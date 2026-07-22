"""Stage 7: Cross-replicate alignment, gap filling, and the EIC spill."""
from __future__ import annotations

import gc
import bisect
import hashlib
import json
import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional, Callable

import numpy as np

from asfam.config import ProcessingConfig
from asfam.core.eic import SegmentEicIndex
from asfam.io.eic_store import (
    SPOTMAP_NAME, STORE_NAME, EicSpillWriter, SpotChromatograms, SpotMap, Trace,
    save_spot_map,
)
from asfam.models import CandidateFeature, Feature, RawSegmentData
from asfam.pipeline.stage0_load import run_stage0_one_sample
from asfam.pipeline.stage7_reconcile import (
    attach_prejoin_alias_folds,
    audit_conservation,
    audit_spots,
    collapse_ms1_natural_peak_aliases,
    gap_fill_target_for_spot,
    reconcile_spots,
)
from metabo_core.algorithms.peak_aggregation import aggregate_feature_gaussian
from metabo_core.alignment.gap_filler import (
    MS1,
    FILLED,
    NO_SIGNAL,
    GapFillConfig,
    GapFillResult,
    GapFillTarget,
    fill_from_chromatogram,
    make_filled_peak,
)
from metabo_core.alignment.joiner import (
    AlignmentSpot, Ms2Reader, build_feature, join_spots,
)
from metabo_core.alignment.refiner import order_spots_by_mz, refine_features
from metabo_core.annotation import is_high_confidence
from metabo_core.utils.memlog import log_memory

logger = logging.getLogger(__name__)
MERGE_MAP_NAME = "alignment_merge_map.json"
GAP_FILL_REUSE_AUDIT_NAME = "gap_fill_natural_peak_reuse.json"


def _candidate_is_high_confidence(peak: CandidateFeature, confidence) -> bool:
    """CandidateFeature equivalent of the Feature confidence predicate.

    CandidateFeature deliberately has no ``selected_annotation`` property, so
    passing it to the shared Feature predicate would classify every candidate
    as unidentified and silently discard annotation quality from keeper choice.
    """
    selected = int(peak.selected_annotation_idx)
    if not 0 <= selected < len(peak.annotation_matches):
        return False
    match = peak.annotation_matches[selected]
    return (
        getattr(match, "score", None) is not None
        and float(match.score) >= confidence.score_threshold
        and int(getattr(match, "n_matched", 0) or 0)
        >= confidence.min_matched_peaks
    )


@dataclass
class GapFillContext:
    """What gap filling needs that the joiner does not: raw data, and somewhere
    to put the chromatograms it extracts.

    Passing this rather than the raw scans themselves keeps the rule PR-3
    established: only one sample's mzML files are ever resident, and alignment
    itself still touches none of them.
    """

    sample_files: dict[str, list[str]]
    output_dir: str
    temp_dir: Optional[Path] = None


@dataclass(frozen=True)
class _NaturalMs1Peak:
    spot_index: int
    peak: CandidateFeature


@dataclass(frozen=True)
class _NaturalPeakReuseBlock:
    sample_id: str
    target_spot_id: str
    source_spot_id: str
    source_peak_id: str
    source_detection: str
    target_mz: float
    source_mz: float
    mz_delta: float
    target_rt: float
    fill_rt_apex: float
    source_rt_apex: float
    rt_delta: float
    fill_rt_left: float
    fill_rt_right: float
    source_rt_left: float
    source_rt_right: float
    fill_height: float
    source_height: float
    segment_name: str


class _NaturalMs1PeakIndex:
    """Natural detections a gap fill is forbidden to reuse in another row."""

    def __init__(
        self,
        spots: list[AlignmentSpot],
        *,
        rt_tolerance: float,
    ) -> None:
        self._rt_tolerance = max(0.0, float(rt_tolerance))
        entries: dict[str, list[tuple[float, _NaturalMs1Peak]]] = {}
        seen: set[tuple[str, int]] = set()
        for spot in spots:
            natural = [
                (sample_id, peak)
                for sample_id, peak in spot.peaks.items()
                if peak.gap_fill_status == "detected"
            ] + [
                (fold.sample_id, fold.peak) for fold in spot.folded_peaks
            ]
            for sample_id, peak in natural:
                token = (sample_id, id(peak))
                if token in seen or peak.quant_route != MS1 or peak.align_mz is None:
                    continue
                seen.add(token)
                entries.setdefault(sample_id, []).append((
                    float(peak.align_mz), _NaturalMs1Peak(spot.index, peak),
                ))
        self._entries = {}
        for sample_id, values in entries.items():
            values.sort(key=lambda item: (item[0], str(item[1].peak.feature_id)))
            self._entries[sample_id] = (
                [item[0] for item in values], values,
            )

    def reused_peak(
        self,
        sample_id: str,
        spot: AlignmentSpot,
        target: GapFillTarget,
        result: GapFillResult,
    ) -> Optional[_NaturalMs1Peak]:
        if target.quant.kind != MS1 or result.status != FILLED:
            return None
        indexed = self._entries.get(sample_id)
        if indexed is None:
            return None
        mz_values, entries = indexed
        lo = bisect.bisect_left(
            mz_values, target.quant.mz - target.quant.tolerance,
        )
        hi = bisect.bisect_right(
            mz_values, target.quant.mz + target.quant.tolerance,
        )
        target_segment = str(target.segment_name or "")
        for _mz, item in entries[lo:hi]:
            peak = item.peak
            if item.spot_index == spot.index:
                continue
            peak_segment = str(peak.segment_name or "")
            if (
                target_segment
                and peak_segment
                and target_segment != peak_segment
            ):
                continue
            fill_apex_in_natural = (
                float(peak.rt_left) <= result.rt_apex <= float(peak.rt_right)
            )
            natural_apex_in_fill = (
                result.rt_left <= float(peak.rt_apex) <= result.rt_right
            )
            overlap = max(
                0.0,
                min(float(result.rt_right), float(peak.rt_right))
                - max(float(result.rt_left), float(peak.rt_left)),
            )
            fill_width = float(result.rt_right) - float(result.rt_left)
            natural_width = float(peak.rt_right) - float(peak.rt_left)
            overlap_ratio = (
                overlap / min(fill_width, natural_width)
                if fill_width > 0 and natural_width > 0
                else 0.0
            )
            natural_apex_in_search = (
                abs(float(peak.rt_apex) - float(target.rt_center))
                <= self._rt_tolerance
            )
            apex_delta = abs(float(result.rt_apex) - float(peak.rt_apex))
            fill_height = float(result.height or 0.0)
            natural_height = float(peak.ms1_height or 0.0)
            same_raw_height = (
                fill_height > 0
                and natural_height > 0
                and math.isclose(
                    fill_height,
                    natural_height,
                    rel_tol=1e-10,
                    abs_tol=1e-6,
                )
            )
            # Mutual apex containment is direct peak-geometry evidence.  A
            # natural detector can nevertheless retain an offset model apex and
            # narrow bounds while gap fill selects the same raw EIC maximum.  In
            # that case the exact (unrounded) height is the direct fingerprint,
            # provided the fitted intervals still intersect and the two apices
            # are within the existing RT tolerance.  Normally the natural apex
            # must also lie in the target search window.  If its fitted apex is
            # just outside that window, retain the older strong-geometry route:
            # the selected fill apex lies inside the natural bounds and at least
            # half of the narrower fitted interval overlaps.  This admits both
            # observed offset-apex forms without treating disjoint or opposite-
            # end, coincidentally equal-height peaks as one.
            if (
                (fill_apex_in_natural and natural_apex_in_fill)
                or (
                    same_raw_height
                    and apex_delta <= self._rt_tolerance
                    and overlap > 0.0
                    and (
                        natural_apex_in_search
                        or (
                            fill_apex_in_natural
                            and overlap_ratio >= 0.5
                        )
                    )
                )
            ):
                return item
        return None


def run_stage7(
    features_by_replicate: dict[str, list[CandidateFeature]],
    config: ProcessingConfig,
    progress_callback: Optional[Callable] = None,
    ms2_reader: Optional[Ms2Reader] = None,
    gap_fill: Optional[GapFillContext] = None,
    stats_out: Optional[dict] = None,
    mapping_output_dir: Optional[str] = None,
) -> list[Feature]:
    """Align features across biological replicates, then fill the gaps.

    The join delegates to the shared MS-DIAL-style joiner: a union master list
    seeded by the reference replicate, bucketed candidate generation, a greedy
    claim, and a representative chosen per spot rather than taken from the
    reference. It reads no raw data — the orchestrator hands it features read
    back from the per-sample ``.mfeat`` / ``.mspec`` spill, with the raw scans
    and the spectral library both already freed.

    Gap filling then reopens the raw data one sample at a time and integrates
    each spot's quantitation ion wherever the joiner found no peak, so the
    exported matrix has a number for every (feature, sample). It writes every
    chromatogram it extracts to ``alignment.eic`` on the way past, which is what
    lets the GUI plot a feature without any raw scan at all. ``spotmap.json``
    goes beside it: the store is keyed by spot id, and the single-sample view
    holds candidate ids, which are a disjoint namespace.

    Finally the refiner marks the spots the union master list produced twice for
    one compound. It runs *after* the isotope plumbing below, and its m/z
    renumbering runs *before* gap filling — both orderings are load-bearing and
    are explained where they happen.
    """
    logger.info("Stage 7: Cross-replicate alignment...")

    if not features_by_replicate:
        return []

    join_config = config.joiner_view()
    join_input, alias_folds, prejoin_stats = collapse_ms1_natural_peak_aliases(
        features_by_replicate,
    )
    spots, join_stats = join_spots(
        join_input, join_config, ms2_reader,
    )
    attach_prejoin_alias_folds(spots, alias_folds)

    confidence = config.confidence_view()
    spots, reconcile_stats = reconcile_spots(
        spots,
        join_config,
        config.refiner_view(),
        is_annotated=lambda peak: _candidate_is_high_confidence(peak, confidence),
        ms2_reader=ms2_reader,
    )
    conservation = audit_conservation(features_by_replicate, spots)
    if (
        conservation.n_unexplained_lost
        or conservation.n_input_keyed_peaks
        != conservation.n_assigned_peaks + conservation.n_collapsed_same_compound
    ):
        raise RuntimeError(
            "Stage 7 natural-peak conservation failed: "
            f"input={conservation.n_input_keyed_peaks}, "
            f"assigned={conservation.n_assigned_peaks}, "
            f"SAME-folded={conservation.n_collapsed_same_compound}, "
            f"unexplained={conservation.n_unexplained_lost}"
        )

    # Step 6.4, hoisted: gap filling keys alignment.eic by feature_id, so the
    # numbering has to be settled before the first id is minted.
    spots = order_spots_by_mz(spots)

    spot_audit = audit_spots(spots, join_config, ms2_reader)
    structural_violations = {
        name: getattr(spot_audit, name)
        for name in (
            "n_mixed_quant_route_spots",
            "n_product_multi_window_spots",
            "n_product_multi_segment_spots",
            "n_same_folds_below_min_matched",
            "n_fold_records_missing_target",
            "n_fold_identity_not_same",
            "n_fold_evidence_mismatch",
            "n_hidden_keeper_with_visible_member",
            "n_duplicate_source_tokens",
            "n_consensus_rt_mismatch",
            "n_consensus_mz_mismatch",
            "n_consensus_window_mismatch",
            "n_consensus_segment_mismatch",
        )
        if getattr(spot_audit, name)
    }
    if structural_violations:
        detail = ", ".join(
            f"{name}={value}" for name, value in structural_violations.items()
        )
        raise RuntimeError(f"Stage 7 structural invariant failed: {detail}")

    # Scientific source->keeper provenance is useful even when raw data are no
    # longer available for gap filling.  Keep it separate from spotmap.json:
    # that file is an index into alignment.eic and must never be refreshed on
    # its own while an older EIC store remains beside it.
    if spots and mapping_output_dir is not None:
        save_spot_map(
            Path(mapping_output_dir) / MERGE_MAP_NAME, _spot_map(spots),
        )

    gap_fill_stats: dict[str, object] = {}
    if spots and gap_fill is not None and config.gap_fill_enabled:
        # Written in lockstep with alignment.eic below, and never on its own: the
        # two are a pair, and a run that refreshed one but not the other would
        # hand the GUI keys into a store minted by some earlier join.
        save_spot_map(Path(gap_fill.output_dir) / SPOTMAP_NAME, _spot_map(spots))
        fill_counts = run_gap_fill(
            spots, sorted(features_by_replicate), config, gap_fill,
            progress_callback,
        )
        if fill_counts is not None:
            gap_fill_stats = dict(fill_counts)
    elif spots and gap_fill is None:
        logger.info("  Gap fill skipped: no raw data context")
    elif spots:
        logger.info("  Gap fill disabled by config")

    spot_audit_values = asdict(spot_audit)
    final_statuses = Counter(
        peak.gap_fill_status for spot in spots for peak in spot.peaks.values()
    )
    spot_audit_values.update({
        "n_detected_cells": final_statuses["detected"],
        "n_filled_cells": final_statuses["filled"],
        "n_no_signal_cells": final_statuses["no_signal"],
    })

    n_samples = len(features_by_replicate)
    result: list[Feature] = []
    for spot in spots:
        feature = build_feature(spot, _feature_id(spot.index), n_samples)
        # The shared core leaves these four unset — they need either ASFAM's
        # gaussian aggregation or ASFAM's dedup-stage bookkeeping, so they are
        # plumbed from the representative here rather than in metabo_core.
        rep = spot.representative
        feature.gaussian_similarity = aggregate_feature_gaussian(
            rep.ms1_gaussian, rep.ms2_gaussian,
        )
        feature.isotope_index = rep.isotope_index
        feature.isotope_group_id = rep.isotope_group_id
        feature.adduct_group_id = rep.adduct_group_id
        result.append(feature)

    logger.info("  Aligned features: %d", len(result))

    # After the loop, never inside it: the refiner's second placement pass skips
    # unidentified isotope satellites, and isotope_index is only true three lines
    # up. Called with a feature missing it, the pass silently does nothing.
    refine_stats = refine_features(
        result,
        config.refiner_view(),
        is_annotated=lambda f: is_high_confidence(f, confidence),
    )

    if stats_out is not None:
        stats_out.update({
            "prejoin": asdict(prejoin_stats),
            "join": asdict(join_stats),
            "reconciliation": asdict(reconcile_stats),
            "conservation": asdict(conservation),
            "spots": spot_audit_values,
            "gap_fill": gap_fill_stats,
            "refinement": asdict(refine_stats),
        })

    if progress_callback:
        progress_callback("stage7", 1, 1, "Alignment complete")

    return result


def _feature_id(index: int) -> str:
    return f"F{index:05d}"


def _spot_map(spots: list[AlignmentSpot]) -> SpotMap:
    """The key map the single-sample EIC view needs, spilled beside the store.

    Call this *after* ``order_spots_by_mz`` — ``spot.index`` is what it mints ids
    from, and that is the line that settles it — and *before* gap filling, for
    two reasons. Gap filling mints a peak into ``spot.peaks`` for every sample
    that had none, under a derived id (``make_filled_peak``: ``rep1_00042@gap:2``)
    that no candidate in ``_work/`` carries; built afterwards, the map would fill
    with keys nothing can ever look up. And ``build_feature`` cannot do this at
    all: ``Feature`` does not record which candidates it was built from.

    A candidate no spot claimed is simply absent — the GUI says so rather than
    plotting a neighbour's chromatogram.
    """
    return SpotMap(
        spot_of={
            peak.feature_id: _feature_id(spot.index)
            for spot in spots
            for peak in spot.peaks.values()
        } | {
            fold.peak.feature_id: _feature_id(spot.index)
            for spot in spots
            for fold in spot.folded_peaks
        },
        representative_of={
            _feature_id(spot.index): spot.representative_sample_id
            for spot in spots if spot.representative_sample_id
        },
        fold_reason_of={
            fold.peak.feature_id: fold.reason
            for spot in spots
            for fold in spot.folded_peaks
        },
        fold_evidence_of={
            fold.peak.feature_id: {
                "target_peak_id": fold.target_peak_id,
                "evidence_kind": fold.evidence_kind,
                "cosine": fold.cosine,
                "n_matched_fragments": fold.n_matched_fragments,
                **fold.evidence_details,
            }
            for spot in spots
            for fold in spot.folded_peaks
        },
    )


# ---------------------------------------------------------------------------
# Chromatogram source
# ---------------------------------------------------------------------------

class _SampleChromatograms:
    """Chromatograms of one loaded sample, routed to the right m/z segment.

    An ASFAM sample is dozens of mzML files split by precursor m/z, a concept
    MS-DIAL has no equivalent of. A spot is served by the segment its tallest
    detected peak was measured in; if that sample is missing the file (a partial
    acquisition), fall back to whichever segment's range covers the ion, using
    the same ``low <= mz <= high + 1`` rule the EIC viewer routes with.

    Exactly one :class:`SegmentEicIndex` is alive at a time. Callers walk the
    spots segment-major, so that costs one build per segment rather than one per
    spot — 23 ms and 5.2 MiB apiece.
    """

    def __init__(self, segments: list[RawSegmentData]):
        self._by_name = {s.segment_name: s for s in segments}
        self._segments = segments
        self._index: Optional[SegmentEicIndex] = None
        self._index_name: str = ""

    def segment_for(self, target: GapFillTarget) -> Optional[RawSegmentData]:
        segment = self._by_name.get(target.segment_name)
        if segment is not None:
            return segment
        probe = (target.quant.mz if target.quant.kind == MS1
                 else float(target.quant.channel))
        for candidate in self._segments:
            if candidate.segment_low <= probe <= candidate.segment_high + 1:
                return candidate
        return None

    def index_for(self, segment: RawSegmentData) -> SegmentEicIndex:
        if self._index_name != segment.segment_name:
            self._index = SegmentEicIndex(segment)
            self._index_name = segment.segment_name
        return self._index

    def release(self) -> None:
        self._index = None
        self._index_name = ""

    def chromatogram(
        self, target: GapFillTarget,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """``(rt, raw intensity)`` over the target's RT window, or ``None``."""
        segment = self.segment_for(target)
        if segment is None:
            return None
        index = self.index_for(segment)
        quant = target.quant
        if quant.kind == MS1:
            dense = index.ms1_eic_sum(quant.mz, quant.tolerance)
        else:
            dense = index.product_eic_sum(quant.channel, quant.mz, quant.tolerance)
        if dense is None:
            return None
        return _slice_rt(index.rt_array, dense, target.rt_lo, target.rt_hi)


def _slice_rt(
    rt: np.ndarray, intensity: np.ndarray, rt_lo: float, rt_hi: float,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    lo = int(np.searchsorted(rt, rt_lo, side="left"))
    hi = int(np.searchsorted(rt, rt_hi, side="right"))
    if hi <= lo:
        return None
    return rt[lo:hi], intensity[lo:hi]


# ---------------------------------------------------------------------------
# Gap fill + EIC spill
# ---------------------------------------------------------------------------

def run_gap_fill(
    spots: list[AlignmentSpot],
    sample_ids: list[str],
    config: ProcessingConfig,
    context: GapFillContext,
    progress_callback: Optional[Callable] = None,
) -> dict[str, object]:
    """Fill every (spot, sample) hole and spill ``alignment.eic``. Mutates ``spots``.

    One sample's raw data is resident at a time, as in ``PeakAligner.cs:105,
    129-130``. Because the chromatograms come out sample-major and the reader
    wants them spot-major, each sample streams into a temporary file and a final
    pass transposes them (``PeakAligner.cs:209-226``).
    """
    gap_config = config.gap_fill_view()
    targets = [gap_fill_target_for_spot(spot, gap_config) for spot in spots]
    natural_ms1 = _NaturalMs1PeakIndex(
        spots,
        rt_tolerance=gap_config.rt_tolerance,
    )
    n_untargeted = sum(1 for t in targets if t is None)
    if n_untargeted:
        logger.warning(
            "  %d spots have no quantitation ion and stay unfilled", n_untargeted,
        )

    # Segment-major, then channel-major: one SegmentEicIndex build per segment
    # instead of per spot, and the product-ion LRU never thrashes.
    order = sorted(
        range(len(spots)),
        key=lambda i: (
            (targets[i].segment_name, targets[i].quant.channel, targets[i].quant.mz)
            if targets[i] is not None else ("", 0, 0.0)
        ),
    )

    keys = [_feature_id(spot.index) for spot in spots]
    filled = Counter()
    blocked_reuses: list[_NaturalPeakReuseBlock] = []
    store_path = Path(context.output_dir) / STORE_NAME

    with EicSpillWriter(keys, temp_dir=context.temp_dir) as writer:
        for n, sample_id in enumerate(sample_ids, start=1):
            if progress_callback:
                progress_callback("stage7", n, len(sample_ids),
                                  f"Gap fill: {sample_id}")
            log_memory(logger, f"gap fill {sample_id} pre")
            segments = run_stage0_one_sample(context.sample_files[sample_id], config)
            source = _SampleChromatograms(segments)
            # The one place this stage's RSS peaks: this sample's raw scans plus
            # one segment's centroid index. "post" below is measured after the
            # release and would report the floor, not the peak.
            log_memory(logger, f"gap fill {sample_id} raw resident")
            writer.add_sample(
                _one_sample(spots, targets, order, sample_id, source,
                            config, gap_config, natural_ms1, filled,
                            blocked_reuses),
            )
            log_memory(logger, f"gap fill {sample_id} spots done")
            source.release()
            del segments, source
            gc.collect()
            log_memory(logger, f"gap fill {sample_id} released")

        size = writer.transpose(store_path)

    total = filled["detected"] + filled["filled"] + filled["no_signal"]
    logger.info(
        "  Gap fill: %d detected, %d filled, %d no signal (%.1f%% of %d cells)",
        filled["detected"], filled["filled"], filled["no_signal"],
        100.0 * (filled["filled"] + filled["no_signal"]) / max(total, 1), total,
    )
    for kind in ("ms1", "product"):
        n = filled[f"kind:{kind}"]
        if n:
            logger.info("    %-8s quantitation ion: %d fills", kind, n)
    if filled["blocked_natural_peak_reuse"]:
        logger.info(
            "    %d MS1 fills rejected: the selected peak is already a natural "
            "detection in another spot",
            filled["blocked_natural_peak_reuse"],
        )
    n_fills = filled["peak_top"] + filled["forced"]
    if n_fills:
        # A force-inserted fill has no peak top within +/-rt_tolerance: its
        # height is background at the expected RT, not a missed peak. Both
        # come out as "filled", so only this split tells them apart.
        logger.info(
            "    %d fills framed a real peak top, %d force-inserted (%.1f%%)",
            filled["peak_top"], filled["forced"],
            100.0 * filled["forced"] / n_fills,
        )
    logger.info("  Wrote %s: %d spots, %.1f MiB", STORE_NAME, len(keys), size / 2**20)
    blocked_payload = [asdict(item) for item in blocked_reuses]
    blocked_blob = json.dumps(
        blocked_payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    )
    debug_dir = Path(context.output_dir) / "_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    audit_path = debug_dir / GAP_FILL_REUSE_AUDIT_NAME
    audit_path.write_text(
        json.dumps(blocked_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary: dict[str, object] = dict(filled)
    summary.update({
        "blocked_natural_peak_reuse_audit": str(audit_path),
        "blocked_natural_peak_reuse_sha256": hashlib.sha256(
            blocked_blob.encode("utf-8"),
        ).hexdigest(),
        "blocked_natural_peak_reuse_examples": blocked_payload[:20],
    })
    return summary


def _one_sample(
    spots: list[AlignmentSpot],
    targets: list[Optional[GapFillTarget]],
    order: list[int],
    sample_id: str,
    source: _SampleChromatograms,
    config: ProcessingConfig,
    gap_config: GapFillConfig,
    natural_ms1: _NaturalMs1PeakIndex,
    filled: Counter,
    blocked_reuses: list[_NaturalPeakReuseBlock],
) -> Iterator[SpotChromatograms]:
    """Fill this sample's holes; yield one :class:`SpotChromatograms` per spot.

    Bodies must come out in spot order for the spill writer, but the work runs
    in ``order`` so the segment index is built once per segment. Holding one
    sample's traces costs ~6 MiB.
    """
    bodies: list[SpotChromatograms] = [SpotChromatograms() for _ in spots]

    for i in order:
        spot, target = spots[i], targets[i]
        if target is None:
            continue

        chromatogram = source.chromatogram(target)
        peak = spot.peaks.get(sample_id)
        if peak is None:
            if chromatogram is None:
                result = GapFillResult(status=NO_SIGNAL)
            else:
                result = fill_from_chromatogram(*chromatogram, target, gap_config)
            blocker = natural_ms1.reused_peak(
                sample_id, spot, target, result,
            )
            if blocker is not None:
                filled["blocked_natural_peak_reuse"] += 1
                source_peak = blocker.peak
                source_mz = float(source_peak.align_mz)
                blocked_reuses.append(_NaturalPeakReuseBlock(
                    sample_id=str(sample_id),
                    target_spot_id=_feature_id(spot.index),
                    source_spot_id=_feature_id(blocker.spot_index),
                    source_peak_id=str(source_peak.feature_id),
                    source_detection=str(source_peak.detection_source),
                    target_mz=float(target.quant.mz),
                    source_mz=source_mz,
                    mz_delta=abs(float(target.quant.mz) - source_mz),
                    target_rt=float(target.rt_center),
                    fill_rt_apex=float(result.rt_apex),
                    source_rt_apex=float(source_peak.rt_apex),
                    rt_delta=abs(float(result.rt_apex) - float(source_peak.rt_apex)),
                    fill_rt_left=float(result.rt_left),
                    fill_rt_right=float(result.rt_right),
                    source_rt_left=float(source_peak.rt_left),
                    source_rt_right=float(source_peak.rt_right),
                    fill_height=float(result.height),
                    source_height=float(source_peak.ms1_height or 0.0),
                    segment_name=str(target.segment_name or ""),
                ))
                result = GapFillResult(status=NO_SIGNAL)
            peak = make_filled_peak(spot.representative, sample_id, target, result)
            spot.peaks[sample_id] = peak
            filled[f"kind:{target.quant.kind}"] += 1
            if result.status != NO_SIGNAL:
                filled["forced" if result.forced else "peak_top"] += 1
        filled[peak.gap_fill_status] += 1

        if chromatogram is not None:
            rt, intensity = chromatogram
            bodies[i].quant.append(Trace(
                label=sample_id, rt=rt, intensity=intensity,
                status=peak.gap_fill_status,
                rt_left=peak.rt_left, rt_right=peak.rt_right, rt_apex=peak.rt_apex,
            ))

        if sample_id == spot.representative_sample_id:
            bodies[i].fragments.extend(
                _fragment_traces(spot, target, source, config),
            )

    return iter(bodies)


def _fragment_traces(
    spot: AlignmentSpot,
    target: GapFillTarget,
    source: _SampleChromatograms,
    config: ProcessingConfig,
) -> list[tuple[float, Trace]]:
    """Product-ion chromatograms for the MS2 panel, from the representative sample."""
    if not spot.representative_ms2:
        return []
    mz, intensity = spot.representative_ms2
    if len(mz) == 0:
        return []

    segment = source.segment_for(target)
    if segment is None:
        return []
    index = source.index_for(segment)
    channel = int(spot.representative.precursor_mz_nominal)

    k = min(int(config.eic_store_top_fragments), len(mz))
    top = np.argsort(intensity)[-k:][::-1]

    traces = []
    for j in top:
        dense = index.product_eic_sum(channel, float(mz[j]), config.eic_mz_tolerance)
        if dense is None:
            break
        window = _slice_rt(index.rt_array, dense, target.rt_lo, target.rt_hi)
        if window is None:
            break
        traces.append((float(mz[j]), Trace(
            label=f"{mz[j]:.3f}", rt=window[0], intensity=window[1],
        )))
    return traces
