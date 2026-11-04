"""ASFAM-only pre-gap-fill spot reconciliation and consensus centres."""
from __future__ import annotations

import math
import hashlib
import json
import bisect
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np

from metabo_core.alignment.gap_filler import (
    GapFillTarget,
    choose_quant_channel,
    consensus_quant_ion,
    gap_fill_target,
)
from metabo_core.alignment.cluster import Cluster, observation_of
from metabo_core.alignment.identity import (
    IdentityEvidence,
    IdentityState,
    identity,
    ms2_identity_evidence,
)
from metabo_core.alignment.joiner import AlignmentSpot, FoldedPeak, Ms2Reader
from metabo_core.config import (
    GapFillConfig, IdentityConfig, JoinerConfig, RefinerConfig,
)
from metabo_core.models import MS1, PRODUCT, CandidateFeature


@dataclass
class ReconcileStats:
    n_input_spots: int = 0
    n_output_spots: int = 0
    n_merged_spots: int = 0
    n_detected_cells_transferred: int = 0
    n_detected_over_fill: int = 0
    n_detected_conflicts_folded: int = 0
    n_product_window_rejected: int = 0
    n_identity_different_rejected: int = 0
    n_identity_unjudgeable_rejected: int = 0
    n_candidate_clusters_examined: int = 0
    n_disjoint_ms1_spots_merged: int = 0
    #: Spots whose joiner folds admit no single cell keeper that is SAME with
    #: every alternative. The assignment is left as the joiner made it.
    n_cell_plans_unresolved: int = 0


@dataclass(frozen=True)
class PrejoinAliasStats:
    n_input_candidates: int = 0
    n_output_candidates: int = 0
    n_ms1_aliases_folded: int = 0
    #: Subset of ``n_ms1_aliases_folded`` collapsed by the same-detection-path
    #: pass, i.e. on ``same_source_exact`` evidence rather than ``dual_path``.
    n_same_path_aliases_folded: int = 0


@dataclass(frozen=True)
class ConservationAudit:
    n_input_keyed_peaks: int
    n_assigned_peaks: int
    n_collapsed_same_compound: int
    n_unexplained_lost: int


@dataclass(frozen=True)
class SpotAudit:
    """Machine-checkable Stage 7 spot and source-membership invariants."""

    n_spots: int
    n_detected_cells: int
    n_filled_cells: int
    n_no_signal_cells: int
    n_fold_records: int
    n_mixed_quant_route_spots: int
    n_product_multi_window_spots: int
    n_product_multi_segment_spots: int
    n_same_folds_below_min_matched: int
    n_fold_records_missing_target: int
    n_fold_identity_not_same: int
    n_fold_evidence_mismatch: int
    n_ms1_natural_peak_evidence_mismatch: int
    n_hidden_keeper_with_visible_member: int
    n_duplicate_source_tokens: int
    n_consensus_rt_mismatch: int
    n_consensus_mz_mismatch: int
    n_consensus_window_mismatch: int
    n_consensus_segment_mismatch: int
    membership_sha256: str
    assigned_detected_sha256: str
    natural_source_sha256: str
    #: INV-3: every natural member of a row is within one isolation window of
    #: the row's. Replaces ``n_product_multi_window_spots``, which only ever
    #: looked at PRODUCT members because only PRODUCT rows could be split by a
    #: window; a route-agnostic row has to hold the rule for all of them.
    n_window_span_violations: int = 0
    #: INV-4: the non-empty acquisition segments of a row's natural members all
    #: agree. Replaces ``n_product_multi_segment_spots``, same reason.
    n_segment_mismatch_spots: int = 0
    #: Rows holding both an ``ms1_detected`` and an ``ms2_only`` natural member —
    #: the thing the route split made impossible. **Not a violation**: it is the
    #: measurement of G1, and ``n_mixed_quant_route_spots`` (which counts the
    #: same rows) stops being enforceable the moment this is intended.
    n_cross_route_spots: int = 0
    #: Rows whose natural members carry two or more *different* high-confidence
    #: annotation names. The mis-merge indicator, and the one number that says
    #: whether looser grouping bought coverage or bought nonsense: splitting one
    #: compound over two rows is redundancy, but merging two compounds into one
    #: makes that row's CV and fold change meaningless. Requires
    #: ``is_annotated``; ``0`` without it, which is not the same as "none".
    n_conflicting_annotation_spots: int = 0
    #: ``[(feature index, sorted names)]``, capped, for looking at.
    conflicting_annotation_examples: tuple = ()


class _Spectra:
    """Small read-through cache for candidate spectra used in reconciliation."""

    def __init__(self, reader: Optional[Ms2Reader], maxsize: int = 4096) -> None:
        self._reader = reader
        self._maxsize = max(1, int(maxsize))
        self._cache: OrderedDict[
            tuple[str, int], list[tuple[float, float]]
        ] = OrderedDict()

    def get(self, sample_id: str, peak: CandidateFeature) -> list[tuple[float, float]]:
        key = (sample_id, id(peak))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        mz = getattr(peak, "ms2_mz", None)
        intensity = getattr(peak, "ms2_intensity", None)
        if mz is not None and intensity is not None and len(mz):
            spectrum = list(zip(np.asarray(mz).tolist(), np.asarray(intensity).tolist()))
        elif self._reader is not None:
            arrays = self._reader(sample_id, peak)
            if arrays is None:
                spectrum = []
            else:
                spectrum = list(zip(
                    np.asarray(arrays[0]).tolist(),
                    np.asarray(arrays[1]).tolist(),
                ))
        else:
            spectrum = []
        self._cache[key] = spectrum
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        return spectrum

    def arrays(self, sample_id: str, peak: CandidateFeature) -> tuple[np.ndarray, np.ndarray]:
        spectrum = self.get(sample_id, peak)
        if not spectrum:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty.copy()
        array = np.asarray(spectrum, dtype=np.float64)
        return array[:, 0].copy(), array[:, 1].copy()


def _best_total_score(peak: CandidateFeature) -> float:
    if not peak.annotation_matches:
        return 0.0
    selected = int(peak.selected_annotation_idx)
    if not 0 <= selected < len(peak.annotation_matches):
        return 0.0
    return float(
        getattr(peak.annotation_matches[selected], "total_score", 0.0)
        or 0.0
    )


def _source_signature(spot: AlignmentSpot) -> tuple:
    return tuple(sorted(
        (sample_id, str(peak.feature_id))
        for sample_id, peak in spot.peaks.items()
        if peak.gap_fill_status == "detected"
    ))


def consensus_window_segment(
    detected: list[CandidateFeature], vote: bool,
) -> tuple[Optional[int], str]:
    """The row's isolation window and acquisition segment.

    ``vote=False`` is the historical rule: whatever the tallest detected cell
    says. That is out of step with ``alignment_mz``, which the detected cells
    *vote* on (``consensus_quant_ion``, after MS-DIAL's ``ResetQuantMasses``) —
    one loud sample could therefore move the window while the ion it names came
    from the majority. ``vote=True`` puts both on the same footing: the mode,
    with total height breaking ties.
    """
    if not detected:
        return None, ""
    tallest = min(
        detected,
        key=lambda peak: (-(peak.ms1_height or 0.0), str(peak.feature_id)),
    )
    if not vote:
        return int(tallest.precursor_mz_nominal), str(tallest.segment_name or "")

    def _mode(key: Callable[[CandidateFeature], object]):
        tally: dict[object, tuple[int, float]] = {}
        for peak in detected:
            count, height = tally.get(key(peak), (0, 0.0))
            tally[key(peak)] = (count + 1, height + float(peak.ms1_height or 0.0))
        return max(tally.items(), key=lambda item: (item[1][0], item[1][1]))[0]

    return (
        int(_mode(lambda peak: int(peak.precursor_mz_nominal))),
        str(_mode(lambda peak: str(peak.segment_name or ""))),
    )


def recompute_alignment_center(
    spot: AlignmentSpot,
    gap_config: Optional[GapFillConfig] = None,
    *,
    vote_window: bool = False,
) -> None:
    """Set the one RT/m/z/window/segment centre every later Stage 7 step uses.

    The m/z is the row's **consensus** quantitation ion, voted across the
    detected cells rather than taken from the tallest one — see
    :func:`~metabo_core.alignment.gap_filler.consensus_quant_ion`. Passing no
    ``gap_config`` keeps the old tallest-cell behaviour, which is what callers
    that only need the RT centre want.

    ``vote_window`` extends the same vote to the window and segment; see
    :func:`consensus_window_segment`. Off by default so the legacy engine keeps
    its output byte-for-byte.
    """
    detected = spot.detected_peaks
    if not detected:
        spot.alignment_rt = None
        spot.alignment_mz = None
        spot.alignment_window = None
        spot.alignment_segment = ""
        return

    tallest = min(
        detected,
        key=lambda peak: (
            -(peak.ms1_height or 0.0),
            str(peak.feature_id),
        ),
    )
    spot.alignment_rt = float(np.mean([peak.rt_apex for peak in detected]))
    consensus = None
    if gap_config is not None:
        representative = (
            spot.peaks.get(spot.representative_sample_id)
            if spot.representative_sample_id else None
        )
        # choose_quant_channel, not consensus_quant_ion: a row may now hold both
        # an exact-precursor cell and a fragment-only one, and averaging a
        # fragment m/z into a precursor m/z would give the row an ion that is
        # neither. On a single-route row the two are the same function.
        consensus = choose_quant_channel(detected, representative, gap_config)
    if consensus is not None:
        spot.alignment_mz = float(consensus.mz)
    else:
        spot.alignment_mz = (
            float(tallest.align_mz) if tallest.align_mz is not None else None
        )
    window, segment = consensus_window_segment(detected, vote_window)
    spot.alignment_window = window
    spot.alignment_segment = segment


def gap_fill_target_for_spot(
    spot: AlignmentSpot, config: GapFillConfig,
) -> Optional[GapFillTarget]:
    """Build a target from, then pin it to, the row's one quantitation channel.

    ``gap_fill_target`` derives its channel from the *tallest* detected cell,
    which is right only where every cell shares a route. A row that holds both
    an exact-precursor cell and a fragment-only one would otherwise be filled on
    whichever kind happened to be loudest, so the channel is re-decided here by
    :func:`~metabo_core.alignment.gap_filler.choose_quant_channel` and the m/z,
    the kind and the tolerance are all taken from it together. Taking only the
    m/z — which is what pinning to ``alignment_mz`` alone did — would integrate a
    precursor m/z over a product-ion slice.
    """
    detected = spot.detected_peaks
    target = gap_fill_target(detected, config)
    if target is None:
        return None
    if spot.alignment_rt is None or spot.alignment_mz is None:
        recompute_alignment_center(spot)
    if spot.alignment_rt is None or spot.alignment_mz is None:
        return target

    representative = (
        spot.peaks.get(spot.representative_sample_id)
        if spot.representative_sample_id else None
    )
    quant = choose_quant_channel(detected, representative, config) or target.quant
    quant = replace(quant, mz=float(spot.alignment_mz))
    if quant.kind == PRODUCT and spot.alignment_window is not None:
        quant = replace(quant, channel=int(spot.alignment_window))

    half = config.rt_expansion * max(target.peak_width, config.min_peak_width)
    return replace(
        target,
        quant=quant,
        rt_center=float(spot.alignment_rt),
        rt_lo=float(spot.alignment_rt) - half,
        rt_hi=float(spot.alignment_rt) + half,
        segment_name=spot.alignment_segment,
    )


def _identity(
    left_sample: str,
    left: CandidateFeature,
    right_sample: str,
    right: CandidateFeature,
    spectra: _Spectra,
    config: JoinerConfig,
) -> IdentityEvidence:
    return ms2_identity_evidence(
        spectra.get(left_sample, left),
        spectra.get(right_sample, right),
        mz_tolerance=config.ms2_mz_tolerance,
        same_threshold=config.ms2_identity_threshold,
        min_fragments=config.ms2_identity_min_fragments,
        min_matched_fragments=config.ms2_identity_min_matched_fragments,
    )


def _spot_identity(
    left: AlignmentSpot,
    right: AlignmentSpot,
    spectra: _Spectra,
    config: JoinerConfig,
) -> IdentityEvidence:
    return _identity(
        left.representative_sample_id,
        left.representative,
        right.representative_sample_id,
        right.representative,
        spectra,
        config,
    )


def _merge_rt_gate(route: str, config: RefinerConfig) -> float:
    """How far apart in RT two rows of one compound may sit and still merge.

    ``rt_gate`` (0.05 min) is the historical value and stays the floor. It was
    narrower than the window the joiner splits on (0.2 min), which made every
    joiner-made split unreachable by this pass — see
    :attr:`RefinerConfig.merge_rt_tolerance`.
    """
    if route in config.merge_identity_routes:
        return max(float(config.merge_rt_tolerance), config.rt_gate)
    return config.rt_gate


def _geometry_matches(
    left: AlignmentSpot, right: AlignmentSpot, config: RefinerConfig,
) -> bool:
    """Are these two rows close enough to be one compound split in two?

    The m/z half of the gate is dropped for the routes in
    ``merge_identity_routes``, and only for them: those rows are quantified on a
    per-sample ``argmax`` ion, so two rows of one compound routinely carry
    different ``alignment_mz`` — the very split this pass exists to undo. They
    still have to survive the reliable MS2 SAME decision in
    :func:`_spots_are_mergeable`; dropping the m/z gate removes a coordinate that
    was never evidence for them, it does not remove the evidence.
    """
    if left.alignment_rt is None or right.alignment_rt is None:
        return False
    route = left.quant_route
    if abs(float(left.alignment_rt) - float(right.alignment_rt)) >= _merge_rt_gate(
        route, config,
    ):
        return False
    if route in config.merge_identity_routes:
        return True
    if left.alignment_mz is None or right.alignment_mz is None:
        return False
    gate = config.mz_gate(float(right.alignment_mz))
    return abs(float(left.alignment_mz) - float(right.alignment_mz)) < gate


def _window_compatible(left: AlignmentSpot, right: AlignmentSpot) -> bool:
    if left.quant_route != PRODUCT:
        return True
    if left.alignment_window != right.alignment_window:
        return False
    # Empty segment names occur in hand-made/core features.  When ASFAM has both,
    # require exact compatibility rather than merging across acquisitions.
    return not (
        left.alignment_segment
        and right.alignment_segment
        and left.alignment_segment != right.alignment_segment
    )


_MZ_BUCKET_WIDTH = 1.0


class _ClusterIndex:
    """Sparse lookup of clusters whose anchor can pass the geometry gates.

    Complete-link identity is still checked against every member of a candidate
    cluster.  The index only removes the tens of thousands of chemically
    impossible clusters that an O(N^2) scan would visit for each spot.
    """

    def __init__(self, config: RefinerConfig) -> None:
        self._config = config
        self._rt_width = float(config.rt_gate)
        self._buckets: dict[tuple, list[int]] = {}
        self._partition_clusters: dict[tuple, list[int]] = {}
        # Identity routes match with no m/z gate at all, so an m/z bucket cannot
        # find their pairs. They get a second index cut on RT alone, inside the
        # same route/window partition.
        self._identity_rt_width = max(
            float(config.merge_rt_tolerance), float(config.rt_gate),
        )
        self._identity_buckets: dict[tuple, list[int]] = {}

    @staticmethod
    def _partition(spot: AlignmentSpot) -> tuple:
        return (
            spot.quant_route,
            spot.alignment_window if spot.quant_route == PRODUCT else None,
        )

    def _uses_identity(self, spot: AlignmentSpot) -> bool:
        return spot.quant_route in self._config.merge_identity_routes

    def add(self, cluster_idx: int, anchor: AlignmentSpot) -> None:
        partition = self._partition(anchor)
        self._partition_clusters.setdefault(partition, []).append(cluster_idx)
        if anchor.alignment_rt is None:
            return
        if self._uses_identity(anchor):
            if self._identity_rt_width <= 0:
                return
            self._identity_buckets.setdefault(
                (partition,
                 int(float(anchor.alignment_rt) // self._identity_rt_width)), [],
            ).append(cluster_idx)
            return
        if anchor.alignment_mz is None or self._rt_width <= 0:
            return
        mz_bucket = int(float(anchor.alignment_mz) // _MZ_BUCKET_WIDTH)
        rt_bucket = int(float(anchor.alignment_rt) // self._rt_width)
        self._buckets.setdefault(
            (partition, mz_bucket, rt_bucket), [],
        ).append(cluster_idx)

    def candidates(self, spot: AlignmentSpot) -> list[int]:
        if spot.alignment_rt is None:
            return []
        partition = self._partition(spot)
        if self._uses_identity(spot):
            if self._identity_rt_width <= 0:
                return []
            rt_bucket = int(float(spot.alignment_rt) // self._identity_rt_width)
            found: set[int] = set()
            for rb in (rt_bucket - 1, rt_bucket, rt_bucket + 1):
                found.update(self._identity_buckets.get((partition, rb), ()))
            return sorted(found)
        if spot.alignment_mz is None or self._rt_width <= 0:
            return []
        mz = float(spot.alignment_mz)
        pivot = float(self._config.ppm_pivot_mz)
        tolerance = max(0.0, float(self._config.mz_tolerance))

        # The gate is evaluated at the anchor's m/z.  Above the ppm pivot an
        # anchor that can match ``mz`` is bounded by a < mz/(1-tol/pivot).
        # Deriving the widest possible gate from that bound keeps the index
        # exhaustive rather than assuming the candidate and anchor gates equal.
        if pivot > 0 and tolerance >= pivot:
            return list(self._partition_clusters.get(partition, ()))
        upper_anchor = mz
        if pivot > 0 and tolerance > 0:
            upper_anchor = max(mz, mz / (1.0 - tolerance / pivot))
        max_gate = max(tolerance, self._config.mz_gate(upper_anchor))
        mz_span = int(math.ceil(max_gate / _MZ_BUCKET_WIDTH))
        mz_bucket = int(mz // _MZ_BUCKET_WIDTH)
        rt_bucket = int(float(spot.alignment_rt) // self._rt_width)

        found: set[int] = set()
        for mb in range(mz_bucket - mz_span, mz_bucket + mz_span + 1):
            for rb in (rt_bucket - 1, rt_bucket, rt_bucket + 1):
                found.update(self._buckets.get((partition, mb, rb), ()))
        return sorted(found)


@dataclass(frozen=True)
class _CellAlternative:
    peak: CandidateFeature
    previous_fold_reason: str = ""
    was_assigned: bool = False


@dataclass(frozen=True)
class _Ms1NaturalPeakEvidence:
    """Direct evidence that ASFAM detected one natural MS1 peak twice."""

    state: IdentityState
    mz_delta: float
    rt_delta: float
    peak_overlap_ratio: float
    height_ratio: float
    cosine: float = 0.0
    n_matched_fragments: int = 0

    @property
    def details(self) -> dict[str, float]:
        return {
            "mz_delta": self.mz_delta,
            "rt_delta": self.rt_delta,
            "peak_overlap_ratio": self.peak_overlap_ratio,
            "height_ratio": self.height_ratio,
        }


@dataclass
class _CellPlan:
    winner: CandidateFeature
    alternatives: list[_CellAlternative]
    evidence_by_peak: dict[
        int, IdentityEvidence | _Ms1NaturalPeakEvidence
    ]


def _ms1_natural_peak_evidence(
    left: CandidateFeature,
    right: CandidateFeature,
) -> Optional[_Ms1NaturalPeakEvidence]:
    """Prove two same-sample candidates are the same integrated MS1 peak.

    This is intentionally stricter than cross-sample alignment geometry.  It
    accepts either independent MS1/MS2-driven views, or an exact MS1-driven
    integration fingerprint from the same raw file.  The latter heals duplicate
    mass-slice candidates; area is not identity evidence because overlapping
    slices can integrate different shoulders around the same apex and bounds.
    """
    if left.quant_route != MS1 or right.quant_route != MS1:
        return None
    left_segment = str(left.segment_name or "")
    right_segment = str(right.segment_name or "")
    if not left_segment or left_segment != right_segment:
        return None
    left_detection = str(left.detection_source or "")
    right_detection = str(right.detection_source or "")
    dual_path = {left_detection, right_detection} == {
        "ms1_driven", "ms2_driven",
    }
    left_source = str(left.source_file or "")
    right_source = str(right.source_file or "")
    if (left_source or right_source) and (
        not left_source or not right_source or left_source != right_source
    ):
        return None
    if left.align_mz is None or right.align_mz is None:
        return None
    mz_delta = abs(float(left.align_mz) - float(right.align_mz))
    if mz_delta > 0.01:
        return None
    rt_delta = abs(float(left.rt_apex) - float(right.rt_apex))
    if rt_delta > 0.1:
        return None

    left_width = float(left.rt_right) - float(left.rt_left)
    right_width = float(right.rt_right) - float(right.rt_left)
    if left_width <= 0 or right_width <= 0:
        return None
    overlap = max(
        0.0,
        min(float(left.rt_right), float(right.rt_right))
        - max(float(left.rt_left), float(right.rt_left)),
    )
    overlap_ratio = overlap / min(left_width, right_width)
    if overlap_ratio < 0.5:
        return None

    left_height = float(left.ms1_height or 0.0)
    right_height = float(right.ms1_height or 0.0)
    if left_height <= 0 or right_height <= 0:
        return None
    height_ratio = min(left_height, right_height) / max(
        left_height, right_height,
    )
    if height_ratio < 0.8:
        return None

    same_source_exact = (
        left.gap_fill_status == "detected"
        and right.gap_fill_status == "detected"
        and left_detection == right_detection == "ms1_driven"
        and bool(left_source)
        and left_source == right_source
        and left.replicate_id == right.replicate_id
        and int(left.isotope_index or 0) == int(right.isotope_index or 0)
        and math.isclose(
            float(left.rt_apex), float(right.rt_apex),
            rel_tol=0.0, abs_tol=1e-12,
        )
        and math.isclose(
            float(left.rt_left), float(right.rt_left),
            rel_tol=0.0, abs_tol=1e-12,
        )
        and math.isclose(
            float(left.rt_right), float(right.rt_right),
            rel_tol=0.0, abs_tol=1e-12,
        )
        and math.isclose(
            left_height, right_height,
            rel_tol=1e-12, abs_tol=1e-9,
        )
    )
    if not dual_path and not same_source_exact:
        return None
    return _Ms1NaturalPeakEvidence(
        state=IdentityState.SAME,
        mz_delta=mz_delta,
        rt_delta=rt_delta,
        peak_overlap_ratio=overlap_ratio,
        height_ratio=height_ratio,
    )


def _same_path_alias_pairs(
    pool: list[CandidateFeature],
) -> list[tuple[int, int, _Ms1NaturalPeakEvidence]]:
    """One-to-one alias pairs *within* one detection path, deterministically.

    :func:`_ms1_natural_peak_evidence` can only clear a same-path pair on its
    ``same_source_exact`` branch — bit-identical integration bounds and height
    out of one raw file — so what this heals is a duplicate mass-slice
    candidate, not a second detector's view of the same peak. That branch was
    structurally unreachable until now: the candidate generation below only ever
    drew edges *between* the two paths, which made ``dual_path`` permanently
    true.

    Within one path this is a general graph rather than a bipartite one, so the
    augmenting-path matcher does not apply. Greedy over pairs ordered by
    evidence quality then cell priority is a *maximal* matching; on a relation
    this tight the components are cliques of near-identical slices, where a
    maximum matching has the same cardinality.

    ``pool`` must already be sorted by ``align_mz``.
    """
    pool_mz = [float(peak.align_mz) for peak in pool]
    candidates: list[tuple[int, int, _Ms1NaturalPeakEvidence]] = []
    for left_idx, left in enumerate(pool):
        hi = bisect.bisect_right(pool_mz, pool_mz[left_idx] + 0.01)
        for right_idx in range(left_idx + 1, hi):
            evidence = _ms1_natural_peak_evidence(left, pool[right_idx])
            if evidence is not None:
                candidates.append((left_idx, right_idx, evidence))
    candidates.sort(key=lambda item: (
        item[2].rt_delta,
        item[2].mz_delta,
        -item[2].peak_overlap_ratio,
        -item[2].height_ratio,
        _cell_priority(pool[item[0]]),
        _cell_priority(pool[item[1]]),
    ))

    matched: list[tuple[int, int, _Ms1NaturalPeakEvidence]] = []
    taken: set[int] = set()
    for left_idx, right_idx, evidence in candidates:
        if left_idx in taken or right_idx in taken:
            continue
        taken.add(left_idx)
        taken.add(right_idx)
        matched.append((left_idx, right_idx, evidence))
    return matched


def collapse_ms1_natural_peak_aliases(
    features_by_sample: dict[str, list[CandidateFeature]],
    *,
    enabled: bool = True,
    same_path: bool = False,
) -> tuple[
    dict[str, list[CandidateFeature]],
    dict[tuple[str, int], list[FoldedPeak]],
    PrejoinAliasStats,
]:
    """Collapse direct dual-path aliases before they enter bipartite matching.

    Without this pass, maximum-cardinality assignment can seat the MS1-driven
    and MS2-driven views of one integrated peak in two neighbouring masters.
    The later spot reconciler cannot safely undo that topology once unrelated
    samples have been attached to both rows.

    ``same_path`` adds a second, opt-in pass over the peaks the first one left
    alone, joining ``ms1_driven`` to ``ms1_driven`` and ``ms2_driven`` to
    ``ms2_driven`` on ``same_source_exact`` evidence (see
    :func:`_same_path_alias_pairs`). It runs *after* the dual-path pass and only
    on what that pass did not consume, so turning it on never perturbs a fold
    the shipped behaviour already made.
    """
    n_input = sum(len(peaks) for peaks in features_by_sample.values())
    if not enabled:
        return features_by_sample, {}, PrejoinAliasStats(
            n_input_candidates=n_input,
            n_output_candidates=n_input,
            n_ms1_aliases_folded=0,
        )

    filtered: dict[str, list[CandidateFeature]] = {}
    folds_by_winner: dict[tuple[str, int], list[FoldedPeak]] = {}
    n_folded = 0
    n_same_path = 0

    for sample_id in sorted(features_by_sample):
        peaks = features_by_sample[sample_id]
        ms1_driven = sorted(
            (
                peak for peak in peaks
                if peak.quant_route == MS1
                and peak.align_mz is not None
                and peak.detection_source == "ms1_driven"
            ),
            key=lambda peak: (float(peak.align_mz), _cell_priority(peak)),
        )
        ms2_driven = sorted(
            (
                peak for peak in peaks
                if peak.quant_route == MS1
                and peak.align_mz is not None
                and peak.detection_source == "ms2_driven"
            ),
            key=lambda peak: (float(peak.align_mz), _cell_priority(peak)),
        )
        ms2_mz = [float(peak.align_mz) for peak in ms2_driven]

        # One physical peak can have at most one view from each detection path.
        # A deterministic maximum-cardinality bipartite match therefore avoids
        # the unsafe one-to-many collapse that a greedy keeper loop permits.
        edges: list[list[tuple[int, _Ms1NaturalPeakEvidence]]] = []
        for left in ms1_driven:
            left_mz = float(left.align_mz)
            lo = bisect.bisect_left(ms2_mz, left_mz - 0.01)
            hi = bisect.bisect_right(ms2_mz, left_mz + 0.01)
            compatible = []
            for right_idx in range(lo, hi):
                right = ms2_driven[right_idx]
                evidence = _ms1_natural_peak_evidence(left, right)
                if evidence is not None:
                    compatible.append((right_idx, evidence))
            compatible.sort(key=lambda item: (
                item[1].rt_delta,
                item[1].mz_delta,
                -item[1].peak_overlap_ratio,
                -item[1].height_ratio,
                _cell_priority(ms2_driven[item[0]]),
            ))
            edges.append(compatible)

        holder: dict[int, tuple[int, _Ms1NaturalPeakEvidence]] = {}

        def augment(left_idx: int, seen_right: set[int]) -> bool:
            for right_idx, evidence in edges[left_idx]:
                if right_idx in seen_right:
                    continue
                seen_right.add(right_idx)
                previous = holder.get(right_idx)
                if previous is None or augment(previous[0], seen_right):
                    holder[right_idx] = (left_idx, evidence)
                    return True
            return False

        for left_idx in sorted(
            range(len(ms1_driven)),
            key=lambda idx: _cell_priority(ms1_driven[idx]),
        ):
            augment(left_idx, set())

        consumed: set[int] = set()

        def fold_into(
            keeper: CandidateFeature,
            alias: CandidateFeature,
            evidence: _Ms1NaturalPeakEvidence,
        ) -> None:
            nonlocal n_folded
            consumed.add(id(alias))
            n_folded += 1
            # The alias may already be keeping folds of its own. They name the
            # alias, not the new keeper, so their ``target_peak_id`` stays as it
            # is; moving the bucket only guarantees they land in the *same* spot,
            # where ``audit_spots`` resolves the alias -> keeper link.
            inherited = folds_by_winner.pop((sample_id, id(alias)), [])
            bucket = folds_by_winner.setdefault((sample_id, id(keeper)), [])
            bucket.append(FoldedPeak(
                sample_id=sample_id,
                peak=alias,
                reason="ms1_natural_peak_alias",
                target_peak_id=str(keeper.feature_id),
                cosine=0.0,
                n_matched_fragments=0,
                evidence_kind="ms1_natural_peak",
                evidence_details=evidence.details,
            ))
            bucket.extend(inherited)

        matched = sorted(
            [
                (
                    ms1_driven[left_idx],
                    ms2_driven[right_idx],
                    evidence,
                )
                for right_idx, (left_idx, evidence) in holder.items()
            ],
            key=lambda item: (
                _cell_priority(item[0]), _cell_priority(item[1]),
            ),
        )
        for left, right, evidence in matched:
            keeper, alias = min((left, right), key=_cell_priority), max(
                (left, right), key=_cell_priority,
            )
            fold_into(keeper, alias, evidence)

        if same_path:
            for path in ("ms1_driven", "ms2_driven"):
                pool = sorted(
                    (
                        peak for peak in peaks
                        if peak.quant_route == MS1
                        and peak.align_mz is not None
                        and peak.detection_source == path
                        and id(peak) not in consumed
                    ),
                    key=lambda peak: (float(peak.align_mz), _cell_priority(peak)),
                )
                for left_idx, right_idx, evidence in _same_path_alias_pairs(pool):
                    left, right = pool[left_idx], pool[right_idx]
                    keeper, alias = min((left, right), key=_cell_priority), max(
                        (left, right), key=_cell_priority,
                    )
                    fold_into(keeper, alias, evidence)
                    n_same_path += 1

        filtered[sample_id] = [
            peak for peak in peaks if id(peak) not in consumed
        ]

    return filtered, folds_by_winner, PrejoinAliasStats(
        n_input_candidates=n_input,
        n_output_candidates=n_input - n_folded,
        n_ms1_aliases_folded=n_folded,
        n_same_path_aliases_folded=n_same_path,
    )


def attach_prejoin_alias_folds(
    spots: list[AlignmentSpot],
    folds_by_winner: dict[tuple[str, int], list[FoldedPeak]],
) -> None:
    """Attach pre-join aliases to the spot that conserved their canonical peak."""
    location: dict[tuple[str, int], AlignmentSpot] = {}
    for spot in spots:
        for sample_id, peak in spot.peaks.items():
            location[(sample_id, id(peak))] = spot
        for fold in spot.folded_peaks:
            location[(fold.sample_id, id(fold.peak))] = spot

    missing = []
    for winner_token, folds in folds_by_winner.items():
        spot = location.get(winner_token)
        if spot is None:
            missing.append(f"{winner_token[0]}:{folds[0].target_peak_id}")
            continue
        spot.folded_peaks.extend(folds)
    if missing:
        raise RuntimeError(
            "pre-join MS1 aliases lost their canonical peak: "
            + ", ".join(missing[:10])
        )


def _cluster_cell_plans(
    spots: list[AlignmentSpot],
    spectra: _Spectra,
    config: JoinerConfig,
) -> tuple[Optional[dict[str, _CellPlan]], Optional[IdentityState]]:
    """Find one direct-SAME keeper for every sample's detected alternatives.

    This is deliberately a star around the final cell keeper, not a transitive
    chain through an earlier assigned peak.  If no candidate has reliable SAME
    evidence to every detected loser, the spots cannot safely reconcile.
    """
    by_sample: dict[str, list[_CellAlternative]] = {}
    for spot in spots:
        for sample_id, peak in spot.peaks.items():
            by_sample.setdefault(sample_id, []).append(_CellAlternative(
                peak=peak, was_assigned=True,
            ))
        for fold in spot.folded_peaks:
            by_sample.setdefault(fold.sample_id, []).append(_CellAlternative(
                peak=fold.peak,
                previous_fold_reason=fold.reason,
                was_assigned=False,
            ))

    plans: dict[str, _CellPlan] = {}
    for sample_id, raw_alternatives in sorted(by_sample.items()):
        alternatives: list[_CellAlternative] = []
        seen: set[int] = set()
        for alternative in raw_alternatives:
            token = id(alternative.peak)
            if token not in seen:
                seen.add(token)
                alternatives.append(alternative)

        ordered = sorted(alternatives, key=lambda item: _cell_priority(item.peak))
        detected = [
            item for item in ordered if item.peak.gap_fill_status == "detected"
        ]
        visible_detected = [
            item for item in detected
            if not item.peak.is_duplicate
        ]
        keeper_candidates = visible_detected or ordered
        failures: list[IdentityState] = []
        chosen: Optional[_CellPlan] = None
        for candidate in keeper_candidates:
            evidence_by_peak: dict[
                int, IdentityEvidence | _Ms1NaturalPeakEvidence
            ] = {}
            candidate_failed = False
            if candidate.peak.gap_fill_status == "detected":
                for other in detected:
                    if other.peak is candidate.peak:
                        continue
                    evidence = None
                    if not config.use_ms2_identity_for_ms1:
                        evidence = _ms1_natural_peak_evidence(
                            other.peak, candidate.peak,
                        )
                    if evidence is None:
                        may_use_ms2 = not (
                            other.peak.quant_route == MS1
                            and candidate.peak.quant_route == MS1
                            and not config.use_ms2_identity_for_ms1
                            and not (
                                other.peak.detection_source == "ms2_driven"
                                and candidate.peak.detection_source == "ms2_driven"
                            )
                        )
                        if may_use_ms2:
                            evidence = _identity(
                                sample_id,
                                other.peak,
                                sample_id,
                                candidate.peak,
                                spectra,
                                config,
                            )
                        else:
                            evidence = IdentityEvidence(
                                IdentityState.UNJUDGEABLE,
                                0.0,
                                0,
                                int(other.peak.n_fragments or 0),
                                int(candidate.peak.n_fragments or 0),
                            )
                    if evidence.state is not IdentityState.SAME:
                        failures.append(evidence.state)
                        candidate_failed = True
                        break
                    evidence_by_peak[id(other.peak)] = evidence
            elif detected:
                candidate_failed = True
            if not candidate_failed:
                chosen = _CellPlan(
                    winner=candidate.peak,
                    alternatives=alternatives,
                    evidence_by_peak=evidence_by_peak,
                )
                break

        if chosen is None:
            state = (
                IdentityState.DIFFERENT
                if IdentityState.DIFFERENT in failures
                else IdentityState.UNJUDGEABLE
            )
            return None, state
        plans[sample_id] = chosen
    return plans, None


def _spots_are_mergeable(
    left: AlignmentSpot,
    right: AlignmentSpot,
    spectra: _Spectra,
    joiner_config: JoinerConfig,
    refiner_config: RefinerConfig,
    stats: ReconcileStats,
) -> bool:
    if left.quant_route != right.quant_route:
        return False
    if not _geometry_matches(left, right, refiner_config):
        return False
    if not _window_compatible(left, right):
        stats.n_product_window_rejected += 1
        return False

    # ASFAM MS1 features carry all-ion spectra for a whole co-eluting segment.
    # Those spectra can vary across samples while the chromatographic MS1 peak
    # is stable, so only the narrowed m/z/RT geometry is identity-bearing here.
    # PRODUCT rows still require the reliable MS2 decision below.
    if (
        left.quant_route == MS1
        and not joiner_config.use_ms2_identity_for_ms1
    ):
        return True

    # When the m/z gate has been dropped for this route the spectra are the only
    # evidence left, so they are held to the redundancy pass's own cosine rather
    # than the joiner's. The joiner's is a *veto* threshold — it only has to
    # catch pairs geometry got wrong — and it is deliberately looser.
    identity_config = joiner_config
    if left.quant_route in refiner_config.merge_identity_routes:
        identity_config = replace(
            joiner_config,
            ms2_identity_threshold=refiner_config.ms2_identity_threshold,
        )
    evidence = _spot_identity(left, right, spectra, identity_config)
    if evidence.state is IdentityState.DIFFERENT:
        stats.n_identity_different_rejected += 1
        return False
    if evidence.state is IdentityState.UNJUDGEABLE:
        stats.n_identity_unjudgeable_rejected += 1
        return False

    return True


def _spot_priority(
    spot: AlignmentSpot, is_annotated: Callable[[CandidateFeature], bool],
) -> tuple:
    rep = spot.representative
    hidden = bool(rep.is_duplicate)
    annotated = bool(is_annotated(rep))
    return (
        hidden,
        not annotated,
        -_best_total_score(rep),
        -(rep.ms1_height or 0.0),
        _source_signature(spot),
    )


def _cell_priority(peak: CandidateFeature) -> tuple:
    status_rank = {"detected": 2, "filled": 1, "no_signal": 0}
    return (
        -status_rank.get(peak.gap_fill_status, -1),
        bool(peak.is_duplicate),
        -int(bool(peak.annotation_matches)),
        -_best_total_score(peak),
        -(peak.ms1_sn or 0.0),
        -(peak.ms1_height or 0.0),
        str(peak.feature_id),
    )


def _choose_representative(spot: AlignmentSpot, spectra: _Spectra) -> None:
    detected = [
        (sample_id, peak)
        for sample_id, peak in sorted(spot.peaks.items())
        if peak.gap_fill_status == "detected"
    ]
    if not detected:
        return
    visible = [
        item for item in detected
        if not item[1].is_duplicate
    ]
    eligible = visible or detected
    with_ms2 = [item for item in eligible if item[1].n_fragments > 0]
    sample_id, peak = min(
        with_ms2 or eligible,
        key=lambda item: (
            -_best_total_score(item[1]),
            -(item[1].ms1_height or 0.0),
            item[0],
            str(item[1].feature_id),
        ),
    )
    unchanged = (
        sample_id == spot.representative_sample_id
        and spot.peaks.get(sample_id) is peak
        and spot.representative_ms2 is not None
    )
    spot.representative_sample_id = sample_id
    if not unchanged:
        spot.representative_ms2 = spectra.arrays(sample_id, peak)


def _merge_cluster(
    keeper: AlignmentSpot,
    losers: list[AlignmentSpot],
    cell_plans: dict[str, _CellPlan],
    spectra: _Spectra,
    stats: ReconcileStats,
    gap_config: Optional[GapFillConfig] = None,
) -> None:
    stats.n_merged_spots += len(losers)
    original_keeper_samples = set(keeper.peaks)
    new_peaks: dict[str, CandidateFeature] = {}
    new_folds: list[FoldedPeak] = []

    for sample_id, plan in sorted(cell_plans.items()):
        winner = plan.winner
        new_peaks[sample_id] = winner
        if sample_id not in original_keeper_samples and winner.gap_fill_status == "detected":
            stats.n_detected_cells_transferred += 1

        for alternative in plan.alternatives:
            rejected = alternative.peak
            if rejected is winner:
                continue
            if (
                winner.gap_fill_status == "detected"
                and rejected.gap_fill_status != "detected"
                and alternative.was_assigned
            ):
                stats.n_detected_over_fill += 1
            if rejected.gap_fill_status != "detected":
                continue
            evidence = plan.evidence_by_peak.get(id(rejected))
            if evidence is None or evidence.state is not IdentityState.SAME:
                raise RuntimeError(
                    "attempted to fold a detected cell without direct SAME evidence"
                )
            new_folds.append(FoldedPeak(
                sample_id=sample_id,
                peak=rejected,
                reason=(alternative.previous_fold_reason
                        or "same_route_redundant_cell"),
                target_peak_id=str(winner.feature_id),
                cosine=evidence.cosine,
                n_matched_fragments=evidence.n_matched_fragments,
                evidence_kind=(
                    "ms1_natural_peak"
                    if isinstance(evidence, _Ms1NaturalPeakEvidence)
                    else "ms2_identity"
                ),
                evidence_details=(
                    evidence.details
                    if isinstance(evidence, _Ms1NaturalPeakEvidence)
                    else {}
                ),
            ))
            if alternative.was_assigned:
                stats.n_detected_conflicts_folded += 1

    keeper.peaks = new_peaks
    keeper.folded_peaks = new_folds

    # Merging changes the candidate pool even when the old representative's
    # cell survived.  Re-run the full representative rule so a stronger
    # spectrum/annotation transferred from a loser can become authoritative.
    _choose_representative(keeper, spectra)
    recompute_alignment_center(keeper, gap_config)


def _natural_sample_ids(spot: AlignmentSpot) -> frozenset[str]:
    """Return samples represented by conserved, naturally detected peaks."""
    return frozenset(
        {
            sample_id
            for sample_id, peak in spot.peaks.items()
            if peak.gap_fill_status == "detected"
        }
        | {fold.sample_id for fold in spot.folded_peaks}
    )


def _uniform_isotope_index(spot: AlignmentSpot) -> Optional[int]:
    """Return one reliable isotope role, rejecting internally mixed spots."""
    indices = {
        int(peak.isotope_index or 0)
        for peak in spot.peaks.values()
        if peak.gap_fill_status == "detected"
    }
    indices.update(int(fold.peak.isotope_index or 0) for fold in spot.folded_peaks)
    if len(indices) != 1:
        return None
    return next(iter(indices))


def _complete_disjoint_ms1_spots(
    spots: list[AlignmentSpot],
    spectra: _Spectra,
    joiner_config: JoinerConfig,
    refiner_config: RefinerConfig,
    is_annotated: Callable[[CandidateFeature], bool],
    stats: ReconcileStats,
    gap_config: Optional[GapFillConfig] = None,
) -> list[AlignmentSpot]:
    """Complete rows split across disjoint samples after centres are stable.

    This is deliberately one non-iterative pass over the first reconciliation
    result.  A candidate must be compatible with every original cluster member;
    merging never exposes an updated centroid to another placement decision.
    Isotope group IDs are sample-local, so the conserved isotope index is the
    only cross-sample role used here.
    """
    ordered = sorted(spots, key=lambda spot: _spot_priority(spot, is_annotated))
    clusters: list[list[AlignmentSpot]] = []
    cluster_samples: list[set[str]] = []
    cluster_roles: list[Optional[int]] = []
    cluster_index = _ClusterIndex(refiner_config)

    for spot in ordered:
        role = _uniform_isotope_index(spot)
        samples = set(_natural_sample_ids(spot))
        eligible = spot.quant_route == MS1 and role is not None and bool(samples)
        destination_idx: Optional[int] = None

        if eligible:
            for cluster_idx in cluster_index.candidates(spot):
                if cluster_roles[cluster_idx] != role:
                    continue
                if samples & cluster_samples[cluster_idx]:
                    continue
                cluster = clusters[cluster_idx]
                stats.n_candidate_clusters_examined += 1
                if not all(
                    # The ppm gate is evaluated from one side by the shared
                    # helper.  Requiring the reverse geometry too makes this
                    # ASFAM completion decision independent of keeper order.
                    _geometry_matches(member, spot, refiner_config)
                    and _spots_are_mergeable(
                        spot,
                        member,
                        spectra,
                        joiner_config,
                        refiner_config,
                        stats,
                    )
                    for member in cluster
                ):
                    continue
                cell_plans, cell_state = _cluster_cell_plans(
                    [*cluster, spot], spectra, joiner_config,
                )
                if cell_plans is None:
                    if cell_state is IdentityState.DIFFERENT:
                        stats.n_identity_different_rejected += 1
                    else:
                        stats.n_identity_unjudgeable_rejected += 1
                    continue
                destination_idx = cluster_idx
                break

        if destination_idx is None:
            clusters.append([spot])
            cluster_samples.append(samples)
            cluster_roles.append(role if eligible else None)
            if eligible:
                cluster_index.add(len(clusters) - 1, spot)
        else:
            clusters[destination_idx].append(spot)
            cluster_samples[destination_idx].update(samples)

    completed: list[AlignmentSpot] = []
    for cluster in clusters:
        keeper = cluster[0]
        if len(cluster) > 1:
            cell_plans, cell_state = _cluster_cell_plans(
                cluster, spectra, joiner_config,
            )
            if cell_plans is None:  # pragma: no cover - checked on insertion
                raise RuntimeError(
                    "disjoint MS1 completion lost its cell plan: "
                    f"{cell_state}"
                )
            _merge_cluster(keeper, cluster[1:], cell_plans, spectra, stats,
                           gap_config)
            stats.n_disjoint_ms1_spots_merged += len(cluster) - 1
        completed.append(keeper)
    return completed


def reconcile_spots(
    spots: list[AlignmentSpot],
    joiner_config: JoinerConfig,
    refiner_config: RefinerConfig,
    *,
    is_annotated: Callable[[CandidateFeature], bool],
    ms2_reader: Optional[Ms2Reader] = None,
    gap_config: Optional[GapFillConfig] = None,
) -> tuple[list[AlignmentSpot], ReconcileStats]:
    """Merge only reliable same-route duplicates before any gap is filled.

    Identity is complete-link within a cluster, not union-find: cosine is not
    transitive, so every new member must be SAME with every existing member.
    """
    stats = ReconcileStats(n_input_spots=len(spots))
    if not spots:
        return [], stats

    spectra = _Spectra(ms2_reader, joiner_config.ms2_cache_size)
    for spot in spots:
        # Reconcile cells already folded by the joiner before keeper priority
        # or isotope protection is evaluated.  A fold can be the only visible
        # natural detection in an otherwise hidden singleton; merely choosing
        # among ``spot.peaks`` would then leave the hidden cell as keeper.
        if spot.folded_peaks:
            cell_plans, _cell_state = _cluster_cell_plans(
                [spot], spectra, joiner_config,
            )
            if cell_plans is None:
                # No single alternative is SAME with every other one. That is a
                # real gap in the evidence, not a broken invariant: a sample can
                # hold an MS1 alias of peak P and another of peak Q where P and
                # Q were joined on their MS2, and an MS1 alias witness is only
                # ever direct — it says "these two rows are one chromatographic
                # peak", and it has nothing to say about a third row. The star
                # cannot close, and no amount of re-planning will close it.
                #
                # Keep the assignment the joiner made. Every fold still names
                # the peak it was folded onto and carries the evidence for that
                # one pair, so conservation and the audit trail are intact; the
                # only thing forgone is a possible visibility upgrade of one
                # cell. Measured incidence: 1 spot of 2,550 folded spots on
                # cancer, 0 on rice and on the benchmark.
                stats.n_cell_plans_unresolved += 1
                _choose_representative(spot, spectra)
                recompute_alignment_center(spot, gap_config)
                continue
            _merge_cluster(spot, [], cell_plans, spectra, stats, gap_config)
        else:
            # Do this for every spot, including singletons: the joiner's
            # strongest spectrum can belong to a row Stage 4-6 already hid.
            # A visible member must represent the combined row before keeper
            # priority is evaluated.
            _choose_representative(spot, spectra)
            recompute_alignment_center(spot, gap_config)

    # Visible spots are considered first.  A pre-hidden row may be absorbed by
    # a visible keeper, but can never establish the slot that hides one.
    ordered = sorted(spots, key=lambda spot: _spot_priority(spot, is_annotated))
    clusters: list[list[AlignmentSpot]] = []
    protected_clusters: set[int] = set()
    cluster_index = _ClusterIndex(refiner_config)

    def is_protected_isotope(spot: AlignmentSpot) -> bool:
        """Keep isotope satellites and internally mixed roles out of merging.

        A fold can carry the opposite isotope role from the representative.
        Looking only at ``representative.isotope_index`` therefore lets a
        mixed M/M+1 spot whose representative happens to be M absorb another
        monoisotopic row.  Completion already requires one uniform role; apply
        the same conservative guard to the ordinary reconciliation pass.
        """
        role = _uniform_isotope_index(spot)
        return role is None or (
            role > 0 and not is_annotated(spot.representative)
        )

    for spot in ordered:
        # Preserve the refiner's safety rule for unidentified isotope
        # satellites, and quarantine spots whose conserved cells disagree on
        # isotope role.  Neither may claim a slot or be folded into a neighbour.
        if is_protected_isotope(spot):
            clusters.append([spot])
            cluster_idx = len(clusters) - 1
            protected_clusters.add(cluster_idx)
            cluster_index.add(cluster_idx, spot)
            continue

        destination = None
        for cluster_idx in cluster_index.candidates(spot):
            # A protected isotope/mixed-role spot cannot establish a keeper
            # slot either.  Without this reverse guard, a later visible row
            # could still be folded into the protected singleton above.
            if cluster_idx in protected_clusters:
                continue
            cluster = clusters[cluster_idx]
            stats.n_candidate_clusters_examined += 1
            if all(_spots_are_mergeable(
                spot,
                member,
                spectra,
                joiner_config,
                refiner_config,
                stats,
            ) for member in cluster):
                cell_plans, cell_state = _cluster_cell_plans(
                    [*cluster, spot], spectra, joiner_config,
                )
                if cell_plans is None:
                    if cell_state is IdentityState.DIFFERENT:
                        stats.n_identity_different_rejected += 1
                    else:
                        stats.n_identity_unjudgeable_rejected += 1
                    continue
                destination = cluster
                break
        if destination is None:
            clusters.append([spot])
            cluster_index.add(len(clusters) - 1, spot)
        else:
            destination.append(spot)

    reconciled: list[AlignmentSpot] = []
    for cluster in clusters:
        keeper = cluster[0]
        if len(cluster) > 1:
            cell_plans, cell_state = _cluster_cell_plans(
                cluster, spectra, joiner_config,
            )
            if cell_plans is None:  # pragma: no cover - checked on insertion
                raise RuntimeError(
                    f"reconciliation cluster lost its cell plan: {cell_state}"
                )
            _merge_cluster(
                keeper, cluster[1:], cell_plans, spectra, stats, gap_config,
            )
        else:
            recompute_alignment_center(keeper, gap_config)
        reconciled.append(keeper)

    completed = _complete_disjoint_ms1_spots(
        reconciled,
        spectra,
        joiner_config,
        refiner_config,
        is_annotated,
        stats,
        gap_config,
    )
    stats.n_output_spots = len(completed)
    return completed, stats


def spots_from_clusters(
    clusters: list[Cluster],
    identity_config: IdentityConfig,
    *,
    ms2_reader: Optional[Ms2Reader] = None,
    gap_config: Optional[GapFillConfig] = None,
    cache_size: int = 4096,
) -> list[AlignmentSpot]:
    """Turn compound clusters into the one-cell-per-sample rows Stage 7 carries.

    Clustering deliberately lets one sample contribute several observations —
    "one cell per sample" is a *quantitation* constraint, not a grouping one, and
    forcing it during grouping is what made the old joiner need a matching
    algorithm plus a conservation loop to rescue its own losers.

    Resolving it here is therefore the whole cost of that freedom, and for now it
    is resolved the cheap way: the highest-priority observation takes the cell and
    the rest are folded, each naming the cell it lost to and carrying the verdict
    that says they are the same compound. Nothing is re-integrated yet — until a
    row has one quantitation channel, there is no ion to re-integrate *on*, and
    picking one is the next step.
    """
    spectra = _Spectra(ms2_reader, cache_size)
    spots: list[AlignmentSpot] = []
    for index, cluster in enumerate(clusters):
        by_sample: dict[str, list[CandidateFeature]] = {}
        for sample_id, peak in cluster.members:
            by_sample.setdefault(sample_id, []).append(peak)

        founder_sample, founder = cluster.members[0]
        spot = AlignmentSpot(
            index=index,
            master_mz=float(founder.align_mz),
            master_rt=float(founder.rt_apex),
            origin_sample_id=founder_sample,
        )
        for sample_id in sorted(by_sample):
            candidates = sorted(by_sample[sample_id], key=_cell_priority)
            winner = candidates[0]
            spot.peaks[sample_id] = winner
            for loser in candidates[1:]:
                verdict = identity(
                    observation_of(loser, lambda p=loser: spectra.get(sample_id, p)),
                    observation_of(winner, lambda p=winner: spectra.get(sample_id, p)),
                    identity_config,
                )
                spot.folded_peaks.append(FoldedPeak(
                    sample_id=sample_id,
                    peak=loser,
                    reason="same_compound_cluster_member",
                    target_peak_id=str(winner.feature_id),
                    cosine=float(verdict.cosine),
                    n_matched_fragments=int(verdict.n_matched_fragments),
                    evidence_kind="tiered_identity",
                    evidence_details={"tier": int(verdict.tier)},
                ))
        _choose_representative(spot, _Spectra(ms2_reader, cache_size))
        recompute_alignment_center(spot, gap_config, vote_window=True)
        spots.append(spot)
    return spots


def audit_conservation(
    features_by_sample: dict[str, list[CandidateFeature]],
    spots: list[AlignmentSpot],
) -> ConservationAudit:
    """Recheck natural-peak conservation after all same-route reconciliation."""
    input_tokens = {
        (sample_id, id(peak))
        for sample_id, peaks in features_by_sample.items()
        for peak in peaks
        if peak.align_mz is not None
    }
    assigned_tokens = {
        (sample_id, id(peak))
        for spot in spots
        for sample_id, peak in spot.peaks.items()
        if peak.gap_fill_status == "detected"
    }
    folded_tokens = {
        (fold.sample_id, id(fold.peak))
        for spot in spots
        for fold in spot.folded_peaks
    }
    unexplained = input_tokens - assigned_tokens - folded_tokens
    return ConservationAudit(
        n_input_keyed_peaks=len(input_tokens),
        n_assigned_peaks=len(assigned_tokens),
        n_collapsed_same_compound=len(folded_tokens),
        n_unexplained_lost=len(unexplained),
    )


def _digest(value) -> str:
    blob = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def audit_spots(
    spots: list[AlignmentSpot],
    joiner_config: JoinerConfig,
    ms2_reader: Optional[Ms2Reader] = None,
    gap_config: Optional[GapFillConfig] = None,
    *,
    identity_config: Optional[IdentityConfig] = None,
    vote_window: bool = False,
    is_annotated: Optional[Callable[[CandidateFeature], bool]] = None,
) -> SpotAudit:
    """Audit final cells without depending on transient object identities.

    ``vote_window`` must match what the run used (see
    :func:`recompute_alignment_center`), or the consensus checks audit a contract
    the run never signed.
    """
    cell_statuses = {"detected": 0, "filled": 0, "no_signal": 0}
    membership: list[list[str]] = []
    assigned_tokens: list[str] = []
    source_tokens: list[str] = []
    seen_source_tokens: set[str] = set()
    duplicate_sources = 0
    mixed_routes = product_multi_window = product_multi_segment = 0
    low_evidence = missing_targets = identity_not_same = evidence_mismatch = 0
    ms1_evidence_mismatch = 0
    hidden_keepers = 0
    rt_mismatch = mz_mismatch = window_mismatch = segment_mismatch = 0
    window_span_violations = segment_mismatch_spots = cross_route_spots = 0
    conflicting_annotation_spots = 0
    conflicting_examples: list[tuple] = []
    n_folds = 0

    spectra = _Spectra(ms2_reader, joiner_config.ms2_cache_size)
    for spot in spots:
        natural: list[tuple[str, CandidateFeature]] = [
            (sample_id, peak)
            for sample_id, peak in spot.peaks.items()
            if peak.gap_fill_status == "detected"
        ] + [
            (fold.sample_id, fold.peak) for fold in spot.folded_peaks
        ]
        group_tokens = sorted(
            f"{sample_id}\0{peak.feature_id}" for sample_id, peak in natural
        )
        membership.append(group_tokens)
        source_tokens.extend(group_tokens)
        for token in group_tokens:
            if token in seen_source_tokens:
                duplicate_sources += 1
            seen_source_tokens.add(token)

        for sample_id, peak in spot.peaks.items():
            status = peak.gap_fill_status
            if status in cell_statuses:
                cell_statuses[status] += 1
            if status == "detected":
                assigned_tokens.append(f"{sample_id}\0{peak.feature_id}")

        routes = {peak.quant_route for _sample_id, peak in natural}
        if len(routes) > 1:
            mixed_routes += 1
        if len({str(peak.signal_type) for _sample_id, peak in natural}) > 1:
            cross_route_spots += 1
        product_peaks = [
            peak for _sample_id, peak in natural if peak.quant_route == PRODUCT
        ]
        if len({int(peak.precursor_mz_nominal) for peak in product_peaks}) > 1:
            product_multi_window += 1
        segments = {str(peak.segment_name or "") for peak in product_peaks}
        segments.discard("")
        if len(segments) > 1:
            product_multi_segment += 1

        if is_annotated is not None:
            names = {
                str(
                    peak.annotation_matches[int(peak.selected_annotation_idx)].name
                    or ""
                )
                for _sample_id, peak in natural
                if peak.annotation_matches
                and 0 <= int(peak.selected_annotation_idx)
                < len(peak.annotation_matches)
                and is_annotated(peak)
            }
            names.discard("")
            if len(names) > 1:
                conflicting_annotation_spots += 1
                if len(conflicting_examples) < 50:
                    conflicting_examples.append((spot.index, tuple(sorted(names))))

        # INV-3 / INV-4, over *every* natural member rather than the PRODUCT ones.
        windows = {int(peak.precursor_mz_nominal) for _sample_id, peak in natural}
        if windows and max(windows) - min(windows) > 1:
            window_span_violations += 1
        all_segments = {
            str(peak.segment_name or "") for _sample_id, peak in natural
        }
        all_segments.discard("")
        if len(all_segments) > 1:
            segment_mismatch_spots += 1

        # A fold may name another fold of the same sample rather than the
        # assigned cell: an MS1 alias witness points at the row it is an alias
        # *of*, and that row can afterwards lose the claim and be folded itself.
        # The chain is still sound — the evidence on each link is checked below
        # against the peak that link actually names — so resolve through it
        # rather than calling the record orphaned.
        folds_by_id = {
            (fold.sample_id, str(fold.peak.feature_id)): fold
            for fold in spot.folded_peaks
        }

        def _fold_target(fold: FoldedPeak) -> Optional[CandidateFeature]:
            """The peak this fold names, whether it is a cell or another fold."""
            assigned = spot.peaks.get(fold.sample_id)
            if (
                assigned is not None
                and assigned.gap_fill_status == "detected"
                and str(fold.target_peak_id) == str(assigned.feature_id)
            ):
                return assigned
            seen: set[str] = {str(fold.peak.feature_id)}
            link = folds_by_id.get((fold.sample_id, str(fold.target_peak_id)))
            while link is not None:
                token = str(link.peak.feature_id)
                if token in seen:
                    return None                     # a cycle is a broken chain
                seen.add(token)
                if (
                    assigned is not None
                    and assigned.gap_fill_status == "detected"
                    and str(link.target_peak_id) == str(assigned.feature_id)
                ):
                    return link.peak
                link = folds_by_id.get((fold.sample_id, str(link.target_peak_id)))
            return None

        for fold in spot.folded_peaks:
            n_folds += 1
            assigned = _fold_target(fold)
            if assigned is None:
                missing_targets += 1
                continue
            if fold.evidence_kind == "ms1_natural_peak":
                evidence = _ms1_natural_peak_evidence(fold.peak, assigned)
                details_match = evidence is not None and all(
                    math.isclose(
                        float(fold.evidence_details.get(key, float("nan"))),
                        float(value),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    for key, value in evidence.details.items()
                )
                details_match = (
                    details_match
                    and fold.n_matched_fragments == 0
                    and math.isclose(
                        float(fold.cosine), 0.0, rel_tol=0.0, abs_tol=1e-12,
                    )
                )
                if not details_match:
                    evidence_mismatch += 1
                    ms1_evidence_mismatch += 1
                continue
            if fold.evidence_kind == "tiered_identity":
                # Same rule the clustering used, re-run on the pair the record
                # names. Verified rather than trusted: the fold is the only
                # remaining trace of an observation that reaches no cell.
                verdict = identity(
                    observation_of(
                        fold.peak,
                        lambda p=fold: spectra.get(p.sample_id, p.peak),
                    ),
                    observation_of(
                        assigned,
                        lambda s=fold.sample_id, p=assigned: spectra.get(s, p),
                    ),
                    identity_config or IdentityConfig(),
                )
                if verdict.state is not IdentityState.SAME:
                    identity_not_same += 1
                if (
                    verdict.n_matched_fragments != fold.n_matched_fragments
                    or not math.isclose(
                        verdict.cosine, fold.cosine, rel_tol=0.0, abs_tol=1e-12,
                    )
                ):
                    evidence_mismatch += 1
                continue
            if fold.evidence_kind != "ms2_identity":
                evidence_mismatch += 1
                continue
            evidence = _identity(
                fold.sample_id,
                fold.peak,
                fold.sample_id,
                assigned,
                spectra,
                joiner_config,
            )
            if evidence.n_matched_fragments < int(
                joiner_config.ms2_identity_min_matched_fragments
            ):
                low_evidence += 1
            if evidence.state is not IdentityState.SAME:
                identity_not_same += 1
            if (
                evidence.n_matched_fragments != fold.n_matched_fragments
                or not math.isclose(
                    evidence.cosine, fold.cosine, rel_tol=0.0, abs_tol=1e-12,
                )
            ):
                evidence_mismatch += 1

        rep = spot.representative
        rep_hidden = bool(rep.is_duplicate)
        has_visible_member = any(
            not peak.is_duplicate
            for _sample_id, peak in natural
        )
        if rep_hidden and has_visible_member:
            hidden_keepers += 1

        detected = spot.detected_peaks
        if detected:
            expected_rt = float(np.mean([peak.rt_apex for peak in detected]))
            tallest = min(
                detected,
                key=lambda peak: (-(peak.ms1_height or 0.0), str(peak.feature_id)),
            )
            # The consensus ion is a vote across the detected cells when a
            # gap-fill view is available, and the tallest cell's ion when it is
            # not — audit whichever contract the run actually used.
            expected_ion = (
                choose_quant_channel(
                    detected,
                    spot.peaks.get(spot.representative_sample_id)
                    if spot.representative_sample_id else None,
                    gap_config,
                )
                if gap_config is not None else None
            )
            expected_mz = (
                float(expected_ion.mz) if expected_ion is not None
                else float(tallest.align_mz) if tallest.align_mz is not None
                else None
            )
            if spot.alignment_rt is None or not math.isclose(
                float(spot.alignment_rt), expected_rt, rel_tol=0.0, abs_tol=1e-12,
            ):
                rt_mismatch += 1
            if expected_mz is None or spot.alignment_mz is None or not math.isclose(
                float(spot.alignment_mz), expected_mz, rel_tol=0.0, abs_tol=1e-12,
            ):
                mz_mismatch += 1
            expected_window, expected_segment = consensus_window_segment(
                detected, vote_window,
            )
            if spot.alignment_window != expected_window:
                window_mismatch += 1
            if spot.alignment_segment != expected_segment:
                segment_mismatch += 1

    membership.sort()
    assigned_tokens.sort()
    source_tokens.sort()
    return SpotAudit(
        n_spots=len(spots),
        n_detected_cells=cell_statuses["detected"],
        n_filled_cells=cell_statuses["filled"],
        n_no_signal_cells=cell_statuses["no_signal"],
        n_fold_records=n_folds,
        n_mixed_quant_route_spots=mixed_routes,
        n_product_multi_window_spots=product_multi_window,
        n_product_multi_segment_spots=product_multi_segment,
        n_same_folds_below_min_matched=low_evidence,
        n_fold_records_missing_target=missing_targets,
        n_fold_identity_not_same=identity_not_same,
        n_fold_evidence_mismatch=evidence_mismatch,
        n_ms1_natural_peak_evidence_mismatch=ms1_evidence_mismatch,
        n_hidden_keeper_with_visible_member=hidden_keepers,
        n_duplicate_source_tokens=duplicate_sources,
        n_consensus_rt_mismatch=rt_mismatch,
        n_consensus_mz_mismatch=mz_mismatch,
        n_consensus_window_mismatch=window_mismatch,
        n_consensus_segment_mismatch=segment_mismatch,
        membership_sha256=_digest(membership),
        assigned_detected_sha256=_digest(assigned_tokens),
        natural_source_sha256=_digest(source_tokens),
        n_window_span_violations=window_span_violations,
        n_segment_mismatch_spots=segment_mismatch_spots,
        n_cross_route_spots=cross_route_spots,
        n_conflicting_annotation_spots=conflicting_annotation_spots,
        conflicting_annotation_examples=tuple(conflicting_examples),
    )
