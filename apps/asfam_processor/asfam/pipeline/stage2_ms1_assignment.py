"""Stage 2: MS1 precise m/z assignment (batch per-channel with exclusive matching)."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional, Callable

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import RawSegmentData, CandidateFeature
from asfam.core.eic import extract_ms1_eic, extract_ms1_precise_mz
from asfam.core.peak_detection import detect_peaks, detect_chrom_peaks
from asfam.core.mass_utils import peak_overlap_ratio
from asfam.constants import C13_DELTA

logger = logging.getLogger(__name__)


def run_stage2(
    data_by_replicate: dict[str, list[RawSegmentData]],
    features_by_replicate: dict[str, list[CandidateFeature]],
    config: ProcessingConfig,
    progress_callback: Optional[Callable] = None,
) -> dict[str, list[CandidateFeature]]:
    """Assign precise MS1 m/z to each candidate feature.

    Uses batch assignment per channel with exclusive matching:
    each MS1 peak can only be assigned to one MS2 feature.
    """
    logger.info("Stage 2: MS1 precise m/z assignment...")

    raw_lookup: dict[tuple[str, int], RawSegmentData] = {}
    for rep_id, segments in data_by_replicate.items():
        for seg in segments:
            raw_lookup[(seg.segment_name, seg.replicate_id)] = seg

    for rep_id, features in features_by_replicate.items():
        n_high = 0
        n_low = 0
        n_low_with_ms1 = 0
        n_window = 0

        # Group features by (segment_name, channel) for batch processing
        channel_groups: dict[tuple[str, int], list[int]] = defaultdict(list)
        n_skipped = 0
        for i, feat in enumerate(features):
            # PR-C: MS1-driven features already carry a precise ROI-centroid m/z from
            # the stage1b mass-slice finder; Stage 2's coarse channel±0.5 reassignment
            # would clobber it. Skip — they are already fully assigned.
            if feat.detection_source == "ms1_driven" and feat.ms1_precursor_mz is not None:
                n_skipped += 1
                continue
            key = (feat.segment_name, feat.precursor_mz_nominal)
            channel_groups[key].append(i)
        ms1_cache: dict[tuple[str, int], tuple[list, RawSegmentData]] = {}
        ms1_cache_relaxed: dict[tuple[str, int], tuple[list, RawSegmentData]] = {}

        for (seg_name, channel), feat_indices in channel_groups.items():
            sample_feat = features[feat_indices[0]]
            raw_data = raw_lookup.get((seg_name, sample_feat.replicate_id))
            if raw_data is None:
                for fi in feat_indices:
                    features[fi].signal_type = "ms2_only"
                    _assign_representative_ion(features[fi])
                    n_low += 1
                continue

            cache_key = (seg_name, channel)

            # Every feature of this window learns the window's recorded centre
            # (the m/z the instrument actually isolated around; the integer
            # channel is only its floor). Stage 2.5 and alignment, which never
            # see raw data, read the window from this field.
            target = raw_data.isolation_target(channel)
            for fi in feat_indices:
                if getattr(features[fi], "isolation_target_mz", None) is None:
                    features[fi].isolation_target_mz = target

            # Detect MS1 peaks (strict)
            if cache_key not in ms1_cache:
                ms1_peaks = _detect_ms1_peaks(raw_data, channel, config)
                ms1_cache[cache_key] = (ms1_peaks, raw_data)
            ms1_peaks, raw_data = ms1_cache[cache_key]

            # Batch exclusive assignment (strict pass)
            assigned_feat_set, assigned_peak_set = _batch_assign_ms1(
                features, feat_indices, ms1_peaks, raw_data, config,
            )

            # Collect apex indices of strictly-assigned MS1 peaks
            # so the relaxed pass won't re-assign them
            used_apex_indices: set[int] = set()
            for pi in assigned_peak_set:
                used_apex_indices.add(ms1_peaks[pi].apex_index)

            for fi in feat_indices:
                if fi in assigned_feat_set:
                    n_high += 1
                else:
                    # Relaxed pass for unassigned features
                    if cache_key not in ms1_cache_relaxed:
                        relaxed_peaks = _detect_ms1_peaks_relaxed(raw_data, channel, config)
                        ms1_cache_relaxed[cache_key] = (relaxed_peaks, raw_data)
                    relaxed_peaks, _ = ms1_cache_relaxed[cache_key]
                    # Filter out relaxed peaks whose apex overlaps a strict-assigned peak
                    available_relaxed = [
                        p for p in relaxed_peaks
                        if p.apex_index not in used_apex_indices
                    ]
                    assigned = _try_relaxed_ms1_assign(
                        features[fi], available_relaxed, raw_data, config,
                    )
                    if assigned:
                        n_low_with_ms1 += 1
                        # Track this peak too for subsequent features
                        if features[fi].signal_type == "ms1_detected":
                            # Find which relaxed peak was used and track it
                            for p in available_relaxed:
                                if abs(p.rt_apex - features[fi].rt_apex) < config.ms1_rt_tolerance * 1.5:
                                    if (features[fi].ms1_height is not None and
                                            abs(p.height - features[fi].ms1_height) < 1.0):
                                        used_apex_indices.add(p.apex_index)
                                        break
                            n_high += 1
                            continue
                    # Isolation-window centroid: the +/-0.5 Da MS1 centroid at
                    # the apex cycle is only *stored* here (``ms1_window_mz``),
                    # because stage 2 is the last stage that holds raw data.
                    # Stage 2.5 tries neutral-loss consensus first and falls
                    # back to this value last (``mz_source="ms1_lowres"``).
                    # Writing it straight into ``ms1_precursor_mz`` here, as the
                    # 2026-06-25 fallback did, pre-empted stage 2.5 for nearly
                    # every MS2-only feature (its entry filter is "no source
                    # yet"). The centroid is also kept on relaxed-assigned
                    # features purely as a cross-check column.
                    if _store_window_centroid(features[fi], raw_data):
                        n_window += 1
                    _assign_representative_ion(features[fi])
                    n_low += 1

        if progress_callback:
            progress_callback("stage2", len(features), len(features),
                              f"Rep {rep_id}: done")

        logger.info(
            "  Replicate %s: %d ms1_detected, %d ms2_only (%d relaxed-assigned, "
            "%d window centroid stored for stage 2.5); "
            "%d ms1_driven passthrough (already precise)",
            rep_id, n_high, n_low, n_low_with_ms1, n_window, n_skipped,
        )

    return features_by_replicate


def _batch_assign_ms1(
    features: list[CandidateFeature],
    feat_indices: list[int],
    ms1_peaks: list,
    raw_data: RawSegmentData,
    config: ProcessingConfig,
) -> tuple[set[int], set[int]]:
    """Batch-assign MS1 peaks to features with exclusivity.

    Each MS1 peak assigned to at most one feature. Scoring uses RT proximity
    and peak overlap ratio. Greedy best-first assignment.

    Returns (assigned_feature_indices, assigned_peak_indices).
    """
    if not ms1_peaks:
        for fi in feat_indices:
            features[fi].signal_type = "ms2_only"
        return set(), set()

    w_shape = config.ms1_shape_weight
    w_rt = 1.0 - w_shape

    # Build all candidate (feature_idx, peak_idx, score) triples
    triples: list[tuple[float, int, int]] = []
    for fi in feat_indices:
        feat = features[fi]
        for pi, peak in enumerate(ms1_peaks):
            rt_diff = abs(peak.rt_apex - feat.rt_apex)
            if rt_diff > config.ms1_rt_tolerance:
                continue
            rt_score = 1.0 - rt_diff / config.ms1_rt_tolerance
            shape_score = peak_overlap_ratio(
                feat.rt_left, feat.rt_right, peak.rt_left, peak.rt_right,
            )
            score = w_rt * rt_score + w_shape * shape_score
            triples.append((score, fi, pi))

    # Greedy: best score first
    triples.sort(key=lambda x: -x[0])

    assigned_features: set[int] = set()
    assigned_peaks: set[int] = set()

    for score, fi, pi in triples:
        if fi in assigned_features or pi in assigned_peaks:
            continue

        feat = features[fi]
        peak = ms1_peaks[pi]
        # The window the instrument isolated is centred on the recorded
        # target, not on the integer channel (its floor) -- see
        # RawSegmentData.isolation_target. Same window as the chromatogram
        # the peak was picked on (_detect_ms1_peaks).
        centre = raw_data.isolation_target(feat.precursor_mz_nominal)
        feat.isolation_target_mz = centre

        precise_mz = extract_ms1_precise_mz(
            raw_data, peak.apex_index, centre - 0.5, centre + 0.5,
        )
        if precise_mz is None:
            continue

        base_mz = _basepeak_mz(
            raw_data, peak.apex_index, centre - 0.5, centre + 0.5,
        )
        feat.ms1_quant_mz = base_mz
        feat.ms1_window_mz = precise_mz          # whole-window centroid, cross-check column
        feat.ms1_precursor_mz = _reported_precursor(precise_mz, base_mz, config)
        feat.ms1_height = peak.height
        feat.ms1_area = peak.area
        feat.ms1_sn = peak.sn_ratio
        feat.ms1_gaussian = peak.gaussian_similarity
        feat.signal_type = "ms1_detected"
        feat.mz_source = "ms1_peak"
        feat.ms1_isotopes = _extract_isotope_pattern(
            raw_data, peak.apex_index, feat.ms1_precursor_mz, config,
        )

        assigned_features.add(fi)
        assigned_peaks.add(pi)

    # Mark unassigned as ms2_only
    for fi in feat_indices:
        if fi not in assigned_features:
            features[fi].signal_type = "ms2_only"

    return assigned_features, assigned_peaks


def _detect_ms1_peaks(
    raw_data: RawSegmentData,
    precursor_nominal: int,
    config: ProcessingConfig,
) -> list:
    """Extract MS1 EIC and detect peaks for one channel. Called once per channel."""
    rt_ms1, int_ms1 = extract_ms1_eic(
        raw_data, raw_data.isolation_target(precursor_nominal), mz_tolerance=0.5,
    )
    if len(int_ms1) == 0 or np.max(int_ms1) < config.ms1_min_height:
        return []

    return detect_chrom_peaks(
        rt_ms1, int_ms1,
        config=config,
        precursor_mz_nominal=precursor_nominal,
        min_amplitude=config.ms1_min_height,
        min_data_points=config.ms1_peak.min_data_points,
        compute_gaussian=True,
        gaussian_threshold=config.ms1_peak.gaussian_threshold,
        min_prominence_ratio=config.ms1_peak.min_prominence_ratio,
        rt_window_min=config.ms1_peak.rt_window_min,
        rt_window_max=config.ms1_peak.rt_window_max,
    )


def _basepeak_mz(
    raw_data: RawSegmentData,
    cycle_index: int,
    mz_low: float,
    mz_high: float,
) -> Optional[float]:
    """m/z of the tallest MS1 centroid in ``[mz_low, mz_high]`` at one cycle.

    ``_detect_ms1_peaks`` quantifies on ``extract_ms1_eic(raw, target, 0.5)``
    with ``target = raw.isolation_target(channel)``, whose value at each cycle
    is exactly this centroid's intensity -- provided both are handed the same
    window, so callers centre both on the recorded target, never on the bare
    integer channel. So this is
    the ion the feature's ``ms1_height`` was measured on, and the one gap
    filling must integrate to produce a comparable number. Its sibling
    ``extract_ms1_precise_mz`` returns the intensity-weighted centroid of the
    same window, which is a better *precursor* estimate but lands on empty m/z
    whenever the window holds more than one ion.
    """
    cycle = raw_data.cycles[cycle_index]
    ms1_mz = cycle.ms1_mz
    ms1_int = cycle.ms1_intensity
    if ms1_mz is None or len(ms1_mz) == 0:
        return None
    mask = (ms1_mz >= mz_low) & (ms1_mz <= mz_high)
    if not np.any(mask):
        return None
    return float(ms1_mz[mask][int(np.argmax(ms1_int[mask]))])


def _extract_isotope_pattern(
    raw_data: RawSegmentData,
    cycle_index: int,
    mono_mz: float,
    config: ProcessingConfig,
    max_isotopes: int = 5,
) -> list[tuple[float, float]]:
    """Extract MS1 isotope peaks at a given cycle."""
    cycle = raw_data.cycles[cycle_index]
    ms1_mz = cycle.ms1_mz
    ms1_int = cycle.ms1_intensity
    tol = config.ms1_isotope_mz_tol

    isotopes = []
    for n in range(max_isotopes):
        target_mz = mono_mz + n * C13_DELTA
        mask = np.abs(ms1_mz - target_mz) <= tol
        if np.any(mask):
            best_idx = np.argmax(ms1_int[mask])
            mz_val = float(ms1_mz[mask][best_idx])
            int_val = float(ms1_int[mask][best_idx])
            if int_val > 0:
                isotopes.append((mz_val, int_val))
            else:
                break
        else:
            break
    return isotopes


def _detect_ms1_peaks_relaxed(
    raw_data: RawSegmentData,
    precursor_nominal: int,
    config: ProcessingConfig,
) -> list:
    """Detect MS1 peaks with relaxed parameters for weak-signal assignment."""
    rt_ms1, int_ms1 = extract_ms1_eic(
        raw_data, raw_data.isolation_target(precursor_nominal), mz_tolerance=0.5,
    )
    if len(int_ms1) == 0 or np.max(int_ms1) < 50:
        return []

    return detect_chrom_peaks(
        rt_ms1, int_ms1,
        config=config,
        precursor_mz_nominal=precursor_nominal,
        min_amplitude=50.0,
        min_data_points=3,
        sn_fold=2.0,         # relaxed S/N gate for weak-signal recovery
        compute_gaussian=True,
        gaussian_threshold=0.3,
    )


def _try_relaxed_ms1_assign(
    feature: CandidateFeature,
    relaxed_peaks: list,
    raw_data: RawSegmentData,
    config: ProcessingConfig,
) -> bool:
    """Try to assign MS1 precise m/z from a relaxed peak list.

    Assigns ms1_precursor_mz only (for high-res m/z value).
    Keeps signal_type as ms2_only. Does NOT set ms1_height/sn/area
    since those come from the representative product ion.

    Returns True if a relaxed MS1 peak was found and m/z assigned.
    """
    if not relaxed_peaks:
        return False

    # Use wider RT tolerance for relaxed matching
    rt_tol = config.ms1_rt_tolerance * 1.5  # e.g., 0.15 min = 9s

    best_peak = None
    best_rt_diff = float("inf")
    for peak in relaxed_peaks:
        rt_diff = abs(peak.rt_apex - feature.rt_apex)
        if rt_diff <= rt_tol and rt_diff < best_rt_diff:
            best_peak = peak
            best_rt_diff = rt_diff

    if best_peak is None:
        return False

    centre = raw_data.isolation_target(feature.precursor_mz_nominal)
    feature.isolation_target_mz = centre
    precise_mz = extract_ms1_precise_mz(
        raw_data, best_peak.apex_index, centre - 0.5, centre + 0.5,
    )
    if precise_mz is None:
        return False

    # Assign precise m/z (window centroid at the relaxed peak's apex; the
    # promoted branch below may replace it with the quantified ion).
    feature.ms1_precursor_mz = precise_mz

    # If the MS1 peak is strong enough, promote to ms1_detected. The
    # SN gate is now applied inside detect_peaks (MS-DIAL three-gate),
    # so here we only re-check the absolute height floor.
    if best_peak.height >= config.ms1_min_height:
        feature.signal_type = "ms1_detected"
        feature.mz_source = "ms1_peak"
        base_mz = _basepeak_mz(
            raw_data, best_peak.apex_index, centre - 0.5, centre + 0.5,
        )
        feature.ms1_quant_mz = base_mz
        feature.ms1_window_mz = precise_mz
        feature.ms1_precursor_mz = _reported_precursor(precise_mz, base_mz, config)
        feature.ms1_height = best_peak.height
        feature.ms1_area = best_peak.area
        feature.ms1_sn = best_peak.sn_ratio
        feature.ms1_gaussian = best_peak.gaussian_similarity
        feature.ms1_isotopes = _extract_isotope_pattern(
            raw_data, best_peak.apex_index, feature.ms1_precursor_mz, config,
        )
    else:
        feature.mz_source = "ms1_relaxed"

    return True


def _reported_precursor(
    window_centroid: float,
    base_peak: Optional[float],
    config: ProcessingConfig,
) -> float:
    """The m/z an MS1-detected, MS2-driven feature reports as its precursor.

    ``config.ms1_precursor_report`` decides: ``"base_peak"`` (default since
    v1.4.2610xx) reports the tallest ion in the isolation window at the MS1 peak
    apex — the ion the feature was quantified on, a real measured m/z —
    ``"window_centroid"`` reports the intensity-weighted centroid of the whole
    1-Da window (pre-v1.4 behaviour; a mixture of every co-isolated ion). Falls
    back to the centroid when no base peak could be read.
    """
    mode = str(getattr(config, "ms1_precursor_report", "base_peak") or "base_peak")
    if mode == "window_centroid" or base_peak is None:
        return float(window_centroid)
    if mode != "base_peak":
        raise ValueError(
            f"ms1_precursor_report must be 'base_peak' or 'window_centroid', got {mode!r}"
        )
    return float(base_peak)


def window_centroid_mz(
    feature: CandidateFeature,
    raw_data: RawSegmentData,
) -> Optional[float]:
    """Intensity-weighted MS1 centroid of the 1-Da isolation window (centred on
    the recorded isolation target, not on the integer channel) at the feature's
    apex cycle, or ``None`` when the window holds no MS1 signal.

    ASFAM acquires a full-resolution MS1 scan every cycle, so even when the
    channel's MS1 is too weak for the strict (``ms1_min_height``) and relaxed
    (50) peak pickers, the window usually still carries signal at the apex.
    The value is a *mixture* centroid of everything co-isolated in the 1-Da
    window — a low-resolution precursor estimate, not a detected peak — and
    that is why it is the last fallback in stage 2.5 and is reported with
    ``mz_confidence="low"``.

    Public so the manuscript's precision analyses can call the production
    algorithm on MS1-detected features (whose strict peak m/z is the truth).
    """
    rt_array = raw_data.rt_array
    if rt_array is None or len(rt_array) == 0:
        return None
    apex_cycle = int(np.argmin(np.abs(rt_array - feature.rt_apex)))
    centre = raw_data.isolation_target(feature.precursor_mz_nominal)
    return extract_ms1_precise_mz(
        raw_data, apex_cycle, centre - 0.5, centre + 0.5,
    )


def _store_window_centroid(
    feature: CandidateFeature,
    raw_data: RawSegmentData,
) -> bool:
    """Store the window centroid on ``feature.ms1_window_mz`` and nothing else.

    Deliberately leaves ``ms1_precursor_mz`` / ``mz_source`` / ``mz_confidence``
    untouched: whether the centroid becomes the reported precursor is stage
    2.5's decision, taken only after neutral-loss consensus has failed or was
    not applicable (``run_stage2b``). Returns True when a centroid was stored.
    """
    mz = window_centroid_mz(feature, raw_data)
    if mz is None:
        return False
    feature.ms1_window_mz = float(mz)
    return True


def _assign_representative_ion(feature: CandidateFeature) -> None:
    """For ms2_only features, use the highest product ion as representative.

    Sets ms1_height, ms1_sn, and ms2_rep_ion_mz from the product ion
    with the highest intensity (ties broken by largest m/z).
    """
    if feature.ms2_intensity is None or len(feature.ms2_intensity) == 0:
        return

    max_int = np.max(feature.ms2_intensity)
    candidates = np.where(feature.ms2_intensity == max_int)[0]
    # Tie-break: largest m/z
    best_idx = candidates[np.argmax(feature.ms2_mz[candidates])]

    feature.ms1_height = float(feature.ms2_intensity[best_idx])
    feature.ms2_rep_ion_mz = float(feature.ms2_mz[best_idx])

    if feature.ms2_sn is not None and best_idx < len(feature.ms2_sn):
        feature.ms1_sn = float(feature.ms2_sn[best_idx])
