"""Stage 2: MS1 precise m/z assignment (batch per-channel with exclusive matching)."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional, Callable

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import RawSegmentData, CandidateFeature
from asfam.core.eic import extract_ms1_eic, extract_ms1_precise_mz
from asfam.core.peak_detection import detect_peaks
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

        # Group features by (segment_name, channel) for batch processing
        channel_groups: dict[tuple[str, int], list[int]] = defaultdict(list)
        for i, feat in enumerate(features):
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
                        relaxed_peaks = _detect_ms1_peaks_relaxed(raw_data, channel)
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
                    _assign_representative_ion(features[fi])
                    n_low += 1

        if progress_callback:
            progress_callback("stage2", len(features), len(features),
                              f"Rep {rep_id}: done")

        logger.info(
            "  Replicate %s: %d ms1_detected, %d ms2_only (%d with MS1 m/z)",
            rep_id, n_high, n_low, n_low_with_ms1,
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
        precursor = feat.precursor_mz_nominal

        precise_mz = extract_ms1_precise_mz(
            raw_data, peak.apex_index,
            float(precursor) - 0.5, float(precursor) + 0.5,
        )
        if precise_mz is None:
            continue

        feat.ms1_precursor_mz = precise_mz
        feat.ms1_height = peak.height
        feat.ms1_area = peak.area
        feat.ms1_sn = peak.sn_ratio
        feat.signal_type = "ms1_detected"
        feat.mz_source = "ms1_peak"
        feat.ms1_isotopes = _extract_isotope_pattern(
            raw_data, peak.apex_index, precise_mz, config,
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
        raw_data, float(precursor_nominal), mz_tolerance=0.5,
    )
    if len(int_ms1) == 0 or np.max(int_ms1) < config.ms1_min_height:
        return []

    return detect_peaks(
        rt_ms1, int_ms1,
        precursor_mz_nominal=precursor_nominal,
        height_threshold=config.ms1_min_height,
        sn_threshold=config.peak_sn_threshold,
        width_min=config.peak_width_min,
        prominence=config.ms1_min_height * 0.5,
        compute_gaussian=True,
        gaussian_threshold=config.peak_gaussian_threshold,
    )


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
) -> list:
    """Detect MS1 peaks with relaxed parameters for weak-signal assignment."""
    rt_ms1, int_ms1 = extract_ms1_eic(
        raw_data, float(precursor_nominal), mz_tolerance=0.5,
    )
    if len(int_ms1) == 0 or np.max(int_ms1) < 50:
        return []

    return detect_peaks(
        rt_ms1, int_ms1,
        precursor_mz_nominal=precursor_nominal,
        height_threshold=50.0,
        sn_threshold=2.0,
        width_min=3,
        prominence=30.0,
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

    precise_mz = extract_ms1_precise_mz(
        raw_data, best_peak.apex_index,
        float(feature.precursor_mz_nominal) - 0.5,
        float(feature.precursor_mz_nominal) + 0.5,
    )
    if precise_mz is None:
        return False

    # Assign precise m/z
    feature.ms1_precursor_mz = precise_mz

    # If the MS1 peak is strong enough, promote to ms1_detected
    if (best_peak.height >= config.ms1_min_height and
            best_peak.sn_ratio >= config.peak_sn_threshold):
        feature.signal_type = "ms1_detected"
        feature.mz_source = "ms1_peak"
        feature.ms1_height = best_peak.height
        feature.ms1_area = best_peak.area
        feature.ms1_sn = best_peak.sn_ratio
        feature.ms1_isotopes = _extract_isotope_pattern(
            raw_data, best_peak.apex_index, precise_mz, config,
        )
    else:
        feature.mz_source = "ms1_relaxed"

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
