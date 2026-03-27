"""Stage 2: MS1 precise m/z assignment (optimized with per-channel caching)."""
from __future__ import annotations

import logging
from typing import Optional, Callable

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import RawSegmentData, CandidateFeature
from asfam.core.eic import extract_ms1_eic, extract_ms1_precise_mz
from asfam.core.peak_detection import detect_peaks
from asfam.constants import C13_DELTA

logger = logging.getLogger(__name__)


def run_stage2(
    data_by_replicate: dict[str, list[RawSegmentData]],
    features_by_replicate: dict[str, list[CandidateFeature]],
    config: ProcessingConfig,
    progress_callback: Optional[Callable] = None,
) -> dict[str, list[CandidateFeature]]:
    """Assign precise MS1 m/z to each candidate feature.

    Optimized: caches MS1 EIC and peak detection per (segment, channel)
    so each is computed only once instead of per-feature.
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

        # Cache: (segment_name, channel) -> (ms1_peaks, raw_data)
        ms1_cache: dict[tuple[str, int], tuple[list, RawSegmentData]] = {}
        # Relaxed cache for second pass
        ms1_cache_relaxed: dict[tuple[str, int], tuple[list, RawSegmentData]] = {}

        for i, feat in enumerate(features):
            raw_data = raw_lookup.get((feat.segment_name, feat.replicate_id))
            if raw_data is None:
                feat.signal_type = "ms2_only"
                _assign_representative_ion(feat)
                n_low += 1
                continue

            cache_key = (feat.segment_name, feat.precursor_mz_nominal)
            if cache_key not in ms1_cache:
                ms1_peaks = _detect_ms1_peaks(
                    raw_data, feat.precursor_mz_nominal, config,
                )
                ms1_cache[cache_key] = (ms1_peaks, raw_data)

            ms1_peaks, raw_data = ms1_cache[cache_key]
            _assign_ms1_from_cache(feat, ms1_peaks, raw_data, config)

            if feat.signal_type == "ms1_detected":
                n_high += 1
            else:
                # Relaxed second pass: try to find weaker MS1 peak for m/z assignment
                if cache_key not in ms1_cache_relaxed:
                    relaxed_peaks = _detect_ms1_peaks_relaxed(
                        raw_data, feat.precursor_mz_nominal,
                    )
                    ms1_cache_relaxed[cache_key] = (relaxed_peaks, raw_data)
                relaxed_peaks, raw_data = ms1_cache_relaxed[cache_key]
                assigned = _try_relaxed_ms1_assign(feat, relaxed_peaks, raw_data, config)
                if assigned:
                    n_low_with_ms1 += 1
                    if feat.signal_type == "ms1_detected":
                        # Promoted to high by relaxed pass
                        n_high += 1
                        continue
                _assign_representative_ion(feat)
                n_low += 1

            if progress_callback and (i + 1) % 500 == 0:
                progress_callback("stage2", i + 1, len(features),
                                  f"Rep {rep_id}: {i+1}/{len(features)}")

        logger.info(
            "  Replicate %s: %d ms1_detected, %d ms2_only (%d with MS1 m/z)",
            rep_id, n_high, n_low, n_low_with_ms1,
        )

    return features_by_replicate


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
    )


def _assign_ms1_from_cache(
    feature: CandidateFeature,
    ms1_peaks: list,
    raw_data: RawSegmentData,
    config: ProcessingConfig,
) -> None:
    """Assign MS1 to a feature using pre-computed MS1 peaks."""
    if not ms1_peaks:
        feature.signal_type = "ms2_only"
        return

    precursor = feature.precursor_mz_nominal

    # Find best matching MS1 peak by RT
    best_peak = None
    best_rt_diff = float("inf")
    for peak in ms1_peaks:
        rt_diff = abs(peak.rt_apex - feature.rt_apex)
        if rt_diff <= config.ms1_rt_tolerance and rt_diff < best_rt_diff:
            best_peak = peak
            best_rt_diff = rt_diff

    if best_peak is None:
        feature.signal_type = "ms2_only"
        return

    precise_mz = extract_ms1_precise_mz(
        raw_data, best_peak.apex_index,
        float(precursor) - 0.5, float(precursor) + 0.5,
    )
    if precise_mz is None:
        feature.signal_type = "ms2_only"
        return

    feature.ms1_precursor_mz = precise_mz
    feature.ms1_height = best_peak.height
    feature.ms1_area = best_peak.area
    feature.ms1_sn = best_peak.sn_ratio
    feature.signal_type = "ms1_detected"
    feature.mz_source = "ms1_peak"

    feature.ms1_isotopes = _extract_isotope_pattern(
        raw_data, best_peak.apex_index, precise_mz, config,
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
