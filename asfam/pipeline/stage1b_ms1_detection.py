"""Stage 1b: MS1-driven feature detection (complementary to MS2-driven Stage 1).

For each nominal m/z channel, detect peaks in the MS1 EIC. For each MS1 peak
that does NOT already match an MS2-driven feature, collect co-eluting MS2
product ions and create a new CandidateFeature.

This captures compounds visible in MS1 but missed by the MS2-driven approach.
"""
from __future__ import annotations

import logging
from typing import Optional, Callable

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import RawSegmentData, CandidateFeature
from asfam.core.eic import extract_ms1_eic, extract_ms1_precise_mz
from asfam.core.peak_detection import detect_peaks

logger = logging.getLogger(__name__)


def run_stage1b(
    data_by_replicate: dict[str, list[RawSegmentData]],
    existing_features: dict[str, list[CandidateFeature]],
    config: ProcessingConfig,
    progress_callback: Optional[Callable] = None,
) -> dict[str, list[CandidateFeature]]:
    """MS1-driven feature detection, complementary to MS2-driven Stage 1.

    For each replicate, scans every channel for MS1 peaks not already covered
    by MS2-driven features. Creates new CandidateFeature objects with
    detection_source="ms1_driven" and merges them into the existing list.

    Returns the updated features_by_replicate dict.
    """
    logger.info("Stage 1b: MS1-driven feature detection...")

    total_new = 0
    total_files = sum(len(segs) for segs in data_by_replicate.values())
    file_count = 0

    for rep_id, segments in data_by_replicate.items():
        existing = existing_features.get(rep_id, [])

        # Build index of existing features: channel -> list of RT apexes
        existing_index: dict[int, list[float]] = {}
        for f in existing:
            ch = f.precursor_mz_nominal
            if ch not in existing_index:
                existing_index[ch] = []
            existing_index[ch].append(f.rt_apex)

        new_features: list[CandidateFeature] = []

        for raw_data in segments:
            seg_new = _process_segment_ms1(
                raw_data, existing_index, config,
            )
            new_features.extend(seg_new)
            file_count += 1

            if progress_callback:
                progress_callback(
                    "stage1b", file_count, total_files,
                    f"Rep {rep_id} {raw_data.segment_name}: {len(seg_new)} new MS1-driven",
                )

        if new_features:
            existing_features[rep_id] = existing + new_features
            total_new += len(new_features)
            logger.info("  Replicate %s: %d new MS1-driven features", rep_id, len(new_features))
        else:
            logger.info("  Replicate %s: 0 new MS1-driven features", rep_id)

    logger.info("  Total new MS1-driven features: %d", total_new)
    return existing_features


def _process_segment_ms1(
    raw_data: RawSegmentData,
    existing_index: dict[int, list[float]],
    config: ProcessingConfig,
) -> list[CandidateFeature]:
    """Process one segment: detect MS1 peaks and build features for novel ones."""
    features: list[CandidateFeature] = []
    feature_counter = 0

    for channel in raw_data.precursor_list:
        # Extract MS1 EIC for this channel
        rt_arr, int_arr = extract_ms1_eic(raw_data, float(channel), mz_tolerance=0.5)
        if len(int_arr) == 0 or np.max(int_arr) < config.ms1_min_height:
            continue

        # Detect MS1 peaks
        ms1_peaks = detect_peaks(
            rt_arr, int_arr,
            precursor_mz_nominal=channel,
            height_threshold=config.ms1_min_height,
            sn_threshold=config.peak_sn_threshold,
            width_min=config.peak_width_min,
            prominence=config.ms1_min_height * 0.5,
        )

        if not ms1_peaks:
            continue

        # Check each MS1 peak against existing features
        existing_rts = existing_index.get(channel, [])

        for peak in ms1_peaks:
            # Skip if overlaps with existing feature (same channel, RT within tolerance)
            is_duplicate = any(
                abs(peak.rt_apex - ert) <= config.ms1_rt_tolerance
                for ert in existing_rts
            )
            if is_duplicate:
                continue

            # Novel MS1 peak: extract precise m/z
            precise_mz = extract_ms1_precise_mz(
                raw_data, peak.apex_index,
                float(channel) - 0.5, float(channel) + 0.5,
            )
            if precise_mz is None:
                continue

            # Collect co-eluting MS2 product ions in the peak RT range
            ms2_mz, ms2_intensity = _collect_ms2_at_peak(
                raw_data, channel, peak.apex_index,
                peak.left_index, peak.right_index,
                config,
            )

            n_frags = len(ms2_mz)
            # Require at least 1 fragment (MS1-driven can have fewer frags)
            if n_frags < 1:
                continue

            feature_id = f"{raw_data.segment_name}_{channel}_ms1_{feature_counter}"
            feature_counter += 1

            feat = CandidateFeature(
                feature_id=feature_id,
                segment_name=raw_data.segment_name,
                replicate_id=raw_data.replicate_id,
                precursor_mz_nominal=channel,
                rt_apex=float(rt_arr[peak.apex_index]),
                rt_left=float(rt_arr[peak.left_index]),
                rt_right=float(rt_arr[peak.right_index]),
                ms2_mz=ms2_mz,
                ms2_intensity=ms2_intensity,
                n_fragments=n_frags,
                ms1_precursor_mz=precise_mz,
                ms1_height=peak.height,
                ms1_area=peak.area,
                ms1_sn=peak.sn_ratio,
                signal_type="ms1_detected",
                detection_source="ms1_driven",
                mz_source="ms1_peak",
                source_file=raw_data.file_path,
            )
            features.append(feat)

    return features


def _collect_ms2_at_peak(
    raw_data: RawSegmentData,
    channel: int,
    apex_idx: int,
    left_idx: int,
    right_idx: int,
    config: ProcessingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect MS2 product ions that co-elute with an MS1 peak.

    For each candidate product ion, builds its EIC within the peak range
    and checks that it actually peaks near the MS1 apex (co-elution check).
    Only includes ions whose peak apex is within a few cycles of the MS1 apex.
    """
    n_cycles = len(raw_data.cycles)
    # Expand range slightly for EIC context
    eic_left = max(0, left_idx - 3)
    eic_right = min(n_cycles - 1, right_idx + 3)
    n_pts = eic_right - eic_left + 1

    # First pass: discover all unique product ions in the peak range
    # Adaptive bin size: wider at higher m/z
    bin_scale = max(100, int(1.0 / max(config.eic_mz_tolerance, channel * 100e-6)))
    ion_bins: dict[int, float] = {}  # bin -> representative mz
    for ci in range(left_idx, right_idx + 1):
        if ci >= n_cycles:
            break
        cycle = raw_data.cycles[ci]
        if channel not in cycle.ms2_scans:
            continue
        prod_mz, prod_int = cycle.ms2_scans[channel]
        for mz_val, int_val in zip(prod_mz, prod_int):
            if int_val > 0:
                key = round(mz_val * bin_scale)
                if key not in ion_bins:
                    ion_bins[key] = float(mz_val)

    if not ion_bins:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Second pass: for each ion, build its EIC and validate co-elution
    max_apex_offset = 3  # max cycles between MS1 apex and MS2 ion apex
    match_tol = max(config.eic_mz_tolerance, channel * 100e-6)  # adaptive
    result_mz = []
    result_int = []

    for key, rep_mz in ion_bins.items():
        # Build EIC for this ion
        eic = np.zeros(n_pts)
        for i, ci in enumerate(range(eic_left, eic_right + 1)):
            if ci >= n_cycles:
                break
            cycle = raw_data.cycles[ci]
            if channel not in cycle.ms2_scans:
                continue
            prod_mz, prod_int = cycle.ms2_scans[channel]
            if len(prod_mz) > 0:
                mask = np.abs(prod_mz - rep_mz) <= match_tol
                if np.any(mask):
                    eic[i] = float(np.max(prod_int[mask]))

        # Check: ion must have signal at or very near the MS1 apex
        apex_local = apex_idx - eic_left  # local index of MS1 apex in eic
        if apex_local < 0 or apex_local >= len(eic):
            continue

        # Find the ion's peak apex within the window
        ion_apex_local = np.argmax(eic)
        ion_apex_height = float(eic[ion_apex_local])

        if ion_apex_height < max(config.peak_height_threshold, 50.0):
            continue

        # Co-elution check: ion apex must be within max_apex_offset of MS1 apex
        if abs(ion_apex_local - apex_local) > max_apex_offset:
            continue

        # Additional check: the ion must have intensity at the MS1 apex cycle
        # (not just a distant peak within the window)
        apex_region = eic[max(0, apex_local - 1):min(len(eic), apex_local + 2)]
        if np.max(apex_region) < max(config.peak_height_threshold, 50.0):
            continue

        result_mz.append(rep_mz)
        result_int.append(ion_apex_height)

    if not result_mz:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Sort by m/z
    order = np.argsort(result_mz)
    return np.array(result_mz, dtype=np.float64)[order], np.array(result_int, dtype=np.float64)[order]
