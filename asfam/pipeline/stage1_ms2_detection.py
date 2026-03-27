"""Stage 1: MS2-driven peak detection."""
from __future__ import annotations

import logging
from typing import Optional, Callable

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import RawSegmentData, CandidateFeature
from asfam.core.eic import extract_product_ion_eics
from asfam.core.peak_detection import detect_peaks
from asfam.core.clustering import cluster_peaks_by_rt

logger = logging.getLogger(__name__)


def run_stage1(
    data_by_replicate: dict[str, list[RawSegmentData]],
    config: ProcessingConfig,
    progress_callback: Optional[Callable] = None,
) -> dict[str, list[CandidateFeature]]:
    """MS2-driven peak detection for all replicates.

    For each MRM-HR channel in each file:
    1. Extract product ion EICs
    2. Detect peaks in each EIC
    3. Cluster peaks by RT to assemble MS2 spectra
    4. Create CandidateFeature objects

    Returns
    -------
    dict: replicate_id -> list[CandidateFeature]
    """
    logger.info("Stage 1: MS2-driven peak detection...")
    results: dict[str, list[CandidateFeature]] = {}

    total_files = sum(len(segs) for segs in data_by_replicate.values())
    file_count = 0

    for rep_id, segments in data_by_replicate.items():
        rep_features: list[CandidateFeature] = []

        for raw_data in segments:
            file_features = _process_one_file(raw_data, config)
            rep_features.extend(file_features)
            file_count += 1

            if progress_callback:
                progress_callback(
                    "stage1", file_count, total_files,
                    f"Rep {rep_id} {raw_data.segment_name}: {len(file_features)} features",
                )

        results[rep_id] = rep_features
        logger.info(
            "  Replicate %s: %d candidate features", rep_id, len(rep_features),
        )

    return results


def _process_one_file(
    raw_data: RawSegmentData,
    config: ProcessingConfig,
) -> list[CandidateFeature]:
    """Process one mzML file through MS2 peak detection."""
    features: list[CandidateFeature] = []
    feature_counter = 0

    for precursor in raw_data.precursor_list:
        # Extract all product ion EICs for this channel
        eics = extract_product_ion_eics(
            raw_data, precursor,
            mz_tolerance=config.eic_mz_tolerance,
            min_nonzero_scans=config.peak_width_min,
        )

        if not eics:
            continue

        # Pre-filter: skip flat-noise EICs (uniform low intensity, no real peaks)
        eics = [eic for eic in eics if _eic_has_signal(eic.intensity_array)]

        if not eics:
            continue

        # Detect peaks in each EIC
        all_peaks = []
        for eic in eics:
            # Detect peaks (detect_peaks handles smoothing internally)
            peaks = detect_peaks(
                eic.rt_array,
                eic.intensity_array,
                precursor_mz_nominal=precursor,
                product_mz=eic.product_mz,
                height_threshold=config.peak_height_threshold,
                sn_threshold=config.peak_sn_threshold,
                width_min=config.peak_width_min,
                prominence=config.peak_prominence,
                smoothing_method=config.eic_smoothing_method,
                smoothing_window=config.eic_smoothing_window,
                smoothing_polyorder=config.eic_smoothing_polyorder,
                compute_gaussian=True,
                gaussian_threshold=config.peak_gaussian_threshold,
            )
            all_peaks.extend(peaks)

        if not all_peaks:
            continue

        # Cluster peaks by RT
        clusters = cluster_peaks_by_rt(all_peaks, config.rt_cluster_tolerance)

        # Assemble features from clusters
        for cluster in clusters:
            if len(cluster) < config.min_fragments_per_feature:
                continue

            # Build MS2 spectrum from cluster
            ms2_mz = np.array([p.product_mz for p in cluster], dtype=np.float64)
            ms2_intensity = np.array([p.height for p in cluster], dtype=np.float64)
            ms2_sn = np.array([p.sn_ratio for p in cluster], dtype=np.float64)

            # Remove flat-top noise ions: ions with identical intensity are instrument baseline
            ms2_mz, ms2_intensity, ms2_sn = _remove_flat_noise(ms2_mz, ms2_intensity, extra_arrays=[ms2_sn])
            if len(ms2_mz) < config.min_fragments_per_feature:
                continue

            # Reject features where all product ions are noise-level intensity.
            # A genuine MS2 spectrum should have at least one dominant fragment.
            if float(np.max(ms2_intensity)) < config.msms_intensity_threshold:
                continue

            # Sort by m/z
            order = np.argsort(ms2_mz)
            ms2_mz = ms2_mz[order]
            ms2_intensity = ms2_intensity[order]
            ms2_sn = ms2_sn[order]

            # Consensus RT (intensity-weighted)
            rts = np.array([p.rt_apex for p in cluster])
            heights = np.array([p.height for p in cluster])
            consensus_rt = float(np.average(rts, weights=heights))

            rt_left = float(min(p.rt_left for p in cluster))
            rt_right = float(max(p.rt_right for p in cluster))

            feature_id = f"{raw_data.segment_name}_{precursor}_{feature_counter}"
            feature_counter += 1

            features.append(CandidateFeature(
                feature_id=feature_id,
                segment_name=raw_data.segment_name,
                replicate_id=raw_data.replicate_id,
                precursor_mz_nominal=precursor,
                rt_apex=consensus_rt,
                rt_left=rt_left,
                rt_right=rt_right,
                ms2_mz=ms2_mz,
                ms2_intensity=ms2_intensity,
                n_fragments=len(ms2_mz),
                ms2_sn=ms2_sn,
                source_file=raw_data.file_path,
            ))

    return features


def _remove_flat_noise(
    mz: np.ndarray, intensity: np.ndarray, min_group_size: int = 2,
    extra_arrays: list[np.ndarray] | None = None,
) -> tuple:
    """Remove flat-top noise ions (identical intensity values).

    Instrument baseline noise appears as multiple ions with exactly the
    same intensity value (e.g., 53 counts). These are not real fragments.

    Strategy: find intensity values that appear >= min_group_size times;
    remove ALL ions with those flat intensities.

    Returns (mz, intensity, *extra_arrays) with noise removed.
    """
    if len(intensity) < 3:
        if extra_arrays:
            return (mz, intensity) + tuple(extra_arrays)
        return mz, intensity

    # Count how many ions share each exact intensity value
    unique_vals, counts = np.unique(intensity, return_counts=True)

    # Flat noise: intensity values appearing multiple times
    flat_values = set(unique_vals[counts >= min_group_size])

    if not flat_values:
        if extra_arrays:
            return (mz, intensity) + tuple(extra_arrays)
        return mz, intensity

    # Keep ions whose intensity is NOT in flat_values
    keep_mask = np.array([float(v) not in flat_values for v in intensity])

    result = [mz[keep_mask], intensity[keep_mask]]
    if extra_arrays:
        for arr in extra_arrays:
            result.append(arr[keep_mask])
    return tuple(result)


def _eic_has_signal(
    intensity: np.ndarray,
    min_dynamic_range: float = 3.0,
) -> bool:
    """Check if an EIC contains real chromatographic signal vs flat noise.

    A noise EIC has uniformly low intensity (max ≈ median). A real signal
    has peaks that stand out clearly above the noise floor.

    Returns True if max / median_nonzero >= min_dynamic_range.
    """
    nonzero = intensity[intensity > 0]
    if len(nonzero) < 3:
        return False
    median_nz = float(np.median(nonzero))
    if median_nz <= 0:
        return False
    return float(np.max(nonzero)) / median_nz >= min_dynamic_range
