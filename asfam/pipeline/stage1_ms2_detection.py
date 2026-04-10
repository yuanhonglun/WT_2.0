"""Stage 1: MS2-driven peak detection."""
from __future__ import annotations

import logging
from typing import Optional, Callable

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import RawSegmentData, CandidateFeature
from asfam.core.eic import extract_product_ion_eics, merge_close_ions
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
        all_eics = extract_product_ion_eics(
            raw_data, precursor,
            mz_tolerance=config.eic_mz_tolerance,
            min_nonzero_scans=config.peak_width_min,
        )

        if not all_eics:
            continue

        # Pre-filter: skip flat-noise EICs (uniform low intensity, no real peaks)
        filtered_eics = [eic for eic in all_eics if _eic_has_signal(eic.intensity_array)]

        if not filtered_eics:
            continue

        # Detect peaks in each EIC
        all_peaks = []
        for eic in filtered_eics:
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

        # Second pass: recall co-eluting ions with relaxed criteria
        recalled_by_idx: dict[int, list[tuple[float, float, float]]] = {}
        if config.recall_enabled:
            for ci, cluster in enumerate(clusters):
                recalled_by_idx[ci] = _recall_ions_for_cluster(
                    cluster, all_eics, precursor, config,
                )

        # Assemble features from clusters
        for ci, cluster in enumerate(clusters):
            recalled = recalled_by_idx.get(ci, [])
            if len(cluster) + len(recalled) < config.min_fragments_per_feature:
                continue

            # Build MS2 spectrum from cluster
            ms2_mz = np.array([p.product_mz for p in cluster], dtype=np.float64)
            ms2_intensity = np.array([p.height for p in cluster], dtype=np.float64)
            ms2_sn = np.array([p.sn_ratio for p in cluster], dtype=np.float64)

            # Append recalled ions
            if recalled:
                r_mz = np.array([r[0] for r in recalled], dtype=np.float64)
                r_int = np.array([r[1] for r in recalled], dtype=np.float64)
                r_sn = np.array([r[2] for r in recalled], dtype=np.float64)
                ms2_mz = np.concatenate([ms2_mz, r_mz])
                ms2_intensity = np.concatenate([ms2_intensity, r_int])
                ms2_sn = np.concatenate([ms2_sn, r_sn])

            # Merge near-duplicate product ions (adaptive tolerance for high mz)
            ms2_mz, ms2_intensity, ms2_sn = merge_close_ions(
                ms2_mz, ms2_intensity,
                precursor_mz_nominal=precursor,
                base_tolerance=config.eic_mz_tolerance,
                extra_arrays=[ms2_sn],
            )

            # Remove flat-top noise ions: ions with identical intensity are instrument baseline
            ms2_mz, ms2_intensity, ms2_sn = _remove_flat_noise(ms2_mz, ms2_intensity, extra_arrays=[ms2_sn])
            if len(ms2_mz) < config.min_fragments_per_feature:
                continue

            # Reject features where all product ions are noise-level intensity.
            # A genuine MS2 spectrum should have at least one dominant fragment.
            if float(np.max(ms2_intensity)) < config.msms_intensity_threshold:
                continue

            # Per-ion relative threshold: remove ions below X% of base peak
            if config.msms_relative_threshold > 0 and len(ms2_intensity) > 0:
                base_peak = float(np.max(ms2_intensity))
                keep = ms2_intensity >= base_peak * config.msms_relative_threshold
                ms2_mz = ms2_mz[keep]
                ms2_intensity = ms2_intensity[keep]
                ms2_sn = ms2_sn[keep]
                if len(ms2_mz) < config.min_fragments_per_feature:
                    continue

            # Sort by m/z
            order = np.argsort(ms2_mz)
            ms2_mz = ms2_mz[order]
            ms2_intensity = ms2_intensity[order]
            ms2_sn = ms2_sn[order]

            # Refine product ion m/z: re-centroid using raw data at apex scans only
            ms2_mz = _refine_product_mz(
                ms2_mz, cluster, raw_data, precursor,
                config.eic_mz_tolerance,
            )
            # Re-sort after refinement (m/z values may shift slightly)
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


def _recall_ions_for_cluster(
    cluster: list,
    all_eics: list,
    precursor_mz_nominal: int,
    config: ProcessingConfig,
) -> list[tuple[float, float, float]]:
    """Recall additional co-eluting ions at the cluster's consensus RT.

    For each EIC not already represented in the cluster, checks for signal
    in a small window around the cluster's apex. Ions that show co-elution
    evidence (sufficient intensity and consecutive nonzero scans) are recalled.

    Returns list of (product_mz, intensity, sn_estimate).
    """
    if not cluster or not all_eics:
        return []

    # Consensus RT and apex cycle index
    rts = np.array([p.rt_apex for p in cluster])
    heights = np.array([p.height for p in cluster])
    consensus_rt = float(np.average(rts, weights=heights))

    rt_array = all_eics[0].rt_array
    apex_idx = int(np.argmin(np.abs(rt_array - consensus_rt)))

    # Window around apex
    window = config.recall_apex_window
    idx_lo = max(0, apex_idx - window)
    idx_hi = min(len(rt_array) - 1, apex_idx + window)

    # Already-detected product ion m/z bins (0.005 Da resolution)
    detected_bins = {round(p.product_mz * 200) for p in cluster}

    recalled: list[tuple[float, float, float]] = []
    max_recalled = 50  # cap to prevent noise accumulation

    for eic in all_eics:
        if len(recalled) >= max_recalled:
            break

        # Skip if already in cluster
        mz_bin = round(eic.product_mz * 200)
        if mz_bin in detected_bins:
            continue

        # Skip precursor region (instrument artifact)
        if abs(eic.product_mz - precursor_mz_nominal) < 1.5:
            continue

        # Check signal in window
        segment = eic.intensity_array[idx_lo:idx_hi + 1]
        max_val = float(np.max(segment)) if len(segment) > 0 else 0.0
        if max_val < config.recall_min_intensity:
            continue

        # Check consecutive nonzero (co-elution evidence)
        consec = 0
        max_consec = 0
        for v in segment:
            if v > 0:
                consec += 1
                if consec > max_consec:
                    max_consec = consec
            else:
                consec = 0
        if max_consec < config.recall_min_consecutive:
            continue

        # Estimate S/N
        n = len(eic.intensity_array)
        flank_lo = max(0, idx_lo - 20)
        flank_hi = min(n, idx_hi + 20)
        flank_vals = np.concatenate([
            eic.intensity_array[flank_lo:idx_lo],
            eic.intensity_array[idx_hi + 1:flank_hi],
        ])
        noise = float(np.median(np.abs(flank_vals))) + 1.0 if len(flank_vals) > 0 else 1.0
        sn = max_val / noise

        recalled.append((eic.product_mz, max_val, sn))
        detected_bins.add(mz_bin)  # prevent duplicate recalls

    return recalled


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


def _refine_product_mz(
    ms2_mz: np.ndarray,
    cluster: list,
    raw_data: RawSegmentData,
    precursor_mz_nominal: int,
    mz_tolerance: float,
) -> np.ndarray:
    """Refine each product ion's m/z using raw data at apex scans only.

    The global EIC bin centroid averages m/z across ALL cycles (including
    noisy off-peak scans). Re-centroiding at the chromatographic apex
    (±1 cycle) uses only the highest-S/N scans, giving a more accurate
    intensity-weighted m/z.

    Falls back to the original m/z if no raw data is available at the apex.
    """
    refined = np.copy(ms2_mz)

    # Build lookup: m/z bin key -> DetectedPeak (for apex_index)
    peak_by_bin: dict[int, object] = {}
    for p in cluster:
        key = round(p.product_mz * 200)  # 0.005 Da bins
        peak_by_bin[key] = p

    # Consensus apex for recalled ions (those without a DetectedPeak)
    rts = np.array([p.rt_apex for p in cluster])
    heights = np.array([p.height for p in cluster])
    consensus_apex_idx = int(np.argmin(np.abs(
        raw_data.rt_array - float(np.average(rts, weights=heights))
    )))

    for ion_idx in range(len(ms2_mz)):
        ion_mz = ms2_mz[ion_idx]

        # Find DetectedPeak for this ion
        mz_bin = round(ion_mz * 200)
        peak = peak_by_bin.get(mz_bin)
        apex_idx = peak.apex_index if peak is not None else consensus_apex_idx

        # Collect raw m/z values from apex ± 1 cycles
        weighted_mz_sum = 0.0
        weight_sum = 0.0

        lo = max(0, apex_idx - 1)
        hi = min(raw_data.n_cycles, apex_idx + 2)

        for ci in range(lo, hi):
            cycle = raw_data.cycles[ci]
            if precursor_mz_nominal not in cycle.ms2_scans:
                continue
            prod_mz_arr, prod_int_arr = cycle.ms2_scans[precursor_mz_nominal]
            if len(prod_mz_arr) == 0:
                continue
            # Find ions within tolerance of this product ion
            diffs = np.abs(prod_mz_arr - ion_mz)
            mask = diffs <= mz_tolerance
            if np.any(mask):
                local_mz = prod_mz_arr[mask]
                local_int = prod_int_arr[mask]
                weighted_mz_sum += float(np.sum(local_mz * local_int))
                weight_sum += float(np.sum(local_int))

        if weight_sum > 0:
            refined[ion_idx] = weighted_mz_sum / weight_sum

    return refined


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
