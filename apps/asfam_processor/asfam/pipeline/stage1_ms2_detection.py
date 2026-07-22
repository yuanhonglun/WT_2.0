"""Stage 1: MS2-driven peak detection."""
from __future__ import annotations

import logging
from typing import Optional, Callable

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import RawSegmentData, CandidateFeature
from asfam.core.eic import extract_product_ion_eics_massslice
from asfam.core.peak_detection import detect_peaks, detect_chrom_peaks
from asfam.core.clustering import cluster_peaks_by_rt
from asfam.core.ms2_amdis_clustering import cluster_peaks_amdis
from metabo_core.algorithms.ms2_cleanup import (
    MS2CleanupConfig,
    clean_ms2_spectrum,
)
from metabo_core.algorithms.shape_correlation import median_pairwise_correlation

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

    ms2_pd = config.ms2_peak
    for precursor in raw_data.precursor_list:
        # Extract all product ion EICs for this channel
        all_eics = extract_product_ion_eics_massslice(raw_data, precursor, config)

        if not all_eics:
            continue

        # Pre-filter: skip flat-noise EICs (uniform low intensity, no real peaks)
        filtered_eics = [eic for eic in all_eics if _eic_has_signal(eic.intensity_array)]

        if not filtered_eics:
            continue

        # Detect peaks in each EIC
        all_peaks = []
        for eic in filtered_eics:
            # Detect peaks (router → builtin detect_peaks or MS-DIAL engine by
            # config.peak_detector; builtin path is byte-for-byte unchanged).
            peaks = detect_chrom_peaks(
                eic.rt_array,
                eic.intensity_array,
                config=config,
                precursor_mz_nominal=precursor,
                product_mz=eic.product_mz,
                min_amplitude=ms2_pd.min_amplitude,
                min_data_points=ms2_pd.min_data_points,
                compute_gaussian=True,
                gaussian_threshold=ms2_pd.gaussian_threshold,
                min_prominence_ratio=ms2_pd.min_prominence_ratio,
                rt_window_min=ms2_pd.rt_window_min,
                rt_window_max=ms2_pd.rt_window_max,
            )
            for p in peaks:
                if eic.basepeak_mz is not None:
                    p.product_mz = float(eic.basepeak_mz[p.apex_index])
            all_peaks.extend(peaks)

        if not all_peaks:
            continue

        # Option-A global near-duplicate dedup before RT clustering: collapse any
        # two peaks within global_dedup_mass_tol + global_dedup_rt_tol (keep the
        # taller), same semantics as MS1's global cleanup. Primarily removes the
        # same ion detected twice by 50%-overlapping mass slices.
        all_peaks = _dedup_peaks_global(
            all_peaks,
            config.msdial_peak.global_dedup_mass_tol,
            config.msdial_peak.global_dedup_rt_tol,
        )

        # Cluster peaks into co-eluting groups. Two paths (config.ms2_clustering):
        if config.ms2_clustering == "amdis":
            # AMDIS component-perception (sharpness NMS + used-bins) already does
            # component-level ownership, so _resolve_peak_ownership (RT-clustering
            # specific) is intentionally NOT applied here; shared fragments may
            # appear in multiple co-eluting features (calibrated in AMDIS plan T6).
            clusters = cluster_peaks_amdis(filtered_eics, all_peaks, config)
        else:
            # RT-proximity clustering (default, unchanged).
            clusters = cluster_peaks_by_rt(
                all_peaks,
                config.rt_cluster_tolerance,
                config.cluster_max_apex_span,
            )
            # One-peak-one-feature ownership resolution.
            # A single chromatographic peak in a given product-ion EIC can only
            # belong to the cluster whose intensity-weighted consensus RT is
            # nearest to that peak's apex. This prevents a low-response feature
            # from absorbing ions that really belong to a neighbouring
            # high-response feature (e.g. a shared base peak).
            clusters = _resolve_peak_ownership(clusters)

        # Drop clusters that no longer carry enough genuine peaks
        clusters = [c for c in clusters if len(c) >= 1]

        # Cross-EIC shape coherence: continuous-baseline-noise "features"
        # have fragment EICs that are independent of each other (median
        # pairwise correlation near 0), while real metabolite features
        # have all fragments rising and falling together (median ≥ 0.85).
        # Single-peak clusters trivially pass.
        if config.shape_corr_threshold > 0 and clusters:
            eic_by_mz = {
                float(eic.product_mz): eic.intensity_array
                for eic in filtered_eics
            }
            coherent: list[list] = []
            for cluster in clusters:
                if len(cluster) < 2:
                    coherent.append(cluster)
                    continue
                lo = min(p.left_index for p in cluster)
                hi = max(p.right_index for p in cluster) + 1
                segments: list[np.ndarray] = []
                for p in cluster:
                    eic_int = eic_by_mz.get(float(p.product_mz))
                    if eic_int is None:
                        continue
                    segments.append(eic_int[lo:hi])
                if len(segments) < 2:
                    coherent.append(cluster)
                    continue
                if median_pairwise_correlation(segments) >= config.shape_corr_threshold:
                    coherent.append(cluster)
            clusters = coherent

        # Second pass: recall co-eluting ions with relaxed criteria
        recalled_by_idx: dict[int, list[tuple[float, float, float]]] = {}
        if config.recall_enabled and clusters:
            # Precompute consensus RT per cluster for ownership-aware recall
            all_consensus_rts: list[float] = []
            for cl in clusters:
                if not cl:
                    all_consensus_rts.append(float("nan"))
                    continue
                rts_c = np.array([p.rt_apex for p in cl], dtype=np.float64)
                hs_c = np.array([max(p.height, 1e-6) for p in cl], dtype=np.float64)
                all_consensus_rts.append(float(np.average(rts_c, weights=hs_c)))
            for ci, cluster in enumerate(clusters):
                recalled_by_idx[ci] = _recall_ions_for_cluster(
                    cluster, all_eics, precursor, config,
                    cluster_index=ci,
                    all_consensus_rts=all_consensus_rts,
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
            ms2_gaussian = np.array(
                [p.gaussian_similarity for p in cluster], dtype=np.float64,
            )

            # Append recalled ions. Recalled ions come from a more
            # permissive co-elution criterion; their per-peak shape is
            # not verified, so they carry gaussian=0.0 (which the
            # aggregator treats as "unknown" and ignores).
            if recalled:
                r_mz = np.array([r[0] for r in recalled], dtype=np.float64)
                r_int = np.array([r[1] for r in recalled], dtype=np.float64)
                r_sn = np.array([r[2] for r in recalled], dtype=np.float64)
                r_gauss = np.zeros(len(recalled), dtype=np.float64)
                ms2_mz = np.concatenate([ms2_mz, r_mz])
                ms2_intensity = np.concatenate([ms2_intensity, r_int])
                ms2_sn = np.concatenate([ms2_sn, r_sn])
                ms2_gaussian = np.concatenate([ms2_gaussian, r_gauss])

            # Unified MS2 cleanup pipeline (merge / flat-noise / threshold /
            # remove-after-precursor / top-N / sort). The cleanup runs through
            # the shared metabo_core implementation so ASFAM and DDA stay in
            # lock-step. ms2_sn is carried along as an auxiliary array.
            cleanup_cfg = MS2CleanupConfig(
                merge_absolute_tol=config.eic_mz_tolerance,
                absolute_intensity_threshold=config.msms_intensity_threshold,
                # ms2_driven path uses its OWN relative floor, decoupled from
                # stage1b's msms_relative_threshold (T1/MSDec). See config.py.
                relative_intensity_threshold=config.ms2_driven_rel_floor,
                # ASFAM stage 1 has no precursor m/z at this point — we only
                # have the nominal int. Disable the "remove after precursor"
                # step here; stage 2 (MS1 assignment) and later stages still
                # work without it.
                remove_after_precursor=False,
                # ASFAM's intensity-floor reject ("reject if max < threshold")
                # is enforced separately via min_fragments_per_feature below;
                # the threshold here keeps low-intensity ions out.
            )
            ms2_mz, ms2_intensity, ms2_sn, ms2_gaussian = clean_ms2_spectrum(
                ms2_mz, ms2_intensity,
                precursor_mz=float(precursor),
                config=cleanup_cfg,
                extra_arrays=[ms2_sn, ms2_gaussian],
            )
            if len(ms2_mz) < config.min_fragments_per_feature:
                continue
            # Feature-admission guard (decoupled from the cleanup floor, option B).
            # Admit a ms2_driven cluster only if its brightest fragment reaches the
            # base-peak quality bar (ms2_driven_feature_floor, default 1000). The
            # cleanup floor above stays at 200 so an ADMITTED feature keeps its
            # weaker [200,1000) real fragments; this bar just stops recall-dominated
            # weak clusters from exploding the feature count (validation 2026-07-02).
            if float(np.max(ms2_intensity)) < config.ms2_driven_feature_floor:
                continue

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
                ms2_gaussian=ms2_gaussian,
                source_file=raw_data.file_path,
            ))

    return features


def _resolve_peak_ownership(
    clusters: list[list],
) -> list[list]:
    """Assign each detected peak to exactly one cluster.

    For every (product_mz_bin, apex_index) peak present across all clusters,
    keep it only in the cluster whose intensity-weighted consensus RT is
    closest to the peak's own apex RT. Ties are broken by the larger cluster,
    then by higher peak height.

    This fixes the case where a high-response feature shares a strong ion
    (e.g. base peak 85) with a neighbouring low-response feature whose own
    cluster would otherwise "steal" that ion even though the real peak
    apex is far away from the low-response consensus RT.
    """
    if not clusters:
        return clusters

    # Consensus RT per cluster (intensity-weighted mean of apex RTs)
    consensus_rts: list[float] = []
    cluster_sizes: list[int] = []
    for cluster in clusters:
        if not cluster:
            consensus_rts.append(0.0)
            cluster_sizes.append(0)
            continue
        rts = np.array([p.rt_apex for p in cluster], dtype=np.float64)
        heights = np.array([max(p.height, 1e-6) for p in cluster], dtype=np.float64)
        consensus_rts.append(float(np.average(rts, weights=heights)))
        cluster_sizes.append(len(cluster))

    # Choose owner for every (product_mz_bin, apex_index) key
    peak_owner: dict[tuple[int, int], int] = {}
    peak_pick: dict[tuple[int, int], tuple[float, int, float]] = {}
    # Stored score = (distance_to_consensus, -cluster_size, -peak_height)
    # smaller score wins (tie-break: larger cluster, taller peak)

    for ci, cluster in enumerate(clusters):
        cons_rt = consensus_rts[ci]
        size = cluster_sizes[ci]
        for p in cluster:
            key = (round(p.product_mz * 200), int(p.apex_index))
            score = (abs(cons_rt - p.rt_apex), -size, -float(p.height))
            best = peak_pick.get(key)
            if best is None or score < best:
                peak_pick[key] = score
                peak_owner[key] = ci

    # Rebuild clusters keeping only peaks owned by this cluster index
    new_clusters: list[list] = [[] for _ in clusters]
    for ci, cluster in enumerate(clusters):
        for p in cluster:
            key = (round(p.product_mz * 200), int(p.apex_index))
            if peak_owner.get(key) == ci:
                new_clusters[ci].append(p)
    return new_clusters


def _dedup_peaks_global(peaks, mass_tol, rt_tol):
    """Global near-duplicate peak dedup (Option A; port of metabo_core
    _further_cleanup to DetectedPeak).

    Sort by product_mz ascending; for any pair within mass_tol and with
    |delta rt_apex| < rt_tol, keep the taller (higher height) peak. When
    heights are exactly equal, exclude the earlier one (matching the C#
    "(target - searched) > 0" semantics). This removes the duplicate peaks of
    the same ion produced by 50%-overlapping mass slices.
    """
    if not peaks:
        return peaks
    s = sorted(peaks, key=lambda p: (p.product_mz, p.rt_apex))
    n = len(s)
    exclude = set()
    for i in range(n):
        target = s[i]
        for j in range(i + 1, n):
            searched = s[j]
            if (searched.product_mz - target.product_mz) > mass_tol:
                break
            if abs(target.rt_apex - searched.rt_apex) < rt_tol:
                if (target.height - searched.height) > 0:
                    exclude.add(j)
                else:
                    exclude.add(i)
    return [s[i] for i in range(n) if i not in exclude]


def _recall_ions_for_cluster(
    cluster: list,
    all_eics: list,
    precursor_mz_nominal: int,
    config: ProcessingConfig,
    cluster_index: int = 0,
    all_consensus_rts: list | None = None,
) -> list[tuple[float, float, float]]:
    """Recall additional co-eluting ions at the cluster's consensus RT.

    For each EIC not already represented in the cluster, checks for signal
    in a small window around the cluster's apex. Ions that show co-elution
    evidence (sufficient intensity and consecutive nonzero scans) are recalled.

    Ownership-aware: if another cluster's consensus RT is closer to the
    window's own local maximum than this cluster's consensus RT, the ion
    is left to that other cluster instead.

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

        # Ownership guard: if another cluster's consensus RT is closer to
        # the local maximum of this EIC within the recall window, skip —
        # that peak belongs to the other cluster.
        if all_consensus_rts is not None and len(all_consensus_rts) > 1:
            local_argmax = int(np.argmax(segment)) + idx_lo
            local_rt = float(rt_array[local_argmax])
            my_dist = abs(consensus_rt - local_rt)
            stolen = False
            for oi, other_rt in enumerate(all_consensus_rts):
                if oi == cluster_index or not np.isfinite(other_rt):
                    continue
                if abs(other_rt - local_rt) < my_dist - 1e-9:
                    stolen = True
                    break
            if stolen:
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
