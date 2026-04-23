"""Stage 7: Cross-replicate alignment and gap filling."""
from __future__ import annotations

import logging
from typing import Optional, Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from asfam.config import ProcessingConfig
from asfam.models import RawSegmentData, CandidateFeature, Feature
from asfam.core.eic import extract_ms1_eic, extract_product_ion_eics
from asfam.core.peak_detection import detect_peaks
from asfam.core.similarity import cosine_similarity

logger = logging.getLogger(__name__)


def run_stage7(
    features_by_replicate: dict[str, list[CandidateFeature]],
    data_by_replicate: dict[str, list[RawSegmentData]],
    config: ProcessingConfig,
    progress_callback: Optional[Callable] = None,
) -> list[Feature]:
    """Align features across biological replicates.

    Uses the replicate with the most features as reference.
    Matches features using the Hungarian algorithm (globally optimal
    assignment) with combined scoring: Gaussian similarity on m/z + RT,
    plus MS2 spectral cosine similarity for disambiguation.
    """
    logger.info("Stage 7: Cross-replicate alignment...")

    rep_ids = sorted(features_by_replicate.keys())
    if not rep_ids:
        return []

    # Choose reference: replicate with most features
    ref_id = max(rep_ids, key=lambda r: len(features_by_replicate[r]))
    ref_features = features_by_replicate[ref_id]
    logger.info("  Reference replicate: %s (%d features)", ref_id, len(ref_features))

    # Build aligned features from reference
    aligned: list[dict] = []
    for feat in ref_features:
        aligned.append({
            "ref_feature": feat,
            "matches": {ref_id: feat},
        })

    # Precompute reference MS2 peak lists for cosine scoring
    ref_peaks_list = []
    for aln in aligned:
        feat = aln["ref_feature"]
        if feat.ms2_mz is not None and len(feat.ms2_mz) > 0:
            ref_peaks_list.append(list(zip(feat.ms2_mz.tolist(), feat.ms2_intensity.tolist())))
        else:
            ref_peaks_list.append([])

    ms2_tol = config.eic_mz_tolerance  # Da, same tolerance used for MS2 matching

    # Match other replicates to reference using Hungarian algorithm
    for rep_id in rep_ids:
        if rep_id == ref_id:
            continue

        target_features = features_by_replicate[rep_id]
        n_ref = len(aligned)
        n_target = len(target_features)

        if n_target == 0:
            continue

        # Precompute target MS2 peak lists
        target_peaks_list = []
        for t_feat in target_features:
            if t_feat.ms2_mz is not None and len(t_feat.ms2_mz) > 0:
                target_peaks_list.append(list(zip(t_feat.ms2_mz.tolist(), t_feat.ms2_intensity.tolist())))
            else:
                target_peaks_list.append([])

        # Build similarity matrix with combined scoring
        sim_matrix = np.zeros((n_ref, n_target))
        for i, aln in enumerate(aligned):
            ref_feat = aln["ref_feature"]
            ref_mz = ref_feat.precursor_mz
            ref_rt = ref_feat.rt_apex
            ref_peaks = ref_peaks_list[i]

            for j, t_feat in enumerate(target_features):
                # Quick filter: skip obviously distant pairs
                if abs(ref_mz - t_feat.precursor_mz) > config.alignment_mz_tolerance * 3:
                    continue
                if abs(ref_rt - t_feat.rt_apex) > config.alignment_rt_tolerance * 3:
                    continue

                # Gaussian m/z + RT score (0-1)
                gauss = _gaussian_similarity(
                    ref_mz, ref_rt,
                    t_feat.precursor_mz, t_feat.rt_apex,
                    config.alignment_mz_tolerance,
                    config.alignment_rt_tolerance,
                    config.alignment_mz_weight,
                    config.alignment_rt_weight,
                )

                # MS2 cosine similarity (0-1)
                ms2_cos = 0.0
                if ref_peaks and target_peaks_list[j]:
                    ms2_cos, _ = cosine_similarity(ref_peaks, target_peaks_list[j], ms2_tol)

                # Combined score: 60% m/z+RT, 40% MS2
                # When MS2 is unavailable (either side has 0 fragments),
                # fall back to pure m/z+RT scoring
                if ref_peaks and target_peaks_list[j]:
                    sim_matrix[i, j] = 0.6 * gauss + 0.4 * ms2_cos
                else:
                    sim_matrix[i, j] = gauss

        # Hungarian algorithm: minimize cost = maximize similarity
        cost_matrix = -sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            if sim_matrix[r, c] > 0.5:
                aligned[r]["matches"][rep_id] = target_features[c]

        n_matched = sum(1 for r, c in zip(row_ind, col_ind) if sim_matrix[r, c] > 0.5)
        logger.info("  Replicate %s: %d/%d features matched", rep_id, n_matched, n_target)

    # Convert to Feature objects
    result: list[Feature] = []
    for i, aln in enumerate(aligned):
        ref_feat = aln["ref_feature"]
        matches = aln["matches"]

        heights = {}
        areas = {}
        for rid, feat in matches.items():
            h = feat.ms1_height if feat.ms1_height else 0.0
            a = feat.ms1_area if feat.ms1_area else 0.0
            heights[rid] = h
            areas[rid] = a

        h_vals = [v for v in heights.values() if v > 0]
        a_vals = [v for v in areas.values() if v > 0]

        mean_h = float(np.mean(h_vals)) if h_vals else 0.0
        mean_a = float(np.mean(a_vals)) if a_vals else 0.0
        cv_h = float(np.std(h_vals) / mean_h) if mean_h > 0 and len(h_vals) > 1 else 0.0

        feature = Feature(
            feature_id=f"F{i:05d}",
            precursor_mz=ref_feat.precursor_mz,
            rt=ref_feat.rt_apex,
            rt_left=ref_feat.rt_left,
            rt_right=ref_feat.rt_right,
            signal_type=ref_feat.signal_type,
            ms2_mz=ref_feat.ms2_mz,
            ms2_intensity=ref_feat.ms2_intensity,
            n_fragments=ref_feat.n_fragments,
            heights=heights,
            areas=areas,
            mean_height=mean_h,
            mean_area=mean_a,
            cv=cv_h,
            formula=ref_feat.inferred_formula,
            adduct=ref_feat.adduct_type,
            sn_ratio=ref_feat.ms1_sn or 0.0,
            ms1_isotopes=ref_feat.ms1_isotopes,
            name=ref_feat.matchms_name,
            height_ion_mz=ref_feat.ms2_rep_ion_mz,
            detection_source=ref_feat.detection_source,
            mz_source=ref_feat.mz_source,
            mz_confidence=ref_feat.mz_confidence,
            is_duplicate=ref_feat.is_duplicate,
            duplicate_group_id=ref_feat.duplicate_group_id,
            duplicate_type=ref_feat.duplicate_type,
        )
        # Propagate annotation matches (top N)
        if ref_feat.annotation_matches:
            feature.annotation_matches = ref_feat.annotation_matches
            feature.selected_annotation_idx = ref_feat.selected_annotation_idx
        result.append(feature)

    logger.info("  Aligned features: %d", len(result))

    if progress_callback:
        progress_callback("stage7", 1, 1, "Alignment complete")

    return result


def _gaussian_similarity(
    mz1: float, rt1: float,
    mz2: float, rt2: float,
    mz_tol: float, rt_tol: float,
    mz_weight: float = 0.5, rt_weight: float = 0.5,
) -> float:
    """Gaussian similarity score for alignment matching."""
    if mz_tol <= 0 or rt_tol <= 0:
        return 0.0

    mz_score = np.exp(-0.5 * ((mz1 - mz2) / mz_tol) ** 2)
    rt_score = np.exp(-0.5 * ((rt1 - rt2) / rt_tol) ** 2)
    return mz_weight * mz_score + rt_weight * rt_score
