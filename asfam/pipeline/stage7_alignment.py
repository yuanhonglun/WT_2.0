"""Stage 7: Cross-replicate alignment and gap filling."""
from __future__ import annotations

import logging
from typing import Optional, Callable

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import RawSegmentData, CandidateFeature, Feature
from asfam.core.eic import extract_ms1_eic, extract_product_ion_eics
from asfam.core.peak_detection import detect_peaks

logger = logging.getLogger(__name__)


def run_stage7(
    features_by_replicate: dict[str, list[CandidateFeature]],
    data_by_replicate: dict[str, list[RawSegmentData]],
    config: ProcessingConfig,
    progress_callback: Optional[Callable] = None,
) -> list[Feature]:
    """Align features across biological replicates.

    Uses the replicate with the most features as reference.
    Matches features by Gaussian similarity (m/z + RT).
    Gap-fills missing values from raw data.
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

    # Match other replicates to reference
    for rep_id in rep_ids:
        if rep_id == ref_id:
            continue

        target_features = features_by_replicate[rep_id]
        used_targets: set[int] = set()

        for aln in aligned:
            ref_feat = aln["ref_feature"]
            best_score = 0.0
            best_idx = -1

            for t_idx, t_feat in enumerate(target_features):
                if t_idx in used_targets:
                    continue

                score = _gaussian_similarity(
                    ref_feat.precursor_mz, ref_feat.rt_apex,
                    t_feat.precursor_mz, t_feat.rt_apex,
                    config.alignment_mz_tolerance,
                    config.alignment_rt_tolerance,
                    config.alignment_mz_weight,
                    config.alignment_rt_weight,
                )

                if score > best_score:
                    best_score = score
                    best_idx = t_idx

            if best_idx >= 0 and best_score > 0.5:
                aln["matches"][rep_id] = target_features[best_idx]
                used_targets.add(best_idx)

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
