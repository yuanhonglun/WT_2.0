"""Stage 3: Merge features across m/z segments within each replicate."""
from __future__ import annotations

import logging
from typing import Optional, Callable

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import CandidateFeature
from asfam.core.similarity import cosine_similarity

logger = logging.getLogger(__name__)


def run_stage3(
    features_by_replicate: dict[str, list[CandidateFeature]],
    config: ProcessingConfig,
    progress_callback: Optional[Callable] = None,
) -> dict[str, list[CandidateFeature]]:
    """Merge features from different m/z segments within each replicate.

    Handles boundary duplicates: features at segment edges that may
    appear in adjacent segments.

    Returns
    -------
    dict: replicate_id -> merged list[CandidateFeature]
    """
    logger.info("Stage 3: Merging features across m/z segments...")

    for rep_id, features in features_by_replicate.items():
        n_before = len(features)

        # Sort by precursor m/z then RT
        features.sort(key=lambda f: (f.precursor_mz, f.rt_apex))

        # Find and remove boundary duplicates
        to_remove: set[int] = set()
        for i in range(len(features)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(features)):
                if j in to_remove:
                    continue

                fi, fj = features[i], features[j]

                # Only check cross-segment pairs
                if fi.segment_name == fj.segment_name:
                    continue

                # Check m/z proximity
                mz_diff = abs(fi.precursor_mz - fj.precursor_mz)
                if mz_diff > config.merge_mz_tolerance:
                    break  # sorted by mz, no more candidates

                # Check RT proximity
                rt_diff = abs(fi.rt_apex - fj.rt_apex)
                if rt_diff > config.merge_rt_tolerance:
                    continue

                # Check MS2 similarity
                peaks_i = list(zip(fi.ms2_mz.tolist(), fi.ms2_intensity.tolist()))
                peaks_j = list(zip(fj.ms2_mz.tolist(), fj.ms2_intensity.tolist()))
                score, _ = cosine_similarity(
                    peaks_i, peaks_j, config.eic_mz_tolerance,
                )

                if score >= config.merge_ms2_cosine_threshold:
                    # Keep the one with better MS1 data
                    if _feature_quality(fi) >= _feature_quality(fj):
                        to_remove.add(j)
                    else:
                        to_remove.add(i)

        # Remove duplicates
        merged = [f for idx, f in enumerate(features) if idx not in to_remove]

        # Reassign feature IDs
        for idx, feat in enumerate(merged):
            feat.feature_id = f"rep{rep_id}_{idx:05d}"

        features_by_replicate[rep_id] = merged

        logger.info(
            "  Replicate %s: %d -> %d features (removed %d boundary duplicates)",
            rep_id, n_before, len(merged), n_before - len(merged),
        )

        if progress_callback:
            progress_callback("stage3", 1, 1, f"Rep {rep_id} merged")

    return features_by_replicate


def _feature_quality(feat: CandidateFeature) -> float:
    """Score feature quality for tiebreaking."""
    score = feat.n_fragments * 100.0
    if feat.ms1_height is not None:
        score += feat.ms1_height
    if feat.ms1_sn is not None:
        score += feat.ms1_sn * 10
    return score
