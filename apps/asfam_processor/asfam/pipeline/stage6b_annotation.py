"""Stage 6.5: ASFAM glue around the metabo_core library annotator.

Algorithm logic now lives in ``metabo_core.annotation.library``. This stage
only iterates ASFAM-organized replicate features and writes match results
back onto each feature.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from asfam.config import ProcessingConfig
from metabo_core.annotation import (
    build_index_from_list,
    build_reranker,
    load_and_index_library,
    match_feature_topn,
    RankingInput,
)
from metabo_core.annotation.adapters import from_annotation_match, to_annotation_match
from metabo_core.models import CandidateFeature

logger = logging.getLogger(__name__)

TOP_N = 5


def run_stage6b_annotation(
    features_by_replicate: dict[str, list[CandidateFeature]],
    config: ProcessingConfig,
    spectral_library_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    preloaded_library: Optional[list] = None,
) -> dict[str, list[CandidateFeature]]:
    if not spectral_library_path and not preloaded_library:
        logger.info("Stage 6.5: No library provided, skipping annotation.")
        return features_by_replicate

    logger.info("Stage 6.5: Library annotation...")

    if preloaded_library is not None:
        library = build_index_from_list(preloaded_library)
    else:
        library = load_and_index_library(spectral_library_path)
    if library is None:
        return features_by_replicate

    mz_index = library["index"]
    spectra = library["spectra"]

    annotation_cfg = config.annotation_view()
    annotation_cfg.top_n = TOP_N
    similarity_cfg = config.similarity_view()

    reranker = build_reranker(config.reranker_view())

    total_matched = 0
    total_features = 0
    for rep_id, features in features_by_replicate.items():
        active = [f for f in features if f.status == "active"]
        n_matched = 0
        for i, feat in enumerate(active):
            if progress_callback and i % 100 == 0:
                progress_callback(
                    "stage6b", i, len(active),
                    f"Rep {rep_id}: annotating {i}/{len(active)}",
                )
            matches = match_feature_topn(
                feat, spectra, mz_index, annotation_cfg, similarity_cfg, top_n=TOP_N,
            )
            if matches:
                if reranker is not None:
                    cands = [from_annotation_match(m) for m in matches]
                    req = RankingInput(
                        feature_id=feat.feature_id,
                        mode="asfam",
                        measured_precursor_mz=feat.precursor_mz,
                        measured_rt=feat.rt_apex,
                        measured_ri=None,
                        measured_spectrum=list(zip(
                            feat.ms2_mz.tolist(), feat.ms2_intensity.tolist()
                        )),
                        candidates=cands,
                    )
                    result = reranker.rerank(req)
                    matches = [to_annotation_match(c) for c in result.candidates]
                    feat.reranker_name = result.reranker_name
                    if result.explanations:
                        feat.annotation_explanations = result.explanations
                best = matches[0]
                # Score-floor guard for restored duplicates (T4/T5 merge).
                # Dedup-removed features (is_duplicate) are restored to active
                # BEFORE this stage so they can be annotated; but a weak/wrong
                # library hit that only passes the count gate (n_matched /
                # matched_pct) must not leak a name back onto them. Require the
                # best match to clear the display-confidence floor
                # (matchms_similarity_threshold, 0.7) before re-annotating a
                # duplicate. Non-duplicate features keep the "emit any hit"
                # behavior — their display gate is applied later at export/GUI.
                if (feat.is_duplicate
                        and best.score < config.matchms_similarity_threshold):
                    feat.annotation_matches = []
                    continue
                feat.annotation_matches = matches
                feat.selected_annotation_idx = 0
                feat.matchms_name = best.name
                feat.matchms_score = best.score
                if best.formula:
                    feat.inferred_formula = best.formula
                n_matched += 1
            else:
                feat.annotation_matches = []

        total_matched += n_matched
        total_features += len(active)
        logger.info(
            "  Replicate %s: %d/%d annotated (%.1f%%)",
            rep_id, n_matched, len(active),
            n_matched / max(len(active), 1) * 100,
        )

    logger.info("  Total: %d/%d features annotated", total_matched, total_features)
    return features_by_replicate
