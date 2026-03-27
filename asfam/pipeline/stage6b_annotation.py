"""Stage 6.5: Library annotation - match ALL features against spectral library.

Returns top N matches per feature for user review.
"""
from __future__ import annotations

import logging
from typing import Optional, Callable

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import CandidateFeature, AnnotationMatch
from asfam.core.similarity import composite_similarity

logger = logging.getLogger(__name__)

TOP_N = 5  # number of annotation candidates to keep per feature


def run_stage6b_annotation(
    features_by_replicate: dict[str, list[CandidateFeature]],
    config: ProcessingConfig,
    spectral_library_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    preloaded_library: Optional[list] = None,
) -> dict[str, list[CandidateFeature]]:
    """Annotate all active features by matching against spectral library.

    Stores top N matches per feature in annotation_matches list.
    """
    if not spectral_library_path and not preloaded_library:
        logger.info("Stage 6.5: No library provided, skipping annotation.")
        return features_by_replicate

    logger.info("Stage 6.5: Library annotation...")

    if preloaded_library is not None:
        library = _build_index_from_list(preloaded_library)
    else:
        library = _load_and_index_library(spectral_library_path)
    if library is None:
        return features_by_replicate

    mz_index = library["index"]
    spectra = library["spectra"]

    total_matched = 0
    total_features = 0

    for rep_id, features in features_by_replicate.items():
        active = [f for f in features if f.status == "active"]
        n_matched = 0

        for i, feat in enumerate(active):
            if progress_callback and i % 100 == 0:
                progress_callback("stage6b", i, len(active),
                                  f"Rep {rep_id}: annotating {i}/{len(active)}")

            matches = _match_feature_topn(feat, spectra, mz_index, config, TOP_N)
            if matches:
                feat.annotation_matches = matches
                feat.selected_annotation_idx = 0
                # Backward-compatible: set top-1 fields
                best = matches[0]
                feat.matchms_name = best.name
                feat.matchms_score = best.score
                if best.formula:
                    feat.inferred_formula = best.formula
                n_matched += 1
            else:
                feat.annotation_matches = []

        total_matched += n_matched
        total_features += len(active)
        logger.info("  Replicate %s: %d/%d annotated (%.1f%%)",
                     rep_id, n_matched, len(active),
                     n_matched / max(len(active), 1) * 100)

    logger.info("  Total: %d/%d features annotated", total_matched, total_features)
    return features_by_replicate


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def _load_and_index_library(path: str) -> Optional[dict]:
    """Load library and build integer m/z index for fast lookup."""
    try:
        from asfam.io.spectral_library import read_msp, read_mgf

        if path.lower().endswith(".msp"):
            spectra = read_msp(path)
        elif path.lower().endswith(".mgf"):
            spectra = read_mgf(path)
        else:
            logger.warning("Unsupported library format: %s", path)
            return None

        if not spectra:
            logger.warning("Library is empty: %s", path)
            return None

        mz_index: dict[int, list[int]] = {}
        for idx, spec in enumerate(spectra):
            pmz = spec.get("metadata", {}).get("precursor_mz")
            if pmz is not None:
                key = int(round(pmz))
                if key not in mz_index:
                    mz_index[key] = []
                mz_index[key].append(idx)

        logger.info("Library loaded: %d spectra, %d unique integer m/z values",
                     len(spectra), len(mz_index))
        return {"spectra": spectra, "index": mz_index}
    except Exception as e:
        logger.warning("Failed to load library: %s", e)
        return None


def _build_index_from_list(spectra: list[dict]) -> Optional[dict]:
    """Build m/z index from a pre-loaded spectrum list."""
    if not spectra:
        return None
    mz_index: dict[int, list[int]] = {}
    for idx, spec in enumerate(spectra):
        pmz = spec.get("metadata", {}).get("precursor_mz")
        if pmz is not None:
            key = int(round(pmz))
            if key not in mz_index:
                mz_index[key] = []
            mz_index[key].append(idx)
    logger.info("Library indexed: %d spectra, %d unique integer m/z values",
                 len(spectra), len(mz_index))
    return {"spectra": spectra, "index": mz_index}


# ---------------------------------------------------------------------------
# Matching: returns top N AnnotationMatch objects
# ---------------------------------------------------------------------------

def _match_feature_topn(
    feature: CandidateFeature,
    spectra: list[dict],
    mz_index: dict[int, list[int]],
    config: ProcessingConfig,
    top_n: int = 5,
) -> list[AnnotationMatch]:
    """Match one feature against indexed library, return top N matches."""
    feat_mz = feature.precursor_mz
    feat_mz_int = int(round(feat_mz))

    candidate_indices = []
    for offset in [-1, 0, 1]:
        candidate_indices.extend(mz_index.get(feat_mz_int + offset, []))

    if not candidate_indices:
        return []

    query_peaks = list(zip(feature.ms2_mz.tolist(), feature.ms2_intensity.tolist()))
    if len(query_peaks) < 2:
        return []

    hits = []
    for idx in candidate_indices:
        spec = spectra[idx]
        ref_mz_list = spec.get("mz", [])
        ref_int_list = spec.get("intensity", [])
        if len(ref_mz_list) < 2:
            continue

        ref_pmz = spec.get("metadata", {}).get("precursor_mz")
        if ref_pmz is not None and abs(ref_pmz - feat_mz) > 1.0:
            continue

        ref_peaks = list(zip(ref_mz_list, ref_int_list))
        ref_pmz_val = float(ref_pmz) if ref_pmz is not None else 0.0
        ref_rt = float(spec.get("metadata", {}).get("rt", 0) or 0)

        score, n_matched = composite_similarity(
            query_peaks, ref_peaks, config.eic_mz_tolerance,
            precursor_query=feat_mz,
            precursor_ref=ref_pmz_val,
            rt_query=feature.rt_apex,
            rt_ref=ref_rt,
            ms1_tolerance=config.ms1_mz_tolerance,
            use_rt=config.matchms_use_rt,
        )

        # Matched peaks percentage
        max_ref_int = max((i for _, i in ref_peaks), default=1)
        n_ref_sig = sum(1 for _, i in ref_peaks if i >= max_ref_int * 0.01)
        matched_pct = n_matched / max(n_ref_sig, 1)

        if (score >= config.matchms_similarity_threshold and
                n_matched >= config.matchms_min_matched_peaks and
                matched_pct >= config.matchms_min_matched_pct):
            meta = spec.get("metadata", {})
            hits.append({
                "name": meta.get("name", ""),
                "formula": meta.get("formula", ""),
                "score": score,
                "n_matched": n_matched,
                "ref_peaks": ref_peaks,
                "ref_precursor_mz": meta.get("precursor_mz"),
                "adduct": meta.get("adduct", ""),
            })

    # Sort by score descending, take top N
    hits.sort(key=lambda h: h["score"], reverse=True)
    return [
        AnnotationMatch(
            rank=i + 1,
            name=h["name"],
            formula=h.get("formula", ""),
            score=h["score"],
            n_matched=h["n_matched"],
            ref_peaks=h.get("ref_peaks"),
            ref_precursor_mz=h.get("ref_precursor_mz"),
            adduct=h.get("adduct", ""),
        )
        for i, h in enumerate(hits[:top_n])
    ]
