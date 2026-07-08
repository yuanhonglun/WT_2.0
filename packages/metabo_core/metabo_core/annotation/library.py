"""Library annotation core algorithm.

Algorithmic core extracted from ASFAM stage 6.5. The core does not depend on
``ProcessingConfig`` or any ASFAM-specific data structure; it accepts plain
candidate features (with ``precursor_mz`` / ``rt_apex`` / ``ms2_mz`` /
``ms2_intensity``) and returns top-N ``AnnotationMatch`` results. ASFAM stage
glue is responsible for iterating replicates and writing results back.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from metabo_core.algorithms.similarity import composite_similarity_breakdown
from metabo_core.config import AnnotationConfig, SimilarityConfig
from metabo_core.io.spectral_library import read_msp, read_mgf
from metabo_core.models import AnnotationMatch

logger = logging.getLogger(__name__)

# The only metadata fields the annotation matcher / AnnotationMatch reads.
# The on-disk LC-MS library (lib/lcms/pos.msp ~6.7GB) also carries inchi /
# smiles / comment / splash / synon / instrument / ... per spectrum, which
# nothing downstream uses but which dominate the in-memory footprint
# (~24GB fully loaded). Loading lean — only these fields, plus float64
# numpy peak arrays — cuts that to ~6-7GB with byte-identical matching.
# Extend this set (and AnnotationMatch) if a field needs to reach the UI.
ANNOTATION_METADATA_FIELDS = frozenset(
    {"name", "precursor_mz", "formula", "adduct", "rt"}
)


def load_and_index_library(path: str) -> Optional[dict]:
    """Load MSP/MGF library and build an integer m/z index for fast lookup.

    Loads the *lean* representation (``ANNOTATION_METADATA_FIELDS`` only,
    float64 numpy peak arrays) so a multi-GB library stays within a few GB
    of RAM. Result-identical to a full load for annotation purposes.
    """
    try:
        keep = set(ANNOTATION_METADATA_FIELDS)
        if path.lower().endswith(".msp"):
            spectra = read_msp(path, keep_metadata=keep, as_arrays=True)
        elif path.lower().endswith(".mgf"):
            spectra = read_mgf(path, keep_metadata=keep, as_arrays=True)
        else:
            logger.warning("Unsupported library format: %s", path)
            return None

        if not spectra:
            logger.warning("Library is empty: %s", path)
            return None

        return build_index_from_list(spectra)
    except Exception as exc:
        logger.warning("Failed to load library: %s", exc)
        return None


def build_index_from_list(spectra: list[dict]) -> Optional[dict]:
    """Build an integer m/z index from a pre-loaded list of spectra.

    As a one-time amortization, cache per-spectrum parsed metadata
    (``precursor_mz`` / ``rt``) onto each spec dict under ``_`` -prefixed
    keys. ``match_feature_topn`` reads them by direct dict lookup instead
    of recomputing per (feature, candidate) pair.
    """
    if not spectra:
        return None
    mz_index: dict[int, list[int]] = {}
    for idx, spec in enumerate(spectra):
        meta = spec.get("metadata", {})
        pmz = meta.get("precursor_mz")
        if pmz is not None:
            key = int(round(pmz))
            mz_index.setdefault(key, []).append(idx)

        # Cache parsed metadata used per-candidate by match_feature_topn.
        spec["_precursor_mz"] = float(pmz) if pmz is not None else None
        rt_raw = meta.get("rt", 0)
        spec["_rt"] = float(rt_raw) if rt_raw else 0.0
    logger.info(
        "Library indexed: %d spectra, %d unique integer m/z values",
        len(spectra), len(mz_index),
    )
    return {"spectra": spectra, "index": mz_index}


def _passes_prefilter(
    query_bins_wide: set[int],
    ref_mz,
    shift: float,
    inv_tol: float,
    min_matched: int,
) -> bool:
    """Cheap binned shared-peak screen — an upper bound on ``n_matched``.

    Returns ``False`` only when *neither* the direct nor the neutral-loss
    (shifted) frame can reach ``min_matched`` shared peaks; such a candidate
    cannot pass the ``n_matched >= min_matched`` gate, so skipping it before
    the expensive composite scorer is result-preserving.

    A reference peak counts as a potential match when its integer m/z bin (or
    an adjacent bin) is occupied by a query peak. Because a true within-±tol
    match always lands in one of those three bins, the count never undercounts
    real matches — the screen never drops a candidate the full scorer would
    have kept. It may overcount (keep a candidate that ultimately fails), which
    only costs an unnecessary full score, never a missed hit.

    The neutral-loss frame mirrors :func:`composite_similarity_breakdown`:
    a query peak matches ``ref_mz + shift`` where ``shift = precursor_ref -
    precursor_query``.
    """
    if min_matched <= 0:
        return True

    direct = 0
    for r in ref_mz:
        if int(r * inv_tol) in query_bins_wide:
            direct += 1
            if direct >= min_matched:
                return True

    shifted = 0
    for r in ref_mz:
        if int((r + shift) * inv_tol) in query_bins_wide:
            shifted += 1
            if shifted >= min_matched:
                return True
    return False


def match_feature_topn(
    feature,
    spectra: list[dict],
    mz_index: dict[int, list[int]],
    annotation: AnnotationConfig,
    similarity: SimilarityConfig,
    top_n: Optional[int] = None,
) -> list[AnnotationMatch]:
    """Match one feature against an indexed library and return top-N hits."""
    feat_mz = feature.precursor_mz
    feat_mz_int = int(round(feat_mz))

    candidate_indices: list[int] = []
    for offset in (-1, 0, 1):
        candidate_indices.extend(mz_index.get(feat_mz_int + offset, []))
    if not candidate_indices:
        return []

    query_peaks = list(zip(feature.ms2_mz.tolist(), feature.ms2_intensity.tolist()))
    if len(query_peaks) < annotation.min_peaks_to_match:
        return []

    limit = top_n if top_n is not None else annotation.top_n

    hits = []
    rt_query = feature.rt_apex
    mz_tol = similarity.mz_tolerance
    ms1_tol = similarity.ms1_tolerance
    use_rt = similarity.use_rt
    sim_thresh = annotation.similarity_threshold
    min_matched = annotation.min_matched_peaks
    min_matched_pct = annotation.min_matched_pct
    min_wdp = annotation.min_wdp

    # Per-feature precompute, amortized across all candidates:
    #  * float64 query arrays reused by the array-based matcher (S3) so the
    #    composite never rebuilds them per candidate;
    #  * widened integer m/z bins of the query for the cheap shared-peak
    #    pre-filter (S2). Bin width == mz_tolerance; each query peak occupies
    #    its bin plus the two neighbours so a ±tol match is never missed.
    q_mz = np.asarray(feature.ms2_mz, dtype=np.float64)
    q_int = np.asarray(feature.ms2_intensity, dtype=np.float64)
    inv_tol = 1.0 / mz_tol if mz_tol > 0 else 0.0
    query_bins_wide: set[int] = set()
    if inv_tol > 0:
        for m in q_mz:
            b = int(m * inv_tol)
            query_bins_wide.add(b - 1)
            query_bins_wide.add(b)
            query_bins_wide.add(b + 1)

    for idx in candidate_indices:
        spec = spectra[idx]
        ref_mz_arr = spec.get("mz")
        ref_int_arr = spec.get("intensity")
        if ref_mz_arr is None or len(ref_mz_arr) < annotation.min_peaks_to_match:
            continue

        # Cached at index build time so this is a plain dict lookup, not
        # a recursive metadata get/parse per candidate.
        ref_pmz = spec.get("_precursor_mz")
        # S1: precursor candidate window tightened from ±1.0 to ±0.5 Da.
        if ref_pmz is not None and abs(ref_pmz - feat_mz) > 0.5:
            continue

        # S2: cheap binned shared-peak pre-filter. Runs *before* the ref_peaks
        # build and the composite scorer, so candidates that cannot reach
        # min_matched (in either frame) cost only a handful of dict lookups
        # instead of the full ~50us scoring path. Result-preserving.
        if inv_tol > 0 and min_matched > 0:
            shift = (ref_pmz - feat_mz) if ref_pmz is not None else 0.0
            if not _passes_prefilter(query_bins_wide, ref_mz_arr, shift,
                                     inv_tol, min_matched):
                continue

        ref_peaks = list(zip(ref_mz_arr, ref_int_arr))
        ref_pmz_val = ref_pmz if ref_pmz is not None else 0.0
        ref_rt = spec.get("_rt", 0.0)

        breakdown = composite_similarity_breakdown(
            query_peaks, ref_peaks, mz_tol,
            precursor_query=feat_mz,
            precursor_ref=ref_pmz_val,
            rt_query=rt_query,
            rt_ref=ref_rt,
            ms1_tolerance=ms1_tol,
            use_rt=use_rt,
            q_arrays=(q_mz, q_int),
            r_arrays=(np.asarray(ref_mz_arr, dtype=np.float64),
                      np.asarray(ref_int_arr, dtype=np.float64)),
        )
        score = breakdown.score
        n_matched = breakdown.n_matched

        # Gate on the MS-DIAL-faithful bounded Matched% from B2's breakdown
        # (same ≥1%-of-base significant-ref denominator, numerator restricted
        # to significant matches). NOTE: B2 widened `score` to the [0,2]
        # TotalScore range, so the score/matched_pct gate sensitivities shift;
        # retuning thresholds for the new scale is deferred to PR-E (spec §5.3).
        matched_pct = breakdown.matched_pct

        # min_wdp guard (default 0.0 = disabled): reject inflated-query false
        # positives whose weighted dot product (true m/z-weighted spectral
        # shape) is near zero even though matched_pct=1.0 / high rdp push the
        # total over the high-confidence line. A real match keeps a
        # substantial wdp, so this never drops a genuine hit.
        if (score >= sim_thresh and
                n_matched >= min_matched and
                matched_pct >= min_matched_pct and
                breakdown.wdp >= min_wdp):
            meta = spec.get("metadata", {})
            hits.append({
                "name": meta.get("name", ""),
                "formula": meta.get("formula", ""),
                "score": score,
                "n_matched": n_matched,
                # Coerce to plain Python floats: the lean loader stores peaks
                # as numpy arrays, and numpy scalars are not JSON-serializable
                # (project save). Mirrors the GC-MS annotator's ref_peaks.
                "ref_peaks": [(float(m), float(v)) for m, v in ref_peaks],
                "ref_precursor_mz": meta.get("precursor_mz"),
                "adduct": meta.get("adduct", ""),
                "wdp": breakdown.wdp,
                "sdp": breakdown.sdp,
                "rdp": breakdown.rdp,
                "matched_pct": matched_pct,
                "total_score": breakdown.total_score,
            })

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
            wdp=h.get("wdp", 0.0),
            sdp=h.get("sdp", 0.0),
            rdp=h.get("rdp", 0.0),
            matched_pct=h.get("matched_pct", 0.0),
            total_score=h.get("total_score", 0.0),
        )
        for i, h in enumerate(hits[:limit])
    ]
