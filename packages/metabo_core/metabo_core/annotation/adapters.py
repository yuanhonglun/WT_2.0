"""Adapters between app-side candidate representations and AnnotationCandidate.

LC-MS (ASFAM, DDA) uses the AnnotationMatch dataclass from
metabo_core.models.features. GC-MS uses a hit-dict shape produced by
apps/gcms_processor/gcms/pipeline/stage_annotate.py. These adapters
normalize both into AnnotationCandidate for the reranker, then convert
back for storage on RunContext.
"""
from __future__ import annotations

from typing import Any

from metabo_core.annotation.reranker import AnnotationCandidate
from metabo_core.models.features import AnnotationMatch


# Fields that ride in AnnotationCandidate.extras for GC-MS round-trip.
# Keep this list in sync with the hit-dict shape in
# apps/gcms_processor/gcms/pipeline/stage_annotate.py:_score_against_library.
#
# Other GCMS hit-dict keys (name, formula, inchikey, adduct, rt, ri,
# wdp, rdp, sdp, score, n_matched, ref_peaks) are mapped to first-class
# AnnotationCandidate fields — see from_gcms_hit / to_gcms_hit. This
# tuple is only for keys that have no first-class equivalent and must
# ride through extras.
_GCMS_EXTRAS_KEYS = (
    "total_score",
    "spectral_score",
    "chrom_score",
    "matched_pct",
    "n_adjacent_subtracted",
    "acquired_ion_count",
)


def from_annotation_match(match: AnnotationMatch) -> AnnotationCandidate:
    return AnnotationCandidate(
        rank=match.rank,
        name=match.name,
        formula=match.formula,
        adduct=match.adduct,
        score=match.score,
        n_matched=match.n_matched,
        wdp=match.wdp,
        sdp=match.sdp,
        rdp=match.rdp,
        ref_peaks=match.ref_peaks,
        ref_precursor_mz=match.ref_precursor_mz,
    )


def to_annotation_match(cand: AnnotationCandidate) -> AnnotationMatch:
    return AnnotationMatch(
        rank=cand.rank,
        name=cand.name,
        formula=cand.formula,
        score=cand.score,
        n_matched=cand.n_matched,
        ref_peaks=cand.ref_peaks,
        ref_precursor_mz=cand.ref_precursor_mz,
        adduct=cand.adduct,
        wdp=cand.wdp,
        sdp=cand.sdp,
        rdp=cand.rdp,
    )


def from_gcms_hit(hit: dict[str, Any], rank: int) -> AnnotationCandidate:
    extras = {k: hit[k] for k in _GCMS_EXTRAS_KEYS if k in hit}
    return AnnotationCandidate(
        rank=rank,
        name=hit.get("name", ""),
        formula=hit.get("formula", ""),
        inchikey=hit.get("inchikey", ""),
        adduct=hit.get("adduct", ""),
        score=float(hit.get("total_score", hit.get("score", 0.0))),
        n_matched=int(hit.get("n_matched", 0)),
        wdp=float(hit.get("wdp", 0.0)),
        sdp=float(hit.get("sdp", 0.0)),
        rdp=float(hit.get("rdp", 0.0)),
        ref_peaks=hit.get("ref_peaks"),
        ref_precursor_mz=None,
        ref_ri=hit.get("ri"),
        ref_rt=hit.get("rt"),
        extras=extras,
    )


def to_gcms_hit(cand: AnnotationCandidate) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": cand.name,
        "formula": cand.formula,
        "inchikey": cand.inchikey,
        "adduct": cand.adduct,
        "rt": cand.ref_rt,
        "ri": cand.ref_ri,
        "wdp": cand.wdp,
        "rdp": cand.rdp,
        "sdp": cand.sdp,
        "ref_peaks": cand.ref_peaks if cand.ref_peaks is not None else [],
        "score": cand.score,
        "n_matched": cand.n_matched,
    }
    # Carry GCMS-specific fields through extras, in order from canonical list.
    for key in _GCMS_EXTRAS_KEYS:
        if key in cand.extras:
            out[key] = cand.extras[key]
    # Ensure total_score is always present (downstream sort uses it).
    out.setdefault("total_score", cand.score)
    return out
