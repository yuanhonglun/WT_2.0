"""End-to-end integration smoke test for the reranker contract.

Exercises adapters + Protocol + default fallback + LLM wrapper together,
without depending on any app's pipeline. Pure metabo_core."""
from __future__ import annotations

from metabo_core.annotation import (
    CosineRiReranker, LlmExplanationWrapper, RankingInput, StubLlmExplainer,
)
from metabo_core.annotation.adapters import (
    from_annotation_match, from_gcms_hit, to_annotation_match, to_gcms_hit,
)
from metabo_core.models.features import AnnotationMatch


def test_full_lcms_round_trip():
    matches = [
        AnnotationMatch(rank=1, name="A", score=0.90, n_matched=5, wdp=0.9, sdp=0.85, rdp=0.92),
        AnnotationMatch(rank=2, name="B", score=0.85, n_matched=4, wdp=0.8, sdp=0.75, rdp=0.82),
    ]
    cands = [from_annotation_match(m) for m in matches]
    req = RankingInput(
        feature_id="F00001", mode="dda",
        measured_precursor_mz=303.05, measured_rt=12.4, measured_ri=None,
        measured_spectrum=[(303.05, 1.0)], candidates=cands,
    )
    reranker = LlmExplanationWrapper(
        inner=CosineRiReranker(),
        explainer=StubLlmExplainer(),
        top_k_explained=2,
    )
    result = reranker.rerank(req)
    matches_back = [to_annotation_match(c) for c in result.candidates]
    assert {m.name for m in matches_back} == {"A", "B"}
    assert all(m.rank in (1, 2) for m in matches_back)
    assert 1 in result.explanations and 2 in result.explanations
    assert result.reranker_name == "cosine_ri+stub"


def test_full_gcms_round_trip_preserves_extras():
    hits = [
        {
            "name": "limonene", "formula": "C10H16", "inchikey": "ABCD",
            "adduct": "", "rt": 12.3, "ri": 1030.5,
            "total_score": 0.86, "spectral_score": 0.91, "chrom_score": 0.74,
            "wdp": 0.92, "rdp": 0.88, "sdp": 0.85,
            "matched_pct": 0.7, "n_adjacent_subtracted": 2,
            "ref_peaks": [(93.0, 1.0)], "score": 0.86, "n_matched": 9,
            "acquired_ion_count": 24,
        },
        {
            "name": "alt", "formula": "C10H16", "inchikey": "EFGH",
            "adduct": "", "rt": 12.4, "ri": 1100.0,
            "total_score": 0.50, "spectral_score": 0.60, "chrom_score": 0.40,
            "wdp": 0.60, "rdp": 0.55, "sdp": 0.58,
            "matched_pct": 0.4, "n_adjacent_subtracted": 0,
            "ref_peaks": [(95.0, 1.0)], "score": 0.50, "n_matched": 6,
            "acquired_ion_count": 18,
        },
    ]
    cands = [from_gcms_hit(h, rank=i + 1) for i, h in enumerate(hits)]
    req = RankingInput(
        feature_id="F00007", mode="gcms_fullscan",
        measured_precursor_mz=None, measured_rt=12.3, measured_ri=1030.5,
        measured_spectrum=[(93.0, 1.0)], candidates=cands,
    )
    result = CosineRiReranker(alpha=0.5).rerank(req)
    hits_back = [to_gcms_hit(c) for c in result.candidates]
    # limonene should still rank first (high lib score + matching RI)
    assert hits_back[0]["name"] == "limonene"
    # All keys preserved on every hit
    for h in hits_back:
        for key in ("total_score", "spectral_score", "chrom_score",
                    "matched_pct", "n_adjacent_subtracted",
                    "acquired_ion_count", "ref_peaks", "ri", "rt"):
            assert key in h, f"GCMS round-trip lost {key}"
