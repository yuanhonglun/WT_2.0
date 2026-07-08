from metabo_core.annotation.reranker import (
    AnnotationCandidate, CandidateReranker, IdentityReranker,
    RankingInput, RankingResult,
)


def _make_request(candidates):
    return RankingInput(
        feature_id="F00001",
        mode="dda",
        measured_precursor_mz=303.05,
        measured_rt=12.4,
        measured_ri=None,
        measured_spectrum=[(303.0, 1.0)],
        candidates=candidates,
    )


def test_identity_reranker_satisfies_protocol():
    r = IdentityReranker()
    assert isinstance(r, CandidateReranker)
    assert r.name == "identity"


def test_identity_reranker_preserves_order_and_ranks():
    cands = [
        AnnotationCandidate(rank=1, name="a", score=0.9),
        AnnotationCandidate(rank=2, name="b", score=0.8),
        AnnotationCandidate(rank=3, name="c", score=0.7),
    ]
    result = IdentityReranker().rerank(_make_request(cands))

    assert isinstance(result, RankingResult)
    assert result.reranker_name == "identity"
    assert [c.name for c in result.candidates] == ["a", "b", "c"]
    assert [c.rank for c in result.candidates] == [1, 2, 3]


def test_identity_reranker_handles_empty_candidates():
    result = IdentityReranker().rerank(_make_request([]))
    assert result.candidates == []
    assert result.explanations == {}


def test_identity_reranker_preserves_extras():
    cand = AnnotationCandidate(rank=1, name="x", extras={"chrom_score": 0.77})
    result = IdentityReranker().rerank(_make_request([cand]))
    assert result.candidates[0].extras["chrom_score"] == 0.77


import math
from metabo_core.annotation.reranker import CosineRiReranker


def test_cosine_ri_reranker_satisfies_protocol():
    r = CosineRiReranker()
    assert isinstance(r, CandidateReranker)
    assert r.name == "cosine_ri"


def test_cosine_ri_reranker_lcms_path_behaves_like_identity():
    """When measured_ri is None, only library score matters → same order."""
    cands = [
        AnnotationCandidate(rank=1, name="a", score=0.9),
        AnnotationCandidate(rank=2, name="b", score=0.8),
    ]
    req = _make_request(cands)  # measured_ri=None per helper
    result = CosineRiReranker().rerank(req)
    assert [c.name for c in result.candidates] == ["a", "b"]
    assert [c.rank for c in result.candidates] == [1, 2]


def test_cosine_ri_reranker_promotes_close_ri_match():
    """Lower lib_score but matching RI should beat higher lib_score with far RI."""
    cands = [
        AnnotationCandidate(rank=1, name="high_lib_far_ri", score=0.95, ref_ri=2000.0),
        AnnotationCandidate(rank=2, name="lower_lib_close_ri", score=0.70, ref_ri=1500.0),
    ]
    req = RankingInput(
        feature_id="F00001",
        mode="gcms_fullscan",
        measured_precursor_mz=None,
        measured_rt=None,
        measured_ri=1500.0,
        measured_spectrum=[],
        candidates=cands,
    )
    # alpha=0.5 to make RI matter as much as lib_score
    result = CosineRiReranker(alpha=0.5, ri_sigma=10.0).rerank(req)
    assert result.candidates[0].name == "lower_lib_close_ri"
    assert result.candidates[0].rank == 1
    assert result.candidates[1].rank == 2


def test_cosine_ri_reranker_handles_empty():
    req = _make_request([])
    result = CosineRiReranker().rerank(req)
    assert result.candidates == []


def test_cosine_ri_reranker_reassigns_rank_field():
    cands = [
        AnnotationCandidate(rank=99, name="x", score=0.5),
        AnnotationCandidate(rank=99, name="y", score=0.9),
    ]
    result = CosineRiReranker().rerank(_make_request(cands))
    assert result.candidates[0].name == "y"
    assert result.candidates[0].rank == 1
    assert result.candidates[1].rank == 2
