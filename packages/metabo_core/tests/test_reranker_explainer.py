from metabo_core.annotation.reranker import (
    AnnotationCandidate, CandidateReranker, IdentityReranker,
    LlmExplainer, LlmExplanationWrapper, RankingInput, RankingResult,
    StubLlmExplainer,
)


def _req(cands):
    return RankingInput(
        feature_id="F00007",
        mode="dda",
        measured_precursor_mz=303.05,
        measured_rt=12.4,
        measured_ri=None,
        measured_spectrum=[(303.0, 1.0)],
        candidates=cands,
    )


def test_stub_explainer_satisfies_protocol():
    e = StubLlmExplainer()
    assert isinstance(e, LlmExplainer)


def test_stub_explainer_returns_deterministic_string():
    e = StubLlmExplainer()
    req = _req([AnnotationCandidate(rank=1, name="caffeine", score=0.91)])
    s1 = e.explain(req, req.candidates[0])
    s2 = e.explain(req, req.candidates[0])
    assert s1 == s2
    assert "caffeine" in s1


def test_wrapper_satisfies_reranker_protocol():
    w = LlmExplanationWrapper(inner=IdentityReranker(), explainer=StubLlmExplainer())
    assert isinstance(w, CandidateReranker)


def test_wrapper_preserves_inner_order():
    cands = [
        AnnotationCandidate(rank=1, name="a", score=0.9),
        AnnotationCandidate(rank=2, name="b", score=0.8),
        AnnotationCandidate(rank=3, name="c", score=0.7),
    ]
    w = LlmExplanationWrapper(inner=IdentityReranker(), explainer=StubLlmExplainer())
    out = w.rerank(_req(cands))
    assert [c.name for c in out.candidates] == ["a", "b", "c"]


def test_wrapper_attaches_explanations_for_top_k():
    cands = [AnnotationCandidate(rank=i + 1, name=f"c{i}", score=1.0 - 0.1 * i) for i in range(5)]
    w = LlmExplanationWrapper(
        inner=IdentityReranker(),
        explainer=StubLlmExplainer(),
        top_k_explained=2,
    )
    out = w.rerank(_req(cands))
    assert set(out.explanations.keys()) == {1, 2}
    assert "c0" in out.explanations[1]
    assert "c1" in out.explanations[2]


def test_wrapper_records_reranker_name_with_explainer_suffix():
    w = LlmExplanationWrapper(inner=IdentityReranker(), explainer=StubLlmExplainer())
    out = w.rerank(_req([AnnotationCandidate(rank=1, name="x")]))
    assert out.reranker_name == "identity+stub"


def test_wrapper_handles_empty_candidates_without_calling_explainer():
    class CountingExplainer:
        calls = 0
        def explain(self, request, candidate):
            CountingExplainer.calls += 1
            return ""
    w = LlmExplanationWrapper(inner=IdentityReranker(), explainer=CountingExplainer())
    out = w.rerank(_req([]))
    assert out.candidates == []
    assert out.explanations == {}
    assert CountingExplainer.calls == 0


def test_wrapper_records_version_with_inner_version_passthrough():
    """The wrapper should overwrite both reranker_name and reranker_version
    on the result so downstream consumers see who actually emitted the
    final ranking, even when the inner reranker has a different version."""

    class DifferentVersionReranker(IdentityReranker):
        name = "other"
        version = "2.0"

    w = LlmExplanationWrapper(inner=DifferentVersionReranker(), explainer=StubLlmExplainer())
    out = w.rerank(_req([AnnotationCandidate(rank=1, name="x")]))
    assert out.reranker_name == "other+stub"
    assert out.reranker_version == "1.0"
