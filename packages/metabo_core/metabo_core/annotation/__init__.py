"""Library annotation algorithms shared across apps."""
from metabo_core.annotation.library import (
    load_and_index_library,
    build_index_from_list,
    match_feature_topn,
)
from metabo_core.annotation.reranker import (
    AnnotationCandidate,
    CandidateReranker,
    CosineRiReranker,
    IdentityReranker,
    LlmExplainer,
    LlmExplanationWrapper,
    RankingInput,
    RankingResult,
    StubLlmExplainer,
    build_reranker,  # NEW
)

__all__ = [
    "load_and_index_library",
    "build_index_from_list",
    "match_feature_topn",
    "AnnotationCandidate",
    "CandidateReranker",
    "CosineRiReranker",
    "IdentityReranker",
    "LlmExplainer",
    "LlmExplanationWrapper",
    "RankingInput",
    "RankingResult",
    "StubLlmExplainer",
    "build_reranker",
]
