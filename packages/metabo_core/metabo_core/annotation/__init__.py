"""Library annotation algorithms shared across apps."""
from metabo_core.annotation.library import (
    ANNOTATION_METADATA_FIELDS,
    load_and_index_library,
    load_library_lean,
    build_index_from_list,
    match_feature_topn,
)
from metabo_core.annotation.confidence import is_high_confidence
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
    "ANNOTATION_METADATA_FIELDS",
    "load_and_index_library",
    "load_library_lean",
    "build_index_from_list",
    "match_feature_topn",
    "is_high_confidence",
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
