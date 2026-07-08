import pytest
from metabo_core.annotation.reranker import (
    CosineRiReranker, IdentityReranker, LlmExplanationWrapper, build_reranker,
)
from metabo_core.config.reranker import RerankerConfig


def test_factory_disabled_returns_none():
    assert build_reranker(RerankerConfig(enabled=False)) is None


def test_factory_default_returns_cosine_ri():
    r = build_reranker(RerankerConfig(enabled=True, mode="default", alpha=0.5, ri_sigma=20.0))
    assert isinstance(r, CosineRiReranker)
    assert r.alpha == 0.5
    assert r.ri_sigma == 20.0


def test_factory_identity_returns_identity():
    r = build_reranker(RerankerConfig(enabled=True, mode="identity"))
    assert isinstance(r, IdentityReranker)


def test_factory_llm_explain_wraps_default():
    r = build_reranker(RerankerConfig(enabled=True, mode="llm_explain"))
    assert isinstance(r, LlmExplanationWrapper)
    assert isinstance(r.inner, CosineRiReranker)
    assert r.top_k_explained == 3


def test_factory_unknown_mode_raises():
    with pytest.raises(ValueError, match="reranker mode"):
        build_reranker(RerankerConfig(enabled=True, mode="not_a_mode"))


def test_factory_student_missing_module_raises():
    with pytest.raises(ValueError, match="student_module"):
        build_reranker(RerankerConfig(enabled=True, mode="student", student_module=None))


def test_factory_student_nonexistent_module_raises_module_not_found():
    """When student_module points to a module that doesn't exist on the
    Python path, importlib raises ModuleNotFoundError. This propagates
    to the caller in Q1 (no try/except wrapper); a future Q3 task will
    add graceful fallback to IdentityReranker."""
    cfg = RerankerConfig(
        enabled=True, mode="student",
        student_module="this_module.definitely.does.not.exist:build",
    )
    with pytest.raises(ModuleNotFoundError):
        build_reranker(cfg)


def test_factory_student_malformed_dotted_path_raises():
    """student_module without the ':' separator raises ValueError."""
    cfg = RerankerConfig(
        enabled=True, mode="student",
        student_module="missing.the.colon",
    )
    with pytest.raises(ValueError, match="module:callable"):
        build_reranker(cfg)
