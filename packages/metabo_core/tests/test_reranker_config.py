from metabo_core.config.reranker import RerankerConfig


def test_default_disabled():
    cfg = RerankerConfig()
    assert cfg.enabled is False
    assert cfg.mode == "default"  # "default" | "identity" | "student" | "llm_explain"
    assert cfg.top_k_explained == 3
    assert cfg.alpha == 0.7
    assert cfg.ri_sigma == 10.0
    assert cfg.student_module is None


def test_explicit_enabled_with_llm_explain():
    cfg = RerankerConfig(enabled=True, mode="llm_explain")
    assert cfg.enabled
    assert cfg.mode == "llm_explain"
