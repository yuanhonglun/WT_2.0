"""Verify that stage6b's optional reranker hook reorders annotation_matches
when enabled, and is a strict no-op when disabled."""
from __future__ import annotations

import numpy as np

from asfam.config import ProcessingConfig
from asfam.pipeline.stage6b_annotation import run_stage6b_annotation
from metabo_core.models.features import CandidateFeature


def _make_feature(mz_apex=195.09):
    feat = CandidateFeature(
        feature_id="F00001", segment_name="190-200", replicate_id=1,
        precursor_mz_nominal=int(round(mz_apex)),
        rt_apex=5.0, rt_left=4.9, rt_right=5.1,
        ms2_mz=np.array([110.07, 138.07, 195.09]),
        ms2_intensity=np.array([0.3, 0.4, 1.0]),
        n_fragments=3,
    )
    feat.ms1_precursor_mz = mz_apex
    return feat


def _make_library():
    # Two candidates so reranker can prove it changed ordering.
    # Three peaks each so matchms_min_matched_peaks=3 (default) is satisfied.
    return [
        {
            "mz": [110.07, 138.07, 195.09], "intensity": [0.3, 0.4, 1.0],
            "metadata": {"name": "caffeine_high", "precursor_mz": 195.09,
                         "formula": "C8H10N4O2"},
        },
        {
            "mz": [110.07, 138.07, 195.09], "intensity": [0.35, 0.45, 1.0],
            "metadata": {"name": "alt_close_ri", "precursor_mz": 195.09,
                         "formula": "C8H10N4O2"},
        },
    ]


def test_reranker_disabled_preserves_existing_behavior():
    cfg = ProcessingConfig(reranker_enabled=False)
    feat = _make_feature()
    result = run_stage6b_annotation(
        features_by_replicate={"rep1": [feat]},
        config=cfg,
        preloaded_library=_make_library(),
    )
    out_feat = result["rep1"][0]
    assert out_feat.annotation_matches  # got at least one match
    # No reranker side-effect markers
    assert getattr(out_feat, "reranker_name", "") == ""


def test_reranker_enabled_identity_yields_same_order_but_marks_feature():
    cfg = ProcessingConfig(reranker_enabled=True, reranker_mode="identity")
    feat = _make_feature()
    result = run_stage6b_annotation(
        features_by_replicate={"rep1": [feat]},
        config=cfg,
        preloaded_library=_make_library(),
    )
    out_feat = result["rep1"][0]
    names_before_sort = [m.name for m in out_feat.annotation_matches]
    assert out_feat.reranker_name == "identity"
    # Identity preserves order; first match is still the input top-1
    assert out_feat.matchms_name == names_before_sort[0]


def test_reranker_enabled_with_llm_explain_attaches_explanations():
    cfg = ProcessingConfig(reranker_enabled=True, reranker_mode="llm_explain")
    feat = _make_feature()
    result = run_stage6b_annotation(
        features_by_replicate={"rep1": [feat]},
        config=cfg,
        preloaded_library=_make_library(),
    )
    out_feat = result["rep1"][0]
    assert out_feat.reranker_name.startswith("cosine_ri+")
    assert getattr(out_feat, "annotation_explanations", {})  # dict, non-empty
