"""Tests for the shared ``pipeline_labels`` module.

These tests pin two things down:

* ``get_stage_titles`` merges :data:`COMMON_STAGE_TITLES` with the
  correct app-specific dictionary.
* ``stage_label`` produces the same rendering as a direct
  :func:`stage_message` call with the right ``extra_titles``.
"""
from __future__ import annotations

import pytest

from metabo_core.pipeline_labels import (
    ASFAM_STAGE_TITLES,
    COMMON_STAGE_TITLES,
    DDA_STAGE_TITLES,
    GCMS_STAGE_TITLES,
    get_stage_titles,
    stage_label,
    stage_message,
)


# ---------------------------------------------------------------------
# get_stage_titles
# ---------------------------------------------------------------------
def test_get_stage_titles_asfam_includes_common_and_app_keys():
    titles = get_stage_titles("asfam")
    # Common keys are present.
    for key in COMMON_STAGE_TITLES:
        assert key in titles
    # ASFAM-specific keys are present.
    for key in ASFAM_STAGE_TITLES:
        assert key in titles
        assert titles[key] == ASFAM_STAGE_TITLES[key]
    # A representative spot check.
    assert titles["stage0"] == "Loading mzML files"
    assert titles["stage4"] == "Deduplicating isotope peaks"


def test_get_stage_titles_dda_includes_common_and_app_keys():
    titles = get_stage_titles("dda")
    for key in COMMON_STAGE_TITLES:
        assert key in titles
    for key in DDA_STAGE_TITLES:
        assert key in titles
        assert titles[key] == DDA_STAGE_TITLES[key]
    # DDA-specific stage names.
    assert "isotope_adduct" in titles
    assert "ms2_assoc" in titles
    # Common name still resolves to the common phrasing.
    assert titles["load"] == "Loading mzML files"


def test_get_stage_titles_gcms_includes_common_and_app_keys():
    titles = get_stage_titles("gcms")
    for key in COMMON_STAGE_TITLES:
        assert key in titles
    for key in GCMS_STAGE_TITLES:
        assert key in titles
        assert titles[key] == GCMS_STAGE_TITLES[key]
    assert titles["deconvolve"] == "Deconvolving co-eluting components"
    assert titles["compute_ri"] == "Computing retention indices"


def test_get_stage_titles_unknown_app_raises():
    with pytest.raises(KeyError):
        get_stage_titles("unknown_app")  # type: ignore[arg-type]


def test_get_stage_titles_returns_independent_dicts():
    """每次调用应返回独立的 dict，避免外部修改污染常量。"""
    a = get_stage_titles("asfam")
    a["stage0"] = "mutated"
    b = get_stage_titles("asfam")
    assert b["stage0"] == "Loading mzML files"
    # And the underlying constant is untouched.
    assert ASFAM_STAGE_TITLES["stage0"] == "Loading mzML files"


# ---------------------------------------------------------------------
# stage_label
# ---------------------------------------------------------------------
def test_stage_label_dda_load_start():
    assert stage_label("dda", "load", "start") == "Loading mzML files…"


def test_stage_label_dda_features_done_with_elapsed():
    msg = stage_label("dda", "features", "done", elapsed=1.234)
    # ``stage_message`` formats elapsed with one decimal place.
    assert msg == "Detecting MS1 features done in 1.2s"


def test_stage_label_asfam_stage4_start_uses_app_phrasing():
    msg = stage_label("asfam", "stage4", "start")
    assert msg == "Deduplicating isotope peaks…"


def test_stage_label_asfam_stage4_done_with_detail():
    msg = stage_label("asfam", "stage4", "done",
                      elapsed=2.5, detail="42 features remain")
    assert msg == "Deduplicating isotope peaks done in 2.5s · 42 features remain"


def test_stage_label_dda_isotope_adduct_start_uses_dda_override():
    """DDA 的 isotope_adduct 阶段不在公共表里，必须落到 DDA 字典。"""
    msg = stage_label("dda", "isotope_adduct", "start")
    assert msg == "Grouping isotope and adduct peaks…"


def test_stage_label_matches_underlying_stage_message():
    """``stage_label`` 应等价于 ``stage_message`` + extra_titles 显式调用。"""
    expected = stage_message(
        "deconvolve",
        "done",
        extra_titles=get_stage_titles("gcms"),
        elapsed=3.0,
    )
    assert stage_label("gcms", "deconvolve", "done", elapsed=3.0) == expected


def test_stage_label_unknown_stage_falls_back_to_key():
    """没有任何映射的 stage key 应直接按 key 渲染，作为安全后备。"""
    msg = stage_label("dda", "totally_unknown", "start")
    assert msg == "totally_unknown…"
