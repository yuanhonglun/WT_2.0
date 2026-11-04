"""features.csv carries the precursor-provenance columns and the MS2-only tier.

Reviewer 4 (2026-10): "the number of supporting fragments is exported with every
inferred precursor" was claimed but no such column existed, nor one naming the
source of each feature's precursor m/z. Both are pinned here, together with
the derived ``window_delta_mda`` cross-check and the ``ms2_only_tier`` label
that downstream statistics filter on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asfam.config import ProcessingConfig
from asfam.models import Feature
from asfam.pipeline.stage8_export import _export_csv, ms2_only_tier, window_delta_mda

NEW_COLUMNS = ("mz_source", "mz_confidence", "inferred_mz", "nl_n_support",
               "ms1_window_mz", "window_delta_mda", "ms2_only_tier")


def _feature(fid, *, signal_type="ms2_only", n_fragments=4, precursor_mz=195.0877,
             **overrides) -> Feature:
    feat = Feature(
        feature_id=fid, precursor_mz=precursor_mz, rt=5.0, rt_left=4.9, rt_right=5.1,
        signal_type=signal_type,
        ms2_mz=np.linspace(100.0, 180.0, n_fragments),
        ms2_intensity=np.linspace(1000.0, 500.0, n_fragments),
        n_fragments=n_fragments,
    )
    for key, value in overrides.items():
        setattr(feat, key, value)
    return feat


def _export(features, tmp_path) -> pd.DataFrame:
    out = tmp_path / "features.csv"
    _export_csv(features, out, ProcessingConfig())
    return pd.read_csv(out, comment="#", keep_default_na=False)


def test_every_provenance_column_is_present_and_filled_on_a_consensus_row(tmp_path):
    feat = _feature("F1", mz_source="nl_consensus", mz_confidence="high",
                    inferred_mz=195.0877, nl_n_support=3, ms1_window_mz=195.1100)
    df = _export([feat], tmp_path)
    for col in NEW_COLUMNS:
        assert col in df.columns, f"missing column {col}"
    row = df.iloc[0]
    assert row["mz_source"] == "nl_consensus"
    assert row["mz_confidence"] == "high"
    assert float(row["inferred_mz"]) == pytest.approx(195.0877)
    assert int(row["nl_n_support"]) == 3
    assert float(row["ms1_window_mz"]) == pytest.approx(195.11)
    # (195.0877 - 195.1100) * 1000 = -22.3 mDa, rounded to 2 decimals
    assert float(row["window_delta_mda"]) == pytest.approx(-22.3, abs=0.01)
    assert row["ms2_only_tier"] == "B"


def test_ms1_detected_rows_have_blank_provenance_and_no_tier(tmp_path):
    feat = _feature("F2", signal_type="ms1_detected", mz_source="ms1_peak")
    row = _export([feat], tmp_path).iloc[0]
    assert row["mz_source"] == "ms1_peak"
    assert row["inferred_mz"] == "" and row["nl_n_support"] == ""
    assert row["ms1_window_mz"] == "" and row["window_delta_mda"] == ""
    assert row["ms2_only_tier"] == ""


def test_window_delta_is_blank_on_a_lowres_row_where_it_would_be_zero_by_construction():
    feat = _feature("F3", mz_source="ms1_lowres", mz_confidence="low",
                    precursor_mz=195.11, ms1_window_mz=195.11)
    assert window_delta_mda(feat) is None
    feat.mz_source = "ms1_relaxed"
    feat.precursor_mz = 195.1000
    assert window_delta_mda(feat) == pytest.approx(-10.0)


@pytest.mark.parametrize("n_fragments, source, confidence, expected", [
    (4, "ms1_peak", "", "A"),
    (3, "ms1_relaxed", "", "A"),
    (5, "nl_consensus", "high", "B"),
    (3, "nl_consensus", "medium", "B"),
    (3, "nl_consensus", "low", "C"),
    (4, "ms1_lowres", "low", "C"),
    (4, "library", "", "C"),
    (6, "", "", "C"),
    (2, "ms1_lowres", "low", "D"),
    (1, "", "", "D"),
    (2, "ms1_peak", "", "D"),
])
def test_ms2_only_tier_definition(n_fragments, source, confidence, expected):
    feat = _feature("T", n_fragments=n_fragments, mz_source=source,
                    mz_confidence=confidence)
    assert ms2_only_tier(feat, 3) == expected


def test_tier_floor_follows_the_inference_fragment_floor():
    feat = _feature("T", n_fragments=3, mz_source="nl_consensus", mz_confidence="high")
    assert ms2_only_tier(feat, 3) == "B"
    assert ms2_only_tier(feat, 4) == "D"
    assert ms2_only_tier(_feature("M", signal_type="ms1_detected", n_fragments=1), 3) == ""
