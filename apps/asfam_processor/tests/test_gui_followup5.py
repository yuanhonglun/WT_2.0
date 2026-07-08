"""Smoke tests for the ASFAM GUI changes that mirror Plan F-followup-5."""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PyQt5")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp_factory():
    from PyQt5.QtWidgets import QApplication

    def _factory():
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    return _factory


def _feat(feature_id="F0", *, score=None, name="", is_dup=False):
    """Build a minimal Feature with optional annotation."""
    from asfam.models import AnnotationMatch, Feature

    f = Feature(
        feature_id=feature_id,
        precursor_mz=100.0,
        rt=5.0,
        rt_left=4.8,
        rt_right=5.2,
        signal_type="ms1_detected",
        ms2_mz=np.array([]),
        ms2_intensity=np.array([]),
        n_fragments=0,
    )
    if score is not None:
        f.annotation_matches = [
            AnnotationMatch(rank=1, name=name or "Compound", score=float(score))
        ]
        f.selected_annotation_idx = 0
        f.name = name or "Compound"
    f.is_duplicate = is_dup
    return f


def test_feature_table_has_score_column(qapp_factory):
    """Plan F-followup-5 mirror: Score column is present (now followed by
    WDP/RDP component columns)."""
    _ = qapp_factory()
    from asfam.gui.feature_table import COLUMNS

    headers = [label for label, _ in COLUMNS]
    assert "Score" in headers, f"Score column missing, got {headers!r}"
    assert "WDP" in headers and "RDP" in headers, f"got {headers!r}"


def test_feature_table_score_renders_three_decimals(qapp_factory):
    _ = qapp_factory()
    from asfam.gui.feature_table import FeatureTableWidget

    table = FeatureTableWidget()
    table.set_features([_feat("F0", score=0.85432)])
    # Inspect the model directly — proxy may have hidden the row.
    idx = table.model.index(0, len(table.model._features[0].__dataclass_fields__) and 13)
    # Easier: walk to the Score column by header.
    from PyQt5.QtCore import Qt
    from asfam.gui.feature_table import COLUMNS

    score_col = next(i for i, (h, _) in enumerate(COLUMNS) if h == "Score")
    cell = table.model.data(table.model.index(0, score_col), Qt.DisplayRole)
    assert cell == "0.854", f"got {cell!r}"


def test_feature_table_annotated_only_filter_uses_threshold(qapp_factory):
    _ = qapp_factory()
    from asfam.gui.feature_table import FeatureTableWidget

    table = FeatureTableWidget()
    feats = [
        _feat("F_high", score=0.95, name="A"),
        _feat("F_low", score=0.40, name="B"),  # below 0.8 threshold
        _feat("F_none"),                        # unannotated
    ]
    table.set_features(feats)
    assert table.proxy.rowCount() == 3  # no filter

    table.proxy.set_annotated_threshold(0.8)
    table.proxy.set_annotated_only(True)
    assert table.proxy.rowCount() == 1  # only F_high


def test_scatter_passes_annotated_gate_uses_score(qapp_factory):
    _ = qapp_factory()
    from asfam.gui.scatter_plot import ScatterPlotWidget

    w = ScatterPlotWidget()
    w.set_annotated_threshold(0.8)
    assert w._passes_annotated_gate(_feat("F", score=0.9)) is True
    assert w._passes_annotated_gate(_feat("F", score=0.5)) is False
    assert w._passes_annotated_gate(_feat("F")) is False


def test_eic_plot_smoothed_button_has_no_custom_stylesheet(qapp_factory):
    """Plan F-followup-5 mirror: drop the custom blue button styleSheet
    so the Smoothed/Raw button matches the GC-MS EIC viewer."""
    _ = qapp_factory()
    from asfam.gui.eic_plot import EICPlotWidget

    w = EICPlotWidget()
    assert w._btn_smooth.styleSheet() == "", (
        f"smoothed button still carries styleSheet: {w._btn_smooth.styleSheet()!r}"
    )
