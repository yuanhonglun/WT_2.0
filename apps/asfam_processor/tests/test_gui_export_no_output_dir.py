"""Regression: Export must respond after opening a project.

Repro of the reported bug: open a project, click Export, nothing happens.
After loading a project the setup panel's output-directory field is empty
(the .asfam file never stored it), so the old ``_on_export`` guard
``if output_dir and self.features`` silently no-opped.
"""
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


def _feat(feature_id="F0", *, score=0.9, name="Compound"):
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
    f.annotation_matches = [AnnotationMatch(rank=1, name=name, score=float(score))]
    f.selected_annotation_idx = 0
    f.name = name
    return f


@pytest.fixture
def main_window(qapp_factory, monkeypatch):
    """A MainWindow with the background update thread disabled."""
    _ = qapp_factory()
    from asfam.gui import main_window as mw_mod

    # Avoid the GitHub update-check QThread (network + dangling thread).
    monkeypatch.setattr(mw_mod._UpdateChecker, "start", lambda self: None)
    win = mw_mod.MainWindow()
    yield win
    win.close()


def test_export_prompts_when_output_dir_unset(main_window, monkeypatch, tmp_path):
    """With features loaded but no output dir set, Export prompts for a
    directory and then runs the exporter (instead of silently no-opping)."""
    mw = main_window
    mw.features = [_feat("F0")]
    mw.setup_panel.out_path.clear()
    assert mw.setup_panel.get_output_dir() == ""

    from PyQt5.QtWidgets import QFileDialog, QMessageBox

    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory",
        lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)

    captured = {}
    import asfam.pipeline.stage8_export as stage8

    def fake_run_stage8(features, output_dir, config, feedback_store=None):
        captured["output_dir"] = output_dir
        captured["n_features"] = len(features)
        return {"features.csv": "ok"}

    monkeypatch.setattr(stage8, "run_stage8", fake_run_stage8)

    mw._on_export()

    assert captured.get("output_dir") == str(tmp_path)
    assert captured.get("n_features") == 1
    # Chosen directory is reflected back into the panel for reuse.
    assert mw.setup_panel.get_output_dir() == str(tmp_path)


def test_export_cancel_dialog_is_noop(main_window, monkeypatch):
    """Cancelling the directory prompt must not call the exporter."""
    mw = main_window
    mw.features = [_feat("F0")]
    mw.setup_panel.out_path.clear()

    from PyQt5.QtWidgets import QFileDialog

    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *a, **k: "")

    called = {"n": 0}
    import asfam.pipeline.stage8_export as stage8
    monkeypatch.setattr(
        stage8, "run_stage8",
        lambda *a, **k: called.__setitem__("n", called["n"] + 1))

    mw._on_export()
    assert called["n"] == 0


def test_export_warns_without_features(main_window, monkeypatch):
    """No results loaded -> warn the user instead of silently no-opping."""
    mw = main_window
    mw.features = []

    from PyQt5.QtWidgets import QMessageBox

    warned = {"n": 0}
    monkeypatch.setattr(
        QMessageBox, "warning",
        lambda *a, **k: warned.__setitem__("n", warned["n"] + 1))

    called = {"n": 0}
    import asfam.pipeline.stage8_export as stage8
    monkeypatch.setattr(
        stage8, "run_stage8",
        lambda *a, **k: called.__setitem__("n", called["n"] + 1))

    mw._on_export()
    assert warned["n"] == 1
    assert called["n"] == 0
