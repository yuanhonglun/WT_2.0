"""P7: exported REFERENCE spectrum is base-peak-normalized to 100 (MS-DIAL
Relative parity), while the raw ``match.ref_peaks`` fed to the scorer stay
byte-identical and the MEASURED export stays raw (real data).

These exercise the real ``MS2PlotWidget._on_export_spectrum`` wiring headlessly
(offscreen Qt from conftest), monkeypatching the file dialog + writer so the
peak list handed to ``write_spectrum`` can be inspected.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

pytest.importorskip("PyQt5")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    from PyQt5.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _feature_with_ref(ref_peaks, *, ms2_mz, ms2_int):
    from asfam.models import AnnotationMatch, Feature

    feat = Feature(
        feature_id="F0",
        precursor_mz=258.05,
        rt=5.0,
        rt_left=4.8,
        rt_right=5.2,
        signal_type="ms2_only",
        ms2_mz=np.asarray(ms2_mz, dtype=float),
        ms2_intensity=np.asarray(ms2_int, dtype=float),
        n_fragments=len(ms2_mz),
    )
    match = AnnotationMatch(rank=1, name="Compound", score=0.9,
                            ref_peaks=list(ref_peaks))
    feat.annotation_matches = [match]
    feat.selected_annotation_idx = 0
    return feat, match


def _patch_export(monkeypatch, mod, tmp_path, name):
    """Capture the peaks passed to write_spectrum; stub dialogs."""
    captured: dict = {}
    monkeypatch.setattr(
        mod, "write_spectrum",
        lambda path, peaks, meta, fmt="msp": (
            captured.update(peaks=peaks, fmt=fmt) or path),
    )
    from PyQt5.QtWidgets import QFileDialog, QMessageBox

    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        lambda *a, **k: (str(tmp_path / name), "MSP (*.msp)"))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    return captured


def test_reference_export_base_peak_100_and_raw_untouched(
        qapp, monkeypatch, tmp_path):
    from asfam.gui import ms2_plot as mod

    raw_ref = [(258.05, 999.0), (229.05, 152.0), (152.01, 5.0)]
    feat, match = _feature_with_ref(
        raw_ref, ms2_mz=[258.05, 229.05], ms2_int=[100.0, 60.0])
    snapshot = [tuple(p) for p in match.ref_peaks]

    w = mod.MS2PlotWidget()
    w.show_feature(feat)
    captured = _patch_export(monkeypatch, mod, tmp_path, "ref.msp")

    w._on_export_spectrum("Reference → MSP")
    w.close()

    peaks = captured["peaks"]
    # Base peak scaled to 100 for MS-DIAL Relative parity.
    assert math.isclose(max(v for _, v in peaks), 100.0)
    assert math.isclose(peaks[1][1], 152.0 / 999.0 * 100.0, rel_tol=1e-9)
    # m/z and order preserved.
    assert [round(m, 4) for m, _ in peaks] == [258.05, 229.05, 152.01]
    # HARD invariant: the raw ref peaks fed to the scorer are byte-identical
    # (display normalization operates on a copy, never mutates the match).
    assert [tuple(p) for p in match.ref_peaks] == snapshot


def test_measured_export_stays_raw(qapp, monkeypatch, tmp_path):
    """Measured spectrum export must NOT be normalized — it is real data.

    Uses a single-peak measured spectrum to sidestep an UNRELATED pre-existing
    bug in the measured branch (``list(feat.ms2_mz or [])`` raises on a
    multi-element numpy array); that bug is outside T5-P7 scope (which only
    touches the reference export at ms2_plot.py:355) and is flagged separately.
    A single peak still proves the measured intensity is passed through raw.
    """
    from asfam.gui import ms2_plot as mod

    feat, _ = _feature_with_ref(
        [(258.05, 999.0)], ms2_mz=[258.05], ms2_int=[4200.0])

    w = mod.MS2PlotWidget()
    w.show_feature(feat)
    captured = _patch_export(monkeypatch, mod, tmp_path, "measured.msp")

    w._on_export_spectrum("Measured → MSP")
    w.close()

    peaks = captured["peaks"]
    # Unchanged raw intensity (no base-peak scaling on measured data).
    assert max(v for _, v in peaks) == 4200.0
