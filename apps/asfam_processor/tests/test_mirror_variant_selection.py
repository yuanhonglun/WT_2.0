"""P7 primary: when the library holds several collision-energy variants of the
SAME compound, the mirror plot DISPLAYS the variant whose spectrum best matches
the query (MS-DIAL parity) — while the counted / selected annotation stays
``annotation_matches[0]`` (top ``total_score``). Scheme (a): auto-pick the
displayed variant, no combo/selection change.

Exercises the real ``MS2PlotWidget._draw_current`` wiring headlessly.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PyQt5")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    from PyQt5.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _feature(matches):
    from asfam.models import Feature

    # Query looks like the 91 / 119 CE variant.
    feat = Feature(
        feature_id="F0",
        precursor_mz=179.07,
        rt=4.64,
        rt_left=4.5,
        rt_right=4.8,
        signal_type="ms2_only",
        ms2_mz=np.array([91.05, 119.05]),
        ms2_intensity=np.array([100.0, 60.0]),
        n_fragments=2,
    )
    feat.annotation_matches = matches
    feat.selected_annotation_idx = 0
    return feat


def test_mirror_displays_best_matching_variant_but_keeps_selection(qapp):
    from asfam.gui import ms2_plot as mod
    from asfam.models import AnnotationMatch

    feat = _feature([
        # top-score variant — base peaks 147/161, does NOT match the query
        AnnotationMatch(rank=1, name="Coniferaldehyde", score=1.20,
                        ref_peaks=[(147.04, 100.0), (161.06, 99.0)]),
        # same compound, different CE — 91/119, matches the query
        AnnotationMatch(rank=2, name="Coniferaldehyde", score=1.05,
                        ref_peaks=[(91.05, 100.0), (119.05, 61.0)]),
    ])
    w = mod.MS2PlotWidget()
    w.show_feature(feat)

    # Displayed reference = the query-matching CE variant (91/119),
    # NOT the top-score entry (147/161).
    assert w._ref_mz is not None
    assert sorted(round(m, 2) for m in w._ref_mz.tolist()) == [91.05, 119.05]

    # HARD invariant: selection / count untouched. Combo still at 0 and the
    # feature's selected annotation is still the top-score entry.
    assert w.annotation_combo.currentIndex() == 0
    assert feat.selected_annotation is feat.annotation_matches[0]
    assert feat.annotation_matches[0].score == 1.20
    w.close()


def test_user_combo_switch_is_respected(qapp):
    """A manual candidate selection (index > 0) is honored — the auto-pick
    only applies in the default (index 0) view."""
    from asfam.gui import ms2_plot as mod
    from asfam.models import AnnotationMatch

    feat = _feature([
        AnnotationMatch(rank=1, name="C", score=1.20,
                        ref_peaks=[(147.04, 100.0), (161.06, 99.0)]),
        AnnotationMatch(rank=2, name="C", score=1.10,
                        ref_peaks=[(200.0, 100.0), (210.0, 80.0)]),
        AnnotationMatch(rank=3, name="C", score=1.00,
                        ref_peaks=[(91.05, 100.0), (119.05, 61.0)]),
    ])
    w = mod.MS2PlotWidget()
    w.show_feature(feat)

    # Default view auto-picks the query-matching variant (91/119).
    assert sorted(round(m, 2) for m in w._ref_mz.tolist()) == [91.05, 119.05]

    # User switches to candidate #2 (index 1): display honors it (200/210),
    # the auto-pick is bypassed.
    w.annotation_combo.setCurrentIndex(1)
    assert sorted(round(m, 2) for m in w._ref_mz.tolist()) == [200.0, 210.0]
    w.close()
