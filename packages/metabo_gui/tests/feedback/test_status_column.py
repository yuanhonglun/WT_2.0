"""FeatureStatusDelegate renders status text for a feature_id."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")


def test_status_text_empty_when_no_entry(qapp):
    from metabo_gui.feedback.status_column import status_text_for_entry
    assert status_text_for_entry(None) == ""


def test_status_text_uses_verified_check_when_only_verified(qapp):
    from metabo_gui.feedback.models import FeatureSignature, FeedbackEntry
    from metabo_gui.feedback.status_column import status_text_for_entry
    e = FeedbackEntry(
        feature_id_at_run="F1",
        feature_signature=FeatureSignature(mz=1, rt=1, mode="dda"),
        tags=[], verified_good=True, comment="",
        created_at="t", updated_at="t", run_timestamp_created="t",
    )
    assert status_text_for_entry(e) == "✓"


def test_status_text_uses_first_tag_short_label_when_issue_present(qapp):
    from metabo_gui.feedback.models import FeatureSignature, FeedbackEntry
    from metabo_gui.feedback.status_column import status_text_for_entry
    e = FeedbackEntry(
        feature_id_at_run="F1",
        feature_signature=FeatureSignature(mz=1, rt=1, mode="dda"),
        tags=["peak_split", "wrong_annot"], verified_good=False, comment="",
        created_at="t", updated_at="t", run_timestamp_created="t",
    )
    assert "split" in status_text_for_entry(e).lower()


def test_status_tooltip_lists_all_tags_and_comment_first_line(qapp):
    from metabo_gui.feedback.models import FeatureSignature, FeedbackEntry
    from metabo_gui.feedback.status_column import tooltip_for_entry
    e = FeedbackEntry(
        feature_id_at_run="F1",
        feature_signature=FeatureSignature(mz=1, rt=1, mode="dda"),
        tags=["peak_split", "wrong_annot"], verified_good=True,
        comment="line one\nline two",
        created_at="t", updated_at="t", run_timestamp_created="t",
    )
    tip = tooltip_for_entry(e)
    assert "peak_split" in tip
    assert "wrong_annot" in tip
    assert "verified" in tip.lower()
    assert "line one" in tip


def test_tooltip_empty_when_no_entry(qapp):
    from metabo_gui.feedback.status_column import tooltip_for_entry
    assert tooltip_for_entry(None) == ""


def test_delegate_renders_without_crashing(qapp):
    from PyQt5.QtCore import QModelIndex, QRect, Qt
    from PyQt5.QtGui import QPainter, QPixmap
    from PyQt5.QtWidgets import QStyleOptionViewItem
    from metabo_gui.feedback.controller import FeedbackController
    from metabo_gui.feedback.models import (
        FeatureSignature, FeedbackEntry, FeedbackStore, RunContext,
    )
    from metabo_gui.feedback.status_column import FeatureStatusDelegate

    ctx = RunContext(app="dda", software_version="0", run_timestamp="t",
                     input_files=[], input_root="", library_path=None,
                     project_file=None, export_dir=None, params={})
    store = FeedbackStore(1, "dda", "0", ctx, [
        FeedbackEntry(
            "F1", FeatureSignature(1, 1, "dda"), ["peak_split"], False, "",
            "t", "t", "t",
        )
    ])
    ctl = FeedbackController("/tmp/x.dda", store, autosave_ms=10_000)
    delegate = FeatureStatusDelegate(controller=ctl, feature_id_provider=lambda idx: "F1")

    pm = QPixmap(100, 30)
    pm.fill(Qt.white)
    painter = QPainter(pm)
    opt = QStyleOptionViewItem()
    opt.rect = QRect(0, 0, 100, 30)
    delegate.paint(painter, opt, QModelIndex())
    painter.end()  # no crash = pass
