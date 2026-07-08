"""NotesDock: tag checkboxes, verified flag, comment editor."""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PyQt5")


@pytest.fixture
def controller(qapp, tmp_path):
    from metabo_gui.feedback.controller import FeedbackController
    from metabo_gui.feedback.models import FeedbackStore, RunContext
    ctx = RunContext(app="dda", metra_version="0", run_timestamp="t",
                     input_files=[], input_root="", library_path=None,
                     project_file=str(tmp_path / "x.dda"), export_dir=None, params={})
    store = FeedbackStore(1, "dda", "0", ctx, [])
    return FeedbackController(tmp_path / "x.dda", store, autosave_ms=10_000)


def test_dock_starts_empty(qapp, controller):
    from metabo_gui.feedback.widget import NotesDock
    dock = NotesDock(controller)
    assert dock.current_feature_id is None
    assert not dock.comment_edit.toPlainText()


def test_set_current_feature_renders_existing_entry(qapp, controller):
    from metabo_gui.feedback.models import FeatureSignature, FeedbackEntry
    from metabo_gui.feedback.widget import NotesDock

    controller.upsert_entry(FeedbackEntry(
        feature_id_at_run="F1",
        feature_signature=FeatureSignature(mz=280.0, rt=5.0, mode="dda"),
        tags=["peak_split"], verified_good=True, comment="hello",
        created_at="t", updated_at="t", run_timestamp_created="t",
    ))
    dock = NotesDock(controller)
    dock.set_current_feature("F1", FeatureSignature(mz=280.0, rt=5.0, mode="dda"))
    assert dock.comment_edit.toPlainText() == "hello"
    assert dock.tag_checkboxes["peak_split"].isChecked()
    assert dock.verified_checkbox.isChecked()


def test_toggling_tag_creates_entry(qapp, controller):
    from metabo_gui.feedback.models import FeatureSignature
    from metabo_gui.feedback.widget import NotesDock

    dock = NotesDock(controller)
    dock.set_current_feature("F1", FeatureSignature(mz=1, rt=1, mode="dda"))
    dock.tag_checkboxes["noise"].setChecked(True)
    entry = controller.get_entry("F1")
    assert entry is not None
    assert entry.tags == ["noise"]


def test_toggling_verified_only_creates_entry(qapp, controller):
    from metabo_gui.feedback.models import FeatureSignature
    from metabo_gui.feedback.widget import NotesDock

    dock = NotesDock(controller)
    dock.set_current_feature("F1", FeatureSignature(mz=1, rt=1, mode="dda"))
    dock.verified_checkbox.setChecked(True)
    entry = controller.get_entry("F1")
    assert entry is not None
    assert entry.verified_good
    assert entry.tags == []


def test_clearing_all_inputs_removes_entry(qapp, controller):
    from metabo_gui.feedback.models import FeatureSignature
    from metabo_gui.feedback.widget import NotesDock

    dock = NotesDock(controller)
    dock.set_current_feature("F1", FeatureSignature(mz=1, rt=1, mode="dda"))
    dock.tag_checkboxes["noise"].setChecked(True)
    assert controller.get_entry("F1") is not None
    dock._clear_current_no_confirm()
    assert controller.get_entry("F1") is None


def test_comment_autosaves_after_debounce(qapp, controller):
    from PyQt5.QtCore import QCoreApplication
    from metabo_gui.feedback.models import FeatureSignature
    from metabo_gui.feedback.widget import NotesDock

    dock = NotesDock(controller, comment_debounce_ms=20)
    dock.set_current_feature("F1", FeatureSignature(mz=1, rt=1, mode="dda"))
    dock.comment_edit.setPlainText("some thought")
    deadline = time.time() + 2.0
    while time.time() < deadline:
        QCoreApplication.processEvents()
        e = controller.get_entry("F1")
        if e and e.comment == "some thought":
            break
    e = controller.get_entry("F1")
    assert e is not None
    assert e.comment == "some thought"


def test_switching_feature_flushes_pending_comment(qapp, controller):
    from metabo_gui.feedback.models import FeatureSignature
    from metabo_gui.feedback.widget import NotesDock

    dock = NotesDock(controller, comment_debounce_ms=10_000)
    sig = FeatureSignature(mz=1, rt=1, mode="dda")
    dock.set_current_feature("F1", sig)
    dock.comment_edit.setPlainText("important")
    dock.set_current_feature("F2", FeatureSignature(mz=2, rt=2, mode="dda"))
    e1 = controller.get_entry("F1")
    assert e1 is not None
    assert e1.comment == "important"


def test_loading_entry_does_not_trigger_writeback(qapp, controller):
    """When the dock loads an existing entry to display it, it must NOT
    re-upsert (would corrupt updated_at timestamp and trigger autosave loop).
    """
    from metabo_gui.feedback.models import FeatureSignature, FeedbackEntry
    from metabo_gui.feedback.widget import NotesDock

    controller.upsert_entry(FeedbackEntry(
        feature_id_at_run="F1",
        feature_signature=FeatureSignature(mz=1, rt=1, mode="dda"),
        tags=["noise"], verified_good=False, comment="original",
        created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00",
        run_timestamp_created="2026-01-01T00:00:00",
    ))
    dock = NotesDock(controller)
    dock.set_current_feature("F1", FeatureSignature(mz=1, rt=1, mode="dda"))
    # Updated_at should NOT have changed just from loading
    e_after_load = controller.get_entry("F1")
    assert e_after_load.updated_at == "2026-01-01T00:00:00"
