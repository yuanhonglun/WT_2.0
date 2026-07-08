"""FeedbackController wraps FeedbackStore with autosave QTimer."""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PyQt5")


@pytest.fixture
def project_path(tmp_path):
    return tmp_path / "demo.dda"


@pytest.fixture
def empty_store():
    from metabo_gui.feedback.models import FeedbackStore, RunContext
    ctx = RunContext(app="dda", metra_version="0.0.0", run_timestamp="t",
                     input_files=[], input_root="", library_path=None,
                     project_file=None, export_dir=None, params={})
    return FeedbackStore(schema_version=1, app="dda", metra_version="0.0.0",
                         run_context=ctx, entries=[])


def test_update_entry_marks_dirty_and_autosaves(qapp, project_path, empty_store):
    from PyQt5.QtCore import QCoreApplication
    from metabo_gui.feedback.controller import FeedbackController
    from metabo_gui.feedback.models import FeatureSignature, FeedbackEntry
    from metabo_gui.feedback.store import sidecar_path_for

    ctl = FeedbackController(project_path, empty_store, autosave_ms=10)

    e = FeedbackEntry(
        feature_id_at_run="F1",
        feature_signature=FeatureSignature(mz=280.0, rt=5.0, mode="dda"),
        tags=["peak_split"], verified_good=False, comment="x",
        created_at="t", updated_at="t", run_timestamp_created="t",
    )
    ctl.upsert_entry(e)
    assert ctl.is_dirty
    deadline = time.time() + 2.0
    while time.time() < deadline:
        QCoreApplication.processEvents()
        if not ctl.is_dirty:
            break
    assert not ctl.is_dirty
    assert sidecar_path_for(project_path).exists()


def test_save_now_flushes_immediately(qapp, project_path, empty_store):
    from metabo_gui.feedback.controller import FeedbackController
    from metabo_gui.feedback.models import FeatureSignature, FeedbackEntry
    from metabo_gui.feedback.store import sidecar_path_for

    ctl = FeedbackController(project_path, empty_store, autosave_ms=10_000)
    e = FeedbackEntry(
        feature_id_at_run="F1",
        feature_signature=FeatureSignature(mz=280.0, rt=5.0, mode="dda"),
        tags=[], verified_good=True, comment="",
        created_at="t", updated_at="t", run_timestamp_created="t",
    )
    ctl.upsert_entry(e)
    ctl.save_now()
    assert not ctl.is_dirty
    assert sidecar_path_for(project_path).exists()


def test_upsert_replaces_existing_entry_for_same_feature_id(qapp, project_path, empty_store):
    from metabo_gui.feedback.controller import FeedbackController
    from metabo_gui.feedback.models import FeatureSignature, FeedbackEntry

    ctl = FeedbackController(project_path, empty_store, autosave_ms=10_000)
    sig = FeatureSignature(mz=280.0, rt=5.0, mode="dda")
    ctl.upsert_entry(FeedbackEntry(
        feature_id_at_run="F1", feature_signature=sig,
        tags=["noise"], verified_good=False, comment="first",
        created_at="t", updated_at="t", run_timestamp_created="t",
    ))
    ctl.upsert_entry(FeedbackEntry(
        feature_id_at_run="F1", feature_signature=sig,
        tags=["peak_split"], verified_good=False, comment="second",
        created_at="t", updated_at="t2", run_timestamp_created="t",
    ))
    assert len(ctl.store.entries) == 1
    assert ctl.store.entries[0].comment == "second"


def test_remove_entry(qapp, project_path, empty_store):
    from metabo_gui.feedback.controller import FeedbackController
    from metabo_gui.feedback.models import FeatureSignature, FeedbackEntry

    ctl = FeedbackController(project_path, empty_store, autosave_ms=10_000)
    ctl.upsert_entry(FeedbackEntry(
        feature_id_at_run="F1",
        feature_signature=FeatureSignature(mz=280.0, rt=5.0, mode="dda"),
        tags=["noise"], verified_good=False, comment="",
        created_at="t", updated_at="t", run_timestamp_created="t",
    ))
    ctl.remove_entry("F1")
    assert ctl.store.entries == []


def test_get_entry_returns_none_for_unknown(qapp, project_path, empty_store):
    from metabo_gui.feedback.controller import FeedbackController
    ctl = FeedbackController(project_path, empty_store, autosave_ms=10_000)
    assert ctl.get_entry("F_missing") is None


def test_store_changed_signal_emitted_on_upsert(qapp, project_path, empty_store):
    from metabo_gui.feedback.controller import FeedbackController
    from metabo_gui.feedback.models import FeatureSignature, FeedbackEntry

    ctl = FeedbackController(project_path, empty_store, autosave_ms=10_000)
    emitted: list[str] = []
    ctl.storeChanged.connect(lambda fid: emitted.append(fid))
    ctl.upsert_entry(FeedbackEntry(
        feature_id_at_run="F1",
        feature_signature=FeatureSignature(mz=1.0, rt=1.0, mode="dda"),
        tags=["noise"], verified_good=False, comment="",
        created_at="t", updated_at="t", run_timestamp_created="t",
    ))
    assert emitted == ["F1"]
