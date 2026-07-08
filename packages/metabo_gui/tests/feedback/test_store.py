"""Sidecar file IO for FeedbackStore."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from metabo_gui.feedback.models import FeedbackStore, RunContext
from metabo_gui.feedback.store import (
    load_alongside,
    save_alongside,
    sidecar_path_for,
)


def _empty_store(app: str = "asfam") -> FeedbackStore:
    return FeedbackStore(
        schema_version=1,
        app=app,
        metra_version="0.0.0",
        run_context=RunContext(
            app=app, metra_version="0.0.0",
            run_timestamp="t", input_files=[], input_root="",
            library_path=None, project_file=None, export_dir=None, params={},
        ),
        entries=[],
    )


def test_sidecar_path_appends_feedback_json(tmp_path):
    p = tmp_path / "my_run.asfam"
    assert sidecar_path_for(p) == tmp_path / "my_run.asfam.feedback.json"


def test_sidecar_path_works_for_any_extension(tmp_path):
    for ext in (".asfam", ".dda", ".gcmsproj"):
        p = tmp_path / f"x{ext}"
        assert sidecar_path_for(p) == tmp_path / f"x{ext}.feedback.json"


def test_load_alongside_returns_none_when_missing(tmp_path):
    p = tmp_path / "my_run.asfam"
    assert load_alongside(p) is None


def test_load_alongside_returns_none_when_invalid_json(tmp_path):
    p = tmp_path / "my_run.asfam"
    sidecar_path_for(p).write_text("{not valid json", encoding="utf-8")
    assert load_alongside(p) is None


def test_save_then_load_roundtrip(tmp_path):
    p = tmp_path / "my_run.asfam"
    store = _empty_store()
    save_alongside(p, store)
    loaded = load_alongside(p)
    assert loaded == store


def test_save_alongside_writes_via_tmp_then_rename(tmp_path):
    """The temp file must be removed and the final file present after save."""
    p = tmp_path / "my_run.asfam"
    save_alongside(p, _empty_store())
    final = sidecar_path_for(p)
    assert final.exists()
    # No stray .tmp files in the directory
    leftovers = [f for f in tmp_path.iterdir() if f.name.endswith(".tmp")]
    assert leftovers == []


def test_load_alongside_handles_unknown_schema_version(tmp_path):
    p = tmp_path / "my_run.asfam"
    bad = {"schema_version": 99, "app": "asfam"}
    sidecar_path_for(p).write_text(json.dumps(bad), encoding="utf-8")
    assert load_alongside(p) is None
