"""dump_feedback_to_export_dir writes feedback.csv + feedback.json."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from metabo_gui.feedback.exporter import dump_feedback_to_export_dir
from metabo_gui.feedback.models import (
    FeatureSignature, FeedbackEntry, FeedbackStore, RunContext,
)


def _store_with(entries: list[FeedbackEntry]) -> FeedbackStore:
    ctx = RunContext(
        app="dda", software_version="0.0.0", run_timestamp="t",
        input_files=["/x/y/a.mzML"], input_root="/x/y",
        library_path=None, project_file=None, export_dir=None, params={},
    )
    return FeedbackStore(schema_version=1, app="dda", software_version="0.0.0",
                        run_context=ctx, entries=entries)


def _entry(fid="F1", tags=("peak_split",), verified=False, comment="hi"):
    return FeedbackEntry(
        feature_id_at_run=fid,
        feature_signature=FeatureSignature(mz=280.0, rt=5.0, mode="dda"),
        tags=list(tags), verified_good=verified, comment=comment,
        created_at="2026-05-14T10:30:00", updated_at="2026-05-14T10:31:00",
        run_timestamp_created="2026-05-14T10:00:00",
    )


def test_dump_writes_both_files(tmp_path):
    store = _store_with([_entry()])
    dump_feedback_to_export_dir(tmp_path, store)
    assert (tmp_path / "feedback.csv").exists()
    assert (tmp_path / "feedback.json").exists()


def test_feedback_csv_column_order(tmp_path):
    dump_feedback_to_export_dir(tmp_path, _store_with([_entry()]))
    with open(tmp_path / "feedback.csv", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == [
        "feature_id", "mz", "rt", "mode",
        "tags", "verified_good", "comment",
        "created_at", "updated_at",
    ]


def test_multi_tag_joined_by_pipe(tmp_path):
    e = _entry(tags=("peak_split", "wrong_annot"))
    dump_feedback_to_export_dir(tmp_path, _store_with([e]))
    rows = list(csv.DictReader(open(tmp_path / "feedback.csv", encoding="utf-8")))
    assert rows[0]["tags"] == "peak_split|wrong_annot"


def test_empty_entries_still_writes_files_with_header(tmp_path):
    dump_feedback_to_export_dir(tmp_path, _store_with([]))
    assert (tmp_path / "feedback.csv").read_text(encoding="utf-8").startswith("feature_id,")
    payload = json.loads((tmp_path / "feedback.json").read_text(encoding="utf-8"))
    assert payload["entries"] == []
    assert payload["run_context"]["app"] == "dda"


def test_feedback_json_carries_run_context_and_export_marker(tmp_path):
    dump_feedback_to_export_dir(tmp_path, _store_with([_entry()]))
    payload = json.loads((tmp_path / "feedback.json").read_text(encoding="utf-8"))
    assert "exported_at" in payload
    assert payload["run_context"]["input_files"] == ["/x/y/a.mzML"]
    assert len(payload["entries"]) == 1


def test_verified_good_only_entry_written_with_empty_tags(tmp_path):
    e = _entry(tags=(), verified=True, comment="looks fine")
    dump_feedback_to_export_dir(tmp_path, _store_with([e]))
    row = next(csv.DictReader(open(tmp_path / "feedback.csv", encoding="utf-8")))
    assert row["tags"] == ""
    assert row["verified_good"] == "True"


def test_export_dir_created_if_missing(tmp_path):
    target = tmp_path / "new_subdir"
    assert not target.exists()
    dump_feedback_to_export_dir(target, _store_with([]))
    assert target.exists()
    assert (target / "feedback.csv").exists()
