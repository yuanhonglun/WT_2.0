"""FeedbackEntry / FeatureSignature / RunContext / FeedbackStore dataclasses."""
from __future__ import annotations

import numpy as np
import pytest

from metabo_gui.feedback.models import (
    FeatureSignature,
    FeedbackEntry,
    FeedbackStore,
    RunContext,
)


def test_feature_signature_roundtrip():
    sig = FeatureSignature(mz=280.1632, rt=5.42, mode="dda")
    d = sig.to_dict()
    sig2 = FeatureSignature.from_dict(d)
    assert sig == sig2


def test_feedback_entry_roundtrip():
    e = FeedbackEntry(
        feature_id_at_run="F00123",
        feature_signature=FeatureSignature(mz=280.1632, rt=5.42, mode="dda"),
        tags=["peak_split", "wrong_annot"],
        verified_good=False,
        comment="hello",
        created_at="2026-05-14T10:30:00",
        updated_at="2026-05-14T10:31:00",
        run_timestamp_created="2026-05-14T10:00:00",
    )
    d = e.to_dict()
    e2 = FeedbackEntry.from_dict(d)
    assert e == e2


def test_run_context_minimal_fields():
    ctx = RunContext(
        app="dda",
        metra_version="0.7.260514.6",
        run_timestamp="2026-05-14T10:00:00",
        input_files=["/a/b/x.mzML", "/a/b/y.mzML"],
        input_root="/a/b",
        library_path=None,
        project_file=None,
        export_dir=None,
        params={"threshold": 0.5},
    )
    d = ctx.to_dict()
    ctx2 = RunContext.from_dict(d)
    assert ctx2 == ctx


def test_store_roundtrip_empty_entries():
    ctx = RunContext(
        app="asfam",
        metra_version="0.7.260514.6",
        run_timestamp="2026-05-14T10:00:00",
        input_files=[],
        input_root="",
        library_path=None,
        project_file=None,
        export_dir=None,
        params={},
    )
    store = FeedbackStore(
        schema_version=1,
        app="asfam",
        metra_version="0.7.260514.6",
        run_context=ctx,
        entries=[],
    )
    d = store.to_dict()
    store2 = FeedbackStore.from_dict(d)
    assert store2 == store


def test_store_authoritative_outer_fields_on_mismatch(caplog):
    """If outer app/metra_version disagree with run_context, outer wins, log warning."""
    raw = {
        "schema_version": 1,
        "app": "asfam",
        "metra_version": "0.7.260514.6",
        "run_context": {
            "app": "dda",
            "metra_version": "0.0.0",
            "run_timestamp": "2026-05-14T10:00:00",
            "input_files": [],
            "input_root": "",
            "library_path": None,
            "project_file": None,
            "export_dir": None,
            "params": {},
        },
        "entries": [],
    }
    with caplog.at_level("WARNING"):
        store = FeedbackStore.from_dict(raw)
    assert store.app == "asfam"
    assert "mismatch" in caplog.text.lower()
