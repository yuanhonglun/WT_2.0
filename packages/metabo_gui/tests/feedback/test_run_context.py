"""Test run context builders: build_run_context, params_to_jsonable, feature_signature_from_components."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from metabo_gui.feedback.run_context import (
    build_run_context,
    common_input_root,
    feature_signature_from_components,
    params_to_jsonable,
)


def test_common_input_root_single_file():
    # Use os.path.commonpath via Path semantics — on Windows, both forward and
    # backslash paths collapse the same way. Use a path the OS can normalize.
    import os
    files = [os.path.join("a", "b", "c", "x.mzML")]
    assert common_input_root(files) == os.path.join("a", "b", "c")


def test_common_input_root_multiple_files():
    import os
    files = [
        os.path.join("a", "b", "c", "x.mzML"),
        os.path.join("a", "b", "c", "y.mzML"),
        os.path.join("a", "b", "c", "sub", "z.mzML"),
    ]
    assert common_input_root(files) == os.path.join("a", "b", "c")


def test_common_input_root_empty():
    assert common_input_root([]) == ""


def test_params_to_jsonable_handles_numpy():
    d = {
        "arr": np.array([1.0, 2.0]),
        "scalar": np.float64(3.14),
        "int_val": np.int32(7),
        "nested": {"path": Path("/a/b")},
    }
    out = params_to_jsonable(d)
    import json
    json.dumps(out)  # must be JSON serializable
    assert out["scalar"] == 3.14
    assert out["int_val"] == 7
    assert out["arr"] == [1.0, 2.0]
    # Path should be coerced to string; accept either separator
    assert isinstance(out["nested"]["path"], str)


def test_params_to_jsonable_repr_fallback_for_unknown_object():
    class Weird:
        def __repr__(self):
            return "<Weird>"
    out = params_to_jsonable({"w": Weird()})
    assert out["w"] == "<Weird>"


def test_build_run_context_full():
    import os
    ctx = build_run_context(
        app="dda",
        metra_version="0.7.260514.10",
        input_files=["/x/y/a.mzML", "/x/y/b.mzML"],
        library_path="/libs/pos.msp",
        project_file=None,
        export_dir=None,
        params={"thr": 0.5},
    )
    assert ctx.app == "dda"
    # Normalize path separators for cross-platform testing
    expected_root = os.path.normpath(os.path.join("/x", "y"))
    actual_root = os.path.normpath(ctx.input_root)
    assert actual_root == expected_root
    assert ctx.library_path == "/libs/pos.msp"
    assert ctx.params == {"thr": 0.5}
    assert ctx.run_timestamp  # auto-filled with a non-empty timestamp


def test_build_run_context_accepts_explicit_timestamp():
    ctx = build_run_context(
        app="asfam",
        metra_version="0.0.0",
        input_files=[],
        run_timestamp="2026-05-14T10:00:00",
    )
    assert ctx.run_timestamp == "2026-05-14T10:00:00"


def test_feature_signature_from_components_basic():
    sig = feature_signature_from_components(mz=280.1632, rt=5.42, mode="dda")
    assert sig.mz == 280.1632
    assert sig.rt == 5.42
    assert sig.mode == "dda"
