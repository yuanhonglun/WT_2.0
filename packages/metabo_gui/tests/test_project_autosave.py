"""Auto-save path generator uses the data dir + standard timestamp pattern."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pytest

from metabo_gui.project_autosave import auto_save_path, first_existing


def test_path_components(tmp_path: Path):
    sample = tmp_path / "sample.mzML"
    sample.write_text("")
    ts = datetime(2026, 4, 30, 14, 23, 17)
    p = auto_save_path(sample, app_prefix="ASFAM", extension="asfam",
                       timestamp=ts)
    assert p.parent == tmp_path
    assert p.name == "ASFAMProject_20260430_142317.asfam"


def test_extension_strips_leading_dot(tmp_path: Path):
    p = auto_save_path(tmp_path / "x.mzML", app_prefix="GCMS",
                       extension=".gcmsproj",
                       timestamp=datetime(2026, 1, 1, 0, 0, 0))
    assert p.suffix == ".gcmsproj"


def test_default_timestamp_uses_now(tmp_path: Path):
    p = auto_save_path(tmp_path / "x.mzML", app_prefix="X", extension="xx")
    # Filename matches the YYYYMMDD_HHMMSS pattern between prefix and ext
    assert re.match(r"XProject_\d{8}_\d{6}\.xx$", p.name)


def test_first_existing_picks_first(tmp_path: Path):
    a = tmp_path / "a.mzML"; a.write_text("")
    b = tmp_path / "b.mzML"
    assert first_existing([b, a]) == a


def test_first_existing_returns_none_when_no_match(tmp_path: Path):
    assert first_existing([tmp_path / "missing1", tmp_path / "missing2"]) is None
