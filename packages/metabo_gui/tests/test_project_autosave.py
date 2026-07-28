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


def test_directory_overrides_the_data_dir(tmp_path: Path):
    """ASFAM saves into the output directory so the project sits with the
    companion files it reads (alignment.eic, spotmap.json, _work/) and the whole
    folder can be archived as one unit."""
    out = tmp_path / "result_0722"
    out.mkdir()
    p = auto_save_path(tmp_path / "raw" / "sample.mzML", app_prefix="ASFAM",
                       extension="asfam", directory=out,
                       timestamp=datetime(2026, 7, 27, 9, 0, 0))
    assert p == out / "ASFAMProject_20260727_090000.asfam"


def test_directory_none_still_means_beside_the_data(tmp_path: Path):
    p = auto_save_path(tmp_path / "sample.mzML", app_prefix="ASFAM",
                       extension="asfam", directory=None,
                       timestamp=datetime(2026, 7, 27, 9, 0, 0))
    assert p.parent == tmp_path


def test_first_existing_picks_first(tmp_path: Path):
    a = tmp_path / "a.mzML"; a.write_text("")
    b = tmp_path / "b.mzML"
    assert first_existing([b, a]) == a


def test_first_existing_returns_none_when_no_match(tmp_path: Path):
    assert first_existing([tmp_path / "missing1", tmp_path / "missing2"]) is None
