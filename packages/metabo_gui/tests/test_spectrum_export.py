"""MSP / MGF round-trip + dispatcher behavior."""
from __future__ import annotations

from pathlib import Path

import pytest

from metabo_gui.spectrum_export import (
    format_mgf,
    format_msp,
    write_spectrum,
    write_spectrum_mgf,
    write_spectrum_msp,
)


PEAKS = [(73.0, 1000.0), (147.05, 850.5), (207.1, 320.0)]
META = {
    "Name": "Test Compound",
    "Formula": "C5H10O",
    "PrecursorMZ": 88.0762,
    "PrecursorType": "[M+H]+",
    "RetentionTime": 5.42,
    "InChIKey": "ABCDEF1234567-LMNOPQ-N",
}


def test_msp_round_trip(tmp_path: Path):
    path = write_spectrum_msp(tmp_path / "out.msp", PEAKS, META)
    text = path.read_text(encoding="utf-8")
    assert "Name: Test Compound" in text
    assert "Formula: C5H10O" in text
    assert "Num Peaks: 3" in text
    assert "73.0000 1000.0000" in text
    assert "147.0500 850.5000" in text
    # Order check: Name comes before Formula (canonical order)
    assert text.index("Name:") < text.index("Formula:")


def test_mgf_round_trip(tmp_path: Path):
    path = write_spectrum_mgf(tmp_path / "out.mgf", PEAKS, META)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("BEGIN IONS\n")
    assert text.rstrip().endswith("END IONS")
    assert "TITLE=Test Compound" in text
    assert "PEPMASS=88.0762" in text
    assert "ADDUCT=[M+H]+" in text
    assert "RTINSECONDS=5.42" in text
    assert "73.0000 1000.0000" in text


def test_dispatcher_picks_format_from_suffix(tmp_path: Path):
    p_msp = write_spectrum(tmp_path / "x.msp", PEAKS, META)
    p_mgf = write_spectrum(tmp_path / "x.mgf", PEAKS, META)
    assert "Num Peaks:" in p_msp.read_text()
    assert "BEGIN IONS" in p_mgf.read_text()


def test_dispatcher_explicit_fmt_overrides_suffix(tmp_path: Path):
    p = write_spectrum(tmp_path / "noext", PEAKS, META, fmt="mgf")
    assert "BEGIN IONS" in p.read_text()


def test_dispatcher_rejects_unknown_format(tmp_path: Path):
    with pytest.raises(ValueError):
        write_spectrum(tmp_path / "x.txt", PEAKS, META)


def test_meta_with_none_values_skipped():
    meta = {"Name": "X", "Formula": None, "InChIKey": ""}
    text = format_msp(PEAKS, meta)
    # Empty / None entries should not be emitted at all
    assert "Formula:" not in text
    assert "InChIKey:" not in text


def test_unknown_meta_keys_pass_through():
    meta = {"Name": "X", "ExperimentTag": "lab1"}
    msp = format_msp(PEAKS, meta)
    mgf = format_mgf(PEAKS, meta)
    assert "ExperimentTag: lab1" in msp
    assert "EXPERIMENTTAG=lab1" in mgf
