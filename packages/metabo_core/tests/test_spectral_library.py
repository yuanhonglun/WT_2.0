"""Regression tests for shared spectral library readers."""
import numpy as np

from metabo_core.io.spectral_library import read_msp, read_mgf


MSP_TEXT = (
    "NAME: Test Compound\n"
    "PRECURSORMZ: 181.0707\n"
    "FORMULA: C9H12O4\n"
    "ADDUCT: [M+H]+\n"
    "Num Peaks: 2\n"
    "100.0 500\n"
    "150.0 250\n"
    "\n"
)


MGF_TEXT = (
    "BEGIN IONS\n"
    "TITLE=Demo MGF\n"
    "PEPMASS=181.0707\n"
    "RTINSECONDS=120.5\n"
    "100.0 500\n"
    "150.0 250\n"
    "END IONS\n"
)


def test_read_msp_parses_metadata_and_peaks(tmp_path):
    path = tmp_path / "demo.msp"
    path.write_text(MSP_TEXT, encoding="utf-8")
    spectra = read_msp(str(path))
    assert len(spectra) == 1
    spec = spectra[0]
    assert spec["mz"] == [100.0, 150.0]
    assert spec["intensity"] == [500.0, 250.0]
    assert spec["metadata"]["name"] == "Test Compound"
    assert spec["metadata"]["precursor_mz"] == 181.0707
    assert spec["metadata"]["formula"] == "C9H12O4"
    assert spec["metadata"]["adduct"] == "[M+H]+"


def test_read_msp_parses_multi_peak_per_line_nist_gcms_format(tmp_path):
    """NIST GC-MS MSP exports often pack multiple peaks per line, separated
    by ';'. The reader must handle that format too.
    """
    msp_text = (
        "Name: Hexanal\n"
        "Formula: C6H12O\n"
        "Num Peaks: 5\n"
        "14 3; 15 21; 17 2;\n"
        "39 200; 41 690;\n"
        "\n"
    )
    path = tmp_path / "demo_gcms.msp"
    path.write_text(msp_text, encoding="utf-8")
    spectra = read_msp(str(path))
    assert len(spectra) == 1
    spec = spectra[0]
    assert spec["mz"] == [14.0, 15.0, 17.0, 39.0, 41.0]
    assert spec["intensity"] == [3.0, 21.0, 2.0, 200.0, 690.0]


def test_read_msp_lean_prunes_metadata_and_returns_arrays(tmp_path):
    """Lean mode (used by the LC-MS annotation loader) drops metadata
    fields the matcher never reads and stores peaks as float64 numpy
    arrays — cutting the in-memory footprint of a multi-GB library by
    ~4x. Peak values are preserved exactly (float64), so matching and
    display are unchanged.
    """
    text = (
        "NAME: A\nPRECURSORMZ: 195.09\nFORMULA: C8H10N4O2\nADDUCT: [M+H]+\n"
        "RETENTIONTIME: 5.0\nINCHIKEY: ABC\nSMILES: CCO\nCOMMENT: a big bulky comment\n"
        "Num Peaks: 2\n110.07 0.3\n195.09 1.0\n\n"
    )
    path = tmp_path / "fat.msp"
    path.write_text(text, encoding="utf-8")
    keep = {"name", "precursor_mz", "formula", "adduct", "rt"}

    spectra = read_msp(str(path), keep_metadata=keep, as_arrays=True)
    assert len(spectra) == 1
    s = spectra[0]
    # Metadata pruned to exactly the keep-set (comment / inchikey / smiles gone).
    assert set(s["metadata"].keys()) == keep
    assert "comment" not in s["metadata"]
    # Peaks are float64 arrays with identical values.
    assert isinstance(s["mz"], np.ndarray)
    assert isinstance(s["intensity"], np.ndarray)
    assert s["mz"].dtype == np.float64
    assert s["mz"].tolist() == [110.07, 195.09]
    assert s["intensity"].tolist() == [0.3, 1.0]
    assert s["metadata"]["precursor_mz"] == 195.09


def test_read_msp_default_keeps_all_metadata_and_lists(tmp_path):
    """Default (no lean args) is byte-for-byte the legacy behavior:
    Python lists for peaks and every metadata field retained."""
    text = "NAME: A\nPRECURSORMZ: 195.09\nCOMMENT: keepme\nNum Peaks: 1\n110.07 0.3\n\n"
    path = tmp_path / "default.msp"
    path.write_text(text, encoding="utf-8")
    s = read_msp(str(path))[0]
    assert s["mz"] == [110.07]            # list, not ndarray
    assert s["metadata"]["comment"] == "keepme"


def test_read_mgf_parses_metadata_and_peaks(tmp_path):
    path = tmp_path / "demo.mgf"
    path.write_text(MGF_TEXT, encoding="utf-8")
    spectra = read_mgf(str(path))
    assert len(spectra) == 1
    spec = spectra[0]
    assert spec["mz"] == [100.0, 150.0]
    assert spec["intensity"] == [500.0, 250.0]
    assert spec["metadata"]["name"] == "Demo MGF"
    assert spec["metadata"]["precursor_mz"] == 181.0707
    assert spec["metadata"]["rt"] == 120.5
