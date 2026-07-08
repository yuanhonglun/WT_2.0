"""Memory-footprint regression: the ASFAM library preload must be *lean*.

The orchestrator pre-loads the spectral library once via
``stage2b_inference._load_library`` and shares it between stage 2.5
(MS2-only inference) and stage 6.5 (annotation). For the production LC-MS
library (``lib/lcms/pos.msp`` ~6.7GB, ~6.4M spectra) the *full*
representation — every metadata field (inchi / smiles / comment / ...)
plus Python-list peaks — is ~3x larger than the lean one and exhausts RAM
on re-annotation / inference-on runs. These tests pin ``_load_library``
to the lean representation (pruned metadata + float64 numpy peaks) and
confirm the stage 2.5 matcher tolerates numpy peak arrays, so the
optimization cannot silently regress.
"""
from types import SimpleNamespace

import numpy as np

from asfam.config import ProcessingConfig
from asfam.pipeline.stage2b_inference import _library_match, _load_library


FAT_MSP = (
    "NAME: A\nPRECURSORMZ: 195.09\nFORMULA: C8H10N4O2\nADDUCT: [M+H]+\n"
    "RETENTIONTIME: 5.0\nINCHIKEY: ABC\nSMILES: CCO\n"
    "COMMENT: a big bulky comment that nothing downstream reads\n"
    "Num Peaks: 2\n110.07 0.3\n195.09 1.0\n\n"
)


def test_load_library_is_lean(tmp_path):
    """_load_library returns float64 numpy peaks and prunes metadata the
    matchers never read — the multi-GB footprint reduction."""
    path = tmp_path / "fat.msp"
    path.write_text(FAT_MSP, encoding="utf-8")

    spectra = _load_library(str(path))
    assert spectra is not None and len(spectra) == 1
    spec = spectra[0]

    # Peaks come back as float64 numpy arrays (not Python lists).
    assert isinstance(spec["mz"], np.ndarray)
    assert isinstance(spec["intensity"], np.ndarray)
    assert spec["mz"].dtype == np.float64
    assert spec["mz"].tolist() == [110.07, 195.09]
    assert spec["intensity"].tolist() == [0.3, 1.0]

    # Bulky metadata the matchers never read is pruned.
    assert "comment" not in spec["metadata"]
    assert "inchikey" not in spec["metadata"]
    assert "smiles" not in spec["metadata"]
    # Fields the matchers DO read are preserved exactly.
    assert spec["metadata"]["name"] == "A"
    assert spec["metadata"]["precursor_mz"] == 195.09
    assert spec["metadata"]["formula"] == "C8H10N4O2"

    # The integer-m/z key stage 2.5 matching relies on is still set.
    assert spec["_mz_int"] == 195


def test_library_match_handles_numpy_peaks():
    """With the lean loader, candidate peaks are numpy arrays. The stage
    2.5 matcher must not do ``if not <ndarray>`` (raises ValueError for a
    multi-element array)."""
    library = [
        {
            "mz": np.array([110.07, 195.09], dtype=np.float64),
            "intensity": np.array([0.3, 1.0], dtype=np.float64),
            "metadata": {"precursor_mz": 195.09, "name": "A", "formula": "X"},
            "_mz_int": 195,
        }
    ]
    feat = SimpleNamespace(
        precursor_mz_nominal=195,
        ms2_mz=np.array([110.07, 195.09], dtype=np.float64),
        ms2_intensity=np.array([0.3, 1.0], dtype=np.float64),
    )
    cfg = ProcessingConfig()
    cfg.library_mz_inference_threshold = 0.0  # accept the self-match
    cfg.matchms_min_matched_peaks = 1

    # Must not raise on the numpy-array candidate, and should self-match.
    match = _library_match(feat, library, cfg)
    assert match is not None
    assert match["precursor_mz"] == 195.09
