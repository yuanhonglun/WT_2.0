"""Spectral library I/O: read/write MGF and MSP formats."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _finalize_spectrum(
    current: dict,
    keep_metadata: Optional[set[str]],
    as_arrays: bool,
) -> dict:
    """Apply optional lean transforms to a just-parsed spectrum.

    ``keep_metadata`` prunes the metadata dict to only those keys (the
    LC-MS annotator reads just ``name`` / ``precursor_mz`` / ``formula`` /
    ``adduct`` / ``rt``; the on-disk library also carries ``inchi`` /
    ``smiles`` / ``comment`` / ``splash`` / ... that nothing downstream
    uses). ``as_arrays`` stores peaks as ``float64`` numpy arrays instead
    of Python lists. Together these cut a multi-GB library's in-memory
    footprint by ~4x. Pruning happens here, at finalize, so the bulky
    fields live for only one spectrum at a time — never accumulated.
    ``float64`` preserves the exact peak values, so matching is unchanged.
    """
    if keep_metadata is not None:
        meta = current["metadata"]
        current["metadata"] = {k: v for k, v in meta.items() if k in keep_metadata}
    if as_arrays:
        current["mz"] = np.asarray(current["mz"], dtype=np.float64)
        current["intensity"] = np.asarray(current["intensity"], dtype=np.float64)
    return current


def read_msp(
    filepath: str,
    keep_metadata: Optional[set[str]] = None,
    as_arrays: bool = False,
) -> list[dict]:
    """Read MSP/NIST format spectral library.

    Handles various MSP formats including MassBank, NIST, MS-DIAL exports.
    Supports case-insensitive field names and Windows line endings.

    ``keep_metadata`` / ``as_arrays`` enable the lean representation used
    by the LC-MS annotation loader to bound memory on multi-GB libraries
    (see :func:`_finalize_spectrum`). Both default to the legacy behavior
    (full metadata, Python-list peaks).
    """
    spectra = []
    current = None
    reading_peaks = False

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip().replace("\r", "")

            if not line:
                if current and current["mz"]:
                    spectra.append(_finalize_spectrum(current, keep_metadata, as_arrays))
                current = None
                reading_peaks = False
                continue

            if reading_peaks:
                # Support two MSP peak formats:
                #   "73 17"                        (one peak per line, common NIST/MS-DIAL)
                #   "14 3; 15 21; 17 2;"           (multi-peak per line, NIST GC-MS)
                # Splitting on ';' covers both: a one-peak line yields a single
                # chunk equal to the line itself. Plan B and Plan C independently
                # arrived at this fix; merged form chosen here for clarity.
                for chunk in line.split(";"):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    parts = chunk.split()
                    if len(parts) >= 2:
                        try:
                            mz_val = float(parts[0])
                            int_val = float(parts[1])
                            current["mz"].append(mz_val)
                            current["intensity"].append(int_val)
                        except ValueError:
                            pass
                continue

            if ":" in line:
                key, val = line.split(":", 1)
                key_upper = key.strip().upper()
                val = val.strip()

                if current is None:
                    current = {"mz": [], "intensity": [], "metadata": {}}

                if key_upper == "NUM PEAKS":
                    try:
                        int(val)
                    except ValueError:
                        pass
                    reading_peaks = True
                elif key_upper in ("PRECURSORMZ", "PRECURSOR_MZ"):
                    try:
                        current["metadata"]["precursor_mz"] = float(val)
                    except ValueError:
                        pass
                elif key_upper == "NAME":
                    current["metadata"]["name"] = val
                elif key_upper == "FORMULA":
                    current["metadata"]["formula"] = val
                elif key_upper in ("RETENTIONTIME", "RT"):
                    try:
                        current["metadata"]["rt"] = float(val)
                    except ValueError:
                        pass
                elif key_upper in ("PRECURSOR_TYPE", "PRECURSORTYPE", "ADDUCT"):
                    current["metadata"]["adduct"] = val
                elif key_upper == "INCHIKEY":
                    current["metadata"]["inchikey"] = val
                elif key_upper == "SMILES":
                    current["metadata"]["smiles"] = val
                elif key_upper in ("COMMENT", "COMMENTS"):
                    current["metadata"]["comment"] = val
                elif key_upper in ("COLLISION_ENERGY", "COLLISIONENERGY"):
                    current["metadata"]["collision_energy"] = val
                else:
                    current["metadata"][key.strip().lower()] = val

    # Don't forget the last entry
    if current and current["mz"]:
        spectra.append(_finalize_spectrum(current, keep_metadata, as_arrays))

    logger.info("Read %d spectra from %s", len(spectra), Path(filepath).name)
    return spectra


def read_mgf(
    filepath: str,
    keep_metadata: Optional[set[str]] = None,
    as_arrays: bool = False,
) -> list[dict]:
    """Read MGF spectral library into list of dicts.

    ``keep_metadata`` / ``as_arrays`` mirror :func:`read_msp` for the lean
    annotation-loader path; both default to the legacy behavior.
    """
    spectra = []
    current = None

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip().replace("\r", "")
            if line == "BEGIN IONS":
                current = {"mz": [], "intensity": [], "metadata": {}}
            elif line == "END IONS":
                if current and current["mz"]:
                    spectra.append(_finalize_spectrum(current, keep_metadata, as_arrays))
                current = None
            elif current is not None:
                if "=" in line:
                    key, val = line.split("=", 1)
                    key_upper = key.strip().upper()
                    val = val.strip()
                    if key_upper == "PEPMASS":
                        try:
                            current["metadata"]["precursor_mz"] = float(val.split()[0])
                        except ValueError:
                            pass
                    elif key_upper == "CHARGE":
                        current["metadata"]["charge"] = val
                    elif key_upper in ("NAME", "TITLE"):
                        current["metadata"]["name"] = val
                    elif key_upper == "FORMULA":
                        current["metadata"]["formula"] = val
                    elif key_upper in ("RTINSECONDS", "RETENTIONTIME"):
                        try:
                            current["metadata"]["rt"] = float(val)
                        except ValueError:
                            pass
                    else:
                        current["metadata"][key.strip().lower()] = val
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            current["mz"].append(float(parts[0]))
                            current["intensity"].append(float(parts[1]))
                        except ValueError:
                            pass

    logger.info("Read %d spectra from %s", len(spectra), Path(filepath).name)
    return spectra
