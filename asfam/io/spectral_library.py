"""Spectral library I/O: read/write MGF and MSP formats."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_msp(filepath: str) -> list[dict]:
    """Read MSP/NIST format spectral library.

    Handles various MSP formats including MassBank, NIST, MS-DIAL exports.
    Supports case-insensitive field names and Windows line endings.
    """
    spectra = []
    current = None
    reading_peaks = False

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip().replace("\r", "")

            if not line:
                if current and current["mz"]:
                    spectra.append(current)
                current = None
                reading_peaks = False
                continue

            if reading_peaks:
                parts = line.split()
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
        spectra.append(current)

    logger.info("Read %d spectra from %s", len(spectra), Path(filepath).name)
    return spectra


def read_mgf(filepath: str) -> list[dict]:
    """Read MGF spectral library into list of dicts."""
    spectra = []
    current = None

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip().replace("\r", "")
            if line == "BEGIN IONS":
                current = {"mz": [], "intensity": [], "metadata": {}}
            elif line == "END IONS":
                if current and current["mz"]:
                    spectra.append(current)
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
