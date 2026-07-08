"""Single-spectrum export to MSP / MGF (one feature at a time).

The MSP and MGF emitters take the same in-memory representation:
``peaks`` is a sequence of ``(mz, intensity)`` pairs, ``meta`` is a flat
``dict`` of header fields. Field naming follows the de-facto convention
(NIST EI MSP, MGF mascot-style) so produced files load in MS-DIAL,
MZmine, sirius and the like without further translation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence


PeakLike = tuple[float, float]


# ----------------------------------------------------------------------
# MSP (NIST EI library text format)
# ----------------------------------------------------------------------

# Fixed write order for the most common MSP header keys; unknown keys
# follow alphabetically.
_MSP_KEY_ORDER: tuple[str, ...] = (
    "Name", "Synon", "Formula", "MW", "ExactMass", "CAS#", "InChIKey",
    "SMILES", "Comment", "RetentionTime", "RetentionIndex",
    "PrecursorMZ", "PrecursorType", "IonMode", "Spectrum_type",
    "Instrument_type", "Instrument", "Collision_energy",
)


def format_msp(peaks: Sequence[PeakLike], meta: Mapping[str, object]) -> str:
    """Format a single spectrum as an MSP record (no trailing blank line)."""
    headers = _ordered_meta(meta)
    lines: list[str] = [f"{k}: {v}" for k, v in headers]
    lines.append(f"Num Peaks: {len(peaks)}")
    for mz, inten in peaks:
        lines.append(f"{float(mz):.4f} {float(inten):.4f}")
    return "\n".join(lines) + "\n"


def write_spectrum_msp(
    path: str | Path,
    peaks: Sequence[PeakLike],
    meta: Mapping[str, object],
) -> Path:
    """Write a single feature's spectrum to an .msp file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(format_msp(peaks, meta), encoding="utf-8")
    return out


# ----------------------------------------------------------------------
# MGF (Mascot Generic Format)
# ----------------------------------------------------------------------

# Canonical MGF header lines (uppercase, key=value, before peaks).
_MGF_KEY_MAP: dict[str, str] = {
    "Name": "TITLE",
    "PrecursorMZ": "PEPMASS",
    "PrecursorType": "ADDUCT",
    "IonMode": "IONMODE",
    "RetentionTime": "RTINSECONDS",
    "Charge": "CHARGE",
    "Formula": "FORMULA",
    "InChIKey": "INCHIKEY",
    "SMILES": "SMILES",
}


def format_mgf(peaks: Sequence[PeakLike], meta: Mapping[str, object]) -> str:
    """Format a single spectrum as an MGF block."""
    lines: list[str] = ["BEGIN IONS"]
    seen_keys: set[str] = set()
    # Emit keys in canonical order first
    for src_key, mgf_key in _MGF_KEY_MAP.items():
        if src_key in meta and meta[src_key] not in (None, ""):
            value = meta[src_key]
            # MGF RT is in seconds; if caller passed a minute value as
            # "RetentionTime", we trust it and pass through. Callers who
            # need conversion should multiply before calling.
            lines.append(f"{mgf_key}={_mgf_value(value)}")
            seen_keys.add(src_key)
    # Pass through anything else verbatim
    for k, v in meta.items():
        if k in seen_keys or v in (None, ""):
            continue
        lines.append(f"{str(k).upper()}={_mgf_value(v)}")
    for mz, inten in peaks:
        lines.append(f"{float(mz):.4f} {float(inten):.4f}")
    lines.append("END IONS")
    return "\n".join(lines) + "\n"


def write_spectrum_mgf(
    path: str | Path,
    peaks: Sequence[PeakLike],
    meta: Mapping[str, object],
) -> Path:
    """Write a single feature's spectrum to an .mgf file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(format_mgf(peaks, meta), encoding="utf-8")
    return out


def write_spectrum(
    path: str | Path,
    peaks: Sequence[PeakLike],
    meta: Mapping[str, object],
    *,
    fmt: str | None = None,
) -> Path:
    """Dispatch to MSP or MGF based on file suffix (or explicit ``fmt``)."""
    suffix = (fmt or Path(path).suffix.lstrip(".") or "").lower()
    if suffix == "msp":
        return write_spectrum_msp(path, peaks, meta)
    if suffix == "mgf":
        return write_spectrum_mgf(path, peaks, meta)
    raise ValueError(
        f"Unsupported spectrum format {suffix!r}; expected 'msp' or 'mgf'."
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _ordered_meta(meta: Mapping[str, object]) -> Iterable[tuple[str, object]]:
    """Yield ``(key, value)`` pairs in the canonical MSP order, then the rest."""
    seen: set[str] = set()
    for key in _MSP_KEY_ORDER:
        if key in meta and meta[key] not in (None, ""):
            yield key, meta[key]
            seen.add(key)
    for key in sorted(meta):
        if key in seen:
            continue
        if meta[key] in (None, ""):
            continue
        yield key, meta[key]


def _mgf_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
