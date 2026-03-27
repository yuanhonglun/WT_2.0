"""Mass spectrometry utility functions: MW calculation, isotope classification, peak overlap."""
from __future__ import annotations

import math
from typing import Optional

from asfam.constants import (
    PROTON_MASS, ISOTOPE_DELTAS, C13_DELTA, MAX_ISOTOPE_STEP,
    ADDUCTS_POSITIVE, ADDUCTS_NEGATIVE,
    mz_from_neutral, neutral_from_mz,
)


# ---------------------------------------------------------------------------
# Neutral mass calculation from precursor m/z + adduct
# ---------------------------------------------------------------------------

# Quick-lookup adduct tables: name -> dict
_ADDUCT_BY_NAME = {}
for _a in ADDUCTS_POSITIVE + ADDUCTS_NEGATIVE:
    _ADDUCT_BY_NAME[_a["name"]] = _a


def neutral_mass_from_adduct(precursor_mz: float, adduct_name: str) -> Optional[float]:
    """Calculate neutral monoisotopic mass given precursor m/z and adduct name.

    Returns None if adduct_name is unknown.
    """
    adduct = _ADDUCT_BY_NAME.get(adduct_name)
    if adduct is None:
        return None
    return neutral_from_mz(precursor_mz, adduct)


# ---------------------------------------------------------------------------
# Isotope gap classification
# ---------------------------------------------------------------------------

def classify_isotope_gap(
    delta_mz: float,
    classic_tol: float = 0.01,
    relaxed_tol: float = 0.02,
    max_step: int = MAX_ISOTOPE_STEP,
) -> Optional[str]:
    """Classify an m/z difference as an isotope gap type.

    Returns:
        "classic" - matches a known element isotope delta (C13, N15, S34, O18, Cl37, Br81)
        "relaxed" - matches a near-integer step (1, 2, 3, or 4 Da)
        None      - not an isotope gap
    """
    abs_delta = abs(delta_mz)

    # Check classic isotope deltas
    for name, iso_delta in ISOTOPE_DELTAS.items():
        if name == "C13":
            # C13 can be n * delta for n = 1..max_step
            for n in range(1, max_step + 1):
                if abs(abs_delta - n * iso_delta) <= classic_tol:
                    return "classic"
        else:
            if abs(abs_delta - iso_delta) <= classic_tol:
                return "classic"

    # Check relaxed near-integer steps
    for n in range(1, max_step + 1):
        if abs(abs_delta - float(n)) <= relaxed_tol:
            return "relaxed"

    return None


# ---------------------------------------------------------------------------
# Peak overlap ratio
# ---------------------------------------------------------------------------

def peak_overlap_ratio(
    rt_left_a: float, rt_right_a: float,
    rt_left_b: float, rt_right_b: float,
) -> float:
    """Compute chromatographic peak overlap ratio.

    Returns overlap_length / min(width_a, width_b), clamped to [0, 1].
    """
    overlap_start = max(rt_left_a, rt_left_b)
    overlap_end = min(rt_right_a, rt_right_b)
    overlap = max(0.0, overlap_end - overlap_start)

    width_a = rt_right_a - rt_left_a
    width_b = rt_right_b - rt_left_b
    min_width = min(width_a, width_b)

    if min_width <= 0:
        return 0.0
    return min(overlap / min_width, 1.0)


# ---------------------------------------------------------------------------
# Adduct pair checking
# ---------------------------------------------------------------------------

def check_adduct_pair(
    mz_a: float,
    mz_b: float,
    ionization_mode: str = "positive",
    mw_tolerance: float = 0.02,
) -> Optional[tuple[str, str]]:
    """Check if two m/z values could be different adducts of the same molecule.

    Returns (adduct_name_a, adduct_name_b) if a valid pair is found, else None.
    """
    adducts = ADDUCTS_POSITIVE if ionization_mode == "positive" else ADDUCTS_NEGATIVE

    for i, adduct_a in enumerate(adducts):
        mw_a = neutral_from_mz(mz_a, adduct_a)
        for j, adduct_b in enumerate(adducts):
            if i == j:
                continue
            mw_b = neutral_from_mz(mz_b, adduct_b)
            if abs(mw_a - mw_b) <= mw_tolerance:
                return (adduct_a["name"], adduct_b["name"])
    return None


# ---------------------------------------------------------------------------
# Intensity ratio validation for isotopes
# ---------------------------------------------------------------------------

def max_c13_m1_ratio(precursor_mz: float) -> float:
    """Estimate maximum plausible M+1/M+0 intensity ratio for C13.

    Based on the heuristic: max carbons ~ mz / 12, and
    M+1/M+0 ~ n_carbons * 0.011.
    Adds a safety margin of 1.5x.
    """
    estimated_carbons = precursor_mz / 12.0
    theoretical_ratio = estimated_carbons * 0.011
    return theoretical_ratio * 1.5  # safety margin
