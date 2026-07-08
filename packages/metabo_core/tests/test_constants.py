"""Regression tests for shared constants."""
import math

from metabo_core.constants.mass import (
    PROTON_MASS,
    ADDUCTS_POSITIVE,
    ADDUCTS_NEGATIVE,
    mz_from_neutral,
    neutral_from_mz,
)


def _adduct(name: str) -> dict:
    for table in (ADDUCTS_POSITIVE, ADDUCTS_NEGATIVE):
        for entry in table:
            if entry["name"] == name:
                return entry
    raise KeyError(name)


def test_proton_mass_constant():
    assert math.isclose(PROTON_MASS, 1.00727646677, rel_tol=0, abs_tol=1e-12)


def test_mz_round_trip_for_protonated_form():
    neutral = 180.0634
    adduct = _adduct("[M+H]+")
    mz = mz_from_neutral(neutral, adduct)
    assert math.isclose(mz, neutral + PROTON_MASS, abs_tol=1e-9)
    assert math.isclose(neutral_from_mz(mz, adduct), neutral, abs_tol=1e-9)


def test_dimer_adduct_round_trip():
    neutral = 250.0
    adduct = _adduct("[2M+H]+")
    mz = mz_from_neutral(neutral, adduct)
    assert math.isclose(neutral_from_mz(mz, adduct), neutral, abs_tol=1e-9)
