"""Regression tests for shared mass utilities."""
import math

from metabo_core.algorithms.mass_utils import (
    classify_isotope_gap,
    peak_overlap_ratio,
    check_adduct_pair,
    neutral_mass_from_adduct,
)


def test_classify_isotope_gap_classic_carbon_steps():
    assert classify_isotope_gap(1.003355) == "classic"
    assert classify_isotope_gap(2.0067) == "classic"


def test_classify_isotope_gap_near_integer_relaxed():
    assert classify_isotope_gap(1.015) == "relaxed"
    assert classify_isotope_gap(3.0) == "relaxed"


def test_classify_isotope_gap_returns_none_for_garbage():
    assert classify_isotope_gap(0.4) is None
    assert classify_isotope_gap(7.0) is None


def test_peak_overlap_ratio_full_overlap():
    assert math.isclose(peak_overlap_ratio(1.0, 2.0, 1.2, 1.8), 1.0)


def test_peak_overlap_ratio_no_overlap():
    assert peak_overlap_ratio(1.0, 1.5, 2.0, 2.5) == 0.0


def test_check_adduct_pair_finds_protonated_vs_sodiated():
    pair = check_adduct_pair(181.0707, 203.0526, ionization_mode="positive", mw_tolerance=0.02)
    assert pair is not None
    assert pair[0] == "[M+H]+"
    assert pair[1] == "[M+Na]+"


def test_neutral_mass_from_known_adduct():
    mw = neutral_mass_from_adduct(181.0707, "[M+H]+")
    assert mw is not None
    assert math.isclose(mw, 181.0707 - 1.00727646677, abs_tol=1e-9)


def test_neutral_mass_from_unknown_adduct_returns_none():
    assert neutral_mass_from_adduct(181.0707, "[M+Banana]+") is None
