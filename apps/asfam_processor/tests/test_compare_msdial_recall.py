"""Unit tests for the new recall-mode helpers added in Task 0.1.

Covers the pure, data-independent helpers:
  * dynamic_mz_tol  -- the growing-tolerance formula
  * recall_match    -- non-greedy any-match (not greedy 1:1 like match_features)
  * _check_sentinel -- sentinel-peak lookup
  * reverse_recall_match -- METRA features with no MS-DIAL partner
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.compare_asfam_msdial import (
    _check_sentinel,
    _is_dup,
    _is_recall_annotated,
    _subset_recall,
    dynamic_mz_tol,
    mz_bin_recall,
    parse_msdial_csv,
    recall_match,
    reverse_recall_match,
)


# --------------------------------------------------------------------------- #
# dynamic_mz_tol
# --------------------------------------------------------------------------- #

def test_dynamic_mz_tol_floor_dominates_at_low_mz():
    """At low m/z, tolerance should be clamped to the floor (0.006 Da)."""
    # 200 Da * 25e-6 = 0.005 < floor=0.006
    assert dynamic_mz_tol(200.0) == 0.006


def test_dynamic_mz_tol_ppm_dominates_at_high_mz():
    """At high m/z, ppm term exceeds floor."""
    # 300 Da * 25e-6 = 0.0075 > floor=0.006
    tol = dynamic_mz_tol(300.0)
    assert abs(tol - 300.0 * 25e-6) < 1e-12


def test_dynamic_mz_tol_crossover():
    """Crossover is at mz = floor / (ppm * 1e-6) = 0.006 / 25e-6 = 240 Da."""
    tol_at_240 = dynamic_mz_tol(240.0)
    assert abs(tol_at_240 - 0.006) < 1e-12   # exactly at the crossover


def test_dynamic_mz_tol_custom_params():
    tol = dynamic_mz_tol(1000.0, ppm=10.0, floor=0.005)
    # 1000 * 10e-6 = 0.01 > floor=0.005
    assert abs(tol - 0.01) < 1e-12


# --------------------------------------------------------------------------- #
# recall_match  (non-greedy)
# --------------------------------------------------------------------------- #

def test_recall_match_basic_hit():
    msdial = [{"mz": 300.0, "rt": 2.0}]
    ours = [{"mz": 300.005, "rt": 2.05}]   # within 25ppm and 0.2 min
    assert recall_match(msdial, ours) == [True]


def test_recall_match_miss_by_mz():
    msdial = [{"mz": 300.0, "rt": 2.0}]
    ours = [{"mz": 300.02, "rt": 2.0}]     # 0.02 Da > dynamic tol ~0.0075
    assert recall_match(msdial, ours) == [False]


def test_recall_match_miss_by_rt():
    msdial = [{"mz": 300.0, "rt": 2.0}]
    ours = [{"mz": 300.005, "rt": 2.25}]   # 0.25 min > rt_tol=0.2
    assert recall_match(msdial, ours) == [False]


def test_recall_match_non_greedy_multiple_metra_one_msdial():
    """Both METRA features can claim the same MS-DIAL peak (non-greedy)."""
    msdial = [{"mz": 300.0, "rt": 2.0}]
    ours = [
        {"mz": 300.001, "rt": 2.0},
        {"mz": 300.003, "rt": 2.0},
    ]
    # Non-greedy: MS-DIAL peak counts as matched (both ours are valid candidates)
    assert recall_match(msdial, ours) == [True]


def test_recall_match_one_hit_one_miss():
    msdial = [
        {"mz": 300.0, "rt": 2.0},
        {"mz": 500.0, "rt": 5.0},
    ]
    ours = [{"mz": 300.003, "rt": 2.05}]   # only first MS-DIAL peak is matched
    result = recall_match(msdial, ours)
    assert result[0] is True
    assert result[1] is False


def test_recall_match_empty_ours():
    msdial = [{"mz": 300.0, "rt": 2.0}]
    assert recall_match(msdial, []) == [False]


def test_recall_match_empty_msdial():
    assert recall_match([], [{"mz": 300.0, "rt": 2.0}]) == []


# --------------------------------------------------------------------------- #
# reverse_recall_match
# --------------------------------------------------------------------------- #

def test_reverse_recall_match_basic():
    msdial = [{"mz": 300.0, "rt": 2.0}]
    ours = [
        {"mz": 300.003, "rt": 2.0},   # matched
        {"mz": 400.0, "rt": 3.0},     # unmatched (no MS-DIAL near it)
    ]
    result = reverse_recall_match(ours, msdial)
    assert result[0] is True
    assert result[1] is False


def test_reverse_recall_match_empty():
    assert reverse_recall_match([], [{"mz": 300.0, "rt": 2.0}]) == []
    assert reverse_recall_match([{"mz": 300.0, "rt": 2.0}], []) == [False]


# --------------------------------------------------------------------------- #
# _check_sentinel
# --------------------------------------------------------------------------- #

def test_check_sentinel_found_matched():
    msdial = [{"mz": 801.20964, "rt": 3.919, "height": 1_700_000.0}]
    matched = [True]
    assert _check_sentinel(msdial, matched) is True


def test_check_sentinel_found_not_matched():
    msdial = [{"mz": 801.20964, "rt": 3.919, "height": 1_700_000.0}]
    matched = [False]
    assert _check_sentinel(msdial, matched) is False


def test_check_sentinel_not_in_msdial():
    msdial = [{"mz": 285.05, "rt": 1.0}]
    matched = [False]
    assert _check_sentinel(msdial, matched) is None


def test_check_sentinel_tolerance_boundary():
    """Sentinel uses ±0.01 Da on m/z and ±0.30 min on RT."""
    # Just inside boundary
    msdial = [{"mz": 801.20964 + 0.009, "rt": 3.919 + 0.29}]
    matched = [True]
    assert _check_sentinel(msdial, matched) is True
    # Just outside m/z boundary
    msdial2 = [{"mz": 801.20964 + 0.011, "rt": 3.919}]
    matched2 = [True]
    assert _check_sentinel(msdial2, matched2) is None


# --------------------------------------------------------------------------- #
# I1 guard: rt=None must not raise TypeError
# --------------------------------------------------------------------------- #

def test_recall_match_rt_none_returns_false_no_raise():
    """Peak with valid mz but rt=None must return False (unmatched), not TypeError."""
    msdial = [
        {"mz": 300.0, "rt": None},          # valid mz, rt is None -- the bug case
        {"mz": 400.0, "rt": 2.0},           # normal peak, should still work
    ]
    ours = [{"mz": 300.003, "rt": 1.5}]
    result = recall_match(msdial, ours)      # must NOT raise TypeError
    assert result[0] is False                # rt=None peak is unmatched
    assert result[1] is False                # no METRA feature near 400.0


def test_reverse_recall_match_rt_none_returns_false_no_raise():
    """METRA feature with valid mz but rt=None must return False, not TypeError."""
    msdial = [{"mz": 300.0, "rt": 2.0}]
    ours = [
        {"mz": 300.003, "rt": None},        # valid mz, rt is None -- the bug case
        {"mz": 300.003, "rt": 2.0},         # normal feature, should still match
    ]
    result = reverse_recall_match(ours, msdial)  # must NOT raise TypeError
    assert result[0] is False                # rt=None feature is unmatched
    assert result[1] is True                 # normal feature matches


# --------------------------------------------------------------------------- #
# M1: _is_recall_annotated unit tests
# --------------------------------------------------------------------------- #

def test_is_recall_annotated_unknown_false():
    assert _is_recall_annotated("Unknown") is False


def test_is_recall_annotated_unknown_compound_true():
    """'Unknown compound' is NOT in the exclusion set; it counts as annotated."""
    assert _is_recall_annotated("Unknown compound") is True


def test_is_recall_annotated_low_score_false():
    assert _is_recall_annotated("low score: foo") is False


def test_is_recall_annotated_no_result_false():
    assert _is_recall_annotated("no result") is False


def test_is_recall_annotated_empty_false():
    assert _is_recall_annotated("") is False


def test_is_recall_annotated_normal_name_true():
    assert _is_recall_annotated("Quercetin") is True


# --------------------------------------------------------------------------- #
# parse_msdial_csv  (integrated named-column CSV, e.g. ASFAM-3.csv)
# --------------------------------------------------------------------------- #

def test_parse_msdial_csv_maps_named_columns(tmp_path):
    p = tmp_path / "ASFAM-3.csv"
    p.write_text(
        "Peak ID,Name,RT (min),Precursor m/z,Height,S/N,Isotope,InChIKey,Adduct,MW\n"
        "1,Quercetin,3.50,301.0354,12345,42.1,0,ABCDEFGHIJ,[M+H]+,300.0\n"
        "2,Unknown,0.80,85.0393,500,3.0,0,,[M+H]+,84.03\n",
        encoding="utf-8",
    )
    rows = parse_msdial_csv(p)
    assert len(rows) == 2
    r0 = rows[0]
    assert abs(r0["mz"] - 301.0354) < 1e-9
    assert abs(r0["rt"] - 3.50) < 1e-9
    assert r0["height"] == 12345.0
    assert abs(r0["sn"] - 42.1) < 1e-9
    assert r0["isotope"] == 0
    assert r0["name"] == "Quercetin"
    assert r0["inchikey"] == "ABCDEFGHIJ"
    assert r0["annotated"] is True
    assert rows[1]["annotated"] is False     # "Unknown" is not confident


def test_parse_msdial_csv_column_order_independent(tmp_path):
    """Columns are matched by NAME, not position -> reordered cols still parse."""
    p = tmp_path / "reordered.csv"
    p.write_text(
        "Height,Precursor m/z,Name,Isotope,RT (min),S/N\n"
        "999,200.1,Foo,0,1.23,7.0\n",
        encoding="utf-8",
    )
    rows = parse_msdial_csv(p)
    assert len(rows) == 1
    assert abs(rows[0]["mz"] - 200.1) < 1e-9
    assert abs(rows[0]["rt"] - 1.23) < 1e-9
    assert rows[0]["height"] == 999.0


def test_parse_msdial_csv_picks_first_sn_when_duplicate(tmp_path):
    """The integration tool emits two S/N cols; pandas renames the 2nd 'S/N.1'.
    parse_msdial_csv must read the first 'S/N'."""
    p = tmp_path / "dupsn.csv"
    p.write_text(
        "Name,RT (min),Precursor m/z,Height,S/N,Isotope,S/N\n"
        "Bar,2.0,150.0,3000,11.5,0,99.9\n",
        encoding="utf-8",
    )
    rows = parse_msdial_csv(p)
    assert abs(rows[0]["sn"] - 11.5) < 1e-9


# --------------------------------------------------------------------------- #
# _subset_recall  /  mz_bin_recall  /  _is_dup
# --------------------------------------------------------------------------- #

def test_subset_recall_counts_all_mono_annotated():
    peaks = [
        {"mz": 100.0, "rt": 1.0, "isotope": 0, "name": "Quercetin"},  # mono, ann
        {"mz": 101.0, "rt": 1.0, "isotope": 1, "name": "Unknown"},    # iso, not ann
        {"mz": 102.0, "rt": 1.0, "isotope": 0, "name": "Unknown"},    # mono, not ann
    ]
    flags = [True, False, True]
    r = _subset_recall("X", peaks, flags)
    assert r["seg"] == "X"
    assert r["n_total"] == 3 and r["n_matched_all"] == 2
    assert abs(r["recall_all"] - 2 / 3) < 1e-9
    assert r["n_mono"] == 2 and r["n_matched_mono"] == 2     # both mono matched
    assert r["recall_mono"] == 1.0
    assert r["n_ann"] == 1 and r["n_matched_ann"] == 1       # only Quercetin
    assert r["recall_annotated"] == 1.0


def test_subset_recall_empty():
    r = _subset_recall("EMPTY", [], [])
    assert r["n_total"] == 0
    assert r["recall_all"] is None
    assert r["recall_mono"] is None
    assert r["recall_annotated"] is None


def test_mz_bin_recall_bins_and_labels():
    peaks = [
        {"mz": 85.0, "rt": 1.0, "isotope": 0, "name": "A"},    # bin 0-100
        {"mz": 150.0, "rt": 1.0, "isotope": 0, "name": "B"},   # bin 100-200
        {"mz": 199.9, "rt": 1.0, "isotope": 0, "name": "C"},   # bin 100-200
        {"mz": 250.0, "rt": 1.0, "isotope": 0, "name": "D"},   # bin 200-300
    ]
    flags = [False, True, True, False]
    bins = mz_bin_recall(peaks, flags, bin_width=100.0)
    assert [b["seg"] for b in bins] == ["0-100", "100-200", "200-300"]
    by = {b["seg"]: b for b in bins}
    assert by["0-100"]["n_total"] == 1 and by["0-100"]["n_matched_all"] == 0
    assert by["100-200"]["n_total"] == 2 and by["100-200"]["n_matched_all"] == 2
    assert by["100-200"]["recall_all"] == 1.0
    assert by["200-300"]["n_matched_all"] == 0


def test_mz_bin_recall_skips_missing_mz():
    peaks = [
        {"mz": None, "rt": 1.0, "isotope": 0, "name": "A"},
        {"mz": 150.0, "rt": 1.0, "isotope": 0, "name": "B"},
    ]
    flags = [True, True]
    bins = mz_bin_recall(peaks, flags, bin_width=100.0)
    assert [b["seg"] for b in bins] == ["100-200"]     # None-mz peak dropped
    assert bins[0]["n_total"] == 1


def test_is_dup_handles_bool_and_str():
    assert _is_dup({"is_duplicate": True}) is True
    assert _is_dup({"is_duplicate": False}) is False
    assert _is_dup({"is_duplicate": "True"}) is True
    assert _is_dup({"is_duplicate": "False"}) is False
    assert _is_dup({"is_duplicate": "true"}) is True
    assert _is_dup({}) is False                  # missing -> not a duplicate
    assert _is_dup({"is_duplicate": None}) is False
