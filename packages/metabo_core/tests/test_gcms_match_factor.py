"""Tests for ``metabo_core.gcms.library_matching.gcms_match_factor``.

Pin the Plan D composite-spectral-score shape:

    spectral_score_raw = (wdp*3 + sdp*3 + rdp*2 + matched_pct) / 9

with adjacent-deconvolution penalty (-0.02 per overlap, clamped to >=0,
on the [0,1] scale), AMDIS spectrum complexity scaling (eq 8/9, a=0.5),
and AMDIS detection-threshold correction (eq 11, default off).

Optional chromatographic component is multiplicative (mode='rt'|'ri'):
    total = spectral_score * chrom_score**chrom_weight
With mode='none' (default), total == spectral_score.

RDP is exposed as a first-class field on the returned dict — per user
2026-04-29 it is the primary judgment metric for fullscan GC-MS.
"""
from __future__ import annotations

import math

import pytest


# ---------------------------------------------------------------------------
# Identity / disjoint
# ---------------------------------------------------------------------------

def test_identity_with_many_balanced_peaks_keeps_spectral_score_high() -> None:
    """Identity matching: a 20-peak spectrum with reasonably balanced
    intensities keeps the AMDIS complexity-scaling factor close to 1.0
    (sum_A is large so w is small per AMDIS eq 9), so spectral_score
    stays high. total == spectral_score when mode='none'.
    """
    from metabo_core.gcms.library_matching import gcms_match_factor

    peaks = [(50.0 + i, 1.0 - 0.04 * i) for i in range(20)]
    out = gcms_match_factor(peaks, peaks, mode="none")
    assert out["spectral_score"] >= 0.85
    assert out["total"] == pytest.approx(out["spectral_score"], abs=1e-12)
    assert out["chrom_score"] is None
    # All score-component keys are present.
    for key in ("wdp", "rdp", "sdp", "matched_pct", "n_adjacent_subtracted"):
        assert key in out


def test_disjoint_spectra_spectral_score_zero() -> None:
    from metabo_core.gcms.library_matching import gcms_match_factor

    a = [(50.0, 100.0)]
    b = [(150.0, 100.0)]
    out = gcms_match_factor(a, b, mode="none")
    assert out["spectral_score"] == 0.0
    assert out["total"] == 0.0
    assert out["wdp"] == 0.0
    assert out["rdp"] == 0.0
    assert out["sdp"] == 0.0


# ---------------------------------------------------------------------------
# Composite weighting matches the existing composite_similarity shape
# (wdp*3 + sdp*3 + rdp*2 + matched_pct) / 9
# ---------------------------------------------------------------------------

def test_composite_weighting_matches_composite_similarity_shape() -> None:
    """Identity test verifying ``raw = (wdp*3 + sdp*3 + rdp*2 + matched_pct)/9``
    is the pre-scaling shape. spectral_score = raw * complexity * threshold,
    with complexity in (0, 1] and threshold = 1 by default.
    """
    from metabo_core.gcms.library_matching import gcms_match_factor

    peaks = [(50.0, 100.0), (75.0, 95.0), (100.0, 90.0), (120.0, 85.0), (140.0, 80.0)]
    out = gcms_match_factor(peaks, peaks, mode="none")

    wdp = out["wdp"]
    sdp = out["sdp"]
    rdp = out["rdp"]
    mp = out["matched_pct"]
    expected_raw = (wdp * 3 + sdp * 3 + rdp * 2 + mp) / 9.0

    # spectral_score is at most expected_raw (complexity scaling can only
    # dampen, not amplify). For this 5-peak case complexity is around
    # 0.85, so spectral_score lies between raw*0.7 and raw*1.0.
    assert out["spectral_score"] <= expected_raw + 1e-9
    assert out["spectral_score"] >= expected_raw * 0.7


# ---------------------------------------------------------------------------
# Peak-count penalty NOT double-applied
# ---------------------------------------------------------------------------

def test_peak_count_penalty_not_double_applied() -> None:
    """For a single-peak reference, WDP/RDP/SDP each apply a 0.75 peak-count
    penalty internally. gcms_match_factor must NOT multiply again.
    """
    from metabo_core.algorithms.similarity import (
        weighted_dot_product,
        reverse_dot_product,
        simple_dot_product,
    )
    from metabo_core.gcms.library_matching import gcms_match_factor

    peaks = [(100.0, 100.0)]
    wdp_raw = weighted_dot_product(peaks, peaks, mz_tolerance=0.01)
    rdp_raw = reverse_dot_product(peaks, peaks, mz_tolerance=0.01)
    sdp_raw = simple_dot_product(peaks, peaks, mz_tolerance=0.01)

    out = gcms_match_factor(peaks, peaks, mode="none")

    # gcms_match_factor's wdp/rdp/sdp output must be exactly the
    # already-penalized primitive values (no extra multiplication).
    assert out["wdp"] == pytest.approx(wdp_raw, abs=1e-9)
    assert out["rdp"] == pytest.approx(rdp_raw, abs=1e-9)
    assert out["sdp"] == pytest.approx(sdp_raw, abs=1e-9)


# ---------------------------------------------------------------------------
# Spectrum complexity scaling (AMDIS eq 8/9)
# ---------------------------------------------------------------------------

def test_spectrum_complexity_scaling_dampens_single_dominant_peak() -> None:
    """A spectrum with one dominant peak and tiny side peaks should have
    its dominant peak diminished by the AMDIS eq-8/9 scaling. The spectral
    score for such a spectrum should be visibly below 1.0 even though the
    measured spectrum is identical to the reference.

    AMDIS paper §"Spectrum Complexity": "with a spectrum containing only
    one prominent peak, setting a=0.5 causes this peak to be diminished
    by a factor of three".
    """
    from metabo_core.gcms.library_matching import gcms_match_factor

    # 1 dominant peak (1.0) plus many small ones.
    peaks = [(50.0, 1.0)] + [(60.0 + i, 0.01) for i in range(10)]
    out_dom = gcms_match_factor(peaks, peaks, mode="none")

    # 5 roughly-equal peaks (multi-peak case): scaling barely changes things.
    balanced = [(50.0, 1.0), (60.0, 0.95), (70.0, 0.9), (80.0, 0.85), (90.0, 0.80)]
    out_bal = gcms_match_factor(balanced, balanced, mode="none")

    # Both identity, but the dominant case should be visibly lower.
    assert out_bal["spectral_score"] > out_dom["spectral_score"]
    # The dominant case should be reduced relative to its raw composite.
    raw_dom = (out_dom["wdp"] * 3 + out_dom["sdp"] * 3 + out_dom["rdp"] * 2
               + out_dom["matched_pct"]) / 9.0
    assert out_dom["spectral_score"] < raw_dom


# ---------------------------------------------------------------------------
# Detection threshold correction (AMDIS eq 11)
# ---------------------------------------------------------------------------

def test_detection_threshold_correction_no_op_at_zero() -> None:
    from metabo_core.gcms.library_matching import gcms_match_factor

    peaks = [(50.0, 100.0), (75.0, 50.0), (100.0, 25.0), (120.0, 10.0)]
    out_no = gcms_match_factor(peaks, peaks, mode="none", detection_threshold=0.0)
    out_def = gcms_match_factor(peaks, peaks, mode="none")
    assert out_no["spectral_score"] == pytest.approx(out_def["spectral_score"], abs=1e-12)


def test_detection_threshold_correction_at_0_1_reduces_score() -> None:
    """With threshold=0.1, spectral_score *= (1 - 0.1^0.3) ≈ 1 - 0.5012 ≈ 0.4988."""
    from metabo_core.gcms.library_matching import gcms_match_factor

    peaks = [(50.0, 100.0), (75.0, 50.0), (100.0, 25.0), (120.0, 10.0)]
    out_no = gcms_match_factor(peaks, peaks, mode="none", detection_threshold=0.0)
    out_thr = gcms_match_factor(peaks, peaks, mode="none", detection_threshold=0.1)
    expected_factor = 1.0 - 0.1 ** 0.3
    assert out_thr["spectral_score"] == pytest.approx(
        out_no["spectral_score"] * expected_factor, abs=1e-9,
    )


# ---------------------------------------------------------------------------
# Mode='rt' / Mode='ri' Gaussian
# ---------------------------------------------------------------------------

def test_mode_rt_perfect_match_total_equals_spectral() -> None:
    from metabo_core.gcms.library_matching import gcms_match_factor

    peaks = [(50.0, 100.0), (75.0, 50.0), (100.0, 25.0)]
    out = gcms_match_factor(
        peaks, peaks, mode="rt",
        rt_query=10.0, rt_ref=10.0, rt_tolerance=0.1,
    )
    assert out["chrom_score"] == pytest.approx(1.0, abs=1e-12)
    assert out["total"] == pytest.approx(out["spectral_score"], abs=1e-12)


def test_mode_rt_one_tolerance_off_drops_chrom_to_exp_minus_half() -> None:
    from metabo_core.gcms.library_matching import gcms_match_factor

    peaks = [(50.0, 100.0), (75.0, 50.0)]
    out = gcms_match_factor(
        peaks, peaks, mode="rt",
        rt_query=10.0, rt_ref=10.1, rt_tolerance=0.1,
    )
    assert out["chrom_score"] == pytest.approx(math.exp(-0.5), abs=1e-9)


def test_mode_ri_perfect_match() -> None:
    from metabo_core.gcms.library_matching import gcms_match_factor

    peaks = [(50.0, 100.0), (75.0, 50.0)]
    out = gcms_match_factor(
        peaks, peaks, mode="ri",
        ri_query=1234.5, ri_ref=1234.5, ri_tolerance=10.0,
    )
    assert out["chrom_score"] == pytest.approx(1.0, abs=1e-12)


def test_mode_ri_offset_drops_chrom_to_gaussian_value() -> None:
    from metabo_core.gcms.library_matching import gcms_match_factor

    peaks = [(50.0, 100.0), (75.0, 50.0)]
    out = gcms_match_factor(
        peaks, peaks, mode="ri",
        ri_query=1234.5, ri_ref=1244.5, ri_tolerance=10.0,
    )
    # 10 RI units offset, tolerance 10 → exp(-0.5)
    assert out["chrom_score"] == pytest.approx(math.exp(-0.5), abs=1e-9)


def test_mode_none_ignores_rt_ri_inputs() -> None:
    from metabo_core.gcms.library_matching import gcms_match_factor

    peaks = [(50.0, 100.0), (75.0, 50.0)]
    out = gcms_match_factor(
        peaks, peaks, mode="none",
        rt_query=10.0, rt_ref=15.0, rt_tolerance=0.1,
        ri_query=1000.0, ri_ref=2000.0, ri_tolerance=10.0,
    )
    assert out["chrom_score"] is None
    assert out["total"] == pytest.approx(out["spectral_score"], abs=1e-12)


# ---------------------------------------------------------------------------
# Adjacent-deconvolution penalty
# ---------------------------------------------------------------------------

def test_adjacent_deconvolution_penalty_subtracts_0_02_per_overlap() -> None:
    """Plan D ordering: raw -= 0.02 * n_adjacent on [0,1] scale (clamped),
    then complexity scaling, then detection-threshold correction. So the
    spectral_score difference is the raw penalty (0.04 for n=2) propagated
    through the same complexity factor.
    """
    from metabo_core.gcms.library_matching import gcms_match_factor

    peaks = [(50.0, 100.0), (75.0, 50.0), (100.0, 25.0), (120.0, 10.0)]
    out_0 = gcms_match_factor(peaks, peaks, mode="none", n_adjacent_subtracted=0)
    out_2 = gcms_match_factor(peaks, peaks, mode="none", n_adjacent_subtracted=2)
    # Both runs share the same complexity factor f. spectral_0 = raw_0 * f,
    # spectral_2 = (raw_0 - 0.04) * f, so:
    # spectral_0 - spectral_2 = 0.04 * f.
    # f for this 4-peak spectrum (complexity factor) is in (0, 1].
    delta = out_0["spectral_score"] - out_2["spectral_score"]
    # The delta must be strictly positive and at most 0.04 (when f=1).
    assert 0 < delta <= 0.04 + 1e-9
    # And at least some lower bound: f for this spectrum is >= 0.4.
    assert delta >= 0.04 * 0.4
    assert out_2["n_adjacent_subtracted"] == 2


def test_adjacent_deconvolution_penalty_clamped_to_zero() -> None:
    """Penalty cannot drop the score below zero."""
    from metabo_core.gcms.library_matching import gcms_match_factor

    a = [(50.0, 100.0)]
    b = [(150.0, 100.0)]
    out = gcms_match_factor(a, b, mode="none", n_adjacent_subtracted=100)
    assert out["spectral_score"] == 0.0
    assert out["total"] == 0.0


# ---------------------------------------------------------------------------
# RDP exposure (first-class output)
# ---------------------------------------------------------------------------

def test_rdp_exposed_as_first_class_field() -> None:
    """Per user 2026-04-29: RDP must be a separate dict field, not just
    folded into ``total``. Fullscan workflows lean heavily on RDP because
    it ignores measured peaks not in the reference."""
    from metabo_core.gcms.library_matching import gcms_match_factor

    peaks = [(50.0, 100.0), (75.0, 50.0), (100.0, 25.0)]
    out = gcms_match_factor(peaks, peaks, mode="none")
    assert "rdp" in out
    assert isinstance(out["rdp"], float)
    assert 0.0 <= out["rdp"] <= 1.0


# ---------------------------------------------------------------------------
# matched_pct exposure
# ---------------------------------------------------------------------------

def test_matched_pct_exposed_as_first_class_field() -> None:
    """matched_pct = matched / max(reference_peak_count, 1)."""
    from metabo_core.gcms.library_matching import gcms_match_factor

    measured = [(50.0, 100.0), (75.0, 50.0)]                # 2 peaks
    reference = [(50.0, 100.0), (75.0, 50.0), (100.0, 25.0), (120.0, 10.0)]  # 4

    out = gcms_match_factor(measured, reference, mode="none")
    # 2 out of 4 reference peaks matched → matched_pct = 0.5.
    assert out["matched_pct"] == pytest.approx(0.5, abs=1e-9)


def test_n_matched_exposed_as_first_class_field() -> None:
    """Plan D follow-up #4: n_matched is the integer count of matched peaks,
    returned directly so callers do not have to round-trip via
    matched_pct * n_ref (which can drift by ±1)."""
    from metabo_core.gcms.library_matching import gcms_match_factor

    measured = [(50.0, 100.0), (75.0, 50.0)]
    reference = [(50.0, 100.0), (75.0, 50.0), (100.0, 25.0), (120.0, 10.0)]

    out = gcms_match_factor(measured, reference, mode="none")
    assert out["n_matched"] == 2
    # Empty inputs → n_matched=0.
    out2 = gcms_match_factor([], reference, mode="none")
    assert out2["n_matched"] == 0


# ---------------------------------------------------------------------------
# total range
# ---------------------------------------------------------------------------

def test_total_score_in_range() -> None:
    from metabo_core.gcms.library_matching import gcms_match_factor

    peaks = [(50.0, 100.0), (75.0, 50.0), (100.0, 25.0)]
    out = gcms_match_factor(peaks, peaks, mode="rt", rt_query=10.0, rt_ref=10.0,
                             rt_tolerance=0.1)
    assert 0.0 <= out["total"] <= 1.0
