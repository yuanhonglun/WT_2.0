"""Tests for the MS-DIAL MSDec MS2 deconvolution port (metabo_core.algorithms.msdec).

Reference: MSDecHandler.cs / MSDecProcess.cs / Ms2Dec.cs. Sub-function
expectations are hand-computed from the C# formulas; end-to-end tests assert
the deconvolution's behavioural contract (single component recovered,
overlapping co-eluters separated, apex gate, m/z not re-centroided).
"""
from __future__ import annotations

import numpy as np
import pytest

from metabo_core.algorithms.msdec import (
    baseline_correct,
    matched_filter,
    region_markers,
    solve_target_coefficient,
    deconvolute_ms2,
)
from metabo_core.config.msdec import lc_msdec_config


def _gauss(n, center, sigma, amp):
    x = np.arange(n, dtype=np.float64)
    return amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)


# --------------------------------------------------------------------------
# baseline_correct — MSDecHandler.cs getBaselineCorrectedPeaklist (L1380-1412)
# int() truncation on (coeff*RT + intercept); endpoints = local minima.
# --------------------------------------------------------------------------
def test_baseline_correct_flat_baseline_subtracts_constant():
    intensity = np.array([10.0, 12.0, 20.0, 12.0, 10.0])
    rt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    out = baseline_correct(intensity, rt, peak_top=2)
    np.testing.assert_allclose(out, [0.0, 2.0, 10.0, 2.0, 0.0])


def test_baseline_correct_sloped_baseline_int_truncation():
    intensity = np.array([10.0, 12.0, 20.0, 14.0, 12.0])
    rt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    out = baseline_correct(intensity, rt, peak_top=2)
    # corrected = I - int(0.5*RT + 10) → [0, 2, 9, 3, 0]
    np.testing.assert_allclose(out, [0.0, 2.0, 9.0, 3.0, 0.0])


def test_baseline_correct_negatives_floored_to_zero():
    intensity = np.array([5.0, 0.0, 8.0, 0.0, 5.0])
    rt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    out = baseline_correct(intensity, rt, peak_top=2)
    assert np.all(out >= 0.0)


# --------------------------------------------------------------------------
# matched_filter — MSDecHandler.cs getMatchedFileterArray (L1461-1481)
# fixed kernel length 21 (halfPoint=10), zero-padded.
# --------------------------------------------------------------------------
def test_matched_filter_unit_impulse_returns_kernel():
    n = 31
    sharp = np.zeros(n)
    sharp[15] = 1.0
    mf = matched_filter(sharp, sigma=0.5, half_point=10)
    # center coefficient is (1 - 0) * exp(0) = 1
    assert mf[15] == pytest.approx(1.0)
    # one step off: coef at x=±1, x/sigma=±2 → (1-4)*exp(-2)
    expected_off = (1 - (1 / 0.5) ** 2) * np.exp(-0.5 * (1 / 0.5) ** 2)
    assert mf[14] == pytest.approx(expected_off)
    assert mf[16] == pytest.approx(expected_off)


def test_matched_filter_zero_input_is_zero():
    mf = matched_filter(np.zeros(25), sigma=0.5, half_point=10)
    np.testing.assert_array_equal(mf, np.zeros(25))


# --------------------------------------------------------------------------
# region_markers — MSDecHandler.cs getRegionMarkers (L1414-1443), margin=5
# --------------------------------------------------------------------------
def test_region_markers_single_peak_one_region():
    n = 30
    mf = np.zeros(n)
    # a rising-then-falling positive bump well inside the margins
    for i, v in zip(range(12, 19), [0.2, 0.5, 0.9, 1.0, 0.9, 0.5, 0.2]):
        mf[i] = v
    regions = region_markers(mf, margin=5)
    assert len(regions) == 1
    begin, end = regions[0]
    # region brackets the bump (begins on the rise, ends as it returns <=0)
    assert begin <= 14
    assert end >= 16


def test_region_markers_ignores_margins():
    n = 20
    mf = np.zeros(n)
    mf[1] = 5.0  # inside left margin → ignored
    mf[18] = 5.0  # inside right margin → ignored
    assert region_markers(mf, margin=5) == []


# --------------------------------------------------------------------------
# solve_target_coefficient — MSDecProcess.cs Gram + LU inverse row-0 · z
# basis = [target, (neighbors...), linear, const]; only target coeff kept.
# --------------------------------------------------------------------------
def test_solve_pure_scaled_target_recovers_scale():
    target = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    exp = 3.0 * target
    coeff = solve_target_coefficient(target, [], exp)
    assert coeff == pytest.approx(3.0)


def test_solve_target_plus_constant_baseline():
    target = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    exp = 3.0 * target + 5.0  # const basis absorbs the offset
    coeff = solve_target_coefficient(target, [], exp)
    assert coeff == pytest.approx(3.0)


def test_solve_target_plus_linear_drift():
    target = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    linear = np.arange(5, dtype=np.float64)
    exp = 3.0 * target + 2.0 * linear  # linear basis absorbs the drift
    coeff = solve_target_coefficient(target, [], exp)
    assert coeff == pytest.approx(3.0)


def test_solve_separates_overlapping_neighbor():
    # exp is purely the neighbor (apex-shifted) component → target coeff ~0.
    target = np.array([0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    neighbor = np.array([0.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0])
    exp = 4.0 * neighbor
    coeff = solve_target_coefficient(target, [neighbor], exp)
    assert coeff == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# deconvolute_ms2 — end-to-end behavioural contract
# --------------------------------------------------------------------------
def test_deconvolute_single_component_recovers_all_fragments():
    n, center = 60, 30
    g = _gauss(n, center, sigma=6.0, amp=1.0)
    ion_mzs = np.array([100.0501, 150.1010, 200.1500])
    ion_eics = np.array([5000.0 * g, 3000.0 * g, 1500.0 * g])
    cfg = lc_msdec_config()
    cfg.min_amplitude = 200.0
    out_mz, out_int = deconvolute_ms2(
        ion_mzs, ion_eics, precursor_apex_scan=center, config=cfg
    )
    # all three co-eluting fragments are recovered
    assert set(np.round(out_mz, 4)) >= {100.0501, 150.1010, 200.1500}
    # relative order of intensities preserved (5000 > 3000 > 1500 component)
    order = {round(float(m), 4): i for m, i in zip(out_mz, out_int)}
    assert order[100.0501] > order[150.1010] > order[200.1500]


def test_deconvolute_preserves_product_mz_no_recentroiding():
    n, center = 60, 30
    g = _gauss(n, center, sigma=6.0, amp=1.0)
    # two product ions only 0.01 Da apart — re-centroiding (merge_close_ions)
    # would collapse them to one weighted m/z; MSDec must keep both.
    ion_mzs = np.array([100.0500, 100.0600])
    ion_eics = np.array([5000.0 * g, 4000.0 * g])
    cfg = lc_msdec_config()
    cfg.min_amplitude = 200.0
    out_mz, out_int = deconvolute_ms2(
        ion_mzs, ion_eics, precursor_apex_scan=center, config=cfg
    )
    assert 100.0500 in np.round(out_mz, 4)
    assert 100.0600 in np.round(out_mz, 4)


def test_deconvolute_apex_gate_rejects_far_precursor():
    n = 60
    g = _gauss(n, 30, sigma=6.0, amp=1.0)
    ion_mzs = np.array([100.05, 150.10])
    ion_eics = np.array([5000.0 * g, 3000.0 * g])
    cfg = lc_msdec_config()
    cfg.min_amplitude = 200.0
    # precursor apex 10 scans away from the model apex (>2) → empty spectrum.
    out_mz, out_int = deconvolute_ms2(
        ion_mzs, ion_eics, precursor_apex_scan=10, config=cfg
    )
    assert out_mz.size == 0
    assert out_int.size == 0


def test_deconvolute_separates_overlapping_coeluter():
    # Two co-eluters in one isolation window: component A (apex at precursor
    # scan) and component B (apex shifted +8 scans). MSDec should attribute
    # A-fragments strongly to the target and suppress B-fragments.
    n, a_center, b_center = 70, 30, 38
    ga = _gauss(n, a_center, sigma=5.0, amp=1.0)
    gb = _gauss(n, b_center, sigma=5.0, amp=1.0)
    ion_mzs = np.array([120.05, 121.05, 320.20, 321.20])
    # 120/121 belong to A (apex 30); 320/321 belong to B (apex 38).
    ion_eics = np.array([6000.0 * ga, 4000.0 * ga, 6000.0 * gb, 4000.0 * gb])
    cfg = lc_msdec_config()
    cfg.min_amplitude = 200.0
    out_mz, out_int = deconvolute_ms2(
        ion_mzs, ion_eics, precursor_apex_scan=a_center, config=cfg
    )
    spec = {round(float(m), 2): float(v) for m, v in zip(out_mz, out_int)}
    a_signal = spec.get(120.05, 0.0) + spec.get(121.05, 0.0)
    b_signal = spec.get(320.20, 0.0) + spec.get(321.20, 0.0)
    # A's fragments dominate the deconvoluted spectrum at the target apex.
    assert a_signal > 0.0
    assert a_signal > 3.0 * (b_signal + 1.0)
