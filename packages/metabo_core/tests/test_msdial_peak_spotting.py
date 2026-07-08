"""Tests for the MS-DIAL peak-spotting engine + primitives (Tasks 1.2-1.3, Track B).

These tests pin the numerically sensitive helpers ported from MS-DIAL's
``ChromatogramGlobalProperty_temp2`` / ``ChroChroChromatogram``
(``Common/CommonStandard/Components/ChromatogramGlobalProperty_temp2.cs``).
They are intentionally written first (TDD) and exercise behaviour, not the
internal numeric scale of the truncated MS-DIAL coefficients.
"""
import numpy as np
import pytest

from metabo_core.algorithms.baseline import lwma_smooth
from metabo_core.algorithms.msdial_peak_spotting import (
    _background_spike_filter,
    _count_spikes,
    _estimate_global_noise,
    _is_noise,
    _lwma_msdial,
    _sg_derivatives,
    _slope_noises,
    _upper_median,
    msdial_detect_peaks_in_chromatogram,
)
from metabo_core.config.msdial_peak_spotting import lc_msdial_config
from metabo_core.models.chromatography import DetectedPeak


# ---------------------------------------------------------------------------
# 0. _lwma_msdial  (faithful MS-DIAL imos LWMA, edge handling)
# ---------------------------------------------------------------------------
#
# Faithful to Smoothing.LinearWeightedMovingAverage imos method (Smoothing.cs
# lines 29-67): full (L+1)**2 normalisation for EVERY point, plus two triangular
# boundary corrections. This DIFFERS from baseline.lwma_smooth at the first/last
# L points -- lwma_smooth drops the out-of-range taps and shrinks the normaliser,
# whereas MS-DIAL keeps the full (L+1)**2 normaliser. Interior points are
# identical. The msdial engine must use THIS variant for chrom/ss/baseline.


def test_lwma_msdial_level1_boundary_and_interior():
    # L=1, x=[10,20,30,40,50]. MS-DIAL imos: dest[0]=(3a+b)/4, dest[-1]=(3e+d)/4
    # (NOT lwma_smooth's (2a+b)/3); interior is the standard 1-2-1 weighting
    # normalised by (1+1)**2 = 4.
    out = _lwma_msdial(np.array([10.0, 20.0, 30.0, 40.0, 50.0]), 1)
    assert out[0] == pytest.approx((3 * 10 + 20) / 4)        # 12.5
    assert out[-1] == pytest.approx((3 * 50 + 40) / 4)       # 47.5
    assert out[1] == pytest.approx((10 + 2 * 20 + 30) / 4)   # 20.0
    assert out[2] == pytest.approx((20 + 2 * 30 + 40) / 4)   # 30.0
    assert out[3] == pytest.approx((30 + 2 * 40 + 50) / 4)   # 40.0


def test_lwma_msdial_level2_boundary():
    # L=2, x=[10..70]. MS-DIAL imos boundary closed forms (hand-derived from the
    # imos definition):
    #   dest[0] = (6a + 2b + c) / 9
    #   dest[1] = (2a + 4b + 2c + d) / 9
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    out = _lwma_msdial(x, 2)
    assert out[0] == pytest.approx((6 * 10 + 2 * 20 + 30) / 9)
    assert out[1] == pytest.approx((2 * 10 + 4 * 20 + 2 * 30 + 40) / 9)


def test_lwma_msdial_interior_matches_lwma_smooth():
    # Only the boundaries differ: on the interior [L : n-L] the faithful imos
    # LWMA and baseline.lwma_smooth must agree exactly (both use the full
    # (L+1)**2 normalisation there). The boundaries genuinely DIFFER.
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 1000.0, 50)
    L = 3
    a = _lwma_msdial(x, L)
    b = lwma_smooth(x, L)
    np.testing.assert_allclose(a[L : x.size - L], b[L : x.size - L])
    assert not np.allclose(a[:L], b[:L])
    assert not np.allclose(a[-L:], b[-L:])


def test_lwma_msdial_constant_in_constant_out():
    # A constant trace maps to the same constant everywhere: the effective
    # weights sum to (L+1)**2 at every point, boundaries included.
    x = np.full(40, 1234.0)
    out = _lwma_msdial(x, 3)
    np.testing.assert_allclose(out, 1234.0)


def test_lwma_msdial_level_zero_returns_copy():
    # level <= 0 returns a float64 copy of the input (independent storage).
    x = np.array([1.0, 2.0, 3.0])
    out = _lwma_msdial(x, 0)
    assert np.array_equal(out, x)
    assert out is not x
    out[0] = 999.0
    assert x[0] == 1.0  # mutating the result must not touch the input


def test_lwma_msdial_empty_returns_empty():
    out = _lwma_msdial(np.zeros(0), 2)
    assert out.shape == (0,)
    assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# 1. _sg_derivatives
# ---------------------------------------------------------------------------


def test_sg_derivatives_linear_ramp():
    # On a linear ramp the SG first derivative is a positive constant equal to
    # the per-index slope (coeffs {-0.2,-0.1,0,0.1,0.2} integrate the slope),
    # and the second derivative is ~0. Boundaries (first/last 2) are exactly 0.
    y = 2.0 * np.arange(10)
    d1, d2 = _sg_derivatives(y)

    # First/last 2 entries are exactly zero (no valid 5-point window).
    assert d1[0] == 0.0 and d1[1] == 0.0
    assert d1[-1] == 0.0 and d1[-2] == 0.0
    assert d2[0] == 0.0 and d2[1] == 0.0
    assert d2[-1] == 0.0 and d2[-2] == 0.0

    interior = slice(2, -2)
    # slope == 2 per index -> d1 ~ 2.0, a positive constant.
    assert np.all(d1[interior] > 0.0)
    np.testing.assert_allclose(d1[interior], 2.0, atol=1e-9)
    # second derivative of a line is 0.
    np.testing.assert_allclose(d2[interior], 0.0, atol=1e-6)


def test_sg_derivatives_parabola_second_derivative_constant_positive():
    # On a discrete parabola the SG second derivative is a positive constant.
    # NOTE: MS-DIAL's SECOND_DIFF_COEFF = {1/7,-1/14,-1/7,-1/14,1/7} is HALF the
    # textbook SG 2nd-derivative, so for y = x**2 (true f'' = 2) it returns ~1.0.
    n = 15
    y = (np.arange(n) - 7.0) ** 2
    d1, d2 = _sg_derivatives(y)

    assert d1[0] == 0.0 and d1[1] == 0.0 and d1[-1] == 0.0 and d1[-2] == 0.0
    assert d2[0] == 0.0 and d2[1] == 0.0 and d2[-1] == 0.0 and d2[-2] == 0.0

    interior = slice(2, -2)
    assert np.all(d2[interior] > 0.0)
    # Constant across the interior.
    assert np.ptp(d2[interior]) < 1e-4
    # Half the textbook value (2.0) because of the MS-DIAL coefficient scale.
    np.testing.assert_allclose(d2[interior], 1.0, atol=1e-3)


def test_sg_derivatives_too_short_all_zero():
    # Fewer than 5 points -> no valid window anywhere.
    for n in (0, 1, 2, 3, 4):
        d1, d2 = _sg_derivatives(np.arange(n, dtype=float))
        assert d1.shape == (n,)
        assert np.all(d1 == 0.0)
        assert np.all(d2 == 0.0)


# ---------------------------------------------------------------------------
# 2. _slope_noises
# ---------------------------------------------------------------------------


def test_slope_noises_flat_plus_gaussian_small_positive():
    # A noisy-flat baseline plus one tall Gaussian: every noise is a real
    # (computed) median -> small, strictly positive, and clearly above the
    # 1e-4 empty-candidate fallback.
    rng = np.random.default_rng(0)
    n = 200
    x = np.arange(n)
    y = 1000.0 + rng.normal(0.0, 5.0, n)
    y = y + 5000.0 * np.exp(-0.5 * ((x - 100.0) / 4.0) ** 2)

    d1, d2 = _sg_derivatives(y)
    amplitude_noise, slope_noise, peaktop_noise = _slope_noises(y, d1, d2)

    for noise in (amplitude_noise, slope_noise, peaktop_noise):
        assert noise > 1e-3           # real median, well above the 1e-4 fallback
        assert noise < 100.0          # small vs the 5000-tall peak
        assert np.isfinite(noise)


def test_slope_noises_constant_trace_falls_back():
    # An all-constant trace has no nonzero diffs -> every candidate set is
    # empty -> MS-DIAL returns the 0.0001 fallback for all three.
    y = np.full(100, 1234.0)
    d1, d2 = _sg_derivatives(y)
    amplitude_noise, slope_noise, peaktop_noise = _slope_noises(y, d1, d2)
    assert amplitude_noise == 1e-4
    assert slope_noise == 1e-4
    assert peaktop_noise == 1e-4


def test_slope_noises_amplitude_uses_backward_max_forward_candidates():
    # Faithful MS-DIAL quirk: the amplitude-noise THRESHOLD max uses the
    # BACKWARD diff |y[i]-y[i-1]| (the 100 jump at i=2), while the CANDIDATES
    # use the FORWARD diff |y[i+1]-y[i]| (the 1's). threshold = 0.05*100 = 5,
    # candidates {1,1,1} all < 5 -> upper-median = 1.0.
    # A wrong forward-max impl would give max=1, threshold=0.05, no candidate
    # passes -> the 1e-4 fallback. So this value (1.0) pins the quirk.
    y = np.array([0, 0, 100, 101, 102, 103, 0], dtype=float)
    d1, d2 = _sg_derivatives(y)
    amplitude_noise, _, _ = _slope_noises(y, d1, d2)
    assert amplitude_noise == 1.0


def test_slope_noises_amplitude_uses_upper_median():
    # Forward candidates = {1,2,3,4} (diffs of the rising interior), backward
    # max = 100 (the jump at i=2) -> threshold 5, all candidates pass.
    # Upper median of {1,2,3,4} = sorted[4//2] = 3.0 (MS-DIAL BasicMathematics),
    # whereas numpy.median would be 2.5. Pins the upper-median rule.
    y = np.array([0, 0, 100, 101, 103, 106, 110, 0], dtype=float)
    d1, d2 = _sg_derivatives(y)
    amplitude_noise, _, _ = _slope_noises(y, d1, d2)
    assert amplitude_noise == 3.0
    assert amplitude_noise != 2.5


# ---------------------------------------------------------------------------
# 3. _estimate_global_noise  (faithful MS-DIAL GetMinimumNoiseLevel)
# ---------------------------------------------------------------------------
#
# Faithful to Chromatogram.GetMinimumNoiseLevel (Chromatogram.cs 512-540) plus
# the "* NoiseFactor" applied at its only call site (Chromatogram.cs line 698):
#   * bins of bin_size over the (already baseline-corrected) trace, KEEPING the
#     trailing partial bin;
#   * per-bin amplitude = max - min, but ONLY bins with min < max count -- flat
#     / zero-amplitude bins are excluded (line 529);
#   * ONLY those nonzero bins count toward the >= min_windows check (line 533);
#   * sufficient -> noise = upper_median(nonzero amplitudes) * factor, where the
#     upper median is InplaceSortMedian = sorted[size // 2] (line 534);
#   * insufficient -> noise = min_noise_level * factor, the MinimumNoiseLevel
#     PARAMETER (line 535) whose MS-DIAL GlobalParameter default is 0d (so the
#     fallback is 0 under defaults), NOT a hardcoded 0 in the algorithm;
#   * estimated_noise = max(1.0, noise / factor)  (line 199).


def _spike_bins(amplitudes, *, bin_size=50, n_flat_bins=0, floor=0.0):
    """Build a corrected trace whose k-th nonzero bin has (max-min)==amplitudes[k].

    Each amplitude bin is a flat ``floor`` with a single spike of height
    ``floor + amp`` (so min==floor, max==floor+amp, amplitude==amp). Optionally
    prepend ``n_flat_bins`` perfectly flat bins (amplitude 0, min==max) that the
    faithful estimator MUST exclude via the ``min < max`` filter.
    """
    bins = []
    for _ in range(n_flat_bins):
        bins.append(np.full(bin_size, floor, dtype=float))  # flat -> min == max
    for amp in amplitudes:
        b = np.full(bin_size, floor, dtype=float)
        b[0] = floor + amp                                  # single spike
        bins.append(b)
    return np.concatenate(bins) if bins else np.zeros(0, dtype=float)


def test_estimate_global_noise_upper_median_times_factor():
    # 10 nonzero bins (EVEN count) with amplitudes 10..100. The MS-DIAL upper
    # median is sorted[10 // 2] = sorted[5] = 60, NOT numpy's average-of-middle
    # (50 + 60) / 2 = 55. So noise = 60 * factor, distinguishing the faithful
    # upper median from numpy.median.
    amps = np.arange(10, 101, 10, dtype=float)  # [10,20,...,100] -> 10 bins
    corrected = _spike_bins(amps, bin_size=50)
    factor = 3.0

    noise, estimated_noise = _estimate_global_noise(
        corrected, bin_size=50, factor=factor, min_windows=10
    )
    assert noise == pytest.approx(60.0 * factor)      # upper median 60
    assert noise != pytest.approx(55.0 * factor)      # would be numpy.median
    assert estimated_noise == pytest.approx(max(1.0, noise / factor))  # 60.0


def test_estimate_global_noise_excludes_zero_bins():
    # 15 perfectly-flat (zero-amplitude) bins + 10 nonzero bins (amps 10..100).
    # The flat bins must be EXCLUDED: they neither contribute amplitudes nor
    # count toward min_windows. If they were wrongly kept, the 15 leading zeros
    # would dominate -> median 0 -> noise 0. The result must depend ONLY on the
    # nonzero bins.
    amps = np.arange(10, 101, 10, dtype=float)  # 10 nonzero bins
    corrected = _spike_bins(amps, bin_size=50, n_flat_bins=15)
    factor = 3.0

    noise, estimated_noise = _estimate_global_noise(
        corrected, bin_size=50, factor=factor, min_windows=10
    )
    assert noise == pytest.approx(60.0 * factor)   # same as without the flats
    assert estimated_noise == pytest.approx(60.0)


def test_estimate_global_noise_keeps_trailing_partial_bin():
    # 9 full nonzero bins (amps 10..90) + 1 trailing PARTIAL nonzero bin (only
    # 20 points, amplitude 100). MS-DIAL keeps the partial bin (buffer sized
    # ceil(_size / binSize)), giving 10 nonzero bins >= min_windows; the upper
    # median over [10..100] is 60. If the partial bin were discarded there would
    # be 9 nonzero bins (< min_windows) and the result would fall back to 0.
    corrected = _spike_bins(np.arange(10, 91, 10, dtype=float), bin_size=50)
    tail = np.zeros(20, dtype=float)
    tail[0] = 100.0  # trailing partial bin, amplitude 100
    corrected = np.concatenate([corrected, tail])
    factor = 3.0

    noise, _ = _estimate_global_noise(
        corrected, bin_size=50, factor=factor, min_windows=10
    )
    assert noise == pytest.approx(60.0 * factor)   # upper median of [10..100]


def test_estimate_global_noise_fallback_default_zero():
    # All-flat trace -> 0 nonzero bins (< min_windows). The MS-DIAL
    # GlobalParameter default MinimumNoiseLevel is 0d, so noise collapses to
    # 0 * factor = 0 and estimated -> max(1, 0) = 1.
    corrected = np.full(50 * 9, 1234.0, dtype=float)  # 9 flat bins, none nonzero
    noise, estimated_noise = _estimate_global_noise(
        corrected, bin_size=50, factor=3.0, min_windows=10
    )
    assert noise == 0.0
    assert estimated_noise == 1.0


def test_estimate_global_noise_fallback_uses_min_noise_level_param():
    # Fewer than min_windows nonzero bins -> fall back to the MinimumNoiseLevel
    # PARAMETER, not a hardcoded 0. Per Chromatogram.cs line 698 the
    # "* NoiseFactor" is applied to the WHOLE GetMinimumNoiseLevel return,
    # including this fallback branch -> noise = min_noise_level * factor.
    corrected = _spike_bins(np.array([10.0, 20.0, 30.0]), bin_size=50)  # 3 bins
    factor = 3.0
    min_noise_level = 7.0

    noise, estimated_noise = _estimate_global_noise(
        corrected,
        bin_size=50,
        factor=factor,
        min_windows=10,
        min_noise_level=min_noise_level,
    )
    assert noise == pytest.approx(min_noise_level * factor)            # 21.0
    assert estimated_noise == pytest.approx(max(1.0, noise / factor))  # 7.0


# ---------------------------------------------------------------------------
# 4. _is_noise
# ---------------------------------------------------------------------------


def _clean_is_noise_kwargs():
    """Baseline arguments under which NO IsNoise condition trips."""
    return dict(
        min_prom=800.0,
        max_prom=1000.0,
        noise=100.0,
        amplitude_noise=10.0,
        min_amplitude=500.0,
        fold=4.0,
        is_high_baseline=False,
        edge_min=5000.0,
        baseline_median=100.0,
    )


def _call_is_noise(**overrides):
    kwargs = _clean_is_noise_kwargs()
    kwargs.update(overrides)
    return _is_noise(
        kwargs["min_prom"],
        kwargs["max_prom"],
        kwargs["noise"],
        kwargs["amplitude_noise"],
        kwargs["min_amplitude"],
        fold=kwargs["fold"],
        is_high_baseline=kwargs["is_high_baseline"],
        edge_min=kwargs["edge_min"],
        baseline_median=kwargs["baseline_median"],
    )


def test_is_noise_clean_case_passes():
    assert _call_is_noise() is False


def test_is_noise_condition1_max_prom_below_noise():
    # max_prom (1000) < noise -> reject. Others untouched.
    assert _call_is_noise(noise=2000.0) is True


def test_is_noise_condition2_min_prom_below_min_amplitude():
    # min_prom (800) < min_amplitude -> reject. (noise 100 keeps cond1 off.)
    assert _call_is_noise(min_amplitude=900.0) is True


def test_is_noise_condition3_min_prom_below_sn_fold():
    # min_prom (800) < amplitude_noise * fold = 300 * 4 = 1200 -> reject.
    assert _call_is_noise(amplitude_noise=300.0) is True


def test_is_noise_condition4_high_baseline_and_low_edge():
    # is_high_baseline AND edge_min (50) < baseline_median (100) -> reject.
    assert _call_is_noise(
        is_high_baseline=True, edge_min=50.0, baseline_median=100.0
    ) is True


def test_is_noise_condition4_requires_both_parts():
    # High baseline but edge NOT below median -> condition does not trip.
    assert _call_is_noise(
        is_high_baseline=True, edge_min=150.0, baseline_median=100.0
    ) is False
    # Edge below median but baseline NOT high -> condition does not trip.
    assert _call_is_noise(
        is_high_baseline=False, edge_min=50.0, baseline_median=100.0
    ) is False


# ---------------------------------------------------------------------------
# 5. msdial_detect_peaks_in_chromatogram  (Task 1.3 derivative engine)
# ---------------------------------------------------------------------------
#
# Faithful port of the MSDIAL5 normal LC path for one chromatogram:
#   PeakSpottingCore.GetChromatogramPeakFeatures(EIC, detector)
#     -> PeakDetection.PeakDetectionVS1(Chromatogram)
#       -> ChroChroChromatogram.DetectPeaks  (derivative walk + GetPeakDetectionResult)
#   followed by GetBackgroundSubtractedPeaks (RAW-EIC spike filter).
# Smoothing chain (PeakSpottingCore.cs:359 + Chromatogram.GetProperty:693-706):
#   chrom = LWMA(raw, L); ss = LWMA(LWMA(chrom,1),1); baseline = LWMA(chrom,20);
#   corrected = max(ss - baseline, 0). baseline_median/max/min come from chrom.


def _gauss(n, center, height, sigma):
    x = np.arange(n, dtype=float)
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def test_detects_single_clean_peak():
    rt = np.linspace(0, 3, 120)
    y = _gauss(120, 60, 10000, 4) + 50.0
    cfg = lc_msdial_config()
    peaks = msdial_detect_peaks_in_chromatogram(rt, y, config=cfg)
    assert len(peaks) == 1
    assert abs(peaks[0].apex_index - 60) <= 2
    assert peaks[0].height > 8000
    assert peaks[0].area > peaks[0].height      # x60 sec convention
    assert peaks[0].sn_ratio > 10


def test_rejects_pure_noise():
    rng = np.random.default_rng(0)
    rt = np.linspace(0, 3, 200)
    y = rng.uniform(0, 80, 200)                 # below MinimumAmplitude=1000
    peaks = msdial_detect_peaks_in_chromatogram(rt, y, config=lc_msdial_config())
    assert peaks == []


def test_splits_two_resolved_peaks():
    rt = np.linspace(0, 5, 200)
    y = _gauss(200, 70, 9000, 4) + _gauss(200, 120, 7000, 4) + 30.0
    peaks = msdial_detect_peaks_in_chromatogram(rt, y, config=lc_msdial_config())
    assert len(peaks) == 2


def test_rejects_below_min_amplitude():
    rt = np.linspace(0, 3, 120)
    y = _gauss(120, 60, 600, 4) + 20.0          # 600 < 1000
    peaks = msdial_detect_peaks_in_chromatogram(rt, y, config=lc_msdial_config())
    assert peaks == []


# --- targeted tests for ported quirks --------------------------------------


def test_all_zero_trace_returns_empty():
    # Flat-zero trace: derivatives are 0, IsPeakStarted never fires.
    rt = np.linspace(0, 3, 120)
    peaks = msdial_detect_peaks_in_chromatogram(
        rt, np.zeros(120), config=lc_msdial_config()
    )
    assert peaks == []


def test_too_short_trace_returns_empty():
    # n < 2*margin+1 (margin = max(min_data_points, 2) = 5): the walk range
    # [margin, n-margin) is empty, so even a tall peak yields nothing.
    rt = np.linspace(0, 1, 8)
    y = _gauss(8, 4, 10000, 1) + 50.0
    peaks = msdial_detect_peaks_in_chromatogram(rt, y, config=lc_msdial_config())
    assert peaks == []


def test_empty_trace_returns_empty():
    peaks = msdial_detect_peaks_in_chromatogram(
        np.zeros(0), np.zeros(0), config=lc_msdial_config()
    )
    assert peaks == []


def test_i_max_j_splits_two_close_peaks():
    # Two resolved-but-close Gaussians (centres 80 and 110, sigma 4). The
    # ``i = max(i, j)`` advance past each resolved peak is what lets the second
    # peak be detected separately rather than swallowed; assert both apices.
    rt = np.linspace(0, 5, 200)
    y = _gauss(200, 80, 9000, 4) + _gauss(200, 110, 7000, 4) + 30.0
    peaks = msdial_detect_peaks_in_chromatogram(rt, y, config=lc_msdial_config())
    assert len(peaks) == 2
    apices = sorted(p.apex_index for p in peaks)
    assert abs(apices[0] - 80) <= 2
    assert abs(apices[1] - 110) <= 2


def test_peaks_sorted_by_rt_apex_and_passthrough_mz():
    # Output is sorted ascending by rt_apex; precursor/product m/z pass through.
    rt = np.linspace(0, 5, 200)
    y = _gauss(200, 70, 9000, 4) + _gauss(200, 120, 7000, 4) + 30.0
    peaks = msdial_detect_peaks_in_chromatogram(
        rt, y, config=lc_msdial_config(), precursor_mz_nominal=255, product_mz=123.45
    )
    assert len(peaks) == 2
    assert peaks[0].rt_apex < peaks[1].rt_apex
    for p in peaks:
        assert p.precursor_mz_nominal == 255
        assert p.product_mz == 123.45


def test_clean_peak_area_sec_and_gaussian_similarity_range():
    # area uses the x60 min->sec convention (so area >> height for a sharp
    # peak) and gaussian_similarity is a finite ratio in [0, 1] for a clean
    # Gaussian (near 1).
    rt = np.linspace(0, 3, 120)
    y = _gauss(120, 60, 10000, 4) + 50.0
    peaks = msdial_detect_peaks_in_chromatogram(rt, y, config=lc_msdial_config())
    assert len(peaks) == 1
    p = peaks[0]
    assert p.area > 0.0
    # ×60 min->sec: without it the area (~2.5e3) would fall *below* the height;
    # with it the sharp peak's area clears 10× the height.
    assert p.area > 10.0 * p.height
    assert 0.0 <= p.gaussian_similarity <= 1.0
    assert p.gaussian_similarity > 0.9    # clean Gaussian


def test_high_baseline_cond4_rejects_low_edge_peak():
    # IsNoise condition 4 (ChromatogramGlobalProperty_temp2.cs:67 via
    # HasBoundaryBelowThreshold): on a HIGH-baseline EIC, a peak whose lower
    # boundary dips below the chrom baseline-median is rejected. The SAME sharp
    # peak on a low flat baseline (not high-baseline) IS detected -- isolating
    # cond4 as the cause.
    n = 200
    rt = np.linspace(0, 5, n)

    # High plateau (10000) with a broad central depression; a sharp peak rises
    # from the bottom of the depression so its edges sit below the median.
    high = (
        np.full(n, 10000.0)
        - _gauss(n, 100, 7500, 18)   # broad valley
        + _gauss(n, 100, 11000, 3)   # sharp peak from the valley floor
    )
    high = np.clip(high, 0.0, None)
    chrom_high = lwma_smooth(high, lc_msdial_config().smoothing_level)
    bmed = _upper_median(chrom_high)
    assert bmed > (float(chrom_high.max()) + float(chrom_high.min())) * 0.5  # high baseline
    high_peaks = msdial_detect_peaks_in_chromatogram(rt, high, config=lc_msdial_config())
    assert high_peaks == []          # rejected by cond4

    # Control: identical sharp peak on a low flat baseline -> detected.
    control = _gauss(n, 100, 11000, 3) + 2500.0
    chrom_ctrl = lwma_smooth(control, lc_msdial_config().smoothing_level)
    assert not (
        _upper_median(chrom_ctrl)
        > (float(chrom_ctrl.max()) + float(chrom_ctrl.min())) * 0.5
    )  # NOT high baseline
    control_peaks = msdial_detect_peaks_in_chromatogram(
        rt, control, config=lc_msdial_config()
    )
    assert len(control_peaks) == 1


# ---------------------------------------------------------------------------
# 6. _count_spikes + _background_spike_filter  (Task 1.3 background-spike filter)
# ---------------------------------------------------------------------------
#
# Faithful port of Chromatogram.CountSpikes (Chromatogram.cs lines 889-911) and
# PeakSpottingCore.GetBackgroundSubtractedPeaks (PeakSpottingCore.cs lines 656-676).
# A spike pair (IsPeakTop -> spikeMax, IsBottom -> spikeMin) increments the counter
# when |spikeMax - spikeMin| / 2 > threshold; a detected peak is DROPPED when the
# combined left+right flank counter >= background_spike_threshold (default 15).


def test_count_spikes_sawtooth_low_threshold_positive():
    # Sawtooth [0, 10, 0, 10, ...] of length 2k=40.
    # Each (peak-top at 10, bottom at 0) pair contributes |10-0|/2 = 5.
    # With threshold=4 < 5 every pair qualifies; expected count = k-1 = 19.
    # (Boundary handling: left_bound=max(0,1)=1, right_bound=min(39,38)=38.
    #  Pairs complete at i=2,4,...,38 -> 19 pairs = k-1.)
    k = 20
    raw = np.tile([0.0, 10.0], k)   # length 40: [0,10,0,10,...,0,10]
    count = _count_spikes(raw, 0, raw.size - 1, 4.0)
    assert count == k - 1   # 19


def test_count_spikes_sawtooth_high_threshold_zero():
    # Same sawtooth; threshold=100 >> 5 -> |10-0|/2 = 5 never exceeds it -> 0 pairs.
    k = 20
    raw = np.tile([0.0, 10.0], k)
    count = _count_spikes(raw, 0, raw.size - 1, 100.0)
    assert count == 0


def test_count_spikes_scales_with_k():
    # For k tiles of [0, 10] (length 2k), count = k-1 under a low threshold.
    # Verifies the count grows deterministically with the number of sawtooth cycles.
    prev = 0
    for k in (5, 10, 20):
        raw = np.tile([0.0, 10.0], k)
        c = _count_spikes(raw, 0, raw.size - 1, 4.0)
        assert c == k - 1, f"k={k}: expected {k - 1}, got {c}"
        assert c > prev
        prev = c


def test_background_spike_filter_reject_and_keep():
    """_background_spike_filter drops a spiky-flanked peak; keeps a clean-flanked one.

    Approach 3a: call _background_spike_filter directly with a synthetic DetectedPeak.

    Setup
    -----
    n=220-point trace.  Peak: left_index=100, apex_index=110, right_index=120.
    tracking = min(10*(120-100), 50) = 50.
    chrom: triangular profile (left/right edges=0, apex=10).
    amp_diff = max(10-0, 10-0) = 10;  spike_threshold = 10/3 ≈ 3.33.

    SPIKY raw -- sawtooth [0, 8] in left flank [50..99] and right flank [120..169]:
      (8-0)/2 = 4 > 3.33 -> every pair counts.
      Left flank: 25 pairs; right flank: 25 pairs -> total 50 >= 15 -> REJECT.

    CLEAN raw -- flat zeros in the flanks, triangular peak in the centre:
      No valid (peak-top, bottom) pairs -> counter = 0 < 15 -> KEEP.

    PROVE: kept_spiky is empty (reject branch ran); kept_clean holds the one peak.
    """
    n = 220
    rt = np.linspace(0.0, 5.0, n)

    # chrom: triangular peak, left=100 (0.0), apex=110 (10.0), right=120 (0.0).
    chrom = np.zeros(n)
    chrom[100:111] = np.linspace(0.0, 10.0, 11)   # rising: 0, 1, ..., 10
    chrom[110:121] = np.linspace(10.0, 0.0, 11)   # falling: 10, 9, ..., 0

    # Synthetic detected peak matching the chrom profile above.
    peak = DetectedPeak(
        precursor_mz_nominal=0,
        product_mz=0.0,
        rt_apex=float(rt[110]),
        rt_left=float(rt[100]),
        rt_right=float(rt[120]),
        apex_index=110,
        left_index=100,
        right_index=120,
        height=float(chrom[110]),   # 10.0
        area=100.0,
        sn_ratio=5.0,
        gaussian_similarity=0.9,
    )

    sawtooth = np.tile([0.0, 8.0], 25)   # 50 elements: [0,8,0,8,...,0,8]

    # ---- SPIKY: sawtooth in both flanks, triangular profile in peak region ----
    raw_spiky = np.zeros(n)
    raw_spiky[50:100] = sawtooth           # left flank [50..99]
    raw_spiky[100:121] = chrom[100:121]    # peak region (IsValidPeakTop passes at 110)
    raw_spiky[120:170] = sawtooth          # right flank [120..169]

    # Sanity-assert the spike counter fires (mirrors _background_spike_filter logic).
    amp_diff = max(peak.height - chrom[100], peak.height - chrom[120])   # 10.0
    tracking = min(10 * (120 - 100), 50)                                   # 50
    spike_thr = amp_diff / 3.0                                             # ≈ 3.33
    left_count = _count_spikes(raw_spiky, 100 - tracking, 100, spike_thr)
    right_count = _count_spikes(raw_spiky, 120, 120 + tracking, spike_thr)
    total = left_count + right_count
    assert total >= 15, (
        f"Expected >=15 spike pairs for spiky flanks, got {left_count}+{right_count}={total}"
    )

    kept_spiky = _background_spike_filter([peak], raw_spiky, chrom, threshold=15)
    assert kept_spiky == [], "Spiky-flanked peak must be REJECTED (counter >= 15)"

    # ---- CLEAN: flat zeros in flanks, same triangular peak ----
    raw_clean = np.zeros(n)
    raw_clean[100:121] = chrom[100:121]

    kept_clean = _background_spike_filter([peak], raw_clean, chrom, threshold=15)
    assert len(kept_clean) == 1, "Clean-flanked peak must be KEPT (counter < 15)"
    assert kept_clean[0] is peak
