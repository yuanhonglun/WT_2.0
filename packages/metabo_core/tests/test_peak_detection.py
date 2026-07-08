"""Regression tests for the MS-DIAL-style peak detector."""
import numpy as np

from metabo_core.algorithms.peak_detection import detect_peaks


def _gaussian_eic(rt_center: float = 5.0, sigma: float = 0.08, amp: float = 5000.0):
    rt = np.linspace(0, 15, 894)
    intensity = amp * np.exp(-0.5 * ((rt - rt_center) / sigma) ** 2)
    intensity = np.maximum(
        intensity + np.random.default_rng(42).normal(0, 20, rt.size), 0,
    )
    return rt, intensity


def test_detect_peaks_finds_single_gaussian():
    rt, intensity = _gaussian_eic()
    peaks = detect_peaks(rt, intensity, min_amplitude=500)
    assert len(peaks) == 1
    assert abs(peaks[0].rt_apex - 5.0) < 0.05
    # Peak height is now measured above the local baseline; the synthetic
    # Gaussian sits on a ~zero baseline so the corrected height should
    # land close to ``amp``.
    assert peaks[0].height > 1000
    # Area is integrated with the NumPy-version-agnostic trapezoid shim
    # (np.trapz was removed in NumPy 2.x); a real Gaussian must yield area > 0.
    assert peaks[0].area > 0


def test_detect_peaks_area_exceeds_height():
    """Peak area integrates over time in SECONDS (MS-DIAL convention), so a
    real chromatographic peak's area must exceed its apex height.

    Readers hand RT to the detector in MINUTES; integrating the trapezoid
    over minutes (peak width ~0.1-0.4 min) made ``area < height`` — wrong.
    The area axis is now seconds (``rt * 60``), restoring ``area > height``.
    """
    rt, intensity = _gaussian_eic(rt_center=7.0, sigma=0.08, amp=5000.0)
    peaks = detect_peaks(rt, intensity, min_amplitude=500)
    assert len(peaks) == 1
    assert peaks[0].area > peaks[0].height


def test_detect_peaks_returns_empty_for_short_input():
    rt = np.linspace(0, 1, 4)
    intensity = np.zeros(4)
    assert detect_peaks(rt, intensity) == []


def test_detect_peaks_filters_below_min_amplitude():
    rt, intensity = _gaussian_eic(amp=200.0)
    peaks = detect_peaks(rt, intensity, min_amplitude=10000)
    assert peaks == []


def test_detect_peaks_baseline_subtracts_drift():
    """A peak riding on a slow baseline drift should still be found, and
    its reported height should reflect the height **above** the drift,
    not the absolute apex."""
    rt = np.linspace(0, 15, 894)
    drift = 500.0 + 100.0 * (rt / rt.max())  # slow linear baseline
    peak_amp = 4000.0
    peak = peak_amp * np.exp(-0.5 * ((rt - 7.0) / 0.08) ** 2)
    noise = np.random.default_rng(0).normal(0, 15, rt.size)
    intensity = np.maximum(drift + peak + noise, 0)

    peaks = detect_peaks(rt, intensity, min_amplitude=500)
    assert len(peaks) == 1
    # Height above baseline should be well above the pre-drift floor
    # (~500) yet below the raw apex (drift + amp ≈ 4600). The wide
    # LWMA baseline (w=20) slightly tracks the peak shoulders so the
    # corrected height lands around 2000 — that's MS-DIAL's behaviour.
    assert 1500 < peaks[0].height < 4500


def test_detect_peaks_rejects_when_amplitude_below_noise():
    """A small bump that does not clear the noise gate must be rejected
    even if it would pass the absolute ``min_amplitude``."""
    rt = np.linspace(0, 15, 894)
    rng = np.random.default_rng(1)
    # Loud, broadband baseline noise.
    noisy = np.maximum(rng.normal(500, 200, rt.size), 0)
    # A tiny bump on top — apex height above baseline is ~150, which is
    # well below the binned-amplitude noise level.
    bump = 150.0 * np.exp(-0.5 * ((rt - 7.0) / 0.05) ** 2)
    intensity = noisy + bump

    peaks = detect_peaks(rt, intensity, min_amplitude=10)
    assert peaks == []


def test_rt_window_drops_peak_before_min():
    """A peak whose apex RT is below ``rt_window_min`` is rejected.
    Catches the solvent-front / dead-time artefact at the start of an
    LC gradient."""
    rt, intensity = _gaussian_eic(rt_center=1.5)
    # Without the window the early peak is found.
    no_win = detect_peaks(rt, intensity, min_amplitude=500)
    assert len(no_win) == 1
    # With rt_window_min=2.0 (apex at 1.5 < 2.0), the peak is rejected.
    with_win = detect_peaks(rt, intensity, min_amplitude=500, rt_window_min=2.0)
    assert with_win == []


def test_rt_window_drops_peak_after_max():
    """A peak whose apex RT is above ``rt_window_max`` is rejected.
    This is the F00096 head/tail artefact in the rice ASFAM feedback:
    a fake slope at the post-gradient re-equilibration tail."""
    rt, intensity = _gaussian_eic(rt_center=13.5)
    no_win = detect_peaks(rt, intensity, min_amplitude=500)
    assert len(no_win) == 1
    with_win = detect_peaks(rt, intensity, min_amplitude=500, rt_window_max=12.0)
    assert with_win == []


def test_rt_window_keeps_peak_within_range():
    """A peak whose apex sits inside ``[rt_window_min, rt_window_max]``
    is not affected."""
    rt, intensity = _gaussian_eic(rt_center=7.0)
    peaks = detect_peaks(
        rt, intensity, min_amplitude=500,
        rt_window_min=1.0, rt_window_max=12.0,
    )
    assert len(peaks) == 1
    assert abs(peaks[0].rt_apex - 7.0) < 0.05


def test_rt_window_defaults_are_off():
    """Default ``rt_window_min=None`` and ``rt_window_max=None`` must
    behave identically to not passing them — preserves back-compat."""
    rt, intensity = _gaussian_eic(rt_center=5.0)
    a = detect_peaks(rt, intensity, min_amplitude=500)
    b = detect_peaks(rt, intensity, min_amplitude=500,
                     rt_window_min=None, rt_window_max=None)
    assert len(a) == len(b) == 1
    assert a[0].rt_apex == b[0].rt_apex


def test_prominence_ratio_keeps_clean_gaussian():
    """A clean Gaussian has min_prom/apex ≈ 0.9, well above the LC
    default ratio of 0.3. It must pass the ratio gate."""
    rt, intensity = _gaussian_eic(rt_center=7.0, amp=5000.0)
    peaks = detect_peaks(
        rt, intensity, min_amplitude=500, min_prominence_ratio=0.3,
    )
    assert len(peaks) == 1


def test_prominence_ratio_is_monotonic():
    """Adding the ratio gate can only remove peaks, never add them."""
    rt, intensity = _gaussian_eic(rt_center=7.0)
    base = detect_peaks(rt, intensity, min_amplitude=500,
                        min_prominence_ratio=0.0)
    strict = detect_peaks(rt, intensity, min_amplitude=500,
                          min_prominence_ratio=0.5)
    assert len(strict) <= len(base)


def test_prominence_ratio_default_is_off():
    """Default ``min_prominence_ratio=0.0`` must behave identically to
    not passing the parameter."""
    rt, intensity = _gaussian_eic(rt_center=5.0)
    a = detect_peaks(rt, intensity, min_amplitude=500)
    b = detect_peaks(rt, intensity, min_amplitude=500, min_prominence_ratio=0.0)
    assert len(a) == len(b) == 1
    assert a[0].rt_apex == b[0].rt_apex
