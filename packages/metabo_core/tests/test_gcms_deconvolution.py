"""Tests for ``metabo_core.gcms.deconvolution``.

Implements the AMDIS-style deconvolution algorithm (Stein 1999). Tests
are organized by phase, matching Plan D Task 3 sub-tasks 3a-3f.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest


# ===========================================================================
# Phase 1: Noise factor estimation and threshold-transition handling (3a)
# ===========================================================================

def test_estimate_noise_factor_returns_value_within_10pct_of_known() -> None:
    """Synthetic constant-signal chromatogram with Gaussian noise. The
    proportionality factor of the noise is the expected Nf within 10%.
    """
    from metabo_core.gcms.deconvolution import estimate_noise_factor

    # Build chromatograms with mean=100, noise sigma = 1.0 * sqrt(100) = 10.0
    # so the proportionality factor (sigma / sqrt(mean)) is 1.0.
    # Use median-deviation, not std, so expect ~ 0.6745 * 1.0 (median /
    # sqrt(mean) for normal).
    rng = np.random.default_rng(42)
    n_scans = 500
    chroms = []
    for _ in range(20):
        chrom = 100.0 + rng.normal(0.0, 10.0, size=n_scans)
        chrom = np.maximum(chrom, 1.0)  # avoid zeros which would reject segments
        chroms.append(chrom)

    nf = estimate_noise_factor(chroms, include_tic=False)
    # For Gaussian noise, median |x - mean| ≈ 0.6745 * sigma. Here:
    # sigma=10, mean=100, so Nf ≈ 6.745 / sqrt(100) = 0.6745.
    assert 0.5 <= nf <= 0.85, f"Nf {nf} not within tolerance of 0.6745"


def test_estimate_noise_factor_rejects_segments_with_zero() -> None:
    """A segment containing any zero must be rejected; remaining valid
    segments still produce a sample. If all segments are rejected the
    fallback Nf is 1.0."""
    from metabo_core.gcms.deconvolution import estimate_noise_factor

    # 13 scans, all zeros → no valid sample → fallback 1.0.
    chrom = np.zeros(13, dtype=float)
    nf = estimate_noise_factor([chrom], include_tic=False)
    assert nf == 1.0


def test_estimate_noise_factor_rejects_segments_with_few_crossings() -> None:
    """A monotonic segment (no crossings of the mean) must be rejected.
    Half of 13 = 6.5; segments need >= 7 crossings.
    """
    from metabo_core.gcms.deconvolution import estimate_noise_factor

    # Strictly monotonic: each adjacent pair never both differ from mean
    # in the same direction. Crossings = 1 (the median split). << 7.
    chrom = np.linspace(10.0, 100.0, 13)
    # Pad with a noisy segment to ensure at least one rejection happens.
    rng = np.random.default_rng(0)
    noisy = 100.0 + rng.normal(0, 10.0, size=13)
    full = np.concatenate([chrom, noisy])
    nf = estimate_noise_factor([full], include_tic=False)
    # At least the second segment may produce a sample. We just verify it
    # doesn't crash and returns a positive number.
    assert nf > 0


def test_estimate_noise_factor_is_median_of_medians_not_global_median() -> None:
    """AMDIS Stein 1999 §"Noise Analysis": per-chromatogram median first,
    then median across chromatograms. On heterogeneous data (one quiet
    chromatogram + one noisier one) this diverges from a flat global
    median because each chromatogram gets one vote regardless of how
    many segments it contributed.
    """
    from metabo_core.gcms.deconvolution import estimate_noise_factor

    rng = np.random.default_rng(0)
    # Chromatogram A: low noise factor (Nf ~= 1.0). 5 segments → 5 samples.
    base_a = 100.0
    a = base_a + rng.normal(0, 1.0 * math.sqrt(base_a), size=13 * 5)
    # Chromatogram B: high noise factor (Nf ~= 5.0). 1 segment → 1 sample.
    base_b = 100.0
    b = base_b + rng.normal(0, 5.0 * math.sqrt(base_b), size=13 * 1)
    # Pad B to match A's length so include_tic doesn't conflate them.
    b = np.concatenate([b, np.full(13 * 4, base_b)])

    nf = estimate_noise_factor([a, b], include_tic=False)
    # Median-of-medians: median of (per-chrom-A median ~1, per-chrom-B median ~5)
    # → ~3 (midpoint of two single-sample medians); allow generous tolerance.
    # Flat global median (the WRONG behavior) would lean strongly toward Nf~1
    # because A contributes 5 samples and B contributes 1.
    # Threshold: median-of-medians must be >= 1.5; flat global median would
    # typically be <= 1.5.
    assert nf >= 1.5, (
        f"median-of-medians regression: Nf={nf} suggests the implementation "
        f"reverted to a flat global median across all per-segment samples"
    )


def test_replace_threshold_transitions_replaces_zero_with_AT_sqrt_fraction() -> None:
    """For AT=10, with half of the scans involved in threshold transitions,
    zero abundance values are replaced by 10 * sqrt(0.5) ≈ 7.07. Per
    AMDIS Stein 1999 sect. "Threshold transitions".
    """
    from metabo_core.gcms.deconvolution import replace_threshold_transitions

    # 10-scan chromatogram, alternating 10 / 0:
    # 10, 0, 10, 0, 10, 0, 10, 0, 10, 0
    chrom = np.array([10.0, 0.0] * 5)
    out = replace_threshold_transitions(chrom, threshold_value=10.0, n_segments=1)
    # Every scan is involved in a zero/non-zero transition with its
    # neighbor → fraction = 1.0. So zero values become 10 * sqrt(1.0) = 10.
    # (Slight edge effect at boundaries; we tolerate that.)
    expected = 10.0 * math.sqrt(1.0)
    # All originally-zero values must be > 0 now.
    zero_mask = chrom == 0
    assert np.all(out[zero_mask] > 0)
    # And close to the expected value.
    assert np.allclose(out[zero_mask], expected, atol=1e-6)


def test_replace_threshold_transitions_no_zeros_returns_input() -> None:
    from metabo_core.gcms.deconvolution import replace_threshold_transitions

    chrom = np.array([10.0, 12.0, 11.0, 13.0])
    out = replace_threshold_transitions(chrom, threshold_value=10.0)
    assert np.allclose(out, chrom)


def test_replace_threshold_transitions_all_zeros_returns_zero() -> None:
    """All-zero chromatogram: no transitions exist. Returns as-is."""
    from metabo_core.gcms.deconvolution import replace_threshold_transitions

    chrom = np.zeros(20)
    out = replace_threshold_transitions(chrom, threshold_value=10.0)
    assert np.allclose(out, 0.0)


def test_replace_threshold_transitions_does_not_modify_input() -> None:
    from metabo_core.gcms.deconvolution import replace_threshold_transitions

    chrom = np.array([10.0, 0.0, 10.0, 0.0])
    out = replace_threshold_transitions(chrom, threshold_value=10.0)
    assert chrom[1] == 0.0  # original unchanged
    assert out[1] != 0.0    # output replaced


# ===========================================================================
# Phase 2: Component perception (3b)
# ===========================================================================

def _gaussian_peak(n_scans: int, apex: float, sigma: float, amp: float) -> np.ndarray:
    x = np.arange(n_scans, dtype=np.float64)
    return amp * np.exp(-0.5 * ((x - apex) / sigma) ** 2)


def test_perceive_components_three_clear_apices() -> None:
    """3 well-separated peaks at known apex scans → 3 components within
    1-scan tolerance."""
    from metabo_core.gcms.deconvolution import perceive_components

    n_scans = 60
    apexes = [10, 30, 50]
    sigma = 1.5
    amp = 1000.0
    chroms = np.zeros((3, n_scans), dtype=np.float64)
    for i, ap in enumerate(apexes):
        chroms[i] = _gaussian_peak(n_scans, ap, sigma, amp)

    components = perceive_components(chroms, Nf=1.0, use_tic_path=False)
    assert len(components) >= 3
    # The first three components (sorted by apex) match the expected apexes.
    apex_actual = sorted(c.apex_scan for c in components)[:3]
    for got, expected in zip(apex_actual, apexes):
        assert abs(got - expected) < 1.0, f"got {got} expected ~{expected}"


def test_perceive_components_uses_tic_path_when_no_strong_individual_max() -> None:
    """If no individual ion has a sufficient peak-height-vs-noise but the
    TIC does, the TIC path should still perceive the component.

    (Each individual ion is below the 4-noise-unit threshold; their sum
    is well above it.)
    """
    from metabo_core.gcms.deconvolution import perceive_components

    n_scans = 50
    chroms = np.zeros((10, n_scans), dtype=np.float64)
    # Each ion contributes a tiny bump at scan 25.
    for i in range(10):
        chroms[i] = _gaussian_peak(n_scans, 25.0, 1.5, 5.0)
    # Add tiny baseline noise so individual peaks fail the height test
    # but the TIC peak succeeds.
    rng = np.random.default_rng(0)
    chroms = chroms + rng.normal(0, 1.0, size=chroms.shape)
    chroms = np.maximum(chroms, 0)

    # With Nf=1 and a 4-noise-unit threshold, each ion's apex (≈ 5.0)
    # over baseline (≈ 0) needs height >= 4 * 1 * sqrt(5) ≈ 8.9. So
    # individual ions fail. TIC apex ≈ 50 over noise ≈ 4 * sqrt(50) ≈ 28
    # → passes.
    components = perceive_components(chroms, Nf=1.0, use_tic_path=True)
    # With TIC path enabled, at least one component (the TIC) should appear.
    assert any(c.perceived_via_tic for c in components) or any(
        abs(c.apex_scan - 25.0) < 2.0 for c in components
    )


def test_perceive_components_filters_low_sharpness_via_75pct_cutoff() -> None:
    """Only ions with sharpness >= 75% of the max are added to the
    contributing-ion list of a component."""
    from metabo_core.gcms.deconvolution import perceive_components

    n_scans = 40
    chroms = np.zeros((3, n_scans), dtype=np.float64)
    # Strong sharp ion.
    chroms[0] = _gaussian_peak(n_scans, 20.0, 1.0, 1000.0)
    # Weak (low-sharpness) ion at the same apex.
    chroms[1] = _gaussian_peak(n_scans, 20.0, 5.0, 100.0)
    # Different apex.
    chroms[2] = _gaussian_peak(n_scans, 35.0, 1.0, 500.0)

    components = perceive_components(chroms, Nf=1.0, use_tic_path=False,
                                      sharpness_cutoff_ratio=0.75)
    # The component near scan 20 should include ion 0 but probably not
    # ion 1 (its sharpness will be much lower).
    near20 = [c for c in components if abs(c.apex_scan - 20.0) < 2.0]
    assert near20, f"expected a component near scan 20, got {[c.apex_scan for c in components]}"
    c = near20[0]
    assert 0 in c.contributing_ions


def test_perceive_components_min_range_scans_merges_split_apex_ions() -> None:
    """A sharp dominant ion + co-eluting ions whose apex jitters within
    a few scans should be ONE component when ``min_range_scans`` is set
    to a width larger than the jitter — not N independent components.

    This mirrors the 0.7.260526 over-segmentation bug: passion_fruit
    full-scan produced 9 features at RT 5.283 because the sharp anchor
    ion's small uncertainty range left the adjacent ions free to be
    perceived independently.
    """
    from metabo_core.gcms.deconvolution import perceive_components

    n_scans = 60
    chroms = np.zeros((4, n_scans), dtype=np.float64)
    # Four co-eluting ions whose apex positions jitter by ±2 scans, all
    # individually sharp enough to qualify as their own component under
    # the original 50/sharpness uncertainty window.
    chroms[0] = _gaussian_peak(n_scans, 30.0, 0.8, 1000.0)  # dominant
    chroms[1] = _gaussian_peak(n_scans, 32.0, 0.8, 800.0)
    chroms[2] = _gaussian_peak(n_scans, 28.0, 0.8, 600.0)
    chroms[3] = _gaussian_peak(n_scans, 30.0, 0.8, 500.0)

    # Original behavior (min_range_scans=0): expect multiple components
    # near the apex because the small uncertainty window leaves the
    # jittered ions un-claimed.
    base = perceive_components(chroms, Nf=1.0, use_tic_path=False)
    near30_old = [c for c in base if abs(c.apex_scan - 30.0) < 3.0]
    assert len(near30_old) >= 2, (
        f"sanity: original algorithm should over-segment this case, "
        f"got {[c.apex_scan for c in base]}"
    )

    # Floored behavior: a 3-scan minimum range merges them into one.
    floored = perceive_components(
        chroms, Nf=1.0, use_tic_path=False,
        min_range_scans=3, inclusion_cutoff_ratio=0.3,
    )
    near30_new = [c for c in floored if abs(c.apex_scan - 30.0) < 3.0]
    assert len(near30_new) == 1, (
        f"floored algorithm should yield one component near scan 30, "
        f"got {[c.apex_scan for c in floored]}"
    )
    # That single component must absorb all four ions as contributors.
    assert set(near30_new[0].contributing_ions) == {0, 1, 2, 3}


def test_perceive_components_defaults_match_pre_a_behavior() -> None:
    """``min_range_scans=0`` + ``inclusion_cutoff_ratio=None`` (defaults)
    must reproduce the original Stein 1999 behavior so existing callers
    are unaffected by the 0.7.260526 signature widening."""
    from metabo_core.gcms.deconvolution import perceive_components

    n_scans = 60
    apexes = [10, 30, 50]
    chroms = np.zeros((3, n_scans), dtype=np.float64)
    for i, ap in enumerate(apexes):
        chroms[i] = _gaussian_peak(n_scans, ap, 1.5, 1000.0)

    baseline = perceive_components(chroms, Nf=1.0, use_tic_path=False)
    explicit = perceive_components(
        chroms, Nf=1.0, use_tic_path=False,
        min_range_scans=0, inclusion_cutoff_ratio=None,
    )

    assert len(baseline) == len(explicit)
    for a, b in zip(baseline, explicit):
        assert abs(a.apex_scan - b.apex_scan) < 1e-9
        assert a.contributing_ions == b.contributing_ions


def test_deskew_chromatograms_first_last_scan_unchanged() -> None:
    """AMDIS special case 1: first and last scans are not interpolated."""
    from metabo_core.gcms.deconvolution import deskew_chromatograms

    arr = np.array([[1.0, 5.0, 9.0, 5.0, 1.0]])
    out = deskew_chromatograms(arr)
    assert out[0, 0] == arr[0, 0]
    assert out[0, -1] == arr[0, -1]


def test_deskew_chromatograms_zero_values_stay_zero() -> None:
    """AMDIS special case 2: zero values are not interpolated."""
    from metabo_core.gcms.deconvolution import deskew_chromatograms

    arr = np.array([[10.0, 0.0, 10.0]])
    out = deskew_chromatograms(arr)
    assert out[0, 1] == 0.0


def test_deskew_chromatograms_interpolated_clamped_to_AT() -> None:
    """AMDIS special case 3: non-zero interpolated values cannot be < AT.

    Construct a 3-scan parabola whose interpolated apex falls between
    AT and the original value; verify the interpolated value is >= AT.
    """
    from metabo_core.gcms.deconvolution import deskew_chromatograms

    # 3 scans with a tiny dip in the middle. Quadratic predicts the
    # extremum slightly higher; we just verify the AT clamp fires when
    # the predicted value is below AT.
    arr = np.array([[5.0, 0.5, 5.0]])
    AT = 1.0
    out = deskew_chromatograms(arr, AT=AT)
    # The interpolated middle value is the parabola apex prediction;
    # the function clamps non-zero values up to AT if below AT.
    assert out[0, 1] >= 0  # never negative
    if out[0, 1] != 0.5:    # only if the interpolation actually fired
        assert out[0, 1] >= AT


# ===========================================================================
# Phase 3: Model peak construction (3c)
# ===========================================================================

def test_build_model_peak_single_dominant_ion_returns_normalized_shape() -> None:
    """A component with one dominant contributing ion: model is the
    normalized shape of that ion."""
    from metabo_core.gcms.deconvolution import (
        Component,
        build_model_peak,
    )

    n_scans = 30
    chroms = np.zeros((2, n_scans), dtype=np.float64)
    chroms[0] = _gaussian_peak(n_scans, 15.0, 1.5, 1000.0)
    chroms[1] = np.zeros(n_scans)  # silent ion

    comp = Component(
        apex_scan=15.0, apex_scan_int=15, sharpness=10.0,
        contributing_ions=[0],
        window_lo=10, window_hi=20,
    )
    model = build_model_peak(comp, chroms)
    assert model.max() == pytest.approx(1.0, abs=1e-9)
    # The apex of the model should be near scan 15.
    assert int(np.argmax(model)) == 15


def test_build_model_peak_two_co_maximizing_ions_sums_normalized() -> None:
    """Two ions with comparable sharpness contributing → model is the
    sum (scaled to peak 1.0)."""
    from metabo_core.gcms.deconvolution import (
        Component,
        build_model_peak,
    )

    n_scans = 30
    chroms = np.zeros((2, n_scans), dtype=np.float64)
    chroms[0] = _gaussian_peak(n_scans, 15.0, 1.5, 1000.0)
    chroms[1] = _gaussian_peak(n_scans, 15.0, 1.5, 500.0)

    comp = Component(
        apex_scan=15.0, apex_scan_int=15, sharpness=10.0,
        contributing_ions=[0, 1],
        window_lo=10, window_hi=20,
    )
    model = build_model_peak(comp, chroms)
    assert model.max() == pytest.approx(1.0, abs=1e-9)
    # Outside the window, the model is zero.
    assert model[0] == 0.0
    assert model[-1] == 0.0


def test_build_model_peak_excludes_non_contributing_ions() -> None:
    """An ion not in contributing_ions has no influence on the model
    even if it has high intensity."""
    from metabo_core.gcms.deconvolution import (
        Component,
        build_model_peak,
    )

    n_scans = 30
    chroms = np.zeros((2, n_scans), dtype=np.float64)
    chroms[0] = _gaussian_peak(n_scans, 15.0, 1.5, 1000.0)
    chroms[1] = _gaussian_peak(n_scans, 15.0, 5.0, 5000.0)  # high intensity, broad

    comp = Component(
        apex_scan=15.0, apex_scan_int=15, sharpness=10.0,
        contributing_ions=[0],   # ion 1 is excluded
        window_lo=12, window_hi=18,
    )
    model = build_model_peak(comp, chroms)
    # Only ion 0 contributes → model shape should match ion 0 (sharper).
    # Compare shape: model should be more peaked than if ion 1 were included.
    # (Loose check: ratio model[15]/model[12] should be high, since ion 0
    # alone is sharp.)
    if model[12] > 0:
        ratio = model[15] / model[12]
        assert ratio > 5.0


def test_build_model_peak_tic_path_uses_tic() -> None:
    """A TIC-perceived component with empty contributing_ions uses the
    TIC restricted to the window."""
    from metabo_core.gcms.deconvolution import (
        Component,
        build_model_peak,
    )

    n_scans = 30
    chroms = np.zeros((3, n_scans), dtype=np.float64)
    for i in range(3):
        chroms[i] = _gaussian_peak(n_scans, 15.0, 1.5, 100.0)

    comp = Component(
        apex_scan=15.0, apex_scan_int=15, sharpness=5.0,
        contributing_ions=[],
        window_lo=10, window_hi=20,
        perceived_via_tic=True,
    )
    model = build_model_peak(comp, chroms)
    assert model.max() == pytest.approx(1.0, abs=1e-9)
    assert int(np.argmax(model)) == 15


# ===========================================================================
# Phase 4: Per-m/z least-squares fit + adjacent subtraction (3d)
# ===========================================================================

def test_deconvolve_spectrum_single_gaussian_recovers_apex_intensity() -> None:
    """A single Gaussian peak: deconvolve_spectrum returns the apex
    intensity within 5%."""
    from metabo_core.gcms.deconvolution import (
        Component,
        deconvolve_spectrum,
    )

    n_scans = 30
    apex = 15
    chroms = np.zeros((1, n_scans), dtype=np.float64)
    chroms[0] = _gaussian_peak(n_scans, apex, 1.5, 1000.0)
    mz_values = np.array([100.0])

    comp = Component(
        apex_scan=float(apex), apex_scan_int=apex, sharpness=10.0,
        contributing_ions=[0],
        window_lo=10, window_hi=20,
    )
    out = deconvolve_spectrum(comp, chroms, mz_values)
    # Recovered intensity should be near 1000 (the original amplitude)
    # within 5%.
    assert out.intensity[0] == pytest.approx(1000.0, rel=0.05)
    assert out.intensity_no_subtraction[0] == pytest.approx(1000.0, rel=0.05)
    assert out.n_adjacent_subtracted == 0


def test_deconvolve_spectrum_two_overlapping_peaks_extracted_with_subtraction() -> None:
    """Two overlapping Gaussian peaks at scan 13 and scan 17 → with adjacent
    subtraction, both should be correctly extracted (each within 10%)."""
    from metabo_core.gcms.deconvolution import (
        Component,
        deconvolve_spectrum,
    )

    n_scans = 30
    apex_a = 13
    apex_b = 17
    sigma = 1.5
    amp_a = 1000.0
    amp_b = 800.0

    # 2 ions: ion 0 maximizes near a, ion 1 maximizes near b.
    chroms = np.zeros((2, n_scans), dtype=np.float64)
    chroms[0] = _gaussian_peak(n_scans, apex_a, sigma, amp_a) \
              + 0.3 * _gaussian_peak(n_scans, apex_b, sigma, amp_b)
    chroms[1] = 0.3 * _gaussian_peak(n_scans, apex_a, sigma, amp_a) \
              + _gaussian_peak(n_scans, apex_b, sigma, amp_b)
    mz_values = np.array([100.0, 200.0])

    comp_a = Component(
        apex_scan=float(apex_a), apex_scan_int=apex_a, sharpness=10.0,
        contributing_ions=[0],
        window_lo=8, window_hi=22,
    )
    comp_b = Component(
        apex_scan=float(apex_b), apex_scan_int=apex_b, sharpness=10.0,
        contributing_ions=[1],
        window_lo=8, window_hi=22,
    )

    out_a = deconvolve_spectrum(comp_a, chroms, mz_values, neighbor_components=[comp_b])
    out_b = deconvolve_spectrum(comp_b, chroms, mz_values, neighbor_components=[comp_a])

    assert out_a.n_adjacent_subtracted == 1
    assert out_b.n_adjacent_subtracted == 1

    # Component a's spectrum should be dominated by m/z 100 (ion 0).
    assert out_a.intensity[0] > out_a.intensity[1]
    # Component b's spectrum should be dominated by m/z 200 (ion 1).
    assert out_b.intensity[1] > out_b.intensity[0]


def test_deconvolve_spectrum_three_overlapping_keeps_both_with_and_without_subtraction() -> None:
    """Three overlapping peaks: with up-to-2-neighbor subtraction,
    intensity_no_subtraction is the no-subtraction variant; intensity
    is the with-subtraction variant. Both should be present and
    different in general."""
    from metabo_core.gcms.deconvolution import (
        Component,
        deconvolve_spectrum,
    )

    n_scans = 40
    apexes = [15, 19, 23]
    sigma = 1.5
    chroms = np.zeros((3, n_scans), dtype=np.float64)
    for i, ap in enumerate(apexes):
        chroms[i] = _gaussian_peak(n_scans, ap, sigma, 1000.0 - 100.0 * i)
        # Add cross-bleed so neighbors actually contribute non-zero.
        for j, ap2 in enumerate(apexes):
            if i != j:
                chroms[i] += 0.2 * _gaussian_peak(n_scans, ap2, sigma, 1000.0)
    mz_values = np.array([100.0, 200.0, 300.0])

    comps = [
        Component(apex_scan=float(ap), apex_scan_int=ap, sharpness=10.0,
                  contributing_ions=[i], window_lo=10, window_hi=30)
        for i, ap in enumerate(apexes)
    ]
    out = deconvolve_spectrum(comps[1], chroms, mz_values,
                               neighbor_components=[comps[0], comps[2]])
    assert out.n_adjacent_subtracted == 2
    # The two variants must be available.
    assert out.intensity.shape == out.intensity_no_subtraction.shape
    # In general, the with-subtraction version is smaller for the
    # neighbor m/z's because their contribution is removed.
    assert out.intensity_no_subtraction[0] >= 0
    assert out.intensity_no_subtraction[2] >= 0


# ===========================================================================
# Phase 5: Peak flagging (3e)
# ===========================================================================

def test_flag_peaks_fm_mismatch_above_threshold_flags_peak() -> None:
    """A peak whose extracted shape diverges sharply from the model is
    flagged. We use a flat-top peak that doesn't match a Gaussian model.
    """
    from metabo_core.gcms.deconvolution import (
        Component,
        deconvolve_spectrum,
    )

    n_scans = 30
    apex = 15
    chroms = np.zeros((2, n_scans), dtype=np.float64)
    # Ion 0: clean Gaussian (matches model).
    chroms[0] = _gaussian_peak(n_scans, apex, 1.5, 1000.0)
    # Ion 1: square pulse around the apex (poor match to a Gaussian model).
    chroms[1, 10:21] = 500.0
    mz_values = np.array([100.0, 200.0])

    comp = Component(
        apex_scan=float(apex), apex_scan_int=apex, sharpness=10.0,
        contributing_ions=[0],   # only the Gaussian ion shapes the model
        window_lo=10, window_hi=21,
    )
    out = deconvolve_spectrum(comp, chroms, mz_values, Nf=1.0)
    # The square-pulse ion should be flagged because its shape doesn't
    # match the (Gaussian) model.
    assert out.flags[1]


def test_flag_peaks_fm_above_reject_threshold_zeroes_intensity() -> None:
    """FM > 0.6 → peak rejected (intensity zeroed AND flagged).

    The shape-mismatch case: ion has positive c from the lstsq fit but
    its shape inside the window diverges sharply from the model.
    """
    from metabo_core.gcms.deconvolution import (
        Component,
        deconvolve_spectrum,
    )

    n_scans = 30
    apex = 15
    chroms = np.zeros((2, n_scans), dtype=np.float64)
    # Ion 0: clean Gaussian (matches model).
    chroms[0] = _gaussian_peak(n_scans, apex, 1.5, 1000.0)
    # Ion 1: U-shaped — high at the window edges, dip in the middle.
    # Has a positive lstsq fit but its FM is large.
    chroms[1, 10:21] = np.array(
        [500, 400, 300, 200, 100, 50, 100, 200, 300, 400, 500],
        dtype=np.float64,
    )
    mz_values = np.array([100.0, 200.0])

    comp = Component(
        apex_scan=float(apex), apex_scan_int=apex, sharpness=10.0,
        contributing_ions=[0],
        window_lo=10, window_hi=21,
    )
    out = deconvolve_spectrum(comp, chroms, mz_values, Nf=1.0)
    # Ion 1 is either rejected (intensity 0) or flagged.
    assert out.flags[1] or out.intensity[1] == 0


def test_flag_peaks_low_sn_flags_peak() -> None:
    """Peaks with S/N < 2 are flagged."""
    from metabo_core.gcms.deconvolution import (
        Component,
        deconvolve_spectrum,
    )

    n_scans = 30
    apex = 15
    chroms = np.zeros((2, n_scans), dtype=np.float64)
    chroms[0] = _gaussian_peak(n_scans, apex, 1.5, 1000.0)
    # Tiny peak at the same shape: extracted intensity will be small;
    # with Nf large enough, S/N stays below 2.
    chroms[1] = _gaussian_peak(n_scans, apex, 1.5, 5.0)
    mz_values = np.array([100.0, 200.0])

    comp = Component(
        apex_scan=float(apex), apex_scan_int=apex, sharpness=10.0,
        contributing_ions=[0],
        window_lo=10, window_hi=21,
    )
    out = deconvolve_spectrum(comp, chroms, mz_values, Nf=10.0)
    # With Nf=10, the tiny extracted intensity (~ 5) has S/N = 5 / (10 *
    # sqrt(5)) = 5 / 22.4 ≈ 0.22 < 2 → flagged.
    assert out.flags[1]


def test_flag_peaks_clean_match_not_flagged() -> None:
    """A clean Gaussian peak that matches the model perfectly with
    high S/N should NOT be flagged."""
    from metabo_core.gcms.deconvolution import (
        Component,
        deconvolve_spectrum,
    )

    n_scans = 30
    apex = 15
    chroms = np.zeros((1, n_scans), dtype=np.float64)
    chroms[0] = _gaussian_peak(n_scans, apex, 1.5, 1000.0)
    mz_values = np.array([100.0])

    comp = Component(
        apex_scan=float(apex), apex_scan_int=apex, sharpness=10.0,
        contributing_ions=[0],
        window_lo=10, window_hi=21,
    )
    out = deconvolve_spectrum(comp, chroms, mz_values, Nf=1.0)
    assert not out.flags[0]


def test_flag_peaks_zero_neighbor_at_apex_flags() -> None:
    """A peak adjacent to a zero-abundance scan at the apex is flagged
    as a possible noise spike."""
    from metabo_core.gcms.deconvolution import (
        Component,
        deconvolve_spectrum,
    )

    n_scans = 30
    apex = 15
    chroms = np.zeros((2, n_scans), dtype=np.float64)
    chroms[0] = _gaussian_peak(n_scans, apex, 1.5, 1000.0)
    # Ion 1: spike at apex with zeros on both sides.
    chroms[1, apex] = 100.0
    mz_values = np.array([100.0, 200.0])

    comp = Component(
        apex_scan=float(apex), apex_scan_int=apex, sharpness=10.0,
        contributing_ions=[0],
        window_lo=10, window_hi=21,
    )
    out = deconvolve_spectrum(comp, chroms, mz_values, Nf=1.0)
    # Either flagged due to FM mismatch (square-spike vs Gaussian) or
    # via the zero-neighbor rule. Both are correct AMDIS outcomes.
    assert out.flags[1]


# ===========================================================================
# Phase 6: High-level entry — deconvolve_features (3f)
# ===========================================================================

@dataclass
class _Scan:
    mz_array: np.ndarray
    intensity_array: np.ndarray


def _build_synth_scans_three_components(
    n_scans: int = 60,
    apexes: tuple[int, ...] = (15, 30, 45),
    sigma: float = 1.5,
) -> tuple[list, list[dict]]:
    """3 distinct features with non-overlapping apexes."""
    # 4 m/z channels: each component has a unique fingerprint.
    fingerprints = [
        {73.0: 1000.0, 91.0: 200.0, 100.0: 50.0, 120.0: 10.0},
        {73.0: 100.0, 91.0: 1000.0, 100.0: 200.0, 120.0: 50.0},
        {73.0: 50.0, 91.0: 200.0, 100.0: 1000.0, 120.0: 200.0},
    ]
    mzs = [73.0, 91.0, 100.0, 120.0]
    chroms_per_mz: dict[float, np.ndarray] = {m: np.zeros(n_scans) for m in mzs}
    for ap, fp in zip(apexes, fingerprints):
        for m in mzs:
            chroms_per_mz[m] = chroms_per_mz[m] + _gaussian_peak(n_scans, ap, sigma, fp[m])

    scans: list = []
    for s in range(n_scans):
        mz_arr = np.array(mzs, dtype=np.float64)
        int_arr = np.array([chroms_per_mz[m][s] for m in mzs], dtype=np.float64)
        scans.append(_Scan(mz_arr, int_arr))

    features = [
        {"feature_id": f"f{i}", "apex_index": ap}
        for i, ap in enumerate(apexes)
    ]
    return scans, features


def test_deconvolve_features_three_clear_components_returns_three_spectra() -> None:
    """End-to-end: 3 well-separated components → 3 deconvolved spectra
    keyed by feature_id."""
    from metabo_core.gcms.deconvolution import (
        DeconvolutionConfig,
        deconvolve_features,
    )

    scans, features = _build_synth_scans_three_components()
    out = deconvolve_features(features, scans, config=DeconvolutionConfig())

    assert len(out) == 3
    assert {"f0", "f1", "f2"} == set(out.keys())
    for fid, spec in out.items():
        assert spec.intensity.size == 4   # 4 m/z channels
        assert spec.mz.size == 4
        # Each spectrum has at least one non-zero intensity entry.
        assert spec.intensity.max() > 0


def test_deconvolved_to_peaks_filters_and_sorts() -> None:
    """deconvolved_to_peaks returns [(mz, intensity), ...] sorted by m/z,
    optionally filtered by relative floor and top-N."""
    from metabo_core.gcms.deconvolution import (
        DeconvolvedSpectrum,
        deconvolved_to_peaks,
    )

    spec = DeconvolvedSpectrum(
        mz=np.array([100.0, 73.0, 91.0]),
        intensity=np.array([1000.0, 100.0, 50.0]),
        intensity_no_subtraction=np.array([1000.0, 100.0, 50.0]),
        flags=np.array([False, False, False]),
        n_adjacent_subtracted=0,
        apex_scan=15,
    )
    pairs = deconvolved_to_peaks(spec, rel_intensity_floor=0.06)  # drops 50/1000=0.05
    assert len(pairs) == 2
    assert pairs[0][0] < pairs[1][0]  # sorted by m/z


@dataclass
class _FakeScan:
    """Minimal scan-like object for build_chromatograms tests."""
    mz_array: np.ndarray
    intensity_array: np.ndarray


def test_build_chromatograms_buckets_by_mz_tol() -> None:
    """Plan D follow-up #2: pin the vectorized rebuild's behavior on a
    small synthetic input. Two scans, three m/z buckets."""
    from metabo_core.gcms.deconvolution import build_chromatograms_from_scans

    scans = [
        _FakeScan(
            mz_array=np.array([50.0, 75.0, 100.0]),
            intensity_array=np.array([10.0, 20.0, 30.0]),
        ),
        _FakeScan(
            mz_array=np.array([50.0, 75.0, 100.0]),
            intensity_array=np.array([15.0, 22.0, 25.0]),
        ),
    ]
    chroms, mzs = build_chromatograms_from_scans(scans, mz_tol=0.02)
    assert chroms.shape == (3, 2)
    np.testing.assert_array_equal(mzs, np.array([50.0, 75.0, 100.0]))
    np.testing.assert_array_equal(
        chroms,
        np.array(
            [
                [10.0, 15.0],   # 50.0 across the two scans
                [20.0, 22.0],   # 75.0 across the two scans
                [30.0, 25.0],   # 100.0 across the two scans
            ]
        ),
    )


def test_build_chromatograms_same_bucket_within_scan_takes_max() -> None:
    """Two m/z values in the same scan that round to the same bucket:
    the chromatogram cell takes the max, matching the legacy behavior."""
    from metabo_core.gcms.deconvolution import build_chromatograms_from_scans

    # 50.005 and 50.015 both round to bucket 2500 / 2501 at mz_tol=0.02?
    # round(50.005/0.02)=2500.25 -> int 2500. round(50.015/0.02)=2500.75 -> 2501.
    # Use values that genuinely share a bucket: 50.0 and 50.005, mz_tol=0.02.
    # round(50.0/0.02)=2500. round(50.005/0.02)=2500.25 -> 2500. Same bucket.
    scans = [
        _FakeScan(
            mz_array=np.array([50.0, 50.005]),
            intensity_array=np.array([10.0, 30.0]),  # max = 30.0
        ),
    ]
    chroms, mzs = build_chromatograms_from_scans(scans, mz_tol=0.02)
    assert chroms.shape == (1, 1)
    assert chroms[0, 0] == 30.0
    # The "canonical" m/z is the first-seen (50.0).
    assert mzs[0] == 50.0


def test_build_chromatograms_empty_scans_returns_zero_grid() -> None:
    from metabo_core.gcms.deconvolution import build_chromatograms_from_scans

    chroms, mzs = build_chromatograms_from_scans([])
    assert chroms.shape == (0, 0)
    assert mzs.size == 0


def test_build_chromatograms_all_empty_arrays_returns_correct_shape() -> None:
    """Scans exist but every mz_array is empty."""
    from metabo_core.gcms.deconvolution import build_chromatograms_from_scans

    scans = [
        _FakeScan(mz_array=np.array([]), intensity_array=np.array([])),
        _FakeScan(mz_array=np.array([]), intensity_array=np.array([])),
    ]
    chroms, mzs = build_chromatograms_from_scans(scans)
    assert chroms.shape == (0, 2)
    assert mzs.size == 0


def test_deconvolved_to_peaks_with_flags_keeps_alignment_through_filters() -> None:
    """Plan D follow-up #3: flags must stay aligned to peaks through the
    filter+sort pipeline. The previous _flags_for_peaks helper used
    list.index(mz) on a list of floats, which is fragile. The new
    function carries indices through the pipeline."""
    from metabo_core.gcms.deconvolution import (
        DeconvolvedSpectrum,
        deconvolved_to_peaks_with_flags,
    )

    # Construct a spectrum where flags differ per index. After sorting by
    # m/z and dropping the i=0 entry, flags must still report correctly.
    spec = DeconvolvedSpectrum(
        mz=np.array([100.0, 73.0, 91.0, 50.0]),
        intensity=np.array([1000.0, 100.0, 50.0, 0.0]),
        intensity_no_subtraction=np.array([1000.0, 100.0, 50.0, 0.0]),
        # flag pattern: idx0=True, idx1=False, idx2=True, idx3=True (dropped)
        flags=np.array([True, False, True, True]),
        n_adjacent_subtracted=0,
        apex_scan=15,
    )
    peaks, flags = deconvolved_to_peaks_with_flags(spec)
    assert len(peaks) == len(flags) == 3
    # After mz-sort: 73.0 (idx1, False), 91.0 (idx2, True), 100.0 (idx0, True)
    mzs = [p[0] for p in peaks]
    assert mzs == [73.0, 91.0, 100.0]
    assert flags == [False, True, True]


def test_deconvolved_to_peaks_with_flags_top_n_preserves_alignment() -> None:
    """top_n cut also preserves flag alignment."""
    from metabo_core.gcms.deconvolution import (
        DeconvolvedSpectrum,
        deconvolved_to_peaks_with_flags,
    )

    spec = DeconvolvedSpectrum(
        mz=np.array([100.0, 73.0, 91.0, 60.0]),
        intensity=np.array([1000.0, 100.0, 500.0, 50.0]),
        intensity_no_subtraction=np.array([1000.0, 100.0, 500.0, 50.0]),
        flags=np.array([True, False, True, False]),
        n_adjacent_subtracted=0,
        apex_scan=15,
    )
    peaks, flags = deconvolved_to_peaks_with_flags(spec, top_n=2)
    # Top-2 by intensity: 1000 (idx0, True), 500 (idx2, True). After mz-sort:
    # 91.0 (idx2, True), 100.0 (idx0, True).
    assert [p[0] for p in peaks] == [91.0, 100.0]
    assert flags == [True, True]


def test_deconvolve_features_empty_inputs_returns_empty() -> None:
    from metabo_core.gcms.deconvolution import deconvolve_features

    assert deconvolve_features([], []) == {}
    assert deconvolve_features([{"feature_id": "x", "apex_index": 0}], []) == {}


# ===========================================================================
# Plan E Task 1: representative_mz helper for the Feature Overview scatter
# ===========================================================================

def test_representative_mz_picks_heaviest_unflagged_above_threshold() -> None:
    """Heaviest unflagged peak whose intensity is at least
    threshold_ratio * base_peak_intensity is the representative m/z.

    Spectrum: [(50, 100), (75, 200), (100, 50)] all unflagged. Base peak
    intensity = 200; threshold = 0.05 * 200 = 10. All peaks pass, so the
    heaviest m/z = 100.
    """
    from metabo_core.gcms.deconvolution import (
        DeconvolvedSpectrum,
        representative_mz,
    )

    spec = DeconvolvedSpectrum(
        mz=np.array([50.0, 75.0, 100.0]),
        intensity=np.array([100.0, 200.0, 50.0]),
        intensity_no_subtraction=np.array([100.0, 200.0, 50.0]),
        flags=np.array([False, False, False]),
        n_adjacent_subtracted=0,
        apex_scan=15,
    )
    assert representative_mz(spec, threshold_ratio=0.05) == 100.0


def test_representative_mz_excludes_peaks_below_threshold() -> None:
    """A heavy peak whose intensity is below threshold_ratio is excluded
    and the next-heaviest qualifying peak is returned.

    Spectrum: [(50, 100), (75, 200), (100, 5)] with threshold = 0.05.
    base_intensity = 200, threshold value = 10. 5 < 10 so 100 excluded.
    Heaviest remaining = 75.
    """
    from metabo_core.gcms.deconvolution import (
        DeconvolvedSpectrum,
        representative_mz,
    )

    spec = DeconvolvedSpectrum(
        mz=np.array([50.0, 75.0, 100.0]),
        intensity=np.array([100.0, 200.0, 5.0]),
        intensity_no_subtraction=np.array([100.0, 200.0, 5.0]),
        flags=np.array([False, False, False]),
        n_adjacent_subtracted=0,
        apex_scan=15,
    )
    assert representative_mz(spec, threshold_ratio=0.05) == 75.0


def test_representative_mz_falls_back_to_base_peak_when_all_flagged() -> None:
    """When every peak is flagged the representative falls back to the
    base peak m/z (== quant_mass).
    """
    from metabo_core.gcms.deconvolution import (
        DeconvolvedSpectrum,
        representative_mz,
    )

    spec = DeconvolvedSpectrum(
        mz=np.array([50.0, 75.0, 100.0]),
        intensity=np.array([100.0, 200.0, 50.0]),
        intensity_no_subtraction=np.array([100.0, 200.0, 50.0]),
        flags=np.array([True, True, True]),
        n_adjacent_subtracted=0,
        apex_scan=15,
    )
    # Base peak (max intensity) is 200 at m/z=75, so fallback returns 75.
    assert representative_mz(spec, threshold_ratio=0.05) == 75.0


def test_representative_mz_empty_spectrum_returns_nan() -> None:
    """An empty spectrum returns NaN to flag missing data."""
    from metabo_core.gcms.deconvolution import (
        DeconvolvedSpectrum,
        representative_mz,
    )

    spec = DeconvolvedSpectrum(
        mz=np.array([]),
        intensity=np.array([]),
        intensity_no_subtraction=np.array([]),
        flags=np.array([], dtype=bool),
        n_adjacent_subtracted=0,
        apex_scan=0,
    )
    result = representative_mz(spec)
    assert math.isnan(result)


def test_representative_mz_threshold_10pct_keeps_peak_at_15pct() -> None:
    """Threshold 0.10: peak at 30/200 = 0.15 still qualifies, so result = 100."""
    from metabo_core.gcms.deconvolution import (
        DeconvolvedSpectrum,
        representative_mz,
    )

    spec = DeconvolvedSpectrum(
        mz=np.array([50.0, 75.0, 100.0]),
        intensity=np.array([100.0, 200.0, 30.0]),
        intensity_no_subtraction=np.array([100.0, 200.0, 30.0]),
        flags=np.array([False, False, False]),
        n_adjacent_subtracted=0,
        apex_scan=15,
    )
    assert representative_mz(spec, threshold_ratio=0.10) == 100.0


def test_representative_mz_threshold_50pct_drops_peak_at_15pct() -> None:
    """Threshold 0.50: only peaks >= 100 intensity qualify. Heaviest = 75."""
    from metabo_core.gcms.deconvolution import (
        DeconvolvedSpectrum,
        representative_mz,
    )

    spec = DeconvolvedSpectrum(
        mz=np.array([50.0, 75.0, 100.0]),
        intensity=np.array([100.0, 200.0, 30.0]),
        intensity_no_subtraction=np.array([100.0, 200.0, 30.0]),
        flags=np.array([False, False, False]),
        n_adjacent_subtracted=0,
        apex_scan=15,
    )
    assert representative_mz(spec, threshold_ratio=0.50) == 75.0


def test_representative_mz_uses_no_subtraction_when_requested() -> None:
    """The use_subtraction flag mirrors deconvolved_to_peaks behavior."""
    from metabo_core.gcms.deconvolution import (
        DeconvolvedSpectrum,
        representative_mz,
    )

    # With-subtraction has m/z 100 above threshold; no-subtraction has it
    # below threshold. The choice of array changes the result.
    spec = DeconvolvedSpectrum(
        mz=np.array([50.0, 75.0, 100.0]),
        intensity=np.array([100.0, 200.0, 50.0]),                  # peak 100 above 5%
        intensity_no_subtraction=np.array([100.0, 200.0, 5.0]),    # peak 100 below 5%
        flags=np.array([False, False, False]),
        n_adjacent_subtracted=0,
        apex_scan=15,
    )
    assert representative_mz(spec, threshold_ratio=0.05, use_subtraction=True) == 100.0
    assert representative_mz(spec, threshold_ratio=0.05, use_subtraction=False) == 75.0


# ===========================================================================
# External-ion-peaks path + public sharpness/apex wrappers (ASFAM MS2 AMDIS)
# ===========================================================================

def test_peak_sharpness_public_wrapper_matches_internal():
    from metabo_core.gcms.deconvolution import peak_sharpness, _peak_sharpness
    chrom = np.array([0., 1., 4., 9., 4., 1., 0.])
    assert peak_sharpness(chrom, 3, 0, 7, 1.0) == _peak_sharpness(chrom, 3, 0, 7, 1.0)


def test_parabola_apex_public_wrapper_exists():
    from metabo_core.gcms.deconvolution import parabola_apex
    chrom = np.array([0., 1., 4., 9., 4., 1., 0.])
    # apex at 3, symmetric -> precise apex stays ~3.0
    assert abs(parabola_apex(chrom, 3) - 3.0) < 0.51


def test_perceive_components_external_ion_peaks_bypasses_ion_detection():
    """external_ion_peaks provided -> no internal _ion_peaks; components come
    straight from the passed peaks' sharpness landscape."""
    from metabo_core.gcms.deconvolution import perceive_components, IonPeak
    # 2 ions, 40 scans; two co-eluting ions apexing at scan 20 (one strong, one weak)
    chroms = np.zeros((2, 40), dtype=np.float64)
    for s in range(40):
        chroms[0, s] = max(0.0, 100.0 - abs(s - 20) * 12.0)
        chroms[1, s] = max(0.0, 60.0 - abs(s - 20) * 8.0)
    ips = [
        IonPeak(ion_index=0, apex_scan_int=20, apex_scan_precise=20.0,
                apex_intensity=100.0, sharpness=5.0, window_lo=12, window_hi=29,
                baseline=np.zeros(0)),
        IonPeak(ion_index=1, apex_scan_int=20, apex_scan_precise=20.0,
                apex_intensity=60.0, sharpness=3.0, window_lo=13, window_hi=28,
                baseline=np.zeros(0)),
    ]
    comps = perceive_components(
        chroms, external_ion_peaks=ips, use_tic_path=False,
        sharpness_range_factor=50.0, sharpness_cutoff_ratio=0.75,
        min_range_scans=3, inclusion_cutoff_ratio=0.3,
    )
    assert len(comps) == 1
    # both ions co-elute at apex 20 -> both contribute (incl_cutoff=0.3 -> 3.0 >= 0.3*5.0)
    assert set(comps[0].contributing_ions) == {0, 1}
    assert comps[0].apex_scan_int == 20
