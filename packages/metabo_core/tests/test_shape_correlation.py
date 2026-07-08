"""Regression tests for cross-EIC shape coherence helper."""
from __future__ import annotations

import numpy as np

from metabo_core.algorithms.shape_correlation import (
    median_pairwise_correlation,
)


def _gaussian(n: int, center: int, sigma: float, amp: float = 1000.0) -> np.ndarray:
    idx = np.arange(n)
    return amp * np.exp(-0.5 * ((idx - center) / sigma) ** 2)


def test_single_eic_returns_one():
    """A single trace is trivially coherent with itself."""
    eic = _gaussian(20, 10, 2.0)
    assert median_pairwise_correlation([eic]) == 1.0


def test_empty_returns_one():
    """No traces -> nothing to compare, treat as coherent (callers
    should also guard against empty clusters separately)."""
    assert median_pairwise_correlation([]) == 1.0


def test_perfect_co_elution_yields_near_one():
    """Three identical-shape Gaussians (only amplitude differs) — the
    archetype of real metabolite co-elution."""
    eics = [
        _gaussian(20, 10, 2.0, amp=1000.0),
        _gaussian(20, 10, 2.0, amp=500.0),
        _gaussian(20, 10, 2.0, amp=2000.0),
    ]
    r = median_pairwise_correlation(eics)
    assert r > 0.999


def test_co_eluting_with_minor_noise_still_high():
    """Real metabolite peaks have small per-EIC noise on top of the
    shared shape; median correlation should still be near 1."""
    rng = np.random.default_rng(0)
    base = _gaussian(50, 25, 3.0, amp=1000.0)
    eics = [base + rng.normal(0, 20, 50) for _ in range(4)]
    r = median_pairwise_correlation(eics)
    assert r > 0.95


def test_independent_noise_yields_near_zero():
    """The diagnostic case: fragment EICs are independent noise (the
    baseline-noise "feature" pattern). Median correlation must be
    near 0 so the caller's threshold (e.g. 0.7) rejects the cluster."""
    rng = np.random.default_rng(42)
    eics = [rng.normal(100, 30, 50) for _ in range(5)]
    r = median_pairwise_correlation(eics)
    assert r < 0.3


def test_one_outlier_among_three_is_caught():
    """Two coherent + one independent: median of the 3 pair-corrs
    drops to the middle (a coherent pair vs two incoherent pairs)
    yielding a value the threshold should still catch."""
    rng = np.random.default_rng(7)
    base = _gaussian(40, 20, 3.0, amp=1000.0)
    coherent_a = base + rng.normal(0, 10, 40)
    coherent_b = base + rng.normal(0, 10, 40)
    outlier = rng.normal(100, 30, 40)
    r = median_pairwise_correlation([coherent_a, coherent_b, outlier])
    # The three pairwise corrs are: (a,b)~0.99, (a,outlier)~0, (b,outlier)~0.
    # Median = 0 (one of the low ones), which is well below 0.7.
    assert r < 0.3


def test_zero_variance_trace_is_excluded():
    """A constant trace has undefined correlation; pairs against it are
    skipped rather than poisoning the median with NaN."""
    flat = np.full(20, 500.0)
    real = _gaussian(20, 10, 2.0, amp=1000.0)
    # If 'flat' weren't excluded, median would be NaN. With it
    # excluded, the only valid pair is (real, real) which doesn't
    # exist; median falls back to 0.0 (no valid pairs).
    r = median_pairwise_correlation([flat, real])
    assert r == 0.0


def test_mixed_with_one_constant_and_two_real():
    """One flat trace + two co-eluting reals: the (real, real) pair
    contributes; the two (flat, real) pairs are excluded; median is
    the one valid value (~1)."""
    flat = np.full(30, 500.0)
    a = _gaussian(30, 15, 2.0, amp=1000.0)
    b = _gaussian(30, 15, 2.0, amp=800.0)
    r = median_pairwise_correlation([flat, a, b])
    assert r > 0.99


def test_anticorrelated_traces_yield_negative():
    """Truly anti-correlated traces produce strongly negative pair
    Pearson; median is negative. Useful as a sanity check that the
    helper is reporting signed correlation rather than its absolute
    value."""
    a = _gaussian(30, 15, 2.0, amp=1000.0)
    b = -a + 2000  # exact anti-correlation
    r = median_pairwise_correlation([a, b])
    assert r < -0.95
