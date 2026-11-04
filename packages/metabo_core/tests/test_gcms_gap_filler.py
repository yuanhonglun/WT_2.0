"""Tests for the MS-DIAL-equivalent GC-MS gap filler."""
from __future__ import annotations

import numpy as np
import pytest

from metabo_core.alignment.gcms_gap_filler import (
    NO_SIGNAL,
    _trapezoid,
    fill_peak,
    needs_fill,
)


def _rt(n: int = 200, step: float = 0.01) -> np.ndarray:
    return np.arange(n, dtype=np.float64) * step


def _gaussian(rt: np.ndarray, center: float, height: float, width: float = 0.05):
    return height * np.exp(-0.5 * ((rt - center) / width) ** 2)


# ---------------------------------------------------------------------------
# needs_fill
# ---------------------------------------------------------------------------

def test_needs_fill_for_an_empty_cell() -> None:
    assert needs_fill(None, 43.0, 0.5) is True


def test_needs_fill_for_a_detected_cell_on_a_different_ion() -> None:
    """The half of ``NeedsGapFill`` that is easy to miss: a *detected* cell
    reported on another ion still has to be re-integrated on the row's ion."""
    assert needs_fill(71.0, 43.0, 0.5) is True


def test_no_fill_when_the_cell_already_uses_the_row_ion() -> None:
    assert needs_fill(43.0, 43.0, 0.5) is False
    # Centroid jitter around one nominal fragment is the same ion.
    assert needs_fill(43.02, 42.98, 0.5) is False


# ---------------------------------------------------------------------------
# fill_peak
# ---------------------------------------------------------------------------

def test_fills_a_clean_peak_and_puts_the_apex_on_it() -> None:
    rt = _rt()
    y = _gaussian(rt, center=1.00, height=10_000.0)
    r = fill_peak(rt, y, center=1.00, ax_tolerance=0.1)
    assert r.found and not r.forced
    assert r.rt_apex == pytest.approx(1.00, abs=0.011)
    assert r.height == pytest.approx(10_000.0, rel=0.01)
    assert r.left_index < r.apex_index < r.right_index
    assert r.area > 0


def test_area_is_in_seconds_like_detect_peaks() -> None:
    """A filled value lands in the same column as detected ones and gets
    averaged with them, so it must use the detector's unit convention: the
    baseline-corrected trace integrated over RT*60."""
    rt = _rt()
    y = _gaussian(rt, center=1.00, height=10_000.0, width=0.05)
    r = fill_peak(rt, y, center=1.00, ax_tolerance=0.1)
    expected = _trapezoid(
        y[r.left_index:r.right_index + 1],
        rt[r.left_index:r.right_index + 1] * 60.0,
    )
    assert r.area == pytest.approx(float(expected))
    # MS-DIAL's "area > height" sanity check for a real peak.
    assert r.area > r.height


def test_picks_the_peak_nearest_the_centre_not_the_tallest() -> None:
    """Two peaks in the window: the row's retention centre decides, because the
    tallest one may belong to a different compound."""
    rt = _rt()
    y = _gaussian(rt, 0.95, 3_000.0, 0.02) + _gaussian(rt, 1.06, 30_000.0, 0.02)
    r = fill_peak(rt, y, center=0.95, ax_tolerance=0.08)
    assert r.found
    assert r.rt_apex == pytest.approx(0.95, abs=0.015)


def test_no_peak_top_in_range_with_force_insert_returns_a_forced_value() -> None:
    """A monotone trace has no local maximum, so the shape test finds nothing
    and ``force_insert`` takes the point nearest the centre."""
    rt = _rt()
    y = 100.0 + np.arange(rt.size, dtype=np.float64) * 0.1     # strictly rising
    r = fill_peak(rt, y, center=1.00, ax_tolerance=0.1, force_insert=True)
    assert r.found and r.forced
    assert r.rt_apex == pytest.approx(1.00, abs=0.011)


def test_noise_usually_does_have_a_peak_top() -> None:
    """Worth pinning so nobody reads ``forced=False`` as "this is a real peak":
    baseline noise contains local maxima that pass the 5-point shape test, so
    the non-forced path fires on noise too. What separates signal from noise is
    the height, which the caller compares against the row's other cells — the
    gap filler itself does not judge."""
    rng = np.random.default_rng(0)
    rt = _rt()
    y = rng.uniform(0.0, 5.0, size=rt.size)
    r = fill_peak(rt, y, center=1.00, ax_tolerance=0.1, force_insert=True)
    assert r.found and not r.forced
    assert r.height < 10.0


def test_flat_trace_without_force_insert_is_no_signal() -> None:
    rt = _rt()
    y = np.zeros_like(rt)
    r = fill_peak(rt, y, center=1.00, ax_tolerance=0.1, force_insert=False)
    assert r == NO_SIGNAL
    assert r.height == 0.0 and r.area == 0.0


def test_empty_or_mismatched_input_is_no_signal() -> None:
    assert fill_peak([], [], center=1.0, ax_tolerance=0.1) == NO_SIGNAL
    assert fill_peak([1.0, 2.0], [1.0], center=1.0, ax_tolerance=0.1) == NO_SIGNAL


def test_peak_outside_the_tolerance_window_is_not_used() -> None:
    """A real peak well outside ``center +- ax_tol`` must not be claimed; with
    force_insert the result is a forced value at the centre instead."""
    rt = _rt()
    y = _gaussian(rt, center=1.60, height=50_000.0, width=0.03)
    r = fill_peak(rt, y, center=1.00, ax_tolerance=0.1, force_insert=True)
    assert r.found and r.forced
    assert r.rt_apex == pytest.approx(1.00, abs=0.1)
    assert r.height < 1.0                       # nothing there, as it should be
    # And with force_insert off, the honest answer is "no signal".
    assert fill_peak(rt, y, 1.00, 0.1, force_insert=False) == NO_SIGNAL


def test_shoulder_does_not_swallow_the_neighbouring_peak() -> None:
    """Boundary walking stops at the valley, so the integrated range covers one
    peak and not its neighbour."""
    rt = _rt()
    y = _gaussian(rt, 1.00, 10_000.0, 0.02) + _gaussian(rt, 1.20, 10_000.0, 0.02)
    r = fill_peak(rt, y, center=1.00, ax_tolerance=0.05)
    assert r.found
    assert rt[r.right_index] < 1.20
