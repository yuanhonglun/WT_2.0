"""Tests for detect_chrom_peaks router (Task 3.3).

TDD: written before the implementation so the first run fails with
ImportError on ``detect_chrom_peaks``.
"""
import numpy as np
import pytest
from dataclasses import replace

from asfam.config import ProcessingConfig
from asfam.core.peak_detection import detect_peaks, detect_chrom_peaks
from metabo_core.algorithms.msdial_peak_spotting import msdial_detect_peaks_in_chromatogram


# ---------------------------------------------------------------------------
# EIC fixture builder
# ---------------------------------------------------------------------------

def _gaussian_eic(height, sigma_scans, n=120, total_time=3.0, center_frac=0.5):
    """Build a clean Gaussian EIC on a regular RT grid.

    Parameters
    ----------
    height : float
        Peak height (apex intensity).
    sigma_scans : float
        Peak width expressed in number of scans (index space).
    n : int
        Number of data points in the chromatogram.
    total_time : float
        Total RT span in minutes.
    center_frac : float
        Fraction of total_time where the Gaussian apex sits (0.5 → centre).
    """
    rt = np.linspace(0.0, total_time, n, dtype=np.float64)
    center_idx = int(n * center_frac)
    idx = np.arange(n, dtype=np.float64)
    y = height * np.exp(-0.5 * ((idx - center_idx) / sigma_scans) ** 2)
    return rt, y


# ---------------------------------------------------------------------------
# Test 1 — metra route is a pure pass-through
# ---------------------------------------------------------------------------

def test_metra_route_matches_detect_peaks_directly():
    """metra routing is byte-for-byte identical to calling detect_peaks directly.

    With config.peak_detector="metra" (default), detect_chrom_peaks must
    forward ALL kwargs unchanged to detect_peaks and return the same result.
    """
    rt, y = _gaussian_eic(height=10000, sigma_scans=10)  # sigma ≈ 0.25 min
    cfg = replace(ProcessingConfig(), peak_detector="metra")  # default is now msdial
    assert cfg.peak_detector == "metra"

    shared_kwargs = dict(
        min_amplitude=200,
        min_data_points=5,
        compute_gaussian=True,
        gaussian_threshold=0.85,
        min_prominence_ratio=0.0,
        rt_window_min=0.0,
        rt_window_max=2.0,
    )

    routed = detect_chrom_peaks(rt, y, config=cfg, **shared_kwargs)
    direct = detect_peaks(rt, y, **shared_kwargs)

    assert len(routed) == len(direct), (
        f"metra route returned {len(routed)} peaks, direct returned {len(direct)}"
    )
    for r, d in zip(routed, direct):
        assert r.apex_index == d.apex_index, (
            f"apex_index mismatch: routed={r.apex_index}, direct={d.apex_index}"
        )
        assert r.height == pytest.approx(d.height, rel=1e-9), (
            f"height mismatch: routed={r.height}, direct={d.height}"
        )


# ---------------------------------------------------------------------------
# Test 2 — msdial route calls msdial_detect_peaks_in_chromatogram
# ---------------------------------------------------------------------------

def test_msdial_route_uses_msdial_engine():
    """msdial routing delegates to msdial_detect_peaks_in_chromatogram.

    The helper must call the MS-DIAL engine, honour the min_amplitude
    override, and return a non-empty list of DetectedPeak whose apex is
    near the Gaussian centre.  It must also return the exact same result as
    calling msdial_detect_peaks_in_chromatogram directly with the same
    overridden config.
    """
    rt, y = _gaussian_eic(height=10000, sigma_scans=10)
    cfg = replace(ProcessingConfig(), peak_detector="msdial")

    peaks = detect_chrom_peaks(rt, y, config=cfg, min_amplitude=200)

    assert len(peaks) > 0, "msdial route should detect the clean Gaussian peak"
    # Gaussian centred at 1.5 min (center_frac=0.5, total_time=3.0)
    apex_rt = rt[peaks[0].apex_index]
    assert abs(apex_rt - 1.5) < 0.15, (
        f"msdial apex RT {apex_rt:.3f} not within 0.15 min of 1.5"
    )

    # Prove the route equals the direct call with the same overridden config
    expected_cfg = replace(cfg.msdial_peak, min_amplitude=200)
    expected = msdial_detect_peaks_in_chromatogram(rt, y, config=expected_cfg)
    assert len(peaks) == len(expected), (
        f"routed {len(peaks)} peaks, direct {len(expected)} peaks"
    )
    if peaks:
        assert peaks[0].apex_index == expected[0].apex_index, (
            f"apex_index: routed={peaks[0].apex_index}, direct={expected[0].apex_index}"
        )


# ---------------------------------------------------------------------------
# Test 3 — msdial min_amplitude override reaches the engine
# ---------------------------------------------------------------------------

def test_msdial_min_amplitude_override_takes_effect():
    """min_amplitude override reaches the MS-DIAL engine.

    The MS-DIAL config defaults to min_amplitude=1000.  The test builds a
    Gaussian with height=600 (below the default floor).  With the caller
    passing min_amplitude=400 the override must make the engine find the
    peak; with min_amplitude=900 the engine must reject it.

    This proves the override is applied (not silently discarded) and keeps
    the A/B comparison about algorithm, not threshold.
    """
    # sigma=10 scans → peak wide enough for MS-DIAL (>5 scans, many bins).
    # After LWMA(level=3) the smoothed apex ≈ 600 * 0.988 ≈ 593.
    rt, y = _gaussian_eic(height=600, sigma_scans=10)
    cfg = replace(ProcessingConfig(), peak_detector="msdial")

    # min_amplitude=400 should accept a ~593-high peak
    found = detect_chrom_peaks(rt, y, config=cfg, min_amplitude=400)
    assert len(found) > 0, (
        f"min_amplitude=400 should accept a 600-height Gaussian "
        f"(smoothed ≈593); got {len(found)} peaks.  "
        f"max(y)={np.max(y):.1f}"
    )

    # min_amplitude=900 should reject a ~593-high peak
    rejected = detect_chrom_peaks(rt, y, config=cfg, min_amplitude=900)
    assert len(rejected) == 0, (
        f"min_amplitude=900 should reject a 600-height Gaussian "
        f"(smoothed ≈593); got {len(rejected)} peaks"
    )
