"""Tests for compute_peak_shape_scores — faithful port of the MS-DIAL
PeakPick.cs GetPeakDetectionResult ShapnessValue / IdealSlopeValue.

Reference: Common/CommonStandard/Algorithm/PeakPick/PeakPick.cs L463-615.
Only the two scores MSDec needs (sharpness for model-peak ranking,
ideal-slope for High/Middle/Low quality) are ported; they depend solely on
the intensity column and the peak-internal index distance.
"""
from __future__ import annotations

import numpy as np
import pytest

from metabo_core.algorithms.peak_shape import compute_peak_shape_scores


def test_perfect_triangle_peak():
    # Symmetric triangle: every slope step is +1/-1, I[top]=4, sqrt(4)=2.
    # leftShapeness = rightShapeness = 0.5 → ShapnessValue = 0.5.
    # All slopes ideal → IdealSlopeValue = 1.0.
    intensity = np.array([0, 1, 2, 3, 4, 3, 2, 1, 0], dtype=np.float64)
    scores = compute_peak_shape_scores(intensity, left_id=0, top_id=4, right_id=8)
    assert scores is not None
    assert scores.shapeness == pytest.approx(0.5)
    assert scores.ideal_slope == pytest.approx(1.0)


def test_nonideal_peak_drops_ideal_slope():
    # One dip on each flank: left ideal=5/nonideal=1, right ideal=5/nonideal=1
    # → IdealSlopeValue = (10-2)/10 = 0.8.
    # Sharpest single step is (4-1)/1/2 = 1.5 on each side → ShapnessValue=1.5.
    intensity = np.array([0, 2, 1, 4, 1, 2, 0], dtype=np.float64)
    scores = compute_peak_shape_scores(intensity, left_id=0, top_id=3, right_id=6)
    assert scores is not None
    assert scores.shapeness == pytest.approx(1.5)
    assert scores.ideal_slope == pytest.approx(0.8)


def test_ideal_slope_clamped_at_zero():
    # Mostly-falling left flank makes (ideal-nonideal) negative → clamp to 0.
    # left: j=2 I3-I2=4-3=1 ideal; j=1 I2-I1=3-1=2 ideal; j=0 I1-I0=1-10=-9 nonideal
    #   left ideal=3 nonideal=9
    # right: j=4 I3-I4=4-0=4 ideal; j=5 I4-I5=0-0=0 ideal; j=6 I5-I6=0-0=0 ideal
    #   right ideal=4 -> total ideal=7 nonideal=9 -> (7-9)/7 <0 -> 0
    intensity = np.array([10, 1, 3, 4, 0, 0, 0], dtype=np.float64)
    scores = compute_peak_shape_scores(intensity, left_id=0, top_id=3, right_id=6)
    assert scores is not None
    assert scores.ideal_slope == 0.0


def test_too_few_points_returns_none():
    # datapoints.Count <= 3 → MS-DIAL returns null → we return None.
    intensity = np.array([0, 1, 2], dtype=np.float64)
    assert compute_peak_shape_scores(intensity, left_id=0, top_id=1, right_id=2) is None


def test_apex_below_both_edges_returns_none():
    # datapoints[top]-datapoints[0] < 0 AND datapoints[top]-datapoints[last] < 0
    # → MS-DIAL returns null.
    intensity = np.array([5, 3, 1, 3, 5], dtype=np.float64)
    assert compute_peak_shape_scores(intensity, left_id=0, top_id=2, right_id=4) is None


def test_window_offset_into_longer_array():
    # left_id/top_id/right_id index into a longer EIC; the same triangle as
    # the first test embedded with padding must give identical scores.
    intensity = np.array([99, 99, 0, 1, 2, 3, 4, 3, 2, 1, 0, 88], dtype=np.float64)
    scores = compute_peak_shape_scores(intensity, left_id=2, top_id=6, right_id=10)
    assert scores is not None
    assert scores.shapeness == pytest.approx(0.5)
    assert scores.ideal_slope == pytest.approx(1.0)
