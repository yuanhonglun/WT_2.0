"""Kovats RT↔RI conversion tests."""
from __future__ import annotations

import math

import pytest

from metabo_core.algorithms.retention_index import (
    kovats_ri_to_rt,
    kovats_rt_to_ri,
    kovats_warn_too_few,
)


CALIB = [(5.0, 800.0), (8.0, 1000.0), (12.0, 1200.0)]


def test_rt_to_ri_at_calibration_point_is_exact():
    assert kovats_rt_to_ri(5.0, CALIB) == pytest.approx(800.0, abs=1e-9)
    assert kovats_rt_to_ri(8.0, CALIB) == pytest.approx(1000.0, abs=1e-9)
    assert kovats_rt_to_ri(12.0, CALIB) == pytest.approx(1200.0, abs=1e-9)


def test_rt_to_ri_interpolates_between_pair():
    # Midpoint between (5, 800) and (8, 1000) -> rt=6.5, ri=900
    assert kovats_rt_to_ri(6.5, CALIB) == pytest.approx(900.0, abs=1e-9)


def test_rt_to_ri_extrapolates_above_max():
    # Slope between last pair (8, 1000) -> (12, 1200) is 50 RI/min.
    # rt=14 is 2 min past the max, so RI = 1200 + 2*50 = 1300.
    assert kovats_rt_to_ri(14.0, CALIB) == pytest.approx(1300.0, abs=1e-9)


def test_rt_to_ri_extrapolates_below_min():
    # Slope between first pair (5, 800) -> (8, 1000) is 200/3 RI/min ≈ 66.67.
    # rt=4 is 1 min below min, so RI = 800 - 66.67 ≈ 733.33.
    expected = 800.0 - 200.0 / 3.0
    assert kovats_rt_to_ri(4.0, CALIB) == pytest.approx(expected, abs=1e-6)


def test_ri_to_rt_round_trip():
    for rt_in in (5.5, 7.0, 10.0):
        ri = kovats_rt_to_ri(rt_in, CALIB)
        rt_back = kovats_ri_to_rt(ri, CALIB)
        assert rt_back == pytest.approx(rt_in, abs=1e-6)


def test_two_point_calibration_works():
    """Robustness: 2 points are the minimum and must work."""
    two = [(5.0, 800.0), (10.0, 1000.0)]
    assert kovats_rt_to_ri(7.5, two) == pytest.approx(900.0, abs=1e-9)
    assert kovats_rt_to_ri(12.0, two) == pytest.approx(1080.0, abs=1e-9)


def test_one_point_raises():
    with pytest.raises(ValueError, match="at least 2"):
        kovats_rt_to_ri(5.0, [(5.0, 800.0)])


def test_zero_points_raises():
    with pytest.raises(ValueError, match="at least 2"):
        kovats_rt_to_ri(5.0, [])


def test_nan_points_dropped_before_count():
    points = [(5.0, 800.0), (float("nan"), 900.0), (8.0, 1000.0)]
    assert kovats_rt_to_ri(6.5, points) == pytest.approx(900.0, abs=1e-9)


def test_unsorted_input_handled():
    """Input order should not matter — internal sort by RT."""
    shuffled = [(12.0, 1200.0), (5.0, 800.0), (8.0, 1000.0)]
    assert kovats_rt_to_ri(6.5, shuffled) == pytest.approx(900.0, abs=1e-9)


def test_nan_query_returns_nan():
    out = kovats_rt_to_ri(float("nan"), CALIB)
    assert math.isnan(out)


def test_warn_too_few_messaging():
    assert kovats_warn_too_few([(5.0, 800.0)]) is not None
    assert "≥2" in kovats_warn_too_few([(5.0, 800.0)])
    assert kovats_warn_too_few([]) is not None
    # 2-5 standards triggers warning
    msg = kovats_warn_too_few([(5.0, 800.0), (8.0, 1000.0)])
    assert msg is not None and "Linear extrapolation" in msg
    msg5 = kovats_warn_too_few([(5.0, 800.0), (6.0, 900.0), (7.0, 1000.0),
                                (8.0, 1100.0), (9.0, 1200.0)])
    assert msg5 is not None
    # 6+ standards is silent
    msg6 = kovats_warn_too_few([(5.0 + i, 800.0 + 100 * i) for i in range(6)])
    assert msg6 is None
