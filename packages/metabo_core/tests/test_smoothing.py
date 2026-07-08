"""Regression tests for shared smoothing helpers."""
import numpy as np

from metabo_core.algorithms.smoothing import smooth_eic


def test_smooth_eic_savgol_preserves_shape_and_nonnegative():
    intensity = np.array([0.0, 1.0, 5.0, 10.0, 5.0, 1.0, 0.0])
    result = smooth_eic(intensity, method="savgol", window_length=5, polyorder=2)
    assert result.shape == intensity.shape
    assert (result >= 0).all()


def test_smooth_eic_short_input_returned_unchanged():
    intensity = np.array([1.0, 2.0])
    result = smooth_eic(intensity, method="savgol", window_length=5, polyorder=2)
    np.testing.assert_array_equal(result, intensity)


def test_smooth_eic_method_none_passes_through():
    intensity = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = smooth_eic(intensity, method="none")
    np.testing.assert_array_equal(result, intensity)


def test_smooth_eic_unknown_method_raises():
    intensity = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    try:
        smooth_eic(intensity, method="banana")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown smoothing method")
