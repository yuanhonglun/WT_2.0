"""Regression tests for the per-feature gaussian aggregator."""
from __future__ import annotations

import numpy as np

from metabo_core.algorithms.peak_aggregation import aggregate_feature_gaussian


def test_scalars_takes_minimum():
    assert aggregate_feature_gaussian(0.95, 0.82, 0.97) == 0.82


def test_list_input_flattened():
    assert aggregate_feature_gaussian([0.9, 0.85, 0.92]) == 0.85


def test_mixed_scalar_and_list():
    assert aggregate_feature_gaussian([0.9, 0.85], 0.7) == 0.7


def test_ndarray_input():
    arr = np.array([0.95, 0.88, 0.91])
    assert aggregate_feature_gaussian(arr) == 0.88


def test_none_inputs_ignored():
    assert aggregate_feature_gaussian(None, 0.8, None) == 0.8


def test_empty_returns_zero():
    assert aggregate_feature_gaussian() == 0.0
    assert aggregate_feature_gaussian(None, []) == 0.0


def test_zero_and_nan_are_dropped():
    """Zero gaussian = peak where shape wasn't computed; NaN = degenerate
    trace. Both are filtered before taking the min."""
    assert aggregate_feature_gaussian(0.0, 0.85, float("nan"), 0.9) == 0.85


def test_all_zero_returns_zero():
    """No valid values → 0.0 (calling code can interpret as 'unknown')."""
    assert aggregate_feature_gaussian(0.0, 0.0, None) == 0.0


def test_single_scalar():
    assert aggregate_feature_gaussian(0.73) == 0.73


def test_mixing_with_ms1_and_ms2():
    """Realistic ASFAM usage: aggregate an MS1 peak's gaussian with the
    per-fragment MS2 gaussians from the same feature."""
    ms1_gaussian = 0.91
    ms2_gaussian = np.array([0.88, 0.95, 0.87, 0.93])
    assert aggregate_feature_gaussian(ms1_gaussian, ms2_gaussian) == 0.87
