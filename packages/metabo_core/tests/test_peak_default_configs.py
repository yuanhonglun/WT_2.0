"""Regression tests for the three named PeakDetectionConfig factories.

These three factories — ``lc_ms1_peak_config``, ``lc_ms2_peak_config``,
``gc_peak_config`` — are the single source of truth for the platform's
peak-detection defaults. ASFAM / DDA / GC-MS app configs all import
them, so the field values pinned here are effectively a cross-app
contract.
"""
from __future__ import annotations

from metabo_core.config import (
    PeakDetectionConfig,
    gc_peak_config,
    lc_ms1_peak_config,
    lc_ms2_peak_config,
)


def test_lc_ms1_peak_config_pinned_defaults() -> None:
    cfg = lc_ms1_peak_config()
    assert isinstance(cfg, PeakDetectionConfig)
    assert cfg.min_amplitude == 500.0
    assert cfg.min_data_points == 3
    assert cfg.smooth_window == 1
    assert cfg.baseline_window == 20
    assert cfg.noise_bin_size == 50
    assert cfg.noise_factor == 3.0
    assert cfg.sn_fold == 4.0
    # Bumped 0.75 -> 0.85 on 2026-05-15 after rice ASFAM noise feedback.
    assert cfg.gaussian_threshold == 0.85
    # 4th prominence gate, ratio-based; LC default 0.3.
    assert cfg.min_prominence_ratio == 0.3
    # RT window is user-set per-run via GUI; defaults disabled.
    assert cfg.rt_window_min is None
    assert cfg.rt_window_max is None


def test_lc_ms2_peak_config_pinned_defaults() -> None:
    cfg = lc_ms2_peak_config()
    assert isinstance(cfg, PeakDetectionConfig)
    # MS2 amp floor is intentionally lower than MS1 — product-ion EICs
    # carry less response than precursor EICs.
    assert cfg.min_amplitude == 200.0
    assert cfg.min_data_points == 3
    assert cfg.smooth_window == 1
    assert cfg.baseline_window == 20
    assert cfg.noise_bin_size == 50
    assert cfg.noise_factor == 3.0
    assert cfg.sn_fold == 4.0
    assert cfg.gaussian_threshold == 0.85
    assert cfg.min_prominence_ratio == 0.3
    assert cfg.rt_window_min is None
    assert cfg.rt_window_max is None


def test_gc_peak_config_pinned_defaults() -> None:
    cfg = gc_peak_config()
    assert isinstance(cfg, PeakDetectionConfig)
    # GC defaults diverge from LC for chromatographic-physics reasons —
    # see ``gc_peak_config`` docstring for the rationale.
    assert cfg.min_amplitude == 1000.0
    assert cfg.min_data_points == 5
    assert cfg.smooth_window == 5
    assert cfg.baseline_window == 100
    assert cfg.noise_bin_size == 50
    assert cfg.noise_factor == 5.0
    assert cfg.sn_fold == 5.0
    assert cfg.gaussian_threshold == 0.85
    # GC peaks are sharp + symmetric, so the ratio gate can be tighter.
    assert cfg.min_prominence_ratio == 0.5
    assert cfg.rt_window_min is None
    assert cfg.rt_window_max is None


def test_factories_return_independent_instances() -> None:
    """Each factory call must return a fresh dataclass so callers cannot
    accidentally mutate a shared default."""
    a = lc_ms1_peak_config()
    b = lc_ms1_peak_config()
    assert a is not b
    a.min_amplitude = 9999.0
    assert b.min_amplitude == 500.0


def test_lc_ms1_vs_ms2_amplitude_floors_differ() -> None:
    assert lc_ms1_peak_config().min_amplitude > lc_ms2_peak_config().min_amplitude


def test_gc_differs_structurally_from_lc() -> None:
    """The GC defaults must NOT be a copy of the LC defaults; the
    smoothing / baseline / SN parameters were chosen specifically for
    GC peak physics. (LC and GC now share gaussian_threshold=0.85 and
    edge_skip_scans=3 after the 2026-05-15 LC noise tuning, so those
    no longer differ — physics-driven knobs still do.)"""
    gc = gc_peak_config()
    lc = lc_ms1_peak_config()
    assert gc.smooth_window != lc.smooth_window
    assert gc.baseline_window != lc.baseline_window
    assert gc.sn_fold != lc.sn_fold
    assert gc.min_data_points != lc.min_data_points
