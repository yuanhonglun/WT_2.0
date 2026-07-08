"""Tests for MsdecConfig — MS-DIAL MSDec parameter surface.

Defaults reproduce the MS-DIAL ``ChromDecBaseParameter`` constants
(ParameterBase.cs ~L1207) and the MSDec engine hardcoded values
(MSDecHandler.cs / MSDecProcess.cs), so the ported engine produces
results directly comparable to MS-DIAL without user tuning.

The faithful MS-DIAL model-peak amplitude floor is 1000; the ASFAM app
view overrides it to a lower value to retain weak MS2 (see asfam config).
"""
from __future__ import annotations

import pytest

from metabo_core.config.msdec import MsdecConfig, lc_msdec_config


def test_lc_msdec_config_returns_msdec_config():
    assert isinstance(lc_msdec_config(), MsdecConfig)


def test_matched_filter_defaults():
    cfg = lc_msdec_config()
    assert cfg.sigma_window == 0.5            # ParameterBase.cs:1212 SigmaWindowValue
    assert cfg.matched_filter_half_point == 10  # MSDecHandler.cs:1462 (kernel len 21)
    assert cfg.region_margin == 5             # MSDecHandler.cs:1418


def test_ms2_curation_defaults():
    cfg = lc_msdec_config()
    assert cfg.centroid_ms2_tolerance == 0.025   # ParameterBase.cs:1118
    assert cfg.amplitude_cutoff == 0.0           # ParameterBase.cs:1214
    assert cfg.relative_amplitude_cutoff == 0.0  # ParameterBase.cs:1228
    assert cfg.kept_isotope_range == 5.0         # ParameterBase.cs:1218
    assert cfg.remove_after_precursor is True    # ParameterBase.cs:1220


def test_model_peak_detection_defaults():
    cfg = lc_msdec_config()
    assert cfg.min_data_points == 5      # ParameterBase.cs:1100 MinimumDatapoints
    assert cfg.min_amplitude == 1000.0   # ParameterBase.cs:1098 (MS-DIAL faithful)
    assert cfg.smoothing_level == 3      # ParameterBase.cs:1096 (LWMA level)
    assert cfg.average_peak_width == 30  # ParameterBase.cs:1216 (edge walk *0.5=15)


def test_model_selection_defaults():
    cfg = lc_msdec_config()
    assert cfg.sharpness_inclusion_fraction == 0.9  # MSDecHandler.cs:975
    assert cfg.ideal_slope_high == 0.999            # MSDecHandler.cs:827
    assert cfg.ideal_slope_middle == 0.9            # MSDecHandler.cs:829
    assert cfg.apex_model_tolerance == 2            # MSDecHandler.cs:719 (<=2 scan gate)
    assert cfg.max_neighbor_models == 4             # MSDecHandler.cs:1138-1141 (2L2R)
    assert cfg.model_min_edge_points == 3           # MSDecHandler.cs:1288-1289


def test_msdec_config_is_mutable():
    # Mirror MsdialPeakSpottingConfig (not frozen) so the ASFAM view can
    # ``replace()``/assign min_amplitude to a weaker-signal floor.
    cfg = lc_msdec_config()
    cfg.min_amplitude = 500.0
    assert cfg.min_amplitude == 500.0
