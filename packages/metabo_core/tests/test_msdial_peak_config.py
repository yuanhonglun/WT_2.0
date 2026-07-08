from metabo_core.config.msdial_peak_spotting import MsdialPeakSpottingConfig, lc_msdial_config

def test_defaults_match_msdial_table():
    c = lc_msdial_config()
    assert c.mass_slice_width == 0.1
    assert c.smoothing_level == 3
    assert c.min_data_points == 5
    assert c.min_amplitude == 1000.0
    assert c.centroid_ms1_tolerance == 0.01
    assert c.noise_factor == 3.0
    assert c.amplitude_noise_fold == 4.0
    assert c.slope_noise_fold == 2.0
    assert c.average_peak_width == 20
    assert c.background_spike_threshold == 15
    assert c.noise_bin_size == 50
    assert c.min_noise_windows == 10

def test_factory_returns_fresh_instances():
    assert lc_msdial_config() is not lc_msdial_config()
