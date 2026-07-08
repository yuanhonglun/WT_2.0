"""AMDIS plan T5: ms2_clustering switch — config default + save/load + routing."""
import numpy as np

from asfam.models import ScanCycle, RawSegmentData
from asfam.config import ProcessingConfig


def _make_min_raw_data(channel=285, n=24):
    """One channel with a clean gaussian pair -> >=1 EIC and >=1 detected peak,
    enough to reach the clustering call in _process_one_file."""
    amp = [0, 0, 0, 400, 1000, 2500, 5000, 8000, 9000, 8000, 5000, 2500, 1000, 400,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0][:n]
    cyc = []
    for i in range(n):
        scans = {}
        if amp[i] > 0:
            scans[channel] = (np.array([100.02, 150.07]),
                              np.array([float(amp[i]), float(amp[i]) * 0.6]))
        cyc.append(ScanCycle(i, 1.0 + 0.05 * i, np.array([]), np.array([]), scans))
    return RawSegmentData("x", "seg", channel, channel + 29, 1, n,
                          np.array([1.0 + 0.05 * i for i in range(n)]),
                          [channel], cyc)


def test_ms2_clustering_defaults_to_rt():
    cfg = ProcessingConfig()
    assert cfg.ms2_clustering == "rt"            # amdis is opt-in; rt unchanged
    assert cfg.ms2_amdis.sharpness_range_factor == 50.0
    assert cfg.ms2_amdis.match_max_window == 12


def test_ms2_amdis_config_survives_save_load(tmp_path):
    from asfam.config import Ms2AmdisConfig
    cfg = ProcessingConfig()
    cfg.ms2_clustering = "amdis"
    cfg.ms2_amdis.sharpness_range_factor = 42.0
    p = tmp_path / "cfg.json"
    cfg.save(p)
    restored = ProcessingConfig.load(p)
    assert restored.ms2_clustering == "amdis"
    # Must be rebuilt as the dataclass, not a plain dict (GUI/CLI project files).
    assert isinstance(restored.ms2_amdis, Ms2AmdisConfig)
    assert restored.ms2_amdis.sharpness_range_factor == 42.0


def test_amdis_switch_routes_to_amdis_clusterer(monkeypatch):
    """ms2_clustering='amdis' -> _process_one_file calls cluster_peaks_amdis,
    NOT cluster_peaks_by_rt."""
    import asfam.pipeline.stage1_ms2_detection as st
    cfg = ProcessingConfig()
    cfg.ms2_clustering = "amdis"
    called = {"amdis": 0, "rt": 0}
    monkeypatch.setattr(st, "cluster_peaks_amdis",
                        lambda *a, **k: (called.__setitem__("amdis", called["amdis"] + 1) or []))
    monkeypatch.setattr(st, "cluster_peaks_by_rt",
                        lambda *a, **k: (called.__setitem__("rt", called["rt"] + 1) or []))
    st._process_one_file(_make_min_raw_data(), cfg)
    assert called["amdis"] >= 1 and called["rt"] == 0


def test_rt_path_is_default_and_calls_rt_clusterer(monkeypatch):
    """Default (rt) still routes to cluster_peaks_by_rt — zero behavior change."""
    import asfam.pipeline.stage1_ms2_detection as st
    cfg = ProcessingConfig()  # default rt
    called = {"amdis": 0, "rt": 0}
    monkeypatch.setattr(st, "cluster_peaks_amdis",
                        lambda *a, **k: (called.__setitem__("amdis", called["amdis"] + 1) or []))
    monkeypatch.setattr(st, "cluster_peaks_by_rt",
                        lambda *a, **k: (called.__setitem__("rt", called["rt"] + 1) or []))
    st._process_one_file(_make_min_raw_data(), cfg)
    assert called["rt"] >= 1 and called["amdis"] == 0
