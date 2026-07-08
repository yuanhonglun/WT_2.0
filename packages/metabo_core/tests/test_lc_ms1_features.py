# packages/metabo_core/tests/test_lc_ms1_features.py
import numpy as np
import pytest
from metabo_core.algorithms.lc_ms1_features import MS1FeatureHit, find_lc_ms1_features
from metabo_core.algorithms.ms1_eic_roi import ROIConfig
from metabo_core.config.peak_detection import lc_ms1_peak_config


class _FakeScan:
    """Scan-like: 只暴露 finder 需要的属性。"""
    def __init__(self, rt, mz, intensity, ms_level=1):
        self.rt = rt
        self.mz_array = np.asarray(mz, dtype=np.float64)
        self.intensity_array = np.asarray(intensity, dtype=np.float64)
        self.ms_level = ms_level


def test_ms1featurehit_fields():
    hit = MS1FeatureHit(
        mz_centroid=285.05, rt_apex=1.0, rt_left=0.9, rt_right=1.1,
        height=1000.0, area=50.0, sn_ratio=20.0, gaussian_similarity=0.95,
        apex_scan_idx=7,
    )
    assert hit.mz_centroid == 285.05
    assert hit.apex_scan_idx == 7
    assert hit.left_scan_idx == 0
    assert hit.right_scan_idx == 0


def test_empty_scans_returns_empty():
    assert find_lc_ms1_features(
        [], roi_config=ROIConfig(mode="lc_ppm"), peak_config=lc_ms1_peak_config(),
    ) == []


def _gaussian(n, center, width, amp):
    x = np.arange(n)
    return amp * np.exp(-0.5 * ((x - center) / width) ** 2)


def test_single_gaussian_peak_detected():
    # 30 个 MS1 scan, 单一 m/z=285.05, 在 scan 15 处一个高斯峰
    n = 30
    rts = np.linspace(0.0, 2.9, n)
    amps = _gaussian(n, center=15, width=2.0, amp=5000.0)
    scans = []
    for i in range(n):
        if amps[i] > 1.0:
            scans.append(_FakeScan(rts[i], [285.05], [amps[i]]))
        else:
            scans.append(_FakeScan(rts[i], [285.05], [0.0]))
    hits = find_lc_ms1_features(
        scans,
        roi_config=ROIConfig(mode="lc_ppm", ppm_tolerance=15.0, min_eic_points=5),
        peak_config=lc_ms1_peak_config(),
    )
    assert len(hits) == 1
    assert abs(hits[0].mz_centroid - 285.05) < 0.01
    assert 13 <= hits[0].apex_scan_idx <= 17
    assert hits[0].height > 1000.0


def test_two_coisolated_mz_in_same_nominal_resolved():
    # 同一整数通道 285 内两个不同精确质量前体 285.05 / 285.42, 不同 RT
    n = 40
    rts = np.linspace(0.0, 3.9, n)
    a1 = _gaussian(n, 10, 2.0, 5000.0)
    a2 = _gaussian(n, 28, 2.0, 4000.0)
    scans = []
    for i in range(n):
        mzs, ints = [], []
        if a1[i] > 1.0:
            mzs.append(285.05); ints.append(a1[i])
        if a2[i] > 1.0:
            mzs.append(285.42); ints.append(a2[i])
        if not mzs:
            mzs, ints = [285.05], [0.0]
        scans.append(_FakeScan(rts[i], mzs, ints))
    hits = find_lc_ms1_features(
        scans,
        roi_config=ROIConfig(mode="lc_ppm", ppm_tolerance=15.0, min_eic_points=5),
        peak_config=lc_ms1_peak_config(),
    )
    centroids = sorted(round(h.mz_centroid, 2) for h in hits)
    assert centroids == [285.05, 285.42]  # 两个独立 feature, 旧 1 Da 塌缩做不到


def test_mz_range_explicit_window_restricts_search():
    # 两个相距很远的 m/z 信号：200.10 和 500.20
    # 只传 mz_range=(199.0, 201.0)，应仅返回 200.10 的 feature
    n = 30
    rts = np.linspace(0.0, 2.9, n)
    a_low = _gaussian(n, center=15, width=2.0, amp=5000.0)   # m/z=200.10
    a_high = _gaussian(n, center=15, width=2.0, amp=5000.0)  # m/z=500.20
    scans = []
    for i in range(n):
        mzs, ints = [], []
        if a_low[i] > 1.0:
            mzs.append(200.10); ints.append(a_low[i])
        if a_high[i] > 1.0:
            mzs.append(500.20); ints.append(a_high[i])
        if not mzs:
            mzs, ints = [200.10], [0.0]
        scans.append(_FakeScan(rts[i], mzs, ints))
    hits = find_lc_ms1_features(
        scans,
        roi_config=ROIConfig(mode="lc_ppm", ppm_tolerance=15.0, min_eic_points=5),
        peak_config=lc_ms1_peak_config(),
        mz_range=(199.0, 201.0),  # 仅包含 200.10；500.20 应被排除
    )
    assert len(hits) >= 1, "预期至少检出 200.10 的 feature"
    # 所有返回 hit 的 mz 必须在窗口内
    for h in hits:
        assert 199.0 <= h.mz_centroid <= 201.0, (
            f"mz_centroid={h.mz_centroid} 超出 mz_range 窗口"
        )
    # 500.20 的 feature 不得出现
    assert all(abs(h.mz_centroid - 500.20) > 1.0 for h in hits), (
        "500.20 的 feature 不应出现在 mz_range=(199,201) 的结果中"
    )
