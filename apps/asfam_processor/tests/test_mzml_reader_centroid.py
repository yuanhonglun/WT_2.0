import numpy as np
from asfam.io.mzml_reader import _centroid_if_needed


def _sparse_centroid_scan():
    # 25 个稀疏 centroid 峰（相邻间距 ~1-2 Da），含 F00981 复现里的“93”真值
    # 与其低邻居——两者是不同离子，绝不能被 3 点加权均值混成 ~92.6。
    mz = np.array([
        54.937, 57.108, 59.304, 65.083, 69.608, 72.830, 77.250, 79.273,
        81.035, 83.059, 85.10, 88.20, 91.054, 93.033, 95.482, 104.876,
        107.538, 110.916, 115.736, 120.962, 125.067, 128.170, 137.642,
        149.023, 208.886,
    ], dtype=np.float64)
    inten = np.array([
        6, 96, 7, 721, 9, 37, 6, 17, 11, 4, 5, 5, 102, 329, 12, 16,
        18, 77, 3, 279, 5, 18, 6, 8519, 165,
    ], dtype=np.float64)
    return mz, inten


def test_sparse_centroid_scan_preserves_mz_no_cross_ion_blend():
    """≥20 点但已是 centroid：必须原样直通，不得把 93.033 拉向 92.6。"""
    mz, inten = _sparse_centroid_scan()
    assert len(mz) >= 20  # would have hit the old buggy branch
    out_mz, out_int = _centroid_if_needed(mz, inten, min_intensity=1.0)
    # 每个输入 m/z 都原样保留（强度全 >= 1）
    assert np.allclose(np.sort(out_mz), np.sort(mz), atol=1e-9)
    # 真值 93.033 仍在、且没有出现被混平均的 ~92.6
    assert np.any(np.abs(out_mz - 93.033) < 1e-6)
    assert not np.any(np.abs(out_mz - 92.60) < 0.05)


def test_centroid_intensity_mask_drops_below_min():
    mz, inten = _sparse_centroid_scan()
    inten = inten.copy(); inten[0] = 0.0  # 54.937 强度置 0
    out_mz, _ = _centroid_if_needed(mz, inten, min_intensity=1.0)
    assert not np.any(np.abs(out_mz - 54.937) < 1e-6)


def test_profile_peak_centroided_within_peak_only():
    """密集 profile 单峰（间距 ~0.005 Da）→ 输出一个峰内加权质心 ≈ 峰顶，
    不与另一个相隔 >gap 的峰混合。"""
    # 峰 A: 中心 100.00，三角强度；峰 B: 中心 100.50（相隔 0.5 Da，独立离子）
    a_mz = np.arange(99.980, 100.021, 0.005)
    a_int = np.array([10, 40, 90, 150, 90, 40, 10, 5, 2][:len(a_mz)], dtype=np.float64)
    b_mz = np.arange(100.480, 100.521, 0.005)
    b_int = np.array([5, 30, 80, 120, 80, 30, 5, 2, 1][:len(b_mz)], dtype=np.float64)
    mz = np.concatenate([a_mz, b_mz]); inten = np.concatenate([a_int, b_int])
    # 18 dense points (each peak sampled ~9x @ 0.005 Da). With <20 points the
    # OLD code passes through untouched (returns 18 ions); the fixed code sees a
    # tiny median gap -> profile -> per-peak centroid -> exactly 2 ions. That
    # contrast is what makes this a real red/green (see fix design 2026-07-01).
    assert len(mz) >= 18
    out_mz, out_int = _centroid_if_needed(mz, inten, min_intensity=1.0)
    # 两个独立峰 → 两个质心，各自落在自己峰内、绝不合成 ~100.25
    assert len(out_mz) == 2
    assert abs(out_mz[0] - 100.00) < 0.01
    assert abs(out_mz[1] - 100.50) < 0.01


def test_fewer_than_three_points_passthrough():
    mz = np.array([200.0, 201.0], dtype=np.float64)
    inten = np.array([5.0, 9.0], dtype=np.float64)
    out_mz, out_int = _centroid_if_needed(mz, inten)
    assert np.array_equal(out_mz, mz) and np.array_equal(out_int, inten)
