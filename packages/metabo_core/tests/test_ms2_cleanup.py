"""单元测试：``metabo_core.algorithms.ms2_cleanup``。

覆盖每个 helper 的边界 + 整体集成 + 性能 baseline。
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from metabo_core.algorithms.ms2_cleanup import (
    MS2CleanupConfig,
    _apply_intensity_threshold,
    _keep_top_n,
    _merge_close_ions,
    _remove_after_precursor,
    _remove_flat_noise,
    clean_ms2_spectrum,
)


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------


def test_merge_close_ions_collapses_near_duplicates():
    """两个相距 < 容差的离子应合并为一个 centroid。"""
    mz = np.array([100.000, 100.003, 200.000], dtype=np.float64)
    intensity = np.array([1000.0, 500.0, 800.0], dtype=np.float64)
    cfg = MS2CleanupConfig(merge_absolute_tol=0.02)
    out_mz, out_int = _merge_close_ions(mz, intensity, precursor_mz=300.0, config=cfg)
    assert len(out_mz) == 2
    # 第一个 centroid 强度加权: (100*1000 + 100.003*500) / 1500 ≈ 100.001
    assert out_mz[0] == pytest.approx(100.001, abs=1e-3)
    # 取强度最大成员: 1000
    assert out_int[0] == 1000.0
    assert out_mz[1] == 200.0
    assert out_int[1] == 800.0


def test_remove_flat_noise_drops_repeated_intensity():
    """3 个相同强度的离子应被全部丢弃，独立强度的离子保留。"""
    mz = np.array([50.0, 60.0, 70.0, 80.0, 90.0], dtype=np.float64)
    intensity = np.array([53.0, 53.0, 53.0, 5000.0, 8000.0], dtype=np.float64)
    out_mz, out_int = _remove_flat_noise(mz, intensity, min_repeats=3)
    assert list(out_mz) == [80.0, 90.0]
    assert list(out_int) == [5000.0, 8000.0]


def test_remove_flat_noise_keeps_when_below_threshold():
    """只有 2 个相同强度（< min_repeats=3），不应被删。"""
    mz = np.array([50.0, 60.0, 70.0], dtype=np.float64)
    intensity = np.array([100.0, 100.0, 200.0], dtype=np.float64)
    out_mz, out_int = _remove_flat_noise(mz, intensity, min_repeats=3)
    assert len(out_mz) == 3


def test_apply_intensity_threshold_uses_max_of_abs_and_rel():
    """绝对阈值与相对阈值之间取大者作为截断线。"""
    mz = np.array([50.0, 60.0, 70.0], dtype=np.float64)
    intensity = np.array([500.0, 5000.0, 50000.0], dtype=np.float64)
    # base = 50000; rel*base = 1000；abs = 1000; cut = 1000
    out_mz, out_int = _apply_intensity_threshold(
        mz, intensity, abs_threshold=1000.0, rel_threshold=0.02,
    )
    assert list(out_mz) == [60.0, 70.0]
    assert list(out_int) == [5000.0, 50000.0]


def test_apply_intensity_threshold_relative_dominates_when_higher():
    """相对阈值高于绝对阈值时应使用相对阈值。"""
    mz = np.array([50.0, 60.0, 70.0], dtype=np.float64)
    intensity = np.array([500.0, 5000.0, 50000.0], dtype=np.float64)
    # base = 50000; rel*base = 5000；abs = 100; cut = 5000
    out_mz, out_int = _apply_intensity_threshold(
        mz, intensity, abs_threshold=100.0, rel_threshold=0.1,
    )
    assert list(out_mz) == [60.0, 70.0]


def test_remove_after_precursor_drops_high_mz():
    """precursor + isotope_range 之后的离子应被删除。"""
    mz = np.array([100.0, 250.0, 270.0, 280.0], dtype=np.float64)
    intensity = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)
    out_mz, out_int = _remove_after_precursor(
        mz, intensity, precursor_mz=270.0, kept_isotope_range_da=1.5,
    )
    # cutoff = 270 + 1.5 = 271.5
    assert list(out_mz) == [100.0, 250.0, 270.0]
    assert list(out_int) == [100.0, 200.0, 300.0]


def test_remove_after_precursor_skips_when_precursor_unknown():
    mz = np.array([100.0, 500.0], dtype=np.float64)
    intensity = np.array([100.0, 200.0], dtype=np.float64)
    out_mz, _ = _remove_after_precursor(
        mz, intensity, precursor_mz=None, kept_isotope_range_da=1.5,
    )
    assert len(out_mz) == 2


def test_keep_top_n_drops_lowest():
    mz = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
    intensity = np.array([100.0, 500.0, 50.0, 200.0, 800.0], dtype=np.float64)
    out_mz, out_int = _keep_top_n(mz, intensity, top_n=3)
    assert len(out_mz) == 3
    # 前 3 强：800, 500, 200 → mz 50, 20, 40（顺序可能乱，由 argpartition 决定）
    assert set(out_int.tolist()) == {800.0, 500.0, 200.0}


def test_keep_top_n_noop_when_n_zero_or_negative():
    mz = np.array([10.0, 20.0], dtype=np.float64)
    intensity = np.array([100.0, 500.0], dtype=np.float64)
    out_mz, _ = _keep_top_n(mz, intensity, top_n=0)
    assert len(out_mz) == 2


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_clean_ms2_spectrum_full_pipeline():
    """构造含平顶噪声 + 弱峰 + precursor 后离子的谱图，验证 6 步清洗。

    输入设计：
      - precursor_mz = 300
      - 5 个强度 = 53 的平顶噪声离子（应被 flat_noise 步骤删）
      - 一个 50 强度的弱峰（应被 abs 阈值 1000 删）
      - 一个 100 强度的弱峰（应被 abs 阈值 1000 删）
      - 一个 mz = 305 的高强度离子（应被 remove_after_precursor 删）
      - 真正的高强度信号：5 个 1000–50000 的 ion
    """
    mz = np.array([
        50.0, 60.0, 70.0, 80.0, 90.0,    # flat noise (53 × 5)
        100.0,                            # 弱峰 50
        110.0,                            # 弱峰 100
        120.0, 130.0, 140.0, 150.0, 160.0,  # 高强度真信号
        305.0,                            # precursor 后离子
    ], dtype=np.float64)
    intensity = np.array([
        53.0, 53.0, 53.0, 53.0, 53.0,
        50.0,
        100.0,
        5000.0, 8000.0, 12000.0, 30000.0, 50000.0,
        20000.0,
    ], dtype=np.float64)

    cfg = MS2CleanupConfig()  # ASFAM 默认
    out_mz, out_int = clean_ms2_spectrum(mz, intensity, precursor_mz=300.0, config=cfg)

    # 平顶 5 个 + 弱峰 2 个 + precursor 后 1 个 = 8 个被删；剩 5 个
    assert len(out_mz) == 5
    assert list(out_mz) == [120.0, 130.0, 140.0, 150.0, 160.0]
    assert list(out_int) == [5000.0, 8000.0, 12000.0, 30000.0, 50000.0]


def test_clean_ms2_spectrum_returns_sorted_by_mz():
    """输出必须按 m/z 升序，即使输入乱序。"""
    mz = np.array([300.0, 100.0, 200.0], dtype=np.float64)
    intensity = np.array([10000.0, 5000.0, 8000.0], dtype=np.float64)
    cfg = MS2CleanupConfig(remove_after_precursor=False)
    out_mz, _ = clean_ms2_spectrum(mz, intensity, precursor_mz=400.0, config=cfg)
    assert list(out_mz) == sorted(out_mz)


def test_clean_ms2_spectrum_handles_empty_input():
    mz = np.array([], dtype=np.float64)
    intensity = np.array([], dtype=np.float64)
    cfg = MS2CleanupConfig()
    out_mz, out_int = clean_ms2_spectrum(mz, intensity, precursor_mz=300.0, config=cfg)
    assert out_mz.size == 0
    assert out_int.size == 0


def test_clean_ms2_spectrum_extra_arrays_stay_aligned():
    """``extra_arrays`` (例如 S/N) 必须随清洗过程保持索引对齐。"""
    mz = np.array([50.0, 100.0, 150.0, 200.0, 305.0], dtype=np.float64)
    intensity = np.array([5000.0, 8000.0, 12000.0, 30000.0, 20000.0], dtype=np.float64)
    sn = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    cfg = MS2CleanupConfig()
    out = clean_ms2_spectrum(
        mz, intensity, precursor_mz=300.0, config=cfg, extra_arrays=[sn],
    )
    assert len(out) == 3  # mz, intensity, sn
    out_mz, out_int, out_sn = out
    # precursor 后那个 305 被删；剩下 4 个，S/N 仍按 m/z 升序对齐
    assert list(out_mz) == [50.0, 100.0, 150.0, 200.0]
    assert list(out_sn) == [1.0, 2.0, 3.0, 4.0]


def test_clean_ms2_spectrum_no_precursor_skips_precursor_step():
    """precursor_mz=None 时不做 remove_after_precursor。"""
    mz = np.array([100.0, 200.0, 500.0], dtype=np.float64)
    intensity = np.array([5000.0, 8000.0, 12000.0], dtype=np.float64)
    cfg = MS2CleanupConfig()
    out_mz, _ = clean_ms2_spectrum(mz, intensity, precursor_mz=None, config=cfg)
    assert len(out_mz) == 3


def test_clean_ms2_spectrum_top_n_enforced():
    """top_n 应保留最强的 N 个。"""
    mz = np.arange(50.0, 250.0, 1.0)  # 200 个
    intensity = np.linspace(2000.0, 100000.0, 200)
    cfg = MS2CleanupConfig(top_n=20, remove_after_precursor=False)
    out_mz, out_int = clean_ms2_spectrum(mz, intensity, precursor_mz=300.0, config=cfg)
    assert len(out_mz) == 20
    # 最弱也应该≥ rel threshold（2% of 100000 = 2000）
    assert float(out_int.min()) >= 2000.0


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_clean_ms2_spectrum_performance_1000_ions():
    """1000 个离子谱清洗应在 5 ms 内完成（向量化预算）。"""
    rng = np.random.default_rng(42)
    mz = np.sort(rng.uniform(50.0, 1000.0, size=1000))
    intensity = rng.uniform(100.0, 100000.0, size=1000)
    cfg = MS2CleanupConfig()

    # warm-up（JIT / cache）
    clean_ms2_spectrum(mz, intensity, precursor_mz=500.0, config=cfg)

    t0 = time.perf_counter()
    for _ in range(10):
        clean_ms2_spectrum(mz, intensity, precursor_mz=500.0, config=cfg)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / 10.0

    # 5 ms 预算
    assert elapsed_ms < 5.0, f"average cleanup took {elapsed_ms:.2f} ms (> 5 ms budget)"
