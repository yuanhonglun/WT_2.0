"""W6 共享 dedup 底座的单元测试。

覆盖三个公共 API:

- ``adaptive_n_correlated_threshold``: max(5, 0.5 × min(peak_width))
- ``eic_coelution_ok``: Pearson + n_correlated 双门
- ``apex_rt_strict_from_ms1_cycles``: 用 cycle.rt 而不是 rt_array
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from metabo_core.algorithms.dedup_relations import (
    adaptive_n_correlated_threshold,
    apex_rt_strict_from_ms1_cycles,
    eic_coelution_ok,
)


# ---------------------------------------------------------------------------
# adaptive_n_correlated_threshold
# ---------------------------------------------------------------------------


def test_adaptive_threshold_small_peaks_hit_floor():
    """min(10, 12) × 0.5 = 5 -> 地板恰好等于 fraction 结果, 返回 5。"""
    assert adaptive_n_correlated_threshold(10, 12) == 5


def test_adaptive_threshold_wide_peaks_use_fraction():
    """min(30, 20) × 0.5 = 10 -> 超过地板 5, 返回 10。"""
    assert adaptive_n_correlated_threshold(30, 20) == 10


def test_adaptive_threshold_very_narrow_peak_uses_floor():
    """很窄的峰 (2 scan) 也必须用地板 5, 不会被打到 1。"""
    assert adaptive_n_correlated_threshold(2, 100) == 5


def test_adaptive_threshold_ceil_not_round():
    """min(11, 100) × 0.5 = 5.5 -> ceil 后 6 (round 会成 5/6 看实现, 我们要 6)。"""
    assert adaptive_n_correlated_threshold(11, 100) == 6


# ---------------------------------------------------------------------------
# eic_coelution_ok
# ---------------------------------------------------------------------------


def _make_gaussian(n_scans: int, apex: int, sigma: float = 2.0) -> np.ndarray:
    x = np.arange(n_scans, dtype=np.float64)
    return np.exp(-0.5 * ((x - apex) / sigma) ** 2)


def test_coelution_two_identical_peaks_pass():
    """两条相同形状的 EIC: Pearson ~ 1, n_correlated 足够 -> True。"""
    n = 40
    rt = np.linspace(0.0, 4.0, n)
    eic_a = _make_gaussian(n, apex=20, sigma=3.0)
    eic_b = _make_gaussian(n, apex=20, sigma=3.0)
    ok = eic_coelution_ok(
        eic_a, eic_b, rt, rt_start=0.0, rt_end=4.0,
        peak_width_a_scans=12, peak_width_b_scans=12,
    )
    assert ok is True


def test_coelution_different_shapes_fail():
    """形状完全不同 (一个是 Gauss, 一个是反向斜坡) -> Pearson 低 -> False。"""
    n = 40
    rt = np.linspace(0.0, 4.0, n)
    eic_a = _make_gaussian(n, apex=10, sigma=3.0)
    eic_b = _make_gaussian(n, apex=30, sigma=3.0)
    ok = eic_coelution_ok(
        eic_a, eic_b, rt, rt_start=0.0, rt_end=4.0,
        peak_width_a_scans=12, peak_width_b_scans=12,
    )
    assert ok is False


def test_coelution_too_few_overlapping_scans_fails():
    """形状一致但只有 3 个 scan 同时非零 -> n_correlated 不足 -> False。

    这里两条 EIC 只在 3 个 scan 上同时 > 0; 其余位置一条是 0、一条不是。
    n_correlated_threshold 至少 5 (地板), 所以应失败。
    """
    n = 40
    rt = np.linspace(0.0, 4.0, n)
    eic_a = np.zeros(n)
    eic_b = np.zeros(n)
    # 让两条只在 3 个相邻 scan 都有信号 (其余各自单独有信号), Pearson 高
    # 但 n_correlated = 3 < floor 5。
    for i in [18, 19, 20]:
        eic_a[i] = 100.0 + i  # 给些变化, Pearson 才不会因常数 0 输出 0
        eic_b[i] = 200.0 + 2 * i
    # 给其他位置加单边信号, 避免 std 为 0
    eic_a[5] = 50.0
    eic_b[35] = 50.0
    ok = eic_coelution_ok(
        eic_a, eic_b, rt, rt_start=0.0, rt_end=4.0,
        peak_width_a_scans=10, peak_width_b_scans=10,
    )
    assert ok is False


def test_coelution_high_pearson_but_n_below_adaptive_fails():
    """峰宽 20 -> 自适应门要求 ≥ 10 重叠点; 给 7 个重叠 -> 应失败。"""
    n = 60
    rt = np.linspace(0.0, 6.0, n)
    eic_a = np.zeros(n)
    eic_b = np.zeros(n)
    # 让 7 个 scan 都有相关信号 (Pearson 高), 但小于 adaptive (0.5×20 = 10)
    for k, i in enumerate(range(25, 32)):
        eic_a[i] = 100.0 + k
        eic_b[i] = 50.0 + 0.5 * k
    # 单边信号撑大 std
    eic_a[5] = 200.0
    eic_b[55] = 200.0
    ok = eic_coelution_ok(
        eic_a, eic_b, rt, rt_start=0.0, rt_end=6.0,
        peak_width_a_scans=20, peak_width_b_scans=20,
    )
    assert ok is False


# ---------------------------------------------------------------------------
# apex_rt_strict_from_ms1_cycles
# ---------------------------------------------------------------------------


@dataclass
class _FakeCycle:
    rt: float


@dataclass
class _FakeRaw:
    cycles: list
    rt_array: np.ndarray


def test_apex_rt_strict_uses_cycle_rt_only():
    """有 ``cycles`` 属性时, 用 cycle.rt 间隔, 忽略 rt_array 里被 MS2 拉短的间隔。

    构造: 4 个 MS1 周期, 每周期间隔 0.5 min。如果错误地用整个 rt_array
    (含 MS2) 的 diff, 中位会变成 0.1 min, 容差会被算小 5 倍。
    """
    cycles = [_FakeCycle(rt=0.0), _FakeCycle(rt=0.5),
              _FakeCycle(rt=1.0), _FakeCycle(rt=1.5)]
    # rt_array 含 MS1 + 4 个 MS2 每周期, 间隔 0.1 min
    rt_array = np.arange(0.0, 2.0, 0.1)
    raw = _FakeRaw(cycles=cycles, rt_array=rt_array)
    data_by_replicate = {"rep1": [raw]}
    tol = apex_rt_strict_from_ms1_cycles(data_by_replicate, n_cycles=2)
    # 2 × median(0.5) = 1.0
    assert tol == pytest.approx(1.0, abs=1e-6)


def test_apex_rt_strict_fallback_when_empty():
    """data 为空 -> 直接返回 fallback。"""
    assert apex_rt_strict_from_ms1_cycles({}, fallback_min=0.07) == pytest.approx(0.07)
    assert apex_rt_strict_from_ms1_cycles(None, fallback_min=0.07) == pytest.approx(0.07)


def test_apex_rt_strict_n_cycles_multiplier():
    cycles = [_FakeCycle(rt=0.0), _FakeCycle(rt=0.2), _FakeCycle(rt=0.4)]
    raw = _FakeRaw(cycles=cycles, rt_array=np.array([0.0, 0.2, 0.4]))
    tol = apex_rt_strict_from_ms1_cycles({"r": [raw]}, n_cycles=3)
    # 3 × median(0.2) = 0.6
    assert tol == pytest.approx(0.6, abs=1e-9)


def test_apex_rt_strict_fallback_to_rt_array_when_no_cycles():
    """没有 cycles 属性时退回到 rt_array 全 scan 中位 diff。"""
    raw = _FakeRaw(cycles=[], rt_array=np.array([0.0, 0.1, 0.2, 0.3]))
    tol = apex_rt_strict_from_ms1_cycles({"r": [raw]}, n_cycles=2)
    # 2 × median(0.1) = 0.2
    assert tol == pytest.approx(0.2, abs=1e-9)
