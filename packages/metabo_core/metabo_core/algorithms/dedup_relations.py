"""统一的同位素 / 加合物 / ISF 关系判定底座。

W6 重构: ASFAM 与 DDA 的 EIC 共流出判定历史上是三套不同实现 (Stage 4 没用,
Stage 5 用 Pearson + 无 n_correlated 门, Stage 6 用 Pearson + 硬限 10
scan)。这里把它们合并成同一套规则:

    Pearson r ≥ ``pearson_threshold``
  且
    n_correlated_points ≥ max(5, 0.5 × min(peak_width_a_scans, peak_width_b_scans))

n_correlated 的自适应公式以两条 EIC 中较窄峰宽 (scan 数) 的一半为参考,
保证宽峰不会被人为放宽 (峰越宽要求重叠点数也越多)、窄峰不会被人为收紧
(地板值 5 scans 防止极窄峰被误丢)。

另外提供 ``apex_rt_strict_from_ms1_cycles``: 严格 apex RT 容差应当只用
MS1 周期间隔的中位数, 不能混入 MS2 scan 把"周期时间"拉长。这是 ASFAM
Stage 4 历史 bug 的修复。
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np

from metabo_core.algorithms.similarity import eic_pearson_in_range


__all__ = [
    "adaptive_n_correlated_threshold",
    "eic_coelution_ok",
    "apex_rt_strict_from_ms1_cycles",
]


def adaptive_n_correlated_threshold(
    peak_width_a_scans: int,
    peak_width_b_scans: int,
    floor: int = 5,
    fraction: float = 0.5,
) -> int:
    """计算两条 EIC 共流出验证所需的最小重叠点数。

    公式: ``max(floor, ceil(fraction × min(peak_width_a, peak_width_b)))``。

    - ``peak_width_*_scans`` 是峰跨越的 scan 数 (不是分钟); 调用方需要
      自行从 ``rt_left/rt_right`` 配合 RT 序列换算, 例如
      ``np.searchsorted`` 出来的区间长度。
    - 地板 5 scans 防止极窄峰被误丢; 0.5 倍较窄峰宽对应"至少半个峰
      重叠才算共流出", 比硬限 10 更适合多种 LC 梯度。
    """
    a = max(0, int(peak_width_a_scans))
    b = max(0, int(peak_width_b_scans))
    smaller = min(a, b)
    # 用 ceil(fraction × smaller) 而非 round, 保证宽峰要求更严
    threshold = int(np.ceil(fraction * smaller))
    return max(int(floor), threshold)


def eic_coelution_ok(
    eic_a: np.ndarray,
    eic_b: np.ndarray,
    rt_array: np.ndarray,
    rt_start: float,
    rt_end: float,
    peak_width_a_scans: int,
    peak_width_b_scans: int,
    pearson_threshold: float = 0.9,
    floor: int = 5,
    fraction: float = 0.5,
) -> bool:
    """两条 EIC 是否在给定 RT 范围内共流出。

    判定:
      1. Pearson r ≥ ``pearson_threshold``
      2. n_correlated_points ≥ ``adaptive_n_correlated_threshold``

    任一不满足返回 False。

    Parameters
    ----------
    eic_a, eic_b : np.ndarray
        与 ``rt_array`` 同长度的强度向量。
    rt_array : np.ndarray
        所有 scan 的 RT。``eic_pearson_in_range`` 会在内部按
        ``rt_start/rt_end`` 截窗。
    rt_start, rt_end : float
        共流出验证使用的 RT 区间 (min)。建议是两个峰的合并边界, 额外
        留 0.05~0.1 min padding。
    peak_width_*_scans : int
        各自 EIC 主峰的宽度 (scan 数)。
    """
    r, n_corr = eic_pearson_in_range(eic_a, eic_b, rt_array, rt_start, rt_end)
    if r < pearson_threshold:
        return False
    needed = adaptive_n_correlated_threshold(
        peak_width_a_scans, peak_width_b_scans,
        floor=floor, fraction=fraction,
    )
    return n_corr >= needed


def apex_rt_strict_from_ms1_cycles(
    data_by_replicate: Optional[dict],
    n_cycles: int = 2,
    fallback_min: float = 0.04,
) -> float:
    """基于 MS1 周期 (非全部 scan) 的中位间隔, 算出严格 apex-RT 容差。

    历史问题: ASFAM Stage 4 旧实现用 ``np.diff(raw.rt_array)`` 取全部
    scan 间隔的中位, 但 ``rt_array`` 里既包含 MS1 也包含每周期的 MS2,
    所以"周期时间"被人为拉短到了 MS2 间隔的量级, 让 apex_rt_strict 比
    它该有的值小一截。

    正确做法: 每个周期 (``ScanCycle``) 只取它的 ``rt`` 作为 MS1 时刻,
    然后对相邻周期 MS1 时刻的间隔求中位数; 这才是一个真正的 MS1->MS1
    周期。容差 = ``n_cycles × cycle_time``。

    Parameters
    ----------
    data_by_replicate : dict | None
        ``{replicate_id: [RawSegmentData, ...]}``; 每个 ``RawSegmentData``
        的 ``cycles`` 属性是 ``ScanCycle`` 列表, 每个 cycle 有 ``rt`` 字段。
        兼容: 若数据结构里没有 ``cycles`` 但有 ``rt_array``, 会退回到
        旧版"全 scan 中位 diff", 保证调用方迁移期不中断。
    n_cycles : int
        容差 = n_cycles × 中位周期; 推荐 2 (~2 个 MS1 周期)。
    fallback_min : float
        无数据可用时的兜底容差 (min)。

    Returns
    -------
    float
        apex RT 容差 (min)。
    """
    n_cycles = max(1, int(n_cycles))
    fallback = float(fallback_min)
    if not data_by_replicate:
        return fallback

    cycle_times: list[float] = []
    for segments in data_by_replicate.values():
        for raw in segments:
            # 首选: 用 cycle.rt 之间的间隔, 这是干净的 MS1 周期
            cycles: Optional[Iterable[Any]] = getattr(raw, "cycles", None)
            if cycles:
                try:
                    rts = np.asarray(
                        [float(c.rt) for c in cycles], dtype=np.float64
                    )
                except Exception:
                    rts = None
                if rts is not None and rts.size >= 2:
                    diffs = np.diff(rts)
                    diffs = diffs[diffs > 0]
                    if diffs.size:
                        cycle_times.append(float(np.median(diffs)))
                        continue
            # 兜底: 用 rt_array 全 scan 中位 diff (旧行为, 偏小但比 fallback 准)
            try:
                rt = np.asarray(raw.rt_array, dtype=np.float64)
                if rt.size >= 2:
                    diffs = np.diff(rt)
                    diffs = diffs[diffs > 0]
                    if diffs.size:
                        cycle_times.append(float(np.median(diffs)))
            except Exception:
                continue

    if not cycle_times:
        return fallback
    cycle_time = float(np.median(cycle_times))
    return n_cycles * cycle_time
