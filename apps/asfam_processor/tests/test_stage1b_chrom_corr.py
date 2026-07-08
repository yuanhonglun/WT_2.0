"""问题 B: Stage 1b MS2 反卷积 (apex 对齐 + 轻量共流出) 单元测试。

历史: ``_collect_ms2_at_peak`` 先后用过两套门:
  - 最早 "apex 落在 MS1 apex ±3 cycle" 粗判;
  - W8 改成 per-ion 色谱相关性硬门 (Pearson ≥ 0.90, 后放宽到 0.70) +
    10% 基峰预过滤 + 自适应共流出 scan 数门。

问题 B (2026-06-26, 对齐 MS-DIAL DIA 的 MSDec): per-ion Pearson 硬门把真实
但窄/带噪的产品离子砍得太狠 → ms1_driven feature 常 n_frags=0 → 无法谱注释。
MS-DIAL 的 MS2 反卷积根本不用 per-ion 相关性; 它真正的结构判据是
**precursor↔product 的 apex 时间对齐 (≤2 scan)**。故档 B:

  - apex 对齐门 (主判别器): 候选 EIC 的 apex 落在 MS1 apex ±``ms1b_apex_scan_tolerance``
    (默认 2) scan 内才保留;
  - per-ion Pearson 硬门关闭 (阈值默认 0.0, 只挡严格反相关), 可经 config 恢复;
  - 轻量共流出 scan 数门 (floor 2) 挡单点尖刺噪声;
  - 相对振幅预过滤降到 1% (近 0, 对齐 MS-DIAL RelativeAmplitudeCutoff=0%)。

这里用合成 EIC 校验上述分支, 并验证 config 可逆性 (恢复 Pearson 门 / 放宽
apex 门)。
"""
from __future__ import annotations

import numpy as np
import pytest

from asfam.config import ProcessingConfig
from asfam.models import RawSegmentData, ScanCycle
from asfam.pipeline.stage1b_ms1_detection import _collect_ms2_at_peak


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian(rt: np.ndarray, center: float, sigma: float, amp: float) -> np.ndarray:
    """简单高斯峰生成器, 不加噪声 (方便构造确定性测试)。"""
    return amp * np.exp(-0.5 * ((rt - center) / sigma) ** 2)


def _build_raw_segment(
    rt: np.ndarray,
    channel: int,
    ms1_intensity: np.ndarray,
    ms2_ion_table: dict[float, np.ndarray],
) -> RawSegmentData:
    """根据每 cycle 的 MS1 / MS2 强度向量, 拼出 RawSegmentData。

    ms2_ion_table : {产品离子 m/z: 长度 == len(rt) 的强度向量}
    """
    n_cycles = len(rt)
    cycles: list[ScanCycle] = []
    # 每 cycle 的 MS2 谱: 把表中所有离子在该 cycle 的强度收集起来,
    # 强度 > 0 才放入 (避免污染 ion_bins 收集)
    for ci in range(n_cycles):
        # MS1 谱: 单峰 m/z = channel + 0.0001 (落在 ±0.5 Da 内)
        ms1_mz_arr = np.array([channel + 0.0001], dtype=np.float64)
        ms1_int_arr = np.array([float(ms1_intensity[ci])], dtype=np.float64)

        prod_mzs: list[float] = []
        prod_ints: list[float] = []
        for ion_mz, ion_eic in ms2_ion_table.items():
            val = float(ion_eic[ci])
            if val > 0:
                prod_mzs.append(ion_mz)
                prod_ints.append(val)
        ms2_mz_arr = np.array(prod_mzs, dtype=np.float64)
        ms2_int_arr = np.array(prod_ints, dtype=np.float64)
        cycles.append(
            ScanCycle(
                cycle_index=ci,
                rt=float(rt[ci]),
                ms1_mz=ms1_mz_arr,
                ms1_intensity=ms1_int_arr,
                ms2_scans={channel: (ms2_mz_arr, ms2_int_arr)},
            )
        )

    return RawSegmentData(
        file_path="synthetic.mzML",
        segment_name="synthetic_seg",
        segment_low=channel - 1,
        segment_high=channel + 1,
        replicate_id=1,
        n_cycles=n_cycles,
        rt_array=rt.copy(),
        precursor_list=[channel],
        cycles=cycles,
    )


def _default_config() -> ProcessingConfig:
    cfg = ProcessingConfig()
    # 这些用例专测档 B (apex) 路径; 默认现为 msdec (2026-06-29), 故显式设 apex。
    cfg.ms2_deconv = "apex"
    # 把末端的相对强度阈值关掉, 避免噪声/弱峰在最后被二次砍掉,
    # 让本测试只考核反卷积 (apex 对齐 + 预过滤 + 共流出) 逻辑
    cfg.msms_relative_threshold = 0.0
    return cfg


# ---------------------------------------------------------------------------
# 用例 A: apex 对齐保留真实 (含弱) 离子, 剔除 apex 远离的脉冲噪声
# ---------------------------------------------------------------------------

def test_collect_ms2_at_peak_keeps_aligned_incl_weak_drops_apex_displaced_pulse():
    """MS1 高斯峰 (12 cycle 宽) + 3 个产品离子:
      - 离子 1: 与 MS1 同步 (apex 对齐) → 应保留
      - 离子 2: 脉冲落在 cycle 5 (apex 远离 MS1 apex 14) → 应被 apex 门剔除
      - 离子 3: 与 MS1 同步但峰高只有基峰 5% → 旧 10% 预过滤会砍掉它,
        档 B 1% 预过滤下应保留 (这正是 "让真实弱 MS2 回来" 的目标)

    旧行为只留离子 1; 档 B 应留离子 1 + 离子 3 (apex 对齐的真实弱离子)。
    """
    # 30 个 cycle, RT 间隔 ~0.005 min (= 0.3 s, 典型 LC 周期)
    rt = np.arange(30) * 0.005
    channel = 200
    # MS1 高斯, 中心 RT 0.07 (cycle 14), sigma=0.02 → 半宽 ~12 cycle
    ms1_signal = _gaussian(rt, center=0.07, sigma=0.02, amp=10000.0)

    # 离子 1: 与 MS1 完全同步, 不同绝对幅度 (apex 对齐)
    ion1 = _gaussian(rt, center=0.07, sigma=0.02, amp=8000.0)
    # 离子 2: 形状完全不同 — 一个尖锐脉冲落在 cycle 5 (apex 远离 MS1 apex)
    ion2 = np.zeros_like(rt)
    ion2[5] = 7500.0
    ion2[6] = 6000.0
    ion2[4] = 5000.0
    # 离子 3: 与 MS1 同步, 峰高只有基峰的 5% (高于档 B 1% 预过滤门)
    ion3 = _gaussian(rt, center=0.07, sigma=0.02, amp=400.0)  # 400 / 8000 = 5%

    ms2_table = {
        100.0: ion1,
        120.0: ion2,
        140.0: ion3,
    }
    raw = _build_raw_segment(rt, channel, ms1_signal, ms2_table)
    cfg = _default_config()

    # MS1 峰区间: 大约 [8, 20], apex cycle 14
    left_idx, right_idx, apex_idx = 8, 20, 14
    out_mz, out_int = _collect_ms2_at_peak(
        raw, channel, apex_idx, left_idx, right_idx,
        cfg,
        ms1_eic=ms1_signal,
        rt_array=rt,
    )

    # apex 对齐的离子 1 + 弱离子 3 都应保留; apex 远离的离子 2 剔除
    got = sorted(round(m, 1) for m in out_mz.tolist())
    assert got == [100.0, 140.0], (
        f"期望保留 apex 对齐的离子 1+3, 剔除 apex 远离的脉冲离子 2, 实际 {got}"
    )


def test_collect_ms2_at_peak_prefilter_1pct_floor():
    """档 B 预过滤地板降到 1%: 5% 离子保留, 0.5% 离子被预过滤剔除。

    base_max 由强离子撑到 10000:
      - mid 离子 5% (500) ≥ 1% 门 (100) → 保留 (且 apex 对齐)
      - tiny 离子 0.5% (50) < 1% 门 → 被预过滤剔除
    """
    rt = np.arange(30) * 0.005
    channel = 200
    ms1_signal = _gaussian(rt, center=0.07, sigma=0.02, amp=10000.0)

    strong = _gaussian(rt, center=0.07, sigma=0.02, amp=10000.0)  # base_max
    mid = _gaussian(rt, center=0.07, sigma=0.02, amp=500.0)       # 5%
    tiny = _gaussian(rt, center=0.07, sigma=0.02, amp=50.0)       # 0.5%

    raw = _build_raw_segment(
        rt, channel, ms1_signal, {100.0: strong, 130.0: mid, 150.0: tiny}
    )
    cfg = _default_config()
    out_mz, _ = _collect_ms2_at_peak(
        raw, channel, apex_idx=14, left_idx=8, right_idx=20,
        config=cfg,
        ms1_eic=ms1_signal,
        rt_array=rt,
    )
    got = sorted(round(m, 1) for m in out_mz.tolist())
    assert got == [100.0, 130.0], (
        f"期望保留 strong+5%mid, 剔除 0.5%tiny, 实际 {got}"
    )


# ---------------------------------------------------------------------------
# 用例 B: apex 对齐门是主判别器 (Pearson 放宽到 0.0 后仍挡 apex 远离离子)
# ---------------------------------------------------------------------------

def test_collect_ms2_at_peak_apex_gate_rejects_displaced_even_without_pearson():
    """档 B 的核心: 取消 Pearson 硬门 (阈值 0.0) 后, apex 远离 MS1 apex 的
    离子仍被 apex 对齐门挡住——证明判别器从 "相关性" 换成了 "apex 对齐"。

    构造一个与 MS1 同 sigma、但 apex 右移 5 scan (cycle 19) 的高斯离子。它与
    MS1 仍正相关 (过 Pearson 0.0 门) 且共流出点数足 (过 n_corr 门), 故若没有
    apex 门、只把 Pearson 降到 0.0, 它会被错误保留。apex 门 (±2 scan) 应拒它。
    放宽 apex 容差 (设很大) 则应保留——坐实 "是 apex 门在起作用"。
    """
    rt = np.arange(30) * 0.005
    channel = 200
    ms1_signal = _gaussian(rt, center=0.07, sigma=0.02, amp=10000.0)  # apex cycle 14
    # apex 右移 5 scan: 中心 0.095 → cycle 19, |19-14| = 5 > 2
    displaced = _gaussian(rt, center=0.095, sigma=0.02, amp=8000.0)

    raw = _build_raw_segment(rt, channel, ms1_signal, {100.0: displaced})

    # Pearson 门显式关闭 (0.0), 隔离出 apex 门的作用
    cfg = _default_config()
    cfg.ms1b_chrom_corr_threshold = 0.0
    out_mz, _ = _collect_ms2_at_peak(
        raw, channel, apex_idx=14, left_idx=8, right_idx=20,
        config=cfg, ms1_eic=ms1_signal, rt_array=rt,
    )
    assert len(out_mz) == 0, (
        f"apex 右移 5 scan 的离子应被 apex 门 (±2) 拒, 实际 {out_mz.tolist()}"
    )

    # 放宽 apex 容差 → 同一离子应被保留 (证明拒它的正是 apex 门, 而非别的门)
    cfg2 = _default_config()
    cfg2.ms1b_chrom_corr_threshold = 0.0
    cfg2.ms1b_apex_scan_tolerance = 99
    out_mz2, _ = _collect_ms2_at_peak(
        raw, channel, apex_idx=14, left_idx=8, right_idx=20,
        config=cfg2, ms1_eic=ms1_signal, rt_array=rt,
    )
    assert len(out_mz2) == 1 and abs(out_mz2[0] - 100.0) < 0.05, (
        f"放宽 apex 容差后, apex 门是唯一拒它的门, 应保留, 实际 {out_mz2.tolist()}"
    )


def test_collect_ms2_at_peak_keeps_low_correlation_when_apex_aligned():
    """档 B 回归 + 可逆性: 一个 apex 对齐但形状与 MS1 不太像 (Pearson≈0.49,
    低于旧 0.70 硬门) 的产品离子。

    - 档 B 默认 (Pearson 门关, 阈值 0.0): apex 对齐 + 共流出足 → 应保留;
    - 恢复旧 Pearson 门 (阈值 0.70): r≈0.49 < 0.70 → 应被剔。

    这正是 "旧门把真实但形状带噪的离子砍掉" 的回归点, 同时验证 Pearson 门
    经 config 可逆恢复。
    """
    rt = np.arange(30) * 0.005
    channel = 200
    ms1_signal = _gaussian(rt, center=0.07, sigma=0.02, amp=10000.0)  # 平滑宽峰, apex 14

    # apex 在 cycle 14 (值 9000 最大), 但两侧是平台 (7000), 与平滑高斯 MS1
    # 相关性只有 ~0.49 (< 旧 0.70 门, > 0.0)
    flat = np.zeros_like(rt)
    for ci in (11, 12, 13, 15, 16, 17):
        flat[ci] = 7000.0
    flat[14] = 9000.0

    raw = _build_raw_segment(rt, channel, ms1_signal, {100.0: flat})

    # 档 B 默认: 保留
    cfg = _default_config()
    out_mz, _ = _collect_ms2_at_peak(
        raw, channel, apex_idx=14, left_idx=8, right_idx=20,
        config=cfg, ms1_eic=ms1_signal, rt_array=rt,
    )
    assert len(out_mz) == 1 and abs(out_mz[0] - 100.0) < 0.05, (
        f"apex 对齐但低 Pearson 的真实离子, 档 B 应保留, 实际 {out_mz.tolist()}"
    )

    # 恢复旧 Pearson 0.70 硬门: 同一离子应被剔 (可逆性)
    cfg_strict = _default_config()
    cfg_strict.ms1b_chrom_corr_threshold = 0.70
    out_strict, _ = _collect_ms2_at_peak(
        raw, channel, apex_idx=14, left_idx=8, right_idx=20,
        config=cfg_strict, ms1_eic=ms1_signal, rt_array=rt,
    )
    assert len(out_strict) == 0, (
        f"恢复 Pearson 0.70 门后, r≈0.49 的离子应被剔, 实际 {out_strict.tolist()}"
    )


# ---------------------------------------------------------------------------
# 用例 C: 轻量共流出 scan 数门 — apex 对齐但重合点数不够仍剔
# ---------------------------------------------------------------------------

def test_collect_ms2_at_peak_rejects_when_n_correlated_too_low():
    """apex 对齐也救不了 "过窄到共流出点数不足" 的噪声尖峰。

    MS1 峰宽 13 cycle → 自适应门 = max(floor 2, ceil(0.3 × min(prod, 13))) = 4。
    构造一个 apex 落在 MS1 apex (cycle 14)、但仅 3 个 scan 有信号的产品离子:
    apex 门通过 (apex 对齐), 但 n_correlated=3 < 4 → 应被共流出门剔。
    """
    rt = np.arange(30) * 0.005
    channel = 200
    ms1_signal = _gaussian(rt, center=0.07, sigma=0.02, amp=10000.0)

    # 离子: 只在 cycle 13,14,15 有信号 (3 个共流 scan), apex 落在 14
    sparse_ion = np.zeros_like(rt)
    sparse_ion[13] = ms1_signal[13] * 0.9
    sparse_ion[14] = ms1_signal[14] * 0.9
    sparse_ion[15] = ms1_signal[15] * 0.9

    raw = _build_raw_segment(rt, channel, ms1_signal, {100.0: sparse_ion})
    cfg = _default_config()
    out_mz, _ = _collect_ms2_at_peak(
        raw, channel, apex_idx=14, left_idx=8, right_idx=20,
        config=cfg,
        ms1_eic=ms1_signal,
        rt_array=rt,
    )
    assert len(out_mz) == 0, (
        f"3 个共流 scan < 自适应门 4, 应被剔, 实际拿到 {out_mz.tolist()}"
    )


def test_collect_ms2_at_peak_keeps_narrow_coelution_apex_aligned():
    """一个 apex 对齐且 4 个 scan 共流的窄产品离子应保留。

    自适应共流出门 = max(2, ceil(0.3 × 13)) = 4; 离子在 cycle 13-16 有信号
    (4 个共流 scan, apex 落在 14) → 4 ≥ 4 且 apex 对齐 → 保留。ASFAM 每个通道
    都采到了 MS2, 这类 "真实但窄" 的产品离子不该被丢。
    """
    rt = np.arange(30) * 0.005
    channel = 200
    ms1_signal = _gaussian(rt, center=0.07, sigma=0.02, amp=10000.0)

    # apex 对齐 (cycle 14 值最大), cycle 13-16 有信号 (4 个共流 scan)
    narrow_ion = np.zeros_like(rt)
    for ci in (13, 14, 15, 16):
        narrow_ion[ci] = ms1_signal[ci] * 0.9

    raw = _build_raw_segment(rt, channel, ms1_signal, {100.0: narrow_ion})
    cfg = _default_config()
    out_mz, _ = _collect_ms2_at_peak(
        raw, channel, apex_idx=14, left_idx=8, right_idx=20,
        config=cfg, ms1_eic=ms1_signal, rt_array=rt,
    )
    assert len(out_mz) == 1 and abs(out_mz[0] - 100.0) < 0.05, (
        f"apex 对齐的 4-scan 窄共流离子应保留, 实际 {out_mz.tolist()}"
    )


# ---------------------------------------------------------------------------
# 健壮性: 空 MS2 / 单调性
# ---------------------------------------------------------------------------

def test_collect_ms2_at_peak_empty_returns_empty():
    rt = np.arange(20) * 0.005
    channel = 200
    ms1_signal = _gaussian(rt, center=0.05, sigma=0.02, amp=8000.0)
    # 完全没有 MS2 信号
    raw = _build_raw_segment(rt, channel, ms1_signal, {})
    cfg = _default_config()
    out_mz, out_int = _collect_ms2_at_peak(
        raw, channel, apex_idx=10, left_idx=5, right_idx=15,
        config=cfg,
        ms1_eic=ms1_signal,
        rt_array=rt,
    )
    assert len(out_mz) == 0 and len(out_int) == 0


def test_collect_ms2_at_peak_pearson_threshold_monotonic():
    """Pearson 门 (即便档 B 默认关闭, 仍可经 config 开启) 越严 → 留下的离子
    不增多 (单调性 + 可逆性 smoke)。"""
    rt = np.arange(30) * 0.005
    channel = 200
    ms1_signal = _gaussian(rt, center=0.07, sigma=0.02, amp=10000.0)
    # 一个完美相关 + 一个轻微噪声扰动的相关离子 (均 apex 对齐)
    perfect = _gaussian(rt, center=0.07, sigma=0.02, amp=8000.0)
    rng = np.random.default_rng(0)
    noisy = perfect + rng.normal(0, 200, size=perfect.size)
    noisy = np.clip(noisy, 0, None)

    raw = _build_raw_segment(rt, channel, ms1_signal, {100.0: perfect, 110.0: noisy})

    cfg_loose = _default_config()
    cfg_loose.ms1b_chrom_corr_threshold = 0.5
    cfg_strict = _default_config()
    cfg_strict.ms1b_chrom_corr_threshold = 0.99

    loose, _ = _collect_ms2_at_peak(
        raw, channel, apex_idx=14, left_idx=8, right_idx=20,
        config=cfg_loose, ms1_eic=ms1_signal, rt_array=rt,
    )
    strict, _ = _collect_ms2_at_peak(
        raw, channel, apex_idx=14, left_idx=8, right_idx=20,
        config=cfg_strict, ms1_eic=ms1_signal, rt_array=rt,
    )
    assert len(strict) <= len(loose)
