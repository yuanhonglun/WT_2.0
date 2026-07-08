"""PR-C: Stage 1b mass-slice MS1 finder + ≥2 MS2 gate removal (Q3) 单元测试。

PR-C 把 stage1b 的 MS1 驱动内核从"每整数通道一条 1 Da EIC, 取 max 塌缩"
换成在全分辨 MS1 survey 上做精细 m/z mass-slice 找峰
(``find_lc_ms1_features``)。两个核心行为变化:

  1. **mass-slice 分辨共隔离前体**: 落在同一 1 Da 整数通道内的两个不同
     精确质量前体 (例如 285.05 / 285.42) 现在各自成为一个 feature; 旧
     "1 Da 塌缩"内核只会产出 1 个 feature。
  2. **移除 ≥2 MS2 强制门 (Q3)**: 弱/无 MS2 的 feature 不再被丢弃, 而是
     用 ``ms2_quality`` 标注 ("correlated" / "sparse" / "none")。

测试拆成两类确定性场景 (见模块内每个用例的 docstring 说明为什么这样拆):

  - Test A: 同 apex 共流出的两前体 → 验证 mass-slice 分辨 + scan-index 对齐
  - Test B/C: 单前体, MS2 反卷积结果 < 2 → 验证 Q3 门移除 (sparse / none)

合成数据沿用 ``test_stage1b_chrom_corr.py`` 的高斯构造约定, 让峰能通过
默认 ``ms1_peak`` 的质量门 (gaussian / prominence / S/N / min_data_points)。
"""
from __future__ import annotations

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import RawSegmentData, ScanCycle
from asfam.pipeline.stage1b_ms1_detection import run_stage1b


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian(rt: np.ndarray, center: float, sigma: float, amp: float) -> np.ndarray:
    """简单高斯峰生成器, 不加噪声 (方便构造确定性测试)。"""
    return amp * np.exp(-0.5 * ((rt - center) / sigma) ** 2)


def _build_segment(
    rt: np.ndarray,
    ms1_ions: list[tuple[float, np.ndarray]],
    channel: int,
    ms2_ion_table: dict[float, np.ndarray],
    precursor_list: list[int] | None = None,
    replicate_id: int = 1,
) -> RawSegmentData:
    """根据每 cycle 的 MS1 / MS2 强度向量拼出 RawSegmentData。

    与 ``test_stage1b_chrom_corr._build_raw_segment`` 的区别: 这里允许
    **每 cycle 注入多个 MS1 m/z** (mass-slice 测试需要在同一整数通道里放
    两个精确质量前体), 而 chrom_corr 的 builder 只注入单个 MS1 m/z。

    Parameters
    ----------
    ms1_ions : list[(m/z, 每 cycle 强度向量)]
        MS1 survey 谱; 每个 cycle 收集所有强度 > 0 的离子。
    channel : int
        该 segment 的 1 Da 隔离通道 (MS2 都挂在这个通道下)。
    ms2_ion_table : {产品离子 m/z: 每 cycle 强度向量}
        ``ms2_scans[channel]`` 的内容; 强度 > 0 才放入 (避免污染 ion_bins)。
    """
    n_cycles = len(rt)
    cycles: list[ScanCycle] = []
    for ci in range(n_cycles):
        # MS1 survey: 收集该 cycle 所有 > 0 的前体离子
        ms1_mzs: list[float] = []
        ms1_ints: list[float] = []
        for mz, eic in ms1_ions:
            val = float(eic[ci])
            if val > 0:
                ms1_mzs.append(mz)
                ms1_ints.append(val)
        ms1_mz_arr = np.array(ms1_mzs, dtype=np.float64)
        ms1_int_arr = np.array(ms1_ints, dtype=np.float64)

        # MS2 谱 (单通道)
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

    if precursor_list is None:
        precursor_list = [channel]
    return RawSegmentData(
        file_path="synthetic.mzML",
        segment_name="synthetic_seg",
        segment_low=channel - 1,
        segment_high=channel + 1,
        replicate_id=replicate_id,
        n_cycles=n_cycles,
        rt_array=rt.copy(),
        precursor_list=precursor_list,
        cycles=cycles,
    )


def _default_config() -> ProcessingConfig:
    cfg = ProcessingConfig()
    # 关掉末端相对强度阈值, 避免弱 MS2 在最后被二次砍掉,
    # 让本测试只考核 mass-slice 找峰 + Q3 门移除逻辑
    cfg.msms_relative_threshold = 0.0
    return cfg


def _run_stage1b(raw: RawSegmentData, cfg: ProcessingConfig) -> list:
    """走真实入口 ``run_stage1b`` (existing_features 为空 → 全是新 MS1 驱动)。"""
    rep = str(raw.replicate_id)
    out = run_stage1b(
        data_by_replicate={rep: [raw]},
        existing_features={rep: []},
        config=cfg,
    )
    return out.get(rep, [])


# ---------------------------------------------------------------------------
# Test A: mass-slice 分辨共隔离前体 + scan-index 对齐
# ---------------------------------------------------------------------------

def test_massslice_resolves_coisolated_precursors():
    """同一整数通道 285 内两个精确质量前体 285.05 / 285.42, **同 apex 共流出**。

    旧 1 Da 塌缩内核: 对通道 285 抽一条 285±0.5 的 EIC, 取 max 后两前体
    叠成一条单峰 → 只产出 1 个 feature, 精确 m/z 是两者强度加权质心
    (≈285.13), 既不是 285.05 也不是 285.42。

    新 mass-slice 内核: 两前体的 ppm 切片彼此远离 (相距 0.37 Da ≫ 15 ppm
    切片宽 ≈0.004 Da) → 分辨成 2 个独立 feature, 精确 m/z 分别 ≈285.05 /
    285.42。

    scan-index 对齐: 强前体的 MS2 是从 ``hit.apex/left/right_scan_idx``
    (= cycle 索引, 因 ms1_survey_scans 与 cycles 1:1 对齐) 对应的隔离窗
    cycle 里取的。若索引坐标系错位, 这些共流出 MS2 离子就取不到, 强前体
    也就不会是 "correlated" / n_correlated_ms2 ≥ 2。所以该断言即对齐证明。

    注: 两前体同 apex 共流出 → MS1 EIC 形状几乎一致, 同一窗口下的 MS2
    会同时与两者相关 (ASFAM 1 Da DIA 的固有局限), 因此这里**不**断言弱
    前体是 sparse; 弱前体的 MS2 质量交给 Q3 门移除的专项用例 (Test B/C)。
    """
    rt = np.arange(30) * 0.005
    channel = 285
    # 两前体同 apex (center=0.07 → cycle 14), 不同强度
    strong_ms1 = _gaussian(rt, center=0.07, sigma=0.02, amp=10000.0)
    weak_ms1 = _gaussian(rt, center=0.07, sigma=0.02, amp=3000.0)
    # 同窗口下 2 个共流出 MS2 产品离子 (与 MS1 同步)
    frag_a = _gaussian(rt, center=0.07, sigma=0.02, amp=5000.0)
    frag_b = _gaussian(rt, center=0.07, sigma=0.02, amp=4000.0)

    raw = _build_segment(
        rt,
        ms1_ions=[(285.05, strong_ms1), (285.42, weak_ms1)],
        channel=channel,
        ms2_ion_table={100.0: frag_a, 150.0: frag_b},
    )
    cfg = _default_config()
    feats = _run_stage1b(raw, cfg)

    # (a) mass-slice 分辨出 2 个独立 feature (旧 1 Da 塌缩只会有 1 个)
    assert len(feats) == 2, (
        f"期望 mass-slice 分辨出 2 个共隔离前体, 实际 {len(feats)} 个: "
        f"{[round(f.ms1_precursor_mz, 3) for f in feats]}"
    )
    feats = sorted(feats, key=lambda f: f.ms1_precursor_mz)
    assert abs(feats[0].ms1_precursor_mz - 285.05) < 0.05
    assert abs(feats[1].ms1_precursor_mz - 285.42) < 0.05
    # 两个 feature 都挂在整数通道 285, 都是 MS1 驱动
    for f in feats:
        assert f.precursor_mz_nominal == 285
        assert f.detection_source == "ms1_driven"
        assert f.mz_source == "ms1_peak"
        assert f.signal_type == "ms1_detected"

    # (b) 强前体 (285.05) 的 MS2 是从正确 cycle 取的 → correlated, n≥2
    strong = feats[0]
    assert strong.ms1_height >= feats[1].ms1_height  # 285.05 是强峰
    assert strong.ms2_quality == "correlated", (
        f"强前体应为 correlated, 实际 {strong.ms2_quality!r}"
    )
    assert strong.n_correlated_ms2 >= 2
    assert strong.n_fragments >= 2
    # apex RT 应落在合成峰中心附近 (= scan-index/RT 对齐的间接证明)
    assert abs(strong.rt_apex - 0.07) < 0.02


# ---------------------------------------------------------------------------
# Test B: 移除 ≥2 MS2 强制门 (Q3) — 保留只有 1 个相关 MS2 的 feature
# ---------------------------------------------------------------------------

def test_drops_ge2_gate_keeps_low_ms2_feature():
    """单前体 300.10 (通道 300), MS2 反卷积只产出 **1 个** 相关离子。

    旧内核在 ``if n_frags < config.min_fragments_per_feature: continue``
    处会直接丢弃该 feature (默认 min_fragments_per_feature=2)。

    新内核 (Q3 移除) 保留它, 并标注 ms2_quality="sparse" (n_correlated_ms2=1)。
    """
    rt = np.arange(30) * 0.005
    channel = 300
    ms1 = _gaussian(rt, center=0.07, sigma=0.02, amp=10000.0)
    # 单个共流出 MS2 离子 → 反卷积后 n_frags == 1
    one_frag = _gaussian(rt, center=0.07, sigma=0.02, amp=8000.0)

    raw = _build_segment(
        rt,
        ms1_ions=[(300.10, ms1)],
        channel=channel,
        ms2_ion_table={120.0: one_frag},
    )
    cfg = _default_config()
    feats = _run_stage1b(raw, cfg)

    assert len(feats) == 1, (
        f"≥2 MS2 门已移除, 单 MS2 feature 应保留, 实际 {len(feats)} 个"
    )
    f = feats[0]
    assert f.precursor_mz_nominal == 300
    assert abs(f.ms1_precursor_mz - 300.10) < 0.05
    assert f.ms2_quality == "sparse", f"期望 sparse, 实际 {f.ms2_quality!r}"
    assert f.n_correlated_ms2 == 1
    assert f.n_fragments == 1


# ---------------------------------------------------------------------------
# Test C: 移除 ≥2 MS2 强制门 (Q3) — 保留没有任何相关 MS2 的 feature
# ---------------------------------------------------------------------------

def test_keeps_feature_with_zero_correlated_ms2():
    """单前体 300.10, 隔离窗内没有任何 MS2 信号。

    旧内核: n_frags=0 < 2 → 丢弃。
    新内核 (Q3 移除): 保留, 标注 ms2_quality="none" (n_correlated_ms2=0)。
    """
    rt = np.arange(30) * 0.005
    channel = 300
    ms1 = _gaussian(rt, center=0.07, sigma=0.02, amp=10000.0)

    raw = _build_segment(
        rt,
        ms1_ions=[(300.10, ms1)],
        channel=channel,
        ms2_ion_table={},  # 隔离窗内无 MS2
    )
    cfg = _default_config()
    feats = _run_stage1b(raw, cfg)

    assert len(feats) == 1, (
        f"≥2 MS2 门已移除, 无 MS2 feature 应保留, 实际 {len(feats)} 个"
    )
    f = feats[0]
    assert f.precursor_mz_nominal == 300
    assert f.ms2_quality == "none", f"期望 none, 实际 {f.ms2_quality!r}"
    assert f.n_correlated_ms2 == 0
    assert f.n_fragments == 0
