"""PR-C / Task C4: Stage 2 should skip MS1-driven features with a precise m/z.

Red phase (before fix): Stage 2 processes ALL features including ms1_driven ones,
overwriting feat_a.ms1_precursor_mz from the ROI-centroid value (285.0501) to the
coarse channel±0.5 blended centroid (≈285.135). First assertion fails.

Green phase (after fix): Stage 2 skips ms1_driven features that already carry a
precise ms1_precursor_mz, leaving them fully untouched. feat_b (ms2_driven) is
still processed normally, proving the skip is targeted.
"""
from __future__ import annotations

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import CandidateFeature, RawSegmentData, ScanCycle
from asfam.pipeline.stage2_ms1_assignment import run_stage2


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _gaussian(rt: np.ndarray, center: float, sigma: float, amp: float) -> np.ndarray:
    """简单高斯峰, 用于构造无噪声确定性 MS1 EIC。"""
    return amp * np.exp(-0.5 * ((rt - center) / sigma) ** 2)


def _build_raw_segment(
    rt: np.ndarray,
    ms1_ions: list[tuple[float, np.ndarray]],
    channel: int,
    replicate_id: int = 1,
    segment_name: str = "seg_285",
) -> RawSegmentData:
    """构造含多个 MS1 精确质量离子的 RawSegmentData (Stage 2 不需要 MS2)。

    Parameters
    ----------
    ms1_ions : [(精确 m/z, 每 cycle 强度向量)]
        MS1 survey 谱; 强度 > 0 的 ion 才放入各 cycle。
    channel : int
        1 Da 整数隔离通道 (precursor_list 中的值)。
    """
    n_cycles = len(rt)
    cycles: list[ScanCycle] = []
    for ci in range(n_cycles):
        ms1_mzs: list[float] = []
        ms1_ints: list[float] = []
        for mz, eic in ms1_ions:
            val = float(eic[ci])
            if val > 0.0:
                ms1_mzs.append(mz)
                ms1_ints.append(val)
        cycles.append(ScanCycle(
            cycle_index=ci,
            rt=float(rt[ci]),
            ms1_mz=np.array(ms1_mzs, dtype=np.float64),
            ms1_intensity=np.array(ms1_ints, dtype=np.float64),
            # MS2 留空; Stage 2 只读 MS1
            ms2_scans={channel: (np.empty(0), np.empty(0))},
        ))
    return RawSegmentData(
        file_path="synthetic.mzML",
        segment_name=segment_name,
        segment_low=channel - 1,
        segment_high=channel + 1,
        replicate_id=replicate_id,
        n_cycles=n_cycles,
        rt_array=rt.copy(),
        precursor_list=[channel],
        cycles=cycles,
    )


def _make_ms1_driven_feature(
    channel: int,
    precise_mz: float,
    rt_apex: float,
    ms1_height: float,
    ms1_sn: float,
    replicate_id: int = 1,
    segment_name: str = "seg_285",
) -> CandidateFeature:
    """仿 Stage 1b mass-slice 输出: detection_source='ms1_driven' + 精确 m/z 已赋值。"""
    half_width = 0.03  # 约 ±6 cycle @ 0.005 min/cycle
    return CandidateFeature(
        feature_id="F_ms1driven",
        segment_name=segment_name,
        replicate_id=replicate_id,
        precursor_mz_nominal=channel,
        rt_apex=rt_apex,
        rt_left=rt_apex - half_width,
        rt_right=rt_apex + half_width,
        ms2_mz=np.array([100.0, 150.0], dtype=np.float64),
        ms2_intensity=np.array([5000.0, 4000.0], dtype=np.float64),
        n_fragments=2,
        ms1_precursor_mz=precise_mz,   # Stage 1b mass-slice 已赋精确 m/z
        ms1_height=ms1_height,
        ms1_area=ms1_height * 0.5,
        ms1_sn=ms1_sn,
        ms1_gaussian=0.95,
        signal_type="ms1_detected",
        detection_source="ms1_driven",  # PR-C 新增字段
        mz_source="ms1_peak",
    )


def _make_ms2_driven_feature(
    channel: int,
    rt_apex: float,
    replicate_id: int = 1,
    segment_name: str = "seg_285",
) -> CandidateFeature:
    """普通 MS2 驱动 feature, 尚未拿到精确 m/z (Stage 2 负责赋值)。"""
    half_width = 0.03
    return CandidateFeature(
        feature_id="F_ms2driven",
        segment_name=segment_name,
        replicate_id=replicate_id,
        precursor_mz_nominal=channel,
        rt_apex=rt_apex,
        rt_left=rt_apex - half_width,
        rt_right=rt_apex + half_width,
        ms2_mz=np.array([120.0], dtype=np.float64),
        ms2_intensity=np.array([3000.0], dtype=np.float64),
        n_fragments=1,
        ms1_precursor_mz=None,          # 尚未赋精确 m/z, Stage 2 处理后才有值
        signal_type="ms1_detected",     # 默认占位; Stage 2 会改写
        detection_source="ms2_driven",
    )


# ---------------------------------------------------------------------------
# 主测试
# ---------------------------------------------------------------------------

def test_stage2_lowres_ms1_fallback_for_ms2_only():
    """问题 C: ASFAM 每张 MS2 都对应一张低分辨 MS1 (采集模式保证: 每周期
    1 张全分辨 MS1 + N 张隔离窗 MS2)。

    当某 ms2_driven feature 的 MS1 信号太弱 —— strict(floor 300) 与
    relaxed(floor 50) 都抽不出可检测峰 —— 旧逻辑走 ``_assign_representative_ion``
    终档, **不赋 ms1_precursor_mz**, ``precursor_mz`` 退回整数通道 (如 200.0)。

    兜底: 取该通道 ±0.5 Da 窗在 feature apex cycle 的 MS1 质心, 给一个低分辨
    精确 m/z (mz_source='ms1_lowres'), 不再退回整数通道。signal_type 仍是
    ms2_only (这是窗质心, 不是检测到的 MS1 峰)。
    """
    rt = np.arange(30) * 0.005
    channel = 200
    apex_rt = rt[14]

    # MS1 在隔离窗内有信号 (精确 m/z 200.3), 但峰高仅 ~35 < relaxed floor 50
    # → strict + relaxed 都抽不出峰 → 旧逻辑退回整数通道 200.0
    weak = _gaussian(rt, center=apex_rt, sigma=0.02, amp=35.0)
    raw = _build_raw_segment(
        rt, ms1_ions=[(200.3, weak)], channel=channel,
        segment_name="seg_200",
    )
    feat = _make_ms2_driven_feature(
        channel=channel, rt_apex=apex_rt, segment_name="seg_200",
    )

    cfg = ProcessingConfig()
    run_stage2({"1": [raw]}, {"1": [feat]}, cfg)

    # 兜底应赋低分辨精确 m/z, 不再退回整数通道
    assert feat.ms1_precursor_mz is not None, "未给 ms2_only feature 补低分辨 MS1 m/z"
    assert feat.mz_source == "ms1_lowres"
    assert feat.mz_confidence == "low"
    assert abs(feat.precursor_mz - 200.3) < 0.05, (
        f"低分辨兜底质心应 ≈200.3 (窗内信号位置), 实际 {feat.precursor_mz}"
    )
    # 仍是 ms2_only (兜底只补 m/z, 不是检测到的 MS1 峰)
    assert feat.signal_type == "ms2_only"


def test_stage2_skips_ms1_driven_precise_mz():
    """Stage 2 不应覆盖 MS1-driven feature 的精确 ROI 质心 m/z。

    场景
    ----
    通道 285 内两个共隔离前体 (285.05 / 285.42), 同 apex (cycle 14, RT=0.07)。
    Stage 1b mass-slice 已给 feat_a 赋 ms1_precursor_mz=285.0501 (精确)。

    Stage 2 粗质心计算:
      centroid = (285.05×10000 + 285.42×3000) / 13000 ≈ 285.135
    若 Stage 2 处理 feat_a, 会把 ms1_precursor_mz 从 285.0501 改成 ≈285.135,
    精度丢失 (两前体被混合)。

    红相验证 (未修复)
    ----------------
    feat_a 以索引 0 先于 feat_b 进入 channel_groups; greedy 优先分配给它
    → _batch_assign_ms1 把 ms1_precursor_mz 改为 ≈285.135
    → 断言 A: abs(feat_a.ms1_precursor_mz - 285.0501) < 1e-6  **FAILS**

    绿相验证 (修复后)
    ----------------
    feat_a 因 detection_source=='ms1_driven' and ms1_precursor_mz is not None
    被跳过, ms1_precursor_mz 保持 285.0501。
    feat_b (ms2_driven) 正常经过 Stage 2, 拿到粗质心 ≈285.135 (或成 ms2_only)。
    两个断言均通过。
    """
    rt = np.arange(30) * 0.005   # 30 cycles, dt=0.005 min/cycle
    channel = 285
    apex_rt = rt[14]              # 0.07 min

    # 两前体的高斯 MS1 EIC (同 apex, 不同强度)
    # Stage 2 _detect_ms1_peaks 用 max 抽 EIC → 峰高=10000, 远超 ms1_min_height=300
    strong_eic = _gaussian(rt, center=apex_rt, sigma=0.02, amp=10000.0)
    weak_eic   = _gaussian(rt, center=apex_rt, sigma=0.02, amp=3000.0)

    raw = _build_raw_segment(
        rt,
        ms1_ions=[(285.05, strong_eic), (285.42, weak_eic)],
        channel=channel,
    )

    cfg = ProcessingConfig()

    PRECISE_MZ = 285.0501    # Stage 1b mass-slice 产出的精确 m/z
    MS1_HEIGHT  = 10000.0
    MS1_SN      = 20.0

    # feat_a 排在索引 0: 无修复时 greedy 优先分配给它 (同分 Python stable sort 保序)
    feat_a = _make_ms1_driven_feature(
        channel=channel,
        precise_mz=PRECISE_MZ,
        rt_apex=apex_rt,
        ms1_height=MS1_HEIGHT,
        ms1_sn=MS1_SN,
    )
    # feat_b 排在索引 1: 无修复时进入 relaxed pass; 修复后成为 channel_groups 唯一成员
    feat_b = _make_ms2_driven_feature(channel=channel, rt_apex=apex_rt)

    features = [feat_a, feat_b]
    run_stage2({"1": [raw]}, {"1": features}, cfg)

    # ---- 断言 A: feat_a 精确 m/z 未被粗质心 (≈285.135) 覆盖 ----
    assert abs(feat_a.ms1_precursor_mz - PRECISE_MZ) < 1e-6, (
        f"feat_a.ms1_precursor_mz 被 Stage 2 覆盖: "
        f"期望 {PRECISE_MZ}, 实际 {feat_a.ms1_precursor_mz:.6f} "
        f"(粗质心 ≈ 285.135, 与 PRECISE_MZ 相差 {abs(feat_a.ms1_precursor_mz - PRECISE_MZ):.4f})"
    )

    # ---- 断言 A2: ms1_height / ms1_sn 也未被改写 ----
    assert feat_a.ms1_height == MS1_HEIGHT, (
        f"feat_a.ms1_height 被改写: 期望 {MS1_HEIGHT}, 实际 {feat_a.ms1_height}"
    )
    assert feat_a.ms1_sn == MS1_SN, (
        f"feat_a.ms1_sn 被改写: 期望 {MS1_SN}, 实际 {feat_a.ms1_sn}"
    )

    # ---- 断言 B: feat_b (ms2_driven) 确实经过了 Stage 2 处理 ----
    # 修复后 feat_b 是 channel_groups 唯一成员, 应拿到 ms1_precursor_mz (≈285.135)
    # 或退化为 ms2_only — 两者均证明 skip 是定向的, 不是全量 bypass
    processed = (feat_b.ms1_precursor_mz is not None) or (feat_b.signal_type == "ms2_only")
    assert processed, (
        f"feat_b 未被 Stage 2 处理: "
        f"ms1_precursor_mz={feat_b.ms1_precursor_mz}, "
        f"signal_type={feat_b.signal_type!r}"
    )
