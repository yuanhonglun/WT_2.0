"""PR-C C5: 单段真实数据 smoke — mass-slice MS1 驱动内核应找到 *显著更多* 真峰。

PR-C 把 stage1b 的 MS1 驱动内核从"每整数通道一条 1 Da EIC, 取 max 塌缩"
换成在全分辨 MS1 survey 上做精细 m/z mass-slice 找峰
(``find_lc_ms1_features``), 并移除了 ≥2 MS2 强制门 (Q3)。本 smoke 用一个
真实 ASFAM 段证明新内核确实捞出远多于旧实现的真峰, 且这些峰是真实色谱
峰 (高斯 + S/N 门已通过), 而不是噪声。

数据: ``test_data/asfam/full_range/RL_ASFAM_285-314_P_3.mzML`` (m/z 段 285-314,
956 cycles, 30 个隔离通道)。

隔离内核: 跑 Stage 0 加载, 然后用 **空 existing_features** 跑 Stage 1b
(没有 MS2 驱动 feature 去重), 这样 result 里所有 feature 都是新建的
MS1 驱动 feature, 直接数 ``detection_source == "ms1_driven"`` 即可。

观测基线 (2026-06-24, 在该段实测):
  - **n_ms1_driven = 864**  (旧版本 该段 *总* feature 数 ≈ 72,
    新内核约 12×, 显著更多)
  - ms1_sn:       median ≈ 71,  min ≈ 4.0 (恰好过 sn_fold=4 门)
  - ms1_gaussian: median ≈ 0.985, 全部 ≥ 0.85 (= ms1_peak 高斯门)
  - ms1_mz:       [284.96, 314.69], 全有限, 落在 285-314 段内
  - ms2_quality:  none=594 / sparse=165 / correlated=105
                  (印证 Q3 门移除: 弱/无 MS2 的峰被保留并标注)
  - 运行时间:     stage0 ≈ 9s + stage1b ≈ 57s ≈ 65s

阈值取法 (按 plan: "显著高于 72 且安全低于实测值, 留 PR-E 调参余量"):
  - 数量门 ``>= 200``: 约旧基线 2.8×, 远低于实测 864, 即使 PR-E 调参
    收紧也稳; 但已无歧义地证明"显著更多"。
  - 质量门读 config 实际门值 (``ms1_peak.gaussian_threshold`` /
    ``sn_fold``), 不硬编码。

慢测试: 加载 ~176 MB mzML + 找峰约 1 分钟, 标 ``@pytest.mark.slow``,
默认 (无 ``ASFAM_RUN_SLOW=1`` 且无 ``-m slow``) 跳过; 数据缺失也跳过。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from asfam.config import ProcessingConfig
from asfam.pipeline.stage0_load import run_stage0
from asfam.pipeline.stage1b_ms1_detection import run_stage1b


# parents[3] 从 apps/asfam_processor/tests/ 上溯到仓库根 metabo-platform
DATA = (
    Path(__file__).resolve().parents[3]
    / "test_data" / "asfam" / "full_range" / "RL_ASFAM_285-314_P_3.mzML"
)

# 旧版本 该段总 feature 数 (1 Da 塌缩 + ≥2 MS2 门), 仅作对照基线
OLD_BASELINE = 72
# 数量门: 显著高于旧基线 (≈2.8×), 远低于实测 864 → 对 PR-E 调参稳健
MIN_MS1_DRIVEN = 200


def _slow_explicitly_requested() -> bool:
    """pytest 是否用 `-m slow` 调用 (best-effort, 与 DDA 慢测试一致)。"""
    import sys
    args = " ".join(sys.argv)
    return ("-m slow" in args) or ("-m=slow" in args)


@pytest.mark.slow
def test_stage1b_massslice_finds_more_real_peaks():
    """单段 smoke: MS1 驱动内核应找到显著更多 (>=200) 真实色谱峰。"""
    if os.environ.get("ASFAM_RUN_SLOW") != "1" and not _slow_explicitly_requested():
        pytest.skip(
            "slow test (~1 min, loads ~176MB mzML). Set ASFAM_RUN_SLOW=1 or run "
            "`pytest -m slow apps/asfam_processor/tests/test_stage1b_massslice_smoke.py`."
        )
    if not DATA.exists():
        pytest.skip(f"real-data smoke input missing: {DATA}")

    cfg = ProcessingConfig()

    # Stage 0: 加载该段 mzML (单文件 → 顺序加载, 不走 multiprocessing)
    data = run_stage0([str(DATA)], cfg, lambda *a, **k: None)
    assert data, "stage0 returned no replicates"

    # Stage 1b: 空 existing → 结果里全是新 MS1 驱动 feature (无 MS2 去重干扰)
    existing = {rep_id: [] for rep_id in data}
    result = run_stage1b(data, existing, cfg)

    ms1_driven = [
        f
        for feats in result.values()
        for f in feats
        if f.detection_source == "ms1_driven"
    ]
    n_ms1_driven = len(ms1_driven)

    # (1) 显著更多: 远超旧 ~72 总数 — 先断言, 避免空数组时 numpy ops 抛 ValueError 掩盖回归
    assert n_ms1_driven >= MIN_MS1_DRIVEN, (
        f"MS1 驱动 feature 数 {n_ms1_driven} 未显著超过旧基线 ~{OLD_BASELINE}; "
        f"期望 >= {MIN_MS1_DRIVEN}"
    )

    # 质量字段必须齐 (内核保证, 这里兜底防回归)
    assert all(f.ms1_sn is not None for f in ms1_driven)
    assert all(f.ms1_gaussian is not None for f in ms1_driven)
    assert all(f.ms1_precursor_mz is not None for f in ms1_driven)

    sn = np.array([f.ms1_sn for f in ms1_driven], dtype=float)
    gauss = np.array([f.ms1_gaussian for f in ms1_driven], dtype=float)
    mz = np.array([f.ms1_precursor_mz for f in ms1_driven], dtype=float)

    g_gauss = cfg.ms1_peak.gaussian_threshold  # 0.85, 读 config 不硬编码
    g_sn = cfg.ms1_peak.sn_fold                # 4.0

    median_sn = float(np.median(sn))
    median_gauss = float(np.median(gauss))
    frac_gauss_ok = float((gauss >= g_gauss).mean())

    # 诊断输出 (用 -s 可见; 与 DDA 慢测试风格一致)
    print(
        f"\nStage1b mass-slice MS1 smoke (RL_ASFAM_285-314_P_3):\n"
        f"  n_ms1_driven   = {n_ms1_driven}  (old baseline total ~{OLD_BASELINE}; "
        f"threshold >= {MIN_MS1_DRIVEN})\n"
        f"  ms1_sn         median={median_sn:.1f} min={sn.min():.1f} max={sn.max():.0f}\n"
        f"  ms1_gaussian   median={median_gauss:.3f} min={gauss.min():.3f} "
        f">= {g_gauss}: {frac_gauss_ok:.3f}\n"
        f"  ms1_mz         [{mz.min():.4f}, {mz.max():.4f}]"
    )

    # (2) 真峰 (非噪声): S/N 中位数远高于 sn_fold 门; 几乎全部过高斯门
    assert median_sn >= g_sn * 2.5, (
        f"ms1_sn 中位数 {median_sn:.1f} 过低 (sn_fold 门={g_sn}, 期望中位数 >= {g_sn * 2.5:.1f}); 疑似噪声"
    )
    assert frac_gauss_ok >= 0.95, (
        f"仅 {frac_gauss_ok:.1%} 的 feature 过高斯门 {g_gauss}; 疑似噪声"
    )
    assert median_gauss >= g_gauss, (
        f"ms1_gaussian 中位数 {median_gauss:.3f} 低于高斯门 {g_gauss}"
    )

    # (3) 前体 m/z 全有限且落在该段 m/z 范围内 (±2 Da 容隔离窗边缘)
    assert np.isfinite(mz).all(), "存在非有限的 ms1_precursor_mz"
    seg_low = min(s.segment_low for segs in data.values() for s in segs)
    seg_high = max(s.segment_high for segs in data.values() for s in segs)
    assert mz.min() >= seg_low - 2.0, (
        f"ms1_precursor_mz 最小值 {mz.min():.4f} 低于段下界 {seg_low}-2"
    )
    assert mz.max() <= seg_high + 2.0, (
        f"ms1_precursor_mz 最大值 {mz.max():.4f} 高于段上界 {seg_high}+2"
    )
