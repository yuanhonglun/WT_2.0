"""Peak-detection configuration shared by every app.

Mirrors the MS-DIAL parameter surface used by both the LC-MS (DDA / DIA)
and GC-MS feature finders. The only knob users normally tune is
``min_amplitude`` — the absolute height a peak must clear above its
local baseline. The remaining parameters control the baseline /
noise estimator (default to MS-DIAL's hardcoded values) and are kept
overridable for advanced testing but not surfaced in the GUI.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PeakDetectionConfig:
    """Parameters for MS-DIAL-style chromatographic peak detection."""

    # ----- User-tunable -----
    # Absolute minimum peak height above local baseline (Gate B).
    min_amplitude: float = 1000.0
    # Minimum peak width in scans.
    min_data_points: int = 3
    # Optional Gaussian shape gate (off by default). Orthogonal to the
    # three absolute-height prominence gates.
    gaussian_threshold: float = 0.0
    # Optional 4th prominence gate, RATIO based. If > 0, require
    # ``min_prom / apex_above_baseline >= min_prominence_ratio``. Kills
    # the "tall peak that barely rises above its valleys" artefact that
    # the absolute gates miss when a high-response continuous baseline
    # noise has a small local maximum on top.
    min_prominence_ratio: float = 0.0
    # Optional user-defined effective RT window. Peaks whose apex RT
    # lies outside ``[rt_window_min, rt_window_max]`` are rejected.
    # ``None`` on either side disables that bound. Used to skip the
    # solvent-front dead time at the start of a gradient and the
    # re-equilibration tail at the end.
    rt_window_min: float | None = None
    rt_window_max: float | None = None

    # ----- Internal (MS-DIAL hard defaults) -----
    smooth_window: int = 1
    baseline_window: int = 20
    noise_bin_size: int = 50
    noise_factor: float = 3.0
    sn_fold: float = 4.0


# ---------------------------------------------------------------------------
# Named default factories — call each to get a fresh ``PeakDetectionConfig``.
#
# 用工厂函数（而不是 module-level 单例）确保每次调用都拿到独立的 dataclass
# 实例，避免多个 app / 多个 ProcessingConfig 共享同一个 mutable 默认值时
# 互相串改。三套默认覆盖整个平台目前的峰检测使用面：
#
#   - LC-MS MS1 EIC     -> lc_ms1_peak_config
#   - LC-MS MS2 EIC     -> lc_ms2_peak_config
#   - GC-MS（fullscan / cSIM）-> gc_peak_config
# ---------------------------------------------------------------------------


def lc_ms1_peak_config() -> PeakDetectionConfig:
    """LC-MS MS1 EIC 峰检测的默认参数。

    适用范围：
      - ASFAM 主流程 stage1b 的 MS1 互补检测（注意 stage1b 还会再用
        ``ms1_min_height`` 覆盖 ``min_amplitude``，用更低的门去捞补偿峰）
      - DDA 的 MS1 EIC 峰检测（``stage_features``）
      - 未来 DIA 的 MS1 EIC 峰检测

    参数选择依据：
      - ``min_amplitude=500``：ASFAM 历史默认，覆盖大多数 LC-MS QTOF /
        Orbitrap 的可用响应范围；DDA 原先 1000 在用户确认后下调至 500
        以统一 baseline。
      - ``min_data_points=3``：LC 峰一般跨 5-15 个 scan，最低 3 既能
        滤掉单点尖刺又不会切掉真正窄的峰。
      - ``gaussian_threshold=0.85``：2026-05-15 rice ASFAM 反馈显示 0.75
        放过了一类 "形状勉强像高斯但实质上是连续噪声" 的伪峰。GC 默认
        已经是 0.85（电子轰击 + 短峰宽），LC 真高斯峰也能轻松过 0.85，
        所以提到 0.85 作为新的 LC 默认，过滤更严格。
      - ``min_prominence_ratio=0.3``：第 4 道比例门。用户在 rice ASFAM
        反馈里指出，许多 noise feature "响应很高但突出度差"——
        ``min_prom / apex_above_baseline`` 极低。真高斯峰该值≈0.9，
        拖尾/前肩峰仍能保持 ≥0.4，所以 LC 默认 0.3，既能滤掉连续基线
        噪声叠加的小局部极大，又不会误杀拖尾峰。
      - ``rt_window_min/max`` 默认 None（全 RT）：用户在 GUI 上为每次
        run 自定义有效梯度区间，跳过死时间 + 平衡尾段产生的伪斜坡峰。
      - 噪声 / S/N 默认沿用 MS-DIAL 三门（noise_factor=3, sn_fold=4）。
    """
    return PeakDetectionConfig(
        min_amplitude=500.0,
        min_data_points=3,
        smooth_window=1,
        baseline_window=20,
        noise_bin_size=50,
        noise_factor=3.0,
        sn_fold=4.0,
        gaussian_threshold=0.85,
        min_prominence_ratio=0.3,
    )


def lc_ms2_peak_config() -> PeakDetectionConfig:
    """LC-MS MS2 EIC 峰检测的默认参数。

    适用范围：
      - ASFAM stage1 的产品离子 EIC 峰检测
      - 未来 DIA 的 MS2 EIC 峰检测

    与 ``lc_ms1_peak_config`` 的差异：
      - ``min_amplitude=200``：MS2 离子响应通常显著低于 MS1，门要
        更宽松；ASFAM 历史在 stage1 用 200 这个量级。
      - 其余参数与 MS1 一致（min_data_points=3, gaussian=0.85,
        min_prominence_ratio=0.3, MS-DIAL 三门），保证不同 stage 的
        峰形 / 噪声判定一致。
    """
    return PeakDetectionConfig(
        min_amplitude=200.0,
        min_data_points=3,
        smooth_window=1,
        baseline_window=20,
        noise_bin_size=50,
        noise_factor=3.0,
        sn_fold=4.0,
        gaussian_threshold=0.85,
        min_prominence_ratio=0.3,
    )


def gc_peak_config() -> PeakDetectionConfig:
    """GC-MS 峰检测的默认参数（fullscan 与 cSIM 共用）。

    GC 谱图的物理特性与 LC 显著不同，参数必须区分：
      - GC 峰宽极窄（典型 1-3 秒，对应 10-20 个 scan），但峰非常密集
        且基线噪声更大。
      - GC EI 谱图的峰形非常接近理想高斯（电子轰击 + 短 GC 峰宽），
        ``gaussian_threshold`` 可以拉到 0.85 而不会误杀真峰。

    参数选择依据：
      - ``min_amplitude=1000``：GC-MS 单位质量 EIC 的响应往往高于 LC，
        噪声门也对应抬高。
      - ``smooth_window=5``：GC 峰窄、基线抖动相对幅度更大，需要更强
        的局部平滑抑制；LC 一般 1 即可。
      - ``baseline_window=100``：GC 整体运行更长、单峰持续短，基线
        估计窗口必须拉长才能稳健覆盖一个真正的“local baseline”
        区段；LC 的 20 在 GC 上会跟着峰一起跑。
      - ``noise_factor=5`` / ``sn_fold=5``：GC 噪声更高，三门要收紧
        才能滤掉电子噪声 / 基线漂移产生的伪峰。
      - ``gaussian_threshold=0.85``：GC EI 峰高度对称且窄，提高高斯
        判据可以滤掉真正畸形 / 重叠的伪结构。
      - ``min_data_points=5``：GC 峰跨度普遍 8-15 个 scan，5 是
        “真正完整一个 GC 峰”的稳健下限。
    """
    return PeakDetectionConfig(
        min_amplitude=1000.0,
        min_data_points=5,
        smooth_window=5,
        baseline_window=100,
        noise_bin_size=50,
        noise_factor=5.0,
        sn_fold=5.0,
        gaussian_threshold=0.85,
        min_prominence_ratio=0.5,
    )
