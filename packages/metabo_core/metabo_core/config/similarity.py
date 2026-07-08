"""谱图相似度配置（LC-MS 与 GC-MS 共享）。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimilarityConfig:
    """谱图相似度评分参数。

    LC-MS 与 GC-MS 共享同一份配置，但默认 ``mz_tolerance`` 不同：
    LC-MS（DDA / ASFAM）默认 0.02 Da；GC-MS 在 ``GcmsConfig`` 中显式
    设置为 0.5 Da（单位质量约定）。其它字段（匹配峰数下限、匹配峰
    比例下限、阈值、是否启用 RT）在 LC-MS 与 GC-MS 之间共享一致。

    字段说明
    --------
    mz_tolerance
        峰匹配 m/z 容差（Da）。LC-MS 默认 0.02；GC-MS 默认 0.5。
    ms1_tolerance
        前体 m/z 的高斯相似度容差（Da），目前 LC-MS 才会用到（GC-MS
        没有前体维度）。
    min_matched_peaks
        视为有效匹配所需的最少共匹配峰数。
    min_matched_pct
        视为有效匹配所需的"匹配峰占参考谱信号峰"的最低比例。
    similarity_threshold
        显示时的综合相似度阈值；流水线侧通常以 0.0 实际入库，由 GUI
        与导出层在显示/聚合时再用该阈值过滤。
    use_rt
        是否在综合分中加入 RT 项（默认关闭，LC-MS RT 漂移大）。
    rt_tolerance
        RT 项的容差（LC-MS 单位为秒，传给底层算法时会换算为分钟）。
    chrom_weight
        仅 GC-MS 使用：色谱项 ``chrom^weight`` 的指数权重，默认 0.5。
    """

    # 峰匹配 m/z 容差。LC 默认 0.02 Da；GC 由 GcmsConfig 显式覆盖为 0.5。
    mz_tolerance: float = 0.02
    ms1_tolerance: float = 0.01
    # 三套相似度共用的最低匹配条件 / 显示阈值。
    min_matched_peaks: int = 3
    min_matched_pct: float = 0.25
    similarity_threshold: float = 0.7
    # RT 项相关（LC-MS 默认关闭）。
    use_rt: bool = False
    rt_tolerance: float = 100.0
    # GC-MS 专属：色谱分权重指数。LC-MS 路径忽略。
    chrom_weight: float = 0.5
