"""Reference-replicate selection for cross-replicate alignment.

What used to live here as well was ``align_features_across_replicates``: a
master list seeded from the reference replicate and never appended to, plus a
dense Hungarian assignment. It dropped every feature unique to a non-reference
replicate without a word (34% of them on ASFAM's rice data) and its dense
``n_ref x n_target`` similarity matrix cost O(n^2) memory. ASFAM replaced it with
``metabo_core.alignment.joiner`` in v1.0.260709.6 and DDA followed; the function
was deleted once DDA's switch-over landed, as its own docstring asked.

``reference_replicate_quality`` outlived it. The joiner still has to decide which
sample seeds the buckets first, and DDA and GC-MS both import it.
"""
from __future__ import annotations

import logging
from statistics import mean
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# W11 修复 2: 复合参考样选择
# ---------------------------------------------------------------------------


def reference_replicate_quality(features: Iterable[Any]) -> tuple[int, float, int]:
    """计算单个 replicate 的"参考样质量分", 返回 ``(n_passes, mean_sn, n_total)``。

    用作 cross-replicate alignment 选参考样的排序键, 替代旧的"特征最多即
    参考"启发式。该启发式在低 S/N 样品里会优先选到伪迹堆出来的 replicate,
    把真信号被迫匹去伪信号。

    规则:

    1. 过滤掉低质量条目: ``sn_ratio < 3``。``sn_ratio`` 字段名同时被
       LC-MS ``Feature``/GC-MS dict 使用; ``CandidateFeature`` 里则叫
       ``ms1_sn``, 这里两种字段都尝试。
    2. 通过质量门的条目数 = ``n_passes``;
    3. 它们的 ``sn_ratio`` 算术平均 = ``mean_sn``;
    4. 兜底字段 ``n_total`` = 该 replicate 全部 feature 数; 仅在
       ``(n_passes, mean_sn)`` 平局时起作用 — 例如所有 feature 都缺
       ``sn_ratio`` 字段时, 退化到旧的"特征数最多者即参考"。

    返回 tuple 用 ``max(..., key=...)`` 字典序比较, 实现
    "先比通过数, 再比 mean_sn, 都平则比总数"。
    """
    pieces: list[float] = []
    n_total = 0
    for f in features or []:
        n_total += 1
        # CandidateFeature 用 ms1_sn, Feature/GC-MS dict 用 sn_ratio
        sn = _read_sn_ratio(f)
        if sn is None or sn < 3.0:
            continue
        pieces.append(float(sn))
    if not pieces:
        return (0, 0.0, n_total)
    return (len(pieces), float(mean(pieces)), n_total)


def _read_sn_ratio(feature: Any) -> float | None:
    """从 feature (dataclass 或 dict) 取 S/N; 兼容两套字段名。"""
    if isinstance(feature, dict):
        v = feature.get("sn_ratio")
        if v is None:
            v = feature.get("ms1_sn")
        # GC-MS 缺 sn_ratio 时退化到 apex_intensity 门 (>1000 当作 OK)
        if v is None:
            apex = feature.get("apex_intensity", feature.get("height"))
            if apex is not None and float(apex) > 1000.0:
                return 3.0  # 一个刚好过门的占位 S/N
            return None
        return float(v)
    # dataclass 风格
    sn = getattr(feature, "sn_ratio", None)
    if sn is None:
        sn = getattr(feature, "ms1_sn", None)
    if sn is None:
        return None
    return float(sn)
