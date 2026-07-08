"""Cross-replicate alignment core algorithm.

Algorithmic core extracted from ASFAM stage 7. The core only consumes shared
``CandidateFeature`` lists and an ``AlignmentConfig``; it has no dependency on
ASFAM raw-segment data or EIC extraction. ASFAM stage glue handles progress
reporting and any acquisition-specific gap filling.
"""
from __future__ import annotations

import logging
from statistics import mean
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from metabo_core.algorithms.similarity import cosine_similarity
from metabo_core.config import AlignmentConfig
from metabo_core.models import CandidateFeature, Feature

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


def align_features_across_replicates(
    features_by_replicate: dict[str, list[CandidateFeature]],
    config: AlignmentConfig,
) -> list[Feature]:
    """Align candidate features across replicates and assemble final features."""
    rep_ids = sorted(features_by_replicate.keys())
    if not rep_ids:
        return []

    # W11 修复 2: 复合排序 (n_passes, mean_sn), 替代 "len(features) 最大者"
    # 单标准 — 后者在含大量低 S/N 伪峰的 replicate 上会选错参考样。
    ref_id = max(
        rep_ids,
        key=lambda r: reference_replicate_quality(features_by_replicate[r]),
    )
    ref_features = features_by_replicate[ref_id]
    logger.info(
        "  Reference replicate: %s (%d features, quality=%s)",
        ref_id,
        len(ref_features),
        reference_replicate_quality(ref_features),
    )

    aligned: list[dict] = []
    for feat in ref_features:
        aligned.append({"ref_feature": feat, "matches": {ref_id: feat}})

    ref_peaks_list: list[list] = []
    for aln in aligned:
        feat = aln["ref_feature"]
        if feat.ms2_mz is not None and len(feat.ms2_mz) > 0:
            ref_peaks_list.append(
                list(zip(feat.ms2_mz.tolist(), feat.ms2_intensity.tolist()))
            )
        else:
            ref_peaks_list.append([])

    ms2_tol = config.ms2_mz_tolerance

    for rep_id in rep_ids:
        if rep_id == ref_id:
            continue

        target_features = features_by_replicate[rep_id]
        n_ref = len(aligned)
        n_target = len(target_features)
        if n_target == 0:
            continue

        target_peaks_list: list[list] = []
        for t_feat in target_features:
            if t_feat.ms2_mz is not None and len(t_feat.ms2_mz) > 0:
                target_peaks_list.append(
                    list(zip(t_feat.ms2_mz.tolist(), t_feat.ms2_intensity.tolist()))
                )
            else:
                target_peaks_list.append([])

        sim_matrix = np.zeros((n_ref, n_target))
        for i, aln in enumerate(aligned):
            ref_feat = aln["ref_feature"]
            ref_mz = ref_feat.precursor_mz
            ref_rt = ref_feat.rt_apex
            ref_peaks = ref_peaks_list[i]
            for j, t_feat in enumerate(target_features):
                if abs(ref_mz - t_feat.precursor_mz) > config.mz_tolerance * 3:
                    continue
                if abs(ref_rt - t_feat.rt_apex) > config.rt_tolerance * 3:
                    continue
                gauss = _gaussian_similarity(
                    ref_mz, ref_rt,
                    t_feat.precursor_mz, t_feat.rt_apex,
                    config.mz_tolerance, config.rt_tolerance,
                    config.mz_weight, config.rt_weight,
                )
                ms2_cos = 0.0
                if ref_peaks and target_peaks_list[j]:
                    ms2_cos, _ = cosine_similarity(
                        ref_peaks, target_peaks_list[j], ms2_tol,
                    )
                if ref_peaks and target_peaks_list[j]:
                    sim_matrix[i, j] = 0.6 * gauss + 0.4 * ms2_cos
                else:
                    sim_matrix[i, j] = gauss

        cost_matrix = -sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_ind, col_ind):
            if sim_matrix[r, c] > config.match_threshold:
                aligned[r]["matches"][rep_id] = target_features[c]
        n_matched = sum(
            1 for r, c in zip(row_ind, col_ind)
            if sim_matrix[r, c] > config.match_threshold
        )
        logger.info(
            "  Replicate %s: %d/%d features matched", rep_id, n_matched, n_target,
        )

    result: list[Feature] = []
    for i, aln in enumerate(aligned):
        ref_feat = aln["ref_feature"]
        matches = aln["matches"]

        heights = {}
        areas = {}
        for rid, feat in matches.items():
            heights[rid] = feat.ms1_height if feat.ms1_height else 0.0
            areas[rid] = feat.ms1_area if feat.ms1_area else 0.0

        h_vals = [v for v in heights.values() if v > 0]
        a_vals = [v for v in areas.values() if v > 0]
        mean_h = float(np.mean(h_vals)) if h_vals else 0.0
        mean_a = float(np.mean(a_vals)) if a_vals else 0.0
        cv_h = float(np.std(h_vals) / mean_h) if mean_h > 0 and len(h_vals) > 1 else 0.0

        feature = Feature(
            feature_id=f"F{i:05d}",
            precursor_mz=ref_feat.precursor_mz,
            rt=ref_feat.rt_apex,
            rt_left=ref_feat.rt_left,
            rt_right=ref_feat.rt_right,
            signal_type=ref_feat.signal_type,
            ms2_mz=ref_feat.ms2_mz,
            ms2_intensity=ref_feat.ms2_intensity,
            n_fragments=ref_feat.n_fragments,
            heights=heights,
            areas=areas,
            mean_height=mean_h,
            mean_area=mean_a,
            cv=cv_h,
            formula=ref_feat.inferred_formula,
            adduct=ref_feat.adduct_type,
            sn_ratio=ref_feat.ms1_sn or 0.0,
            ms1_isotopes=ref_feat.ms1_isotopes,
            name=ref_feat.matchms_name,
            height_ion_mz=ref_feat.ms2_rep_ion_mz,
            detection_source=ref_feat.detection_source,
            mz_source=ref_feat.mz_source,
            mz_confidence=ref_feat.mz_confidence,
            is_duplicate=ref_feat.is_duplicate,
            duplicate_group_id=ref_feat.duplicate_group_id,
            duplicate_type=ref_feat.duplicate_type,
        )
        if ref_feat.annotation_matches:
            feature.annotation_matches = ref_feat.annotation_matches
            feature.selected_annotation_idx = ref_feat.selected_annotation_idx
        result.append(feature)

    logger.info("  Aligned features: %d", len(result))
    return result


def _gaussian_similarity(
    mz1: float, rt1: float,
    mz2: float, rt2: float,
    mz_tol: float, rt_tol: float,
    mz_weight: float = 0.5, rt_weight: float = 0.5,
) -> float:
    if mz_tol <= 0 or rt_tol <= 0:
        return 0.0
    mz_score = np.exp(-0.5 * ((mz1 - mz2) / mz_tol) ** 2)
    rt_score = np.exp(-0.5 * ((rt1 - rt2) / rt_tol) ** 2)
    return mz_weight * mz_score + rt_weight * rt_score
