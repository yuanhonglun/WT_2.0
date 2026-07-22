"""Stage 4: 同位素去重。

W6 重构后的判定流程 (与 DDA / Stage 5 / Stage 6 共享统一底座):

  1. **基础门 1 — apex RT 严格**: 两个 feature 的 apex RT 距离 ≤
     ``apex_rt_strict`` (由 ``apex_rt_strict_from_ms1_cycles`` 从 MS1
     周期间隔算出, 不再混入 MS2 scan 把周期拉短)。
  2. **基础门 2 — EIC 共流出**: 两条 MS1 EIC 用
     ``eic_coelution_ok`` 校验, Pearson r ≥ 0.9 且
     n_correlated_points ≥ ``max(5, 0.5 × min(peak_width_scans))``。
  3. **基础门 3 — Δm/z 命中已知同位素步**: 通过 ``classify_isotope_gap``
     识别 C13×n (n ≤ 4) 或其他 ISOTOPE_DELTAS。
  4. **ASFAM 增强 (Tier 0)**: 轻同位素 top-N MS2 离子在重同位素 MS2
     中有 +Δm/z 回显, 命中率 ≥ ``isotope_step_min_ratio``。这是 ASFAM
     独有的高质量信号 — 即使基础门 1-3 没全过, 单凭 Tier 0 + 基础门
     1 (apex RT) + Δm/z 命中也可以直接连同位素边。
  5. **C13 强度比 sanity check**: 重同位素强度高于物理上限 ->
     否决该边。

历史的 Tier 1 (MS1 isotope list 反向命中) / Tier 2 (modified cosine) /
Tier 3 (relaxed gap + 中性损失 cosine) 已经全部删除 — 基础门 1-3 已
经够强, 而 Tier 1/2/3 增加 ~200 行未必带来更多召回, 还引入"修正
cosine 阈值 / 中性损失 cosine 阈值"这种难以调参的超参数。
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from asfam.config import ProcessingConfig
from asfam.core.clustering import connected_components, select_representative
from asfam.core.eic import extract_ms1_eic
from asfam.core.mass_utils import (
    classify_isotope_gap,
    max_c13_m1_ratio,
)
from asfam.core.similarity import ms2_isotope_step_score
from asfam.constants import C13_DELTA, ISOTOPE_DELTAS
from asfam.models import CandidateFeature, RawSegmentData
from metabo_core.algorithms.dedup_relations import (
    apex_rt_strict_from_ms1_cycles,
    eic_coelution_ok,
)


logger = logging.getLogger(__name__)


def run_stage4(
    features_by_replicate: dict[str, list[CandidateFeature]],
    config: ProcessingConfig,
    progress_callback: Optional[Callable] = None,
    data_by_replicate: Optional[dict[str, list[RawSegmentData]]] = None,
) -> dict[str, list[CandidateFeature]]:
    """对每个 replicate 做同位素去重 (统一 EIC 共流出 + Tier 0 MS2 步态)。"""
    apex_rt_strict = apex_rt_strict_from_ms1_cycles(
        data_by_replicate,
        n_cycles=int(getattr(config, "isotope_apex_rt_n_cycles", 2)),
        fallback_min=float(getattr(config, "isotope_apex_rt_fallback", 0.04)),
    )
    logger.info(
        "Stage 4: Isotope deduplication (apex_rt_strict=%.4f min)...",
        apex_rt_strict,
    )

    # raw 数据查表: 一个 segment + replicate 对应一个 RawSegmentData
    raw_lookup: dict[tuple[str, int], RawSegmentData] = {}
    if data_by_replicate:
        for rep_id, segments in data_by_replicate.items():
            for seg in segments:
                raw_lookup[(seg.segment_name, seg.replicate_id)] = seg

    for rep_id, features in features_by_replicate.items():
        active = [f for f in features if f.status == "active"]
        n_before = len(active)

        # 排序后比较, 早期终止
        active.sort(key=lambda f: (f.rt_apex, f.precursor_mz))

        # EIC 缓存: 同一 feature 多次比较时复用 (feature_id -> (eic, rt_array))
        eic_cache: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}

        def _get_eic_and_width(feat: CandidateFeature):
            """返回 (eic, rt_array, peak_width_scans), 取不到则返回 None。"""
            cached = eic_cache.get(feat.feature_id)
            if cached is not None:
                return cached
            raw = raw_lookup.get((feat.segment_name, feat.replicate_id))
            if raw is None:
                return None
            rt_arr, eic = extract_ms1_eic(raw, feat.precursor_mz, 0.5)
            # 峰宽 (scan 数): rt_left / rt_right 在 rt_array 上的区间长度
            lo = int(np.searchsorted(rt_arr, feat.rt_left, side="left"))
            hi = int(np.searchsorted(rt_arr, feat.rt_right, side="right"))
            width = max(1, hi - lo)
            cached = (eic, rt_arr, width)
            eic_cache[feat.feature_id] = cached
            return cached

        adjacency: dict[int, set[int]] = {i: set() for i in range(len(active))}
        n_edges = 0

        for i in range(len(active)):
            fi = active[i]
            for j in range(i + 1, len(active)):
                fj = active[j]

                # 基础门 1: apex RT 严格 (用 MS1 周期算出的容差)
                rt_diff = abs(fj.rt_apex - fi.rt_apex)
                # 排序后比较, 一旦 RT 差超过 3 倍宽松容差就可以终止 j 循环
                if rt_diff > config.isotope_rt_tolerance * 3:
                    break
                if rt_diff > apex_rt_strict:
                    continue

                # 基础门 3 (放在前面快速过滤): Δm/z 命中已知同位素步
                delta_mz = abs(fj.precursor_mz - fi.precursor_mz)
                gap_type = classify_isotope_gap(
                    delta_mz,
                    classic_tol=config.isotope_mz_tolerance,
                    relaxed_tol=config.isotope_integer_step_tolerance,
                    max_step=config.isotope_max_step,
                )
                if gap_type is None:
                    continue

                # 找轻 / 重同位素, 后面 Tier 0 和 sanity check 都要用
                if fi.precursor_mz <= fj.precursor_mz:
                    f_light, f_heavy = fi, fj
                else:
                    f_light, f_heavy = fj, fi

                # C13 sanity check: 重同位素强度不应高于物理上限
                if not _intensity_ratio_ok(fi, fj, delta_mz):
                    continue

                # 基础门 2: EIC 共流出 (统一 Pearson + n_correlated)
                eic_i = _get_eic_and_width(fi)
                eic_j = _get_eic_and_width(fj)
                base_ok = False
                if eic_i is not None and eic_j is not None:
                    eic_a, rt_arr_a, w_a = eic_i
                    eic_b, rt_arr_b, w_b = eic_j
                    # 跨 segment 的 boundary 同位素对: fi / fj 各自从自己 segment
                    # 的 raw 抽 EIC, rt_array 长度可能差 ±1 cycle。统一用 fi 所在
                    # raw 重抽 fj 的 EIC, 保证两条 EIC 与 rt_arr_a 等长。fj 的 m/z
                    # 若落在 fi 段 MS1 范围之外, 重抽结果是全 0, Pearson 自然为 0,
                    # base_ok=False, 退回到 Tier 0 + Δm/z 去判定。
                    if len(eic_a) != len(eic_b):
                        raw_a = raw_lookup.get(
                            (fi.segment_name, fi.replicate_id)
                        )
                        if raw_a is not None:
                            _, eic_b = extract_ms1_eic(
                                raw_a, fj.precursor_mz, 0.5
                            )
                        else:
                            eic_b = None
                    if eic_b is not None and len(eic_a) == len(eic_b):
                        rt_start = min(fi.rt_left, fj.rt_left) - 0.1
                        rt_end = max(fi.rt_right, fj.rt_right) + 0.1
                        base_ok = eic_coelution_ok(
                            eic_a, eic_b, rt_arr_a, rt_start, rt_end,
                            peak_width_a_scans=w_a,
                            peak_width_b_scans=w_b,
                            pearson_threshold=config.isf_eic_pearson_threshold,
                        )

                # ASFAM Tier 0 增强: MS2 步态独立判定
                peaks_light = f_light.ms2_as_list()
                peaks_heavy = f_heavy.ms2_as_list()
                eff_delta = _nearest_isotope_step(delta_mz)
                top_n_dyn = min(
                    config.isotope_step_top_n,
                    max(1, len(peaks_light)),
                )
                step_match, step_total = ms2_isotope_step_score(
                    peaks_light, peaks_heavy,
                    isotope_delta=eff_delta,
                    mz_tolerance=config.isotope_step_mz_tolerance,
                    top_n=top_n_dyn,
                )
                step_ratio = step_match / step_total if step_total > 0 else 0.0
                tier0_ok = (
                    step_total >= 2
                    and step_match >= 2
                    and step_ratio >= config.isotope_step_min_ratio
                )

                if base_ok or tier0_ok:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
                    n_edges += 1

        # 连通分量 -> 按 apex RT 拆 (防止链式传递把 RT-far feature 拉到一起)
        raw_components = connected_components(adjacency)
        max_rt_gap = apex_rt_strict
        components: list[list[int]] = []
        for comp in raw_components:
            if len(comp) <= 1:
                components.append(comp)
                continue
            sorted_comp = sorted(comp, key=lambda idx: active[idx].rt_apex)
            sub = [sorted_comp[0]]
            for idx in sorted_comp[1:]:
                if active[idx].rt_apex - active[sub[-1]].rt_apex <= max_rt_gap:
                    sub.append(idx)
                else:
                    components.append(sub)
                    sub = [idx]
            components.append(sub)

        group_id = 0
        n_removed = 0
        for comp in components:
            if len(comp) <= 1:
                continue

            rep_idx = select_representative(
                comp,
                get_mz=lambda idx: active[idx].precursor_mz,
                get_intensity=lambda idx: active[idx].ms1_height or 0.0,
            )

            # select_representative picks the monoisotopic peak (lowest m/z), so
            # every member has delta >= 0 and only the representative has
            # delta == 0 -> isotope_index 0 (the `else 0` branch encodes that
            # rep-is-min-m/z contract). Charge state is assumed 1: classify_
            # isotope_gap only admits ~integer-Da gaps, so z>=2 envelopes never
            # cluster here; multi-charge would need round(delta * z / C13_DELTA).
            rep_mz = active[rep_idx].precursor_mz
            for idx in comp:
                active[idx].isotope_group_id = group_id
                active[idx].duplicate_group_id = group_id
                active[idx].duplicate_type = "isotope"
                delta = active[idx].precursor_mz - rep_mz
                active[idx].isotope_index = (
                    int(round(delta / C13_DELTA)) if delta > 0 else 0
                )
                if idx != rep_idx:
                    active[idx].status = "isotope_excluded"
                    active[idx].is_duplicate = True
                    n_removed += 1
            group_id += 1

        logger.info(
            "  Replicate %s: %d -> %d (%d isotope groups, %d removed, %d edges)",
            rep_id, n_before, n_before - n_removed, group_id, n_removed, n_edges,
        )

        if progress_callback:
            progress_callback("stage4", 1, 1, f"Rep {rep_id} done")

    return features_by_replicate


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _nearest_isotope_step(delta_mz: float) -> float:
    """把观测到的 Δm/z 对齐到最近的已知同位素步, 用于 Tier 0 MS2 回显查找。

    候选: ISOTOPE_DELTAS (C13 / N15 / S34 / ...) 以及 C13 × n (n ≤ 4)。
    超出 50 mDa 没匹配则退回到 ``round(Δm/z) × C13_DELTA``。
    """
    abs_d = abs(delta_mz)
    candidates: list[float] = []
    for name, d in ISOTOPE_DELTAS.items():
        if name == "C13":
            for n in range(1, 5):
                candidates.append(n * d)
        else:
            candidates.append(d)
    if candidates:
        best = min(candidates, key=lambda c: abs(c - abs_d))
        if abs(best - abs_d) <= 0.05:
            return best
    n_round = max(1, int(round(abs_d)))
    return n_round * C13_DELTA


def _intensity_ratio_ok(
    fa: CandidateFeature, fb: CandidateFeature, delta_mz: float,
) -> bool:
    """C13 同位素对的强度比 sanity check: 重同位素不应高于物理上限。"""
    if fa.ms1_height is None or fb.ms1_height is None:
        return True
    if 0.9 < delta_mz < 1.1:
        lighter = fa if fa.precursor_mz < fb.precursor_mz else fb
        heavier = fb if fa.precursor_mz < fb.precursor_mz else fa
        if lighter.ms1_height > 0:
            ratio = heavier.ms1_height / lighter.ms1_height
            max_ratio = max_c13_m1_ratio(lighter.precursor_mz)
            if ratio > max_ratio:
                return False
    return True
