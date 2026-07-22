"""Stage 6: In-source fragmentation (ISF) detection and removal."""
from __future__ import annotations

import logging
from typing import Optional, Callable

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import RawSegmentData, CandidateFeature
from asfam.core.eic import extract_ms1_eic
from asfam.core.similarity import eic_pearson_in_range
from metabo_core.algorithms.dedup_relations import (
    adaptive_n_correlated_threshold,
)

logger = logging.getLogger(__name__)


def run_stage6(
    features_by_replicate: dict[str, list[CandidateFeature]],
    data_by_replicate: dict[str, list[RawSegmentData]],
    config: ProcessingConfig,
    progress_callback: Optional[Callable] = None,
) -> dict[str, list[CandidateFeature]]:
    """Detect and remove in-source fragmentation artifacts.

    Uses MassCube dual-criteria approach:
    1. Candidate ISF's m/z appears in parent's MS2 product ions
    2. EIC scan-by-scan Pearson correlation >= threshold
    """
    logger.info("Stage 6: In-source fragmentation detection...")

    raw_lookup: dict[tuple[str, int], RawSegmentData] = {}
    for rep_id, segments in data_by_replicate.items():
        for seg in segments:
            raw_lookup[(seg.segment_name, seg.replicate_id)] = seg

    for rep_id, features in features_by_replicate.items():
        active = [f for f in features if f.status == "active"]
        n_before = len(active)

        # Sort by m/z ascending (smaller m/z = potential ISF child)
        active.sort(key=lambda f: f.precursor_mz)
        n_removed = 0

        # 同一 rep 内 EIC + 峰宽缓存: 一个 feature 会反复作为不同 (child,parent)
        # 对中的成员被查询; 复用避免对同一 (raw, precursor_mz) 反复扫描。
        # value = (eic_intensities, peak_width_scans)
        eic_cache: dict[tuple[int, str], tuple[np.ndarray, int]] = {}

        def _eic_and_width(raw_data, feat):
            key = (id(raw_data), feat.feature_id)
            cached = eic_cache.get(key)
            if cached is not None:
                return cached
            _, eic = extract_ms1_eic(raw_data, feat.precursor_mz, 0.5)
            rt_arr = raw_data.rt_array
            w = max(
                1,
                int(np.searchsorted(rt_arr, feat.rt_right, side="right"))
                - int(np.searchsorted(rt_arr, feat.rt_left, side="left")),
            )
            cached = (eic, w)
            eic_cache[key] = cached
            return cached

        for i in range(len(active)):
            child = active[i]
            if child.status != "active":
                continue

            child_mz = child.precursor_mz

            # active 已按 precursor_mz 升序; 所有 j <= i 的 parent.precursor_mz
            # 都 <= child_mz, 必然不通过下方 `parent.precursor_mz > child_mz`
            # 关卡, 直接从 i+1 开始可省去 i 次循环 (i 与 j 都是 0..N-1)。
            for j in range(i + 1, len(active)):
                parent = active[j]
                if parent.status != "active":
                    continue

                # Parent must have larger m/z
                if parent.precursor_mz <= child_mz:
                    continue

                # RT proximity check (dedicated ISF gate, not the adduct one)
                if abs(parent.rt_apex - child.rt_apex) > config.isf_rt_tolerance:
                    continue

                # Criterion 1: child's m/z appears in parent's MS2
                if not _mz_in_ms2(child_mz, parent.ms2_mz, config.isf_ms2_mz_tolerance):
                    continue

                # Criterion 2: EIC correlation
                raw_data = raw_lookup.get(
                    (child.segment_name, child.replicate_id)
                )
                if raw_data is None:
                    continue

                rt_start = min(child.rt_left, parent.rt_left) - 0.1
                rt_end = max(child.rt_right, parent.rt_right) + 0.1

                eic_child, w_child = _eic_and_width(raw_data, child)
                eic_parent, w_parent = _eic_and_width(raw_data, parent)

                # 自适应 n_correlated 门 (统一底座): max(5, 0.5 × min(peak_width))
                # 替代旧的硬限 10 scans, 对窄峰更友好、对宽峰更严格
                needed_n = adaptive_n_correlated_threshold(w_child, w_parent)

                r, n_pts = eic_pearson_in_range(
                    eic_child, eic_parent, raw_data.rt_array,
                    rt_start, rt_end,
                )

                if r >= config.isf_eic_pearson_threshold and n_pts >= needed_n:
                    child.status = "isf_excluded"
                    child.is_duplicate = True
                    child.duplicate_type = "isf"
                    child.isf_parent_id = parent.feature_id
                    # Link parent and child via duplicate_group_id
                    isf_gid = 200000 + n_removed
                    child.duplicate_group_id = isf_gid
                    if parent.duplicate_group_id is None:
                        parent.duplicate_group_id = isf_gid
                        parent.duplicate_type = parent.duplicate_type or "isf"
                    n_removed += 1
                    break  # child is ISF, stop looking for parents

        # Keep all features (removed ones are marked is_duplicate=True)

        logger.info(
            "  Replicate %s: %d -> %d (%d ISF removed)",
            rep_id, n_before, n_before - n_removed, n_removed,
        )

    return features_by_replicate


def _mz_in_ms2(
    target_mz: float,
    ms2_mz_array: np.ndarray,
    tolerance: float,
) -> bool:
    """Check if target m/z appears in an MS2 spectrum."""
    if len(ms2_mz_array) == 0:
        return False
    return bool(np.any(np.abs(ms2_mz_array - target_mz) <= tolerance))
