"""Stage 5: Adduct deduplication."""
from __future__ import annotations

import logging
from typing import Optional, Callable

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import RawSegmentData, CandidateFeature
from asfam.core.mass_utils import check_adduct_pair
from asfam.core.eic import extract_ms1_eic
from asfam.core.clustering import connected_components
from metabo_core.algorithms.dedup_relations import eic_coelution_ok

logger = logging.getLogger(__name__)


def run_stage5(
    features_by_replicate: dict[str, list[CandidateFeature]],
    data_by_replicate: dict[str, list[RawSegmentData]],
    config: ProcessingConfig,
    progress_callback: Optional[Callable] = None,
) -> dict[str, list[CandidateFeature]]:
    """Adduct deduplication for each replicate."""
    logger.info("Stage 5: Adduct deduplication...")

    # Build raw data lookup
    raw_lookup: dict[tuple[str, int], RawSegmentData] = {}
    for rep_id, segments in data_by_replicate.items():
        for seg in segments:
            raw_lookup[(seg.segment_name, seg.replicate_id)] = seg

    for rep_id, features in features_by_replicate.items():
        active = [f for f in features if f.status == "active"]
        n_before = len(active)

        # Group by RT clusters
        active.sort(key=lambda f: f.rt_apex)
        rt_groups = _group_by_rt(active, config.adduct_rt_tolerance)

        adjacency: dict[int, set[int]] = {i: set() for i in range(len(active))}
        adduct_labels: dict[int, tuple[str, str]] = {}
        n_edges = 0

        # 同一 rep 内复用 EIC: 同一 feature 会出现在多个候选 adduct 对中。
        # key = (id(raw_data), feature_id); 避免对同一 (raw, mz) 反复扫描。
        eic_cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}

        def _cached_eic(raw_data, feat):
            key = (id(raw_data), feat.feature_id)
            cached = eic_cache.get(key)
            if cached is None:
                cached = extract_ms1_eic(raw_data, feat.precursor_mz, 0.5)
                eic_cache[key] = cached
            return cached

        for group_indices in rt_groups:
            if len(group_indices) < 2:
                continue

            for ii in range(len(group_indices)):
                i = group_indices[ii]
                fi = active[i]
                for jj in range(ii + 1, len(group_indices)):
                    j = group_indices[jj]
                    fj = active[j]

                    # Direct pairwise apex-RT gate (stage4/stage6 have one;
                    # stage5 historically relied only on the running-median
                    # bucketing, whose chaining lets a bucket span > tolerance
                    # so EIC-tail overlap can link RT-separated peaks — e.g.
                    # [M+Na]+/[M+NH4]+ 0.17 min apart). Adds the missing direct
                    # apex-RT constraint without touching coelution / bucketing.
                    if abs(fi.rt_apex - fj.rt_apex) > config.adduct_rt_tolerance:
                        continue

                    # Check if m/z pair matches adduct rules
                    pair = check_adduct_pair(
                        fi.precursor_mz, fj.precursor_mz,
                        config.ionization_mode, config.adduct_mw_tolerance,
                    )
                    if pair is None:
                        continue

                    # EIC 共流出验证 (统一 Pearson + 自适应 n_correlated 门)
                    raw_data = raw_lookup.get(
                        (fi.segment_name, fi.replicate_id)
                    )
                    if raw_data is not None:
                        if not _check_coelution(fi, fj, raw_data, config, _cached_eic):
                            continue

                    adjacency[i].add(j)
                    adjacency[j].add(i)
                    adduct_labels[(i, j)] = pair
                    n_edges += 1

        # Find connected components
        components = connected_components(adjacency)
        group_id = 0
        n_removed = 0

        for comp in components:
            if len(comp) <= 1:
                continue

            # Keep highest intensity feature
            rep_idx = max(comp, key=lambda idx: active[idx].ms1_height or 0.0)

            for idx in comp:
                active[idx].adduct_group_id = group_id
                active[idx].duplicate_group_id = group_id + 100000  # offset to avoid collision with isotope ids
                active[idx].duplicate_type = "adduct"
                if idx != rep_idx:
                    active[idx].status = "adduct_excluded"
                    active[idx].is_duplicate = True
                    # Try to assign adduct type from labels
                    key1 = (min(idx, rep_idx), max(idx, rep_idx))
                    if key1 in adduct_labels:
                        pair = adduct_labels[key1]
                        if idx < rep_idx:
                            active[idx].adduct_type = pair[0]
                            active[rep_idx].adduct_type = pair[1]
                        else:
                            active[idx].adduct_type = pair[1]
                            active[rep_idx].adduct_type = pair[0]
                    n_removed += 1
            group_id += 1

        # Keep all features (removed ones are marked is_duplicate=True)

        logger.info(
            "  Replicate %s: %d -> %d (%d adduct groups, %d removed)",
            rep_id, n_before, n_before - n_removed, group_id, n_removed,
        )

    return features_by_replicate


def _group_by_rt(
    features: list[CandidateFeature], tolerance: float,
) -> list[list[int]]:
    """Group features by RT proximity. Features must be sorted by rt_apex."""
    groups: list[list[int]] = []
    current: list[int] = [0] if features else []

    for i in range(1, len(features)):
        median_rt = np.median([features[j].rt_apex for j in current])
        if abs(features[i].rt_apex - median_rt) <= tolerance:
            current.append(i)
        else:
            groups.append(current)
            current = [i]

    if current:
        groups.append(current)
    return groups


def _check_coelution(
    fa: CandidateFeature,
    fb: CandidateFeature,
    raw_data: RawSegmentData,
    config: ProcessingConfig,
    eic_provider=None,
) -> bool:
    """统一 EIC 共流出判定: Pearson + 自适应 n_correlated 门。

    ``eic_provider``: optional ``callable(raw_data, feature) -> (rt_arr, eic)``
    用于跨 pair 复用同一 feature 的 EIC; 若为 ``None`` 则现场计算 (向后兼容)。
    """
    rt_start = min(fa.rt_left, fb.rt_left) - 0.1
    rt_end = max(fa.rt_right, fb.rt_right) + 0.1
    if eic_provider is not None:
        rt_arr, eic_a = eic_provider(raw_data, fa)
        _, eic_b = eic_provider(raw_data, fb)
    else:
        rt_arr, eic_a = extract_ms1_eic(raw_data, fa.precursor_mz, 0.5)
        _, eic_b = extract_ms1_eic(raw_data, fb.precursor_mz, 0.5)

    # 峰宽 (scan 数): rt_left / rt_right 在 rt_array 上的区间长度
    w_a = _peak_width_scans(rt_arr, fa.rt_left, fa.rt_right)
    w_b = _peak_width_scans(rt_arr, fb.rt_left, fb.rt_right)

    return eic_coelution_ok(
        eic_a, eic_b, rt_arr, rt_start, rt_end,
        peak_width_a_scans=w_a,
        peak_width_b_scans=w_b,
        pearson_threshold=config.adduct_eic_pearson_threshold,
    )


def _peak_width_scans(rt_array: np.ndarray, rt_left: float, rt_right: float) -> int:
    lo = int(np.searchsorted(rt_array, rt_left, side="left"))
    hi = int(np.searchsorted(rt_array, rt_right, side="right"))
    return max(1, hi - lo)
