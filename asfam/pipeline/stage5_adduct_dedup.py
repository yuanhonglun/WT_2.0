"""Stage 5: Adduct deduplication."""
from __future__ import annotations

import logging
from typing import Optional, Callable

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import RawSegmentData, CandidateFeature
from asfam.core.mass_utils import check_adduct_pair
from asfam.core.eic import extract_ms1_eic
from asfam.core.similarity import eic_pearson_in_range
from asfam.core.clustering import connected_components

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

        for group_indices in rt_groups:
            if len(group_indices) < 2:
                continue

            for ii in range(len(group_indices)):
                i = group_indices[ii]
                fi = active[i]
                for jj in range(ii + 1, len(group_indices)):
                    j = group_indices[jj]
                    fj = active[j]

                    # Check if m/z pair matches adduct rules
                    pair = check_adduct_pair(
                        fi.precursor_mz, fj.precursor_mz,
                        config.ionization_mode, config.adduct_mw_tolerance,
                    )
                    if pair is None:
                        continue

                    # EIC correlation validation
                    raw_data = raw_lookup.get(
                        (fi.segment_name, fi.replicate_id)
                    )
                    if raw_data is not None:
                        corr = _compute_eic_correlation(
                            fi, fj, raw_data, config,
                        )
                        if corr < config.adduct_eic_pearson_threshold:
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
                if idx != rep_idx:
                    active[idx].status = "adduct_removed"
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

        features_by_replicate[rep_id] = [
            f for f in features if f.status == "active"
        ]

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


def _compute_eic_correlation(
    fa: CandidateFeature,
    fb: CandidateFeature,
    raw_data: RawSegmentData,
    config: ProcessingConfig,
) -> float:
    """Compute EIC Pearson correlation between two features."""
    rt_start = min(fa.rt_left, fb.rt_left) - 0.1
    rt_end = max(fa.rt_right, fb.rt_right) + 0.1

    _, eic_a = extract_ms1_eic(raw_data, fa.precursor_mz, 0.5)
    _, eic_b = extract_ms1_eic(raw_data, fb.precursor_mz, 0.5)

    r, _ = eic_pearson_in_range(eic_a, eic_b, raw_data.rt_array, rt_start, rt_end)
    return r
