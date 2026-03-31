"""Stage 5b: Duplicate detection — flag near-RT features with similar MS2.

After isotope and adduct deduplication, some features may still represent the
same compound detected in overlapping ASFAM segments or adjacent precursor
windows.  This stage identifies such duplicates by comparing MS2 cosine
similarity among features with close RT and m/z.

Duplicates are flagged (is_duplicate=True) but NOT removed from the active
feature list, so they remain available for inspection in the GUI.
"""
from __future__ import annotations

import logging
from typing import Optional, Callable

from asfam.config import ProcessingConfig
from asfam.models import CandidateFeature
from asfam.core.similarity import cosine_similarity
from asfam.core.clustering import connected_components

logger = logging.getLogger(__name__)


def run_stage5b(
    features_by_replicate: dict[str, list[CandidateFeature]],
    config: ProcessingConfig,
    progress_callback: Optional[Callable] = None,
) -> dict[str, list[CandidateFeature]]:
    """Flag duplicate features within each replicate.

    Two active features are linked as duplicates if:
      1. |RT_a - RT_b| <= duplicate_rt_tolerance  (default 0.2 min)
      2. |mz_a - mz_b| <= duplicate_mz_tolerance  (default 0.5 Da)
      3. cosine(ms2_a, ms2_b) >= duplicate_cosine_threshold  (default 0.85)
         with >= duplicate_min_matched matched peaks

    Connected components of the duplicate graph are grouped.  Within each
    group, the feature with the highest MS1 intensity is kept as
    representative; others get is_duplicate=True.

    Returns the same dict (modified in place).
    """
    logger.info("Stage 5b: Duplicate detection...")
    total_flagged = 0

    for rep_id, features in features_by_replicate.items():
        active = [f for f in features if f.status == "active"]
        if len(active) < 2:
            continue

        # Sort by RT for efficient pairwise search
        active.sort(key=lambda f: f.rt_apex)

        adjacency: dict[int, set[int]] = {}
        n_edges = 0

        for i in range(len(active)):
            fi = active[i]
            peaks_i = fi.ms2_as_list()
            if len(peaks_i) < config.duplicate_min_matched:
                continue

            for j in range(i + 1, len(active)):
                fj = active[j]

                # RT check (sorted, so can break early)
                rt_diff = abs(fj.rt_apex - fi.rt_apex)
                if rt_diff > config.duplicate_rt_tolerance:
                    break

                # m/z check
                mz_diff = abs(fj.precursor_mz - fi.precursor_mz)
                if mz_diff > config.duplicate_mz_tolerance:
                    continue

                # MS2 cosine similarity
                peaks_j = fj.ms2_as_list()
                if len(peaks_j) < config.duplicate_min_matched:
                    continue

                cos, n_matched = cosine_similarity(
                    peaks_i, peaks_j, config.eic_mz_tolerance,
                )
                if cos >= config.duplicate_cosine_threshold and n_matched >= config.duplicate_min_matched:
                    adjacency.setdefault(i, set()).add(j)
                    adjacency.setdefault(j, set()).add(i)
                    n_edges += 1

        if not adjacency:
            logger.info("  Replicate %s: 0 duplicate groups", rep_id)
            continue

        # Find connected components
        components = connected_components(adjacency)
        group_id_base = total_flagged
        n_groups = 0
        n_flagged = 0

        for comp in components:
            if len(comp) <= 1:
                continue
            n_groups += 1
            gid = group_id_base + n_groups

            # Select representative: highest MS1 height, then most fragments
            rep_idx = max(
                comp,
                key=lambda idx: (active[idx].ms1_height or 0, active[idx].n_fragments),
            )

            for idx in comp:
                active[idx].duplicate_group_id = gid
                if idx != rep_idx:
                    active[idx].is_duplicate = True
                    n_flagged += 1

        total_flagged += n_flagged
        logger.info(
            "  Replicate %s: %d duplicate groups, %d features flagged",
            rep_id, n_groups, n_flagged,
        )

    logger.info("  Total flagged duplicates: %d", total_flagged)
    return features_by_replicate
