"""Stage 6: In-source fragmentation (ISF) detection and removal."""
from __future__ import annotations

import logging
from typing import Optional, Callable

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import RawSegmentData, CandidateFeature
from asfam.core.eic import extract_ms1_eic
from asfam.core.similarity import eic_pearson_in_range

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

        for i in range(len(active)):
            child = active[i]
            if child.status != "active":
                continue

            child_mz = child.precursor_mz

            for j in range(len(active)):
                if i == j:
                    continue
                parent = active[j]
                if parent.status != "active":
                    continue

                # Parent must have larger m/z
                if parent.precursor_mz <= child_mz:
                    continue

                # RT proximity check
                if abs(parent.rt_apex - child.rt_apex) > config.adduct_rt_tolerance:
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

                _, eic_child = extract_ms1_eic(
                    raw_data, child_mz, 0.5,
                )
                _, eic_parent = extract_ms1_eic(
                    raw_data, parent.precursor_mz, 0.5,
                )

                r, n_pts = eic_pearson_in_range(
                    eic_child, eic_parent, raw_data.rt_array,
                    rt_start, rt_end,
                )

                if r >= config.isf_eic_pearson_threshold and \
                   n_pts >= config.isf_min_correlated_scans:
                    child.status = "isf_removed"
                    child.isf_parent_id = parent.feature_id
                    n_removed += 1
                    break  # child is ISF, stop looking for parents

        features_by_replicate[rep_id] = [
            f for f in features if f.status == "active"
        ]

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
