"""Stage 4: Isotope deduplication."""
from __future__ import annotations

import logging
from typing import Optional, Callable

from asfam.config import ProcessingConfig
from asfam.models import CandidateFeature
from asfam.core.mass_utils import (
    classify_isotope_gap, peak_overlap_ratio, max_c13_m1_ratio,
)
from asfam.core.similarity import modified_cosine, neutral_loss_cosine
from asfam.core.clustering import connected_components, select_representative

logger = logging.getLogger(__name__)


def run_stage4(
    features_by_replicate: dict[str, list[CandidateFeature]],
    config: ProcessingConfig,
    progress_callback: Optional[Callable] = None,
) -> dict[str, list[CandidateFeature]]:
    """Isotope deduplication for each replicate.

    Builds an isotope graph, finds connected components, keeps
    the representative (lowest m/z, highest intensity) from each group.
    """
    logger.info("Stage 4: Isotope deduplication...")

    for rep_id, features in features_by_replicate.items():
        active = [f for f in features if f.status == "active"]
        n_before = len(active)

        # Build isotope graph
        adjacency: dict[int, set[int]] = {i: set() for i in range(len(active))}
        n_edges = 0

        # Sort by RT for efficient pairwise comparison
        active.sort(key=lambda f: (f.rt_apex, f.precursor_mz))

        for i in range(len(active)):
            fi = active[i]
            for j in range(i + 1, len(active)):
                fj = active[j]

                # RT check (early termination since sorted by RT)
                rt_diff = abs(fj.rt_apex - fi.rt_apex)
                if rt_diff > config.isotope_rt_tolerance * 3:
                    break

                if rt_diff > config.isotope_rt_tolerance:
                    continue

                # Peak overlap check
                overlap = peak_overlap_ratio(
                    fi.rt_left, fi.rt_right, fj.rt_left, fj.rt_right,
                )
                if overlap < config.isotope_overlap_ratio:
                    continue

                # m/z gap classification
                delta_mz = abs(fj.precursor_mz - fi.precursor_mz)
                gap_type = classify_isotope_gap(
                    delta_mz,
                    classic_tol=config.isotope_mz_tolerance,
                    relaxed_tol=config.isotope_integer_step_tolerance,
                    max_step=config.isotope_max_step,
                )
                if gap_type is None:
                    continue

                # MS1 isotope support check
                if _has_ms1_isotope_support(fi, fj, config.isotope_mz_tolerance):
                    adjacency[i].add(j)
                    adjacency[j].add(i)
                    n_edges += 1
                    continue

                # MS2 spectral similarity check
                peaks_i = fi.ms2_as_list()
                peaks_j = fj.ms2_as_list()
                prec_i = fi.precursor_mz
                prec_j = fj.precursor_mz

                if gap_type == "classic":
                    cos, n_matched = modified_cosine(
                        peaks_i, peaks_j, prec_i, prec_j,
                        config.isotope_fragment_mz_tolerance,
                        config.isotope_precursor_exclusion,
                    )
                    if cos >= config.isotope_modified_cos_threshold and \
                       n_matched >= config.isotope_min_matches:
                        # Intensity ratio check for C13
                        if _intensity_ratio_ok(fi, fj, delta_mz):
                            adjacency[i].add(j)
                            adjacency[j].add(i)
                            n_edges += 1

                elif gap_type == "relaxed":
                    cos, n_matched = modified_cosine(
                        peaks_i, peaks_j, prec_i, prec_j,
                        config.isotope_fragment_mz_tolerance,
                        config.isotope_precursor_exclusion,
                    )
                    nl_cos, nl_matched = neutral_loss_cosine(
                        peaks_i, peaks_j, prec_i, prec_j,
                        config.isotope_fragment_mz_tolerance,
                    )
                    if (cos >= config.isotope_modified_cos_relaxed and
                            n_matched >= config.isotope_min_matches_relaxed and
                            nl_cos >= config.isotope_nl_cos_threshold and
                            nl_matched >= config.isotope_min_nl_matches):
                        adjacency[i].add(j)
                        adjacency[j].add(i)
                        n_edges += 1

        # Find connected components
        components = connected_components(adjacency)
        group_id = 0
        n_removed = 0
        for comp in components:
            if len(comp) <= 1:
                continue

            # Select representative
            rep_idx = select_representative(
                comp,
                get_mz=lambda idx: active[idx].precursor_mz,
                get_intensity=lambda idx: active[idx].ms1_height or 0.0,
            )

            for idx in comp:
                active[idx].isotope_group_id = group_id
                if idx != rep_idx:
                    active[idx].status = "isotope_removed"
                    n_removed += 1
            group_id += 1

        # Update feature list: keep active only
        features_by_replicate[rep_id] = [
            f for f in features if f.status == "active"
        ]

        logger.info(
            "  Replicate %s: %d -> %d (%d isotope groups, %d removed)",
            rep_id, n_before, n_before - n_removed, group_id, n_removed,
        )

        if progress_callback:
            progress_callback("stage4", 1, 1, f"Rep {rep_id} done")

    return features_by_replicate


def _has_ms1_isotope_support(
    fa: CandidateFeature, fb: CandidateFeature, tol: float,
) -> bool:
    """Check if MS1 isotope patterns support a relationship."""
    if fa.ms1_isotopes and fb.ms1_precursor_mz is not None:
        for mz, _ in fa.ms1_isotopes:
            if abs(mz - fb.ms1_precursor_mz) <= tol:
                return True
    if fb.ms1_isotopes and fa.ms1_precursor_mz is not None:
        for mz, _ in fb.ms1_isotopes:
            if abs(mz - fa.ms1_precursor_mz) <= tol:
                return True
    return False


def _intensity_ratio_ok(
    fa: CandidateFeature, fb: CandidateFeature, delta_mz: float,
) -> bool:
    """Validate intensity ratio for C13-like isotope pairs."""
    # Only validate if both have MS1 heights
    if fa.ms1_height is None or fb.ms1_height is None:
        return True  # can't validate, assume OK

    # For C13 (delta ~1.003): lighter isotope should be more intense
    if 0.9 < delta_mz < 1.1:
        lighter = fa if fa.precursor_mz < fb.precursor_mz else fb
        heavier = fb if fa.precursor_mz < fb.precursor_mz else fa
        if lighter.ms1_height > 0:
            ratio = heavier.ms1_height / lighter.ms1_height
            max_ratio = max_c13_m1_ratio(lighter.precursor_mz)
            if ratio > max_ratio:
                return False
    return True
