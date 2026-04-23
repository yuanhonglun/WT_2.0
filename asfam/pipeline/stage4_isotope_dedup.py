"""Stage 4: Isotope deduplication."""
from __future__ import annotations

import logging
from typing import Optional, Callable

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import CandidateFeature, RawSegmentData
from asfam.core.mass_utils import (
    classify_isotope_gap, peak_overlap_ratio, max_c13_m1_ratio,
)
from asfam.constants import C13_DELTA, ISOTOPE_DELTAS
from asfam.core.similarity import (
    modified_cosine, neutral_loss_cosine, greedy_match,
    ms2_isotope_step_score,
)
from asfam.core.clustering import connected_components, select_representative

logger = logging.getLogger(__name__)


def _resolve_apex_rt_strict(
    config: ProcessingConfig,
    data_by_replicate: Optional[dict] = None,
) -> float:
    """Compute the hard apex-RT gate from the data's actual scan cycle time.

    isotope_apex_rt_n_cycles cycles * median(cycle_time). Falls back to
    isotope_apex_rt_fallback if cycle time cannot be determined.
    """
    n_cycles = max(1, int(getattr(config, "isotope_apex_rt_n_cycles", 2)))
    fallback = float(getattr(config, "isotope_apex_rt_fallback", 0.04))
    if not data_by_replicate:
        return fallback
    cycle_times: list[float] = []
    for segments in data_by_replicate.values():
        for raw in segments:
            try:
                rt = np.asarray(raw.rt_array, dtype=np.float64)
                if rt.size >= 2:
                    diffs = np.diff(rt)
                    diffs = diffs[diffs > 0]
                    if diffs.size:
                        cycle_times.append(float(np.median(diffs)))
            except Exception:
                continue
    if not cycle_times:
        return fallback
    cycle_time = float(np.median(cycle_times))
    return n_cycles * cycle_time


def run_stage4(
    features_by_replicate: dict[str, list[CandidateFeature]],
    config: ProcessingConfig,
    progress_callback: Optional[Callable] = None,
    data_by_replicate: Optional[dict] = None,
) -> dict[str, list[CandidateFeature]]:
    """Isotope deduplication for each replicate.

    Builds an isotope graph, finds connected components, keeps
    the representative (lowest m/z, highest intensity) from each group.

    Evidence tiers (in order of trust):
      Tier 0  — MS2 step-pattern: of the lighter feature's top-N high-response
                ions, at least `isotope_step_min_ratio` have a +isotope_delta
                ion in the heavier feature's MS2. Strongest signal for true
                isotope partners; survives intensity differences that defeat
                cosine.
      Tier 1  — MS1 isotope pattern explicitly links the two precursors.
      Tier 2  — Modified cosine (classic gaps): >= threshold + min matches
                + intensity-ratio plausibility.
      Tier 3  — Relaxed near-integer gap: requires step-pattern OR strict
                modified+NL cosine.

    All tiers also require apex RT within ~2 scan cycles (computed from the
    data's actual cycle time, see _resolve_apex_rt_strict).
    """
    apex_rt_strict = _resolve_apex_rt_strict(config, data_by_replicate)
    logger.info(
        "Stage 4: Isotope deduplication (apex_rt_strict=%.4f min)...",
        apex_rt_strict,
    )

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

                # Strict apex RT gate: isotopes must have very close apex RTs
                # (within ~2 scan cycles, see _resolve_apex_rt_strict).
                if rt_diff > apex_rt_strict:
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

                # Identify lighter / heavier feature for step-pattern check
                if fi.precursor_mz <= fj.precursor_mz:
                    f_light, f_heavy = fi, fj
                else:
                    f_light, f_heavy = fj, fi
                peaks_light = f_light.ms2_as_list()
                peaks_heavy = f_heavy.ms2_as_list()
                # Effective isotope delta to test in MS2 echo: snap to the
                # nearest known isotope step (default C13 = 1.003355).
                eff_delta = _nearest_isotope_step(delta_mz)

                # Tier 0 (primary): MS2 step-pattern echo of the lighter
                # spectrum into the heavier spectrum at the isotope delta.
                step_match, step_total = ms2_isotope_step_score(
                    peaks_light, peaks_heavy,
                    isotope_delta=eff_delta,
                    mz_tolerance=config.isotope_step_mz_tolerance,
                    top_n=config.isotope_step_top_n,
                )
                step_ratio = step_match / step_total if step_total > 0 else 0.0
                if (step_total >= 2
                        and step_match >= 2
                        and step_ratio >= config.isotope_step_min_ratio
                        and _intensity_ratio_ok(fi, fj, delta_mz)):
                    adjacency[i].add(j)
                    adjacency[j].add(i)
                    n_edges += 1
                    continue

                # Tier 1: MS1 isotope pattern explicitly links the precursors
                if _has_ms1_isotope_support(fi, fj, config.isotope_mz_tolerance,
                                            rt_tolerance=apex_rt_strict):
                    adjacency[i].add(j)
                    adjacency[j].add(i)
                    n_edges += 1
                    continue

                # Tier 2 — modified cosine fallback (classic gaps only).
                # Step-pattern is the primary detector; cosine here is a
                # safety net for spectra where the heavy-atom-bearing
                # fragments are not in the top-N of the lighter spectrum.
                if gap_type == "classic":
                    prec_i = fi.precursor_mz
                    prec_j = fj.precursor_mz
                    cos, n_matched = modified_cosine(
                        peaks_light, peaks_heavy, f_light.precursor_mz, f_heavy.precursor_mz,
                        config.isotope_fragment_mz_tolerance,
                        config.isotope_precursor_exclusion,
                    )
                    if (cos >= config.isotope_modified_cos_threshold
                            and n_matched >= config.isotope_min_matches
                            and _intensity_ratio_ok(fi, fj, delta_mz)):
                        adjacency[i].add(j)
                        adjacency[j].add(i)
                        n_edges += 1

        # Find connected components
        raw_components = connected_components(adjacency)

        # Split components by RT: isotopes must co-elute tightly.
        # Connected components can form transitive chains spanning a huge RT
        # range.  Sort each component by RT and break at any gap larger than
        # the strict apex tolerance — this cleanly separates RT-distant
        # features into independent sub-groups.
        max_rt_gap = apex_rt_strict
        components = []
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

            # Select representative
            rep_idx = select_representative(
                comp,
                get_mz=lambda idx: active[idx].precursor_mz,
                get_intensity=lambda idx: active[idx].ms1_height or 0.0,
            )

            for idx in comp:
                active[idx].isotope_group_id = group_id
                active[idx].duplicate_group_id = group_id
                active[idx].duplicate_type = "isotope"
                if idx != rep_idx:
                    active[idx].status = "isotope_removed"
                    active[idx].is_duplicate = True
                    n_removed += 1
            group_id += 1

        # Keep all features (removed ones are marked is_duplicate=True)
        # Downstream stages filter by status=="active" when building their
        # working set, so removed features won't interfere.

        logger.info(
            "  Replicate %s: %d -> %d (%d isotope groups, %d removed)",
            rep_id, n_before, n_before - n_removed, group_id, n_removed,
        )

        if progress_callback:
            progress_callback("stage4", 1, 1, f"Rep {rep_id} done")

    return features_by_replicate


def _nearest_isotope_step(delta_mz: float) -> float:
    """Snap an observed Δmz to the nearest known isotope step.

    Uses ISOTOPE_DELTAS (C13/N15/S34/...) and integer multiples of C13_DELTA
    up to step 4. Falls back to round(delta_mz) * C13_DELTA for unrecognised
    near-integer gaps so the step-pattern check still has a sensible target.
    """
    abs_d = abs(delta_mz)
    candidates: list[float] = []
    for name, d in ISOTOPE_DELTAS.items():
        if name == "C13":
            for n in range(1, 5):
                candidates.append(n * d)
        else:
            candidates.append(d)
    # Pick the nearest known step
    if candidates:
        best = min(candidates, key=lambda c: abs(c - abs_d))
        if abs(best - abs_d) <= 0.05:  # within 50 mDa is "the same step"
            return best
    # Fallback for unrecognised near-integer gaps
    n_round = max(1, int(round(abs_d)))
    return n_round * C13_DELTA


def _has_ms1_isotope_support(
    fa: CandidateFeature, fb: CandidateFeature, tol: float,
    rt_tolerance: float = 0.05,
) -> bool:
    """Check if MS1 isotope patterns support a relationship."""
    # Require RT proximity — true isotopes co-elute
    if abs(fa.rt_apex - fb.rt_apex) > rt_tolerance:
        return False

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
