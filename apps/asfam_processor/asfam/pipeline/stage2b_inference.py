"""Stage 2.5: precursor m/z for MS2-only features.

Neutral-loss consensus first (optionally preceded by library matching), the
stage-2 isolation-window centroid as the last fallback. See ``run_stage2b``.
"""
from __future__ import annotations

import logging
from typing import Optional, Callable

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import CandidateFeature
from asfam.core.similarity import composite_similarity
from metabo_core.annotation import load_library_lean

logger = logging.getLogger(__name__)

# Common neutral losses (name, exact mass) for NL consensus method
NEUTRAL_LOSSES = [
    ("H", 1.00783), ("H2", 2.01565), ("H2O", 18.01056), ("2H2O", 36.02113),
    ("NH3", 17.02655), ("CO", 27.99491), ("HCN", 27.01090), ("CH2O", 30.01056),
    ("CO2", 43.98983), ("C2H2O", 42.01057), ("CH3", 15.02348), ("C2H4", 28.03130),
    ("C2H2", 26.01565), ("CHO2", 44.99765), ("C3H6", 42.04695),
    ("CH3OH", 32.02621), ("C2H4O", 44.02621), ("CH2O2", 46.00548),
    ("C2H5", 29.03913), ("C3H4", 40.03130), ("NH2", 16.01872),
    ("NO", 29.99799), ("NO2", 45.99290), ("HCl", 35.97668),
    ("SO2", 63.96190), ("SO3", 79.95682), ("C2H3N", 41.02655),
    ("C3H3N", 53.02655), ("C4H8", 56.06260), ("CH4O", 32.02621),
]


#: Counter names ``run_stage2b`` reports through ``stats`` (and the orchestrator
#: prints in ``processing_report.txt``), in report order.
STAGE2B_COUNTERS = (
    "ms2_only_in",
    "library_assigned",
    "nl_assigned",
    "window_fallback",
    "no_precursor",
    "retained_fragment_poor",
    "discarded_fragment_poor",
)


def run_stage2b(
    features_by_replicate: dict[str, list[CandidateFeature]],
    config: ProcessingConfig,
    spectral_library_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    preloaded_library: Optional[list] = None,
    stats: Optional[dict] = None,
) -> dict[str, list[CandidateFeature]]:
    """Give the MS2-only features stage 2 left without a precursor one.

    Order of evidence (the design order; restored in v1.4.2610xx after the
    2026-06-25 stage-2 fallback had been pre-empting this stage):

    1. ``ms1_peak`` / ``ms1_relaxed`` — assigned in stage 2 from a detected MS1
       peak; such features already carry ``mz_source`` and never enter here.
    2. Library-matching inference (``enable_library_mz_inference``, off by
       default): ``mz_source="library"``.
    3. Neutral-loss consensus, for features with at least
       ``min_fragments_for_inference`` fragments (:func:`_nl_consensus_predict`):
       ``mz_source="nl_consensus"``, ``mz_confidence`` high / medium / low for
       >= 3 / 2 / 1 supporting fragments, the count itself in ``nl_n_support``.
    4. Isolation-window centroid stored by stage 2 (``ms1_window_mz``), for
       whatever is still without a source — NL failed, or too few fragments:
       ``ms1_precursor_mz`` takes the centroid, ``mz_source="ms1_lowres"``,
       ``mz_confidence="low"``. Last, because it is an intensity-weighted
       mixture centroid of the whole 1-Da window.
    5. Nothing: ``precursor_mz`` stays the integer channel (``mz_source=""``).

    Fragment-poor features (fewer than ``min_fragments_for_inference``
    fragments) cannot take step 3. With ``retain_fragment_poor_ms2_only`` (the
    default) they go straight to step 4 / 5, stay in every table and are
    labelled tier D at export; with the flag off the pre-v1.4 behaviour returns
    and they are dropped (``status="discarded_few_fragments"``, the one true
    deletion in the pipeline).

    ``stats``, when given, receives the :data:`STAGE2B_COUNTERS` summed over
    the replicates passed in.
    """
    logger.info("Stage 2.5: MS2-only m/z inference...")

    library = preloaded_library
    if (library is None and spectral_library_path
            and getattr(config, "enable_library_mz_inference", False)):
        library = _load_library(spectral_library_path)
    elif not getattr(config, "enable_library_mz_inference", False):
        # Library-matching inference disabled by config — skip library step entirely.
        library = None

    retain_poor = bool(getattr(config, "retain_fragment_poor_ms2_only", True))
    min_frags = int(config.min_fragments_for_inference)
    totals = {k: 0 for k in STAGE2B_COUNTERS}

    for rep_id, features in features_by_replicate.items():
        ms2_only = [f for f in features if f.signal_type == "ms2_only"
                     and not f.mz_source]
        if not ms2_only:
            continue

        logger.info("  Replicate %s: %d ms2_only signals to process", rep_id, len(ms2_only))

        counts = {k: 0 for k in STAGE2B_COUNTERS}
        counts["ms2_only_in"] = len(ms2_only)

        for i, feat in enumerate(ms2_only):
            if progress_callback and i % 50 == 0:
                progress_callback("stage2b", i, len(ms2_only), f"Rep {rep_id}: inferring m/z")

            if feat.n_fragments < min_frags:
                if not retain_poor:
                    feat.status = "discarded_few_fragments"
                    counts["discarded_fragment_poor"] += 1
                    continue
                counts["retained_fragment_poor"] += 1
            else:
                # Step 2: Library matching (opt-in)
                if library is not None:
                    match = _library_match(feat, library, config)
                    if match is not None:
                        feat.inferred_mz = match["precursor_mz"]
                        feat.matchms_score = match["score"]
                        feat.matchms_name = match.get("name", "")
                        feat.inferred_formula = match.get("formula", "")
                        feat.mz_source = "library"
                        counts["library_assigned"] += 1
                        continue

                # Step 3: Neutral loss consensus
                nl_result = _nl_consensus_predict(feat)
                if nl_result is not None:
                    feat.inferred_mz = nl_result["mz"]
                    feat.mz_source = "nl_consensus"
                    feat.mz_confidence = nl_result["confidence"]
                    feat.nl_n_support = int(nl_result["n_support"])
                    counts["nl_assigned"] += 1
                    continue

            # Step 4 / 5: window-centroid fallback, else the integer channel.
            if _apply_window_fallback(feat):
                counts["window_fallback"] += 1
            else:
                counts["no_precursor"] += 1

        if not retain_poor:
            features_by_replicate[rep_id] = [
                f for f in features if f.status != "discarded_few_fragments"
            ]

        logger.info(
            "  Replicate %s: %d library, %d NL consensus, %d window-centroid "
            "fallback, %d no precursor; fragment-poor %d retained / %d discarded",
            rep_id, counts["library_assigned"], counts["nl_assigned"],
            counts["window_fallback"], counts["no_precursor"],
            counts["retained_fragment_poor"], counts["discarded_fragment_poor"],
        )
        for k in STAGE2B_COUNTERS:
            totals[k] += counts[k]

    if stats is not None:
        stats.update(totals)
    return features_by_replicate


def _apply_window_fallback(feature: CandidateFeature) -> bool:
    """Make the stored isolation-window centroid the reported precursor.

    Only for a feature that still has no precursor at all; the centroid goes
    into ``ms1_precursor_mz`` (as the pre-v1.4 stage-2 fallback did, so the
    GUI, alignment corroboration and export read it from the same place) with
    ``mz_source="ms1_lowres"`` / ``mz_confidence="low"``. Returns False when
    stage 2 stored no centroid (no MS1 signal in the window).
    """
    if feature.ms1_precursor_mz is not None or feature.mz_source:
        return False
    mz = getattr(feature, "ms1_window_mz", None)
    if mz is None:
        return False
    feature.ms1_precursor_mz = float(mz)
    feature.mz_source = "ms1_lowres"
    feature.mz_confidence = "low"
    return True


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def _load_library(library_path: str) -> Optional[list[dict]]:
    """Load the spectral library as a flat list of lean spectra.

    Thin wrapper over ``metabo_core.annotation.load_library_lean``: it adds
    only the ``_mz_int`` cache that :func:`_library_match` uses to skip
    candidates outside the feature's nominal MS1 channel.

    Returns the *flat list*, not an index — stage 2.5 walks it directly,
    while stage 6.5 builds an index from the same list via
    ``build_index_from_list``. Sharing one load between the two stages is
    what makes the preload in the orchestrator worthwhile.

    The orchestrator owns the release: this list must be dropped right
    after stage 6.5 (see the library-lifetime invariant in CLAUDE.md).
    """
    spectra = load_library_lean(library_path)
    if spectra is None:
        return None

    for spec in spectra:
        pmz = spec.get("metadata", {}).get("precursor_mz")
        spec["_mz_int"] = int(round(float(pmz))) if pmz is not None else None

    logger.info("Loaded %d spectra from library %s", len(spectra), library_path)
    return spectra


# ---------------------------------------------------------------------------
# Library matching
# ---------------------------------------------------------------------------

def _library_match(
    feature: CandidateFeature,
    library: list[dict],
    config: ProcessingConfig,
) -> Optional[dict]:
    """Match a feature's MS2 against library, restricted by MS1 channel."""
    channel = feature.precursor_mz_nominal
    query_peaks = list(zip(feature.ms2_mz.tolist(), feature.ms2_intensity.tolist()))

    best_score = 0.0
    best_match = None

    for spec in library:
        ref_mz_int = spec.get("_mz_int")
        if ref_mz_int is None or abs(ref_mz_int - channel) > 1:
            continue
        ref_mz_list = spec.get("mz", [])
        ref_int_list = spec.get("intensity", [])
        # Under the lean loader ``mz`` is a float64 numpy array, so guard on
        # length — ``if not <array>`` is ambiguous (ValueError) for arrays.
        if len(ref_mz_list) == 0:
            continue

        ref_peaks = list(zip(ref_mz_list, ref_int_list))
        ref_pmz = spec.get("metadata", {}).get("precursor_mz")
        ref_pmz_val = float(ref_pmz) if ref_pmz is not None else 0.0

        score, n_matched = composite_similarity(
            query_peaks, ref_peaks, config.eic_mz_tolerance,
            precursor_query=float(channel),
            precursor_ref=ref_pmz_val,
            ms1_tolerance=config.ms1_mz_tolerance,
        )

        # W7：Stage 2.5 反推用独立阈值（默认 0.8），比 GUI 显示阈值
        # 更严，因为反推直接落库影响下游所有阶段。
        # ⚠ PR-B(0.7.260624.3)：composite_similarity 综合分已改为 MS-DIAL
        # TotalScore，量程 [0,1]→[0,2]，0.8 在新量程下不再"更严"（偏松）；
        # 按新量程重调阈值留 PR-E（spec §5.3 / 任务 #6）。默认
        # enable_library_mz_inference=False，默认流程不受影响。
        inference_threshold = getattr(
            config, "library_mz_inference_threshold", 0.8,
        )
        if (score >= inference_threshold and
                n_matched >= config.matchms_min_matched_peaks and
                score > best_score):
            best_score = score
            meta = spec.get("metadata", {})
            best_match = {
                "precursor_mz": float(meta.get("precursor_mz", channel)),
                "score": score,
                "n_matched": n_matched,
                "name": meta.get("name", ""),
                "formula": meta.get("formula", ""),
            }

    return best_match


# ---------------------------------------------------------------------------
# Neutral loss consensus method
# ---------------------------------------------------------------------------

def _nl_consensus_predict(feature: CandidateFeature) -> Optional[dict]:
    """Predict precursor m/z from MS2 fragments using neutral loss consensus.

    For each fragment + common NL, computes candidate precursor m/z.
    Clusters candidates within 0.01 Da and selects the cluster supported
    by the most distinct fragment ions (intensity-weighted).
    """
    # The window the instrument isolated is centred on the recorded isolation
    # target (stamped on the feature by stage 1 / 2), not on the integer
    # channel, which is only floor(target): above m/z ~800 the two differ by
    # half a window, and a candidate in the upper half would be lost.
    channel = feature.precursor_mz_nominal
    centre = getattr(feature, "isolation_target_mz", None)
    centre = float(channel) if centre is None else float(centre)
    lo, hi = centre - 0.5, centre + 0.5

    candidates = []
    for mz, intensity in zip(feature.ms2_mz, feature.ms2_intensity):
        for nl_name, nl_mass in NEUTRAL_LOSSES:
            precursor = float(mz) + nl_mass
            if lo <= precursor <= hi:
                candidates.append((precursor, float(mz), float(intensity), nl_name))

    if not candidates:
        return None

    # Cluster by precursor m/z (0.01 Da window)
    candidates.sort(key=lambda c: c[0])
    clusters: list[list] = [[candidates[0]]]
    for c in candidates[1:]:
        if c[0] - clusters[-1][-1][0] < 0.01:
            clusters[-1].append(c)
        else:
            clusters.append([c])

    # Score: count distinct fragment ions supporting each cluster
    best_cluster = None
    best_score = 0
    for cluster in clusters:
        unique_frags = set(round(c[1] * 100) for c in cluster)
        n_unique = len(unique_frags)
        total_intensity = sum(c[2] for c in cluster)
        score = n_unique * 1e12 + total_intensity
        if score > best_score:
            best_score = score
            best_cluster = cluster

    if best_cluster is None:
        return None

    unique_frags = set(round(c[1] * 100) for c in best_cluster)
    n_support = len(unique_frags)

    weights = np.array([c[2] for c in best_cluster])
    mzs = np.array([c[0] for c in best_cluster])
    predicted_mz = float(np.average(mzs, weights=weights))

    if n_support >= 3:
        confidence = "high"
    elif n_support >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    return {"mz": predicted_mz, "confidence": confidence, "n_support": n_support}
