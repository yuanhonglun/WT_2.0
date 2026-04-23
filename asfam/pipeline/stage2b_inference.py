"""Stage 2.5: MS2-only m/z inference using library matching + neutral loss consensus."""
from __future__ import annotations

import logging
from typing import Optional, Callable

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import CandidateFeature
from asfam.core.similarity import composite_similarity

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


def run_stage2b(
    features_by_replicate: dict[str, list[CandidateFeature]],
    config: ProcessingConfig,
    spectral_library_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    preloaded_library: Optional[list] = None,
) -> dict[str, list[CandidateFeature]]:
    """Infer precise m/z for ms2_only signals.

    Step 1: Library matching (MS1-restricted, composite similarity)
    Step 2: Neutral loss consensus method
    Discard features with too few fragments.
    """
    logger.info("Stage 2.5: MS2-only m/z inference...")

    library = preloaded_library
    if (library is None and spectral_library_path
            and getattr(config, "enable_library_mz_inference", False)):
        library = _load_library(spectral_library_path)
    elif not getattr(config, "enable_library_mz_inference", False):
        # Library-matching inference disabled by config — skip library step entirely.
        library = None

    for rep_id, features in features_by_replicate.items():
        ms2_only = [f for f in features if f.signal_type == "ms2_only"
                     and not f.mz_source]
        if not ms2_only:
            continue

        logger.info("  Replicate %s: %d ms2_only signals to process", rep_id, len(ms2_only))

        n_library = 0
        n_nl = 0
        n_discarded = 0

        for i, feat in enumerate(ms2_only):
            if progress_callback and i % 50 == 0:
                progress_callback("stage2b", i, len(ms2_only), f"Rep {rep_id}: inferring m/z")

            if feat.n_fragments < config.min_fragments_for_inference:
                feat.status = "discarded_few_fragments"
                n_discarded += 1
                continue

            # Step 1: Library matching
            if library is not None:
                match = _library_match(feat, library, config)
                if match is not None:
                    feat.inferred_mz = match["precursor_mz"]
                    feat.matchms_score = match["score"]
                    feat.matchms_name = match.get("name", "")
                    feat.inferred_formula = match.get("formula", "")
                    feat.mz_source = "library"
                    n_library += 1
                    continue

            # Step 2: Neutral loss consensus
            nl_result = _nl_consensus_predict(feat)
            if nl_result is not None:
                feat.inferred_mz = nl_result["mz"]
                feat.mz_source = "nl_consensus"
                feat.mz_confidence = nl_result["confidence"]
                n_nl += 1
                continue

            n_discarded += 1

        features_by_replicate[rep_id] = [f for f in features if f.status != "discarded_few_fragments"]

        logger.info("  Replicate %s: %d library, %d NL consensus, %d no inference",
                     rep_id, n_library, n_nl, n_discarded)

    return features_by_replicate


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def _load_library(library_path: str) -> Optional[list[dict]]:
    """Load spectral library from MSP or MGF."""
    try:
        from asfam.io.spectral_library import read_mgf, read_msp
        if library_path.lower().endswith(".mgf"):
            spectra = read_mgf(library_path)
        elif library_path.lower().endswith(".msp"):
            spectra = read_msp(library_path)
        else:
            logger.warning("Unsupported library format: %s", library_path)
            return None

        for spec in spectra:
            pmz = spec.get("metadata", {}).get("precursor_mz")
            if pmz is not None:
                spec["_mz_int"] = int(round(float(pmz)))
            else:
                spec["_mz_int"] = None

        logger.info("Loaded %d spectra from library %s", len(spectra), library_path)
        return spectra
    except Exception as e:
        logger.warning("Failed to load library %s: %s", library_path, e)
        return None


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
        if not ref_mz_list:
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

        if (score >= config.matchms_similarity_threshold and
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
    channel = feature.precursor_mz_nominal
    lo, hi = channel - 0.5, channel + 0.5

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
