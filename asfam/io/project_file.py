"""Project file save/load for resuming analysis without reprocessing."""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import Feature, CandidateFeature

logger = logging.getLogger(__name__)

PROJECT_EXTENSION = ".asfam"
PROJECT_VERSION = 1


def save_project(
    path: str,
    config: ProcessingConfig,
    features: list[Feature],
    mzml_paths: list[str],
    library_path: Optional[str] = None,
    stage_stats: Optional[dict] = None,
    candidates_by_rep: Optional[dict] = None,
) -> None:
    """Save project to .asfam file (pickle-based).

    Saves: config, features, file paths, stats, and optionally
    per-replicate candidate features for detailed inspection.
    """
    project = {
        "version": PROJECT_VERSION,
        "config": asdict(config),
        "mzml_paths": mzml_paths,
        "library_path": library_path,
        "stage_stats": stage_stats or {},
        "features": _features_to_dicts(features),
    }
    if candidates_by_rep is not None:
        project["candidates_by_rep"] = {
            rep_id: _candidates_to_dicts(cands)
            for rep_id, cands in candidates_by_rep.items()
        }

    path = str(path)
    if not path.endswith(PROJECT_EXTENSION):
        path += PROJECT_EXTENSION

    with open(path, "wb") as f:
        pickle.dump(project, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Project saved to %s (%d features)", path, len(features))


def load_project(path: str) -> dict:
    """Load project from .asfam file.

    Returns dict with keys:
    - config: ProcessingConfig
    - features: list[Feature]
    - mzml_paths: list[str]
    - library_path: Optional[str]
    - stage_stats: dict
    - candidates_by_rep: Optional[dict] (if saved)
    """
    with open(path, "rb") as f:
        project = pickle.load(f)

    if project.get("version", 0) != PROJECT_VERSION:
        logger.warning("Project version mismatch: expected %d, got %d",
                       PROJECT_VERSION, project.get("version", 0))

    config = ProcessingConfig(**{
        k: v for k, v in project["config"].items()
        if k in ProcessingConfig.__dataclass_fields__
    })
    features = _dicts_to_features(project["features"])

    result = {
        "config": config,
        "features": features,
        "mzml_paths": project.get("mzml_paths", []),
        "library_path": project.get("library_path"),
        "stage_stats": project.get("stage_stats", {}),
    }

    if "candidates_by_rep" in project:
        result["candidates_by_rep"] = {
            rep_id: _dicts_to_candidates(cands)
            for rep_id, cands in project["candidates_by_rep"].items()
        }

    logger.info("Project loaded from %s (%d features)", path, len(features))
    return result


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _features_to_dicts(features: list[Feature]) -> list[dict]:
    """Convert Feature objects to serializable dicts."""
    result = []
    for f in features:
        d = {
            "feature_id": f.feature_id,
            "precursor_mz": f.precursor_mz,
            "rt": f.rt,
            "rt_left": f.rt_left,
            "rt_right": f.rt_right,
            "signal_type": f.signal_type,
            "ms2_mz": f.ms2_mz.tolist(),
            "ms2_intensity": f.ms2_intensity.tolist(),
            "n_fragments": f.n_fragments,
            "heights": f.heights,
            "areas": f.areas,
            "mean_height": f.mean_height,
            "mean_area": f.mean_area,
            "cv": f.cv,
            "name": f.name,
            "formula": f.formula,
            "adduct": f.adduct,
            "inchikey": f.inchikey,
            "sn_ratio": f.sn_ratio,
            "gaussian_similarity": f.gaussian_similarity,
            "ms1_isotopes": f.ms1_isotopes,
            "height_ion_mz": f.height_ion_mz,
            "mz_source": f.mz_source,
            "mz_confidence": f.mz_confidence,
            "detection_source": f.detection_source,
            "is_duplicate": f.is_duplicate,
            "duplicate_group_id": f.duplicate_group_id,
            "duplicate_type": f.duplicate_type,
            # Annotation matches (top N)
            "annotation_matches": [
                {"rank": m.rank, "name": m.name, "formula": m.formula,
                 "score": m.score, "n_matched": m.n_matched,
                 "ref_peaks": m.ref_peaks, "ref_precursor_mz": m.ref_precursor_mz,
                 "adduct": m.adduct}
                for m in f.annotation_matches
            ],
            "selected_annotation_idx": f.selected_annotation_idx,
        }
        result.append(d)
    return result


def _dicts_to_features(dicts: list[dict]) -> list[Feature]:
    """Convert dicts back to Feature objects."""
    features = []
    for d in dicts:
        f = Feature(
            feature_id=d["feature_id"],
            precursor_mz=d["precursor_mz"],
            rt=d["rt"],
            rt_left=d["rt_left"],
            rt_right=d["rt_right"],
            signal_type=d["signal_type"],
            ms2_mz=np.array(d["ms2_mz"], dtype=np.float64),
            ms2_intensity=np.array(d["ms2_intensity"], dtype=np.float64),
            n_fragments=d["n_fragments"],
            heights=d.get("heights", {}),
            areas=d.get("areas", {}),
            mean_height=d.get("mean_height", 0.0),
            mean_area=d.get("mean_area", 0.0),
            cv=d.get("cv", 0.0),
            name=d.get("name"),
            formula=d.get("formula"),
            adduct=d.get("adduct"),
            inchikey=d.get("inchikey"),
            sn_ratio=d.get("sn_ratio", 0.0),
            gaussian_similarity=d.get("gaussian_similarity", 0.0),
            ms1_isotopes=d.get("ms1_isotopes"),
            height_ion_mz=d.get("height_ion_mz"),
            mz_source=d.get("mz_source", ""),
            mz_confidence=d.get("mz_confidence", ""),
            detection_source=d.get("detection_source", "ms2_driven"),
            is_duplicate=d.get("is_duplicate", False),
            duplicate_group_id=d.get("duplicate_group_id"),
            duplicate_type=d.get("duplicate_type", ""),
        )
        # Backward compatibility: map old signal_type values
        _st = f.signal_type
        if _st == "high_response":
            f.signal_type = "ms1_detected"
        elif _st == "low_response":
            f.signal_type = "ms2_only"
        # Restore annotation matches
        from asfam.models import AnnotationMatch
        am_data = d.get("annotation_matches", [])
        if am_data:
            f.annotation_matches = [AnnotationMatch(**m) for m in am_data]
            f.selected_annotation_idx = d.get("selected_annotation_idx", 0)
        else:
            # Backward compatibility: convert old _ref_spectrum format
            ref_spec = d.get("_ref_spectrum")
            if ref_spec:
                f.annotation_matches = [AnnotationMatch(
                    rank=1, name=d.get("_ref_name", f.name or ""),
                    formula=f.formula or "",
                    score=d.get("_ref_score", 0.0),
                    n_matched=0, ref_peaks=ref_spec,
                )]
        features.append(f)
    return features


def _candidates_to_dicts(candidates: list[CandidateFeature]) -> list[dict]:
    """Convert CandidateFeature objects to dicts."""
    result = []
    for c in candidates:
        d = {
            "feature_id": c.feature_id,
            "segment_name": c.segment_name,
            "replicate_id": c.replicate_id,
            "precursor_mz_nominal": c.precursor_mz_nominal,
            "rt_apex": c.rt_apex,
            "rt_left": c.rt_left,
            "rt_right": c.rt_right,
            "ms2_mz": c.ms2_mz.tolist(),
            "ms2_intensity": c.ms2_intensity.tolist(),
            "n_fragments": c.n_fragments,
            "ms2_sn": c.ms2_sn.tolist() if c.ms2_sn is not None else None,
            "ms2_rep_ion_mz": c.ms2_rep_ion_mz,
            "ms1_precursor_mz": c.ms1_precursor_mz,
            "ms1_height": c.ms1_height,
            "ms1_area": c.ms1_area,
            "ms1_sn": c.ms1_sn,
            "ms1_isotopes": c.ms1_isotopes,
            "signal_type": c.signal_type,
            "inferred_mz": c.inferred_mz,
            "inferred_formula": c.inferred_formula,
            "matchms_score": c.matchms_score,
            "matchms_name": c.matchms_name,
            "source_file": c.source_file,
            "status": c.status,
            "isotope_group_id": c.isotope_group_id,
            "adduct_group_id": c.adduct_group_id,
            "adduct_type": c.adduct_type,
            "isf_parent_id": c.isf_parent_id,
            "detection_source": c.detection_source,
            "mz_source": c.mz_source,
            "mz_confidence": c.mz_confidence,
            "is_duplicate": c.is_duplicate,
            "duplicate_group_id": c.duplicate_group_id,
            "duplicate_type": c.duplicate_type,
            # Annotation matches (top N)
            "annotation_matches": [
                {"rank": m.rank, "name": m.name, "formula": m.formula,
                 "score": m.score, "n_matched": m.n_matched,
                 "ref_peaks": m.ref_peaks, "ref_precursor_mz": m.ref_precursor_mz,
                 "adduct": m.adduct}
                for m in c.annotation_matches
            ],
            "selected_annotation_idx": c.selected_annotation_idx,
        }
        result.append(d)
    return result


def _dicts_to_candidates(dicts: list[dict]) -> list[CandidateFeature]:
    """Convert dicts back to CandidateFeature objects."""
    candidates = []
    for d in dicts:
        c = CandidateFeature(
            feature_id=d["feature_id"],
            segment_name=d["segment_name"],
            replicate_id=d["replicate_id"],
            precursor_mz_nominal=d["precursor_mz_nominal"],
            rt_apex=d["rt_apex"],
            rt_left=d["rt_left"],
            rt_right=d["rt_right"],
            ms2_mz=np.array(d["ms2_mz"], dtype=np.float64),
            ms2_intensity=np.array(d["ms2_intensity"], dtype=np.float64),
            n_fragments=d["n_fragments"],
            ms2_sn=np.array(d["ms2_sn"], dtype=np.float64) if d.get("ms2_sn") is not None else None,
            ms2_rep_ion_mz=d.get("ms2_rep_ion_mz"),
            ms1_precursor_mz=d.get("ms1_precursor_mz"),
            ms1_height=d.get("ms1_height"),
            ms1_area=d.get("ms1_area"),
            ms1_sn=d.get("ms1_sn"),
            ms1_isotopes=d.get("ms1_isotopes"),
            signal_type=d.get("signal_type", "ms1_detected"),
            inferred_mz=d.get("inferred_mz"),
            inferred_formula=d.get("inferred_formula"),
            matchms_score=d.get("matchms_score"),
            matchms_name=d.get("matchms_name"),
            source_file=d.get("source_file"),
            status=d.get("status", "active"),
            isotope_group_id=d.get("isotope_group_id"),
            adduct_group_id=d.get("adduct_group_id"),
            adduct_type=d.get("adduct_type"),
            isf_parent_id=d.get("isf_parent_id"),
            detection_source=d.get("detection_source", "ms2_driven"),
            mz_source=d.get("mz_source", ""),
            mz_confidence=d.get("mz_confidence", ""),
            is_duplicate=d.get("is_duplicate", False),
            duplicate_group_id=d.get("duplicate_group_id"),
            duplicate_type=d.get("duplicate_type", ""),
        )
        # Backward compatibility: map old signal_type values
        if c.signal_type == "high_response":
            c.signal_type = "ms1_detected"
        elif c.signal_type == "low_response":
            c.signal_type = "ms2_only"
        # Restore annotation matches
        from asfam.models import AnnotationMatch
        am_data = d.get("annotation_matches", [])
        if am_data:
            c.annotation_matches = [AnnotationMatch(**m) for m in am_data]
            c.selected_annotation_idx = d.get("selected_annotation_idx", 0)
        else:
            ref_spec = d.get("_ref_spectrum")
            if ref_spec:
                c.annotation_matches = [AnnotationMatch(
                    rank=1, name=d.get("_ref_name", c.matchms_name or ""),
                    formula=c.inferred_formula or "",
                    score=c.matchms_score or 0.0,
                    n_matched=0, ref_peaks=ref_spec,
                )]
        candidates.append(c)
    return candidates
