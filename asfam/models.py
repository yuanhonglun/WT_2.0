"""Data models for the ASFAM processing pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Raw data representations
# ---------------------------------------------------------------------------

@dataclass
class ScanCycle:
    """One scan cycle: 1 MS1 scan + N MRM-HR scans."""
    cycle_index: int
    rt: float                          # retention time (min), from MS1
    ms1_mz: np.ndarray                 # MS1 m/z array (profile)
    ms1_intensity: np.ndarray          # MS1 intensity array
    ms2_scans: dict                    # int(precursor_mz) -> (product_mz_arr, product_int_arr)


@dataclass
class RawSegmentData:
    """All data from one mzML file = one segment + one replicate."""
    file_path: str
    segment_name: str                  # e.g., "100-129"
    segment_low: int                   # e.g., 100
    segment_high: int                  # e.g., 129
    replicate_id: int
    n_cycles: int
    rt_array: np.ndarray               # shape (n_cycles,)
    precursor_list: list               # [100, 101, ..., 129]
    cycles: list                       # list[ScanCycle]
    collision_energy: float = 0.0


# ---------------------------------------------------------------------------
# EIC and peak detection intermediates
# ---------------------------------------------------------------------------

@dataclass
class ProductIonEIC:
    """EIC for one product ion in one MRM-HR channel."""
    precursor_mz_nominal: int          # integer precursor channel
    product_mz: float                  # high-resolution product ion m/z
    rt_array: np.ndarray
    intensity_array: np.ndarray
    smoothed_intensity: Optional[np.ndarray] = None


@dataclass
class DetectedPeak:
    """A chromatographic peak detected in an EIC."""
    precursor_mz_nominal: int
    product_mz: float
    rt_apex: float
    rt_left: float
    rt_right: float
    apex_index: int
    left_index: int
    right_index: int
    height: float
    area: float
    sn_ratio: float = 0.0
    gaussian_similarity: float = 0.0


# ---------------------------------------------------------------------------
# Annotation match result
# ---------------------------------------------------------------------------

@dataclass
class AnnotationMatch:
    """One library match candidate for a feature."""
    rank: int                          # 1-based rank (1 = best)
    name: str = ""
    formula: str = ""
    score: float = 0.0
    n_matched: int = 0
    ref_peaks: Optional[list] = None   # list of (mz, intensity) tuples
    ref_precursor_mz: Optional[float] = None
    adduct: str = ""


# ---------------------------------------------------------------------------
# Candidate and final features
# ---------------------------------------------------------------------------

@dataclass
class CandidateFeature:
    """Feature assembled from RT-clustered product ion peaks."""
    feature_id: str                    # unique ID
    segment_name: str
    replicate_id: int
    precursor_mz_nominal: int          # integer precursor channel
    rt_apex: float
    rt_left: float
    rt_right: float
    # MS2 spectrum assembled from product ions
    ms2_mz: np.ndarray
    ms2_intensity: np.ndarray
    n_fragments: int
    ms2_sn: Optional[np.ndarray] = None    # SN for each product ion (parallel to ms2_mz)
    # MS1 assignment (filled in Stage 2)
    ms1_precursor_mz: Optional[float] = None
    ms1_height: Optional[float] = None
    ms1_area: Optional[float] = None
    ms1_sn: Optional[float] = None
    ms1_isotopes: Optional[list] = None  # [(mz, intensity), ...]
    signal_type: str = "ms1_detected"   # or "ms2_only"
    ms2_rep_ion_mz: Optional[float] = None  # representative product ion m/z (for ms2_only)
    # m/z source tracking: "ms1_peak", "ms1_relaxed", "library", "nl_consensus", ""
    mz_source: str = ""
    mz_confidence: str = ""  # for nl_consensus: "high", "medium", "low", ""
    # Stage 2.5 inference
    inferred_mz: Optional[float] = None
    inferred_formula: Optional[str] = None
    matchms_score: Optional[float] = None
    matchms_name: Optional[str] = None
    # Source tracking
    source_file: Optional[str] = None
    # Deduplication annotations
    status: str = "active"  # active/isotope_removed/adduct_removed/isf_removed
    isotope_group_id: Optional[int] = None
    adduct_group_id: Optional[int] = None
    adduct_type: Optional[str] = None
    isf_parent_id: Optional[str] = None
    detection_source: str = "ms2_driven"
    # Duplicate detection (Stage 5b)
    is_duplicate: bool = False
    duplicate_group_id: Optional[int] = None
    duplicate_type: str = ""  # "isotope", "adduct", "isf", "spectral"
    # Library annotation (top N matches)
    annotation_matches: list = field(default_factory=list)  # list[AnnotationMatch]
    selected_annotation_idx: int = 0

    @property
    def precursor_mz(self) -> float:
        """Best available precise m/z."""
        if self.ms1_precursor_mz is not None:
            return self.ms1_precursor_mz
        if self.inferred_mz is not None:
            return self.inferred_mz
        return float(self.precursor_mz_nominal)

    def ms2_as_list(self) -> list:
        """Return MS2 as list of (mz, intensity) tuples."""
        return list(zip(self.ms2_mz.tolist(), self.ms2_intensity.tolist()))


@dataclass
class Feature:
    """Final feature after alignment across replicates."""
    feature_id: str
    precursor_mz: float
    rt: float
    rt_left: float
    rt_right: float
    signal_type: str
    ms2_mz: np.ndarray
    ms2_intensity: np.ndarray
    n_fragments: int
    # Cross-replicate quantification
    heights: dict = field(default_factory=dict)   # rep_id -> height
    areas: dict = field(default_factory=dict)      # rep_id -> area
    mean_height: float = 0.0
    mean_area: float = 0.0
    cv: float = 0.0
    # Identification
    name: Optional[str] = None
    formula: Optional[str] = None
    adduct: Optional[str] = None
    inchikey: Optional[str] = None
    # Quality
    sn_ratio: float = 0.0
    gaussian_similarity: float = 0.0
    ms1_isotopes: Optional[list] = None
    height_ion_mz: Optional[float] = None  # if height from product ion, which one
    mz_source: str = ""       # "ms1_peak", "ms1_relaxed", "library", "nl_consensus", ""
    mz_confidence: str = ""   # for nl_consensus: "high", "medium", "low"
    detection_source: str = "ms2_driven"
    # Duplicate detection (Stage 5b)
    is_duplicate: bool = False
    duplicate_group_id: Optional[int] = None
    duplicate_type: str = ""  # "isotope", "adduct", "isf", "spectral"
    # Library annotation (top N matches)
    annotation_matches: list = field(default_factory=list)  # list[AnnotationMatch]
    selected_annotation_idx: int = 0

    @property
    def selected_annotation(self) -> Optional[AnnotationMatch]:
        if self.annotation_matches and 0 <= self.selected_annotation_idx < len(self.annotation_matches):
            return self.annotation_matches[self.selected_annotation_idx]
        return None

    def ms2_as_str(self) -> str:
        """Format MS2 as 'mz1:int1 mz2:int2 ...' string."""
        pairs = []
        for m, i in zip(self.ms2_mz, self.ms2_intensity):
            pairs.append(f"{m:.5f}:{i:.0f}")
        return " ".join(pairs)
