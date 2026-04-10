"""Processing configuration with all tunable parameters."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ProcessingConfig:
    """All processing parameters with sensible defaults."""

    # -- General --
    ionization_mode: str = "positive"  # "positive" or "negative"
    n_workers: int = 4                 # multiprocessing pool size

    # -- Stage 1: MS2 peak detection --
    eic_mz_tolerance: float = 0.02     # Da, product ion EIC extraction
    eic_smoothing_method: str = "savgol"
    eic_smoothing_window: int = 7      # Savitzky-Golay window length
    eic_smoothing_polyorder: int = 3
    peak_height_threshold: float = 200.0
    peak_sn_threshold: float = 5.0
    peak_width_min: int = 3            # minimum peak width in scans
    peak_prominence: float = 100.0
    peak_gaussian_threshold: float = 0.6  # minimum gaussian similarity (0 = off)
    rt_cluster_tolerance: float = 0.02  # min, for grouping co-eluting ions
    min_fragments_per_feature: int = 2

    # -- Stage 1: MS2 ion recall (second pass) --
    recall_enabled: bool = True
    recall_min_intensity: float = 50.0   # minimum raw intensity at apex for recalled ion
    recall_min_consecutive: int = 2      # minimum consecutive nonzero cycles around apex
    recall_apex_window: int = 2          # +/- cycles around consensus apex to search

    # -- Stage 2: MS1 assignment --
    ms1_mz_tolerance: float = 0.01     # Da
    ms1_rt_tolerance: float = 0.05      # min (3s, tighter to prevent cross-assignment)
    ms1_min_height: float = 100.0
    ms1_isotope_mz_tol: float = 0.01   # Da, for isotope pattern extraction
    ms1_shape_weight: float = 0.3      # weight for peak shape in MS1 assignment scoring

    # -- Stage 2.5: MS2-only m/z inference --
    min_fragments_for_inference: int = 3
    matchms_similarity_threshold: float = 0.8   # total score cutoff
    matchms_min_matched_peaks: int = 3          # MS-DIAL default: 3
    matchms_min_matched_pct: float = 0.25       # MS-DIAL: 25% of ref peaks
    matchms_use_rt: bool = False                # use RT in scoring (user toggle)

    # -- Stage 3: Segment merge --
    merge_rt_tolerance: float = 0.05   # min
    merge_mz_tolerance: float = 0.02   # Da
    merge_ms2_cosine_threshold: float = 0.8

    # -- Stage 4: Isotope deduplication --
    isotope_rt_tolerance: float = 0.1           # min (search window; overlap ratio is primary criterion)
    isotope_apex_rt_strict: float = 0.05        # min, hard max for apex RT difference between isotope pairs
    isotope_overlap_ratio: float = 0.80
    isotope_mz_tolerance: float = 0.01          # Da, classic gaps
    isotope_integer_step_tolerance: float = 0.02  # Da, relaxed gaps
    isotope_fragment_mz_tolerance: float = 0.02
    isotope_precursor_exclusion: float = 1.5    # Da
    isotope_modified_cos_threshold: float = 0.85
    isotope_modified_cos_relaxed: float = 0.90
    isotope_min_matches: int = 3
    isotope_min_matches_relaxed: int = 4
    isotope_nl_cos_threshold: float = 0.85
    isotope_min_nl_matches: int = 3
    isotope_max_step: int = 4

    # -- Stage 5: Adduct deduplication --
    adduct_rt_tolerance: float = 0.05  # min
    adduct_mw_tolerance: float = 0.02  # Da
    adduct_eic_pearson_threshold: float = 0.9

    # -- Stage 5b: Duplicate detection --
    duplicate_rt_tolerance: float = 0.2    # min
    duplicate_mz_tolerance: float = 0.5    # Da
    duplicate_cosine_threshold: float = 0.85
    duplicate_min_matched: int = 3

    # -- Stage 6: ISF detection --
    isf_eic_pearson_threshold: float = 0.9
    isf_min_correlated_scans: int = 10
    isf_ms2_mz_tolerance: float = 0.02  # Da

    # -- Stage 7: Cross-replicate alignment --
    alignment_rt_tolerance: float = 0.1  # min
    alignment_mz_tolerance: float = 0.02  # Da
    alignment_mz_weight: float = 0.5
    alignment_rt_weight: float = 0.5
    gap_fill_enabled: bool = True
    gap_fill_rt_expansion: float = 1.5

    # -- Stage 8: Export --
    export_mgf: bool = True
    export_msp: bool = True
    export_report: bool = True
    export_include_duplicates: bool = False

    # -- Quality filtering --
    final_height_threshold: float = 1000.0
    final_sn_threshold: float = 5.0
    final_gaussian_threshold: float = 0.6
    msms_intensity_threshold: float = 1000.0
    msms_relative_threshold: float = 0.01
    msms_min_ions: int = 1

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save config to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "ProcessingConfig":
        """Load config from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
