"""Feature-oriented dataclasses shared across pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class AnnotationMatch:
    """One library match candidate for a feature.

    ``score`` is the composite total returned by ``composite_similarity``
    (precursor + MS2 + optional RT). ``wdp`` / ``sdp`` / ``rdp`` are the
    individual MS2 components — surfaced so users can see weighted-dot and
    reverse-dot scores alongside the composite in feature tables and
    exports. ``matched_pct`` (MS-DIAL Matched%, the bounded [0,1] fraction
    of significant reference peaks matched) and ``total_score`` (MS-DIAL
    TotalScore, identical to ``score``) are surfaced alongside them.
    """
    rank: int
    name: str = ""
    formula: str = ""
    score: float = 0.0
    n_matched: int = 0
    ref_peaks: Optional[list] = None
    ref_precursor_mz: Optional[float] = None
    adduct: str = ""
    wdp: float = 0.0
    sdp: float = 0.0
    rdp: float = 0.0
    matched_pct: float = 0.0
    total_score: float = 0.0


@dataclass
class CandidateFeature:
    """Feature assembled from RT-clustered product ion peaks."""
    feature_id: str
    segment_name: str
    replicate_id: int
    precursor_mz_nominal: int
    rt_apex: float
    rt_left: float
    rt_right: float
    ms2_mz: np.ndarray
    ms2_intensity: np.ndarray
    n_fragments: int
    ms2_sn: Optional[np.ndarray] = None
    # Per-fragment chromatographic-shape similarity, parallel to ms2_sn.
    # Populated by stage 1 from DetectedPeak.gaussian_similarity. Aggregated
    # to Feature.gaussian_similarity by aggregate_feature_gaussian().
    ms2_gaussian: Optional[np.ndarray] = None
    ms1_precursor_mz: Optional[float] = None
    ms1_height: Optional[float] = None
    ms1_area: Optional[float] = None
    ms1_sn: Optional[float] = None
    # Per-feature MS1-peak shape similarity (None when no MS1 was assigned).
    ms1_gaussian: Optional[float] = None
    ms1_isotopes: Optional[list] = None
    signal_type: str = "ms1_detected"
    ms2_rep_ion_mz: Optional[float] = None
    mz_source: str = ""
    mz_confidence: str = ""
    inferred_mz: Optional[float] = None
    inferred_formula: Optional[str] = None
    matchms_score: Optional[float] = None
    matchms_name: Optional[str] = None
    source_file: Optional[str] = None
    status: str = "active"
    isotope_group_id: Optional[int] = None
    # Position within the isotope envelope: 0 = monoisotopic representative,
    # n = M+n member. Assigned by Stage 4 (round(delta_mz / C13_DELTA)) so each
    # isotope peak can be surfaced as an independent feature (MS-DIAL style).
    isotope_index: int = 0
    adduct_group_id: Optional[int] = None
    adduct_type: Optional[str] = None
    isf_parent_id: Optional[str] = None
    detection_source: str = "ms2_driven"
    ms2_quality: str = ""          # "correlated" / "sparse" / "none" — MS2 deconvolution quality tag (PR-C)
    n_correlated_ms2: int = 0      # number of chromatographically-correlated MS2 ions kept
    is_duplicate: bool = False
    duplicate_group_id: Optional[int] = None
    duplicate_type: str = ""
    annotation_matches: list = field(default_factory=list)
    selected_annotation_idx: int = 0
    # Inferred charge state from isotope envelope. ``None`` means not
    # estimated yet; ``1`` is the default for singly-charged ions.
    charge_state: Optional[int] = None

    @property
    def precursor_mz(self) -> float:
        if self.ms1_precursor_mz is not None:
            return self.ms1_precursor_mz
        if self.inferred_mz is not None:
            return self.inferred_mz
        return float(self.precursor_mz_nominal)

    def ms2_as_list(self) -> list:
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
    heights: dict = field(default_factory=dict)
    areas: dict = field(default_factory=dict)
    mean_height: float = 0.0
    mean_area: float = 0.0
    cv: float = 0.0
    name: Optional[str] = None
    formula: Optional[str] = None
    adduct: Optional[str] = None
    inchikey: Optional[str] = None
    sn_ratio: float = 0.0
    gaussian_similarity: float = 0.0
    ms1_isotopes: Optional[list] = None
    height_ion_mz: Optional[float] = None
    mz_source: str = ""
    mz_confidence: str = ""
    detection_source: str = "ms2_driven"
    is_duplicate: bool = False
    duplicate_group_id: Optional[int] = None
    duplicate_type: str = ""
    # PR-D: position within the isotope envelope (0 = monoisotopic
    # representative, n = M+n member), the shared isotope cluster id, and the
    # shared adduct cluster id — all plumbed from the representative
    # CandidateFeature by Stage 7 so each isotope/adduct copy can be surfaced as
    # an independent feature (MS-DIAL style).
    isotope_index: int = 0
    isotope_group_id: Optional[int] = None
    adduct_group_id: Optional[int] = None
    annotation_matches: list = field(default_factory=list)
    selected_annotation_idx: int = 0
    # Inferred charge state from isotope envelope.
    charge_state: Optional[int] = None

    @property
    def selected_annotation(self) -> Optional[AnnotationMatch]:
        if self.annotation_matches and 0 <= self.selected_annotation_idx < len(self.annotation_matches):
            return self.annotation_matches[self.selected_annotation_idx]
        return None

    def ms2_as_str(self) -> str:
        pairs = []
        for m, i in zip(self.ms2_mz, self.ms2_intensity):
            pairs.append(f"{m:.5f}:{i:.0f}")
        return " ".join(pairs)
