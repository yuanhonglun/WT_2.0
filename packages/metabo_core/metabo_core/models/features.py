"""Feature-oriented dataclasses shared across pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

#: The two chromatograms a feature's height can be read off. A quantitation
#: value only means something beside another value read off the *same* kind of
#: trace, so these are what the joiner keeps apart and the gap filler
#: integrates on. ``gap_filler`` re-exports both.
MS1 = "ms1"
PRODUCT = "product"


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
    # The MS1 ion ``ms1_height`` / ``ms1_area`` were actually measured on, which
    # is NOT always ``ms1_precursor_mz``. An MS2-driven feature gets its
    # precursor m/z from the intensity-weighted centroid of the whole 1-Da
    # isolation window, and that centroid sits > 0.01 Da away from the ion the
    # height came from in 36% of peaks — so a tolerance-width chromatogram
    # around it can be empty. Gap filling integrates THIS ion; the reported
    # precursor m/z stays ``ms1_precursor_mz``. ``None`` on ms2_only features,
    # which are quantified on ``ms2_rep_ion_mz`` instead.
    ms1_quant_mz: Optional[float] = None
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
    # Gating flag for the *dedup* stages only: a feature moves from "active" to
    # "isotope_excluded" / "adduct_excluded" / "isf_excluded" once an earlier
    # dedup stage has claimed it, so later dedup stages skip it as a candidate.
    # It does NOT mean deleted — every feature, excluded or not, goes on to
    # annotation, alignment and export. Visibility is driven by ``is_duplicate``.
    # ("discarded_few_fragments" is the sole status that really removes a
    # feature: stage 2.5 drops those rows from the list outright.)
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
    # "detected" for every peak the pipeline actually picked; alignment's gap
    # filler mints extra peaks carrying "filled" / "no_signal". Only "detected"
    # peaks may represent a spot or enter its mean / CV (MS-DIAL PeakID >= 0).
    gap_fill_status: str = "detected"

    @property
    def precursor_mz(self) -> float:
        """The precursor m/z to *report*. Not the ion to align or integrate on.

        On an ASFAM MS2-driven feature this is the intensity-weighted centroid of
        a whole 1-Da isolation window, which is not any one real ion — see
        :attr:`align_mz`.
        """
        if self.ms1_precursor_mz is not None:
            return self.ms1_precursor_mz
        if self.inferred_mz is not None:
            return self.inferred_mz
        return float(self.precursor_mz_nominal)

    @property
    def quant_route(self) -> str:
        """:data:`MS1` or :data:`PRODUCT` — which chromatogram the height came off.

        The two are not interchangeable. An MS1 height is read off a +/-0.01 Da
        SUM window on ``ms1_quant_mz``; a product-ion height off a +/-0.1 Da SUM
        slice on ``ms2_rep_ion_mz``. Peaks of different routes must never share
        an alignment spot: their heights would sit in one row measured on two
        different things, and the row's CV and fold change would be meaningless.

        This is the same predicate ``gap_filler.quant_ion`` branches on, and it
        has to stay that way: the route the joiner groups by and the trace the
        gap filler integrates are one decision, not two that happen to agree.
        """
        return PRODUCT if self.signal_type == "ms2_only" else MS1

    @property
    def align_mz(self) -> Optional[float]:
        """The ion this feature was quantified on — the ion gap fill integrates.

        **Not** :attr:`precursor_mz`. ASFAM is all-ion fragmentation: nothing
        selected a precursor, so an MS2-driven feature's ``ms1_precursor_mz`` is
        the intensity-weighted centroid of the entire 1-Da isolation window, and
        it drifts with whatever else happened to co-isolate in *this* sample.
        Two samples' peaks for one compound then land > mz_tolerance apart and
        the joiner splits them across two rows — the tolerance never had a
        chance, because the key does not mean the same thing twice. The
        quantitation ion does: it is a real ion, the one the height was read off,
        and the one a missing sample gets integrated on. Align on that.

        (Gap fill learned this already — integrating ``ms1_precursor_mz`` filled
        28.4% of cells with zero. The joiner never did.)

        ``None`` when the route has no ion at all: such a feature cannot be
        aligned onto a quantitation ion *or* gap-filled, so the joiner counts it
        rather than keying it on something it was not measured on. Measured
        incidence on both ASFAM datasets: zero.

        The MS1 route falls back to ``ms1_precursor_mz`` when ``ms1_quant_mz`` is
        unset, which is not a fudge: DDA never sets ``ms1_quant_mz`` because in
        DDA the precursor *is* a picked MS1 ion — ``ms1_precursor_mz`` is its ROI
        centroid and ``ms1_height`` is the height of that same ROI. The fiction
        is specific to all-ion fragmentation.
        """
        if self.signal_type == "ms2_only":
            return self.ms2_rep_ion_mz
        if self.ms1_quant_mz is not None:
            return self.ms1_quant_mz
        return self.ms1_precursor_mz

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
    #: ``sample_id -> "detected" | "filled" | "no_signal"``. Every sample in the
    #: run appears once gap filling has run, so ``heights`` has no holes.
    #: ``mean_height`` / ``mean_area`` / ``cv`` / ``sn_ratio`` are computed over
    #: the "detected" samples only.
    gap_fill_status: dict = field(default_factory=dict)
    #: Samples whose peak was actually picked, and that count over *every*
    #: sample in the run. Gap-filled cells carry a height but are not detections,
    #: so ``heights[sid] > 0`` is not the same test — post-gap-fill it is true
    #: almost everywhere. Set from ``AlignmentSpot.n_detected``.
    n_detected: int = 0
    detection_rate: float = 0.0
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
    #: The ion this spot was quantified on, copied off the representative
    #: (:attr:`CandidateFeature.align_mz`). **Not** :attr:`precursor_mz`, which is
    #: only what the row *reports*: on an ASFAM ms2_only row that is the
    #: intensity-weighted centroid of a whole 1-Da isolation window, taken from a
    #: cycle where the compound's MS1 was too weak for any peak picker to find —
    #: a noise value that reproduces neither across samples nor across routes.
    #: The refiner keys redundancy on this. ``None`` only when the route had no
    #: quantitation ion at all (measured incidence on both ASFAM datasets: zero).
    #:
    #: Also **not** :attr:`height_ion_mz`, which is the representative's
    #: ``ms2_rep_ion_mz`` — a fragment, and set on MS1-route rows too.
    align_mz: Optional[float] = None
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
    #: Consensus alignment centre, distinct from representative display fields.
    #: ``rt`` is the consensus RT after ASFAM Stage 7 reconciliation;
    #: ``representative_rt`` records the spectrum/annotation donor's apex.
    representative_rt: Optional[float] = None
    #: Stable ASFAM acquisition identity retained through refinement.  These are
    #: optional so DDA/GC-MS objects and old project files remain valid.
    alignment_window: Optional[int] = None
    alignment_segment: str = ""
    #: Cross-route relation metadata.  A partial MS1 coverage relation leaves
    #: PRODUCT visible and never moves its intensities.
    alignment_relation: str = ""
    alignment_related_feature_id: str = ""

    @property
    def quant_route(self) -> str:
        """:data:`MS1` or :data:`PRODUCT` — which chromatogram the height came off.

        The same predicate :attr:`CandidateFeature.quant_route` branches on, and
        it has to stay that way: the route the joiner groups by, the trace the gap
        filler integrates, and the routes the refiner refuses to compare an m/z
        across are one decision, not three that happen to agree.
        """
        return PRODUCT if self.signal_type == "ms2_only" else MS1

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
