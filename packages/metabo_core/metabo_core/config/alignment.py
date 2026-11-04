"""Cross-replicate alignment configuration shared by core and apps."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AlignmentConfig:
    """Parameters for cross-replicate alignment."""
    rt_tolerance: float = 0.1
    mz_tolerance: float = 0.02
    mz_weight: float = 0.5
    rt_weight: float = 0.5
    ms2_mz_tolerance: float = 0.02
    match_threshold: float = 0.5


@dataclass
class JoinerConfig:
    """Parameters for the MS-DIAL-style peak joiner.

    Deliberately *not* :class:`AlignmentConfig`: the two weight the score
    differently. ``AlignmentConfig.mz_weight`` / ``rt_weight`` split a Gaussian
    sub-score that is then blended 0.6/0.4 with an MS2 cosine, whereas the three
    weights here are peers summing to 1 over the whole score::

        score = mz_weight * exp(-0.5 * (dmz / mz_tolerance) ** 2)
              + rt_weight * exp(-0.5 * (drt / rt_tolerance) ** 2)
              + ms2_weight * cosine(master_ms2, target_ms2)

    ``mz_weight=0.3, rt_weight=0.3, ms2_weight=0.4`` reproduces the old blend
    exactly. ``ms2_weight = 0`` is strict MS-DIAL (``LcmsPeakJoiner.cs:55-58``,
    which has no MS2 term at all).

    When either side of a pair carries no MS2, the cosine is undefined and the
    score falls back to m/z + RT alone, with the two weights renormalized to sum
    to 1 — so ``match_threshold`` keeps the same meaning on both paths.
    """
    #: ``RetentionTimeAlignmentTolerance``. 0.2 rather than MS-DIAL's 0.1: the
    #: real cross-sample RT drift on the ASFAM benchmarks has a p90 of
    #: 0.13-0.19 min, so +/-0.1 covered only 85-89% of the peak pairs that are
    #: the same compound. It is jitter, not a warp — the median signed drift per
    #: sample is under a second — so widening the window is the fix and RT
    #: correction is not (correcting it moves coverage by 0.0 points).
    rt_tolerance: float = 0.2
    mz_tolerance: float = 0.02
    mz_weight: float = 0.3
    rt_weight: float = 0.3
    ms2_weight: float = 0.4
    ms2_mz_tolerance: float = 0.02
    #: Reject a claim scoring at or below this. **MS-DIAL has no such gate** —
    #: its claim is ``if (matchIdx.HasValue)``, and the box is the gate. Ours
    #: was 0.5, which was strictly harsher than the box the build step had
    #: already used to suppress the same peak from founding a master of its own,
    #: so a peak could be silently double-killed: no spot, and no master either.
    #: Worse, the score renormalizes to m/z + RT when either side lacks MS2, so
    #: *having* MS2 on both sides made a pair harder to match than having none —
    #: and co-eluting ASFAM compounds in one isolation window are exactly where
    #: the cosine is low. 0 restores MS-DIAL's behaviour.
    #:
    #: DDA passes its own ``AlignmentConfig.match_threshold`` through
    #: ``DDAConfig.joiner_view()`` and keeps the gate; there the MS2 cosine is a
    #: real precursor-selected spectrum and the threshold means what it says.
    match_threshold: float = 0.0
    #: The MS2 identity gate on the box (see :func:`~metabo_core.alignment.joiner._matches`).
    #: Two peaks share a box only if their spectra cannot tell them apart —
    #: cosine at or above this. Without it the box is a *proximity* test, and a
    #: proximity test used to *suppress* a peak from founding a master deletes it
    #: outright: the build step suppressed every peak that fell inside any
    #: master's box, while the claim step lets one sample put only one peak in a
    #: master, so k peaks in one box left k-1 with no master of their own and no
    #: master to claim. On the ASFAM benchmarks that silently deleted 1,242
    #: (rice) / 2,026 (cancer) peaks whose MS2 says they are a *different
    #: compound* from the peak that beat them.
    #:
    #: ``0`` disables the gate and restores the pure proximity box.
    ms2_identity_threshold: float = 0.5
    #: Below this many fragments a cosine is not evidence of anything, so the
    #: gate abstains and the pair stays in one box. Abstaining has to mean
    #: "suppress", not "found a new master": the opposite would give every
    #: fragment-poor peak a row of its own. Read off ``n_fragments``, which costs
    #: no spectrum read — the gate touches the disk only for pairs it can judge.
    ms2_identity_min_fragments: int = 3
    # ``None`` selects the reference automatically via reference_replicate_quality.
    # After the union master list the reference only decides who seeds the
    # buckets first, so this matters far less than it used to.
    reference_sample: Optional[str] = None
    # LRU size for master MS2 spectra, which are scored against every target
    # peak whose window covers them. Target spectra are read straight through
    # and never cached. Sized to bound the alignment stage's memory: a cached
    # spectrum is a list of (mz, intensity) tuples, ~3.5 KB at 30 fragments, so
    # 4096 entries is ~15 MiB. A run with more master peaks than this re-reads
    # the evicted ones — slower, but the memory ceiling holds.
    ms2_cache_size: int = 4096
    #: A pair can carry many fragments yet share only one common ion.  In the
    #: reliable identity path that is still insufficient evidence; this is the
    #: minimum number the spectrum matcher itself must pair.
    ms2_identity_min_matched_fragments: int = 3
    #: Opt in to the three-state identity helper.  False preserves the legacy
    #: joiner contract for DDA and other callers; ASFAM enables it explicitly.
    use_reliable_ms2_identity: bool = False
    #: Opt in to deterministic augmenting-path assignment plus promotion of
    #: every DIFFERENT/UNJUDGEABLE loser.  False keeps the historical greedy
    #: claim byte-for-byte for existing shared-core callers.
    conserve_detected_peaks: bool = False
    #: Whether MS2 may veto and score an MS1-route match.  True preserves the
    #: shared-core/DDA contract. ASFAM disables it, historically on the grounds
    #: that an MS1 feature carries an all-ion spectrum for its whole co-eluting
    #: segment rather than a precursor-selected one.
    #:
    #: **That justification does not survive the code.** Both ASFAM detection
    #: paths deconvolve — stage 1 by co-elution clustering plus a shape-
    #: correlation gate, stage 1b by correlating against the MS1 peak's own
    #: elution curve — so an MS1-route feature's spectrum is specific to that
    #: peak. The real finding is weaker and different: spectra produced by the
    #: two paths differ enough that cross-path cosines run 0.82-0.84 median
    #: against 0.87-0.96 same-path, so an MS1 cosine is usable evidence but only
    #: at a low bar. :attr:`use_tiered_identity` is the replacement: it vetoes at
    #: ``IdentityConfig.theta_veto`` (0.2) instead of exempting the route. Kept
    #: as-is for callers that have not opted into the tiered rule.
    #:
    #: PRODUCT matching always keeps the MS2 gate regardless of this switch.
    use_ms2_identity_for_ms1: bool = True
    #: RT half-width for the **drift path**: same quantitation ion, same route
    #: and window, but further apart in RT than ``rt_tolerance``. ``0`` (default)
    #: disables it and leaves the box exactly as wide as ``rt_tolerance``.
    #:
    #: Why it exists: ``rt_tolerance`` was set from the drift of peak pairs that
    #: *already matched*, which is survivorship bias — a pair pushed outside the
    #: window by drift never enters that measurement. Measured on the peaks that
    #: founded a master instead of matching one, 23% (rice) / 30% (cancer) sit
    #: within ``mz_tolerance`` of an existing master and outside ``rt_tolerance``
    #: of it, and MS2 calls 67% / 77% of those the same compound.
    drift_rt_tolerance: float = 0.0
    #: RT half-width for the **identity path**: the spectra agree, so the
    #: quantitation ion need not. ``0`` (default) disables the path entirely and
    #: restores the pure ``align_mz`` key.
    #:
    #: Why it exists: both quantitation ions are a per-sample ``argmax``.
    #: ``ms2_rep_ion_mz`` is the tallest fragment, ``ms1_quant_mz`` the tallest
    #: MS1 centroid in the isolation window at the apex cycle — so a near-tie
    #: resolves differently in another sample and one compound keys on two
    #: different ions. On three technical replicates of one sample 18.4% of all
    #: newly founded masters are at the same RT, in the same window, MS2-identical
    #: to an existing master, and keyed on a different ion — 57% of them more than
    #: 0.5 Da away, i.e. a wholly different ion rather than a mass shift.
    #:
    #: MS-DIAL hits the same problem in GC-MS, where there is no precursor at all,
    #: and answers it the same way: ``GcmsPeakJoiner.IsSimilarTo`` matches on RT
    #: plus whole-spectrum similarity and never on a representative ion, and
    #: ``ResetQuantMasses`` then gives the spot one consensus quant mass
    #: (``GcmsPeakJoiner.cs:72-122``).
    identity_rt_tolerance: float = 0.0
    #: Quantitation routes allowed to take the identity path. Empty (default)
    #: means none. ASFAM enables ``(PRODUCT,)``: an ``ms2_only`` peak's spectrum
    #: *is* its identity. :data:`MS1` is deliberately excluded — an ASFAM MS1
    #: feature carries the all-ion spectrum of its whole co-eluting segment, so
    #: two different compounds in one isolation window have near-identical
    #: spectra and the path would merge a compound with its own isotope.
    identity_match_routes: tuple[str, ...] = ()
    #: Cosine the identity path demands. ``0`` (default) falls back to
    #: :attr:`ms2_identity_threshold`.
    #:
    #: They are not the same question and should not share a number.
    #: ``ms2_identity_threshold`` is a **veto**: geometry already vouched for the
    #: pair and the spectra only have to fail to contradict it, so a low bar is
    #: right. The identity path is a **permit**: it drops the m/z key entirely
    #: and the spectra are the only evidence left, so the bar has to be the one
    #: the redundancy pass uses when it makes the same call on the same pair
    #: (``RefinerConfig.ms2_identity_threshold``). Two passes scoring one pair of
    #: spectra must not disagree about what they say.
    identity_match_threshold: float = 0.0
    #: Opt in to the four-tier identity rule
    #: (:func:`metabo_core.alignment.identity.identity`) in place of the box veto
    #: ``_same_compound`` and the identity-path permit ``_reliably_same``. The
    #: joiner's own geometry — route, window, ``align_mz``, the three RT paths —
    #: is untouched; only the "are these one compound" answer changes hands.
    #: ``False`` (default) keeps DDA and GC-MS byte-for-byte on the old rule.
    use_tiered_identity: bool = False
    #: Thresholds for the above. ``None`` means :class:`IdentityConfig` defaults.
    identity: Optional["IdentityConfig"] = None


@dataclass(frozen=True)
class IdentityConfig:
    """Thresholds of the four-tier identity rule (:func:`metabo_core.alignment.identity.identity`).

    Every value is the operating point Phase 0 measured on the three ASFAM
    benchmarks; the rationale for each sits on the field. **These defaults do not
    switch anything on** — the rule only runs where a caller opts in
    (:attr:`JoinerConfig.use_tiered_identity`), so DDA and GC-MS never reach it.

    Read only by stage 7, so every app field that feeds this must live in that
    app's checkpoint-fingerprint exclusion set.
    """

    #: Tier 1 (both sides carry an exact precursor). A *veto*: geometry already
    #: put the pair in one box on the same quantitation ion, and the spectra only
    #: have to fail to contradict it, so the bar is low.
    #:
    #: 0.2 rather than 0.35: on the cancer benchmark 0.35 vetoed 109 pairs known
    #: to share an annotation name against 97 known to differ — worse than a coin
    #: flip on the labelled subset. A structural reason it has to stay low: pairs
    #: detected by *different* paths score 0.82-0.84 median against 0.87-0.96
    #: for same-path pairs, because the two paths deconvolve differently. Much of
    #: this veto's fire therefore lands on one compound seen two ways.
    theta_veto: float = 0.2
    #: Tier 2 (neither side has a precursor, so the spectrum is the whole case).
    #: A *permit*, which is why it is far above :attr:`theta_veto`. Same number
    #: the redundancy pass uses on the same pair of spectra.
    theta_same: float = 0.7
    #: Tier 3 (one side has a precursor, the other only fragments). Higher than
    #: :attr:`theta_same` because two observations in one isolation window at one
    #: RT have *naturally* similar deconvolved spectra — a high cosine may only
    #: say they co-elute. Measured at 0.8: recall 0.35 / 0.53 / 0.62 of
    #: same-name pairs, and 0.15 / 0.08 / 0.19 of known different-name pairs
    #: still merge. A starting point, not a conclusion: expect it to be swept.
    theta_cross: float = 0.8
    #: :attr:`theta_cross`, relaxed when the fragment-only side's isolation-window
    #: centroid corroborates the other side's precursor (see :attr:`lowres_tol`).
    theta_cross_supported: float = 0.7
    #: Tier 1 / tier 2 RT half-width. Costs 3.3-6.2 points of same-name coverage
    #: against 0.5, which is the price of not dragging in neighbours.
    rt_window: float = 0.3
    #: Tier 3 RT half-width, deliberately wider. A fragment chromatogram and its
    #: precursor's do not peak at the same time — measured median apex gap for
    #: one compound is 0.33-0.40 min — so 0.3 costs 10.2-12.8 points of coverage
    #: on exactly the tier that needs it most.
    rt_window_cross: float = 0.5
    #: Slack on the isolation-window edge in tier 3's containment test, in Da.
    eps_win: float = 0.2
    #: Tier 3's chemical anchor: the fragment-only side's quantitation ion must
    #: appear in the precursor side's spectrum at this fraction of its base peak.
    #: MS-DIAL's ``QuantMassExists`` uses the same 10%.
    anchor_rel: float = 0.10
    #: How close the fragment-only side's window centroid must be to the other
    #: side's precursor to count as support, in Da. **Never 0.02**: the centroid
    #: is an intensity-weighted *mixture*, so when the target compound is not the
    #: dominant contributor it reports a different ion entirely. At 0.3 Da it
    #: keeps 94-97% of same-name pairs while excluding 44-65% of different-name
    #: ones — but it is still only ever allowed to *add* confidence (see
    #: :func:`~metabo_core.alignment.identity.identity`), because those excluded
    #: 3-6% of true pairs are the low-abundance ones ASFAM exists to find.
    lowres_tol: float = 0.3
    #: Precursor / fragment m/z agreement, in Da.
    mz_tolerance: float = 0.02
    #: Fragment-matching tolerance of every cosine here.
    ms2_mz_tolerance: float = 0.02
    #: Fewer fragments than this on either side and no cosine is evidence.
    n_min: int = 3
    #: Two rich spectra can share one ubiquitous ion; identity needs this many
    #: actual matched pairs before the cosine counts.
    m_min: int = 3


@dataclass(frozen=True)
class ClusterConfig:
    """Candidate generation and complete-link clustering
    (:func:`metabo_core.alignment.cluster.cluster_observations`)."""

    identity: IdentityConfig = field(default_factory=IdentityConfig)
    #: RT bucket width of the candidate index, in minutes. Neighbours are the
    #: bucket and its two adjacent ones, so the reach is ``2 * rt_bucket``.
    #:
    #: Must not be narrower than the widest RT window any tier uses, or that
    #: tier's candidates are gone before the rule is ever asked — enforced below,
    #: because the failure is silent: the run completes and simply never merges
    #: the pairs the wide window existed for.
    rt_bucket: float = 0.5
    #: LRU size for observation spectra. Bounded on purpose: caching all of them
    #: would make peak memory grow with the sample count.
    ms2_cache_size: int = 4096
    #: Keep observations with different ``isotope_index`` out of one cluster.
    #:
    #: Isotope partners are one compound and share nearly every fragment, so on
    #: the tiers that never compare exact masses (2 and 3) no spectral evidence
    #: can separate M from M+1. The per-sample isotope stage already decided;
    #: without this the satellite is absorbed into the monoisotopic row and stops
    #: existing as a row of its own, which is the one thing the pipeline's
    #: mark-never-delete rule forbids.
    respect_isotope_role: bool = True
    #: Allow one sample to contribute several observations to a cluster. ``False``
    #: reinstates the old "one peak per sample per row" rule *at grouping time*,
    #: which is only useful as an A/B: it is the constraint whose enforcement
    #: during grouping the whole one-pass design exists to remove.
    allow_intra_sample_merge: bool = True

    def __post_init__(self) -> None:
        widest = max(self.identity.rt_window, self.identity.rt_window_cross)
        if self.rt_bucket < widest:
            raise ValueError(
                f"rt_bucket={self.rt_bucket} is narrower than the widest "
                f"identity RT window ({widest}); tier-3 candidates would be "
                f"dropped by the index before identity() ever sees them"
            )


@dataclass
class GapFillConfig:
    """Parameters of ``LcmsGapFiller``, plus the product-ion branch ASFAMProcessor adds.

    See :mod:`metabo_core.alignment.gap_filler`. The two m/z tolerances are not
    free knobs — each must match the window the *detected* peaks of that kind
    were integrated over, or a filled value is measured on a different scale
    than the numbers it sits beside in the quantitation matrix.
    """

    #: Half-width of the peak-top search around the spot's RT centre.
    #: ``LcmsGapFiller._rtTol`` = ``RetentionTimeAlignmentTolerance``.
    rt_tolerance: float = 0.1
    #: Half-width of the MS1 m/z window. ``CentroidMs1Tolerance``.
    ms1_mz_tolerance: float = 0.01
    #: Half-width of the product-ion m/z window. MS-DIAL has no product-ion gap
    #: fill; this matches ``mass_slice_width``, the window ``build_slice_eics_sum``
    #: summed when the detected fragment intensities were measured.
    product_mz_tolerance: float = 0.1
    #: LWMA smoothing level. ``SmoothingLevel`` (kernel = 2*level+1).
    smoothing_level: int = 3
    #: ``IsForceInsertForGapFilling``, MS-DIAL default true: with no broad peak
    #: top in range, take the point nearest the centre as the apex.
    force_insert: bool = True
    #: The chromatogram spans ``rt_center +/- rt_expansion * max(peak_width,
    #: min_peak_width)``. ``FromTimes(c, w).ExtendRelative(1)`` = ``c +/- 1.5w``.
    rt_expansion: float = 1.5
    min_peak_width: float = 0.2


@dataclass(frozen=True)
class RefinerConfig:
    """Gates for the cross-sample redundancy pass. Both are derived, not free knobs.

    See :mod:`metabo_core.alignment.refiner`. ``mz_tolerance`` / ``rt_tolerance``
    are the alignment tolerances — the same two the joiner matched with. The
    refiner then *narrows* both, exactly as ``TryMergeToMaster`` does
    (``LcmsAlignmentRefiner.cs:92-102``), because two spots have to be much
    closer than the match window to be one compound split in two.

    The m/z gate only ever decides *same-route* redundancy. Two rows quantified on
    different ions — an MS1 precursor and a product ion — have no comparable m/z
    at all, and :func:`~metabo_core.alignment.refiner.mark_ms1_covered` judges
    those on RT and MS2 identity instead. That is what
    ``ms2_identity_threshold`` / ``ms2_mz_tolerance`` are for.
    """

    mz_tolerance: float = 0.02
    rt_tolerance: float = 0.1
    #: Cosine at or above which a visible MS1-route row and a visible
    #: product-route row at the same RT are one compound, so the product row is
    #: not an MS2-only detection. MS2 is the only identity signal in ASFAM that
    #: does not depend on the route: nothing selected a precursor, so both rows'
    #: spectra come from the same all-ion segment and deconvolve to the same
    #: component. Distinct compounds co-eluting in one isolation window do *not*
    #: score high here — MSDEC separates them, which is what makes the cosine
    #: worth trusting and the reported precursor m/z (a window centroid) not.
    ms2_identity_threshold: float = 0.7
    #: Fragment-matching tolerance of that cosine. The joiner's
    #: ``JoinerConfig.ms2_mz_tolerance``; keep them equal, or the same pair of
    #: spectra scores differently in the two passes.
    ms2_mz_tolerance: float = 0.02
    #: ``Math.Min(param.RetentionTimeAlignmentTolerance, 0.1)`` (``:92``). Without
    #: this cap a user who widens the alignment RT tolerance in the GUI silently
    #: widens the redundancy gate with it — at 0.3 min it would be 3x MS-DIAL's
    #: and start merging distinct compounds.
    rt_tolerance_cap: float = 0.1
    #: ``rtTol * 0.5`` (``:102``).
    rt_gate_factor: float = 0.5
    #: Above this m/z, ``TryMergeToMaster`` (``:96-98``) re-expresses the m/z
    #: tolerance as the ppm it represents at the pivot and re-evaluates it at the
    #: spot's own m/z, so the gate grows with mass. Below the pivot it is flat.
    #: ``0`` disables the widening.
    ppm_pivot_mz: float = 500.0
    #: Same-route spots are reconciled before gap fill in ASFAM.  Its Feature
    #: refiner therefore disables this legacy mark-only pass; other callers keep
    #: it by default.
    same_route_redundancy: bool = True
    #: Reliable identity controls are opt-in so shared-core defaults retain the
    #: old behaviour.  ASFAM enables them for cross-route coverage (and any
    #: caller that elects to keep same-route Feature refinement).
    use_reliable_ms2_identity: bool = False
    ms2_identity_min_fragments: int = 3
    ms2_identity_min_matched_fragments: int = 3
    same_route_ms2_identity_threshold: float = 0.5
    #: A hidden isotope/adduct/spectral/ISF row cannot hide a visible row when
    #: this correctness mode is enabled.
    visible_keepers_only: bool = False
    #: PRODUCT rows are comparable only inside the same isolation window.
    require_product_window_match: bool = False
    #: Cross-route identity must come from compatible ASFAM acquisition
    #: segments/windows; conservative mismatch leaves PRODUCT visible.
    require_cross_route_window_match: bool = False
    #: Preserve a PRODUCT row when it owns natural detections absent from its
    #: MS1 coverer, and record a partial relation instead.  ASFAM opts in;
    #: False retains the historical shared-core mark-only behaviour.
    preserve_cross_route_unique_detections: bool = False
    #: RT half-width for merging two rows the *joiner* split, in minutes. ``0``
    #: (default) keeps :attr:`rt_gate`.
    #:
    #: :attr:`rt_gate` is 0.05 min and :attr:`mz_gate` is 0.02 Da, joined by AND.
    #: The joiner splits a compound in two exactly when the pair is more than
    #: 0.02 Da apart **or** more than ``JoinerConfig.rt_tolerance`` (0.2 min)
    #: apart. Those two regions do not intersect: **no split the joiner can make
    #: is reachable by this pass**, which is why it merged 801 of 51,835 rice
    #: spots while 30% of the rows were still the same compound as another row.
    #: This widens the RT half of the gate to the joiner's reach so the pass can
    #: see what the joiner leaves behind.
    merge_rt_tolerance: float = 0.0
    #: Routes where the m/z half of the gate may be dropped because the spectra
    #: carry identity instead — the mirror of
    #: :attr:`JoinerConfig.identity_match_routes`, and only meaningful for routes
    #: this pass already requires a reliable SAME decision on. Empty (default)
    #: keeps the m/z gate mandatory everywhere.
    merge_identity_routes: tuple[str, ...] = ()
    #: Identity rule for the cross-route pass, per SPEC 2026-08-02 §10.4. When
    #: set, tier 3 has already settled "one MS1 row and one product row are one
    #: compound" during grouping, so this pass keeps only the residue that rule
    #: could **not** judge: a pair it calls SAME or DIFFERENT is left alone, and
    #: what remains records :attr:`Feature.alignment_relation` **without** setting
    #: ``is_duplicate`` — a suspicion is not grounds for hiding a row.
    #: ``None`` (default) keeps the historical mark-as-duplicate behaviour, which
    #: is what DDA and GC-MS run.
    ms1_covered_identity: Optional[IdentityConfig] = None

    @property
    def rt_gate(self) -> float:
        """Half-width of the RT gate, in minutes. Capped, then halved."""
        return min(self.rt_tolerance, self.rt_tolerance_cap) * self.rt_gate_factor

    def mz_gate(self, mz: float) -> float:
        """Half-width of the m/z gate at ``mz``, in Da."""
        if self.ppm_pivot_mz > 0 and mz > self.ppm_pivot_mz:
            return self.mz_tolerance * mz / self.ppm_pivot_mz
        return self.mz_tolerance
