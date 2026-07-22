"""Cross-replicate alignment configuration shared by core and apps."""
from __future__ import annotations

from dataclasses import dataclass
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
    #: shared-core/DDA contract. ASFAM disables it explicitly because an MS1
    #: feature carries an AIF spectrum for the co-eluting segment, not a
    #: precursor-selected identity spectrum. PRODUCT matching always keeps the
    #: MS2 gate regardless of this switch.
    use_ms2_identity_for_ms1: bool = True


@dataclass
class GapFillConfig:
    """Parameters of ``LcmsGapFiller``, plus the product-ion branch ASFAM adds.

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

    @property
    def rt_gate(self) -> float:
        """Half-width of the RT gate, in minutes. Capped, then halved."""
        return min(self.rt_tolerance, self.rt_tolerance_cap) * self.rt_gate_factor

    def mz_gate(self, mz: float) -> float:
        """Half-width of the m/z gate at ``mz``, in Da."""
        if self.ppm_pivot_mz > 0 and mz > self.ppm_pivot_mz:
            return self.mz_tolerance * mz / self.ppm_pivot_mz
        return self.mz_tolerance
