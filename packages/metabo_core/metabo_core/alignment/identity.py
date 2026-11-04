"""Reliable, three-state MS2 identity evidence for alignment.

Geometric proximity says that two features *may* be the same compound.  MS2
can either support that identity, contradict it, or be too sparse to say.  The
last case must stay distinct from both answers: treating one shared fragment as
identity is a false merge, while treating missing spectra as different compounds
needlessly splits every sparse feature.

:func:`ms2_identity_evidence` answers that question from the spectra alone and
is what DDA and GC-MS use. :func:`identity` answers it from **everything an
observation carries** — precursor, isolation window, acquisition segment,
representative fragment and spectrum — and dispatches on *what evidence the two
sides happen to have* rather than on which chromatogram they were quantified
from. That distinction is the point: an all-ion run gives one compound an exact
precursor in the sample where its MS1 peak cleared the amplitude floor and only
fragments in the sample where it did not, and grouping by quantitation route
guarantees those two observations can never meet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from metabo_core.algorithms.similarity import cosine_similarity
from metabo_core.config import IdentityConfig


class IdentityState(str, Enum):
    """The only three scientifically meaningful identity outcomes."""

    SAME = "same"
    DIFFERENT = "different"
    UNJUDGEABLE = "unjudgeable"


@dataclass(frozen=True)
class IdentityEvidence:
    """Identity decision plus the measurements that produced it."""

    state: IdentityState
    cosine: float
    n_matched_fragments: int
    n_left_fragments: int
    n_right_fragments: int

    @property
    def is_reliable(self) -> bool:
        return self.state is not IdentityState.UNJUDGEABLE


def ms2_identity_evidence(
    left: Iterable[tuple[float, float]],
    right: Iterable[tuple[float, float]],
    *,
    mz_tolerance: float,
    same_threshold: float,
    min_fragments: int = 3,
    min_matched_fragments: int = 3,
) -> IdentityEvidence:
    """Return reliable three-state MS2 identity evidence.

    A cosine is identity evidence only when both spectra contain enough total
    fragments *and* the comparison actually matches enough fragments.  The
    latter guard matters when two rich spectra share only one ubiquitous ion:
    their cosine may still be high, but one match is not a molecular identity.
    """
    left_peaks = list(left)
    right_peaks = list(right)
    n_left, n_right = len(left_peaks), len(right_peaks)
    min_total = max(0, int(min_fragments))
    min_matched = max(0, int(min_matched_fragments))

    if n_left < min_total or n_right < min_total:
        return IdentityEvidence(
            IdentityState.UNJUDGEABLE, 0.0, 0, n_left, n_right,
        )

    cosine, n_matched = cosine_similarity(
        left_peaks, right_peaks, float(mz_tolerance),
    )
    if n_matched < min_matched:
        return IdentityEvidence(
            IdentityState.UNJUDGEABLE,
            float(cosine),
            int(n_matched),
            n_left,
            n_right,
        )

    state = (
        IdentityState.SAME
        if cosine >= float(same_threshold)
        else IdentityState.DIFFERENT
    )
    return IdentityEvidence(
        state, float(cosine), int(n_matched), n_left, n_right,
    )


# ---------------------------------------------------------------------------
# Route-agnostic identity: four tiers over what each side actually carries
# ---------------------------------------------------------------------------

#: Label of the corroborating value when it is the isolation-window centroid
#: that stage 2 stored on a fragment-only feature (``ms1_window_mz``) rather
#: than the feature's reported precursor.
WINDOW_SOURCE = "ms1_window"

#: ``lowres_source`` values that tier 3 accepts as a corroborating low-resolution
#: precursor estimate. ``ms1_lowres`` / ``ms1_relaxed`` are ``mz_source`` values:
#: the reported precursor itself is a window centroid, or a real MS1 peak that
#: merely failed the amplitude floor (the stronger of the two; treated alike
#: until there is data to separate them). :data:`WINDOW_SOURCE` is the stored
#: window centroid of a feature whose reported precursor came from elsewhere —
#: since v1.4.2610xx that is the neutral-loss consensus path, which no longer
#: writes the centroid into ``ms1_precursor_mz``. The corroboration itself is
#: the same measurement it always was.
LOWRES_SOURCES = ("ms1_lowres", "ms1_relaxed", WINDOW_SOURCE)


def lowres_evidence(peak: Any) -> tuple[Optional[float], str]:
    """``(lowres_mz, lowres_source)`` for an observation built from ``peak``.

    When the peak's own reported precursor is a low-resolution estimate
    (``mz_source`` in ``ms1_lowres`` / ``ms1_relaxed``) that value corroborates,
    exactly as before. Otherwise the isolation-window centroid stage 2 stored
    on every fragment-only feature (``ms1_window_mz``) plays the same role
    under :data:`WINDOW_SOURCE`, whichever path later supplied the reported
    precursor. A neutral-loss consensus m/z is deliberately never used here: it
    is computed from the very fragments the cosine already weighs, so it would
    corroborate nothing independently — the same reason ``inferred_mz`` is
    absent from :class:`IdentityObservation`.
    """
    source = str(getattr(peak, "mz_source", "") or "")
    if source in ("ms1_lowres", "ms1_relaxed"):
        mz = getattr(peak, "ms1_precursor_mz", None)
        if mz is not None:
            return float(mz), source
    window = getattr(peak, "ms1_window_mz", None)
    if window is not None:
        return float(window), WINDOW_SOURCE
    return None, ""


def static_spectrum(
    peaks: Sequence[tuple[float, float]],
) -> Callable[[], Sequence[tuple[float, float]]]:
    """Adapter for a caller whose spectrum is already in memory."""
    return lambda: peaks


@dataclass(frozen=True)
class IdentityObservation:
    """What one observation can offer as identity evidence.

    Deliberately not a ``CandidateFeature``: the rule must be usable from the
    joiner (where one side is an aggregated master) and from a clustering pass,
    and it must not be able to reach for anything outside this list. In
    particular ``inferred_mz`` is absent on purpose — a precursor inferred from a
    library hit would make the chain annotate -> align -> pick a representative
    -> annotate circular.

    ``read_spectrum`` is a thunk so that the two free gates — fragment counts —
    can reject a pair before anything touches the disk.
    """

    #: Peak apex, in minutes.
    rt_apex: float
    #: Fragment count, readable without loading the spectrum.
    n_fragments: int
    #: ``() -> [(mz, intensity), ...]``. Called at most once per side.
    read_spectrum: Callable[[], Sequence[tuple[float, float]]]
    #: Exact precursor: the MS1 ROI centroid, or the base peak in the isolation
    #: window at the apex cycle. ``None`` for an observation that was only ever
    #: seen through its fragments — which is what selects the tier.
    ms1_quant_mz: Optional[float] = None
    #: Tallest fragment. On a fragment-only observation this is the ion it was
    #: quantified on.
    ms2_rep_ion_mz: Optional[float] = None
    #: Integer isolation window (``precursor_mz_nominal``): an acquisition fact,
    #: true of every observation, and the one hard constraint on tier 3.
    window: Optional[int] = None
    #: Centre m/z of that isolation window as the instrument recorded it. The
    #: integer ``window`` is floor(centre), so the isolated range is
    #: ``centre ± 0.5`` and not ``window ± 0.5`` (above m/z ~800 the two differ
    #: by half a window). ``None`` when the producer did not record it, in
    #: which case the integer window is used as the centre, as before.
    window_mz: Optional[float] = None
    #: Acquisition segment. Empty means "unknown", not "no segment".
    segment: str = ""
    #: Intensity-weighted centroid of the isolation window, when that is all the
    #: precursor evidence there is. Corroborating only — see :func:`identity`.
    lowres_mz: Optional[float] = None
    #: Which of :data:`LOWRES_SOURCES` produced ``lowres_mz``.
    lowres_source: str = ""

    @property
    def has_precursor(self) -> bool:
        return self.ms1_quant_mz is not None


@dataclass(frozen=True)
class IdentityVerdict:
    """The decision, the tier that made it, and what it was made on."""

    state: IdentityState
    #: 1 both sides have a precursor / 2 neither does / 3 exactly one does /
    #: 4 the pair never reached a tier, because a hard gate rejected it.
    tier: int
    cosine: float = 0.0
    n_matched_fragments: int = 0
    #: Evidence that may only ever *raise* confidence. Never a veto.
    support: Mapping[str, Any] = field(default_factory=dict)
    #: Tier 2 fell through to the representative-fragment m/z because the
    #: spectra could not be judged. A merge, but on the weakest evidence here.
    weak: bool = False

    @property
    def is_same(self) -> bool:
        return self.state is IdentityState.SAME


def _judgeable(
    left: IdentityObservation, right: IdentityObservation, config: IdentityConfig,
) -> tuple[bool, float, int]:
    """``(is_evidence, cosine, n_matched)``; the cosine is 0 when it is not evidence.

    Both free gates first: a fragment count is on the observation, a spectrum may
    be behind a disk read.
    """
    if left.n_fragments < config.n_min or right.n_fragments < config.n_min:
        return False, 0.0, 0
    left_peaks = left.read_spectrum()
    right_peaks = right.read_spectrum()
    if not left_peaks or not right_peaks:
        return False, 0.0, 0
    cosine, n_matched = cosine_similarity(
        left_peaks, right_peaks, float(config.ms2_mz_tolerance),
    )
    if n_matched < config.m_min:
        return False, float(cosine), int(n_matched)
    return True, float(cosine), int(n_matched)


def _gates_pass(
    left: IdentityObservation, right: IdentityObservation,
    config: IdentityConfig, rt_window: float,
) -> bool:
    """Same segment, adjacent-or-equal isolation window, close enough in RT.

    A missing segment or window is unknown rather than mismatched, so it abstains
    — the alternative would reject every pair from a caller that does not carry
    acquisition metadata at all.
    """
    if left.segment and right.segment and left.segment != right.segment:
        return False
    if left.window is not None and right.window is not None:
        if abs(int(left.window) - int(right.window)) > 1:
            return False
    return abs(float(left.rt_apex) - float(right.rt_apex)) <= rt_window


def _anchor_present(
    spectrum: Sequence[tuple[float, float]], mz: float, config: IdentityConfig,
) -> bool:
    """Is ``mz`` a real ion of ``spectrum`` rather than a trace at its noise floor?

    MS-DIAL asks the same question of a quantitation mass before it will use one
    (``QuantMassExists``), and with the same 10% of the base peak.
    """
    if not spectrum:
        return False
    base = max(intensity for _mz, intensity in spectrum)
    if base <= 0:
        return False
    floor = base * float(config.anchor_rel)
    tolerance = float(config.ms2_mz_tolerance)
    return any(
        abs(float(peak_mz) - float(mz)) <= tolerance and float(intensity) >= floor
        for peak_mz, intensity in spectrum
    )


def _tier3(
    precursor_side: IdentityObservation,
    fragment_side: IdentityObservation,
    config: IdentityConfig,
) -> IdentityVerdict:
    """One side has an exact precursor, the other only fragments.

    Three conditions, all required, cheapest first:

    ``containment``  the precursor must lie inside the fragment side's isolation
                     window, and the fragment it was quantified on cannot be
                     heavier than that precursor. Free, and a strong exclusion.
    ``spectrum``     the cosine, at :attr:`IdentityConfig.theta_cross`. On its own
                     this is *not* enough: two observations in one window at one
                     RT deconvolve from the same co-eluting mixture, so a high
                     cosine can mean no more than "these co-elute".
    ``anchor``       the fragment side's quantitation ion has to be a real,
                     substantial ion of the precursor side's spectrum. This is
                     what actually tests "the fragment-only row is this
                     compound's fragment view" rather than "they are neighbours".

    The window centroid, when the fragment side has one, only relaxes the cosine
    bar. It is an intensity-weighted *mixture* centroid: where the compound of
    interest is not the dominant contributor it reports somebody else's ion, so
    letting it veto would systematically drop exactly the low-abundance features
    all-ion acquisition exists to recover.
    """
    support: dict[str, Any] = {}
    theta = float(config.theta_cross)
    if (
        fragment_side.lowres_mz is not None
        and fragment_side.lowres_source in LOWRES_SOURCES
    ):
        delta = abs(
            float(precursor_side.ms1_quant_mz) - float(fragment_side.lowres_mz)
        )
        support["lowres_mz_delta"] = delta
        support["lowres_source"] = fragment_side.lowres_source
        if delta <= float(config.lowres_tol):
            theta = float(config.theta_cross_supported)

    def undecided(cosine: float = 0.0, n_matched: int = 0) -> IdentityVerdict:
        return IdentityVerdict(
            IdentityState.UNJUDGEABLE, 3, cosine, n_matched, support,
        )

    rep_mz = fragment_side.ms2_rep_ion_mz
    precursor = float(precursor_side.ms1_quant_mz)
    if rep_mz is None or fragment_side.window is None:
        return undecided()
    # Containment is judged against the window the instrument actually
    # isolated: centred on the recorded target when the producer stored it,
    # on the integer window otherwise.
    centre = (
        float(fragment_side.window_mz)
        if fragment_side.window_mz is not None
        else float(fragment_side.window)
    )
    half_window = 0.5 + float(config.eps_win)
    if not (centre - half_window <= precursor <= centre + half_window):
        return undecided()
    if float(rep_mz) > precursor + float(config.mz_tolerance):
        return undecided()

    ok, cosine, n_matched = _judgeable(precursor_side, fragment_side, config)
    if not ok or cosine < theta:
        return undecided(cosine, n_matched)
    if not _anchor_present(precursor_side.read_spectrum(), float(rep_mz), config):
        return undecided(cosine, n_matched)
    return IdentityVerdict(IdentityState.SAME, 3, cosine, n_matched, support)


def identity(
    left: IdentityObservation,
    right: IdentityObservation,
    config: Optional[IdentityConfig] = None,
    *,
    check_geometry: bool = True,
) -> IdentityVerdict:
    """Are these two observations the same compound?

    Dispatch is on **what each side carries**, never on which chromatogram it was
    quantified from, so an ``ms1_detected`` observation and an ``ms2_only`` one
    can reach a verdict at all (tier 3) instead of being separated before the
    question is asked.

    ``UNJUDGEABLE`` means *do not merge*, the opposite of the abstain-toward-one-
    box convention that suits a gate sitting behind a geometric box. Here
    geometry is not on its own a reason to merge, so silence cannot be consent.

    ``check_geometry=False`` skips the segment / window / RT gates for a caller
    that has already applied its own — the joiner, whose three RT paths are wider
    than :attr:`IdentityConfig.rt_window` and must not be silently narrowed by
    swapping the identity rule underneath them.
    """
    config = config or IdentityConfig()
    both = left.has_precursor and right.has_precursor
    neither = not left.has_precursor and not right.has_precursor
    tier = 1 if both else (2 if neither else 3)

    if check_geometry:
        window = config.rt_window if tier != 3 else config.rt_window_cross
        if not _gates_pass(left, right, config, float(window)):
            return IdentityVerdict(IdentityState.UNJUDGEABLE, 4)

    if tier == 1:
        # Different exact precursors means different formulae. Similar spectra
        # then say "homologue" or "isomer", not "same compound", so there is
        # nothing to decide rather than something to merge.
        if abs(
            float(left.ms1_quant_mz) - float(right.ms1_quant_mz)
        ) > float(config.mz_tolerance):
            return IdentityVerdict(IdentityState.UNJUDGEABLE, 1)
        ok, cosine, n_matched = _judgeable(left, right, config)
        if ok and cosine < float(config.theta_veto):
            return IdentityVerdict(IdentityState.DIFFERENT, 1, cosine, n_matched)
        return IdentityVerdict(IdentityState.SAME, 1, cosine, n_matched)

    if tier == 2:
        ok, cosine, n_matched = _judgeable(left, right, config)
        if ok:
            state = (
                IdentityState.SAME
                if cosine >= float(config.theta_same)
                else IdentityState.DIFFERENT
            )
            return IdentityVerdict(state, 2, cosine, n_matched)
        # No usable spectra. Fall back to the representative fragment m/z --
        # weak evidence, and demoted from the primary key it used to be: it is
        # the argmax of a 0.1 Da slice, so a near-tie between two fragments
        # resolves differently in another sample and one compound ends up
        # keyed on two ions.
        if (
            left.ms2_rep_ion_mz is not None
            and right.ms2_rep_ion_mz is not None
            and abs(
                float(left.ms2_rep_ion_mz) - float(right.ms2_rep_ion_mz)
            ) <= float(config.mz_tolerance)
        ):
            return IdentityVerdict(
                IdentityState.SAME, 2, cosine, n_matched, weak=True,
            )
        return IdentityVerdict(IdentityState.UNJUDGEABLE, 2, cosine, n_matched)

    if left.has_precursor:
        return _tier3(left, right, config)
    return _tier3(right, left, config)
