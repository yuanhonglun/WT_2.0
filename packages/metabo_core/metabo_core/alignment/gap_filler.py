"""MS-DIAL-style gap filling: integrate a spot's quantitation ion in the samples
where no peak was picked, so the quantitation matrix has no holes.

Ported from ``LcmsGapFiller.cs:37-177`` and the ``Chromatogram`` helpers it
calls (``IsBroadPeakTop`` / ``IsPeakTop`` / ``SearchLeftEdge[Hard]`` /
``SearchRightEdge[Hard]`` / ``CalculateArea``, ``Chromatogram.cs:359-418,
571-608, 758-760``).

**Every feature is quantified on exactly one ion**, and gap filling is one
algorithm parameterized by where that ion's chromatogram comes from:

===========================  =====================  ==========================
feature                      quantitation ion       chromatogram
===========================  =====================  ==========================
``ms2_only``                 ``ms2_rep_ion_mz``     product-ion SUM slice
``ms1_detected``             ``ms1_quant_mz``       MS1 SUM window
===========================  =====================  ==========================

``ms2_only`` features are already quantified on a fragment — Stage 2 sets
``ms1_height = ms2_intensity[argmax]`` — so the fragment, not the precursor,
is what a missing sample must be integrated on. That also sidesteps a trap:
most ``ms2_only`` precursor m/z values are the integer floor of a 1-Da
isolation window, and a tolerance-width window around an integer is noise.

``ms1_quant_mz`` rather than ``ms1_precursor_mz`` for the MS1 branch, for the
same class of reason — see the field's comment on ``CandidateFeature``.

Which chromatogram *kind* a spot uses is decided by its tallest **detected**
peak, the same peak MS-DIAL takes the m/z centre from
(``LcmsGapFiller.cs:46``), so the filled value lands on the scale the spot's
existing numbers already live on.

That last clause only holds because the joiner refuses to put two quantitation
routes in one spot. Until it did, 11% of exported rows mixed them: the tallest
detected peak could be ``ms2_only`` while its neighbours in the same row were
``ms1_detected``, and the fill would then integrate a product-ion slice into a
row of MS1 heights. Both sides read from
:attr:`~metabo_core.models.CandidateFeature.quant_route`, deliberately -- the
route a spot is grouped by and the trace it is filled on are one decision.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np

from metabo_core.algorithms.msdial_peak_spotting import _lwma_msdial
from metabo_core.config import GapFillConfig
from metabo_core.models import MS1, PRODUCT, CandidateFeature

DETECTED = "detected"
FILLED = "filled"
NO_SIGNAL = "no_signal"

__all__ = [
    "DETECTED", "FILLED", "NO_SIGNAL", "MS1", "PRODUCT",
    "QuantIon", "GapFillTarget", "GapFillResult", "ChromatogramProvider",
    "quant_ion", "gap_fill_target", "fill_from_chromatogram",
    "make_filled_peak", "fill_spot",
    "needs_requantitation", "requantify_peak", "consensus_quant_ion",
    "choose_quant_channel",
]


@dataclass(frozen=True)
class QuantIon:
    """The ion a feature is quantified on, and how to pull its chromatogram."""

    mz: float
    kind: str            # MS1 | PRODUCT
    tolerance: float
    #: MS2 isolation-window key; meaningless for :data:`MS1`.
    channel: int = 0


@dataclass(frozen=True)
class GapFillTarget:
    """Everything a missing sample needs, derived once per spot."""

    quant: QuantIon
    rt_center: float
    rt_lo: float
    rt_hi: float
    peak_width: float
    estimated_noise: float
    #: The m/z segment the tallest detected peak was measured in (ASFAM splits
    #: one sample across many mzML files); empty for single-file acquisitions.
    segment_name: str = ""


@dataclass
class GapFillResult:
    status: str
    height: float = 0.0
    area: float = 0.0
    rt_apex: float = 0.0
    rt_left: float = 0.0
    rt_right: float = 0.0
    sn_ratio: float = 0.0
    #: True when no broad peak top existed within ``+/-rt_tolerance`` and the
    #: peak was framed around the point nearest the RT centre instead
    #: (``IsForceInsertForGapFilling``). The height is then whatever the trace
    #: happened to carry there, so this separates "the picker missed a real
    #: peak" from "there is no peak, only background" — a distinction the three
    #: ``gap_fill_status`` states cannot make, since force-inserting into noise
    #: still yields a nonzero height and therefore ``filled``.
    forced: bool = False


#: ``(sample_id, target) -> (rt, intensity)`` over ``[target.rt_lo, rt_hi]``, or
#: ``None`` when that sample has no chromatogram there at all (segment missing,
#: channel not acquired). Intensities are raw; this module smooths them.
ChromatogramProvider = Callable[[str, GapFillTarget], Optional[tuple[np.ndarray, np.ndarray]]]


# ---------------------------------------------------------------------------
# Quantitation ion
# ---------------------------------------------------------------------------

def quant_ion(feature: CandidateFeature, config: GapFillConfig) -> Optional[QuantIon]:
    """The ion ``feature.ms1_height`` was measured on. ``None`` if unknowable.

    Branches on ``CandidateFeature.quant_route``, which is also what the joiner
    groups a spot's peaks by — one decision, in one place. A spot whose peaks all
    share a route is therefore a spot every sample can be filled on the same kind
    of trace.

    ``None`` rather than a fallback ion: a fill has to land on the trace the
    detected heights beside it were read off, and there is no second ion that
    qualifies. (The joiner's ``align_mz`` *does* fall back, because it needs a
    key for every peak it has to place. Different questions.)
    """
    if feature.quant_route == PRODUCT:
        if feature.ms2_rep_ion_mz is None:
            return None
        return QuantIon(
            mz=float(feature.ms2_rep_ion_mz),
            kind=PRODUCT,
            tolerance=config.product_mz_tolerance,
            channel=int(feature.precursor_mz_nominal),
        )
    if feature.ms1_quant_mz is None:
        return None
    return QuantIon(
        mz=float(feature.ms1_quant_mz),
        kind=MS1,
        tolerance=config.ms1_mz_tolerance,
        channel=int(feature.precursor_mz_nominal),
    )


def _estimated_noise(peaks: list[CandidateFeature]) -> float:
    """``GetEstimatedNoise``: the loudest noise floor among the detected peaks.

    ASFAMProcessor stores S/N rather than the noise itself, so invert it. A peak with no
    S/N contributes nothing instead of a zero that would win the ``max``.
    """
    noises = [
        p.ms1_height / p.ms1_sn
        for p in peaks
        if p.ms1_sn and p.ms1_sn > 0 and p.ms1_height
    ]
    return max(noises) if noises else 0.0


def gap_fill_target(
    detected: list[CandidateFeature], config: GapFillConfig,
) -> Optional[GapFillTarget]:
    """Derive a spot's fill target from its detected peaks (``LcmsGapFiller:44-49``).

    RT centre is the arithmetic mean of the detected apices; the m/z and the
    chromatogram kind both come from the *tallest* detected peak (a
    representative value, deliberately not an average); the peak width is the
    widest detected peak.
    """
    if not detected:
        return None
    tallest = max(detected, key=lambda p: p.ms1_height or 0.0)
    quant = quant_ion(tallest, config)
    if quant is None:
        return None

    rt_center = float(np.mean([p.rt_apex for p in detected]))
    peak_width = max(p.rt_right - p.rt_left for p in detected)
    half = config.rt_expansion * max(peak_width, config.min_peak_width)
    return GapFillTarget(
        quant=quant,
        rt_center=rt_center,
        rt_lo=rt_center - half,
        rt_hi=rt_center + half,
        peak_width=peak_width,
        estimated_noise=_estimated_noise(detected),
        segment_name=tallest.segment_name,
    )


# ---------------------------------------------------------------------------
# Chromatogram.cs helpers
# ---------------------------------------------------------------------------

def _is_peak_top(y: np.ndarray, i: int) -> bool:
    """``Chromatogram.IsPeakTop`` (``:571-577``)."""
    if i < 1 or i >= y.size - 1:
        return False
    return (
        y[i - 1] <= y[i] and y[i] >= y[i + 1]
        and (y[i - 1] != y[i] or y[i] != y[i + 1])
    )


def _is_broad_peak_top(y: np.ndarray, i: int) -> bool:
    """``Chromatogram.IsBroadPeakTop`` (``:602-608``).

    Note ``&&`` binds tighter than ``||`` in C#: the trailing disjunction is
    ``(i-2 >= 0 && ...) || (i+2 < n && ...)``.
    """
    n = y.size
    if i < 1 or i >= n - 1:
        return False
    if not (y[i - 1] <= y[i] and y[i] >= y[i + 1]):
        return False
    return (i - 2 >= 0 and y[i - 2] <= y[i - 1]) or (i + 2 < n and y[i + 1] >= y[i + 2])


def _search_left_edge(y: np.ndarray, start: int, limit: int, hard: bool) -> int:
    """``SearchLeftEdge`` / ``SearchLeftEdgeHard`` (``:359-381``).

    Walks left off the flank. ``hard`` stops on a plateau as well as on a rise.
    """
    left = max(start, 0)
    limit = max(limit, 0)
    while limit < left:
        if (y[left - 1] >= y[left]) if hard else (y[left - 1] > y[left]):
            break
        left -= 1
    return left


def _search_right_edge(y: np.ndarray, start: int, limit: int, hard: bool) -> int:
    """``SearchRightEdge`` / ``SearchRightEdgeHard`` (``:396-418``)."""
    right = min(start, y.size - 1)
    limit = min(limit, y.size - 1)
    while right < limit:
        if (y[right] <= y[right + 1]) if hard else (y[right] < y[right + 1]):
            break
        right += 1
    return right


def _trapezoid(y: np.ndarray, rt: np.ndarray, i: int, j: int) -> float:
    """``Chromatogram.CalculateArea`` (``:758-760``)."""
    return float((y[i] + y[j]) * abs(rt[i] - rt[j]) / 2.0)


# ---------------------------------------------------------------------------
# The fill itself
# ---------------------------------------------------------------------------

def fill_from_chromatogram(
    rt: np.ndarray,
    intensity: np.ndarray,
    target: GapFillTarget,
    config: GapFillConfig,
) -> GapFillResult:
    """``GapFillCore`` + ``GetNearestPeak`` + ``SetAlignmentChromPeakFeature``.

    ``intensity`` is raw; it is LWMA-smoothed here exactly as MS-DIAL smooths
    before measuring, and every number below is read off the smoothed trace.
    Area is above-baseline and in seconds (x60), matching what ``DetectedPeak``
    stores for a real peak.
    """
    if rt.size == 0:
        return GapFillResult(status=NO_SIGNAL)

    y = _lwma_msdial(np.asarray(intensity, dtype=np.float64), config.smoothing_level)
    n = y.size
    center = target.rt_center
    rt_tol = config.rt_tolerance

    nearest_top = -1
    for i in range(n):
        if i - 2 < 0 or i + 2 >= n:
            continue
        if rt[i] < center - rt_tol:
            continue
        if center + rt_tol < rt[i]:
            break
        if _is_broad_peak_top(y, i) and (
            nearest_top < 0
            or abs(rt[nearest_top] - center) > abs(rt[i] - center)
        ):
            nearest_top = i

    forced = nearest_top < 0
    if forced:
        if not config.force_insert:
            return GapFillResult(status=NO_SIGNAL)
        # No broad peak top in range: frame a peak around the point nearest the
        # centre, walking at most 5 points out on each side.
        top = int(np.argmin(np.abs(rt - center)))
        left = _search_left_edge(y, max(top - 1, 0), max(top - 5, 0), hard=False)
        right = _search_right_edge(y, min(top + 1, n - 1), min(top + 5, n - 1), hard=False)
    else:
        left = _search_left_edge(y, max(nearest_top - 2, 0), 0, hard=True)
        right = _search_right_edge(y, min(nearest_top + 2, n - 1), n - 1, hard=True)
        if not config.force_insert and (nearest_top - left < 2 or right - nearest_top < 2):
            return GapFillResult(status=NO_SIGNAL)
        top = nearest_top
        for i in range(left + 1, right):
            if _is_peak_top(y, i) and y[top] - y[i] < 0.0:
                top = i

    area_above_zero = sum(_trapezoid(y, rt, i, i + 1) for i in range(left, right)) * 60.0
    baseline = _trapezoid(y, rt, left, right) * 60.0
    height = float(y[top])

    return GapFillResult(
        status=FILLED if height > 0.0 else NO_SIGNAL,
        height=height,
        area=area_above_zero - baseline,
        rt_apex=float(rt[top]),
        rt_left=float(rt[left]),
        rt_right=float(rt[right]),
        sn_ratio=(height / target.estimated_noise) if target.estimated_noise > 0 else 0.0,
        forced=forced,
    )


#: MS-DIAL's ``ResetQuantMasses`` ignores cells below a tenth of the row's
#: tallest and demands the chosen mass carry a tenth of the representative
#: spectrum's base peak (``GcmsPeakJoiner.cs:99-101, 135``).
_QUANT_MASS_MIN_RELATIVE = 0.1


def consensus_quant_ion(
    detected: list[CandidateFeature],
    representative: Optional[CandidateFeature],
    config: GapFillConfig,
) -> Optional[QuantIon]:
    """The one ion a whole row is quantified on — a vote, not an argmax.

    Ported from ``GcmsPeakJoiner.ResetQuantMasses`` (``GcmsPeakJoiner.cs:72-122``),
    which exists because GC-MS spots are matched on spectrum and RT and so
    arrive holding one quant mass per sample. The rule there, and here:

    1. among detected cells carrying at least a tenth of the row's tallest
       height, take the **most common** quantitation ion (ties broken toward the
       taller group, then the lower m/z), and average that group;
    2. keep it only if the representative's spectrum actually carries that ion
       above a tenth of its base peak — ``QuantMassExists``;
    3. otherwise fall back to the representative's own ion, then to the base
       peak of its spectrum.

    Why a mode rather than "the tallest peak's ion", which is what this used to
    be: the tallest cell is one sample, and the quantitation ion is a per-sample
    ``argmax`` over near-ties. Letting the single loudest sample name the ion
    lets one sample's coin-flip decide what the whole row is measured on. A vote
    across the detected cells is stable under exactly the near-ties that make
    the argmax unstable.
    """
    if not detected:
        return None
    heights = [float(peak.ms1_height or 0.0) for peak in detected]
    ceiling = max(heights) if heights else 0.0
    floor = ceiling * _QUANT_MASS_MIN_RELATIVE

    voters = [
        (peak, quant_ion(peak, config))
        for peak, height in zip(detected, heights)
        if height >= floor
    ]
    voters = [(peak, ion) for peak, ion in voters if ion is not None]
    if not voters:
        voters = [(peak, quant_ion(peak, config)) for peak in detected]
        voters = [(peak, ion) for peak, ion in voters if ion is not None]
    if not voters:
        return None

    template = voters[0][1]
    groups: dict[float, list[tuple[CandidateFeature, QuantIon]]] = {}
    for peak, ion in voters:
        # Bin at the width the ion was integrated over: two cells whose ions are
        # closer than that were never measured apart.
        placed = False
        for centre in groups:
            if abs(centre - ion.mz) <= template.tolerance:
                groups[centre].append((peak, ion))
                placed = True
                break
        if not placed:
            groups[ion.mz] = [(peak, ion)]

    def group_rank(item):
        centre, members = item
        tallest = max(float(peak.ms1_height or 0.0) for peak, _ion in members)
        return (-len(members), -tallest, centre)

    _centre, winner = min(groups.items(), key=group_rank)
    candidate = float(np.mean([ion.mz for _peak, ion in winner]))
    chosen = replace(template, mz=candidate,
                     channel=winner[0][1].channel)

    if representative is None:
        return chosen
    own = quant_ion(representative, config)
    if _ion_present(representative, candidate, config):
        return chosen
    if own is not None and _ion_present(representative, own.mz, config):
        return own
    base = _base_peak_mz(representative)
    if base is not None:
        return replace(chosen, mz=base)
    return own or chosen


def choose_quant_channel(
    detected: list[CandidateFeature],
    representative: Optional[CandidateFeature],
    config: GapFillConfig,
) -> Optional[QuantIon]:
    """The one channel a whole row is quantified on, route-agnostically.

    A row may now hold both an exact-precursor observation and a fragment-only
    one, and they are measured on chromatograms that are not comparable: an MS1
    window sum at +/-0.01 Da against a product-ion slice sum at +/-0.1 Da. One
    row, one channel, therefore, and this picks it.

    **Any exact precursor wins.** An MS1 precursor is a quantity that compares
    across rows, across runs and against MS-DIAL; a fragment's response does not
    — it depends on how the compound happened to fragment in that window. So
    when even one sample saw the precursor, that is what the row is measured on
    and the fragment-only samples are re-integrated onto it.

    The vote is then restricted to the cells of the winning kind: letting a
    fragment-only cell vote on an MS1 channel would average a fragment m/z into
    a precursor m/z. Within a kind this is
    :func:`consensus_quant_ion` unchanged, so on a single-route row — every row
    the route-partitioned joiner can build — the two functions agree exactly.
    """
    if not detected:
        return None
    ms1_cells = [peak for peak in detected if peak.quant_route == MS1]
    if ms1_cells:
        # ``representative`` is only used for the spectral presence check, which
        # is a no-op on MS1 (see :func:`_ion_present`); passing it here would
        # only risk falling back to a *fragment* when the representative happens
        # to be a fragment-only cell.
        return consensus_quant_ion(ms1_cells, None, config)
    product_rep = (
        representative
        if representative is not None and representative.quant_route == PRODUCT
        else None
    )
    return consensus_quant_ion(detected, product_rep, config)


def _ion_present(
    feature: CandidateFeature, mz: float, config: GapFillConfig,
) -> bool:
    """``QuantMassExists``: the spectrum carries this ion above 10% of its base.

    Deliberately a no-op on the :data:`MS1` route, and this is the reason rather
    than an unfinished branch. MS-DIAL's ``QuantMassExists`` is a GC-MS check,
    where the quantitation mass is by construction one of the spectrum's own
    ions, so "is it in the spectrum" is a real question about a real candidate.
    An ASFAM MS1 quantitation ion is not a candidate drawn from the MS2 spectrum
    at all — it is the ROI centroid, or the base peak of the isolation window at
    the apex cycle — and the MS2 spectrum is of the *fragments*, where a
    precursor ion has no reason to appear. Asking the question there would
    reject every correct MS1 ion.

    The check that does apply to an MS1 ion is whether its chromatogram carries
    signal, and that is what gap filling itself measures: a channel with nothing
    on it comes back ``no_signal``.
    """
    if feature.quant_route != PRODUCT:
        return True
    mz_array = getattr(feature, "ms2_mz", None)
    intensity = getattr(feature, "ms2_intensity", None)
    if mz_array is None or intensity is None or len(mz_array) == 0:
        return False
    ceiling = float(np.max(intensity))
    if ceiling <= 0:
        return False
    near = np.abs(np.asarray(mz_array, dtype=np.float64) - mz) <= config.product_mz_tolerance
    if not np.any(near):
        return False
    return bool(np.max(np.asarray(intensity)[near]) > ceiling * _QUANT_MASS_MIN_RELATIVE)


def _base_peak_mz(feature: CandidateFeature) -> Optional[float]:
    mz_array = getattr(feature, "ms2_mz", None)
    intensity = getattr(feature, "ms2_intensity", None)
    if mz_array is None or intensity is None or len(mz_array) == 0:
        return None
    return float(np.asarray(mz_array)[int(np.argmax(np.asarray(intensity)))])


def needs_requantitation(
    peak: CandidateFeature, target: GapFillTarget, config: GapFillConfig,
) -> bool:
    """Is this *detected* peak's height measured on a different ion than the row?

    MS-DIAL's ``GcmsGapFiller.NeedsGapFill`` is exactly this — ``peak.Mass !=
    spot.QuantMass`` — and it applies to naturally detected cells, not only to
    holes (``GcmsGapFiller.cs:33-36, 52-63``). GC-MS needs it because a spot's
    peaks are matched on spectrum and RT, so each sample brought its own quant
    mass; ASFAM's PRODUCT route needs it for the same reason, and its MS1 route
    needs it whenever the tallest centroid in the isolation window was a
    different ion in this sample.

    Without it a row can hold sample A's height off fragment 150 beside sample
    B's off fragment 305. Those are two different measurements in one row, and
    the row's CV and fold change mean nothing.
    """
    if peak.gap_fill_status != DETECTED:
        return False
    own = quant_ion(peak, config)
    if own is None:
        return False
    if own.kind != target.quant.kind:
        return True
    return abs(own.mz - target.quant.mz) > target.quant.tolerance


def requantify_peak(
    peak: CandidateFeature, target: GapFillTarget, result: GapFillResult,
) -> CandidateFeature:
    """Re-read a detected peak's quantitation off the row's consensus channel.

    Only the numbers move. The peak stays ``detected`` — it *was* detected, the
    picker simply reported it on a different ion — so it still counts toward
    ``n_detected``, may still represent the spot, and still carries its MS2 and
    its annotation. MS-DIAL keeps the same distinction: ``GapFillCore`` only
    demotes ``MasterPeakID`` to ``-2`` ``if (result.MasterPeakID < 0)``, so a
    re-quantified natural peak keeps its ``PeakID >= 0``.

    When the row's channel is a different *kind* from the one this peak was
    measured on, ``signal_type`` moves with the numbers. It has to: everything
    downstream — ``quant_route``, ``align_mz``, :func:`quant_ion`,
    :func:`needs_requantitation` — reads the route off it, and a cell whose
    height came from an MS1 window while its route still says PRODUCT is a cell
    that reports a fragment m/z for a number measured on a precursor. What the
    move costs (that this sample only ever saw the compound through its
    fragments) is not lost: the caller records the original ion and reading as
    the row's ``alternate_channel_*``.

    A re-integration that finds nothing (``no_signal``) is returned unchanged.
    Whether that is acceptable is the caller's decision and it differs by case:
    within one kind the old height is off by a hair of m/z, across kinds it is a
    different measurement entirely.
    """
    if result.status == NO_SIGNAL or result.height <= 0:
        return peak
    quant_mz = float(target.quant.mz)
    updated = replace(
        peak,
        rt_apex=result.rt_apex,
        rt_left=result.rt_left,
        rt_right=result.rt_right,
        ms1_height=result.height,
        ms1_area=result.area,
        ms1_sn=result.sn_ratio,
    )
    if target.quant.kind == PRODUCT:
        updated.ms2_rep_ion_mz = quant_mz
        updated.signal_type = "ms2_only"
    else:
        updated.ms1_quant_mz = quant_mz
        updated.signal_type = "ms1_detected"
    return updated


def make_filled_peak(
    template: CandidateFeature,
    sample_id: str,
    target: GapFillTarget,
    result: GapFillResult,
) -> CandidateFeature:
    """A peak object for a sample that had none, carrying only quantitation.

    Identity fields are copied from the spot's representative so downstream code
    reading e.g. ``segment_name`` sees something sane, but the MS2 spectrum and
    the annotation are dropped: a filled peak must never speak for a spot. The
    empty ``annotation_matches`` and ``n_fragments = 0`` make that true even if
    a caller forgets to gate on :data:`DETECTED`.
    """
    empty = np.empty(0, dtype=np.float64)
    return replace(
        template,
        feature_id=f"{template.feature_id}@gap:{sample_id}",
        ms2_mz=empty,
        ms2_intensity=empty,
        ms2_sn=None,
        ms2_gaussian=None,
        n_fragments=0,
        rt_apex=result.rt_apex,
        rt_left=result.rt_left,
        rt_right=result.rt_right,
        ms1_height=result.height,
        ms1_area=result.area,
        ms1_sn=result.sn_ratio,
        ms1_gaussian=None,
        ms1_isotopes=None,
        annotation_matches=[],
        selected_annotation_idx=0,
        matchms_score=None,
        matchms_name=None,
        gap_fill_status=result.status,
    )


def fill_spot(
    spot,
    sample_ids: list[str],
    provider: ChromatogramProvider,
    config: GapFillConfig,
) -> dict[str, GapFillResult]:
    """Fill every sample of ``spot`` that has no detected peak, in place.

    Returns the per-sample results for the samples that were filled. A spot
    whose target cannot be derived (no quantitation ion) is left with holes —
    the caller decides whether that is worth logging.
    """
    detected = spot.detected_peaks
    target = gap_fill_target(detected, config)
    if target is None:
        return {}

    template = spot.representative
    results: dict[str, GapFillResult] = {}
    for sample_id in sample_ids:
        if sample_id in spot.peaks:
            continue
        chromatogram = provider(sample_id, target)
        if chromatogram is None:
            result = GapFillResult(status=NO_SIGNAL)
        else:
            result = fill_from_chromatogram(*chromatogram, target, config)
        spot.peaks[sample_id] = make_filled_peak(template, sample_id, target, result)
        results[sample_id] = result
    return results
