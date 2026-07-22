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

    ASFAM stores S/N rather than the noise itself, so invert it. A peak with no
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
