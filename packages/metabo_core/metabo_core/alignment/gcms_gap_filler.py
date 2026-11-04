"""GC-MS gap filling — a port of MS-DIAL's ``GcmsGapFiller``.

``MsdialGcMsApi/Algorithm/Alignment/GcmsGapFiller.cs`` in MSDIAL5.

Two jobs, and the second one is easy to miss::

    NeedsGapFill(spot, file) => peak.Mass != spot.QuantMass

That is true for an **empty** cell, and equally true for a **detected** cell
whose own quantitation ion differs from the row's consensus ion. The second kind
gets re-integrated on the consensus ion and **stays detected** — it really was
observed, it was just reported on a different ion. Turning it into a "filled"
value would deflate every ``n_detected`` in the table.

Why re-integrate at all: ``ms1_height`` / ``ms1_area`` are not abstract "peak
size", they are readings off *one specific ion's* chromatogram. A row holding
sample A's height on fragment 150 beside sample B's on fragment 305 has a CV and
a fold change that mean nothing.

Deliberate difference from MS-DIAL
---------------------------------
MS-DIAL extracts a fresh chromatogram, smooths it, and measures the peak against
a **chord baseline** through the two boundary points. This module instead takes
the caller's already **baseline-corrected and smoothed** trace — the very array
``detect_peaks`` ran on when it produced the detected peaks in the same row — and
integrates that.

The reason is the invariant the whole exercise serves: a filled number sits in
the same row as detected numbers and gets averaged with them. Same ion is not
enough; it has to be the same chromatogram, the same smoothing and the same
baseline, or the filled value is measured on a different scale than its
neighbours. ASFAMProcessor's detector already publishes exactly that trace, so using it
is both closer to the invariant and cheaper than re-extracting.

Peak framing below is MS-DIAL's, line for line.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

_trapezoid = getattr(np, "trapezoid", None)
if _trapezoid is None:  # NumPy < 2.0
    _trapezoid = np.trapz

# ``GetPeakRange``: half-width of the initial bracket around a real peak top
# (``GcmsGapFiller.cs:151``), and the furthest a forced insert may walk out from
# the nearest point when there is no peak top at all (``:131``).
_PEAK_MARGIN = 2
_FORCE_INSERT_RANGE = 5


@dataclass(frozen=True)
class FillResult:
    """What a gap fill found. ``found=False`` means "no signal", all zeros."""

    found: bool
    #: True when there was no peak top in range and ``force_insert`` took the
    #: point nearest the centre instead. The value is real but unvouched-for.
    forced: bool = False
    apex_index: int = -1
    left_index: int = -1
    right_index: int = -1
    rt_apex: float = 0.0
    rt_left: float = 0.0
    rt_right: float = 0.0
    #: Apex height on the caller's baseline-corrected trace.
    height: float = 0.0
    #: Baseline-corrected trace integrated over time in **seconds**, matching
    #: ``metabo_core.algorithms.peak_detection.detect_peaks``.
    area: float = 0.0


NO_SIGNAL = FillResult(found=False)


def needs_fill(cell_mz: Optional[float], quant_mz: float, tolerance: float) -> bool:
    """``GcmsGapFiller.NeedsGapFill`` (``:34-37``).

    ``None`` (an empty cell) always needs filling. A present cell needs
    re-integration when its own ion is not the row's consensus ion.

    MS-DIAL compares with ``!=``; here the comparison is within ``tolerance``,
    because a GC-MS quantitation ion is an intensity-weighted centroid and two
    samples reporting the same nominal fragment land on 43.02 and 42.98. Those
    are the same ion and re-integrating one on the other would extract the same
    chromatogram and cost a peak-framing pass for nothing.
    """
    if cell_mz is None:
        return True
    return abs(float(cell_mz) - float(quant_mz)) > tolerance


def fill_peak(
    rt_array: Sequence[float] | np.ndarray,
    corrected: Sequence[float] | np.ndarray,
    center: float,
    ax_tolerance: float,
    force_insert: bool = True,
    search_array: Optional[Sequence[float] | np.ndarray] = None,
) -> FillResult:
    """``GapFillCore`` (``GcmsGapFiller.cs:74-96``) on a corrected trace.

    ``center`` is the row's retention centre (mean of the detected cells) and
    ``ax_tolerance`` the half-width to look for a peak top in — the same
    ``RetentionTimeAlignmentTolerance`` / ``RetentionIndexAlignmentTolerance``
    the joiner matched with.

    ``rt_array`` must be ascending and the same length as ``corrected``.

    ``search_array`` lets the peak-top search run on a **different axis** than
    the integration. In RI mode ``center`` and ``ax_tolerance`` are retention
    *indices*, so the search happens on the per-scan RI axis while the area
    stays an intensity-over-seconds integral of ``rt_array`` — which is what
    MS-DIAL does too (``SetAlignmentChromPeakFeature``, ``:182-190``, integrates
    over ``ChromXs.RT`` regardless of the index type). Defaults to ``rt_array``.
    """
    x = np.asarray(rt_array, dtype=np.float64)
    y = np.asarray(corrected, dtype=np.float64)
    if x.size == 0 or x.size != y.size:
        return NO_SIGNAL
    if search_array is None:
        s = x
    else:
        s = np.asarray(search_array, dtype=np.float64)
        if s.size != x.size:
            return NO_SIGNAL

    candidates, min_id = _peak_top_candidates(s, y, center, ax_tolerance)
    apex, left, right, forced = _peak_range(
        candidates, s, y, min_id, center, force_insert,
    )
    if apex < 0 or left < 0 or right < 0:
        return NO_SIGNAL

    height = float(y[apex])
    # Seconds, matching detect_peaks — see this module's docstring.
    area = float(_trapezoid(y[left:right + 1], x[left:right + 1] * 60.0))
    return FillResult(
        found=True,
        forced=forced,
        apex_index=int(apex),
        left_index=int(left),
        right_index=int(right),
        rt_apex=float(x[apex]),
        rt_left=float(x[left]),
        rt_right=float(x[right]),
        height=height,
        area=area,
    )


def _peak_top_candidates(
    x: np.ndarray, y: np.ndarray, center: float, ax_tol: float,
) -> tuple[list[int], int]:
    """``GetPeakTopCandidates`` (``:98-124``).

    Every local maximum inside ``center +- ax_tol``, plus the index closest to
    the centre as a fallback for the forced-insert path. Indices within two
    points of either end are skipped: the shape tests read ``i-2`` and ``i+2``.
    """
    n = x.size
    candidates: list[int] = []
    min_id = -1
    min_diff = float("inf")

    start = bisect.bisect_left(x.tolist(), center - ax_tol)
    for i in range(start, n):
        if i - 2 < 0 or i + 2 >= n:
            continue
        if x[i] < center - ax_tol:
            continue
        if x[i] > center + ax_tol:
            break
        rising = y[i - 2] <= y[i - 1] <= y[i] and y[i] > y[i + 1]
        falling = y[i - 1] < y[i] and y[i] >= y[i + 1] >= y[i + 2]
        if rising or falling:
            candidates.append(i)
        diff = abs(float(x[i]) - center)
        if diff < min_diff:
            min_diff = diff
            min_id = i

    if min_id == -1:
        min_id = n // 2
    return candidates, min_id


def _peak_range(
    candidates: list[int],
    x: np.ndarray,
    y: np.ndarray,
    min_id: int,
    center: float,
    force_insert: bool,
) -> tuple[int, int, int, bool]:
    """``GetPeakRange`` (``:126-178``). Returns ``(apex, left, right, forced)``."""
    n = x.size
    if not candidates:
        if not force_insert:
            return -1, -1, -1, False
        # No peak top in range: take the point nearest the centre and walk out
        # at most _FORCE_INSERT_RANGE points while the trace keeps descending.
        apex = min_id
        left = max(apex - 1, 0)
        right = min(apex + 1, n - 1)

        limit = max(apex - _FORCE_INSERT_RANGE, 0)
        while limit < left:
            if y[left - 1] > y[left]:
                break
            left -= 1
        limit = min(apex + _FORCE_INSERT_RANGE, n - 1)
        while right < limit:
            if y[right] < y[right + 1]:
                break
            right += 1
        return apex, left, right, True

    apex = min(candidates, key=lambda i: abs(float(x[i]) - center))

    left = max(apex - _PEAK_MARGIN, 0)
    while 0 < left:
        if y[left - 1] >= y[left]:
            break
        left -= 1

    right = min(apex + _PEAK_MARGIN, n - 1)
    while right < n - 1:
        if y[right] <= y[right + 1]:
            break
        right += 1

    if not force_insert and (apex - left < 2 or right - apex < 2):
        return -1, -1, -1, False

    # A taller local maximum inside the bracket wins the apex (``:167-174``).
    for i in range(left + 1, right):
        is_top = (
            (y[i - 1] <= y[i] and y[i] > y[i + 1])
            or (y[i - 1] < y[i] and y[i] >= y[i + 1])
        )
        if is_top and y[apex] < y[i]:
            apex = i
    return apex, left, right, False
