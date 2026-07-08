"""Peak-shape scores for MS-DIAL-style model-peak quality grading.

Faithful port of the two ``PeakDetectionResult`` scores that the MSDec
engine relies on (MS-DIAL ``PeakPick.cs`` ``GetPeakDetectionResult``,
L463-615):

  - ``ShapnessValue`` — average of the steepest normalised slope on each
    flank; used to rank which product-ion model peaks are sharpest and to
    decide which ions join a model chromatogram (``>= 0.9 * max``).
  - ``IdealSlopeValue`` — fraction of monotonic (ideal) slope vs total
    slope on both flanks; used to grade a model peak High / Middle / Low.

These two scores depend solely on the intensity column and the
peak-internal index distance (NOT on RT), so they can be computed from a
1-D intensity trace plus the peak's left/top/right indices. The remaining
``GetPeakDetectionResult`` outputs (symmetry, base-peak, peak-pure,
gaussian-area) are not needed by MSDec and are intentionally omitted.

The current ``msdial_detect_peaks_in_chromatogram`` does not expose these
shape scores (see its ``_assemble_peak`` note); this module fills that gap
without touching the byte-identical peak-detection path.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np


class PeakShapeScores(NamedTuple):
    """MS-DIAL model-peak shape scores needed by MSDec."""

    shapeness: float
    ideal_slope: float


def compute_peak_shape_scores(
    intensity: np.ndarray,
    left_id: int,
    top_id: int,
    right_id: int,
) -> PeakShapeScores | None:
    """Return ``(shapeness, ideal_slope)`` for the peak ``[left_id, right_id]``.

    Mirrors ``PeakPick.cs`` L470-582. ``intensity`` is a 1-D trace; the peak
    occupies the inclusive index range ``[left_id, right_id]`` with its apex
    at ``top_id``. Indices may point into a longer array (only the slice is
    used).

    Returns ``None`` when MS-DIAL's ``GetPeakDetectionResult`` would return
    ``null``: a peak window of ``<= 3`` points, or an apex that sits below
    both edges. The apex intensity is assumed ``> 0`` (PeakDetectionVS1 only
    calls this for peaks clearing the amplitude/noise gates); a non-positive
    apex also yields ``None``.
    """
    # datapoints = intensity[left_id .. right_id] inclusive.
    seg = np.asarray(intensity, dtype=np.float64)[left_id : right_id + 1]
    count = seg.shape[0]
    peak_top = top_id - left_id

    # PeakPick.cs L472-473.
    if count <= 3:
        return None
    if seg[peak_top] - seg[0] < 0 and seg[peak_top] - seg[-1] < 0:
        return None

    apex = float(seg[peak_top])
    if apex <= 0:
        return None
    sqrt_apex = math.sqrt(apex)

    ideal_slope = 0.0
    non_ideal_slope = 0.0

    # ----- Left flank: j from peak_top-1 down to 0 (L480-500) -----
    left_shapeness = -math.inf
    for j in range(peak_top - 1, -1, -1):
        s = (apex - seg[j]) / (peak_top - j) / sqrt_apex
        if left_shapeness < s:
            left_shapeness = s
        diff = seg[j + 1] - seg[j]
        if diff >= 0:
            ideal_slope += abs(diff)
        else:
            non_ideal_slope += abs(diff)

    # ----- Right flank: j from peak_top+1 up to count-1 (L504-524) -----
    right_shapeness = -math.inf
    for j in range(peak_top + 1, count):
        s = (apex - seg[j]) / (j - peak_top) / sqrt_apex
        if right_shapeness < s:
            right_shapeness = s
        diff = seg[j - 1] - seg[j]
        if diff >= 0:
            ideal_slope += abs(diff)
        else:
            non_ideal_slope += abs(diff)

    shapeness = (left_shapeness + right_shapeness) / 2.0

    # L580-582.
    if ideal_slope > 0:
        ideal_slope = (ideal_slope - non_ideal_slope) / ideal_slope
    if ideal_slope < 0:
        ideal_slope = 0.0

    return PeakShapeScores(shapeness=float(shapeness), ideal_slope=float(ideal_slope))
