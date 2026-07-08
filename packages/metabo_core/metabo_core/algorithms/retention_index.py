"""Kovats retention index conversion.

Linear interpolation between flanking calibration points (RT, RI) with
linear extrapolation outside the calibrated range. Two pure functions:

  - ``kovats_rt_to_ri(rt, points)``   measured RT → RI
  - ``kovats_ri_to_rt(ri, points)``   library RI → expected RT

``points`` is a sequence of ``(rt, ri)`` pairs — typically alkane
standards with ``ri = 100 * n_carbon`` (C8 → 800, C9 → 900, …).

Robustness:
  - Requires ≥2 points (linear interp/extrap needs a slope).
  - Duplicates allowed; sorted internally.
  - NaN / non-finite RT or RI values are dropped before sort.
  - The companion :func:`kovats_warn_too_few` returns a warning string
    callers can show in their UI when calibration coverage is sparse.

The legacy GC-MS library builder (``gcms.tools.library_builder``)
previously had its own private copy of this logic — that copy is now
a thin wrapper around the functions in this module.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


PointLike = tuple[float, float]


def _normalize_points(points: Sequence[PointLike]) -> tuple[np.ndarray, np.ndarray]:
    """Validate, drop NaNs, sort by RT. Returns (rt_arr, ri_arr)."""
    if points is None:
        raise ValueError("kovats: points must not be None")
    cleaned: list[PointLike] = []
    for p in points:
        try:
            rt, ri = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not (math.isfinite(rt) and math.isfinite(ri)):
            continue
        cleaned.append((rt, ri))
    if len(cleaned) < 2:
        raise ValueError(
            f"kovats: at least 2 (RT, RI) calibration points required, "
            f"got {len(cleaned)}"
        )
    cleaned.sort(key=lambda p: p[0])
    rt_arr = np.array([p[0] for p in cleaned], dtype=np.float64)
    ri_arr = np.array([p[1] for p in cleaned], dtype=np.float64)
    return rt_arr, ri_arr


def kovats_rt_to_ri(rt: float, points: Sequence[PointLike]) -> float:
    """Convert a measured RT (min) to a Kovats RI given calibration points.

    Linear interpolation between the flanking pair within the calibrated
    range, linear extrapolation outside. Returns ``float`` so callers
    can round / clamp as they see fit.
    """
    rt_arr, ri_arr = _normalize_points(points)
    rt = float(rt)
    if not math.isfinite(rt):
        return float("nan")

    if rt <= rt_arr[0]:
        rt_a, rt_b = rt_arr[0], rt_arr[1]
        ri_a, ri_b = ri_arr[0], ri_arr[1]
    elif rt >= rt_arr[-1]:
        rt_a, rt_b = rt_arr[-2], rt_arr[-1]
        ri_a, ri_b = ri_arr[-2], ri_arr[-1]
    else:
        idx = int(np.searchsorted(rt_arr, rt, side="left"))
        if idx == 0:
            rt_a, rt_b = rt_arr[0], rt_arr[1]
            ri_a, ri_b = ri_arr[0], ri_arr[1]
        else:
            rt_a, rt_b = rt_arr[idx - 1], rt_arr[idx]
            ri_a, ri_b = ri_arr[idx - 1], ri_arr[idx]

    if rt_b == rt_a:
        return float(ri_a)
    return float(ri_a + (ri_b - ri_a) * (rt - rt_a) / (rt_b - rt_a))


def kovats_ri_to_rt(ri: float, points: Sequence[PointLike]) -> float:
    """Inverse of :func:`kovats_rt_to_ri` — RI → RT."""
    rt_arr, ri_arr = _normalize_points(points)
    ri = float(ri)
    if not math.isfinite(ri):
        return float("nan")

    # Search by RI (sort cleaned points by RI for searching).
    order = np.argsort(ri_arr)
    rt_sorted = rt_arr[order]
    ri_sorted = ri_arr[order]

    if ri <= ri_sorted[0]:
        ri_a, ri_b = ri_sorted[0], ri_sorted[1]
        rt_a, rt_b = rt_sorted[0], rt_sorted[1]
    elif ri >= ri_sorted[-1]:
        ri_a, ri_b = ri_sorted[-2], ri_sorted[-1]
        rt_a, rt_b = rt_sorted[-2], rt_sorted[-1]
    else:
        idx = int(np.searchsorted(ri_sorted, ri, side="left"))
        if idx == 0:
            ri_a, ri_b = ri_sorted[0], ri_sorted[1]
            rt_a, rt_b = rt_sorted[0], rt_sorted[1]
        else:
            ri_a, ri_b = ri_sorted[idx - 1], ri_sorted[idx]
            rt_a, rt_b = rt_sorted[idx - 1], rt_sorted[idx]

    if ri_b == ri_a:
        return float(rt_a)
    return float(rt_a + (rt_b - rt_a) * (ri - ri_a) / (ri_b - ri_a))


def kovats_warn_too_few(points: Sequence[PointLike]) -> Optional[str]:
    """Return a warning message if the calibration coverage is sparse.

    None means "enough points, no warning".

    Heuristic: 2-5 calibration points is technically sufficient but
    extrapolation outside the calibrated RT range is unreliable. 6+
    points is silent.
    """
    n = sum(1 for p in points if _is_valid_point(p))
    if n < 2:
        return (
            f"Only {n} valid alkane standard(s); ≥2 are required to "
            f"compute RI (linear interpolation needs a slope)."
        )
    if n < 6:
        return (
            f"Only {n} alkane standards configured. Linear extrapolation "
            f"will be used outside the calibrated RT range; values may "
            f"be approximate. Add more standards if available."
        )
    return None


def _is_valid_point(p: object) -> bool:
    try:
        rt, ri = float(p[0]), float(p[1])  # type: ignore[index]
    except (TypeError, ValueError, IndexError):
        return False
    return math.isfinite(rt) and math.isfinite(ri)
