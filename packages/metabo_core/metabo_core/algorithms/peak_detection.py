"""Chromatographic peak detection.

The algorithm follows MS-DIAL's pipeline (shared across LC-MS, DIA and
GC-MS workflows there — there is no per-workflow variant):

1. ``estimate_baseline_and_noise`` produces a baseline-corrected EIC
   and a global noise level. See ``baseline.py`` for the math.
2. ``scipy.signal.find_peaks`` locates local maxima on the corrected
   trace using a minimal height and width filter.
3. Each candidate is gated by three prominence checks
   (``passes_prominence_gates``):

      a. ``max_prom = apex - min(left_edge, right_edge) >= noise_level``
      b. ``min_prom = apex - max(left_edge, right_edge) >= min_amplitude``
      c. ``min_prom >= amplitude_noise * sn_fold``

4. Optionally the candidate is checked against a Gaussian shape
   similarity gate (orthogonal to the three above; off by default).

Only one user-facing knob survives: ``min_amplitude`` — the absolute
height a peak must reach above its local baseline. Everything else is
fixed at the MS-DIAL defaults (``noise_factor=3``, ``sn_fold=4``,
``baseline_window=20``, ``smooth_window=1``); apps may tune them via
``PeakDetectionConfig`` but they are not exposed as user UI knobs.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from metabo_core.algorithms.baseline import (
    estimate_baseline_and_noise,
    passes_prominence_gates,
    peak_prominences,
)
from metabo_core.models.chromatography import DetectedPeak

# ``np.trapz`` was renamed to ``np.trapezoid`` in NumPy 2.0 and removed from the
# main namespace in later 2.x releases; ``np.trapezoid`` does not exist before
# 2.0. Bind whichever name the installed NumPy exposes so the codebase stays
# compatible across the declared ``numpy>=1.21`` range.
_trapezoid = getattr(np, "trapezoid", None)
if _trapezoid is None:  # NumPy < 2.0
    _trapezoid = np.trapz


def detect_peaks(
    rt_array: np.ndarray,
    intensity_array: np.ndarray,
    precursor_mz_nominal: int = 0,
    product_mz: float = 0.0,
    min_amplitude: float = 1000.0,
    min_data_points: int = 3,
    smooth_window: int = 1,
    baseline_window: int = 20,
    noise_bin_size: int = 50,
    noise_factor: float = 3.0,
    sn_fold: float = 4.0,
    compute_gaussian: bool = True,
    gaussian_threshold: float = 0.0,
    min_prominence_ratio: float = 0.0,
    rt_window_min: float | None = None,
    rt_window_max: float | None = None,
) -> list[DetectedPeak]:
    """Detect chromatographic peaks in an EIC trace (MS-DIAL style).

    Parameters
    ----------
    rt_array, intensity_array : np.ndarray
        Same length, ordered by RT. ``intensity_array`` is the raw EIC.
    precursor_mz_nominal, product_mz : labels passed through to the
        returned :class:`DetectedPeak` objects.
    min_amplitude : float, default 1000.0
        User-defined hard floor on peak height above local baseline
        (Gate B). The only parameter users typically tune.
    min_data_points : int, default 3
        Minimum peak width in scans.
    smooth_window, baseline_window, noise_bin_size, noise_factor :
        Forwarded to :func:`estimate_baseline_and_noise`.
    sn_fold : float, default 4.0
        Multiplier for the S/N gate (Gate C).
    compute_gaussian : bool, default True
        If True, compute Gaussian shape similarity for each peak and
        surface it on the returned object. Cheap (moment-based fit, no
        curve_fit) and lets downstream stages filter on
        ``DetectedPeak.gaussian_similarity`` without first opting in.
    gaussian_threshold : float, default 0.0
        If > 0, reject peaks whose Gaussian similarity falls below this.
    min_prominence_ratio : float, default 0.0
        If > 0, reject peaks whose ``min_prom / apex_above_baseline``
        falls below this. Catches the "tall apex sitting on a noisy
        plateau" artefact that absolute-height prominence gates miss.
    rt_window_min, rt_window_max : float | None, default None
        Optional user-defined effective-RT window. Peaks whose apex RT
        is outside ``[rt_window_min, rt_window_max]`` are rejected.
        Use ``None`` on either side to disable that bound. Intended to
        skip dead time at the start of a gradient and the post-elution
        re-equilibration tail at the end.

    Returns
    -------
    list[DetectedPeak]
        Detected peaks sorted by ``rt_apex``. Each peak's ``height`` is
        the apex height **above baseline** (the MS-DIAL convention);
        ``area`` is the baseline-corrected trace integrated over time in
        **seconds** (RT × 60) within the detected boundaries, so a real
        peak has ``area > height`` (MS-DIAL convention); ``sn_ratio`` is
        ``apex_corrected / amplitude_noise``.
    """
    n = len(intensity_array)
    if n < min_data_points + 2:
        return []

    bl = estimate_baseline_and_noise(
        intensity_array,
        smooth_window=smooth_window,
        baseline_window=baseline_window,
        noise_bin_size=noise_bin_size,
        noise_factor=noise_factor,
    )
    corrected = bl.corrected

    # Use scipy.find_peaks only as a cheap local-maxima finder: we want
    # every local maximum that clears a tiny floor, then we filter via
    # the three MS-DIAL gates. Passing the strict gates to scipy would
    # double-filter and confuse the bookkeeping.
    peak_indices, _ = find_peaks(
        corrected,
        height=1.0,
        width=min_data_points,
    )

    peaks: list[DetectedPeak] = []
    for idx in peak_indices:
        # User-defined effective RT window. Drops peaks whose apex is
        # outside the analyst's declared useful gradient region —
        # typically the dead time at the start and the re-equilibration
        # tail at the end. Both bounds are optional.
        apex_rt = float(rt_array[idx])
        if rt_window_min is not None and apex_rt < rt_window_min:
            continue
        if rt_window_max is not None and apex_rt > rt_window_max:
            continue
        left_idx = _find_boundary_left(corrected, idx)
        right_idx = _find_boundary_right(corrected, idx)

        # Width check (post-boundary): the find_peaks 'width' arg uses
        # rel_height=0.5 which can disagree with our valley-walking
        # boundary finder. Re-check explicitly.
        if right_idx - left_idx + 1 < min_data_points:
            continue

        # Continuous-signal sanity: real chromatographic peaks have
        # contiguous nonzero RAW intensity across most of their range.
        # This catches synthetic / noise-spike features that the
        # baseline subtraction lets through.
        raw_segment = intensity_array[left_idx:right_idx + 1]
        if _max_consecutive_nonzero(raw_segment) < min_data_points:
            continue
        n_pts = len(raw_segment)
        n_zeros = int(np.sum(raw_segment == 0))
        if n_pts > 0 and n_zeros / n_pts > 0.20:
            continue

        # Three-gate filter (Section 3 of the MS-DIAL port).
        min_prom, max_prom = peak_prominences(corrected, int(idx), left_idx, right_idx)
        if not passes_prominence_gates(
            min_prom, max_prom,
            noise_level=bl.noise_level,
            amplitude_noise=bl.amplitude_noise,
            min_amplitude=min_amplitude,
            sn_fold=sn_fold,
        ):
            continue

        # Optional 4th gate: prominence-to-apex ratio. Filters "tall
        # apex sitting on a noisy plateau" — where the apex height is
        # high in absolute terms but it only rises a tiny fraction above
        # its valleys, which is the characteristic shape of a baseline
        # noise local maximum.
        apex_corrected = float(corrected[idx])
        if min_prominence_ratio > 0.0 and apex_corrected > 0.0:
            if (min_prom / apex_corrected) < min_prominence_ratio:
                continue

        gauss_sim = 0.0
        if compute_gaussian or gaussian_threshold > 0:
            gauss_sim = _gaussian_similarity(
                rt_array, corrected, int(idx), left_idx, right_idx,
            )
            if gaussian_threshold > 0 and gauss_sim < gaussian_threshold:
                continue

        apex_height = apex_corrected
        # Peak area = baseline-corrected trace integrated over time in
        # SECONDS (MS-DIAL convention → area > height). Every reader hands
        # the detector RT in MINUTES (LC: metabo_core.io.mzml._scan_rt_minutes;
        # GC-MS: gcms mzml_reader, also minutes), so integrating over the raw
        # minutes axis gave peak-width ~0.1-0.4 min and area < height. The
        # ×60 makes the area axis seconds — unit-consistent across ASFAM /
        # DDA / GC-MS, and robust to scan rate.
        area = float(_trapezoid(
            corrected[left_idx:right_idx + 1],
            rt_array[left_idx:right_idx + 1] * 60.0,
        ))
        sn = apex_height / max(bl.amplitude_noise, 1.0)

        peaks.append(DetectedPeak(
            precursor_mz_nominal=precursor_mz_nominal,
            product_mz=product_mz,
            rt_apex=float(rt_array[idx]),
            rt_left=float(rt_array[left_idx]),
            rt_right=float(rt_array[right_idx]),
            apex_index=int(idx),
            left_index=int(left_idx),
            right_index=int(right_idx),
            height=apex_height,
            area=area,
            sn_ratio=sn,
            gaussian_similarity=gauss_sim,
        ))

    peaks.sort(key=lambda p: p.rt_apex)
    return peaks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _max_consecutive_nonzero(arr: np.ndarray) -> int:
    """Return the length of the longest run of consecutive nonzero values."""
    if len(arr) == 0:
        return 0
    max_run = 0
    current = 0
    for v in arr:
        if v > 0:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return max_run


def _find_boundary_left(trace: np.ndarray, apex_idx: int) -> int:
    """Walk left from apex to the nearest valley or 5%-of-apex point."""
    threshold = trace[apex_idx] * 0.05
    i = apex_idx
    while i > 0:
        if trace[i] <= threshold:
            break
        if trace[i - 1] > trace[i]:  # valley
            break
        i -= 1
    return i


def _find_boundary_right(trace: np.ndarray, apex_idx: int) -> int:
    """Walk right from apex to the nearest valley or 5%-of-apex point."""
    threshold = trace[apex_idx] * 0.05
    n = len(trace)
    i = apex_idx
    while i < n - 1:
        if trace[i] <= threshold:
            break
        if trace[i + 1] > trace[i]:  # valley
            break
        i += 1
    return i


def _gaussian_similarity(
    rt: np.ndarray, intensity: np.ndarray,
    apex_idx: int, left_idx: int, right_idx: int,
) -> float:
    """Pearson correlation between the peak segment and a moment-fitted Gaussian."""
    segment_rt = rt[left_idx:right_idx + 1]
    segment_int = intensity[left_idx:right_idx + 1]

    if len(segment_rt) < 4:
        return 0.0

    weights = np.maximum(segment_int, 0)
    total_w = np.sum(weights)
    if total_w < 1e-10:
        return 0.0

    mu = np.sum(segment_rt * weights) / total_w
    variance = np.sum(weights * (segment_rt - mu) ** 2) / total_w
    sigma = max(np.sqrt(variance), 1e-10)
    amp = float(np.max(segment_int))

    fitted = amp * np.exp(-0.5 * ((segment_rt - mu) / sigma) ** 2)

    std_int = np.std(segment_int)
    std_fit = np.std(fitted)
    if std_int < 1e-10 or std_fit < 1e-10:
        return 0.0

    n = len(segment_int)
    mean_int = np.mean(segment_int)
    mean_fit = np.mean(fitted)
    r = float(np.sum((segment_int - mean_int) * (fitted - mean_fit)) / (n * std_int * std_fit))
    return max(0.0, min(r, 1.0))
