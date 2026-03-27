"""Peak detection in EIC traces."""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import find_peaks

from asfam.core.smoothing import smooth_eic
from asfam.models import DetectedPeak


def detect_peaks(
    rt_array: np.ndarray,
    intensity_array: np.ndarray,
    precursor_mz_nominal: int = 0,
    product_mz: float = 0.0,
    height_threshold: float = 50.0,
    sn_threshold: float = 3.0,
    width_min: int = 3,
    prominence: float = 30.0,
    smoothing_method: str = "savgol",
    smoothing_window: int = 7,
    smoothing_polyorder: int = 3,
    compute_gaussian: bool = False,
    gaussian_threshold: float = 0.0,
) -> list[DetectedPeak]:
    """Detect chromatographic peaks in an EIC trace.

    Parameters
    ----------
    rt_array, intensity_array : arrays of same length
    precursor_mz_nominal : int, for labeling peaks
    product_mz : float, for labeling peaks
    height_threshold : minimum peak apex intensity
    sn_threshold : minimum signal-to-noise ratio
    width_min : minimum peak width in scans
    prominence : minimum peak prominence
    smoothing_method, smoothing_window, smoothing_polyorder : smoothing params
    compute_gaussian : whether to compute gaussian similarity
    gaussian_threshold : minimum gaussian similarity (0 = no filter)

    Returns
    -------
    List of DetectedPeak objects, sorted by rt_apex.
    """
    if len(intensity_array) < width_min + 2:
        return []

    smoothed = smooth_eic(
        intensity_array, smoothing_method, smoothing_window, smoothing_polyorder
    )

    peak_indices, properties = find_peaks(
        smoothed,
        height=height_threshold,
        prominence=prominence,
        width=width_min,
        rel_height=0.5,
    )

    peaks = []
    for idx in peak_indices:
        apex_height = float(intensity_array[idx])
        left_idx = _find_boundary_left(smoothed, idx)
        right_idx = _find_boundary_right(smoothed, idx)

        # Validate against raw (unsmoothed) data:
        # The raw EIC in peak range must have consecutive nonzero points >= width_min.
        # This filters 3-point noise spikes that smoothing broadens into fake peaks.
        raw_segment = intensity_array[left_idx:right_idx + 1]
        if _max_consecutive_nonzero(raw_segment) < width_min:
            continue

        # Reject peaks with too many zeros within peak boundaries.
        # Real chromatographic peaks have continuous signal; noise spikes have
        # scattered nonzero points on a zero background.
        n_pts = len(raw_segment)
        n_zeros = int(np.sum(raw_segment == 0))
        if n_pts > 0 and n_zeros / n_pts > 0.25:
            continue

        area = float(np.trapz(intensity_array[left_idx:right_idx + 1],
                               rt_array[left_idx:right_idx + 1]))
        noise = _estimate_noise(intensity_array, left_idx, right_idx)
        sn = apex_height / max(noise, 1.0)

        if sn < sn_threshold:
            continue

        gauss_sim = 0.0
        if compute_gaussian or gaussian_threshold > 0:
            gauss_sim = _gaussian_similarity(
                rt_array, intensity_array, idx, left_idx, right_idx
            )
            if gaussian_threshold > 0 and gauss_sim < gaussian_threshold:
                continue

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
# ---------------------------------------------------------------------------

def _find_boundary_left(smoothed: np.ndarray, apex_idx: int) -> int:
    """Walk left from apex to find peak left boundary."""
    threshold = smoothed[apex_idx] * 0.05
    i = apex_idx
    while i > 0:
        if smoothed[i] <= threshold:
            break
        if smoothed[i - 1] > smoothed[i]:  # valley
            break
        i -= 1
    return i


def _find_boundary_right(smoothed: np.ndarray, apex_idx: int) -> int:
    """Walk right from apex to find peak right boundary."""
    threshold = smoothed[apex_idx] * 0.05
    n = len(smoothed)
    i = apex_idx
    while i < n - 1:
        if smoothed[i] <= threshold:
            break
        if smoothed[i + 1] > smoothed[i]:  # valley
            break
        i += 1
    return i


def _estimate_noise(
    intensity: np.ndarray, left_idx: int, right_idx: int, flank: int = 20
) -> float:
    """Estimate noise from flanking regions of a peak."""
    n = len(intensity)
    left_flank_start = max(0, left_idx - flank)
    right_flank_end = min(n, right_idx + flank)

    flank_values = np.concatenate([
        intensity[left_flank_start:left_idx],
        intensity[right_idx + 1:right_flank_end],
    ])
    if len(flank_values) == 0:
        return 1.0
    return float(np.median(np.abs(flank_values))) + 1.0


def _gaussian_similarity(
    rt: np.ndarray, intensity: np.ndarray,
    apex_idx: int, left_idx: int, right_idx: int,
) -> float:
    """Compute Gaussian shape similarity using analytical moment-based fit.

    Much faster than curve_fit: uses intensity-weighted moments to estimate
    Gaussian parameters, then computes Pearson r between data and fit.
    """
    segment_rt = rt[left_idx:right_idx + 1]
    segment_int = intensity[left_idx:right_idx + 1]

    if len(segment_rt) < 4:
        return 0.0

    # Ensure non-negative for weighting
    weights = np.maximum(segment_int, 0)
    total_w = np.sum(weights)
    if total_w < 1e-10:
        return 0.0

    # Analytical Gaussian fit via intensity-weighted moments
    mu = np.sum(segment_rt * weights) / total_w
    variance = np.sum(weights * (segment_rt - mu) ** 2) / total_w
    sigma = max(np.sqrt(variance), 1e-10)
    amp = float(np.max(segment_int))

    # Generate fitted Gaussian
    fitted = amp * np.exp(-0.5 * ((segment_rt - mu) / sigma) ** 2)

    # Pearson correlation
    std_int = np.std(segment_int)
    std_fit = np.std(fitted)
    if std_int < 1e-10 or std_fit < 1e-10:
        return 0.0

    n = len(segment_int)
    mean_int = np.mean(segment_int)
    mean_fit = np.mean(fitted)
    r = float(np.sum((segment_int - mean_int) * (fitted - mean_fit)) / (n * std_int * std_fit))
    return max(0.0, min(r, 1.0))
