"""Chromatogram smoothing functions."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


def smooth_eic(
    intensity: np.ndarray,
    method: str = "savgol",
    window_length: int = 7,
    polyorder: int = 3,
    sigma: float = 1.0,
) -> np.ndarray:
    """Smooth an EIC intensity trace.

    Parameters
    ----------
    intensity : array
        Raw intensity values.
    method : str
        "savgol", "gaussian", "moving_average", or "none".
    window_length : int
        Window length for Savitzky-Golay (must be odd).
    polyorder : int
        Polynomial order for Savitzky-Golay.
    sigma : float
        Standard deviation for Gaussian smoothing.

    Returns
    -------
    np.ndarray
        Smoothed intensity (same length), non-negative.
    """
    if len(intensity) < 3:
        return intensity.copy()

    if method == "savgol":
        wl = min(window_length, len(intensity))
        if wl % 2 == 0:
            wl -= 1
        if wl < polyorder + 2:
            return intensity.copy()
        result = savgol_filter(intensity, wl, polyorder)
    elif method == "gaussian":
        result = gaussian_filter1d(intensity, sigma)
    elif method == "moving_average":
        kernel_size = min(window_length, len(intensity))
        if kernel_size % 2 == 0:
            kernel_size -= 1
        kernel = np.ones(kernel_size) / kernel_size
        result = np.convolve(intensity, kernel, mode="same")
    elif method == "none":
        return intensity.copy()
    else:
        raise ValueError(f"Unknown smoothing method: {method}")

    return np.maximum(result, 0.0)
