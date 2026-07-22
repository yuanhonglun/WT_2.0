"""Chromatographic baseline + noise estimation, MS-DIAL-style.

This is a direct port of MS-DIAL's chromatogram preprocessing used by
both the LC-MS (DDA / DIA) and GC-MS feature finders. The C# sources
are:

- ``Common/CommonStandard/Components/Chromatogram.cs``
  (``GetProperty`` lines 693–706, ``PeakHeightFromBounds`` lines 732–737)
- ``Common/CommonStandard/Algorithm/PeakPick/PeakPick.cs``
  (noise estimation lines 405–437)

Algorithm summary
-----------------

1. Smooth the raw EIC twice with a short-window Linear Weighted Moving
   Average (LWMA) to suppress shot noise without distorting peak shape.
2. Smooth the raw EIC once with a wide-window LWMA — this approximates
   the slowly varying baseline / chemical noise floor.
3. Subtract: ``corrected = signal - baseline``.
4. Estimate a single noise level by chunking ``corrected`` into bins of
   ``noise_bin_size`` points, taking the (max - min) amplitude in each
   bin, then multiplying ``median(amplitudes)`` by ``noise_factor``.

Downstream peak picking then operates on ``corrected`` (so the
prominence checks compare apex height above the local baseline) and
filters candidates through three thresholds — see
``metabo_core.algorithms.peak_detection``.

The same algorithm and the same defaults are used for LC-MS (DDA, DIA)
and GC-MS in MS-DIAL — there is no workflow-specific variant — so this
module lives in ``metabo_core`` and is shared by every app.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Linear Weighted Moving Average (MS-DIAL Smoothing.LinearWeightedMovingAverage)
# ---------------------------------------------------------------------------


def lwma_smooth(intensity: np.ndarray, window: int) -> np.ndarray:
    """Linear weighted moving average over ``intensity``.

    At interior position ``i`` the weights run ``1..w+1..1`` across
    ``[i-w, i+w]`` normalised by ``(w+1)**2`` -- identical to MS-DIAL's
    ``Smoothing.LinearWeightedMovingAverage``. At the array boundaries,
    however, this implementation simply DROPS the out-of-range taps and
    shrinks BOTH the window-sum and the normaliser, which DIFFERS from
    MS-DIAL: MS-DIAL's imos method keeps the full ``(w+1)**2`` normaliser at
    every point and adds triangular boundary-correction terms (e.g. for
    ``w=1`` MS-DIAL gives ``out[0] = (3*x[0]+x[1])/4`` while this returns
    ``(2*x[0]+x[1])/3``). The two agree only on the interior ``[w : n-w]``.

    This boundary behaviour is intentional and is relied upon by the builtin
    detector; the faithful MS-DIAL variant (used by the MS-DIAL engine port)
    lives in ``metabo_core.algorithms.msdial_peak_spotting._lwma_msdial``.

    Parameters
    ----------
    intensity : 1-D array of nonneg floats.
    window : half-width ``w`` (the full window length is ``2*w + 1``).
        ``window=0`` returns a copy of ``intensity``.

    Returns
    -------
    np.ndarray
        Smoothed array, same length as ``intensity``.
    """
    if window <= 0:
        return intensity.astype(np.float64, copy=True)

    x = np.asarray(intensity, dtype=np.float64)
    n = x.size
    if n == 0:
        return x.copy()

    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        acc = 0.0
        weight_sum = 0.0
        for k in range(-window, window + 1):
            j = i + k
            if j < 0 or j >= n:
                continue
            w = window + 1 - abs(k)  # weights 1..w+1..1
            acc += x[j] * w
            weight_sum += w
        out[i] = acc / weight_sum if weight_sum > 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# Baseline + noise estimation (MS-DIAL Chromatogram.GetProperty + PeakPick)
# ---------------------------------------------------------------------------


@dataclass
class ChromatogramBaseline:
    """Container returned by :func:`estimate_baseline_and_noise`.

    Attributes
    ----------
    signal : np.ndarray
        Lightly smoothed trace (preserves peak shape).
    baseline : np.ndarray
        Broadly smoothed trace (follows the drift).
    corrected : np.ndarray
        ``signal - baseline``, the trace peak picking should operate on.
        Clipped at zero — MS-DIAL keeps the negative excursions, but for
        downstream prominence checks negative values would distort
        ``apex - max(left, right)``, so we floor at 0.
    noise_level : float
        Global noise floor: ``median(bin_amplitudes) * noise_factor``.
        Below-this candidate peaks are rejected by gate A in
        :func:`metabo_core.algorithms.peak_detection.detect_peaks`.
    amplitude_noise : float
        Same as ``noise_level / noise_factor`` — the raw median amplitude
        used by gate C (``sn_fold * amplitude_noise``).
    """

    signal: np.ndarray
    baseline: np.ndarray
    corrected: np.ndarray
    noise_level: float
    amplitude_noise: float


def estimate_baseline_and_noise(
    intensity: np.ndarray,
    smooth_window: int = 1,
    baseline_window: int = 20,
    noise_bin_size: int = 50,
    noise_factor: float = 3.0,
    min_noise_windows: int = 10,
) -> ChromatogramBaseline:
    """Compute baseline-corrected trace and global noise level.

    Parameters
    ----------
    intensity : np.ndarray
        Raw 1-D EIC intensity, evenly spaced in scan index (RT is
        irrelevant here — only the index ordering matters).
    smooth_window : int, default 1
        Half-width of the LWMA used twice for signal preservation.
        MS-DIAL hardcodes this at 1.
    baseline_window : int, default 20
        Half-width of the broad LWMA used to estimate the baseline.
        MS-DIAL hardcodes this at 20; in narrow / fast scans this can
        be lowered (effective window ``2*w+1``).
    noise_bin_size : int, default 50
        Number of consecutive scans per bin used for the binned
        amplitude noise estimate. MS-DIAL default.
    noise_factor : float, default 3.0
        Multiplier applied to ``median(bin_amplitudes)`` to produce the
        gate-A noise threshold. MS-DIAL default.
    min_noise_windows : int, default 10
        Minimum number of bins required for the binned median to be
        meaningful. Below this the noise level falls back to a small
        positive constant (avoids divide-by-zero in S/N gates on very
        short traces).

    Returns
    -------
    ChromatogramBaseline
    """
    x = np.asarray(intensity, dtype=np.float64)
    if x.size == 0:
        zero = np.zeros(0, dtype=np.float64)
        return ChromatogramBaseline(
            signal=zero, baseline=zero, corrected=zero,
            noise_level=1.0, amplitude_noise=1.0,
        )

    # 1+2. Two-pass narrow LWMA = signal; wide LWMA = baseline.
    signal_once = lwma_smooth(x, smooth_window)
    signal = lwma_smooth(signal_once, smooth_window)
    baseline = lwma_smooth(x, baseline_window)

    # 3. Baseline-correct, floor at zero.
    corrected = np.maximum(signal - baseline, 0.0)

    # 4. Binned amplitude noise: for each bin of ``noise_bin_size``
    # consecutive points, take max - min; the median of these amplitudes
    # is the noise estimate. MS-DIAL multiplies by NoiseFactor (default 3)
    # to get the gate-A threshold.
    n_bins = corrected.size // noise_bin_size
    if n_bins >= min_noise_windows:
        trimmed = corrected[: n_bins * noise_bin_size].reshape(
            n_bins, noise_bin_size,
        )
        amplitudes = trimmed.max(axis=1) - trimmed.min(axis=1)
        amplitude_noise = float(np.median(amplitudes))
    else:
        # Trace too short for a stable binned-median noise estimate.
        # MS-DIAL's behaviour here is to disable the S/N gate and rely
        # on ``MinimumAmplitude`` alone; we mirror that by collapsing
        # the noise floor to 1.0 (a no-op for any peak that already
        # clears the absolute height gate).
        amplitude_noise = 1.0

    # Floor at 1.0 to prevent the gates from collapsing on zero traces.
    amplitude_noise = max(amplitude_noise, 1.0)
    noise_level = amplitude_noise * noise_factor

    return ChromatogramBaseline(
        signal=signal,
        baseline=baseline,
        corrected=corrected,
        noise_level=noise_level,
        amplitude_noise=amplitude_noise,
    )


# ---------------------------------------------------------------------------
# Prominence gate (MS-DIAL Chromatogram.PeakHeightFromBounds + IsNoise)
# ---------------------------------------------------------------------------


def peak_prominences(
    trace: np.ndarray, apex: int, left: int, right: int,
) -> tuple[float, float]:
    """Return (min_prominence, max_prominence) for one peak.

    Mirrors MS-DIAL ``Chromatogram.PeakHeightFromBounds``:

    - ``min_prominence = apex_height - max(left_edge, right_edge)``
      (the strict gate — must clear the *higher* shoulder)
    - ``max_prominence = apex_height - min(left_edge, right_edge)``
      (the lenient gate — clears the *lower* shoulder)

    All three indices refer to positions in ``trace`` (typically the
    baseline-corrected signal). Negative results are floored at 0.
    """
    if not (0 <= left <= apex <= right < trace.size):
        return 0.0, 0.0
    top = float(trace[apex])
    le = float(trace[left])
    re = float(trace[right])
    min_prom = top - max(le, re)
    max_prom = top - min(le, re)
    return max(min_prom, 0.0), max(max_prom, 0.0)


def passes_prominence_gates(
    min_prominence: float,
    max_prominence: float,
    *,
    noise_level: float,
    amplitude_noise: float,
    min_amplitude: float,
    sn_fold: float = 4.0,
) -> bool:
    """Apply MS-DIAL's three-gate filter to one peak's prominences.

    All three gates must pass:

    - A. ``max_prominence >= noise_level``
         (peak must clear the global noise floor)
    - B. ``min_prominence >= min_amplitude``
         (user-defined absolute height floor — the only knob users
         normally need to touch)
    - C. ``min_prominence >= amplitude_noise * sn_fold``
         (S/N gate)

    Returns
    -------
    bool
    """
    if max_prominence < noise_level:
        return False
    if min_prominence < min_amplitude:
        return False
    if min_prominence < amplitude_noise * sn_fold:
        return False
    return True
