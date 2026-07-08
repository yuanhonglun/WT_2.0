"""MS-DIAL peak-spotting engine + primitives (Track B, Tasks 1.2-1.3).

Faithful port of the numerically sensitive helpers used by MS-DIAL's
derivative-based chromatogram peak picker. The C# reference is::

    Common/CommonStandard/Components/ChromatogramGlobalProperty_temp2.cs

which defines ``ChroChroChromatogram`` (the per-EIC detector) and
``ChromatogramGlobalProperty_temp2`` (the smoothed/baseline-corrected
trace plus its derivative coefficients and slope noises). The global
binned-noise estimate lives in ``Components/Chromatogram.cs``
(``GetMinimumNoiseLevel`` / ``GetProperty``) and the median primitive in
``Mathematics/Mathematics/BasicMathematics.cs``.

This module contains the Task-1.2 private primitives:

  * :func:`_sg_derivatives`     -- 5-point Savitzky-Golay 1st/2nd derivatives
  * :func:`_slope_noises`       -- amplitude / slope / peak-top slope noises
  * :func:`_estimate_global_noise` -- binned global noise + estimated noise
  * :func:`_is_noise`           -- the 4-condition IsNoise rejection gate

plus the Task-1.3 public engine :func:`msdial_detect_peaks_in_chromatogram`
(the derivative peak walk, ``ShrinkPeakRange``, ``GetPeakDetectionResult``
assembly, and the RAW-EIC background-spike filter). It is a drop-in
replacement for ``metabo_core.algorithms.peak_detection.detect_peaks``
consumers: same :class:`~metabo_core.models.chromatography.DetectedPeak`
return type, ``area`` in seconds (x60), peaks sorted by ``rt_apex``. The
coarse->fine m/z recalc and cross-slice dedup are Task 2.2 and live elsewhere.

Numbers are ported verbatim from the C# literals; see the per-function
comments for the exact source lines. All logic is a faithful port of the
MS-DIAL reference (no intentional divergence): in particular the global-noise
estimate reproduces ``Chromatogram.GetMinimumNoiseLevel`` rather than
borrowing ``baseline.py``'s estimator, and ``baselineMedian`` / max / min are
taken from the LWMA(SmoothingLevel)-smoothed trace (``chrom``) exactly as
``Chromatogram.GetProperty`` computes them on ``this`` -- the chromatogram
``PeakSpottingCore.GetChromatogramPeakFeatures`` smooths before detection.
"""
from __future__ import annotations

import numpy as np

from metabo_core.config.msdial_peak_spotting import MsdialPeakSpottingConfig
from metabo_core.models.chromatography import DetectedPeak

# ---------------------------------------------------------------------------
# Constants ported verbatim from ChromatogramGlobalProperty_temp2.cs
# ---------------------------------------------------------------------------

# Savitzky-Golay coefficients, ChromatogramGlobalProperty_temp2.cs lines 287-288:
#   FIRST_DIFF_COEFF  = { -0.2, -0.1, 0, 0.1, 0.2 }                (= {-2,-1,0,1,2}/10)
#   SECOND_DIFF_COEFF = { 0.14285714, -0.07142857, -0.1428571,
#                         -0.07142857, 0.14285714 }               (= {1,-0.5,-1,-0.5,1}/7)
# NB: the 2nd-deriv coeffs are HALF the textbook SG 2nd-derivative, so they
# return half the true curvature. The scale is irrelevant downstream because
# the peak-top noise threshold is derived from these same coefficients.
_FIRST_DIFF_COEFF = np.array([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=np.float64)
_SECOND_DIFF_COEFF = np.array(
    [0.14285714, -0.07142857, -0.1428571, -0.07142857, 0.14285714], dtype=np.float64
)
# halfDatapoint = FIRST_DIFF_COEFF.Length / 2 (== 2); the first/last 2 points
# have no valid 5-point window, ChromatogramGlobalProperty_temp2.cs line 322.
_HALF = 2

# Empty-candidate fallback for the slope noises,
# ChromatogramGlobalProperty_temp2.cs lines 369-371 (the C# "0.0001").
_NOISE_FALLBACK = 1e-4
# Small-diff fraction for the slope-noise candidate window,
# ChromatogramGlobalProperty_temp2.cs line 356 (the C# "* 0.05").
_SMALL_DIFF_FRAC = 0.05


def _upper_median(values: np.ndarray) -> float:
    """Return MS-DIAL's median: the upper-middle element for even counts.

    Ports ``BasicMathematics.Median`` / ``InplaceSortMedian``
    (BasicMathematics.cs lines 122-132 / 154-161): sort ascending and return
    ``sorted[count // 2]``. For an even count this is the *upper* of the two
    central elements, NOT numpy's average-of-two-middle. Callers must pass a
    non-empty array.
    """
    s = np.sort(np.asarray(values, dtype=np.float64))
    return float(s[s.size // 2])


# ---------------------------------------------------------------------------
# 0. Linear Weighted Moving Average -- faithful MS-DIAL imos edge handling
# ---------------------------------------------------------------------------


def _lwma_msdial(intensity: np.ndarray, level: int) -> np.ndarray:
    """Linear weighted moving average, faithful to MS-DIAL's imos method.

    Verbatim port of ``Smoothing.LinearWeightedMovingAverage(ValuePeak[]
    peaklist, ValuePeak[] dest, int datasize, int smoothingLevel)`` -- the
    "imos method" in ``Common/CommonStandard/Algorithm/ChromSmoothing/
    Smoothing.cs`` lines 29-67.

    Unlike :func:`metabo_core.algorithms.baseline.lwma_smooth`, which at the
    array boundaries drops the out-of-range taps and shrinks BOTH the window-sum
    and the normaliser, MS-DIAL keeps the full ``(L+1)**2`` normalisation for
    EVERY point and instead adds two triangular boundary-correction terms. The
    two implementations are bit-identical on the interior ``[L : n-L]`` but
    differ on the first/last ``L`` points -- e.g. for ``L=1`` MS-DIAL gives
    ``dest[0] = (3*x[0] + x[1]) / 4`` whereas ``lwma_smooth`` gives
    ``(2*x[0] + x[1]) / 3``. The MS-DIAL engine must use THIS variant so the
    first/last ``baseline_window`` points (which feed the global-noise estimate)
    match the reference.

    Parameters
    ----------
    intensity : 1-D array of floats (raw or pre-smoothed chromatogram).
    level : smoothing half-width ``L``. ``level <= 0`` returns a float64 copy.

    Returns
    -------
    np.ndarray
        Smoothed trace, same length as ``intensity`` (float64).
    """
    x = np.asarray(intensity, dtype=np.float64)
    n = x.size
    if level <= 0:
        return x.copy()
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    L = int(level)
    norm = (L + 1) * (L + 1)
    size = n + 2 * L + 2

    # Difference array (Smoothing.cs lines 34-45): term1 +x[i], term2
    # -2*x[i-L-1], term3 +x[i-2L-2]. The vectorised slices below are the exact
    # equivalent of the C# per-index ``if`` ladder (hand-verified). For term3 the
    # C# omits an upper-bound check because i-2L-2 is always < n for i in [0,size).
    d = np.zeros(size, dtype=np.float64)
    d[:n] += x
    d[L + 1 : L + 1 + n] -= 2.0 * x
    d[2 * L + 2 : 2 * L + 2 + n] += x

    # Two prefix sums (Smoothing.cs lines 47-53).
    intensities = np.cumsum(np.cumsum(d))

    # Triangular boundary corrections (Smoothing.cs lines 55-61). The coefficient
    # (L-i+1)*(L-i)/2 is a triangular number -> exact integer; keep it integer so
    # it is bit-exact, matching the C# int arithmetic before the double multiply.
    m = min(L, n)
    for i in range(m):
        coeff = (L - i + 1) * (L - i) // 2
        intensities[i + L] += x[i] * coeff
    for i in range(m):
        coeff = (L - i + 1) * (L - i) // 2
        intensities[n - 1 - i + L] += x[n - 1 - i] * coeff

    # dest[i] = intensities[i + L] / normalizationValue (Smoothing.cs lines 63-65).
    return intensities[L : L + n] / norm


# ---------------------------------------------------------------------------
# 1. Savitzky-Golay first / second derivatives
# ---------------------------------------------------------------------------


def _sg_derivatives(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """5-point Savitzky-Golay 1st and 2nd derivatives of ``y``.

    Ports ``ChromatogramGlobalProperty_temp2.GenerateDifferencialCoefficients``
    (lines 312-350). For each interior index ``i`` (``2 <= i < len-2``)::

        d1[i] = sum_j FIRST_DIFF_COEFF[j]  * y[i + j - 2]   for j in 0..4
        d2[i] = sum_j SECOND_DIFF_COEFF[j] * y[i + j - 2]   for j in 0..4

    The first and last 2 entries have no valid window and are exactly 0
    (lines 325-334). Returns ``(d1, d2)`` with the same length as ``y``.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    d1 = np.zeros(n, dtype=np.float64)
    d2 = np.zeros(n, dtype=np.float64)
    if n < 2 * _HALF + 1:  # no valid 5-point window anywhere
        return d1, d2

    centers = np.arange(_HALF, n - _HALF)  # interior i in [2, n-2)
    # window[:, j] = y[i + j - halfDatapoint], matching Intensity(i + j - 2).
    window = np.stack(
        [y[centers + (j - _HALF)] for j in range(_FIRST_DIFF_COEFF.size)], axis=1
    )
    d1[centers] = window @ _FIRST_DIFF_COEFF
    d2[centers] = window @ _SECOND_DIFF_COEFF
    return d1, d2


# ---------------------------------------------------------------------------
# 2. Slope noises (amplitude / slope / peak-top)
# ---------------------------------------------------------------------------


def _slope_noises(
    y: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    small_diff_frac: float = _SMALL_DIFF_FRAC,
) -> tuple[float, float, float]:
    """Return ``(amplitude_noise, slope_noise, peaktop_noise)``.

    Ports ``GenerateDifferencialCoefficients`` (max accumulators, lines
    344-347) + ``CalculateSlopeNoises`` (lines 352-372). Each noise is the
    median of the "small" elements of one difference series -- elements with
    ``0 < |x| < small_diff_frac * max|x|`` -- with the C# upper-median; an
    empty candidate set falls back to ``0.0001`` (lines 369-371).

    Source series (loop range ``i in [2, len-2)`` for all three):

    * amplitude : candidate ``|y[i+1] - y[i]|`` (line 360), threshold from
      ``maxAmplitudeDiff = max|y[i] - y[i-1]|`` (line 346). NB the max uses the
      backward diff while the candidate uses the forward diff -- a genuine
      MS-DIAL index quirk, reproduced here.
    * slope    : candidate / max ``|d1[i]|`` (lines 344, 363).
    * peak-top : candidate / max ``|d2[i]|`` but only where ``d2[i] < 0``
      (lines 345, 365-367).
    """
    y = np.asarray(y, dtype=np.float64)
    d1 = np.asarray(d1, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)
    n = y.size
    if n < 2 * _HALF + 1:  # interior loop range [2, n-2) is empty
        return _NOISE_FALLBACK, _NOISE_FALLBACK, _NOISE_FALLBACK

    # --- amplitude noise (from raw-trace consecutive intensity diffs) -------
    # max over i in [2, n-2) of |y[i] - y[i-1]| (line 346, backward diff).
    back_diffs = np.abs(y[_HALF : n - _HALF] - y[_HALF - 1 : n - _HALF - 1])
    max_amp = float(back_diffs.max()) if back_diffs.size else 0.0
    # candidates over i in [2, n-2) of |y[i+1] - y[i]| (line 360, forward diff).
    fwd_diffs = np.abs(y[_HALF + 1 : n - _HALF + 1] - y[_HALF : n - _HALF])
    amp_thresh = max_amp * small_diff_frac
    amp_cand = fwd_diffs[(fwd_diffs > 0.0) & (fwd_diffs < amp_thresh)]
    amplitude_noise = _upper_median(amp_cand) if amp_cand.size else _NOISE_FALLBACK

    # --- slope noise (from 1st-derivative magnitudes) ----------------------
    d1_int = np.abs(d1[_HALF : n - _HALF])
    max_first = float(d1_int.max()) if d1_int.size else 0.0
    slope_thresh = max_first * small_diff_frac
    slope_cand = d1_int[(d1_int > 0.0) & (d1_int < slope_thresh)]
    slope_noise = _upper_median(slope_cand) if slope_cand.size else _NOISE_FALLBACK

    # --- peak-top noise (from negative 2nd-derivative magnitudes) ----------
    d2_int = d2[_HALF : n - _HALF]
    neg_abs = -d2_int[d2_int < 0.0]  # |d2| where d2 < 0 (all strictly > 0)
    max_second = float(neg_abs.max()) if neg_abs.size else 0.0
    peaktop_thresh = max_second * small_diff_frac
    peaktop_cand = neg_abs[(neg_abs > 0.0) & (neg_abs < peaktop_thresh)]
    peaktop_noise = (
        _upper_median(peaktop_cand) if peaktop_cand.size else _NOISE_FALLBACK
    )

    return amplitude_noise, slope_noise, peaktop_noise


# ---------------------------------------------------------------------------
# 3. Global binned noise + estimated noise
# ---------------------------------------------------------------------------


def _estimate_global_noise(
    corrected: np.ndarray,
    *,
    bin_size: int = 50,
    factor: float = 3.0,
    min_windows: int = 10,
    min_noise_level: float = 0.0,
) -> tuple[float, float]:
    """Return ``(noise, estimated_noise)`` for an already-corrected trace.

    Faithful port of ``Chromatogram.GetMinimumNoiseLevel`` (Chromatogram.cs
    lines 512-540) followed by the ``* NoiseFactor`` applied at its only call
    site in ``GetProperty`` (Chromatogram.cs line 698). The defaults mirror
    ``NoiseEstimateParameter.GlobalParameter`` -- the single parameter set the
    whole peak picker uses (NoiseEstimateParameter.cs lines 36-41:
    ``NoiseEstimateBin = 50``, ``MinimumNoiseWindowSize = 10``,
    ``MinimumNoiseLevel = 0d``, ``NoiseFactor = 3``).

    Algorithm (each step cites the C# line):

      * Walk ``corrected`` in consecutive bins of ``bin_size`` and KEEP the
        trailing partial bin -- the C# ``while``/``for`` walks ``i`` to
        ``_size`` and still pushes the final short bin; the buffer is sized
        ``ceil(_size / binSize)`` (Chromatogram.cs lines 516, 520-532).
      * Per-bin amplitude is ``max - min``, but ONLY bins with ``min < max``
        count -- flat / zero-amplitude bins are excluded (line 529).
      * ONLY those nonzero bins count toward the ``>= min_windows`` sufficiency
        check (``size`` is bumped solely for ``min < max`` bins; line 533).
      * Sufficient -> ``noise = upper_median(nonzero amplitudes) * factor``,
        where the upper median is ``BasicMathematics.InplaceSortMedian`` =
        ``sorted[size // 2]`` (line 534; BasicMathematics.cs lines 154-161).
      * Insufficient -> fall back to the ``MinimumNoiseLevel`` PARAMETER, not a
        hardcoded 0 (line 535). NB ``GlobalParameter.MinimumNoiseLevel == 0d``,
        so under the MS-DIAL defaults this fallback is 0.

    ``noise`` is the post-factor value: line 698 multiplies the WHOLE
    ``GetMinimumNoiseLevel`` return -- including the fallback branch -- by
    ``NoiseFactor``. This is the ``ChromatogramGlobalProperty_temp2.Noise``
    field consumed by ``IsNoise`` (ChromatogramGlobalProperty_temp2.cs lines
    67 / 298). ``estimated_noise`` mirrors
    ``EstimatedNoise = Math.Max(1f, Noise / noiseFactor)``
    (ChromatogramGlobalProperty_temp2.cs line 199).

    The input is ASSUMED already baseline-corrected (no re-baselining here);
    MS-DIAL computes it as ``ssChromatogram.Difference(baselineChromatogram)``
    (Chromatogram.cs line 697) before calling ``GetMinimumNoiseLevel``.
    """
    corrected = np.asarray(corrected, dtype=np.float64)
    n = corrected.size

    # Bin into chunks of bin_size, KEEPING the trailing partial bin, and collect
    # the amplitude (max - min) of every bin whose min < max (Chromatogram.cs
    # lines 520-532). Dropping the flat bins here is what also keeps them out of
    # the min_windows count below (only min < max bins increment size, line 533).
    n_full = n // bin_size
    parts: list[np.ndarray] = []
    if n_full > 0:
        full = corrected[: n_full * bin_size].reshape(n_full, bin_size)
        lo = full.min(axis=1)
        hi = full.max(axis=1)
        nonzero = lo < hi  # min < max (Chromatogram.cs line 529)
        parts.append((hi - lo)[nonzero])
    if n_full * bin_size < n:  # trailing partial bin, kept (lines 525-531)
        tail = corrected[n_full * bin_size :]
        tail_lo = float(tail.min())
        tail_hi = float(tail.max())
        if tail_lo < tail_hi:  # min < max (line 529)
            parts.append(np.array([tail_hi - tail_lo], dtype=np.float64))
    amplitudes = (
        np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
    )

    # size >= minWindowSize ? InplaceSortMedian(buffer, size) : minNoiseLevel
    # (Chromatogram.cs lines 533-535). InplaceSortMedian returns 0 for an empty
    # buffer (BasicMathematics.cs lines 156-158); guard for the degenerate
    # min_windows <= 0 case so the upper-median branch never indexes empty.
    if amplitudes.size >= min_windows:
        base = _upper_median(amplitudes) if amplitudes.size else 0.0
    else:
        base = float(min_noise_level)
    # "* NoiseFactor" is applied to BOTH branches at the call site (line 698).
    noise = base * factor

    # EstimatedNoise = Math.Max(1f, Noise / noiseFactor)
    # (ChromatogramGlobalProperty_temp2.cs line 199).
    estimated_noise = max(1.0, noise / factor)
    return noise, estimated_noise


def _chromatogram_estimated_noise(
    raw_intensity: np.ndarray, config: MsdialPeakSpottingConfig
) -> float:
    """Per-chromatogram ``EstimatedNoise``, faithful to the engine's noise chain.

    Reproduces the smoothing + baseline-subtraction + ``GetMinimumNoiseLevel``
    chain that :func:`msdial_detect_peaks_in_chromatogram` runs internally (steps
    1-2 of its pipeline) and returns ONLY the ``estimated_noise`` field
    (``EstimatedNoise = Math.Max(1, Noise / NoiseFactor)``).

    Task 2.2's coarse->fine recalc
    (``PeakSpottingCore.GetRecalculatedChromPeakFeaturesByMs1MsTolerance``,
    line 926) re-derives a peak's S/N from
    ``peakFeature.PeakShape.EstimatedNoise`` — the EstimatedNoise computed during
    the COARSE per-slice detection. The engine does not surface that value on
    :class:`~metabo_core.models.chromatography.DetectedPeak`, so the orchestrator
    calls this helper once per slice (on the SAME dense EIC it hands the engine)
    to obtain the identical constant-per-slice EstimatedNoise without modifying
    the engine. The chain here is byte-for-byte the engine's steps 1-2.

    Parameters
    ----------
    raw_intensity : 1-D float array
        The dense (0-filled) slice EIC — exactly the array passed to the engine.
    config : MsdialPeakSpottingConfig
        Pins ``smoothing_level`` / ``baseline_window`` / ``noise_*`` so the
        result matches the engine's internal estimate.

    Returns
    -------
    float
        ``estimated_noise`` (>= 1.0). For an empty trace returns ``1.0``.
    """
    raw = np.asarray(raw_intensity, dtype=np.float64)
    if raw.size == 0:
        return 1.0
    smoothing_level = int(config.smoothing_level)
    baseline_window = int(config.baseline_window)
    chrom = _lwma_msdial(raw, smoothing_level)
    ss = _lwma_msdial(_lwma_msdial(chrom, 1), 1)
    baseline = _lwma_msdial(chrom, baseline_window)
    corrected = np.maximum(ss - baseline, 0.0)
    _noise, estimated_noise = _estimate_global_noise(
        corrected,
        bin_size=int(config.noise_bin_size),
        factor=float(config.noise_factor),
        min_windows=int(config.min_noise_windows),
    )
    return float(estimated_noise)


# ---------------------------------------------------------------------------
# 4. IsNoise rejection gate
# ---------------------------------------------------------------------------


def _is_noise(
    min_prom: float,
    max_prom: float,
    noise: float,
    amplitude_noise: float,
    min_amplitude: float,
    *,
    fold: float = 4.0,
    is_high_baseline: bool,
    edge_min: float,
    baseline_median: float,
) -> bool:
    """Return True if a candidate peak should be rejected as noise.

    Ports ``ChroChroChromatogram.IsNoise``
    (ChromatogramGlobalProperty_temp2.cs lines 66-68) combined with
    ``Chromatogram.HasBoundaryBelowThreshold`` (Chromatogram.cs lines 718-720).
    The peak is noise if ANY of the four conditions holds:

      1. ``max_prom < noise``                     (max height below global noise)
      2. ``min_prom < min_amplitude``             (below absolute amplitude floor)
      3. ``min_prom < amplitude_noise * fold``    (below S/N amplitude gate)
      4. ``is_high_baseline and edge_min < baseline_median``
         (high-baseline EIC whose lower boundary dips below the baseline median)

    The caller supplies the already-computed prominences and edge values:
    ``max_prom = top - min(leftEdge, rightEdge)``,
    ``min_prom = top - max(leftEdge, rightEdge)``
    (Chromatogram.PeakHeightFromBounds, lines 732-737); and
    ``edge_min = min(I[start], I[end-1])`` (HasBoundaryBelowThreshold).
    """
    cond1 = float(max_prom) < float(noise)
    cond2 = float(min_prom) < float(min_amplitude)
    cond3 = float(min_prom) < float(amplitude_noise) * float(fold)
    cond4 = bool(is_high_baseline) and (float(edge_min) < float(baseline_median))
    return bool(cond1 or cond2 or cond3 or cond4)


# ---------------------------------------------------------------------------
# 5. Shape predicates on a trace (Chromatogram.cs)
# ---------------------------------------------------------------------------
#
# Each predicate is a verbatim port of the matching ``Chromatogram`` method
# (cited per function). The trace ``y`` is the same numpy float array that the
# C# ``Chromatogram`` wraps -- ``ss`` (SmoothedChromatogram) for the walk
# predicates and the RAW EIC for the spike filter.


def _is_peak_top(y: np.ndarray, t: int) -> bool:
    """Chromatogram.IsPeakTop (Chromatogram.cs lines 571-577).

    Strict clause ``(y[t-1] != y[t] || y[t] != y[t+1])`` preserved so a flat
    plateau is NOT a peak top.
    """
    n = y.size
    return bool(
        1 <= t < n - 1
        and y[t - 1] <= y[t]
        and y[t] >= y[t + 1]
        and (y[t - 1] != y[t] or y[t] != y[t + 1])
    )


def _is_large_peak_top(y: np.ndarray, t: int) -> bool:
    """Chromatogram.IsLargePeakTop (Chromatogram.cs lines 587-592)."""
    n = y.size
    return bool(
        t - 2 >= 0
        and t + 2 < n
        and y[t - 2] <= y[t - 1]
        and _is_peak_top(y, t)
        and y[t + 1] >= y[t + 2]
    )


def _is_broad_peak_top(y: np.ndarray, t: int) -> bool:
    """Chromatogram.IsBroadPeakTop (Chromatogram.cs lines 602-608)."""
    n = y.size
    return bool(
        1 <= t < n - 1
        and y[t - 1] <= y[t]
        and y[t] >= y[t + 1]
        and (
            (t - 2 >= 0 and y[t - 2] <= y[t - 1])
            or (t + 2 < n and y[t + 1] >= y[t + 2])
        )
    )


def _is_flat(y: np.ndarray, center: int, amplitude_noise: float) -> bool:
    """Chromatogram.IsFlat (Chromatogram.cs lines 619-623)."""
    n = y.size
    return bool(
        center - 1 >= 0
        and center + 1 < n
        and abs(y[center - 1] - y[center]) < amplitude_noise
        and abs(y[center] - y[center + 1]) < amplitude_noise
    )


def _is_bottom(y: np.ndarray, b: int) -> bool:
    """Chromatogram.IsBottom (Chromatogram.cs lines 635-639)."""
    n = y.size
    return bool(
        b - 1 >= 0
        and b + 1 < n
        and y[b - 1] >= y[b]
        and y[b] <= y[b + 1]
    )


def _is_large_bottom(y: np.ndarray, b: int) -> bool:
    """Chromatogram.IsLargeBottom (Chromatogram.cs lines 649-654)."""
    n = y.size
    return bool(
        b - 2 >= 0
        and b + 2 < n
        and y[b - 2] >= y[b - 1]
        and _is_bottom(y, b)
        and y[b + 1] <= y[b + 2]
    )


def _is_broad_bottom(y: np.ndarray, b: int) -> bool:
    """Chromatogram.IsBroadBottom (Chromatogram.cs lines 661-665)."""
    n = y.size
    return bool(
        _is_bottom(y, b)
        and (
            (b - 2 >= 0 and y[b - 2] >= y[b - 1])
            or (b + 2 < n and y[b + 1] <= y[b + 2])
        )
    )


def _is_valid_peak_top(y: np.ndarray, t: int) -> bool:
    """Chromatogram.IsValidPeakTop (Chromatogram.cs lines 873-876).

    Both neighbours strictly positive -- used by the RAW-EIC spike filter.
    """
    n = y.size
    return bool(
        t - 1 >= 0
        and t + 1 <= n - 1
        and y[t - 1] > 0
        and y[t + 1] > 0
    )


# ---------------------------------------------------------------------------
# 6. Edge walking (ChromatogramGlobalProperty_temp2.cs + Chromatogram.cs)
# ---------------------------------------------------------------------------


def _search_real_left_edge(ss: np.ndarray, i: int) -> int:
    """ChromatogramGlobalProperty_temp2.SearchRealLeftEdge (lines 375-383).

    Walk left up to 5 points on ``ss`` until the trace stops rising
    (``ss[i-j] - ss[i-j-1] <= 0``) or the boundary; otherwise ``i - 6``.
    """
    for j in range(0, 6):
        if i - j - 1 < 0 or (ss[i - j] - ss[i - j - 1]) <= 0:
            return i - j
    return i - 6


def _search_right_edge_candidate(
    ss: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    i: int,
    min_data_points: float,
    slope_fold: float,
    peaktop_noise: float,
    slope_noise: float,
    amplitude_noise: float,
) -> int:
    """ChromatogramGlobalProperty_temp2.SearchRightEdgeCandidate (lines 27-64).

    Walk right on ``ss`` (with ``d1``/``d2``) to the candidate right edge. The
    loop terminates with the C# ``!=`` test ``j + 2 != len - 1``; indices up to
    ``j + 2`` are accessed inside, which the caller's bounds guarantee are valid
    (``i`` always satisfies ``i <= len - 3``).

    Faithful operator-precedence quirk in the first peak-top check (``&&`` binds
    tighter than ``||``)::

        (!foundTop && d1[j-1]>0 && d1[j]<0)
          || (d1[j-1]>0 && d1[j+1]<0 && d2[j] < -peaktopNoise)

    The second disjunct has NO ``!foundTop`` guard, and ``d2[j] < -peaktopNoise``
    binds to the second disjunct only.
    """
    n = ss.size
    j = i
    found_peak_top = False
    peaktop_check_point = j
    while j + 2 != n - 1:
        j += 1

        # peak top check (operator-precedence quirk; see docstring)
        if (
            (not found_peak_top and d1[j - 1] > 0 and d1[j] < 0)
            or (d1[j - 1] > 0 and d1[j + 1] < 0 and d2[j] < -1.0 * peaktop_noise)
        ):
            found_peak_top = True
            peaktop_check_point = j

        if (not found_peak_top) and _is_large_peak_top(ss, j):
            found_peak_top = True
            peaktop_check_point = j

        # peak top check force (inert for min_data_points >= 1.5)
        if (
            (not found_peak_top)
            and min_data_points < 1.5
            and _is_broad_peak_top(ss, j)
        ):
            found_peak_top = True
            peaktop_check_point = j

        minimum_point_from_top = 1 if min_data_points <= 3 else min_data_points * 0.5
        if found_peak_top and peaktop_check_point + minimum_point_from_top <= j - 1:
            if d1[j] > -1.0 * slope_noise * slope_fold:
                break
            if _is_flat(ss, j - 1, amplitude_noise):
                break
            if _is_large_bottom(ss, j):
                break
            # peak right check force (inert for min_data_points >= 1.5)
            if min_data_points < 1.5 and _is_broad_bottom(ss, j):
                found_peak_top = True
                peaktop_check_point = j
    return j


def _search_real_right_edge(
    ss: np.ndarray, i: int, state: dict
) -> tuple[int, bool]:
    """ChromatogramGlobalProperty_temp2.SearchRealRightEdge (lines 385-421).

    Refine the right edge within 5 points in either direction on ``ss``. The
    ``state`` dict carries the cross-iteration infinite-loop guard
    (``infinitLoopCheck`` / ``infinitLoopID``); returns ``(j, is_break)`` where
    ``is_break`` propagates to break the outer DetectPeaks loop.
    """
    n = ss.size
    is_too_long_right_edge = False
    trackcounter = 0
    is_break = False

    # case: wrong edge is in right of real edge (walk left)
    for j in range(0, 6):
        if i - j - 1 < 0 or (ss[i - j] - ss[i - j - 1]) <= 0:
            break
        is_too_long_right_edge = True
        trackcounter += 1
    if is_too_long_right_edge:
        k = i - trackcounter
        if state["check"] and k == state["id"] and k > n - 10:
            is_break = True
        else:
            state["check"] = True
            state["id"] = k
        return k, is_break

    # case: wrong edge is in left of real edge (walk right)
    for j in range(0, 6):
        if i + j + 1 > n - 1:
            break
        if (ss[i + j] - ss[i + j + 1]) <= 0:
            break
        trackcounter += 1
    return i + trackcounter, is_break


def _get_peak_top_id(chrom: np.ndarray, start: int, end: int) -> int:
    """Chromatogram.GetPeakTopId (Chromatogram.cs lines 826-836).

    Index of the maximum in ``chrom[start:end]`` (end exclusive), first
    occurrence on ties -- matching the strict ``<`` update in the C# loop.
    """
    return start + int(np.argmax(chrom[start:end]))


def _shrink_peak_range(
    chrom: np.ndarray, start: int, end: int, average_peak_width: int
) -> tuple[int, int, int]:
    """Chromatogram.ShrinkPeakRange (Chromatogram.cs lines 788-814) on ``chrom``.

    ``end`` is EXCLUSIVE in and out. Re-finds the top via ``GetPeakTopId`` and
    trims only when the peak is wider than ``average_peak_width`` (otherwise the
    inner loops never execute and the bounds are returned unchanged).
    """
    peak_top_id = _get_peak_top_id(chrom, start, end)

    new_start = start
    j = peak_top_id - average_peak_width
    while j >= start:
        if j - 1 < start:
            break
        if chrom[j - 1] >= chrom[j]:
            new_start = j
            break
        j -= 1

    new_end = end
    j = peak_top_id + average_peak_width
    while j < end:
        if j + 1 >= end:
            break
        if chrom[j] <= chrom[j + 1]:
            new_end = j + 1
            break
        j += 1

    return new_start, peak_top_id, new_end


def _ieee_div(a: float, b: float) -> float:
    """Divide with IEEE-754 semantics (x/0 -> inf, 0/0 -> nan) like C# double.

    Python's ``float`` division raises ``ZeroDivisionError``; the gaussian
    similarity ratios in ``GetPeakDetectionResult`` rely on the C# silent
    inf/nan behaviour for degenerate (zero-area) peaks.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(a) / np.float64(b))


def _assemble_peak(
    chrom: np.ndarray,
    rt: np.ndarray,
    peak_top_id: int,
    start: int,
    end: int,
    max_peak_height: float,
    estimated_noise: float,
    precursor_mz_nominal: int,
    product_mz: float,
) -> DetectedPeak | None:
    """Port of ChroChroChromatogram.GetPeakDetectionResult (lines 70-225).

    All intensity/time accessors operate on ``chrom`` (the LWMA-smoothed trace,
    MS-DIAL ``_chromatogram``); ``rt`` supplies ``Time``. ``end`` is EXCLUSIVE.
    Returns the :class:`DetectedPeak`, or ``None`` for the two null guards
    (``end - start <= 3`` or ``maxHeight < 0``). The omitted PeakDetectionResult
    scores (shapeness / symmetry / basePeak / peakPure / idealSlope) are not
    carried by ``DetectedPeak``; ``area`` (x60), ``height``, ``sn_ratio`` and
    ``gaussian_similarity`` are computed faithfully.
    """
    # Null guards (lines 72, 74).
    if end - start <= 3:
        return None
    if max_peak_height < 0:
        return None

    top_int = chrom[peak_top_id]
    start_int = chrom[start]
    end_int = chrom[end - 1]

    # --- HWHM ids (lines 83-130, only the half-height ids are needed) -------
    # Left scan: id minimising |(top-start)/2 - (chrom[j]-start)| (lines 84-87).
    top_minus_start = top_int - start_int
    peak_half_diff = np.inf
    left_peak_half_id = -1
    for j in range(peak_top_id, start - 1, -1):
        diff = abs(top_minus_start / 2.0 - (chrom[j] - start_int))
        if peak_half_diff > diff:
            peak_half_diff = diff
            left_peak_half_id = j
    # Right scan: id minimising |(top-end)/2 - (chrom[j]-end)| (lines 111-114).
    top_minus_end = top_int - end_int
    peak_half_diff = np.inf
    right_peak_half_id = -1
    for j in range(peak_top_id, end):
        diff = abs(top_minus_end / 2.0 - (chrom[j] - end_int))
        if peak_half_diff > diff:
            peak_half_diff = diff
            right_peak_half_id = j

    # gaussianNormalize / peakHalfId selection (lines 135-144).
    if (start_int - end_int) <= 0:  # IntensityDifference(start, end-1) <= 0
        gaussian_normalize = top_minus_start
        peak_half_id = left_peak_half_id
    else:
        gaussian_normalize = top_minus_end
        peak_half_id = right_peak_half_id

    # --- gaussian + real areas (lines 152-189) -----------------------------
    peak_hwhm = abs(rt[peak_half_id] - rt[peak_top_id])  # TimeDifference
    gaussian_sigma = peak_hwhm / np.sqrt(2.0 * np.log(2.0))
    gaussian_area = gaussian_normalize * gaussian_sigma * np.sqrt(2.0 * np.pi) / 2.0

    real_area_above_zero = 0.0
    left_peak_area = 0.0
    right_peak_area = 0.0
    for j in range(start, end - 1):
        real_area_above_zero += (
            (chrom[j + 1] + chrom[j]) * abs(rt[j + 1] - rt[j]) / 2.0
        )  # CalculateArea(j+1, j)
        if j == peak_top_id - 1:
            left_peak_area = real_area_above_zero
        elif j == end - 2:
            right_peak_area = real_area_above_zero - left_peak_area

    # realAreaAboveBaseline = realAreaAboveZero - CalculateArea(end-1, start).
    real_area_above_baseline = real_area_above_zero - (
        (end_int + start_int) * abs(rt[end - 1] - rt[start]) / 2.0
    )

    # Subtract the edge-linear baseline contribution from each half (lines 172-179).
    if (start_int - end_int) <= 0:
        left_peak_area -= start_int * (rt[peak_top_id] - rt[start])
        right_peak_area -= start_int * (rt[end - 1] - rt[peak_top_id])
    else:
        left_peak_area -= end_int * (rt[peak_top_id] - rt[start])
        right_peak_area -= end_int * (rt[end - 1] - rt[peak_top_id])

    # Symmetric gaussian-area ratio average (lines 181-189).
    if gaussian_area >= left_peak_area:
        gauss_left = _ieee_div(left_peak_area, gaussian_area)
    else:
        gauss_left = _ieee_div(gaussian_area, left_peak_area)
    if gaussian_area >= right_peak_area:
        gauss_right = _ieee_div(right_peak_area, gaussian_area)
    else:
        gauss_right = _ieee_div(gaussian_area, right_peak_area)
    gaussian_similarity = (gauss_left + gauss_right) / 2.0

    # AreaAboveBaseline = realAreaAboveBaseline * 60 (line 205);
    # SignalToNoise = maxPeakHeight / estimatedNoise (line 223).
    return DetectedPeak(
        precursor_mz_nominal=int(precursor_mz_nominal),
        product_mz=float(product_mz),
        rt_apex=float(rt[peak_top_id]),
        rt_left=float(rt[start]),
        rt_right=float(rt[end - 1]),
        apex_index=int(peak_top_id),
        left_index=int(start),
        right_index=int(end - 1),
        height=float(top_int),
        area=float(real_area_above_baseline * 60.0),
        sn_ratio=float(max_peak_height / estimated_noise),
        gaussian_similarity=float(gaussian_similarity),
    )


# ---------------------------------------------------------------------------
# 7. Background-spike filter (PeakSpottingCore.cs:656-676 + CountSpikes)
# ---------------------------------------------------------------------------


def _count_spikes(
    raw: np.ndarray, left_id: int, right_id: int, threshold: float
) -> int:
    """Chromatogram.CountSpikes (Chromatogram.cs lines 889-911) on the RAW EIC.

    Scans ``[max(left_id, 1), min(right_id, n-2)]`` inclusive, tracking the
    latest ``IsPeakTop`` -> spikeMax (the ``else if`` means a peak-top point
    never also sets spikeMin) and ``IsBottom`` -> spikeMin; whenever both are
    set it increments when ``|spikeMax - spikeMin| / 2 > threshold`` and resets.
    """
    n = raw.size
    left_bound = max(left_id, 1)
    right_bound = min(right_id, n - 2)
    counter = 0
    spike_max = None
    spike_min = None
    for i in range(left_bound, right_bound + 1):
        if _is_peak_top(raw, i):
            spike_max = raw[i]
        elif _is_bottom(raw, i):
            spike_min = raw[i]
        if spike_max is not None and spike_min is not None:
            if abs(spike_max - spike_min) / 2.0 > threshold:
                counter += 1
            spike_max = None
            spike_min = None
    return counter


def _background_spike_filter(
    peaks: list[DetectedPeak],
    raw: np.ndarray,
    chrom: np.ndarray,
    threshold: int,
) -> list[DetectedPeak]:
    """PeakSpottingCore.GetBackgroundSubtractedPeaks (lines 656-676) on RAW.

    Drops a peak whose flanking spike ``counter >= threshold`` (15) or whose
    apex fails ``IsValidPeakTop`` on the RAW EIC. ``ampDiff`` uses the
    chrom-based edge heights (``PeakHeightTop/Left/Right`` =
    ``IntensityAtPeakTop/Left/Right``); the spike scan itself runs on RAW.
    """
    kept: list[DetectedPeak] = []
    for peak in peaks:
        top = peak.apex_index
        left = peak.left_index
        right = peak.right_index
        if not _is_valid_peak_top(raw, top):
            continue
        tracking = 10 * (right - left)
        if tracking > 50:
            tracking = 50
        amp_diff = max(peak.height - chrom[left], peak.height - chrom[right])
        counter = _count_spikes(raw, left - tracking, left, amp_diff / 3.0)
        counter += _count_spikes(raw, right, right + tracking, amp_diff / 3.0)
        if counter < threshold:
            kept.append(peak)
    return kept


# ---------------------------------------------------------------------------
# 8. Public engine: per-chromatogram derivative peak picking
# ---------------------------------------------------------------------------


def msdial_detect_peaks_in_chromatogram(
    rt_array: np.ndarray,
    intensity_array: np.ndarray,
    *,
    config: MsdialPeakSpottingConfig,
    precursor_mz_nominal: int = 0,
    product_mz: float = 0.0,
) -> list[DetectedPeak]:
    """Detect peaks in one chromatogram, faithful to the MSDIAL5 normal LC path.

    Pipeline (each step cites the C# source):

      1. Smoothing chain (PeakSpottingCore.cs:359 + Chromatogram.GetProperty,
         Chromatogram.cs:693-706)::

            chrom     = LWMA(raw, smoothing_level)          # the property trace
            ss        = LWMA(LWMA(chrom, 1), 1)             # SmoothedChromatogram
            baseline  = LWMA(chrom, baseline_window)
            corrected = max(ss - baseline, 0)               # Difference

      2. Global noise from ``corrected`` (GetMinimumNoiseLevel x NoiseFactor)
         and ``baselineMedian`` / max / min from ``chrom`` (NOT raw): MS-DIAL
         calls ``GetIntensityMedian/Maximum/Minimum`` on ``this`` == the
         LWMA(smoothing_level)-smoothed chromatogram passed into
         ``PeakDetectionVS1``.
      3. SG derivatives + slope noises on ``ss``.
      4. The DetectPeaks walk (ChromatogramGlobalProperty_temp2.cs:227-264):
         IsPeakStarted -> SearchRealLeftEdge -> SearchRightEdgeCandidate ->
         SearchRealRightEdge -> ShrinkPeakRange (on ``chrom``) ->
         PeakHeightFromBounds -> IsNoise -> GetPeakDetectionResult.
      5. RAW-EIC background-spike filter (PeakSpottingCore.cs:656-676).

    Returns a list of :class:`DetectedPeak` sorted by ``rt_apex`` (``area`` in
    seconds, ``height`` = chrom apex intensity). ``precursor_mz_nominal`` /
    ``product_mz`` are passed straight through (apex m/z is assigned at the
    slice level in Task 2.2). Empty / too-short / flat traces yield ``[]``.
    """
    rt = np.asarray(rt_array, dtype=np.float64)
    raw = np.asarray(intensity_array, dtype=np.float64)
    n = raw.size
    if n == 0 or rt.size != n:
        return []

    smoothing_level = int(config.smoothing_level)
    baseline_window = int(config.baseline_window)

    # 1. Smoothing chain (uses the faithful MS-DIAL imos LWMA, _lwma_msdial,
    # whose edge handling matches the reference -- NOT baseline.lwma_smooth).
    chrom = _lwma_msdial(raw, smoothing_level)
    ss = _lwma_msdial(_lwma_msdial(chrom, 1), 1)
    baseline = _lwma_msdial(chrom, baseline_window)
    corrected = np.maximum(ss - baseline, 0.0)

    # 2. Global noise (post-factor) + estimated noise from corrected.
    noise, estimated_noise = _estimate_global_noise(
        corrected,
        bin_size=int(config.noise_bin_size),
        factor=float(config.noise_factor),
        min_windows=int(config.min_noise_windows),
    )
    # baselineMedian / max / min from chrom (Chromatogram.GetProperty:701-704).
    baseline_median = _upper_median(chrom)
    max_int = float(chrom.max())
    min_int = float(chrom.min())
    is_high_baseline = baseline_median > (max_int + min_int) * 0.5

    # 3. SG derivatives + slope noises on ss.
    d1, d2 = _sg_derivatives(ss)
    amplitude_noise, slope_noise, peaktop_noise = _slope_noises(
        ss, d1, d2, small_diff_frac=float(config.slope_noise_small_diff_frac)
    )

    # 4. DetectPeaks walk.
    min_data_points = float(config.min_data_points)
    margin = max(int(min_data_points), 2)
    slope_fold = float(config.slope_noise_fold)
    amplitude_fold = float(config.amplitude_noise_fold)
    min_amplitude = float(config.min_amplitude)
    average_peak_width = int(config.average_peak_width)

    candidates: list[DetectedPeak] = []
    state = {"check": False, "id": 0}
    i = margin
    while i < n - margin:
        # IsPeakStarted (ChromatogramGlobalProperty_temp2.cs:22-25).
        if (
            d1[i] > slope_noise * slope_fold
            and d1[i + 1] > slope_noise * slope_fold
        ):
            start = _search_real_left_edge(ss, i)
            j = _search_right_edge_candidate(
                ss, d1, d2, i, min_data_points, slope_fold,
                peaktop_noise, slope_noise, amplitude_noise,
            )
            j, is_break = _search_real_right_edge(ss, j, state)
            if is_break:
                break
            i = max(i, j)  # advance past this peak (splits two resolved peaks)
            if j - start + 1 >= min_data_points:
                new_start, peak_top_id, new_end = _shrink_peak_range(
                    chrom, start, j + 1, average_peak_width
                )
                top_int = chrom[peak_top_id]
                left_int = chrom[new_start]
                right_int = chrom[new_end - 1]
                # PeakHeightFromBounds (Chromatogram.cs:732-737), NOT floored.
                min_peak_height = top_int - max(left_int, right_int)
                max_peak_height = top_int - min(left_int, right_int)
                edge_min = min(left_int, right_int)
                if not _is_noise(
                    min_peak_height,
                    max_peak_height,
                    noise,
                    amplitude_noise,
                    min_amplitude,
                    fold=amplitude_fold,
                    is_high_baseline=is_high_baseline,
                    edge_min=edge_min,
                    baseline_median=baseline_median,
                ):
                    peak = _assemble_peak(
                        chrom, rt, peak_top_id, new_start, new_end,
                        max_peak_height, estimated_noise,
                        precursor_mz_nominal, product_mz,
                    )
                    # GetChromatogramPeakFeatures skips IntensityAtPeakTop <= 0.
                    if peak is not None and peak.height > 0:
                        candidates.append(peak)
        i += 1

    # 5. Background-spike filter on the RAW EIC, then sort by apex RT.
    kept = _background_spike_filter(
        candidates, raw, chrom, int(config.background_spike_threshold)
    )
    kept.sort(key=lambda p: p.rt_apex)
    return kept
