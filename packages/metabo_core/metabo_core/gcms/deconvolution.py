"""AMDIS-style spectrum deconvolution for GC-MS.

This module is the Plan D core that replaces the simpler "apex + denoise"
spectrum reconstruction in ``apps/gcms_processor``. It follows Stein 1999
(NIST AMDIS, J Am Soc Mass Spectrom 10:770-781), implementing four phases:

1. **Noise analysis** (this file, Task 3a):
   - Estimate ion-counting noise factor Nf via 13-scan segments,
     median-of-medians, with crossing-count rejection.
   - Replace zero-abundance values via threshold-transition correction.
2. **Component perception** via sharpness bins (Task 3b).
3. **Model peak construction** (Task 3c).
4. **Deconvolution** via per-m/z least-squares with adjacent-component
   subtraction (Task 3d).
5. **Peak flagging** (Task 3e).
6. **Top-level entry**: ``deconvolve_features`` (Task 3f).

The algorithm is intentionally faithful to the AMDIS paper. Departures
that are explicit Plan D decisions (RT/RI Gaussian instead of linear
penalty; composite spectral shape via Plan C primitives instead of AMDIS
eq 7's flag-weighted pure:impure mix; AMDIS eq 10 component-purity
correction not implemented) live in ``gcms_match_factor`` upstairs in
``library_matching.py`` — not in this module. This module is the pure
deconvolution side.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Public dataclasses (used by Tasks 3b, 3c, 3d, 3f)
# ---------------------------------------------------------------------------

@dataclass
class IonPeak:
    """A single ion-chromatogram peak that passed the 4-step baseline test."""
    ion_index: int           # which ion chromatogram (index into the array list)
    apex_scan_int: int       # integer apex scan
    apex_scan_precise: float # parabola-fit precise apex (in scan units)
    apex_intensity: float    # height above baseline at the apex
    sharpness: float         # average sharpness over both sides
    window_lo: int           # inclusive
    window_hi: int           # exclusive
    baseline: np.ndarray     # baseline values across [window_lo, window_hi)
    is_tic: bool = False     # True if this peak comes from the TIC


@dataclass
class Component:
    """A perceived chromatographic component."""
    apex_scan: float                # in scan units (precise)
    apex_scan_int: int              # nearest integer
    sharpness: float                # max bin sharpness assigning this component
    contributing_ions: list[int]    # ion indices that maximize at this apex
                                    #   AND have sharpness >= 75% of the max
    window_lo: int = 0              # inferred from the contributing ions
    window_hi: int = 0
    perceived_via_tic: bool = False # True iff perception came from TIC path


@dataclass
class DeconvolvedSpectrum:
    """Result of per-m/z deconvolution for a single component."""
    mz: np.ndarray                          # 1D array of m/z values
    intensity: np.ndarray                   # 1D array of deconvolved intensities
                                             #  (same length as ``mz``)
    intensity_no_subtraction: np.ndarray    # without-subtraction variant
                                             #  (Stein 1999 keeps both)
    flags: np.ndarray                       # bool array; True = flagged
    n_adjacent_subtracted: int              # 0..2 per AMDIS limit
    apex_scan: int


# ---------------------------------------------------------------------------
# Phase 1: Noise analysis
# ---------------------------------------------------------------------------

# AMDIS Stein 1999 §"Noise Analysis": 13 scans per segment.
NOISE_SEGMENT_LEN = 13
# "If the number of crossings is less than one-half the number scans in the
# segment (six or less), the segment is rejected." Half of 13 = 6.5, so we
# require >= 7 crossings to accept the segment.
MIN_CROSSINGS = 7
# Number of equal-length chromatogram segments for threshold-transition
# fraction computation (paper says 10).
THRESHOLD_TRANSITION_SEGMENTS = 10


def _segment_noise_factor(segment: np.ndarray) -> Optional[float]:
    """Single-segment Nf sample, or ``None`` if rejected.

    Per AMDIS Stein 1999:
    - Reject if any abundance is exactly zero.
    - Reject if the number of mean-crossings is < 7 (half of 13 = 6.5;
      "less than one-half" → <= 6 means rejected, so >= 7 to accept).
    - Otherwise: median of |x - mean(x)| / sqrt(mean(x)).
    """
    if segment.size != NOISE_SEGMENT_LEN:
        return None
    if np.any(segment == 0):
        return None
    mean = float(np.mean(segment))
    if mean <= 0:
        return None
    # Crossings: pairs of consecutive scans where one is above the mean
    # and the other is below.
    above = segment > mean
    transitions = int(np.sum(above[1:] != above[:-1]))
    if transitions < MIN_CROSSINGS:
        return None
    # Median deviation from the mean.
    median_dev = float(np.median(np.abs(segment - mean)))
    return median_dev / math.sqrt(mean)


def estimate_noise_factor(
    chromatograms: Sequence[np.ndarray],
    *,
    include_tic: bool = True,
) -> float:
    """Estimate the noise factor ``Nf`` per AMDIS Stein 1999 §"Noise Analysis".

    Each input array is an ion chromatogram (1D abundance vs scan). If
    ``include_tic`` and more than one chromatogram is provided, the TIC
    (sum across chromatograms) is also fed through the same procedure.

    Parameters
    ----------
    chromatograms
        Sequence of 1D float arrays of equal length (number of scans).
    include_tic
        If True (default), the TIC is appended to the chromatogram set
        before analysis (matches AMDIS).

    Returns
    -------
    float
        Median-of-medians sample noise factor. Returns 1.0 if no valid
        segment can be found anywhere in the data (degenerate fallback;
        callers can decide whether to warn).
    """
    if not chromatograms:
        return 1.0

    arrays = [np.asarray(c, dtype=np.float64).ravel() for c in chromatograms]
    if include_tic and len(arrays) > 1:
        # Stack only when all arrays are aligned; else skip TIC.
        n_scans = arrays[0].size
        if all(a.size == n_scans for a in arrays):
            tic = np.sum(np.stack(arrays, axis=0), axis=0)
            arrays = list(arrays) + [tic]

    # AMDIS Stein 1999 §"Noise Analysis": per-chromatogram median of
    # accepted segment Nf samples first; then median across chromatograms.
    # On heterogeneous chromatograms (clean ions next to noisy ions) this
    # diverges from a flat global median because each chromatogram gets one
    # vote regardless of how many segments survived.
    per_chrom_medians: list[float] = []
    for chrom in arrays:
        n = chrom.size
        if n < NOISE_SEGMENT_LEN:
            continue
        # Non-overlapping 13-scan segments. AMDIS does not strictly require
        # non-overlapping; non-overlap is the simplest and matches the
        # paper's illustration.
        n_segments = n // NOISE_SEGMENT_LEN
        chrom_samples: list[float] = []
        for k in range(n_segments):
            seg = chrom[k * NOISE_SEGMENT_LEN : (k + 1) * NOISE_SEGMENT_LEN]
            sample = _segment_noise_factor(seg)
            if sample is not None and math.isfinite(sample):
                chrom_samples.append(sample)
        if chrom_samples:
            per_chrom_medians.append(float(np.median(chrom_samples)))

    if not per_chrom_medians:
        return 1.0
    return float(np.median(per_chrom_medians))


def replace_threshold_transitions(
    chromatogram: np.ndarray,
    *,
    threshold_value: Optional[float] = None,
    n_segments: int = THRESHOLD_TRANSITION_SEGMENTS,
) -> np.ndarray:
    """Replace zero-abundance values per AMDIS Stein 1999 §"Threshold transitions".

    Steps:
    1. Establish ``AT`` = smallest non-zero abundance in the chromatogram
       (or use the explicit ``threshold_value`` argument if given).
    2. Split the chromatogram into ``n_segments`` equal-length segments.
    3. In each segment, count the fraction of scans involved in
       transitions from zero to non-zero (or non-zero to zero).
    4. Replace zero abundances within that segment with
       ``AT * sqrt(fraction_of_threshold_transitions)``.

    Parameters
    ----------
    chromatogram
        1D abundance array.
    threshold_value
        Optional fixed AT. Otherwise inferred from the smallest non-zero
        value.
    n_segments
        Number of equal-length segments (paper recommends 10).

    Returns
    -------
    np.ndarray
        New chromatogram array with zero values replaced. The input is
        not modified in place.
    """
    arr = np.asarray(chromatogram, dtype=np.float64).copy()
    n = arr.size
    if n == 0:
        return arr

    nonzero = arr[arr > 0]
    if threshold_value is None:
        if nonzero.size == 0:
            # All zeros: no threshold transitions exist; return as-is.
            return arr
        AT = float(np.min(nonzero))
    else:
        AT = float(threshold_value)
    if AT <= 0:
        return arr

    # Split into equal-length segments. Last segment may be slightly larger
    # if n is not divisible by n_segments.
    n_segments = max(1, int(n_segments))
    bounds = np.linspace(0, n, n_segments + 1, dtype=int)

    is_zero = arr == 0
    # Mark transitions: scans where neighbor differs in zero-ness.
    transitions = np.zeros(n, dtype=bool)
    if n > 1:
        # Either left or right neighbor crosses the zero/non-zero boundary.
        diff = is_zero[:-1] != is_zero[1:]
        transitions[:-1] |= diff
        transitions[1:] |= diff

    for k in range(n_segments):
        lo, hi = bounds[k], bounds[k + 1]
        if hi <= lo:
            continue
        segment_zero_idx = np.where(is_zero[lo:hi])[0] + lo
        if segment_zero_idx.size == 0:
            continue
        seg_len = hi - lo
        # Fraction of scans in this segment involved in any zero/non-zero
        # transition.
        n_trans = int(np.sum(transitions[lo:hi]))
        fraction = n_trans / float(seg_len) if seg_len > 0 else 0.0
        replacement = AT * math.sqrt(max(fraction, 0.0))
        arr[segment_zero_idx] = replacement

    return arr


# ---------------------------------------------------------------------------
# Phase 2: Component perception (Task 3b)
# ---------------------------------------------------------------------------

# AMDIS Stein 1999 §"Component Perception" parameters.
DEFAULT_DECONV_WINDOW = 12         # max scans on each side of the apex
DEFAULT_JUMP_NOISE_UNITS = 5.0     # "more than five noise units greater"
DEFAULT_FALL_FRACTION = 0.05       # window stops when intensity < 5% of max
DEFAULT_HEIGHT_NOISE_UNITS = 4.0   # peak height rejection threshold
DEFAULT_SHARPNESS_BINS_PER_SCAN = 10
DEFAULT_SHARPNESS_RANGE_FACTOR = 50  # range of bins = 50 / sharpness
DEFAULT_SHARPNESS_CUTOFF_RATIO = 0.75  # ions within 75% of max sharpness


def deskew_chromatograms(
    chromatograms: np.ndarray,
    *,
    AT: float = 0.0,
) -> np.ndarray:
    """De-skew a stack of ion chromatograms via 3-point quadratic interpolation.

    AMDIS Stein 1999 §"Component Perception", first paragraph: scanning
    instruments acquire different m/z peaks of a single component at
    distinctly different parts of the elution profile. De-skewing time-
    aligns by fitting a parabola through scans n-1, n, n+1 and shifting.

    Three special cases (AMDIS):
      1. Abundance values in the first and last scan are NOT interpolated.
      2. Zero abundance values are NOT interpolated (they keep their zeros).
      3. Non-zero interpolated values cannot be less than ``AT``.

    Parameters
    ----------
    chromatograms
        2D array (n_ions, n_scans) of intensities.
    AT
        Detection threshold; interpolated non-zero values below ``AT`` are
        clamped up to ``AT``.

    Returns
    -------
    np.ndarray
        New 2D array with the same shape; first/last columns unchanged.

    Notes
    -----
    A simple 3-point quadratic fit at index n predicts the abundance at
    n + delta where delta is the fractional shift required to time-align
    the m/z peak with the underlying component apex. Since the per-ion
    skew is unknown without an instrument-specific model, we default
    delta=0 (no shift) for ions whose 3-point local fit doesn't show a
    clear maximum in the [n-0.5, n+0.5] window. This preserves AMDIS's
    "abundance values in the first and last scans are not interpolated"
    rule and keeps the output identical to the input when no skew is
    detected.

    The function is the AMDIS "de-skewing" hook: callers can feed it
    raw intensities and get a smoothly shifted version. For a more
    aggressive de-skew, callers can pre-fit a per-ion delta and apply
    it externally.
    """
    arr = np.asarray(chromatograms, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("chromatograms must be 2D (n_ions, n_scans)")
    n_ions, n_scans = arr.shape
    if n_scans < 3:
        return arr.copy()

    out = arr.copy()

    # Vectorized 3-point quadratic over every interior column at once.
    # 等价于 ``for i, for n: a,b,c = arr[i,n-1:n+2]`` 的逐元素逻辑, 但
    # 把 (n_ions × (n_scans-2)) 个独立小算法合并成一组 numpy 表达式,
    # 消除 Python 双层 for 的解释器开销。
    a = arr[:, :-2]     # chrom[n - 1]
    b = arr[:, 1:-1]    # chrom[n]
    c = arr[:, 2:]      # chrom[n + 1]

    A = (a + c) / 2.0 - b
    B = (c - a) / 2.0
    C = b

    A_nonzero = A != 0
    # 避免 A=0 引起的 divide-by-zero / inf 污染: 用 1 占位, 后面 mask 过滤掉。
    A_safe = np.where(A_nonzero, A, 1.0)
    delta = -B / (2.0 * A_safe)

    value = A * delta * delta + B * delta + C

    # Special case 3: positive non-zero values that fall below AT get bumped
    # back up to AT (only when AT > 0). Mirrors `if value > 0 and AT > 0 and
    # value < AT: value = AT`.
    if AT > 0:
        below_AT = (value > 0) & (value < AT)
        value = np.where(below_AT, AT, value)

    # 更新条件 (与原 if 链等价):
    #   b != 0           # special case 2
    #   A != 0           # 线性段不外推
    #   |delta| <= 0.5   # 极值在 [-0.5, +0.5] 之内
    #   value >= 0       # 负的外推视作噪声, 保留原值
    update_mask = (
        (b != 0)
        & A_nonzero
        & (np.abs(delta) <= 0.5)
        & (value >= 0)
    )

    inner = out[:, 1:-1]
    inner[update_mask] = value[update_mask]
    return out


def _set_window(
    chrom: np.ndarray,
    apex: int,
    Nf: float,
    *,
    max_window: int = DEFAULT_DECONV_WINDOW,
    jump_noise_units: float = DEFAULT_JUMP_NOISE_UNITS,
    fall_fraction: float = DEFAULT_FALL_FRACTION,
) -> tuple[int, int]:
    """Set the deconvolution window per AMDIS step (a).

    Sweep outward from ``apex`` up to ``max_window`` scans on each side.
    Stop early on either:
      - A scan whose abundance is more than ``jump_noise_units`` noise
        units greater than the smallest abundance between that scan and
        the apex (presumed neighbor component), measured for the smallest
        abundance.
      - A scan where intensity falls below ``fall_fraction`` of the peak
        max.
    """
    n = chrom.size
    apex_int = float(chrom[apex])
    fall = fall_fraction * apex_int

    def _sweep(direction: int) -> int:
        cur_min = apex_int
        last = apex
        for step in range(1, max_window + 1):
            scan = apex + direction * step
            if scan < 0 or scan >= n:
                break
            v = float(chrom[scan])
            if v < cur_min:
                cur_min = v
            # Jump check.
            if cur_min > 0 and v - cur_min > jump_noise_units * Nf * math.sqrt(max(cur_min, 1e-12)):
                # Window ends at the previous scan.
                break
            last = scan
            if v <= fall:
                break
        return last

    lo = _sweep(-1)
    hi = _sweep(+1)
    return min(lo, apex), max(hi, apex) + 1  # half-open [lo, hi)


def _compute_baseline(
    chrom: np.ndarray,
    lo: int,
    hi: int,
) -> np.ndarray:
    """4-step baseline (AMDIS step b/c) across [lo, hi).

    1. Tentative line through the lowest end points.
    2. Adjusted so no point falls below the line.
    3. Least-squares line using the lowest-half points (measured from
       the tentative baseline).

    Returns a 1D baseline array of length (hi - lo).
    """
    seg = chrom[lo:hi]
    n = seg.size
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.array([float(seg[0])])
    # Step 1: line through end points (lowest by default = end-point
    # values themselves).
    y0, y1 = float(seg[0]), float(seg[-1])
    x = np.arange(n, dtype=np.float64)
    base = y0 + (y1 - y0) * x / (n - 1)
    # Step 2: adjust if any point is below.
    diff = seg - base
    if diff.min() < 0:
        base = base + diff.min()  # shift baseline down so no point dips below.
    # Step 3: least-squares line on the lowest-half points (measured
    # from the adjusted baseline).
    above = seg - base
    sorted_idx = np.argsort(above)
    half = max(2, n // 2)
    chosen = sorted_idx[:half]
    cx = x[chosen]
    cy = seg[chosen]
    # Linear fit y = a + b*x.
    if cx.size >= 2 and np.std(cx) > 0:
        b, a = np.polyfit(cx, cy, 1)
        ls_base = a + b * x
    else:
        ls_base = base
    return ls_base


def _peak_sharpness(
    chrom: np.ndarray,
    apex: int,
    lo: int,
    hi: int,
    Nf: float,
) -> float:
    """Average peak sharpness per AMDIS eq (2): max over both sides of
    ``(Amax - An) / (n * Nf * sqrt(Amax))``.
    """
    Amax = float(chrom[apex])
    if Amax <= 0:
        return 0.0
    sqrt_amax = math.sqrt(Amax)
    Nf_safe = max(Nf, 1e-9)

    left_max = 0.0
    for n in range(1, apex - lo + 1):
        idx = apex - n
        if idx < lo:
            break
        An = float(chrom[idx])
        s = (Amax - An) / (n * Nf_safe * sqrt_amax)
        if s > left_max:
            left_max = s
    right_max = 0.0
    for n in range(1, hi - apex):
        idx = apex + n
        if idx >= hi:
            break
        An = float(chrom[idx])
        s = (Amax - An) / (n * Nf_safe * sqrt_amax)
        if s > right_max:
            right_max = s
    return 0.5 * (left_max + right_max)


def peak_sharpness(chrom, apex, lo, hi, Nf):
    """Public wrapper: AMDIS eq.(2) sharpness of an EXISTING apex (no peak detection).

    Lets non-GC callers (ASFAM MS2 AMDIS clustering) compute the sharpness of a
    peak found by their own detector without duplicating the formula. The private
    ``_peak_sharpness`` stays the GC internal name (scalar path unchanged).
    """
    return _peak_sharpness(chrom, apex, lo, hi, Nf)


def _parabola_apex(chrom: np.ndarray, apex: int) -> float:
    """3-point parabolic fit for the precise apex time (in scan units)."""
    n = chrom.size
    if apex <= 0 or apex >= n - 1:
        return float(apex)
    a = float(chrom[apex - 1])
    b = float(chrom[apex])
    c = float(chrom[apex + 1])
    # delta = (a - c) / (2 * (a - 2b + c))
    denom = 2.0 * (a - 2.0 * b + c)
    if denom == 0:
        return float(apex)
    delta = (a - c) / denom
    return float(apex) + max(min(delta, 0.5), -0.5)


def parabola_apex(chrom, apex):
    """Public wrapper: 3-point parabolic sub-scan apex refinement of an existing apex."""
    return _parabola_apex(chrom, apex)


def _detect_local_maxima(chrom: np.ndarray) -> list[int]:
    """Indices of strict local maxima (chrom[n-1] < chrom[n] > chrom[n+1])."""
    n = chrom.size
    if n < 3:
        return []
    out = []
    for i in range(1, n - 1):
        if chrom[i] > chrom[i - 1] and chrom[i] > chrom[i + 1] and chrom[i] > 0:
            out.append(i)
    return out


def _ion_peaks(
    chromatograms: np.ndarray,
    Nf: float,
    *,
    max_window: int = DEFAULT_DECONV_WINDOW,
    jump_noise_units: float = DEFAULT_JUMP_NOISE_UNITS,
    fall_fraction: float = DEFAULT_FALL_FRACTION,
    height_noise_units: float = DEFAULT_HEIGHT_NOISE_UNITS,
) -> list[IonPeak]:
    """Run AMDIS steps (a)-(d) on every ion chromatogram, returning the
    set of accepted ion-chromatogram peaks (IonPeak instances).

    Performance hot path: real-data files have many "silent" ions that
    never reach the height-rejection threshold. We pre-screen each
    candidate apex BEFORE running the costly window + baseline +
    sharpness pipeline: an apex with raw intensity below
    ``height_noise_units * Nf * sqrt(apex_intensity)`` cannot pass
    the eventual baseline-subtracted height test either.
    """
    n_ions, n_scans = chromatograms.shape
    out: list[IonPeak] = []
    Nf_safe = max(Nf, 1e-9)
    for i in range(n_ions):
        chrom = chromatograms[i]
        if chrom.size == 0:
            continue
        max_int = float(np.max(chrom))
        if max_int <= 0:
            continue
        # Skip ions whose max can't ever pass the height test.
        if max_int < height_noise_units * Nf_safe * math.sqrt(max_int):
            continue
        for apex in _detect_local_maxima(chrom):
            apex_intensity = float(chrom[apex])
            # Cheap early pre-screen on raw apex intensity (the
            # baseline-subtracted height is at most apex_intensity).
            if apex_intensity < height_noise_units * Nf_safe * math.sqrt(apex_intensity):
                continue
            lo, hi = _set_window(
                chrom, apex, Nf,
                max_window=max_window,
                jump_noise_units=jump_noise_units,
                fall_fraction=fall_fraction,
            )
            if hi - lo < 3:
                continue
            baseline = _compute_baseline(chrom, lo, hi)
            apex_height = float(chrom[apex] - baseline[apex - lo])
            if apex_height < height_noise_units * Nf * math.sqrt(max(apex_intensity, 1e-12)):
                continue
            sharpness = _peak_sharpness(chrom, apex, lo, hi, Nf)
            apex_precise = _parabola_apex(chrom, apex)
            out.append(IonPeak(
                ion_index=i,
                apex_scan_int=apex,
                apex_scan_precise=apex_precise,
                apex_intensity=apex_height,
                sharpness=sharpness,
                window_lo=lo,
                window_hi=hi,
                baseline=baseline,
            ))
    return out


def perceive_components(
    chromatograms: np.ndarray,
    *,
    Nf: float = 1.0,
    max_window: int = DEFAULT_DECONV_WINDOW,
    jump_noise_units: float = DEFAULT_JUMP_NOISE_UNITS,
    fall_fraction: float = DEFAULT_FALL_FRACTION,
    height_noise_units: float = DEFAULT_HEIGHT_NOISE_UNITS,
    sharpness_bins_per_scan: int = DEFAULT_SHARPNESS_BINS_PER_SCAN,
    sharpness_range_factor: float = DEFAULT_SHARPNESS_RANGE_FACTOR,
    sharpness_cutoff_ratio: float = DEFAULT_SHARPNESS_CUTOFF_RATIO,
    use_tic_path: bool = True,
    min_range_scans: int = 0,
    inclusion_cutoff_ratio: Optional[float] = None,
    external_ion_peaks: Optional[list["IonPeak"]] = None,
) -> list[Component]:
    """Perceive chromatographic components per AMDIS Stein 1999 §"Component
    Perception".

    Algorithm:
      1. For each ion chromatogram, find local maxima passing the 4-step
         window/baseline/height test (returns IonPeak instances).
      2. Bin each IonPeak's sharpness value into a ``bins_per_scan``-bin
         grid spanning the entire scan axis.
      3. A component is identified at bin ``b`` if the maximum of
         (b-2, b-1, b, b+1, b+2) is at bin ``b``, AND no bin within
         ``range = range_factor / sharpness`` bins has a larger value.
      4. Contributing ions: those whose peak maximizes within the same
         bin range AND has sharpness >= ``cutoff_ratio * max_sharpness``.
      5. **TIC path** (independent): also run the same procedure on the
         TIC. Components perceived only by the TIC path are kept (this
         catches weak components with many minor ions that don't show
         a strong individual maximum but do show a TIC maximum).

    Parameters
    ----------
    chromatograms
        2D array (n_ions, n_scans).
    Nf
        Noise factor (use ``estimate_noise_factor`` upstream).
    min_range_scans
        Lower bound for the ion-inclusion window, expressed in scans.
        ``0`` (default) keeps the original Stein 1999 behavior where the
        window shrinks to ``sharpness_range_factor / sharpness`` bins
        with no floor — fine for the original AMDIS use case but causes
        sharp peaks to produce one component per ion in GC-MS practice.
        Set to e.g. ``apex_window`` (3) to guarantee that all ions
        co-eluting within ~3 scans of the component apex are eligible
        contributors and ``used_bins`` covers the same range so the
        next-sharpest ion of the same physical peak is not perceived as
        a separate component.
    inclusion_cutoff_ratio
        Sharpness ratio (vs the component's dominant sharpness ``s``)
        an ion peak must clear to count as a contributor. ``None``
        (default) falls back to ``sharpness_cutoff_ratio`` (AMDIS 0.75)
        so existing callers behave identically. Passing a lower value
        (e.g. 0.3) lets subsidiary fragment ions with smaller peak
        height — and therefore lower sharpness — be included in the
        component, which is necessary for full-spectrum EI features.

    Returns
    -------
    list[Component]
        Sorted by ascending apex scan.
    """
    arr = np.asarray(chromatograms, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("chromatograms must be 2D")
    n_ions, n_scans = arr.shape
    if n_scans < 3:
        return []

    bins_per_scan = max(1, int(sharpness_bins_per_scan))
    n_bins = n_scans * bins_per_scan
    # Inclusion-window floor: bins corresponding to ``min_range_scans``
    # full scans. Stops a very sharp component from "claiming" only ~2
    # bins, which would (a) exclude its own co-eluting ions and (b)
    # leave the adjacent bins free for the next-sharpest ion to be
    # perceived as an independent component of the same physical peak.
    min_range_bins = (
        max(1, int(min_range_scans) * bins_per_scan)
        if min_range_scans > 0 else 0
    )
    # Inclusion cutoff: separate knob from sharpness_cutoff_ratio so
    # subsidiary fragment ions with naturally lower sharpness can still
    # be assigned to a component without changing the perception logic.
    incl_cutoff = (
        float(inclusion_cutoff_ratio)
        if inclusion_cutoff_ratio is not None
        else float(sharpness_cutoff_ratio)
    )

    if external_ion_peaks is None:
        ion_peaks = _ion_peaks(
            arr, Nf,
            max_window=max_window,
            jump_noise_units=jump_noise_units,
            fall_fraction=fall_fraction,
            height_noise_units=height_noise_units,
        )
    else:
        # ASFAM/LC path: chromatographic peak detection is upstream (ASFAM
        # detect_peaks); these IonPeaks carry precomputed sharpness. Nf is
        # unused here. TIC path must be off (use_tic_path=False) so no
        # _ion_peaks call fires anywhere.
        ion_peaks = external_ion_peaks

    # Bin sharpness for ion-chromatogram peaks.
    bin_max_sharp = np.zeros(n_bins, dtype=np.float64)
    bin_peaks: list[list[IonPeak]] = [[] for _ in range(n_bins)]
    for p in ion_peaks:
        b = int(round(p.apex_scan_precise * bins_per_scan))
        b = max(0, min(n_bins - 1, b))
        bin_peaks[b].append(p)
        if p.sharpness > bin_max_sharp[b]:
            bin_max_sharp[b] = p.sharpness

    components: list[Component] = []
    used_bins: set[int] = set()

    # Sort bins by descending max sharpness so strong components are
    # claimed first.
    sorted_bins = sorted(
        ((bin_max_sharp[b], b) for b in range(n_bins) if bin_max_sharp[b] > 0),
        reverse=True,
    )

    for s, b in sorted_bins:
        if b in used_bins:
            continue
        if s <= 0:
            continue
        # Range of uncertainty: 50 / sharpness bins on each side, floored
        # by ``min_range_bins`` so we never claim less than one physical
        # peak's worth of scans.
        raw_range_bins = int(round(sharpness_range_factor / max(s, 1e-9)))
        range_bins = max(raw_range_bins, min_range_bins)
        lo_b = max(0, b - range_bins)
        hi_b = min(n_bins, b + range_bins + 1)
        # Check if b's bin has the max in (b-2..b+2) AND no greater bin
        # within range_bins.
        local = bin_max_sharp[max(0, b - 2): min(n_bins, b + 3)]
        if local.size == 0 or local.max() > s + 1e-12:
            continue
        window_max = bin_max_sharp[lo_b:hi_b].max() if hi_b > lo_b else 0.0
        if window_max > s + 1e-12:
            continue

        # Contributing ions: those in [lo_b, hi_b) bins with sharpness
        # >= incl_cutoff * s.
        contrib_indices: list[int] = []
        contrib_lo = n_scans
        contrib_hi = 0
        for bb in range(lo_b, hi_b):
            for p in bin_peaks[bb]:
                if p.sharpness >= incl_cutoff * s:
                    contrib_indices.append(p.ion_index)
                    contrib_lo = min(contrib_lo, p.window_lo)
                    contrib_hi = max(contrib_hi, p.window_hi)

        # Mark this and adjacent bins as used so we don't double-perceive.
        for bb in range(lo_b, hi_b):
            used_bins.add(bb)

        if not contrib_indices:
            continue

        components.append(Component(
            apex_scan=b / float(bins_per_scan),
            apex_scan_int=int(round(b / float(bins_per_scan))),
            sharpness=float(s),
            contributing_ions=sorted(set(contrib_indices)),
            window_lo=contrib_lo if contrib_lo < n_scans else 0,
            window_hi=contrib_hi if contrib_hi > 0 else n_scans,
            perceived_via_tic=False,
        ))

    # TIC path.
    if use_tic_path and n_ions > 1:
        tic = arr.sum(axis=0).reshape(1, -1)
        tic_peaks = _ion_peaks(
            tic, Nf,
            max_window=max_window,
            jump_noise_units=jump_noise_units,
            fall_fraction=fall_fraction,
            height_noise_units=height_noise_units,
        )
        # For each TIC peak, check if there is already a component within
        # range_bins bins. If so, skip; otherwise add a TIC-only component.
        for tp in tic_peaks:
            bp = int(round(tp.apex_scan_precise * bins_per_scan))
            bp = max(0, min(n_bins - 1, bp))
            range_bins = int(round(sharpness_range_factor / max(tp.sharpness, 1e-9)))
            tic_lo = max(0, bp - range_bins)
            tic_hi = min(n_bins, bp + range_bins + 1)
            already = any(
                tic_lo <= int(round(c.apex_scan * bins_per_scan)) < tic_hi
                for c in components
            )
            if already:
                continue
            components.append(Component(
                apex_scan=bp / float(bins_per_scan),
                apex_scan_int=int(round(bp / float(bins_per_scan))),
                sharpness=tp.sharpness,
                contributing_ions=[],  # TIC-only; ions filled in later
                window_lo=tp.window_lo,
                window_hi=tp.window_hi,
                perceived_via_tic=True,
            ))

    components.sort(key=lambda c: c.apex_scan)
    return components


# ---------------------------------------------------------------------------
# Phase 3: Model peak construction (Task 3c)
# ---------------------------------------------------------------------------

def build_model_peak(
    component: Component,
    chromatograms: np.ndarray,
    *,
    sharpness_cutoff_ratio: float = DEFAULT_SHARPNESS_CUTOFF_RATIO,
    Nf: float = 1.0,
) -> np.ndarray:
    """Construct the model peak shape for a component (AMDIS sect.
    "Component Perception", final paragraph).

    The model is the sum of the contributing ion chromatograms
    (already filtered to those within range and >= 75% of the max
    sharpness during ``perceive_components``), restricted to the
    union of their deconvolution windows, then normalized so the
    apex value is 1.0.

    For TIC-only components (``perceived_via_tic=True``), the model
    is the TIC restricted to the same window.

    Parameters
    ----------
    component
        Component instance produced by ``perceive_components``.
    chromatograms
        Original 2D chromatogram array used to perceive the component.
    sharpness_cutoff_ratio
        Plumbed through for callers that re-apply the 75% threshold
        for explicit re-filtering. Currently unused (the filtering
        already happened in perceive_components).
    Nf
        Noise factor; reserved for future use.

    Returns
    -------
    np.ndarray
        Model intensity vector of the same length as the chromatograms,
        with non-zero values inside ``[component.window_lo, window_hi)``
        and zeros elsewhere. Normalized so the maximum is 1.0.
    """
    arr = np.asarray(chromatograms, dtype=np.float64)
    n_ions, n_scans = arr.shape
    model = np.zeros(n_scans, dtype=np.float64)

    lo = max(0, component.window_lo)
    hi = min(n_scans, component.window_hi)
    if hi <= lo:
        return model

    if component.perceived_via_tic:
        # Use TIC inside the window.
        model[lo:hi] = arr[:, lo:hi].sum(axis=0)
    else:
        if not component.contributing_ions:
            return model
        # Sum the contributing ions in the window.
        for i in component.contributing_ions:
            if 0 <= i < n_ions:
                model[lo:hi] += arr[i, lo:hi]

    max_val = float(np.max(model))
    if max_val <= 0:
        return np.zeros(n_scans, dtype=np.float64)
    return model / max_val


# ---------------------------------------------------------------------------
# Phase 4: Per-m/z least-squares fit with adjacent subtraction (Task 3d)
# ---------------------------------------------------------------------------

def _fit_per_mz(
    A: np.ndarray,
    M: np.ndarray,
    neighbors: list[np.ndarray],
) -> tuple[float, float]:
    """Solve A(n) = a + b*n + c*M(n) [+ d*Y(n) + e*Z(n) ...] for a single
    m/z and return ``(c, residual_norm)``.

    ``c`` is the deconvolved abundance contribution at the apex of M
    (since M is normalized so M_max = 1, the model contribution to A at
    the apex is exactly c). On failure (singular system, or c <= 0),
    returns (0.0, +inf).
    """
    n = A.size
    if n < 2:
        return 0.0, float("inf")
    cols = [np.ones(n), np.arange(n, dtype=np.float64), M]
    for Y in neighbors[:2]:  # AMDIS: at most 2 explicitly subtracted
        cols.append(Y)
    X = np.column_stack(cols)
    try:
        sol, _residuals, rank, _sv = np.linalg.lstsq(X, A, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, float("inf")
    if sol.size < 3:
        return 0.0, float("inf")
    c = float(sol[2])
    if not math.isfinite(c) or c <= 0:
        return 0.0, float("inf")
    fitted = X @ sol
    residual = float(np.linalg.norm(A - fitted))
    return c, residual


def deconvolve_spectrum(
    target_component: Component,
    chromatograms: np.ndarray,
    mz_values: np.ndarray,
    neighbor_components: Sequence[Component] = (),
    *,
    sharpness_cutoff_ratio: float = DEFAULT_SHARPNESS_CUTOFF_RATIO,
    Nf: float = 1.0,
) -> DeconvolvedSpectrum:
    """Deconvolve the spectrum for ``target_component`` using AMDIS Stein
    1999 §"Deconvolution".

    For each m/z (each row in ``chromatograms``), solve

        A(n) = a + b*n + c*M(n) + d*Y(n) + e*Z(n)

    via ordinary least squares (numpy.linalg.lstsq → SVD/QR). M is the
    target component's normalized model peak, Y/Z are neighbor models
    (at most 2; AMDIS upper bound). The deconvolved abundance at m/z is
    ``c * M(n_max) = c`` (since M is normalized so M_max = 1).

    Both with-subtraction and without-subtraction spectra are produced,
    per Stein 1999. Callers (in gcms_match_factor) can choose which one
    to use; AMDIS itself feeds both into the match factor and lets the
    score pick.

    Parameters
    ----------
    target_component
        Component whose spectrum we want.
    chromatograms
        2D array (n_ions, n_scans) of intensities (one row per m/z).
    mz_values
        1D array of m/z values, same length as ``chromatograms.shape[0]``.
    neighbor_components
        Other components whose model peaks should be subtracted (up to 2).
    sharpness_cutoff_ratio, Nf
        Reserved for future use; currently unused in the per-m/z fit.

    Returns
    -------
    DeconvolvedSpectrum
        Contains both the with-subtraction (``intensity``) and the
        without-subtraction (``intensity_no_subtraction``) spectra.
    """
    arr = np.asarray(chromatograms, dtype=np.float64)
    n_ions, n_scans = arr.shape
    mz_arr = np.asarray(mz_values, dtype=np.float64)
    if mz_arr.size != n_ions:
        raise ValueError("mz_values length must match chromatograms.shape[0]")

    lo = max(0, target_component.window_lo)
    hi = min(n_scans, target_component.window_hi)
    if hi <= lo:
        return DeconvolvedSpectrum(
            mz=mz_arr.copy(),
            intensity=np.zeros(n_ions),
            intensity_no_subtraction=np.zeros(n_ions),
            flags=np.zeros(n_ions, dtype=bool),
            n_adjacent_subtracted=0,
            apex_scan=target_component.apex_scan_int,
        )

    M_full = build_model_peak(target_component, arr)
    M_window = M_full[lo:hi]
    if not np.any(M_window > 0):
        return DeconvolvedSpectrum(
            mz=mz_arr.copy(),
            intensity=np.zeros(n_ions),
            intensity_no_subtraction=np.zeros(n_ions),
            flags=np.zeros(n_ions, dtype=bool),
            n_adjacent_subtracted=0,
            apex_scan=target_component.apex_scan_int,
        )

    # Build neighbor model arrays for the subtraction step.
    neighbor_models: list[np.ndarray] = []
    for nc in list(neighbor_components)[:2]:
        Y = build_model_peak(nc, arr)
        Y_window = Y[lo:hi]
        # Drop neighbors that are silent in this window.
        if np.any(Y_window > 0):
            neighbor_models.append(Y_window)
    n_adj = len(neighbor_models)

    intens_with = np.zeros(n_ions, dtype=np.float64)
    intens_no = np.zeros(n_ions, dtype=np.float64)

    for i in range(n_ions):
        A = arr[i, lo:hi]
        if not np.any(A > 0):
            continue
        # No-subtraction fit.
        c_no, _ = _fit_per_mz(A, M_window, [])
        intens_no[i] = c_no
        if n_adj == 0:
            intens_with[i] = c_no
        else:
            c_with, _ = _fit_per_mz(A, M_window, neighbor_models)
            if c_with > 0:
                intens_with[i] = c_with
            else:
                # Fallback: use the no-subtraction value.
                intens_with[i] = c_no

    flags = _flag_peaks(
        arr=arr,
        mz_arr=mz_arr,
        lo=lo, hi=hi,
        intens_with=intens_with,
        intens_no=intens_no,
        M_window=M_window,
        n_adjacent_subtracted=n_adj,
        Nf=Nf,
    )

    return DeconvolvedSpectrum(
        mz=mz_arr.copy(),
        intensity=intens_with,
        intensity_no_subtraction=intens_no,
        flags=flags,
        n_adjacent_subtracted=n_adj,
        apex_scan=target_component.apex_scan_int,
    )


# ---------------------------------------------------------------------------
# Phase 5: Peak flagging (Task 3e)
# ---------------------------------------------------------------------------

# AMDIS Stein 1999 §"Criteria for peak flagging and rejection".
DEFAULT_FM_FLAG_THRESHOLD = 0.2     # FM > 0.2 → flag (eq 5)
DEFAULT_FM_REJECT_THRESHOLD = 0.6   # FM > 0.6 → reject
DEFAULT_FRACTION_AFTER_SUBTRACTION = 0.10  # < 10% of total extracted → flag
DEFAULT_MIN_S_TO_N = 2.0            # peaks with S/N < 2 → flag


def _flag_peaks(
    arr: np.ndarray,
    mz_arr: np.ndarray,
    lo: int,
    hi: int,
    intens_with: np.ndarray,
    intens_no: np.ndarray,
    M_window: np.ndarray,
    n_adjacent_subtracted: int,
    Nf: float,
    *,
    fm_flag_threshold: float = DEFAULT_FM_FLAG_THRESHOLD,
    fm_reject_threshold: float = DEFAULT_FM_REJECT_THRESHOLD,
    fraction_after_subtraction: float = DEFAULT_FRACTION_AFTER_SUBTRACTION,
    min_s_to_n: float = DEFAULT_MIN_S_TO_N,
) -> np.ndarray:
    """Compute the per-m/z flag mask per AMDIS Stein 1999 §"Criteria for
    peak flagging and rejection".

    Returns
    -------
    np.ndarray (bool, shape (n_ions,))
        True for flagged peaks. Rejected peaks (FM > 0.6) are also set
        to True; their intensity is zeroed in-place inside ``intens_with``
        and ``intens_no`` so callers don't need to re-check.
    """
    n_ions = arr.shape[0]
    flags = np.zeros(n_ions, dtype=bool)

    # Normalize the model so sum(M) = 1 over the window.
    if M_window.sum() > 0:
        M_norm = M_window / M_window.sum()
    else:
        return flags

    total_extracted = float(np.sum(intens_with))
    Nf_safe = max(Nf, 1e-9)

    for i in range(n_ions):
        c = float(intens_with[i])
        if c <= 0:
            continue
        signal = arr[i, lo:hi]
        # Normalize the extracted signal contribution: I(n) = c * M(n).
        # Per AMDIS we look at "signal that didn't match the model".
        sig_sum = float(np.sum(signal))
        if sig_sum <= 0:
            continue
        I_norm = signal / sig_sum
        FM = float(np.sum(np.abs(I_norm - M_norm))) / 2.0
        # Empirical adjustment for weak signals (eq 6).
        # 20 / (sum_A^0.5/Nf + 20). The ¥A1/2/Nf term measures
        # deviation from the model in noise units; for big signals this
        # term is small and FM threshold remains 0.2.
        sum_A = float(np.sum(np.abs(signal - c * M_window)))
        if sum_A > 0:
            adjustment = 20.0 / (math.sqrt(sum_A) / Nf_safe + 20.0)
        else:
            adjustment = 0.0
        threshold = fm_flag_threshold + adjustment

        if FM > fm_reject_threshold:
            # Reject: zero out intensity and flag.
            intens_with[i] = 0.0
            intens_no[i] = 0.0
            flags[i] = True
            continue
        if FM > threshold:
            flags[i] = True

        # Fraction of extracted abundance (only matters when neighbors
        # were subtracted).
        if n_adjacent_subtracted > 0 and total_extracted > 0:
            if c < fraction_after_subtraction * total_extracted:
                flags[i] = True

        # S/N: extracted intensity vs noise level. AMDIS uses
        # noise = Nf * sqrt(c) (ion-counting noise at level c).
        noise_at_c = Nf_safe * math.sqrt(max(c, 1e-12))
        if noise_at_c > 0 and c / noise_at_c < min_s_to_n:
            flags[i] = True

        # Possible noise spike: if the peak at the apex is adjacent to
        # zero-abundance scans (in the original chromatogram), AMDIS
        # flags it when the peak occurrence probability is > 0.1. We
        # approximate as: if either neighbor of the apex inside the
        # window is exactly 0, flag.
        apex_in_window = lo + int(np.argmax(M_window))
        if apex_in_window > lo and arr[i, apex_in_window - 1] == 0:
            flags[i] = True
        if apex_in_window < hi - 1 and arr[i, apex_in_window + 1] == 0:
            flags[i] = True

    return flags


# ---------------------------------------------------------------------------
# Phase 6: High-level entry — deconvolve_features (Task 3f)
# ---------------------------------------------------------------------------

@dataclass
class DeconvolutionConfig:
    """Configuration for the high-level deconvolution entry point.

    Defaults follow AMDIS Stein 1999.
    """
    max_window: int = DEFAULT_DECONV_WINDOW
    jump_noise_units: float = DEFAULT_JUMP_NOISE_UNITS
    fall_fraction: float = DEFAULT_FALL_FRACTION
    height_noise_units: float = DEFAULT_HEIGHT_NOISE_UNITS
    sharpness_bins_per_scan: int = DEFAULT_SHARPNESS_BINS_PER_SCAN
    sharpness_range_factor: float = DEFAULT_SHARPNESS_RANGE_FACTOR
    sharpness_cutoff_ratio: float = DEFAULT_SHARPNESS_CUTOFF_RATIO
    fm_flag_threshold: float = DEFAULT_FM_FLAG_THRESHOLD
    fm_reject_threshold: float = DEFAULT_FM_REJECT_THRESHOLD
    fraction_after_subtraction: float = DEFAULT_FRACTION_AFTER_SUBTRACTION
    min_s_to_n: float = DEFAULT_MIN_S_TO_N
    use_tic_path: bool = True
    max_neighbors_subtracted: int = 2  # AMDIS upper bound
    # Inclusion-window floor (in scans) for perceive_components. See the
    # function docstring for the motivation. ``0`` keeps original AMDIS
    # behavior; GC-MS callers typically pass ``apex_window`` (3).
    min_range_scans: int = 0
    # Sharpness cutoff for ion inclusion (vs the component's dominant
    # sharpness). ``None`` falls back to ``sharpness_cutoff_ratio``;
    # GC-MS callers typically pass 0.3 so subsidiary fragment ions are
    # not filtered out.
    inclusion_cutoff_ratio: Optional[float] = None


def build_chromatograms_from_scans(
    scans: Sequence,
    *,
    mz_tol: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a 2D (n_ions, n_scans) chromatogram array from a list of
    scan-like objects each having ``mz_array`` and ``intensity_array``.

    All scans are pooled into a unified m/z grid by quantizing m/z to
    ``mz_tol`` buckets. The returned ``mz_values`` array gives the
    bucket-canonical m/z (first-seen exact value) for each row.

    Plan D follow-up #2 (2026-04-29): the previous implementation ran two
    full nested Python loops over (scan, m/z) — the dominant fullscan
    bottleneck (passion_fruit.mzML ~14000 scans × ~500 m/z = 7M
    Python-level iterations, exceeding 3 minutes). This version is fully
    vectorized via concat → ``np.unique(return_index=True)`` →
    ``np.searchsorted`` → ``np.maximum.at``. Output is byte-identical to
    the old implementation on the same input.

    Returns
    -------
    (chromatograms, mz_values)
        chromatograms : ndarray (n_ions, n_scans), float64.
        mz_values     : ndarray (n_ions,), float64, sorted ascending
                        (sorted by bucket, which is monotone in m/z).
    """
    if mz_tol <= 0:
        raise ValueError("mz_tol must be positive")
    n_scans = len(scans)
    if n_scans == 0:
        return np.zeros((0, 0), dtype=np.float64), np.zeros(0, dtype=np.float64)

    # Flatten (scan_idx, mz, intensity) triples across all scans. The Python
    # loop here is unavoidable because each scan can have a variable-size
    # m/z array; everything below this loop is vectorized.
    scan_idx_chunks: list[np.ndarray] = []
    mz_chunks: list[np.ndarray] = []
    int_chunks: list[np.ndarray] = []
    for s, scan in enumerate(scans):
        mz_arr = np.asarray(scan.mz_array, dtype=np.float64)
        if mz_arr.size == 0:
            continue
        int_arr = np.asarray(scan.intensity_array, dtype=np.float64)
        scan_idx_chunks.append(np.full(mz_arr.size, s, dtype=np.int64))
        mz_chunks.append(mz_arr)
        int_chunks.append(int_arr)

    if not mz_chunks:
        return (
            np.zeros((0, n_scans), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )

    flat_scan_idx = np.concatenate(scan_idx_chunks)
    flat_mz = np.concatenate(mz_chunks)
    flat_int = np.concatenate(int_chunks)

    # Quantize to buckets. ``round`` matches the previous implementation.
    flat_buckets = np.round(flat_mz / mz_tol).astype(np.int64)

    # ``np.unique`` returns sorted unique buckets and the index of the
    # first occurrence of each bucket in the input. Because buckets are a
    # monotone function of m/z, sorting by bucket = sorting by m/z, so
    # ``mz_values`` is naturally ascending. The "first occurrence" index
    # also matches the old per-scan-then-per-mz iteration order: scans
    # are concatenated in order, and within each scan m/z values keep
    # their original array order.
    unique_buckets, first_idx = np.unique(flat_buckets, return_index=True)
    mz_values = flat_mz[first_idx]
    n_ions = unique_buckets.size

    # Vectorized scatter-max: for duplicate (row, scan) pairs, take the max
    # intensity. Matches the old "if inten > chroms[row, s]: chroms[row, s] = inten".
    rows = np.searchsorted(unique_buckets, flat_buckets)
    chroms = np.zeros((n_ions, n_scans), dtype=np.float64)
    np.maximum.at(chroms, (rows, flat_scan_idx), flat_int)

    return chroms, mz_values


def deconvolve_features(
    features: Sequence[dict],
    scans: Sequence,
    *,
    config: Optional[DeconvolutionConfig] = None,
    mz_tol: float = 0.02,
    feature_id_key: str = "feature_id",
    apex_key: str = "apex_index",
    use_perceive_components: bool = False,
) -> dict:
    """High-level entry point: deconvolve the spectrum for each feature.

    W9 重构 (用户拍板):

    deconvolve **没有** 自己的 peak detection — sharpness 算的是已存在
    峰顶的锐度。stage_features 已经在每个 EIC channel 上做过真正的峰
    检测, 它的输出 (features) 应该直接做为 deconvolve 的目标列表, 不
    需要再用 ``perceive_components`` 重做一遍组分聚类。

    因此默认行为现在是:
      1. 对每个 input feature, 用它的 apex_index 直接构造一个 synthetic
         Component (无 sharpness 步骤)。
      2. 邻居成分也从 features 列表中拿 — 离当前 feature apex 最近的
         其他 features (跳过自身), 限制最多 ``max_neighbors_subtracted``。
      3. 跑 ``deconvolve_spectrum`` 做 per-m/z least-squares +
         adjacent-component subtraction + flagging。

    历史路径 ``use_perceive_components=True`` 仍可用 — AMDIS 原汁原味,
    会在 chroms 上跑 ``perceive_components`` 来"再次发现"组分。不推荐
    在 stage_features 已经定位过峰的流水线里用。

    Parameters
    ----------
    features
        Iterable of feature dicts. Each must have ``feature_id_key`` and
        ``apex_key`` fields.
    scans
        Sequence of scan-like objects (``mz_array``, ``intensity_array``).
    config
        Deconvolution config; uses defaults if None.
    mz_tol
        m/z tolerance for chromatogram bucketing.
    feature_id_key, apex_key
        Field names in the feature dicts.
    use_perceive_components
        Default False (W9). If True, fall back to the legacy AMDIS-style
        sharpness-based perception. Kept for compatibility / debugging.

    Returns
    -------
    dict[feature_id, DeconvolvedSpectrum]
        One entry per input feature.
    """
    if config is None:
        config = DeconvolutionConfig()

    if not features or not scans:
        return {}

    # Build the chromatograms and m/z values from all scans.
    chroms, mz_values = build_chromatograms_from_scans(scans, mz_tol=mz_tol)
    if chroms.size == 0:
        return {fid: _empty_spectrum() for fid in
                (f[feature_id_key] for f in features)}

    # Estimate Nf once per file.
    Nf = estimate_noise_factor(list(chroms), include_tic=True)
    # Replace zeros via threshold-transition correction.
    chroms = np.stack(
        [replace_threshold_transitions(chrom) for chrom in chroms],
        axis=0,
    )

    if use_perceive_components:
        # 历史路径: AMDIS sharpness-based component perception.
        components = perceive_components(
            chroms, Nf=Nf,
            max_window=config.max_window,
            jump_noise_units=config.jump_noise_units,
            fall_fraction=config.fall_fraction,
            height_noise_units=config.height_noise_units,
            sharpness_bins_per_scan=config.sharpness_bins_per_scan,
            sharpness_range_factor=config.sharpness_range_factor,
            sharpness_cutoff_ratio=config.sharpness_cutoff_ratio,
            use_tic_path=config.use_tic_path,
        )
        return _deconvolve_via_perception(
            features, chroms, mz_values, Nf, components, config,
            feature_id_key=feature_id_key, apex_key=apex_key,
        )

    # W9 默认路径: feature-driven, 不做 sharpness 组分感知。
    return _deconvolve_via_features(
        features, chroms, mz_values, Nf, config,
        feature_id_key=feature_id_key, apex_key=apex_key,
    )


def _deconvolve_via_features(
    features: Sequence[dict],
    chroms: np.ndarray,
    mz_values: np.ndarray,
    Nf: float,
    config: DeconvolutionConfig,
    *,
    feature_id_key: str,
    apex_key: str,
) -> dict:
    """W9 默认路径: 每个 feature → 直接 synthetic target Component;
    邻居取最近的其他 feature。跳过 ``perceive_components``。"""
    # 把每个 feature 的 apex 都做成一个 synthetic Component。这样每个
    # feature 都既能做 target 又能做邻居。
    feat_components: list[tuple[str, int, Component]] = []
    for feat in features:
        fid = feat[feature_id_key]
        apex_index = int(feat[apex_key])
        comp = _synthetic_target(apex_index, chroms, Nf, config)
        feat_components.append((fid, apex_index, comp))

    out: dict = {}
    for fid_t, apex_t, target in feat_components:
        # 邻居: 其他 feature 的 synthetic Component, 按 apex 距离排序,
        # 只取真正与 target.window 有重叠的, 上限 max_neighbors_subtracted。
        neighbor_pool = sorted(
            (item for item in feat_components if item[0] != fid_t),
            key=lambda it: abs(it[1] - apex_t),
        )
        neighbors: list[Component] = []
        for _, _, nc in neighbor_pool:
            if nc.window_hi <= target.window_lo or nc.window_lo >= target.window_hi:
                continue
            neighbors.append(nc)
            if len(neighbors) >= config.max_neighbors_subtracted:
                break
        decon = deconvolve_spectrum(
            target, chroms, mz_values,
            neighbor_components=neighbors,
            sharpness_cutoff_ratio=config.sharpness_cutoff_ratio,
            Nf=Nf,
        )
        out[fid_t] = decon
    return out


def _deconvolve_via_perception(
    features: Sequence[dict],
    chroms: np.ndarray,
    mz_values: np.ndarray,
    Nf: float,
    components: list["Component"],
    config: DeconvolutionConfig,
    *,
    feature_id_key: str,
    apex_key: str,
) -> dict:
    """历史路径: 用 perceived components 当邻居池, 必要时 fallback 到
    synthetic target。保留用于调试 / 与 AMDIS 原版做参照对比。"""
    out: dict = {}
    for feat in features:
        fid = feat[feature_id_key]
        apex_index = int(feat[apex_key])
        candidates = sorted(
            components,
            key=lambda c: abs(c.apex_scan_int - apex_index),
        )
        target = None
        if candidates and abs(candidates[0].apex_scan_int - apex_index) <= 2:
            target = candidates[0]
            neighbor_pool = candidates[1:]
        else:
            target = _synthetic_target(apex_index, chroms, Nf, config)
            neighbor_pool = candidates
        neighbors: list[Component] = []
        for nc in neighbor_pool:
            if nc is target:
                continue
            if nc.window_hi <= target.window_lo or nc.window_lo >= target.window_hi:
                continue
            neighbors.append(nc)
            if len(neighbors) >= config.max_neighbors_subtracted:
                break
        decon = deconvolve_spectrum(
            target, chroms, mz_values,
            neighbor_components=neighbors,
            sharpness_cutoff_ratio=config.sharpness_cutoff_ratio,
            Nf=Nf,
        )
        out[fid] = decon
    return out


def _synthetic_target(
    apex_index: int,
    chromatograms: np.ndarray,
    Nf: float,
    config: DeconvolutionConfig,
) -> Component:
    """Build a Component for an apex index that didn't show up in
    ``perceive_components`` (typically a feature passed in from upstream
    peak detection).

    The model uses the strongest ion at the apex as the contributing ion
    and the AMDIS step (a) window setup.
    """
    n_ions, n_scans = chromatograms.shape
    apex = max(0, min(n_scans - 1, apex_index))
    intensities_at_apex = chromatograms[:, apex]
    if intensities_at_apex.max() <= 0:
        # No signal: zero-window component.
        return Component(
            apex_scan=float(apex), apex_scan_int=apex, sharpness=0.0,
            contributing_ions=[], window_lo=apex, window_hi=apex,
        )
    top_ion = int(np.argmax(intensities_at_apex))
    lo, hi = _set_window(
        chromatograms[top_ion], apex, Nf,
        max_window=config.max_window,
        jump_noise_units=config.jump_noise_units,
        fall_fraction=config.fall_fraction,
    )
    # Allow up to 2 contributing ions: the top one plus any other ion
    # whose apex is within ±1 scan and intensity >= 50% of the top.
    contrib = [top_ion]
    apex_intens = intensities_at_apex[top_ion]
    for i in range(n_ions):
        if i == top_ion:
            continue
        local = chromatograms[i, max(0, apex - 1): min(n_scans, apex + 2)]
        if local.size == 0:
            continue
        local_max = float(local.max())
        if local_max >= 0.5 * apex_intens:
            contrib.append(i)
    return Component(
        apex_scan=float(apex), apex_scan_int=apex, sharpness=1.0,
        contributing_ions=contrib,
        window_lo=lo, window_hi=hi,
    )


def _empty_spectrum() -> DeconvolvedSpectrum:
    return DeconvolvedSpectrum(
        mz=np.zeros(0),
        intensity=np.zeros(0),
        intensity_no_subtraction=np.zeros(0),
        flags=np.zeros(0, dtype=bool),
        n_adjacent_subtracted=0,
        apex_scan=0,
    )


def deconvolved_to_peaks(
    spectrum: DeconvolvedSpectrum,
    *,
    use_subtraction: bool = True,
    rel_intensity_floor: float = 0.0,
    top_n: int = 0,
) -> list[tuple[float, float]]:
    """Convert a ``DeconvolvedSpectrum`` to a sorted ``[(mz, intensity), ...]``
    list suitable for downstream library matching.

    Parameters
    ----------
    spectrum
        The deconvolved spectrum to convert.
    use_subtraction
        If True (default), use the with-subtraction intensity. If False,
        use the no-subtraction variant.
    rel_intensity_floor
        Drop ions with intensity < ``rel_intensity_floor * max_intensity``.
    top_n
        If > 0, keep only the top-N most intense ions.
    """
    peaks, _flags = deconvolved_to_peaks_with_flags(
        spectrum,
        use_subtraction=use_subtraction,
        rel_intensity_floor=rel_intensity_floor,
        top_n=top_n,
    )
    return peaks


def representative_mz(
    spectrum: DeconvolvedSpectrum,
    *,
    threshold_ratio: float = 0.05,
    use_subtraction: bool = True,
) -> float:
    """Return the heaviest non-flagged m/z whose intensity is at least
    ``threshold_ratio * base_peak_intensity`` in this deconvolved spectrum.

    This is the per-feature value used by the GC-MS Feature Overview
    scatter Y-axis. The motivation: the base-peak m/z (= ``quant_mass``)
    is often a small fragment ion that does not reflect the analyte's
    molecular weight. The heaviest peak that is still above a small
    fraction of the base peak (default 5%) is a much better visual proxy
    for molecular weight.

    Fall-back order:
      1. If the spectrum is empty (size 0), return NaN to flag missing data.
      2. If every peak is flagged, the result falls back to the base peak
         m/z (the m/z with the largest intensity, ignoring flags).
      3. Otherwise: among the non-flagged peaks whose intensity is at
         least ``threshold_ratio * max_intensity``, return the heaviest
         m/z. If none qualify (e.g. all non-flagged peaks are very weak),
         the result is the heaviest non-flagged peak regardless of
         intensity.

    Parameters
    ----------
    spectrum
        A ``DeconvolvedSpectrum`` instance.
    threshold_ratio
        The minimum intensity (relative to the base peak) for a peak to
        be eligible as the representative. Default 0.05 = 5%.
    use_subtraction
        If True (default), use ``spectrum.intensity`` (with-subtraction).
        If False, use ``spectrum.intensity_no_subtraction``.
    """
    intens = spectrum.intensity if use_subtraction else spectrum.intensity_no_subtraction
    mzs = spectrum.mz
    flags = spectrum.flags
    if intens.size == 0:
        return float("nan")
    max_int = float(intens.max())
    if max_int <= 0:
        return float("nan")

    # Eligible: non-flagged AND intensity >= threshold_ratio * base_peak.
    threshold_value = float(threshold_ratio) * max_int
    eligible_mask = (~flags) & (intens >= threshold_value)
    if np.any(eligible_mask):
        return float(mzs[eligible_mask].max())

    # Fallback: heaviest non-flagged peak regardless of intensity.
    nonflagged_mask = ~flags
    if np.any(nonflagged_mask):
        return float(mzs[nonflagged_mask].max())

    # Final fallback: all peaks are flagged — return the base-peak m/z
    # (= ``quant_mass`` equivalent).
    base_idx = int(np.argmax(intens))
    return float(mzs[base_idx])


def deconvolved_to_peaks_with_flags(
    spectrum: DeconvolvedSpectrum,
    *,
    use_subtraction: bool = True,
    rel_intensity_floor: float = 0.0,
    top_n: int = 0,
) -> tuple[list[tuple[float, float]], list[bool]]:
    """Convert a ``DeconvolvedSpectrum`` to a peaks list AND an
    index-aligned flag list, in a single filter+sort pass.

    This is the preferred entry point when downstream code needs both the
    peaks and AMDIS-style per-peak flags. The previous workflow of calling
    ``deconvolved_to_peaks`` and then re-aligning flags via
    ``mz_arr.index(mz)`` was fragile against any float perturbation of
    the m/z axis. Here we carry indices through the filter chain and
    project flags out at the end.

    Returns
    -------
    (peaks, flags) : tuple
        ``peaks`` is the same shape as ``deconvolved_to_peaks``. ``flags``
        is a list of bool, the same length as ``peaks``, with True meaning
        AMDIS-flagged (uncertain) for that peak.
    """
    intens = spectrum.intensity if use_subtraction else spectrum.intensity_no_subtraction
    mzs = spectrum.mz
    flags = spectrum.flags
    if intens.size == 0:
        return [], []
    max_int = float(intens.max())
    if max_int <= 0:
        return [], []

    # Carry the original index through the filter chain so flags remain
    # aligned. The triples are (mz, intensity, idx); we project flags at
    # the very end.
    triples: list[tuple[float, float, int]] = [
        (float(mz), float(i), idx)
        for idx, (mz, i) in enumerate(zip(mzs, intens))
        if i > 0
    ]
    if rel_intensity_floor > 0:
        threshold = rel_intensity_floor * max_int
        triples = [(m, v, idx) for (m, v, idx) in triples if v >= threshold]
    if top_n > 0 and len(triples) > top_n:
        triples.sort(key=lambda t: -t[1])
        triples = triples[:top_n]
    triples.sort(key=lambda t: t[0])

    peaks = [(m, v) for (m, v, _) in triples]
    flags_out = [bool(flags[idx]) for (_, _, idx) in triples]
    return peaks, flags_out
