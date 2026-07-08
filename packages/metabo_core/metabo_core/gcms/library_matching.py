"""GC-MS library-matching strategies (fullscan + cSIM) and Plan D match factor.

This module implements:

* ``fullscan_cosine``: standard cosine between measured spectrum and the
  full reference spectrum.
* ``csim_intersected_cosine``: intersect the reference spectrum with the
  acquired-ion set first, then compute cosine.
* ``gcms_match_factor`` (Plan D): AMDIS / MS-DIAL style composite match
  factor that combines weighted dot product, reverse dot product, simple
  dot product, and matched-fraction into a single spectral score, with
  optional RT or RI Gaussian similarity multiplied on top.

The acquired-ion set is recovered **from the data itself**, not from any
acquisition method file. This rule is inherited from the legacy WTV2
Data Analyzer:

1. Take a small RT window around the candidate peak's apex
   (default ± 3 scans, configurable).
2. For every scan in that window, collect every m/z whose intensity > 0.
3. The union of those m/z is the "acquired-ion set" for that RT.

Why no method file:

- Unscanned points are exactly zero; scanned-but-quiet points still have
  noise > 0, so the data itself is sufficient to recover what was acquired.
- Some vendors (e.g. Thermo) encrypt their method files, so reading the
  method is not always possible.
- Avoiding method-file reads keeps the cSIM workflow uniform across vendors.

See also: ``CLAUDE.md`` "GC-MS guidance" and the design doc
``docs/superpowers/specs/2026-04-28-metabo-platform-multi-app-near-term-design.md``
section 4.2.1.
"""
from __future__ import annotations

import math
from typing import Iterable, Literal, Optional, Protocol, Sequence

import numpy as np

from metabo_core.algorithms.similarity import (
    _compute_three_scores,
    _match_peaks,
    cosine_similarity,
    gaussian_similarity,
    reverse_dot_product,
    simple_dot_product,
    weighted_dot_product,
)


class _ScanLike(Protocol):
    """Anything with ``mz_array`` and ``intensity_array`` numpy attributes.

    Both the GC-MS app's ``GcmsScan`` dataclass and ad-hoc test doubles
    satisfy this. The function does not import the app type so that
    metabo_core's boundary is preserved.
    """
    mz_array: np.ndarray
    intensity_array: np.ndarray


def acquired_ion_set(
    scans: Sequence[_ScanLike],
    apex_index: int,
    window: int = 3,
    mz_tol: float = 0.01,
) -> list[float]:
    """Return the union of m/z (rounded to ``mz_tol``) where intensity > 0
    across ``scans[apex_index - window : apex_index + window + 1]``.

    Out-of-range bounds are clamped to ``[0, len(scans))``.

    The deduplication is done by quantizing m/z to a multiple of ``mz_tol``
    so values within tolerance collapse to the same bucket. The returned
    m/z values are the bucket centers (sorted ascending).
    """
    if not scans:
        return []
    if mz_tol <= 0:
        raise ValueError("mz_tol must be positive")

    n = len(scans)
    lo = max(0, apex_index - window)
    hi = min(n, apex_index + window + 1)

    # 把窗口内所有 (intensity>0) 离子按"扫描顺序 + 扫描内顺序"拼接一次,
    # 再用 np.unique(return_index=True) 取每个 bucket 的"首次出现 m/z"。
    # 等价于原来的 Python 循环逐个判 `bucket not in dict`, 但避免双层 loop。
    mz_pieces: list[np.ndarray] = []
    for s in range(lo, hi):
        scan = scans[s]
        mz_arr = np.asarray(scan.mz_array, dtype=np.float64)
        int_arr = np.asarray(scan.intensity_array, dtype=np.float64)
        if mz_arr.size == 0:
            continue
        mask = int_arr > 0
        if not np.any(mask):
            continue
        mz_pieces.append(mz_arr[mask])

    if not mz_pieces:
        return []

    all_mz = np.concatenate(mz_pieces)
    if all_mz.size == 0:
        return []
    buckets = np.round(all_mz / mz_tol).astype(np.int64)
    # np.unique(return_index=True) 返回每个唯一 bucket 在 buckets 中
    # 的"首次出现下标" — 与 dict[bucket]=mz 的 first-seen 语义一致。
    _, first_idx = np.unique(buckets, return_index=True)
    canonical = all_mz[first_idx]
    canonical.sort()
    return canonical.tolist()


def fullscan_cosine(
    measured: list[tuple[float, float]],
    reference: list[tuple[float, float]],
    mz_tol: float = 0.01,
) -> tuple[float, int]:
    """Standard cosine between measured spectrum and the full reference
    spectrum, with no intersection step.

    Returns ``(score, n_matched_peaks)`` where the score is in ``[0, 1]``.
    """
    return cosine_similarity(measured, reference, mz_tol)


def csim_intersected_cosine(
    measured: list[tuple[float, float]],
    reference: list[tuple[float, float]],
    acquired_set: Iterable[float],
    mz_tol: float = 0.01,
) -> tuple[float, int]:
    """Intersect the reference with the acquired-ion set, then compute cosine
    against the measured spectrum.

    Parameters
    ----------
    measured
        ``[(mz, intensity), ...]`` from the candidate peak's apex spectrum.
    reference
        ``[(mz, intensity), ...]`` from the spectral library entry.
    acquired_set
        Iterable of m/z values that were actually acquired around the apex
        (typically computed by :func:`acquired_ion_set`).
    mz_tol
        m/z tolerance, used both for the intersection and for the inner
        cosine peak-matching step.

    Returns
    -------
    ``(score, n_matched_peaks)`` where the score is in ``[0, 1]``.
    """
    acquired_arr = np.asarray(sorted(set(float(m) for m in acquired_set)), dtype=np.float64)
    if acquired_arr.size == 0:
        return 0.0, 0

    if not reference:
        return 0.0, 0

    # 用 searchsorted 把 O(N_ref * N_acq) 的 np.min(np.abs(...)) 替换为
    # O((N_ref + N_acq) log N_acq); acquired_arr 已经升序排序。
    ref_mz_arr = np.fromiter(
        (float(mz) for mz, _ in reference),
        dtype=np.float64,
        count=len(reference),
    )
    insert_pos = np.searchsorted(acquired_arr, ref_mz_arr)
    n_acq = acquired_arr.size
    left_idx = np.clip(insert_pos - 1, 0, n_acq - 1)
    right_idx = np.clip(insert_pos, 0, n_acq - 1)
    dist_left = np.abs(acquired_arr[left_idx] - ref_mz_arr)
    dist_right = np.abs(acquired_arr[right_idx] - ref_mz_arr)
    keep_mask = np.minimum(dist_left, dist_right) <= mz_tol

    intersected_ref: list[tuple[float, float]] = [
        (float(ref_mz_arr[i]), float(reference[i][1]))
        for i in np.flatnonzero(keep_mask)
    ]

    if not intersected_ref:
        return 0.0, 0

    return cosine_similarity(measured, intersected_ref, mz_tol)


# ---------------------------------------------------------------------------
# Plan D: GC-MS composite match factor
# ---------------------------------------------------------------------------

ChromMode = Literal["none", "rt", "ri"]


def _spectrum_complexity_factor(
    spectrum: list[tuple[float, float]],
    a: float = 0.5,
) -> float:
    """AMDIS Stein 1999 spectrum complexity scaling, eqs (8) and (9).

    The match-factor formula multiplies each peak abundance by
    ``1 / (1 + w * A)`` where ``w = 1 / (a + sum(A) - 1)`` and abundances
    are normalized so the base peak is 1.0. The dominant-peak case
    (``sum ≈ 1``) reduces the dominant peak by a factor of 3 with
    ``a=0.5``; multi-equal-peak spectra are barely affected.

    Plan D applies this as a single scaling factor on the composite
    spectral score (rather than per-peak inside the dot product), so we
    return a scalar in (0, 1] that approximates the average per-peak
    damping. We compute it as ``sum(A * 1/(1+wA)) / sum(A)`` over the
    base-peak-normalized spectrum, which gives the intensity-weighted
    average of the per-peak scaling factor.
    """
    if not spectrum:
        return 1.0
    intensities = [float(i) for _, i in spectrum if float(i) > 0]
    if not intensities:
        return 1.0
    max_i = max(intensities)
    if max_i <= 0:
        return 1.0
    norm = [i / max_i for i in intensities]
    s = sum(norm)
    denom = a + s - 1.0
    if denom <= 0:
        # Degenerate: too few peaks to apply the scaling. AMDIS notes this
        # only matters for "few dominant peaks" cases; for a 1-peak case,
        # denom = a, so this branch would only fire for exotic inputs.
        denom = max(denom, 1e-9)
    w = 1.0 / denom
    weighted_sum = 0.0
    for amp in norm:
        weighted_sum += amp * (1.0 / (1.0 + w * amp))
    factor = weighted_sum / s
    return max(min(factor, 1.0), 0.0)


def _detection_threshold_correction(threshold: float) -> float:
    """AMDIS Stein 1999 eq (11): match factor *= (1 - threshold ** 0.3).

    ``threshold`` is the relative detection floor (base peak = 1.0). The
    default 0.0 produces a no-op multiplier of 1.0.
    """
    if threshold <= 0:
        return 1.0
    if threshold >= 1.0:
        return 0.0
    return 1.0 - threshold ** 0.3


def gcms_match_factor(
    measured: list[tuple[float, float]],
    reference: list[tuple[float, float]],
    *,
    mz_tolerance: float = 0.01,
    mode: ChromMode = "none",
    rt_query: Optional[float] = None,
    rt_ref: Optional[float] = None,
    ri_query: Optional[float] = None,
    ri_ref: Optional[float] = None,
    rt_tolerance: float = 0.1,
    ri_tolerance: float = 10.0,
    chrom_weight: float = 0.5,
    n_adjacent_subtracted: int = 0,
    detection_threshold: float = 0.0,
) -> dict:
    """Composite GC-MS match factor following AMDIS (Stein 1999) and MS-DIAL.

    Spectral component (always computed):
        - WDP / RDP / SDP via the Plan C primitives in
          ``metabo_core.algorithms.similarity``. Each of WDP/RDP/SDP
          already applies the peak-count penalty internally; we do NOT
          re-multiply.
        - Composite spectral score uses the same shape as the LC-MS
          ``composite_similarity`` for cross-mode consistency:
              raw = (wdp*3 + sdp*3 + rdp*2 + matched_pct) / 9
          with ``matched_pct = matched / max(reference_peak_count, 1)``.
        - Adjacent-deconvolution penalty: -0.02 per overlap on the [0,1]
          scale, clamped at 0. (Plan D scale; AMDIS paper uses [0,100].)
        - AMDIS spectrum complexity scaling (eq 8/9, a=0.5) applied as a
          multiplicative scaling on the post-penalty score.
        - AMDIS detection-threshold correction (eq 11): multiply by
          ``(1 - threshold**0.3)``. Default 0.0 = no-op.

    Chromatographic component (mode='none' skips):
        - mode='rt': gaussian_similarity(rt_query, rt_ref, rt_tolerance)
        - mode='ri': gaussian_similarity(ri_query, ri_ref, ri_tolerance)

    Total:
        - mode='none': total = spectral_score
        - mode='rt' or 'ri': total = spectral_score * (chrom_score ** chrom_weight)
          (multiplicative; downweights bad chromatographic match without
           crushing otherwise-good spectral matches; chrom_weight=0 makes
           the chrom score informational only)

    Returns
    -------
    dict with keys:
        spectral_score : float in [0, 1]
        chrom_score    : float|None
        total          : float in [0, 1]
        wdp, rdp, sdp  : float in [0, 1]
        matched_pct    : float in [0, 1]
        n_adjacent_subtracted : int

    Notes
    -----
    RDP is exposed as a separate first-class field. Per user 2026-04-29
    GC-MS fullscan workflows lean heavily on RDP because it ignores
    measured peaks not in the reference, which suits noisy fullscan apex
    spectra. Result CSV and GUI table must surface RDP as its own column.

    Intentional departures from AMDIS Stein 1999 are documented in
    ``docs/superpowers/plans/2026-04-29-d-gcms-algorithm-upgrade.md``
    section "Algorithm priority order" and "Intentional departures":
    (1) RT/RI uses Gaussian, not linear penalty;
    (2) composite shape matches composite_similarity, not AMDIS eq 7
    flag-weighted pure:impure mix;
    (3) AMDIS eq 10 component-purity correction not implemented in Plan D.
    """
    # Empty inputs → all-zero result.
    if not measured or not reference:
        return {
            "spectral_score": 0.0,
            "chrom_score": _compute_chrom_score(
                mode, rt_query, rt_ref, rt_tolerance,
                ri_query, ri_ref, ri_tolerance,
            ),
            "total": 0.0,
            "wdp": 0.0,
            "rdp": 0.0,
            "sdp": 0.0,
            "matched_pct": 0.0,
            "n_matched": 0,
            "n_adjacent_subtracted": int(n_adjacent_subtracted),
        }

    # Single-pass three-score computation (same approach as
    # composite_similarity's _compute_three_scores). Saves 3x peak-matching
    # work versus calling weighted_dot_product, reverse_dot_product,
    # simple_dot_product independently — important for library searches
    # (216 features x 2530 library entries x 3 dot products each).
    matched = _match_peaks(measured, reference, mz_tolerance)
    n_matched = len(matched)
    max_q = max(i for _, i in measured)
    max_r = max(i for _, i in reference)
    if max_q < 1e-12 or max_r < 1e-12:
        wdp, rdp, sdp = 0.0, 0.0, 0.0
    else:
        wdp, sdp, rdp = _compute_three_scores(
            matched, measured, reference, max_q, max_r,
        )
    wdp = float(wdp)
    rdp = float(rdp)
    sdp = float(sdp)

    n_ref = len(reference)
    matched_pct = float(n_matched) / float(max(n_ref, 1))
    matched_pct = min(matched_pct, 1.0)

    raw = (wdp * 3.0 + sdp * 3.0 + rdp * 2.0 + matched_pct) / 9.0
    raw = max(min(raw, 1.0), 0.0)

    # Adjacent-deconvolution penalty (Plan D scale: 0.02 per overlap on [0,1]).
    raw -= 0.02 * float(max(int(n_adjacent_subtracted), 0))
    raw = max(raw, 0.0)

    # AMDIS spectrum complexity scaling (eq 8/9): use the *measured* spectrum
    # since it is what the user submits to the library search.
    complexity = _spectrum_complexity_factor(measured, a=0.5)
    spectral_score = raw * complexity

    # AMDIS detection-threshold correction (eq 11).
    spectral_score *= _detection_threshold_correction(detection_threshold)
    spectral_score = max(min(spectral_score, 1.0), 0.0)

    chrom_score = _compute_chrom_score(
        mode, rt_query, rt_ref, rt_tolerance,
        ri_query, ri_ref, ri_tolerance,
    )

    if chrom_score is None:
        total = spectral_score
    else:
        # Multiplicative: total = spectral * chrom^chrom_weight.
        # chrom_weight=0 → total == spectral (chrom is informational).
        if chrom_weight <= 0:
            total = spectral_score
        elif chrom_score <= 0:
            total = 0.0
        else:
            total = spectral_score * (chrom_score ** float(chrom_weight))
    total = max(min(total, 1.0), 0.0)

    return {
        "spectral_score": float(spectral_score),
        "chrom_score": chrom_score,
        "total": float(total),
        "wdp": wdp,
        "rdp": rdp,
        "sdp": sdp,
        "matched_pct": float(matched_pct),
        "n_matched": int(n_matched),
        "n_adjacent_subtracted": int(n_adjacent_subtracted),
    }


def _compute_chrom_score(
    mode: ChromMode,
    rt_query: Optional[float],
    rt_ref: Optional[float],
    rt_tolerance: float,
    ri_query: Optional[float],
    ri_ref: Optional[float],
    ri_tolerance: float,
) -> Optional[float]:
    """Return the chromatographic Gaussian similarity for the active mode.

    Returns ``None`` when ``mode='none'``. For 'rt' or 'ri' modes,
    returns a float in ``[0, 1]``; if the query/ref chromatographic
    value is missing, ``gaussian_similarity`` returns 0.0.
    """
    if mode == "none":
        return None
    if mode == "rt":
        return float(gaussian_similarity(rt_query, rt_ref, rt_tolerance))
    if mode == "ri":
        return float(gaussian_similarity(ri_query, ri_ref, ri_tolerance))
    raise ValueError(f"unknown chromatographic mode: {mode!r}")
