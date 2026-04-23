"""Spectral and chromatographic similarity measures."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# MS2 isotope step-pattern evidence
# ---------------------------------------------------------------------------

def ms2_isotope_step_score(
    peaks_lighter: list[tuple[float, float]],
    peaks_heavier: list[tuple[float, float]],
    isotope_delta: float = 1.003355,
    mz_tolerance: float = 0.01,
    top_n: int = 6,
) -> tuple[int, int]:
    """Count how many of the lighter feature's top-N high-response ions
    have a corresponding +isotope_delta ion in the heavier feature's MS2.

    True isotope pairs (M and M+1) typically share the property that the
    heavier feature's MS2 "echoes" the lighter feature's MS2 with each
    fragment shifted by +isotope_delta (when the fragment retains the
    heavy atom). This is independent of cosine similarity — it survives
    even when relative intensities differ.

    Parameters
    ----------
    peaks_lighter, peaks_heavier : list of (mz, intensity)
    isotope_delta : Da, e.g. 1.003355 for C13
    mz_tolerance : Da, ±tol around target m/z
    top_n : take the top N most intense ions of the lighter spectrum

    Returns
    -------
    (matched_count, considered_count)
    """
    if not peaks_lighter or not peaks_heavier:
        return 0, 0

    # Top N most intense ions of the lighter spectrum
    sorted_l = sorted(peaks_lighter, key=lambda x: -x[1])[:top_n]
    heavy_mz = np.asarray([m for m, _ in peaks_heavier], dtype=np.float64)

    matched = 0
    for mz_l, _ in sorted_l:
        target = mz_l + isotope_delta
        if heavy_mz.size and float(np.min(np.abs(heavy_mz - target))) <= mz_tolerance:
            matched += 1
    return matched, len(sorted_l)


# ---------------------------------------------------------------------------
# Greedy peak matching (adapted from user's existing script)
# ---------------------------------------------------------------------------

def greedy_match(
    peaks_a: list[tuple[float, float]],
    peaks_b: list[tuple[float, float]],
    mz_tolerance: float,
    shift: float = 0.0,
) -> tuple[float, int]:
    """Greedy peak matching with optional m/z shift.

    Matches peaks_a to peaks_b (shifted by `shift`). For each pair,
    candidates are ranked by product of intensities; assignments are
    greedy (no duplicate assignments).

    Parameters
    ----------
    peaks_a, peaks_b : list of (mz, intensity)
    mz_tolerance : Da
    shift : m/z shift applied to peaks_b before matching

    Returns
    -------
    (sum_of_products, n_matched)
    """
    if not peaks_a or not peaks_b:
        return 0.0, 0

    candidates = []
    for i, (mz_a, int_a) in enumerate(peaks_a):
        for j, (mz_b, int_b) in enumerate(peaks_b):
            if abs(mz_a - (mz_b + shift)) <= mz_tolerance:
                candidates.append((int_a * int_b, i, j))

    candidates.sort(key=lambda x: x[0], reverse=True)

    used_a = set()
    used_b = set()
    sum_products = 0.0
    n_matched = 0

    for product, i, j in candidates:
        if i not in used_a and j not in used_b:
            sum_products += product
            used_a.add(i)
            used_b.add(j)
            n_matched += 1

    return sum_products, n_matched


def _vector_norm(peaks: list[tuple[float, float]]) -> float:
    """L2 norm of intensity vector."""
    return math.sqrt(sum(i * i for _, i in peaks))


# ---------------------------------------------------------------------------
# Modified cosine similarity
# ---------------------------------------------------------------------------

def modified_cosine(
    peaks_a: list[tuple[float, float]],
    peaks_b: list[tuple[float, float]],
    precursor_a: float,
    precursor_b: float,
    mz_tolerance: float,
    precursor_exclusion: float = 1.5,
) -> tuple[float, int]:
    """Modified cosine similarity between two MS2 spectra.

    Tries both direct matching and shifted matching
    (shift = precursor_b - precursor_a), returns the better result.

    Parameters
    ----------
    peaks_a, peaks_b : list of (mz, intensity)
    precursor_a, precursor_b : precursor m/z values
    mz_tolerance : fragment m/z tolerance (Da)
    precursor_exclusion : exclude fragments within this range of precursor

    Returns
    -------
    (similarity_score, n_matched_peaks)
    """
    # Filter out precursor region
    filt_a = [(m, i) for m, i in peaks_a
              if abs(m - precursor_a) > precursor_exclusion]
    filt_b = [(m, i) for m, i in peaks_b
              if abs(m - precursor_b) > precursor_exclusion]

    if not filt_a or not filt_b:
        return 0.0, 0

    norm_a = _vector_norm(filt_a)
    norm_b = _vector_norm(filt_b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0, 0

    # Direct matching
    sum_direct, count_direct = greedy_match(filt_a, filt_b, mz_tolerance, shift=0.0)
    # Shifted matching
    shift = precursor_b - precursor_a
    sum_shifted, count_shifted = greedy_match(filt_a, filt_b, mz_tolerance, shift=shift)

    if sum_shifted > sum_direct:
        best_sum, best_count = sum_shifted, count_shifted
    else:
        best_sum, best_count = sum_direct, count_direct

    score = best_sum / (norm_a * norm_b)
    return min(score, 1.0), best_count


# ---------------------------------------------------------------------------
# Neutral loss cosine similarity
# ---------------------------------------------------------------------------

def neutral_loss_cosine(
    peaks_a: list[tuple[float, float]],
    peaks_b: list[tuple[float, float]],
    precursor_a: float,
    precursor_b: float,
    mz_tolerance: float,
) -> tuple[float, int]:
    """Cosine similarity based on neutral losses.

    Converts fragment m/z to neutral losses (precursor - fragment),
    then computes cosine similarity.
    """
    nl_a = [(precursor_a - m, i) for m, i in peaks_a if precursor_a - m > 0]
    nl_b = [(precursor_b - m, i) for m, i in peaks_b if precursor_b - m > 0]

    if not nl_a or not nl_b:
        return 0.0, 0

    norm_a = _vector_norm(nl_a)
    norm_b = _vector_norm(nl_b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0, 0

    sum_prod, n_matched = greedy_match(nl_a, nl_b, mz_tolerance, shift=0.0)
    score = sum_prod / (norm_a * norm_b)
    return min(score, 1.0), n_matched


# ---------------------------------------------------------------------------
# Simple cosine similarity (no shift)
# ---------------------------------------------------------------------------

def cosine_similarity(
    peaks_a: list[tuple[float, float]],
    peaks_b: list[tuple[float, float]],
    mz_tolerance: float,
) -> tuple[float, int]:
    """Standard cosine similarity between two spectra."""
    if not peaks_a or not peaks_b:
        return 0.0, 0

    norm_a = _vector_norm(peaks_a)
    norm_b = _vector_norm(peaks_b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0, 0

    sum_prod, n_matched = greedy_match(peaks_a, peaks_b, mz_tolerance)
    score = sum_prod / (norm_a * norm_b)
    return min(score, 1.0), n_matched


# ---------------------------------------------------------------------------
# MS-DIAL-style three-component spectral similarity (library matching)
# ---------------------------------------------------------------------------

def _peak_count_penalty(n_peaks: int) -> float:
    """Peak count penalty from MS-DIAL (Stein 1999)."""
    if n_peaks <= 1:
        return 0.75
    if n_peaks == 2:
        return 0.88
    if n_peaks == 3:
        return 0.94
    if n_peaks <= 5:
        return 0.97
    return 1.0


def weighted_dot_product(
    peaks_query: list[tuple[float, float]],
    peaks_ref: list[tuple[float, float]],
    mz_tolerance: float,
) -> float:
    """Weighted dot product (MS-DIAL style): uses m/z as weight."""
    if not peaks_query or not peaks_ref:
        return 0.0

    # Normalize intensities
    max_q = max(i for _, i in peaks_query)
    max_r = max(i for _, i in peaks_ref)
    if max_q < 1e-12 or max_r < 1e-12:
        return 0.0

    # Match peaks
    matched_pairs = _match_peaks(peaks_query, peaks_ref, mz_tolerance)
    if not matched_pairs:
        return 0.0

    covariance = 0.0
    scalar_q = 0.0
    scalar_r = 0.0

    for (mz_q, int_q), (mz_r, int_r) in matched_pairs:
        nq = int_q / max_q
        nr = int_r / max_r
        mz_avg = (mz_q + mz_r) / 2
        covariance += math.sqrt(nq * nr) * mz_avg
        scalar_q += nq * mz_avg
        scalar_r += nr * mz_avg

    # Add unmatched contributions to scalars
    matched_q = {round(mz, 4) for (mz, _), _ in matched_pairs}
    matched_r = {round(mz, 4) for _, (mz, _) in matched_pairs}
    for mz, i in peaks_query:
        if round(mz, 4) not in matched_q:
            scalar_q += (i / max_q) * mz
    for mz, i in peaks_ref:
        if round(mz, 4) not in matched_r:
            scalar_r += (i / max_r) * mz

    if scalar_q < 1e-12 or scalar_r < 1e-12:
        return 0.0

    score = (covariance ** 2) / (scalar_q * scalar_r)
    penalty = _peak_count_penalty(len(peaks_ref))
    return min(score * penalty, 1.0)


def reverse_dot_product(
    peaks_query: list[tuple[float, float]],
    peaks_ref: list[tuple[float, float]],
    mz_tolerance: float,
) -> float:
    """Reverse dot product: focuses on reference peaks only."""
    if not peaks_query or not peaks_ref:
        return 0.0

    max_q = max(i for _, i in peaks_query)
    max_r = max(i for _, i in peaks_ref)
    if max_q < 1e-12 or max_r < 1e-12:
        return 0.0

    matched_pairs = _match_peaks(peaks_query, peaks_ref, mz_tolerance)

    covariance = 0.0
    scalar_q = 0.0
    scalar_r = 0.0

    for (mz_q, int_q), (mz_r, int_r) in matched_pairs:
        nq = int_q / max_q
        nr = int_r / max_r
        mz_avg = (mz_q + mz_r) / 2
        covariance += math.sqrt(nq * nr) * mz_avg
        scalar_q += nq * mz_avg
        scalar_r += nr * mz_avg

    # Only add unmatched reference peaks to scalar_r
    matched_r = {round(mz, 4) for _, (mz, _) in matched_pairs}
    for mz, i in peaks_ref:
        if round(mz, 4) not in matched_r:
            scalar_r += (i / max_r) * mz

    # scalar_q only from matched peaks (already computed)
    if scalar_q < 1e-12 or scalar_r < 1e-12:
        return 0.0

    score = (covariance ** 2) / (scalar_q * scalar_r)
    penalty = _peak_count_penalty(len(peaks_ref))
    return min(score * penalty, 1.0)


def simple_dot_product(
    peaks_query: list[tuple[float, float]],
    peaks_ref: list[tuple[float, float]],
    mz_tolerance: float,
) -> float:
    """Simple dot product without m/z weighting."""
    if not peaks_query or not peaks_ref:
        return 0.0

    max_q = max(i for _, i in peaks_query)
    max_r = max(i for _, i in peaks_ref)
    if max_q < 1e-12 or max_r < 1e-12:
        return 0.0

    matched_pairs = _match_peaks(peaks_query, peaks_ref, mz_tolerance)

    covariance = 0.0
    scalar_q = sum((i / max_q) for _, i in peaks_query)
    scalar_r = sum((i / max_r) for _, i in peaks_ref)

    for (_, int_q), (_, int_r) in matched_pairs:
        covariance += math.sqrt((int_q / max_q) * (int_r / max_r))

    if scalar_q < 1e-12 or scalar_r < 1e-12:
        return 0.0

    score = (covariance ** 2) / (scalar_q * scalar_r)
    penalty = _peak_count_penalty(len(peaks_ref))
    return min(score * penalty, 1.0)


def composite_similarity(
    peaks_query: list[tuple[float, float]],
    peaks_ref: list[tuple[float, float]],
    mz_tolerance: float = 0.02,
    precursor_query: float = 0.0,
    precursor_ref: float = 0.0,
    rt_query: float = 0.0,
    rt_ref: float = 0.0,
    ms1_tolerance: float = 0.01,
    rt_tolerance: float = 100.0,
    use_rt: bool = False,
) -> tuple[float, int]:
    """MS-DIAL-style spectral similarity for library matching.

    Optimized: matches peaks ONCE and computes all three dot products
    from the same matched pairs, avoiding redundant computation.
    """
    if not peaks_query or not peaks_ref:
        return 0.0, 0

    # Match peaks once (try both product-ion and neutral-loss modes)
    matched = _match_peaks(peaks_query, peaks_ref, mz_tolerance)
    if precursor_query > 0 and precursor_ref > 0:
        nl_matched = _match_peaks_shifted(
            peaks_query, peaks_ref, mz_tolerance, precursor_ref - precursor_query)
        if len(nl_matched) > len(matched):
            matched = nl_matched
    n_matched = len(matched)

    # Precompute normalizations once
    max_q = max(i for _, i in peaks_query)
    max_r = max(i for _, i in peaks_ref)
    if max_q < 1e-12 or max_r < 1e-12:
        return 0.0, 0

    # Compute all three dot products from the same matched pairs
    wdp, sdp, rdp = _compute_three_scores(
        matched, peaks_query, peaks_ref, max_q, max_r)

    # Matched peaks percentage
    n_ref_sig = sum(1 for _, i in peaks_ref if i >= max_r * 0.01)
    matched_pct = min(n_matched / max(n_ref_sig, 1), 1.0)

    # MS2 score (MS-DIAL weights)
    ms2_score = (wdp * 3 + sdp * 3 + rdp * 2 + matched_pct) / 9.0

    # Precursor m/z Gaussian similarity
    precursor_sim = 1.0
    if precursor_query > 0 and precursor_ref > 0 and ms1_tolerance > 0:
        precursor_sim = math.exp(-0.5 * ((precursor_query - precursor_ref) / ms1_tolerance) ** 2)

    # Total score
    if use_rt and rt_tolerance > 0 and rt_query > 0 and rt_ref > 0:
        rt_sim = math.exp(-0.5 * ((rt_query - rt_ref) / (rt_tolerance / 60.0)) ** 2)
        total_score = (precursor_sim + ms2_score * 3 + rt_sim) / 5.0
    else:
        total_score = (precursor_sim + ms2_score * 3) / 4.0

    return min(total_score, 1.0), n_matched


def _compute_three_scores(
    matched: list,
    peaks_query: list[tuple[float, float]],
    peaks_ref: list[tuple[float, float]],
    max_q: float,
    max_r: float,
) -> tuple[float, float, float]:
    """Compute WDP, SDP, RDP from pre-matched pairs in a single pass."""
    n_ref = len(peaks_ref)
    penalty = _peak_count_penalty(n_ref)

    # Build matched sets for identifying unmatched peaks
    matched_q_idx = set()
    matched_r_idx = set()

    # Accumulators for all three scores
    cov_w = 0.0   # weighted covariance (with mz)
    cov_s = 0.0   # simple covariance (no mz)
    sq_w = 0.0    # WDP scalar_q (matched)
    sr_w = 0.0    # WDP scalar_r (matched)
    sq_r = 0.0    # RDP scalar_q (matched only)
    sr_r = 0.0    # RDP scalar_r (matched)

    for (mz_q, int_q), (mz_r, int_r) in matched:
        nq = int_q / max_q
        nr = int_r / max_r
        mz_avg = (mz_q + mz_r) / 2
        sqrt_nqnr = math.sqrt(nq * nr)

        cov_w += sqrt_nqnr * mz_avg
        cov_s += sqrt_nqnr
        sq_w += nq * mz_avg
        sr_w += nr * mz_avg
        sq_r += nq * mz_avg
        sr_r += nr * mz_avg

        # Track matched peaks by rounded mz (for unmatched identification)
        matched_q_idx.add(round(mz_q, 4))
        matched_r_idx.add(round(mz_r, 4))

    # Unmatched contributions
    sq_w_unmatched = 0.0
    sr_w_unmatched = 0.0
    sr_r_unmatched = 0.0
    sq_s = 0.0  # SDP uses all query peaks
    sr_s = 0.0  # SDP uses all ref peaks

    for mz, i in peaks_query:
        ni = i / max_q
        sq_s += ni
        if round(mz, 4) not in matched_q_idx:
            sq_w_unmatched += ni * mz

    for mz, i in peaks_ref:
        ni = i / max_r
        sr_s += ni
        if round(mz, 4) not in matched_r_idx:
            sr_w_unmatched += ni * mz
            sr_r_unmatched += ni * mz

    # WDP: all peaks in both scalars
    total_sq_w = sq_w + sq_w_unmatched
    total_sr_w = sr_w + sr_w_unmatched
    wdp = 0.0
    if total_sq_w > 1e-12 and total_sr_w > 1e-12:
        wdp = min((cov_w ** 2) / (total_sq_w * total_sr_w) * penalty, 1.0)

    # SDP: all peaks, no mz weighting
    sdp = 0.0
    if sq_s > 1e-12 and sr_s > 1e-12:
        sdp = min((cov_s ** 2) / (sq_s * sr_s) * penalty, 1.0)

    # RDP: query scalar from matched only, ref scalar from all
    total_sr_rdp = sr_r + sr_r_unmatched
    rdp = 0.0
    if sq_r > 1e-12 and total_sr_rdp > 1e-12:
        rdp = min((cov_w ** 2) / (sq_r * total_sr_rdp) * penalty, 1.0)

    return wdp, sdp, rdp


def _match_peaks_shifted(
    peaks_a: list[tuple[float, float]],
    peaks_b: list[tuple[float, float]],
    mz_tolerance: float,
    precursor_diff: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Match peaks via neutral loss: a_mz matches b_mz + precursor_diff."""
    candidates = []
    for i, (mz_a, int_a) in enumerate(peaks_a):
        for j, (mz_b, int_b) in enumerate(peaks_b):
            if abs(mz_a - mz_b - precursor_diff) <= mz_tolerance:
                candidates.append((int_a * int_b, i, j))

    candidates.sort(key=lambda x: x[0], reverse=True)
    used_a, used_b = set(), set()
    matched = []
    for _, i, j in candidates:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            matched.append((peaks_a[i], peaks_b[j]))
    return matched


def _match_peaks(
    peaks_a: list[tuple[float, float]],
    peaks_b: list[tuple[float, float]],
    mz_tolerance: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Greedy peak matching, returns list of (peak_a, peak_b) matched pairs."""
    candidates = []
    for i, (mz_a, int_a) in enumerate(peaks_a):
        for j, (mz_b, int_b) in enumerate(peaks_b):
            if abs(mz_a - mz_b) <= mz_tolerance:
                candidates.append((int_a * int_b, i, j))

    candidates.sort(key=lambda x: x[0], reverse=True)
    used_a, used_b = set(), set()
    matched = []

    for _, i, j in candidates:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            matched.append((peaks_a[i], peaks_b[j]))

    return matched


# ---------------------------------------------------------------------------
# EIC correlation
# ---------------------------------------------------------------------------

def eic_pearson_correlation(
    eic_a: np.ndarray,
    eic_b: np.ndarray,
    min_points: int = 3,
) -> tuple[float, int]:
    """Scan-by-scan Pearson correlation of two EIC traces.

    Only uses scans where both EICs have nonzero signal.

    Returns
    -------
    (pearson_r, n_correlated_points)
    """
    mask = (eic_a > 0) & (eic_b > 0)
    n_points = int(np.sum(mask))
    if n_points < min_points:
        return 0.0, n_points

    a = eic_a[mask]
    b = eic_b[mask]

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0, n_points

    r, _ = pearsonr(a, b)
    return float(r), n_points


def eic_pearson_in_range(
    eic_a: np.ndarray,
    eic_b: np.ndarray,
    rt_array: np.ndarray,
    rt_start: float,
    rt_end: float,
    min_points: int = 3,
) -> tuple[float, int]:
    """Pearson correlation of two EICs within an RT range."""
    mask_rt = (rt_array >= rt_start) & (rt_array <= rt_end)
    return eic_pearson_correlation(eic_a[mask_rt], eic_b[mask_rt], min_points)
