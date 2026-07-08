"""Spectral and chromatographic similarity measures."""
from __future__ import annotations

import math
from typing import NamedTuple, Optional

import numpy as np
from scipy.stats import pearsonr


class CompositeSimilarityResult(NamedTuple):
    """:func:`composite_similarity_breakdown` 的逐分量结果。

    ``score`` 对齐 MS-DIAL 导出的 ``Total score``（``GetTotalScore``）。
    谱-only 命中的公式为::

        TotalScore = (sqrt(WDP) + sqrt(SDP) + sqrt(RDP)) / 3 + Matched%

    其中 ``WDP/SDP/RDP`` 是 :func:`_compute_three_scores` 返回的平方点积
    （各 ∈[0,1]，即 ``cov²/(sq·sr)·penalty``）；MS-DIAL 内部存的就是这组
    平方值，求和前会先取 ``sqrt``，所以这里对每项取 ``math.sqrt``。
    ``Matched%`` = ``n_matched_sig / n_sig_ref`` ∈[0,1]，分母为强度 ≥ 参考
    基峰 1% 的显著参考峰数（本函数内局部计算 ``n_sig_ref``，floor 为 1），
    分子为其中被 query 匹配到的显著参考峰数。

    无 ``/2``、无 ``min(., 1.0)`` 钳制——分数本身落在 ``[0, 2]``，
    ``use_rt`` 时 RT 高斯作为加和项再叠加，可超过 2。``score`` 与
    ``total_score`` 数值相同（``score == total_score``）。
    """
    score: float        # MS-DIAL TotalScore（== total_score；范围 [0,2]，use_rt 时更高）
    n_matched: int
    wdp: float          # 加权点积（按 m/z 加权，平方形式）
    sdp: float          # 简单点积（不加权，平方形式）
    rdp: float          # 反向点积（query 端只用匹配峰，平方形式）
    matched_pct: float = 0.0   # 显著参考峰被匹配占比 n_matched_sig / n_sig_ref ∈[0,1]（MS-DIAL Matched%）
    total_score: float = 0.0   # MS-DIAL TotalScore（= score；范围 [0,2]，use_rt 时更高）


# ---------------------------------------------------------------------------
# Standalone Gaussian similarity helper
# ---------------------------------------------------------------------------

def gaussian_similarity(
    value: Optional[float],
    ref: Optional[float],
    tolerance: float,
) -> float:
    """Continuous Gaussian similarity ``exp(-0.5 * ((value - ref) / tol)^2)``.

    Returns ``1.0`` when ``value == ref``, ``~0.6065`` at one tolerance away
    and ``~0.1353`` at two tolerances away. Returns ``0.0`` when either
    input is ``None`` or when ``tolerance`` is non-positive.

    The helper has no unit conversion baked in. LC-MS callers that want
    the historical "rt_tolerance is in seconds even though rt is in minutes"
    convention must do their own /60 before calling. GC-MS callers (Plan D
    ``gcms_match_factor``) can pass the tolerance directly in the same
    unit as the values.

    Examples
    --------
    >>> round(gaussian_similarity(10.0, 10.0, 0.1), 6)
    1.0
    >>> round(gaussian_similarity(10.1, 10.0, 0.1), 4)
    0.6065
    """
    if value is None or ref is None:
        return 0.0
    if tolerance <= 0:
        return 0.0
    delta = float(value) - float(ref)
    return math.exp(-0.5 * (delta / float(tolerance)) ** 2)


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
    if heavy_mz.size == 0:
        return 0, len(sorted_l)
    # Sort once so we can binary-search per query, replacing O(N) np.min(np.abs)
    # per query with O(log N). Output is unchanged: we only compare the nearest
    # distance to ``mz_tolerance`` and the nearest is invariant under sorting.
    heavy_mz_sorted = np.sort(heavy_mz)
    n_h = heavy_mz_sorted.size

    matched = 0
    for mz_l, _ in sorted_l:
        target = mz_l + isotope_delta
        pos = int(np.searchsorted(heavy_mz_sorted, target))
        dl = abs(heavy_mz_sorted[pos - 1] - target) if pos > 0 else float("inf")
        dr = abs(heavy_mz_sorted[pos] - target) if pos < n_h else float("inf")
        if min(dl, dr) <= mz_tolerance:
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


def composite_similarity_breakdown(
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
    q_arrays: Optional[tuple] = None,
    r_arrays: Optional[tuple] = None,
) -> CompositeSimilarityResult:
    """对齐 MS-DIAL ``Total score`` 的综合谱图相似度，返回逐分量分数。

    LC-MS（ASFAM 与 DDA）统一使用本函数。谱-only 命中的公式为::

        spectral_total = (sqrt(WDP) + sqrt(SDP) + sqrt(RDP)) / 3 + Matched%
        score = spectral_total                          if use_rt=False
        score = spectral_total + rt_sim                 if use_rt=True

    其中 ``WDP/SDP/RDP`` 是 :func:`_compute_three_scores` 的平方点积
    （各 ∈[0,1]），与 MS-DIAL 内部存储口径一致，求和前先取 ``sqrt``。
    ``Matched%`` = ``n_matched_sig / n_sig_ref`` ∈[0,1]，分母 ``n_sig_ref``
    为强度 ≥ 参考基峰 1% 的显著参考峰数（本函数内局部计算，floor 为 1）
    ——不是 ``len(peaks_ref)``；分子 ``n_matched_sig`` 只数这些显著参考峰
    中被 query 匹配者，故占比恒 ≤1。

    这是 MS-DIAL ``GetTotalScore`` 的无权加和：没有 ``/2``、没有
    ``min(., 1.0)`` 钳制；分数本身落在 ``[0, 2]``，``use_rt`` 时再叠加
    RT 高斯（加和项，非旧的 ``/4`` 平均）后可更高。保留钳制会破坏与
    MS-DIAL 导出列的数值可比性。

    ``precursor_query`` / ``precursor_ref`` / ``ms1_tolerance`` 仍然
    保留在签名中，但目前仅用于触发中性损失（neutral-loss）方向的
    替代峰匹配尝试；不再作为独立项贡献综合分。

    RT 项的 ``rt_tolerance / 60.0`` 分钟-秒换算保留：RT 数据通常以
    分钟存储而容差以秒输入，统一在此处转换以避免每个调用点重复。
    """
    if not peaks_query or not peaks_ref:
        return CompositeSimilarityResult(0.0, 0, 0.0, 0.0, 0.0)

    # 优先 product-ion 匹配；如果中性损失方向能匹到更多峰，则改用 NL 路径。
    # ``q_arrays`` / ``r_arrays``（可选）是预先构造好的 float64 (mz, intensity)
    # 数组；库匹配热循环传入它们，使 direct 与 NL 两个方向共用同一对数组，
    # 省掉每个 (feature, candidate) 调用里 ``np.fromiter`` 的重复构造。匹配对
    # 取值与 tuple 路径逐位一致，故三套点积/Matched% 完全不变。
    if q_arrays is not None and r_arrays is not None:
        mz_q_a, int_q_a = q_arrays
        mz_r_a, int_r_a = r_arrays
        matched = _greedy_match_arrays(
            mz_q_a, int_q_a, mz_r_a, int_r_a, mz_tolerance, 0.0)
        if precursor_query > 0 and precursor_ref > 0:
            nl_matched = _greedy_match_arrays(
                mz_q_a, int_q_a, mz_r_a, int_r_a, mz_tolerance,
                precursor_ref - precursor_query)
            if len(nl_matched) > len(matched):
                matched = nl_matched
    else:
        matched = _match_peaks(peaks_query, peaks_ref, mz_tolerance)
        if precursor_query > 0 and precursor_ref > 0:
            nl_matched = _match_peaks_shifted(
                peaks_query, peaks_ref, mz_tolerance,
                precursor_ref - precursor_query)
            if len(nl_matched) > len(matched):
                matched = nl_matched
    n_matched = len(matched)

    # 一次性算好归一化基准
    max_q = max(i for _, i in peaks_query)
    max_r = max(i for _, i in peaks_ref)
    if max_q < 1e-12 or max_r < 1e-12:
        return CompositeSimilarityResult(0.0, 0, 0.0, 0.0, 0.0)

    # 共用同一组匹配对，一次性算出三套点积
    wdp, sdp, rdp = _compute_three_scores(
        matched, peaks_query, peaks_ref, max_q, max_r)

    # Matched%（MS-DIAL counter/libCounter）：分母 = 强度 ≥ 参考基峰 1% 的显著参考峰数
    # （局部 n_sig_ref，floor 1）；分子 = 这些显著参考峰中有 query 匹配者。
    # 限定分子到显著参考峰，使 matched_pct 落在 [0,1]，与 MS-DIAL counter 一致——
    # 否则匹配到 <1% 基峰的噪声参考峰会让占比 >1。
    thresh = max_r * 0.01
    n_sig_ref = max(sum(1 for _, i in peaks_ref if i >= thresh), 1)
    n_matched_sig = sum(1 for (_q, (_mz_r, int_r)) in matched if int_r >= thresh)
    matched_pct = n_matched_sig / n_sig_ref

    # MS-DIAL TotalScore（谱-only）：三点积取 sqrt 后平均，加 Matched%。无 /2，无钳制。
    # max(x, 0.0) 是 sqrt 定义域的防御性保护：_compute_three_scores 的平方点积本就 ≥0
    # 且已钳到 [0,1]，这里只是兜底防止浮点负零/越界传入 math.sqrt。
    spectral_total = (
        math.sqrt(max(wdp, 0.0))
        + math.sqrt(max(sdp, 0.0))
        + math.sqrt(max(rdp, 0.0))
    ) / 3.0 + matched_pct

    if use_rt and rt_tolerance > 0 and rt_query > 0 and rt_ref > 0:
        # MS-DIAL GetTotalScore 中 RtSimilarity 是加和项；容差秒→分 /60。
        rt_sim = gaussian_similarity(rt_query, rt_ref, rt_tolerance / 60.0)
        total_score = spectral_total + rt_sim
    else:
        total_score = spectral_total

    return CompositeSimilarityResult(
        score=total_score,
        n_matched=n_matched,
        wdp=wdp,
        sdp=sdp,
        rdp=rdp,
        matched_pct=matched_pct,
        total_score=total_score,
    )


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

    Returns ``(composite_score, n_matched)``. For the per-component WDP/SDP/RDP
    breakdown, call :func:`composite_similarity_breakdown` instead.
    """
    result = composite_similarity_breakdown(
        peaks_query, peaks_ref, mz_tolerance,
        precursor_query, precursor_ref,
        rt_query, rt_ref,
        ms1_tolerance, rt_tolerance, use_rt,
    )
    return result.score, result.n_matched


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
    return _greedy_match_pairs(peaks_a, peaks_b, mz_tolerance, precursor_diff)


def _match_peaks(
    peaks_a: list[tuple[float, float]],
    peaks_b: list[tuple[float, float]],
    mz_tolerance: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Greedy peak matching, returns list of (peak_a, peak_b) matched pairs."""
    return _greedy_match_pairs(peaks_a, peaks_b, mz_tolerance, 0.0)


def _greedy_match_pairs(
    peaks_a: list[tuple[float, float]],
    peaks_b: list[tuple[float, float]],
    mz_tolerance: float,
    shift: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Vectorised greedy match.

    Original ASFAM implementation was a pure-Python double loop —
    O(N*M) per call with high constant. Library matching calls this
    twice per (feature, library-spectrum) pair, so on a 50 K-spectrum
    library it dominates annotation time. Numpy broadcasting reduces
    the inner loop to one bool mask + ``np.where``, which empirically
    gives ~10x speedup on real DDA query sizes (~100 query peaks vs
    ~30 reference peaks).
    """
    n_a = len(peaks_a)
    n_b = len(peaks_b)
    if n_a == 0 or n_b == 0:
        return []

    mz_a = np.fromiter((p[0] for p in peaks_a), dtype=np.float64, count=n_a)
    int_a = np.fromiter((p[1] for p in peaks_a), dtype=np.float64, count=n_a)
    mz_b = np.fromiter((p[0] for p in peaks_b), dtype=np.float64, count=n_b)
    int_b = np.fromiter((p[1] for p in peaks_b), dtype=np.float64, count=n_b)

    # |mz_a - mz_b - shift| <= tol; broadcast across all pairs
    diff = np.abs(mz_a[:, None] - mz_b[None, :] - shift)
    i_arr, j_arr = np.where(diff <= mz_tolerance)
    if i_arr.size == 0:
        return []

    products = int_a[i_arr] * int_b[j_arr]
    # Sort candidate pairs by product descending (greedy keeps the
    # highest-product unique assignment).
    order = np.argsort(-products, kind="stable")

    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for k in order:
        i = int(i_arr[k])
        j = int(j_arr[k])
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append((peaks_a[i], peaks_b[j]))
    return matched


def _greedy_match_arrays(
    mz_a: np.ndarray,
    int_a: np.ndarray,
    mz_b: np.ndarray,
    int_b: np.ndarray,
    mz_tolerance: float,
    shift: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Array-input twin of :func:`_greedy_match_pairs`.

    Identical greedy logic, but takes pre-built ``float64`` arrays instead of
    Python ``(mz, intensity)`` tuple lists. The library-matching hot loop
    builds the query arrays once per feature and reuses one (query, reference)
    array pair across the direct and neutral-loss frames, eliminating the
    per-(feature, candidate) ``np.fromiter`` round-trip that
    :func:`_greedy_match_pairs` pays on every call. Returns the same matched
    ``((mz_q, int_q), (mz_r, int_r))`` pairs, in the same order, with the same
    values — so the downstream WDP/SDP/RDP/Matched% are byte-identical.
    """
    n_a = mz_a.shape[0]
    n_b = mz_b.shape[0]
    if n_a == 0 or n_b == 0:
        return []

    diff = np.abs(mz_a[:, None] - mz_b[None, :] - shift)
    i_arr, j_arr = np.where(diff <= mz_tolerance)
    if i_arr.size == 0:
        return []

    products = int_a[i_arr] * int_b[j_arr]
    order = np.argsort(-products, kind="stable")

    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for k in order:
        i = int(i_arr[k])
        j = int(j_arr[k])
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append(
            ((float(mz_a[i]), float(int_a[i])),
             (float(mz_b[j]), float(int_b[j])))
        )
    return matched


# ---------------------------------------------------------------------------
# Weighted dot-product distance with intensity-ratio fidelity boost
# (legacy WTV2 algorithm; used by apps/gcms_processor companion tools)
# ---------------------------------------------------------------------------

def weighted_dot_product_distance(
    compare_df,
    fr_factor: int,
    *,
    intensity_exp: float = 0.5,
    mz_exp: float = 2.0,
) -> float:
    """Composite weighted dot product (WDPD) with optional FR boost.

    Inputs
    ------
    compare_df : pandas.DataFrame
        Two-column DataFrame indexed by m/z. Column 0 = query intensities;
        column 1 = reference intensities. The index must be numeric (or
        cast-able to float).
    fr_factor : int
        Minimum number of shared peaks required to apply the FR (peak
        intensity ratio fidelity) boost. If shared peaks < fr_factor, the
        plain weighted cosine ``ss`` is returned.

    Notes
    -----
    The weighted vectors are ``w = intensity^intensity_exp * mz^mz_exp``;
    the legacy WTV2 defaults are ``intensity_exp=0.5, mz_exp=2.0``. The FR
    boost averages a clamped intensity-ratio consistency across consecutive
    shared peaks, then blends it with the cosine via
    ``(NU*ss + m*ave_FR) / (NU + m)`` where ``NU = total compared peaks``
    and ``m = shared peak count``.

    This function is shared between the GC-MS library builder and method
    generator. Do not duplicate.
    """
    import numpy as _np
    import pandas as _pd

    m_q = _pd.Series(compare_df.index).astype(float).to_numpy()
    i_q = _np.asarray(compare_df.iloc[:, 0], dtype=float)
    i_r = _np.asarray(compare_df.iloc[:, 1], dtype=float)

    w_q = _np.power(i_q, intensity_exp) * _np.power(m_q, mz_exp)
    w_r = _np.power(i_r, intensity_exp) * _np.power(m_q, mz_exp)

    sum_q = float(_np.sum(w_q))
    sum_r = float(_np.sum(w_r))
    if sum_q == 0.0 or sum_r == 0.0:
        ss = 0.0
    else:
        denom = float(_np.sum(w_q ** 2) * _np.sum(w_r ** 2))
        ss = float(_np.sum(w_q * w_r) ** 2 / denom) if denom > 0 else 0.0

    shared_mask = (i_q != 0) & (i_r != 0)
    m = int(_np.sum(shared_mask))
    if m < fr_factor or m < 2:
        return ss

    iq_shared = i_q[shared_mask]
    ir_shared = i_r[shared_mask]
    # Ratio consistency across consecutive shared peaks; clamp s ∈ (0, 1].
    ratios = (iq_shared[1:] / iq_shared[:-1]) * (ir_shared[:-1] / ir_shared[1:])
    # Replace 0 / inf / nan with neutral 1 to avoid divide-by-zero (legacy
    # would raise; we degrade gracefully).
    with _np.errstate(divide="ignore", invalid="ignore"):
        ratios = _np.where(_np.isfinite(ratios), ratios, 1.0)
    ratios = _np.where(ratios > 1.0, 1.0 / ratios, ratios)
    ave_fr = float(_np.sum(ratios) / (m - 1))

    nu = int(len(compare_df))
    return (nu * ss + m * ave_fr) / (nu + m)


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
