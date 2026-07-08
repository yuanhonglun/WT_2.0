"""Regression tests for shared similarity helpers."""
import math

import numpy as np
import pandas as pd

from metabo_core.algorithms.similarity import (
    composite_similarity_breakdown,
    cosine_similarity,
    modified_cosine,
    neutral_loss_cosine,
    ms2_isotope_step_score,
    weighted_dot_product_distance,
)


def test_cosine_identical_spectra_scores_one():
    peaks = [(50.0, 100.0), (75.0, 50.0), (100.0, 25.0)]
    score, n = cosine_similarity(peaks, peaks, mz_tolerance=0.01)
    assert math.isclose(score, 1.0, abs_tol=1e-9)
    assert n == 3


def test_cosine_disjoint_spectra_score_zero():
    score, n = cosine_similarity(
        [(50.0, 100.0)], [(75.0, 100.0)], mz_tolerance=0.01,
    )
    assert score == 0.0
    assert n == 0


def test_modified_cosine_identical_spectra_score_one():
    peaks = [(50.0, 1000.0), (60.0, 500.0), (70.0, 250.0)]
    score, n = modified_cosine(
        peaks, peaks, precursor_a=200.0, precursor_b=200.0, mz_tolerance=0.01,
    )
    assert n == 3
    assert math.isclose(score, 1.0, abs_tol=1e-9)


def test_neutral_loss_cosine_aligns_neutral_losses():
    a = [(50.0, 100.0), (60.0, 50.0)]
    b = [(150.0, 100.0), (160.0, 50.0)]
    score, n = neutral_loss_cosine(a, b, precursor_a=70.0, precursor_b=170.0, mz_tolerance=0.01)
    assert n == 2
    assert math.isclose(score, 1.0, abs_tol=1e-9)


def test_ms2_isotope_step_score_counts_echoes():
    lighter = [(100.0, 1000.0), (200.0, 500.0)]
    heavier = [(101.003355, 1000.0), (201.003355, 500.0)]
    matched, considered = ms2_isotope_step_score(lighter, heavier)
    assert matched == 2
    assert considered == 2


# ---------------------------------------------------------------------------
# weighted_dot_product_distance (legacy WTV2 algorithm; used by
# apps/gcms_processor method generator and library builder)
# ---------------------------------------------------------------------------

def _legacy_wdpd(compare_df: pd.DataFrame, fr_factor: int) -> float:
    """Bit-for-bit copy of the legacy CombineRtMsp.weighted_dot_product_distance.

    Used as a reference oracle in the test below. Do not use in production.
    """
    m_q = pd.Series(compare_df.index).astype(float)
    i_q = np.array(compare_df.iloc[:, 0])
    i_r = np.array(compare_df.iloc[:, 1])
    k = 0.5
    l = 2
    w_q = np.power(i_q, k) * np.power(m_q, l)
    w_r = np.power(i_r, k) * np.power(m_q, l)
    if np.sum(w_q) == 0 or np.sum(w_r) == 0:
        ss = 0.0
    else:
        ss = (np.sum(w_q * w_r) ** 2) / (np.sum(w_q ** 2) * np.sum(w_r ** 2))
    shared = pd.DataFrame(np.vstack((i_q, i_r)))
    shared = shared.loc[:, (shared != 0).all(axis=0)]
    m = int(shared.shape[1])
    if m >= fr_factor:
        FR = 0.0
        for i in range(1, m):
            s = (shared.iat[0, i] / shared.iat[0, i - 1]) * (
                shared.iat[1, i - 1] / shared.iat[1, i]
            )
            if s > 1:
                s = 1 / s
            FR += s
        ave_FR = FR / (m - 1)
        NU = int(len(compare_df))
        return ((NU * ss) + (m * ave_FR)) / (NU + m)
    return ss


def test_weighted_dot_product_distance_identical_spectra_scores_one():
    df = pd.DataFrame({0: [100, 500, 999], 1: [100, 500, 999]}, index=[50, 75, 100])
    score = weighted_dot_product_distance(df, fr_factor=2)
    assert math.isclose(score, 1.0, abs_tol=1e-9)


def test_weighted_dot_product_distance_disjoint_spectra():
    """No shared peaks → ss == 0 (no FR boost since shared count == 0)."""
    df = pd.DataFrame({0: [100, 0], 1: [0, 500]}, index=[50, 75])
    score = weighted_dot_product_distance(df, fr_factor=2)
    assert score == 0.0


def test_weighted_dot_product_distance_matches_legacy_oracle():
    """Random-but-fixed spectrum agrees with the legacy implementation."""
    rng = np.random.default_rng(42)
    mz = list(range(40, 80))
    iq = rng.integers(low=0, high=999, size=len(mz)).tolist()
    ir = rng.integers(low=0, high=999, size=len(mz)).tolist()
    df = pd.DataFrame({0: iq, 1: ir}, index=mz)

    legacy = _legacy_wdpd(df, fr_factor=2)
    new = weighted_dot_product_distance(df, fr_factor=2)
    assert math.isclose(new, legacy, abs_tol=1e-9)


def test_weighted_dot_product_distance_fr_factor_branching():
    """If shared-peak count < fr_factor, score collapses to ss only."""
    # Two ions; one peak shared (the second). m == 1, so fr_factor=2 → no FR boost.
    df = pd.DataFrame({0: [100, 200], 1: [0, 200]}, index=[50, 75])
    legacy = _legacy_wdpd(df, fr_factor=2)
    new = weighted_dot_product_distance(df, fr_factor=2)
    assert math.isclose(new, legacy, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# composite_similarity_breakdown: MS-DIAL TotalScore (PR-B / Task B2)
# ---------------------------------------------------------------------------

def test_totalscore_identical_spectrum_equals_two():
    # 6 个相同峰: penalty(6)=1.0, wdp=sdp=rdp=1.0, matched_pct=1.0
    # TotalScore = (1+1+1)/3 + 1.0 = 2.0（无 /2）
    peaks = [(100.0, 50.0), (150.0, 80.0), (200.0, 100.0),
             (250.0, 60.0), (300.0, 40.0), (350.0, 90.0)]
    r = composite_similarity_breakdown(peaks, peaks, mz_tolerance=0.02)
    assert abs(r.wdp - 1.0) < 1e-6
    assert abs(r.sdp - 1.0) < 1e-6
    assert abs(r.rdp - 1.0) < 1e-6
    assert abs(r.matched_pct - 1.0) < 1e-9
    assert abs(r.total_score - 2.0) < 1e-6
    assert abs(r.score - r.total_score) < 1e-12   # score 即 total_score（不钳制）


def test_matched_pct_is_matched_over_significant_ref():
    # query 含 ref 的一半峰 → matched_pct = 3/6 = 0.5（全部 ref 峰均显著）
    ref = [(100.0, 50.0), (150.0, 80.0), (200.0, 100.0),
           (250.0, 60.0), (300.0, 40.0), (350.0, 90.0)]
    query = [(100.0, 50.0), (200.0, 100.0), (300.0, 40.0)]
    r = composite_similarity_breakdown(query, ref, mz_tolerance=0.02)
    assert abs(r.matched_pct - 0.5) < 1e-9
    # total_score = (sqrt(wdp)+sqrt(sdp)+sqrt(rdp))/3 + 0.5 ∈ [0,2]
    expected = (math.sqrt(r.wdp) + math.sqrt(r.sdp) + math.sqrt(r.rdp)) / 3.0 + r.matched_pct
    assert abs(r.total_score - expected) < 1e-9
    assert 0.0 <= r.total_score <= 2.0


def test_matched_pct_denominator_uses_significant_ref_peaks():
    # ref 有 1 个 <1% 基峰的噪声峰(强度 0.5 vs 基峰 100)，不计入分母。
    # 显著 ref 峰 = 2 个(100,50)；query 命中这 2 个 → matched_pct = 2/2 = 1.0，而非 2/3。
    ref = [(100.0, 100.0), (150.0, 50.0), (999.0, 0.5)]
    query = [(100.0, 100.0), (150.0, 50.0)]
    r = composite_similarity_breakdown(query, ref, mz_tolerance=0.02)
    assert abs(r.matched_pct - 1.0) < 1e-9


def test_totalscore_not_clamped_above_one():
    # 恒等谱 total_score≈2.0 必须 >1（验证未被 min(.,1) 钳制）。
    peaks = [(100.0, 50.0), (150.0, 80.0), (200.0, 100.0),
             (250.0, 60.0), (300.0, 40.0), (350.0, 90.0)]
    r = composite_similarity_breakdown(peaks, peaks, mz_tolerance=0.02)
    assert r.total_score > 1.0


def test_matched_pct_bounded_when_query_matches_subthreshold_ref_noise():
    # ref: 1 显著基峰 + 3 个 <1% 基峰(0.5 vs 基峰 100)的噪声峰；query 命中全部 4 个。
    # MS-DIAL counter 只数显著匹配 → matched_pct = 1/1 = 1.0（不是 4/1=4.0）。
    ref = [(100.0, 100.0), (200.0, 0.5), (300.0, 0.5), (400.0, 0.5)]
    query = [(100.0, 100.0), (200.0, 0.5), (300.0, 0.5), (400.0, 0.5)]
    r = composite_similarity_breakdown(query, ref, mz_tolerance=0.02)
    assert abs(r.matched_pct - 1.0) < 1e-9
    assert r.matched_pct <= 1.0
    assert r.total_score <= 2.0


def test_rt_term_is_additive():
    # use_rt 时 total = 谱total + rt_sim（加和）。差值应恰为 rt_sim。
    peaks = [(50.0, 100.0), (75.0, 50.0)]
    no_rt = composite_similarity_breakdown(peaks, peaks, mz_tolerance=0.01)
    with_rt = composite_similarity_breakdown(
        peaks, peaks, mz_tolerance=0.01,
        rt_query=10.0, rt_ref=10.5, rt_tolerance=30.0, use_rt=True)
    rt_sim = math.exp(-0.5)  # gaussian(10.0, 10.5, 30s/60=0.5min)
    assert abs((with_rt.total_score - no_rt.total_score) - rt_sim) < 1e-9
