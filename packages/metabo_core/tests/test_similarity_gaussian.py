"""Tests pinning the standalone ``gaussian_similarity`` helper.

The helper is factored out of :func:`composite_similarity` so both LC-MS
(via the existing composite path) and GC-MS (via Plan D
``gcms_match_factor``) can call the same shape.
"""
from __future__ import annotations

import math

import pytest

from metabo_core.algorithms.similarity import (
    composite_similarity,
    gaussian_similarity,
)


# ---------------------------------------------------------------------------
# Shape pinning
# ---------------------------------------------------------------------------

def test_gaussian_similarity_value_equals_ref_returns_one() -> None:
    assert gaussian_similarity(10.0, 10.0, tolerance=0.1) == pytest.approx(1.0, abs=1e-12)


def test_gaussian_similarity_one_tolerance_away_returns_exp_minus_half() -> None:
    val = gaussian_similarity(10.1, 10.0, tolerance=0.1)
    assert val == pytest.approx(math.exp(-0.5), abs=1e-9)
    assert val == pytest.approx(0.6065, abs=1e-3)


def test_gaussian_similarity_two_tolerances_away_returns_exp_minus_two() -> None:
    val = gaussian_similarity(10.2, 10.0, tolerance=0.1)
    assert val == pytest.approx(math.exp(-2.0), abs=1e-9)
    assert val == pytest.approx(0.1353, abs=1e-3)


def test_gaussian_similarity_is_symmetric_in_value_and_ref() -> None:
    a = gaussian_similarity(10.05, 10.0, tolerance=0.1)
    b = gaussian_similarity(10.0, 10.05, tolerance=0.1)
    assert a == pytest.approx(b, abs=1e-12)


def test_gaussian_similarity_returns_zero_for_none_inputs() -> None:
    assert gaussian_similarity(None, 10.0, tolerance=0.1) == 0.0
    assert gaussian_similarity(10.0, None, tolerance=0.1) == 0.0
    assert gaussian_similarity(None, None, tolerance=0.1) == 0.0


def test_gaussian_similarity_returns_zero_for_nonpositive_tolerance() -> None:
    assert gaussian_similarity(10.0, 10.0, tolerance=0.0) == 0.0
    assert gaussian_similarity(10.0, 10.0, tolerance=-1.0) == 0.0


# ---------------------------------------------------------------------------
# composite_similarity must remain numerically identical after refactor
# ---------------------------------------------------------------------------

def test_composite_similarity_three_peak_identity_returns_msdial_totalscore() -> None:
    """对齐 MS-DIAL ``Total score``：``(sqrt(WDP)+sqrt(SDP)+sqrt(RDP))/3 + Matched%``。

    3 个等同峰各自 WDP=SDP=RDP=1.0×penalty(3)=0.94；sqrt(0.94)=0.9695360；
    三个参考峰均显著且全部命中 → Matched%=1.0；``use_rt=False``。
    因此 TotalScore = 0.9695360 + 1.0 ≈ 1.96954（无 /2、无钳制）。
    """
    peaks = [(50.0, 100.0), (75.0, 50.0), (100.0, 25.0)]
    score, n = composite_similarity(peaks, peaks, mz_tolerance=0.01)
    assert score == pytest.approx(1.96954, abs=1e-4)
    assert n == 3


def test_composite_similarity_rt_branch_uses_seconds_to_minutes_hack() -> None:
    """LC-MS RT 路径将 ``rt_tolerance`` 除以 60 后再交给高斯函数。

    rt_query=10.0 min, rt_ref=10.5 min, rt_tolerance=30 秒 (=> 0.5 min)
    => 高斯: exp(-0.5 * (0.5/0.5)^2) = exp(-0.5) ≈ 0.6065307

    对齐 MS-DIAL ``GetTotalScore``：RT 高斯是**加和项**，不再是旧的 ``/4`` 平均。
    两个相同峰：WDP=SDP=RDP=1.0×penalty(2)=0.88；sqrt(0.88)=0.9380832；
    Matched%=1.0 → spectral_total = 0.9380832 + 1.0 ≈ 1.93808。
    => score = 1.93808 + 0.6065307 ≈ 2.54461。
    """
    peaks = [(50.0, 100.0), (75.0, 50.0)]
    score_with, _ = composite_similarity(
        peaks, peaks,
        mz_tolerance=0.01,
        precursor_query=200.0, precursor_ref=200.0, ms1_tolerance=0.01,
        rt_query=10.0, rt_ref=10.5, rt_tolerance=30.0,
        use_rt=True,
    )
    assert score_with == pytest.approx(2.54461, abs=1e-4)


def test_composite_similarity_no_rt_default_unchanged() -> None:
    """``use_rt=False`` 时 RT 项不进入综合分。

    两个相同峰：WDP=SDP=RDP=0.88；sqrt(0.88)=0.9380832；Matched%=1.0。
    => score = (sqrt(0.88))*3/3 + 1.0 = 0.9380832 + 1.0 ≈ 1.93808。
    """
    peaks = [(50.0, 100.0), (75.0, 50.0)]
    score, _ = composite_similarity(
        peaks, peaks,
        mz_tolerance=0.01,
        precursor_query=200.0, precursor_ref=200.0, ms1_tolerance=0.01,
        rt_query=10.0, rt_ref=10.5, rt_tolerance=30.0,
        use_rt=False,
    )
    assert score == pytest.approx(1.93808, abs=1e-4)
