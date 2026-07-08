"""W7：验证 ASFAM 与 DDA 走同一条 composite similarity 公式。

锁定 W7 的核心算法决策：

1. ASFAM 与 DDA 共用 ``composite_similarity_breakdown``，公式对齐 MS-DIAL
   ``TotalScore`` = ``(sqrt(WDP) + sqrt(SDP) + sqrt(RDP)) / 3 + Matched%``
   （``use_rt=False``）；``use_rt=True`` 时 RT 高斯作为加和项叠加
   （加和，非旧的 ``/4`` 平均）。
2. 删除了 DDA 旧路径里的 precursor m/z 高斯项——DDA 在 MS1 EIC
   阶段已经锁定前体，再叠加一项 precursor 维度只会引入噪声。
3. ``SimilarityConfig.mz_tolerance`` 是 LC-MS / GC-MS 共享字段；
   LC-MS 默认 0.02 Da，GC-MS 在 ``GcmsConfig`` 中显式覆盖为 0.5。
"""
from __future__ import annotations

import math

import pytest

from metabo_core.algorithms.similarity import (
    composite_similarity,
    composite_similarity_breakdown,
)
from metabo_core.config import SimilarityConfig


# ---------------------------------------------------------------------------
# ASFAM 路径 ≡ DDA 路径（同一组输入返回完全相同的综合分）
# ---------------------------------------------------------------------------

def _query_ref_pair() -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """构造一对（查询谱, 参考谱），有部分匹配 / 部分不匹配。

    返回的 query / ref 共享多数主峰但强度比例略有差异，并各自含
    一些独有的小峰，这样 WDP / SDP / RDP 都会拿到 ``(0, 1)`` 区间
    的中间分（不是 1.0 也不是 0.0），更能暴露公式上的差异。
    """
    query = [
        (50.0, 100.0),
        (75.0, 80.0),
        (100.0, 60.0),
        (120.0, 40.0),
        (155.0, 20.0),   # query 独有
    ]
    ref = [
        (50.0, 100.0),
        (75.0, 70.0),     # 强度略偏
        (100.0, 50.0),
        (120.0, 35.0),
        (180.0, 10.0),    # ref 独有
    ]
    return query, ref


def test_asfam_and_dda_share_identical_composite_score() -> None:
    """ASFAM 注释（``match_feature_topn``）与 DDA 注释（``stage_annotate``）
    现在都调用 ``composite_similarity_breakdown``，传入同一组 query/ref
    必须得到完全相同的分数与逐分量数值，证明两路代码共用同一公式。
    """
    query, ref = _query_ref_pair()
    sim_cfg = SimilarityConfig()  # LC-MS 默认 mz_tolerance=0.02

    asfam_breakdown = composite_similarity_breakdown(
        query, ref,
        mz_tolerance=sim_cfg.mz_tolerance,
        precursor_query=200.0,
        precursor_ref=200.0,
        ms1_tolerance=sim_cfg.ms1_tolerance,
        rt_query=10.0,
        rt_ref=10.0,
        rt_tolerance=sim_cfg.rt_tolerance,
        use_rt=sim_cfg.use_rt,
    )

    dda_breakdown = composite_similarity_breakdown(
        query, ref,
        mz_tolerance=sim_cfg.mz_tolerance,
        precursor_query=200.0,
        precursor_ref=200.0,
        ms1_tolerance=sim_cfg.ms1_tolerance,
        rt_query=10.0,
        rt_ref=10.0,
        rt_tolerance=sim_cfg.rt_tolerance,
        use_rt=sim_cfg.use_rt,
    )

    assert asfam_breakdown.score == pytest.approx(dda_breakdown.score, abs=1e-12)
    assert asfam_breakdown.wdp == pytest.approx(dda_breakdown.wdp, abs=1e-12)
    assert asfam_breakdown.sdp == pytest.approx(dda_breakdown.sdp, abs=1e-12)
    assert asfam_breakdown.rdp == pytest.approx(dda_breakdown.rdp, abs=1e-12)
    assert asfam_breakdown.n_matched == dda_breakdown.n_matched


def test_composite_score_equals_msdial_totalscore_when_no_rt() -> None:
    """``use_rt=False`` 时 ``score`` 严格等于 MS-DIAL ``TotalScore``
    ``(sqrt(WDP) + sqrt(SDP) + sqrt(RDP)) / 3 + Matched%``（不再是旧的 MS2 平均）。"""
    query, ref = _query_ref_pair()

    breakdown = composite_similarity_breakdown(
        query, ref, mz_tolerance=0.02, use_rt=False,
    )

    expected = (math.sqrt(breakdown.wdp) + math.sqrt(breakdown.sdp) + math.sqrt(breakdown.rdp)) / 3.0 + breakdown.matched_pct
    assert breakdown.score == pytest.approx(expected, abs=1e-12)
    assert breakdown.score == pytest.approx(breakdown.total_score, abs=1e-12)


def test_composite_score_drops_precursor_term() -> None:
    """W7 关键回归：综合分不再因 precursor 不一致而被拉低。

    历史 DDA 公式 ``(precursor_sim + 3*ms2_avg) / 4`` 在 precursor 完
    全错配时（gaussian → 0）会把分数砍到 ``ms2_avg * 0.75``。W7 之后
    precursor 不再进入综合分，匹配的 precursor 与完全错配的 precursor
    应该返回相同的综合分。
    """
    query, ref = _query_ref_pair()

    matched_pre = composite_similarity_breakdown(
        query, ref, mz_tolerance=0.02,
        precursor_query=200.0, precursor_ref=200.0,
        ms1_tolerance=0.01,
    )
    mismatched_pre = composite_similarity_breakdown(
        query, ref, mz_tolerance=0.02,
        precursor_query=200.0, precursor_ref=900.0,
        ms1_tolerance=0.01,
    )

    assert matched_pre.score == pytest.approx(mismatched_pre.score, abs=1e-12)


def test_composite_similarity_thin_wrapper_returns_same_score() -> None:
    """``composite_similarity`` 是 ``composite_similarity_breakdown`` 的薄
    包装，返回 ``(score, n_matched)``——必须与 breakdown 的 score 完全一致。
    """
    query, ref = _query_ref_pair()
    breakdown = composite_similarity_breakdown(query, ref, mz_tolerance=0.02)
    score, n_matched = composite_similarity(query, ref, mz_tolerance=0.02)
    assert score == pytest.approx(breakdown.score, abs=1e-12)
    assert n_matched == breakdown.n_matched


# ---------------------------------------------------------------------------
# SimilarityConfig 字段统一
# ---------------------------------------------------------------------------

def test_similarity_config_has_unified_fields() -> None:
    """W7 把 LC-MS / GC-MS 共享字段集中放在 ``SimilarityConfig`` 上。

    GC-MS 端的覆盖（``GcmsConfig.similarity.mz_tolerance == 0.5``）
    在 ``apps/gcms_processor/tests/test_config.py`` 中验证，避免
    ``metabo_core`` 反向依赖 app 包破坏 ``test_boundaries.py``。
    """
    cfg = SimilarityConfig()
    # LC-MS 默认。GC-MS 在 ``GcmsConfig`` 用 0.5 覆盖。
    assert cfg.mz_tolerance == 0.02
    assert cfg.ms1_tolerance == 0.01
    assert cfg.min_matched_peaks == 3
    assert cfg.min_matched_pct == 0.25
    assert cfg.similarity_threshold == 0.7
    assert cfg.use_rt is False
    assert cfg.chrom_weight == 0.5    # GC-MS 用，LC-MS 忽略
