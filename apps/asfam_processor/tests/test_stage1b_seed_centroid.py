"""T1 R1: MSDec 种子 0.025 Da 质心归并 (对齐 MS-DIAL Ms2Dec curatedSpectra)。

WHY (意图, 非表面行为):
  ``_collect_ms2_msdec`` 的产品离子种子原先直接取 apex 单 scan 的 vendor
  centroid 点 (``ap_mz[keep]``)。QTOF 会把同一真峰吐成间距 <0.025 Da 的
  相邻 centroid 点 (例 F03858: 148.99975 与 149.01975, Δ0.02)。这两个点被
  当成两个离子、各自抽同一条强色谱、被 MSDec 各回归各发一峰 → base 峰被劈成
  重复峰 → 与 MS-DIAL 谱的 cosine 被拉低 (F03858 0.71, F00451 0.60)。

  MS-DIAL Ms2Dec 在建种子时先做 0.025 Da 质心归并 (curatedSpectra):相邻
  Δm/z ≤ tol 的点合成一个离子, m/z = 峰内强度加权质心, intensity = 峰内 max。
  本测试锁定该语义:合并、峰内加权、不跨离子、健壮性。

不变量 (CLAUDE.md): 质心必须 intensity 加权 (非算术均值); 绝不跨离子平均
(否则重蹈 loader 旧 bug)。
"""
from __future__ import annotations

import numpy as np

from asfam.pipeline.stage1b_ms1_detection import _centroid_seed


def test_merges_adjacent_within_tol_weighted():
    """F03858 结构: 148.99975(184) + 149.01975(51690), Δ0.02 < 0.025 → 合一。

    合并后 m/z = 峰内强度加权质心 (被强峰主导, 逼近 149.01975);
    intensity = 峰内 max (51690), 不是求和、不是均值。
    """
    mz = np.array([148.99975, 149.01975])
    it = np.array([184.0, 51690.0])
    omz, oit = _centroid_seed(mz, it, 0.025)
    assert omz.size == 1
    # 强度加权: (148.99975*184 + 149.01975*51690)/(184+51690) ≈ 149.0197
    expected = (148.99975 * 184.0 + 149.01975 * 51690.0) / (184.0 + 51690.0)
    assert abs(omz[0] - expected) < 1e-6          # 精确等于峰内加权质心
    assert abs(omz[0] - 149.01975) < 5e-4         # 被强峰主导 (弱峰几乎不拉动)
    assert oit[0] == 51690.0                       # 峰内 max, 非 sum/mean


def test_weighted_not_arithmetic_mean():
    """两个等 m/z 间距、强度悬殊的点:加权质心必须偏向强峰, 而非落在算术中点。

    这是"intensity 加权 vs 算术均值"不变量的直接守门。
    """
    mz = np.array([100.000, 100.020])
    it = np.array([100.0, 900.0])       # 强峰在 100.020
    omz, _ = _centroid_seed(mz, it, 0.025)
    assert omz.size == 1
    arithmetic_mid = 100.010
    assert omz[0] > arithmetic_mid                 # 偏向强峰, 不是中点
    assert abs(omz[0] - 100.018) < 1e-6            # (100*100 + 100.02*900)/1000


def test_keeps_distinct_ions_beyond_tol():
    """间距 > tol 的点保持独立, 不被合并。"""
    mz = np.array([100.00, 100.05, 149.02])
    it = np.array([10.0, 10.0, 500.0])
    omz, _ = _centroid_seed(mz, it, 0.025)
    assert omz.size == 3                            # 0.05 > 0.025 → 不合并


def test_no_cross_ion_averaging():
    """相隔远的两离子绝不被平均到中间 (loader 旧 bug 的回归护栏)。"""
    mz = np.array([120.0, 140.0])
    it = np.array([100.0, 100.0])
    omz, _ = _centroid_seed(mz, it, 0.025)
    assert set(np.round(omz, 3).tolist()) == {120.0, 140.0}


def test_chain_merge_within_tol():
    """一串相邻点 (每步 <tol) 应合成一个离子, m/z 为整簇加权质心。

    三点 100.00/100.02/100.04, 每步 Δ0.02 < 0.025 → 链式合一。
    """
    mz = np.array([100.00, 100.02, 100.04])
    it = np.array([100.0, 800.0, 100.0])
    omz, oit = _centroid_seed(mz, it, 0.025)
    assert omz.size == 1
    expected = (100.00 * 100 + 100.02 * 800 + 100.04 * 100) / 1000.0
    assert abs(omz[0] - expected) < 1e-6           # 峰内加权质心
    assert oit[0] == 800.0                          # 峰内 max


def test_unsorted_input_is_sorted():
    """乱序输入应被排序后处理, 输出 m/z 升序。"""
    mz = np.array([149.02, 100.00, 120.00])
    it = np.array([500.0, 10.0, 20.0])
    omz, _ = _centroid_seed(mz, it, 0.025)
    assert list(omz) == sorted(omz.tolist())


def test_empty_input_returns_empty():
    """空输入返回空 (健壮性)。"""
    omz, oit = _centroid_seed(
        np.array([], dtype=np.float64), np.array([], dtype=np.float64), 0.025
    )
    assert omz.size == 0 and oit.size == 0
