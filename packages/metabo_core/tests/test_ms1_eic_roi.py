"""ROI 风格 MS1 EIC 构建器单元测试 (W3)。

覆盖 4 个场景:
1. LC 合成数据: 两个相距 50 mDa 的离子能被分到 2 个 ROI;
2. GC 合成数据: 单位质量 m/z=149 与 150 能被分到 2 个 ROI;
3. 漂移合并 (修复 DDA 当前 _build_mz_traces 漂移缺陷的关键测试):
   同一个离子在 10 个 scan 内 m/z 慢漂 10 ppm, 必须归为一个 ROI;
4. 真实 mzML smoke: DDA 文件能跑通且返回合理数量 ROI。
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from metabo_core.algorithms.ms1_eic_roi import (
    ROIConfig,
    ROIEIC,
    build_eics_roi,
)


# ---------------------------------------------------------------------------
# 辅助: 一个最小的 Scan-like 对象, 避免和 Scan dataclass 的其他必填字段耦合
# ---------------------------------------------------------------------------


class _FakeScan:
    """只暴露 build_eics_roi 需要的属性 (rt, mz_array, intensity_array)。"""

    __slots__ = ("rt", "mz_array", "intensity_array")

    def __init__(self, rt: float, mz: list[float] | np.ndarray, intens: list[float] | np.ndarray):
        self.rt = float(rt)
        self.mz_array = np.asarray(mz, dtype=np.float64)
        self.intensity_array = np.asarray(intens, dtype=np.float64)


# ---------------------------------------------------------------------------
# 测试 1: LC 合成数据, 两个明显的离子
# ---------------------------------------------------------------------------


def test_lc_two_ions_split_into_two_rois():
    """两个相距 0.05 Da (~250 ppm @ mz=200) 的离子应分为 2 个 ROI。"""
    scans = []
    n_scans = 10
    for i in range(n_scans):
        rt = 1.0 + i * 0.05
        # 两个稳定的离子: 200.05 和 200.10
        scans.append(_FakeScan(
            rt=rt,
            mz=[200.05, 200.10],
            intens=[1000.0, 800.0],
        ))

    cfg = ROIConfig(
        mode="lc_ppm",
        ppm_tolerance=15.0,
        min_eic_points=5,
        start_mz=199.0,
        end_mz=201.0,
    )
    rois = build_eics_roi(scans, cfg)
    assert len(rois) == 2, f"expected 2 ROIs, got {len(rois)}: {[r.mz_centroid for r in rois]}"

    centroids = sorted(r.mz_centroid for r in rois)
    assert abs(centroids[0] - 200.05) < 1e-4, centroids
    assert abs(centroids[1] - 200.10) < 1e-4, centroids

    # 强度应分别接近 1000 和 800
    by_mz = sorted(rois, key=lambda r: r.mz_centroid)
    assert abs(by_mz[0].intensity_array.max() - 1000.0) < 1e-6
    assert abs(by_mz[1].intensity_array.max() - 800.0) < 1e-6


# ---------------------------------------------------------------------------
# 测试 2: GC 合成数据, m/z=149 与 150
# ---------------------------------------------------------------------------


def test_gc_unit_mass_two_ions_split():
    """GC mode 下 m/z 149 和 150 应分到两个 ROI (slice 宽 0.5 Da)。"""
    scans = []
    n_scans = 8
    for i in range(n_scans):
        rt = 5.0 + i * 0.02
        scans.append(_FakeScan(
            rt=rt,
            mz=[149.0, 150.0],
            intens=[500.0, 300.0],
        ))

    cfg = ROIConfig(
        mode="gc_da",
        da_slice_width=0.5,
        overlap_fraction=0.5,
        min_eic_points=4,
        start_mz=140.0,
        end_mz=160.0,
    )
    rois = build_eics_roi(scans, cfg)
    assert len(rois) == 2, f"expected 2 ROIs, got {len(rois)}: {[r.mz_centroid for r in rois]}"

    centroids = sorted(r.mz_centroid for r in rois)
    assert abs(centroids[0] - 149.0) < 1e-3
    assert abs(centroids[1] - 150.0) < 1e-3


# ---------------------------------------------------------------------------
# 测试 3: 慢漂合并 — DDA 当前 _build_mz_traces 缺陷的关键测试
# ---------------------------------------------------------------------------


def test_slow_drift_stays_one_roi():
    """同一个离子在 10 scan 内 m/z 从 300.000 漂到 300.005 (约 16 ppm),
    必须归到一个 ROI。

    这是修复 legacy DDA ``_build_mz_traces`` 漂移问题的关键测试: 老算法把
    点按 m/z 一次性排序后线性聚簇, 慢漂会逐步把质心拉走从而把 trace 切
    成多段; 新的 slice+50% overlap+后期合并应该一气呵成保留单个 ROI。
    """
    n_scans = 10
    scans = []
    for i in range(n_scans):
        # 从 300.000 线性漂到 300.005 (~16.7 ppm 总漂移)
        mz = 300.000 + (0.005 * i / (n_scans - 1))
        scans.append(_FakeScan(
            rt=2.0 + i * 0.05,
            mz=[mz],
            intens=[1000.0 + i * 10],
        ))

    cfg = ROIConfig(
        mode="lc_ppm",
        ppm_tolerance=20.0,  # 让 slice 容得下 ~16 ppm 漂移
        min_eic_points=5,
        start_mz=299.0,
        end_mz=301.0,
    )
    rois = build_eics_roi(scans, cfg)
    assert len(rois) == 1, (
        f"slow drift must collapse to ONE ROI, got {len(rois)}: "
        f"{[r.mz_centroid for r in rois]}"
    )
    roi = rois[0]
    # centroid 应落在 [300.000, 300.005] 区间
    assert 300.000 <= roi.mz_centroid <= 300.006, roi.mz_centroid
    # 该 ROI 应覆盖大部分 scan; 后期合并保留强者, 边界 scan 可能损失 1-2 个,
    # 关键是不要被切成多段。
    assert roi.n_points >= n_scans - 2, (
        f"slow-drift ROI should retain most scans, got {roi.n_points}/{n_scans}"
    )


# ---------------------------------------------------------------------------
# 测试 4: 真实 DDA mzML smoke + 性能
# ---------------------------------------------------------------------------


# Local-data smoke test: point ``ASFAM_TEST_MZML`` at an mzML file to enable it.
# Skipped by default, since no raw data ships with this repository.
_REAL_MZML_ENV = os.environ.get("ASFAM_TEST_MZML", "")
_REAL_MZML = Path(_REAL_MZML_ENV) if _REAL_MZML_ENV else None


@pytest.mark.skipif(
    _REAL_MZML is None or not _REAL_MZML.exists(),
    reason="real mzML not present (set ASFAM_TEST_MZML to enable)",
)
def test_real_dda_mzml_smoke():
    """用真实 mzML 检验性能 + 输出量级。"""
    # 注意: 这里只在测试里读 mzML, 不违反 metabo_core 不依赖 app 的边界。
    file_path = _REAL_MZML

    ms1_scans = _read_ms1_scans_minimal(file_path)
    assert len(ms1_scans) > 0, "no MS1 scans read from mzML"

    cfg = ROIConfig(
        mode="lc_ppm",
        ppm_tolerance=15.0,
        min_eic_points=5,
        start_mz=50.0,
        end_mz=1100.0,
    )

    t0 = time.time()
    rois = build_eics_roi(ms1_scans, cfg)
    elapsed = time.time() - t0

    print(
        f"\n[ms1_eic_roi smoke] n_ms1_scans={len(ms1_scans)} "
        f"n_rois={len(rois)} elapsed={elapsed:.2f}s"
    )

    # 性能门: < 10s
    assert elapsed < 10.0, f"build_eics_roi too slow: {elapsed:.2f}s"
    # ROI 数量量级合理
    assert 100 <= len(rois) <= 50000, f"unexpected ROI count: {len(rois)}"


def _read_ms1_scans_minimal(file_path: Path):
    """用 pymzml 直接读 MS1 scan; 不依赖 apps.dda."""
    import urllib.request  # noqa: F401  (pymzml 兼容)
    import pymzml

    scans = []
    run = pymzml.run.Reader(str(file_path))
    for idx, spec in enumerate(run):
        if spec.ms_level != 1:
            continue
        try:
            rt = float(spec.scan_time_in_minutes())
        except Exception:
            rt = 0.0
        try:
            peaks = spec.peaks("raw")
        except Exception:
            peaks = None
        if peaks is None or len(peaks) == 0:
            mz = np.array([], dtype=np.float64)
            intens = np.array([], dtype=np.float64)
        else:
            mz = np.asarray(peaks[:, 0], dtype=np.float64)
            intens = np.asarray(peaks[:, 1], dtype=np.float64)
        scans.append(_FakeScan(rt=rt, mz=mz, intens=intens))
    return scans
