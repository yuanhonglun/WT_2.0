"""Tests for :mod:`metabo_core.io.mzml`.

测试在真实的 DDA mzML 文件上验证 iter_scans 的行为：

- 能正确解析出 MS1 / MS2 scan 流；
- DDA MS2 能解析出 precursor m/z；
- DDA MS2 能解析出 isolation window 上下界。

若测试数据不存在则整个文件 skip，方便在 CI 等缺数据环境运行。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from metabo_core.io.mzml import (
    iter_scans,
    _isolation_window_bounds,
    _precursor_mz_intensity,
    _scan_rt_minutes,
    _spectrum_arrays,
)
from metabo_core.models import Scan


# Local-data test: point ``ASFAM_TEST_MZML`` at a DDA mzML file to enable it.
# Skipped by default, since no raw data ships with this repository.
_REAL_DDA_ENV = os.environ.get("ASFAM_TEST_MZML", "")
REAL_DDA = Path(_REAL_DDA_ENV) if _REAL_DDA_ENV else None


pytestmark = pytest.mark.skipif(
    REAL_DDA is None or not REAL_DDA.exists(),
    reason="real DDA mzML not present (set ASFAM_TEST_MZML to enable)",
)


def test_iter_scans_yields_scan_objects_with_arrays():
    """iter_scans 应能产出 Scan 对象，且每张 scan 都带 mz/intensity 数组。"""
    n = 0
    for scan in iter_scans(str(REAL_DDA)):
        assert isinstance(scan, Scan)
        assert scan.ms_level in (1, 2)
        assert isinstance(scan.mz_array, np.ndarray)
        assert isinstance(scan.intensity_array, np.ndarray)
        assert scan.mz_array.dtype == np.float64
        assert scan.intensity_array.dtype == np.float64
        assert len(scan.mz_array) == len(scan.intensity_array)
        n += 1
        if n >= 50:
            break
    assert n > 0, "iter_scans yielded zero scans on real data"


def test_iter_scans_splits_ms1_and_ms2():
    """ms_level 过滤应分别只返回 MS1 / MS2。"""
    ms1_count = sum(1 for _ in iter_scans(str(REAL_DDA), ms_level=1))
    ms2_count = sum(1 for _ in iter_scans(str(REAL_DDA), ms_level=2))
    total_count = sum(1 for _ in iter_scans(str(REAL_DDA)))

    assert ms1_count > 0
    assert ms2_count > 0
    # 全量 = MS1 + MS2（DDA 数据里没有 MS3 等其它级别）
    assert ms1_count + ms2_count == total_count


def test_iter_scans_ms1_has_no_precursor_fields():
    """MS1 scan 不应被填充前体/隔离窗信息。"""
    for scan in iter_scans(str(REAL_DDA), ms_level=1):
        assert scan.precursor_mz is None
        assert scan.precursor_intensity is None
        assert scan.isolation_window_lower is None
        assert scan.isolation_window_upper is None
        break  # 一张就够


def test_iter_scans_ms2_has_precursor_and_isolation_window():
    """DDA MS2 scan 应当能解析出 precursor m/z 与 isolation window。"""
    seen_precursor = False
    seen_window = False
    for scan in iter_scans(str(REAL_DDA), ms_level=2):
        if scan.precursor_mz is not None:
            seen_precursor = True
            assert scan.precursor_mz > 0
        if (
            scan.isolation_window_lower is not None
            and scan.isolation_window_upper is not None
        ):
            seen_window = True
            assert scan.isolation_window_lower < scan.isolation_window_upper
        if seen_precursor and seen_window:
            break
    assert seen_precursor, "未在任何 MS2 scan 中解析到 precursor_mz"
    assert seen_window, "未在任何 MS2 scan 中解析到 isolation window 上下界"


def test_iter_scans_rt_is_monotonic_in_minutes():
    """连续 scan 的 RT 应单调非减且看起来像分钟而非秒。"""
    prev_rt = -1.0
    max_rt = 0.0
    n = 0
    for scan in iter_scans(str(REAL_DDA)):
        assert scan.rt >= prev_rt - 1e-6
        prev_rt = scan.rt
        max_rt = max(max_rt, scan.rt)
        n += 1
        if n >= 200:
            break
    # 真实 DDA 文件长度在分钟量级。若 RT 误把秒当分钟，会出现 >> 100 的值。
    assert max_rt < 200.0


# ---------------------------------------------------------------------------
# Helper-level smoke tests (don't need real data; just need the symbols to import)
# ---------------------------------------------------------------------------


def test_helpers_importable():
    """确保私有 helper 都能被 import（其它 reader 直接依赖它们）。"""
    assert callable(_scan_rt_minutes)
    assert callable(_spectrum_arrays)
    assert callable(_precursor_mz_intensity)
    assert callable(_isolation_window_bounds)


def test_isolation_window_returns_none_when_offsets_missing():
    """当 spectrum 没有 lower/upper offset 时应返回 (None, None)。"""

    class _FakeSpec:
        def get(self, _accession):
            return None

    lo, hi = _isolation_window_bounds(_FakeSpec(), precursor_mz=500.0)
    assert lo is None and hi is None
