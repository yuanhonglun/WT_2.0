"""共享 mzML 读取工具。

本模块下沉了 ASFAM / DDA / GC-MS 三个 app 中重复出现的 pymzml 读取逻辑：

- :func:`iter_scans`：单遍迭代 mzML 文件，按 ``ms_level`` 过滤后返回
  :class:`metabo_core.models.Scan` 对象。MS2 scan 会附带 precursor_mz、
  precursor_intensity 以及 isolation window 上下界。
- :func:`_scan_rt_minutes`：从 pymzml spectrum 提取保留时间（分钟），按
  ``scan_time_in_minutes()`` → CV ``MS:1000016`` → scan index 顺序降级。
- :func:`_spectrum_arrays`：从 pymzml spectrum 提取 ``(mz, intensity)``，
  均以 ``float64`` 数组返回。
- :func:`_precursor_mz_intensity`：DDA 用，按 ``selected_precursors`` →
  CV ``MS:1000827`` → CV ``MS:1000744`` 的顺序解析前体 m/z；
  precursor intensity 使用 CV ``MS:1000042``。
- :func:`_isolation_window_bounds`：DDA 用，根据 CV ``MS:1000827/828/829``
  计算 ``[target - lower_offset, target + upper_offset]``。

需要在 import pymzml **之前** import ``urllib.request``。pymzml 在加载
ontology 时依赖动态 import ``urllib.request``，而 PyInstaller 打包的可执
行文件不会自动收集该子模块；提前 import 可以保证打包后的程序也能正常
读取 mzML。
"""
from __future__ import annotations

import urllib.request  # noqa: F401 — must be imported before pymzml for PyInstaller
from typing import Iterator, Optional

import numpy as np
import pymzml

from metabo_core.models import Scan


# ---------------------------------------------------------------------------
# CV accessions（PSI-MS 受控词表）
# ---------------------------------------------------------------------------

_CV_SCAN_TIME = "MS:1000016"             # scan start time
_CV_ISOLATION_TARGET = "MS:1000827"      # isolation window target m/z
_CV_ISOLATION_LOWER_OFFSET = "MS:1000828"  # isolation window lower offset
_CV_ISOLATION_UPPER_OFFSET = "MS:1000829"  # isolation window upper offset
_CV_SELECTED_ION_MZ = "MS:1000744"       # selected ion m/z
_CV_PEAK_INTENSITY = "MS:1000042"        # peak intensity (precursor intensity)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def iter_scans(
    filepath: str,
    ms_level: Optional[int] = None,
) -> Iterator[Scan]:
    """单遍流式迭代 mzML 文件，按需过滤 ms_level。

    Parameters
    ----------
    filepath : str
        mzML 文件路径。
    ms_level : Optional[int]
        若指定，则只返回该 MS 级别的 scan（1=MS1，2=MS2）；None 表示全部。

    Yields
    ------
    Scan
        :class:`metabo_core.models.Scan` 对象。对 MS2 scan，会额外填充
        ``precursor_mz``、``precursor_intensity``、``isolation_window_lower``、
        ``isolation_window_upper`` 字段（若仪器在 mzML 中有报告）。
    """
    run = pymzml.run.Reader(str(filepath))
    for scan_idx, spectrum in enumerate(run):
        level = spectrum.ms_level
        if level is None:
            continue
        if ms_level is not None and level != ms_level:
            continue

        rt_min = _scan_rt_minutes(spectrum)
        mz_array, int_array = _spectrum_arrays(spectrum)

        scan = Scan(
            scan_id=scan_idx,
            ms_level=int(level),
            rt=float(rt_min),
            mz_array=mz_array,
            intensity_array=int_array,
        )

        if level == 2:
            p_mz, p_int = _precursor_mz_intensity(spectrum)
            lo, hi = _isolation_window_bounds(spectrum, p_mz)
            scan.precursor_mz = p_mz
            scan.precursor_intensity = p_int
            scan.isolation_window_lower = lo
            scan.isolation_window_upper = hi

        yield scan


# ---------------------------------------------------------------------------
# Helpers — 作为下划线开头的 module-level 函数公开给三个 app 复用
# ---------------------------------------------------------------------------


def _scan_rt_minutes(spectrum) -> float:
    """从 pymzml spectrum 提取保留时间（分钟）。

    优先级：``spectrum.scan_time_in_minutes()`` → CV ``MS:1000016``（秒，
    需除以 60）→ 0.0。当所有路径都失败时返回 0.0；调用方若需要 scan index
    作为最终降级（如 GC-MS reader），需在 wrapper 中自行处理。
    """
    try:
        return float(spectrum.scan_time_in_minutes())
    except Exception:
        try:
            rt_sec = spectrum.get(_CV_SCAN_TIME)
            if rt_sec is not None:
                return float(rt_sec) / 60.0
        except Exception:
            pass
    return 0.0


def _spectrum_arrays(spectrum) -> tuple[np.ndarray, np.ndarray]:
    """从 pymzml spectrum 提取 ``(mz, intensity)`` 数组，均为 float64。

    使用 ``spectrum.peaks("raw")`` 读取原始 centroid 数据；空 spectrum 会
    返回两个长度为 0 的数组。
    """
    try:
        peaks = spectrum.peaks("raw")
    except Exception:
        peaks = None
    if peaks is None or len(peaks) == 0:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
    mz = np.asarray(peaks[:, 0], dtype=np.float64)
    intensity = np.asarray(peaks[:, 1], dtype=np.float64)
    return mz, intensity


def _precursor_mz_intensity(
    spectrum,
) -> tuple[Optional[float], Optional[float]]:
    """从 MS2 spectrum 提取 ``(precursor_mz, precursor_intensity)``。

    解析顺序：

    1. ``spectrum.selected_precursors[0]`` 同时给出 mz 和 intensity；
    2. CV ``MS:1000827``（isolation window target m/z）作为 mz 的兜底；
    3. CV ``MS:1000744``（selected ion m/z）作为 mz 的最终兜底；
    4. CV ``MS:1000042``（peak intensity）作为 intensity 的兜底。
    """
    p_mz: Optional[float] = None
    p_int: Optional[float] = None

    try:
        precursors = spectrum.selected_precursors
        if precursors:
            first = precursors[0]
            if "mz" in first and first["mz"] is not None:
                p_mz = float(first["mz"])
            if "i" in first and first["i"] is not None:
                p_int = float(first["i"])
    except (KeyError, TypeError, IndexError, AttributeError):
        pass

    if p_mz is None:
        try:
            target = spectrum.get(_CV_ISOLATION_TARGET)
            if target is not None:
                p_mz = float(target)
        except (KeyError, TypeError, ValueError):
            pass
    if p_mz is None:
        try:
            sel = spectrum.get(_CV_SELECTED_ION_MZ)
            if sel is not None:
                p_mz = float(sel)
        except (KeyError, TypeError, ValueError):
            pass

    if p_int is None:
        try:
            inten = spectrum.get(_CV_PEAK_INTENSITY)
            if inten is not None:
                p_int = float(inten)
        except (KeyError, TypeError, ValueError):
            pass

    return p_mz, p_int


def _isolation_window_bounds(
    spectrum,
    precursor_mz: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    """返回 isolation window 的 ``(lower, upper)`` 上下界（m/z）。

    mzML 同时记录 isolation target 和上下偏移量。若 target 缺失，会用
    ``precursor_mz`` 作为中心；若 lower/upper offset 缺失，则返回
    ``(None, None)``（无法可靠推断窗口宽度）。
    """
    target: Optional[float] = None
    lower_off: Optional[float] = None
    upper_off: Optional[float] = None

    try:
        target_v = spectrum.get(_CV_ISOLATION_TARGET)
        if target_v is not None:
            target = float(target_v)
    except (KeyError, TypeError, ValueError):
        pass
    try:
        lo_v = spectrum.get(_CV_ISOLATION_LOWER_OFFSET)
        if lo_v is not None:
            lower_off = float(lo_v)
    except (KeyError, TypeError, ValueError):
        pass
    try:
        up_v = spectrum.get(_CV_ISOLATION_UPPER_OFFSET)
        if up_v is not None:
            upper_off = float(up_v)
    except (KeyError, TypeError, ValueError):
        pass

    centre = target if target is not None else precursor_mz

    if centre is None:
        return None, None
    if lower_off is None or upper_off is None:
        return None, None

    return centre - lower_off, centre + upper_off
