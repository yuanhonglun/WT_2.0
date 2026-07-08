"""LC-MS MS2 谱图清洗（ASFAM / DDA 共用）。

此模块把 ASFAM Stage 1 与 DDA Stage 2 中重复出现的 MS2 后处理步骤
统一到一个流水线里：

    1. 近邻 m/z 合并（自适应容差，强度加权 centroid）
    2. 平顶噪声删除（多个离子等强度的仪器基线签名）
    3. 强度截断（绝对阈值 AND 相对阈值）
    4. 前体后离子删除（保留前体 + isotope_range 之内）
    5. 取 top_n 最强离子
    6. 按 m/z 升序排序

GC-MS 不调用此模块——deconvolution 输出后处理在
``metabo_core.gcms`` 中单独维护，参数体系完全不同。

仅依赖 numpy 向量化：``np.searchsorted`` / ``np.unique`` /
``np.argpartition`` 等，禁止 Python 双层循环。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from metabo_core.algorithms.peak_merge import merge_close_ions


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MS2CleanupConfig:
    """LC-MS MS2 谱图清洗参数（ASFAM/DDA 共用）。

    默认值取自 ASFAM 的较严格设定（绝对 1000 + 相对 2%），DDA 沿用同样
    的默认；如果某个 app 需要放宽，构造时显式传更松的值即可。
    """

    # 自适应合并容差: max(absolute_tol, mz × ppm/1e6)
    merge_absolute_tol: float = 0.02       # Da
    merge_ppm_tol: float = 100.0           # ppm

    # 强度截断（双重 AND）
    absolute_intensity_threshold: float = 1000.0
    relative_intensity_threshold: float = 0.02   # 2% of base peak

    # 平顶噪声删除（多个离子等强度 → 仪器基线签名）
    remove_flat_noise: bool = True
    flat_noise_min_repeats: int = 3        # 同强度出现 ≥ N 次才删

    # 前体后截断
    remove_after_precursor: bool = True
    kept_isotope_range_da: float = 1.5     # 保留 precursor + 1.5 Da 之内

    # Top N
    top_n: int = 100


# ---------------------------------------------------------------------------
# Internal helpers (每个清洗步骤独立成函数，方便单元测试)
# ---------------------------------------------------------------------------


def _merge_close_ions(
    mz: np.ndarray,
    intensity: np.ndarray,
    precursor_mz: Optional[float],
    config: MS2CleanupConfig,
    extra_arrays: Optional[Sequence[np.ndarray]] = None,
) -> tuple:
    """步骤 1：使用共享的两阶段合并器合并近邻 m/z。

    精度容差为 ``max(merge_absolute_tol, mz × merge_ppm_tol / 1e6)``。
    当 ``precursor_mz`` 未知时，退化为 ``merge_absolute_tol``。

    ``extra_arrays`` 中的每个数组与 ``mz`` 同长度且同顺序，会随合并
    过程一起被保留（每个合并组取强度最大成员对应的值）。
    """
    extras = list(extra_arrays) if extra_arrays else []
    if mz.size <= 1:
        if extras:
            return (mz, intensity, *extras)
        return mz, intensity

    # peak_merge 内部使用 precursor_mz_nominal * 100 ppm 作为参考容差，
    # 这里把它换算成等价的"标称 m/z"以便复用同一个函数。
    if precursor_mz is None or not np.isfinite(precursor_mz):
        nominal = 0
    else:
        # peak_merge 内部按 nominal * 100e-6 计算；我们的 ppm 可调，所以
        # 等效 nominal = precursor_mz * (merge_ppm_tol / 100)。
        nominal = int(round(float(precursor_mz) * (config.merge_ppm_tol / 100.0)))

    merged = merge_close_ions(
        mz,
        intensity,
        precursor_mz_nominal=nominal,
        base_tolerance=config.merge_absolute_tol,
        extra_arrays=extras if extras else None,
    )
    return merged


def _apply_mask(
    mask: np.ndarray,
    mz: np.ndarray,
    intensity: np.ndarray,
    extras: list[np.ndarray],
) -> tuple:
    """工具函数：对所有数组应用同一布尔掩码。"""
    if extras:
        return (mz[mask], intensity[mask], *(arr[mask] for arr in extras))
    return mz[mask], intensity[mask]


def _remove_flat_noise(
    mz: np.ndarray,
    intensity: np.ndarray,
    min_repeats: int,
    extra_arrays: Optional[Sequence[np.ndarray]] = None,
) -> tuple:
    """步骤 2：删除平顶噪声。

    仪器基线噪声常表现为若干离子强度完全相等（如 53 counts × 多个 m/z）。
    凡是强度值出现 ≥ ``min_repeats`` 次的离子全部丢弃。

    向量化实现：``np.unique`` 找重复值，``np.isin`` 一次过滤。
    """
    extras = list(extra_arrays) if extra_arrays else []
    if intensity.size < min_repeats:
        return _apply_mask(np.ones(intensity.size, dtype=bool), mz, intensity, extras)

    unique_vals, counts = np.unique(intensity, return_counts=True)
    flat_values = unique_vals[counts >= min_repeats]
    if flat_values.size == 0:
        return _apply_mask(np.ones(intensity.size, dtype=bool), mz, intensity, extras)

    keep = ~np.isin(intensity, flat_values)
    return _apply_mask(keep, mz, intensity, extras)


def _apply_intensity_threshold(
    mz: np.ndarray,
    intensity: np.ndarray,
    abs_threshold: float,
    rel_threshold: float,
    extra_arrays: Optional[Sequence[np.ndarray]] = None,
) -> tuple:
    """步骤 3：双重强度截断。

    保留同时满足：
      - ``intensity >= abs_threshold``
      - ``intensity >= rel_threshold × base_peak``
    的离子。
    """
    extras = list(extra_arrays) if extra_arrays else []
    if intensity.size == 0:
        return _apply_mask(np.ones(0, dtype=bool), mz, intensity, extras)

    base_peak = float(intensity.max())
    rel_cut = base_peak * float(rel_threshold)
    cut = max(float(abs_threshold), rel_cut)
    keep = intensity >= cut
    return _apply_mask(keep, mz, intensity, extras)


def _remove_after_precursor(
    mz: np.ndarray,
    intensity: np.ndarray,
    precursor_mz: Optional[float],
    kept_isotope_range_da: float,
    extra_arrays: Optional[Sequence[np.ndarray]] = None,
) -> tuple:
    """步骤 4：去掉前体 + isotope_range 之后的离子。

    若 ``precursor_mz`` 未知则跳过（返回原谱）。
    """
    extras = list(extra_arrays) if extra_arrays else []
    if precursor_mz is None or not np.isfinite(precursor_mz):
        return _apply_mask(np.ones(intensity.size, dtype=bool), mz, intensity, extras)
    cutoff = float(precursor_mz) + float(kept_isotope_range_da)
    keep = mz <= cutoff
    return _apply_mask(keep, mz, intensity, extras)


def _keep_top_n(
    mz: np.ndarray,
    intensity: np.ndarray,
    top_n: int,
    extra_arrays: Optional[Sequence[np.ndarray]] = None,
) -> tuple:
    """步骤 5：保留强度最高的 ``top_n`` 个离子（不排序）。

    使用 ``np.argpartition`` 做 O(N) 选择；后续步骤 6 再统一按 m/z 排序。
    若 ``top_n <= 0`` 视为不限制。
    """
    extras = list(extra_arrays) if extra_arrays else []
    if top_n <= 0 or intensity.size <= top_n:
        if extras:
            return (mz, intensity, *extras)
        return mz, intensity
    keep_idx = np.argpartition(intensity, -top_n)[-top_n:]
    if extras:
        return (mz[keep_idx], intensity[keep_idx], *(arr[keep_idx] for arr in extras))
    return mz[keep_idx], intensity[keep_idx]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def clean_ms2_spectrum(
    mz: np.ndarray,
    intensity: np.ndarray,
    precursor_mz: Optional[float],
    config: MS2CleanupConfig,
    extra_arrays: Optional[Sequence[np.ndarray]] = None,
) -> tuple:
    """清洗 MS2 谱图，返回 ``(mz_clean, intensity_clean[, *extras_clean])``，按 m/z 升序。

    清洗步骤顺序：

      1. 合并近邻 m/z（自适应容差，强度加权 centroid）
      2. 移除平顶噪声（≥ N 个离子等强度）
      3. 强度截断（绝对 AND 相对）
      4. 移除 ``precursor + kept_isotope_range_da`` 之后的离子
      5. 保留 ``top_n`` 个最强离子
      6. 按 m/z 升序排序

    ``extra_arrays`` 是若干与 ``mz`` 同长度同顺序的辅助数组（例如 S/N），
    在整个清洗流程中会随离子一同被过滤/合并。当未提供时，仅返回
    ``(mz, intensity)`` 二元组以保持调用方简洁。

    输入数组可以是任意 ndim==1 的数值数组；返回均为 ``float64``。
    """
    mz_arr = np.asarray(mz, dtype=np.float64)
    int_arr = np.asarray(intensity, dtype=np.float64)
    extras = (
        [np.asarray(arr, dtype=np.float64) for arr in extra_arrays]
        if extra_arrays else []
    )
    has_extras = bool(extras)

    def _pack(m, i, ex):
        if has_extras:
            return (m, i, *ex)
        return m, i

    if mz_arr.size == 0:
        return _pack(mz_arr, int_arr, extras)

    # 1) 合并近邻 m/z
    merged = _merge_close_ions(
        mz_arr, int_arr, precursor_mz, config,
        extra_arrays=extras if has_extras else None,
    )
    if has_extras:
        mz_arr, int_arr, *extras = merged
    else:
        mz_arr, int_arr = merged
    if mz_arr.size == 0:
        return _pack(mz_arr, int_arr, extras)

    # 2) 平顶噪声
    if config.remove_flat_noise:
        filtered = _remove_flat_noise(
            mz_arr, int_arr, config.flat_noise_min_repeats,
            extra_arrays=extras if has_extras else None,
        )
        if has_extras:
            mz_arr, int_arr, *extras = filtered
        else:
            mz_arr, int_arr = filtered
        if mz_arr.size == 0:
            return _pack(mz_arr, int_arr, extras)

    # 3) 强度截断
    filtered = _apply_intensity_threshold(
        mz_arr,
        int_arr,
        config.absolute_intensity_threshold,
        config.relative_intensity_threshold,
        extra_arrays=extras if has_extras else None,
    )
    if has_extras:
        mz_arr, int_arr, *extras = filtered
    else:
        mz_arr, int_arr = filtered
    if mz_arr.size == 0:
        return _pack(mz_arr, int_arr, extras)

    # 4) 前体后截断
    if config.remove_after_precursor:
        filtered = _remove_after_precursor(
            mz_arr, int_arr, precursor_mz, config.kept_isotope_range_da,
            extra_arrays=extras if has_extras else None,
        )
        if has_extras:
            mz_arr, int_arr, *extras = filtered
        else:
            mz_arr, int_arr = filtered
        if mz_arr.size == 0:
            return _pack(mz_arr, int_arr, extras)

    # 5) Top N
    filtered = _keep_top_n(
        mz_arr, int_arr, config.top_n,
        extra_arrays=extras if has_extras else None,
    )
    if has_extras:
        mz_arr, int_arr, *extras = filtered
    else:
        mz_arr, int_arr = filtered

    # 6) 按 m/z 升序排序
    order = np.argsort(mz_arr)
    mz_arr = mz_arr[order]
    int_arr = int_arr[order]
    extras = [arr[order] for arr in extras]
    return _pack(mz_arr, int_arr, extras)
