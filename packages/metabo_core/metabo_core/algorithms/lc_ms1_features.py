# packages/metabo_core/metabo_core/algorithms/lc_ms1_features.py
"""LC-MS MS1 mass-slice 特征发现（DDA/DIA/ASFAM 共用，GC-MS 不得使用）。

把 DDA stage_features 已验证的 "build_eics_roi → dense EIC → detect_peaks"
模式抽取为平台共享函数。返回纯算法层 MS1FeatureHit, 不绑定任何 app 的
Scan / CandidateFeature 类型。
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from metabo_core.algorithms.ms1_eic_roi import ROIConfig, ROIEIC, build_eics_roi
from metabo_core.algorithms.peak_detection import detect_peaks
from metabo_core.config.peak_detection import PeakDetectionConfig


@dataclass
class MS1FeatureHit:
    """一个 MS1 mass-slice 峰的算法层结果。

    携带 apex/left/right scan 索引, 便于下游（PR-C ASFAM）直接喂相关性
    反卷积 ``_collect_ms2_at_peak``, 无需从 RT 反推。

    索引坐标系（重要, 给下游消费者）
    ------------------------------------
    ``apex_scan_idx`` / ``left_scan_idx`` / ``right_scan_idx`` 是
    **进入 ``find_lc_ms1_features`` 所传 ``ms1_scans`` 序列的 0-based
    位置索引**（与该序列的 RT 顺序对齐）——不是绝对 scan 号, 也不是
    混合 MS1/MS2 列表里的索引。下游做回映（如按 apex 取对应隔离窗 MS2）
    时必须持有同一份已过滤的 MS1 列表; 若改用别的列表（例如含 MS2 的
    全 scan 列表）会静默指向错误的 scan。``rt_apex/rt_left/rt_right``
    提供等价的 RT 兜底, 不依赖索引坐标系。
    """
    mz_centroid: float
    rt_apex: float
    rt_left: float
    rt_right: float
    height: float
    area: float
    sn_ratio: float
    gaussian_similarity: float
    apex_scan_idx: int
    left_scan_idx: int = 0
    right_scan_idx: int = 0


def find_lc_ms1_features(
    ms1_scans,
    *,
    roi_config: ROIConfig,
    peak_config: PeakDetectionConfig,
    mz_range: tuple[float, float] | None = None,
) -> list[MS1FeatureHit]:
    """在全分辨 MS1 survey 上做 mass-slice 找峰。

    Parameters
    ----------
    ms1_scans : Scan-like 序列
        每个需 ``.rt`` / ``.mz_array`` / ``.intensity_array``。调用方负责只传
        MS1（``ms_level == 1``）。
    roi_config : ROIConfig
        建议 ``mode="lc_ppm"``；**注意**：其 ``start_mz/end_mz`` 字段
        **始终被覆盖**——搜索窗口取自 ``mz_range``（若非 None）或取自
        ``ms1_scans`` 的观测 m/z 范围；调用方在 ROIConfig 中设置的
        ``start_mz/end_mz`` 无效，勿依赖。
    peak_config : PeakDetectionConfig
        三门峰检测参数（质量门）。
    mz_range : (lo, hi) 或 None
        限定搜索窗口的 m/z 区间，是唯一有效的 m/z 限制方式。``None``
        表示使用 ``ms1_scans`` 中出现过的实际 m/z 范围。

    Returns
    -------
    list[MS1FeatureHit]
        每个检出峰一个 hit。其 ``apex_scan_idx/left_scan_idx/
        right_scan_idx`` 为**进入入参 ``ms1_scans`` 序列的 0-based 位置
        索引**（与其 RT 顺序对齐）——下游需持有同一份 MS1 列表方能
        正确回映, 详见 :class:`MS1FeatureHit`。
    """
    if not ms1_scans:
        return []

    rt_array = np.array([float(s.rt) for s in ms1_scans], dtype=np.float64)
    n_scans = len(ms1_scans)

    if mz_range is not None:
        start_mz, end_mz = float(mz_range[0]), float(mz_range[1])
    else:
        start_mz, end_mz = _observed_mz_range(ms1_scans)
    if not np.isfinite(start_mz) or not np.isfinite(end_mz) or end_mz <= start_mz:
        return []

    # ROIConfig 是 dataclass：覆盖 start/end_mz 以匹配本次数据范围
    roi_cfg = replace(roi_config, start_mz=start_mz, end_mz=end_mz)
    rois = build_eics_roi(ms1_scans, roi_cfg)

    hits: list[MS1FeatureHit] = []
    for roi in rois:
        eic = _roi_to_dense_eic(roi, n_scans)
        mz_centroid = float(roi.mz_centroid)
        peaks = detect_peaks(
            rt_array=rt_array,
            intensity_array=eic,
            precursor_mz_nominal=int(round(mz_centroid)),
            product_mz=mz_centroid,
            min_amplitude=peak_config.min_amplitude,
            min_data_points=peak_config.min_data_points,
            smooth_window=peak_config.smooth_window,
            baseline_window=peak_config.baseline_window,
            noise_bin_size=peak_config.noise_bin_size,
            noise_factor=peak_config.noise_factor,
            sn_fold=peak_config.sn_fold,
            compute_gaussian=peak_config.gaussian_threshold > 0,
            gaussian_threshold=peak_config.gaussian_threshold,
            min_prominence_ratio=peak_config.min_prominence_ratio,
            rt_window_min=peak_config.rt_window_min,
            rt_window_max=peak_config.rt_window_max,
        )
        for peak in peaks:
            hits.append(MS1FeatureHit(
                mz_centroid=mz_centroid,
                rt_apex=peak.rt_apex,
                rt_left=peak.rt_left,
                rt_right=peak.rt_right,
                height=peak.height,
                area=peak.area,
                sn_ratio=peak.sn_ratio,
                gaussian_similarity=peak.gaussian_similarity,
                apex_scan_idx=int(peak.apex_index),
                left_scan_idx=int(peak.left_index),
                right_scan_idx=int(peak.right_index),
            ))
    return hits


def _roi_to_dense_eic(roi: ROIEIC, n_scans: int) -> np.ndarray:
    out = np.zeros(n_scans, dtype=np.float64)
    if roi.scan_indices.size == 0:
        return out
    out[roi.scan_indices] = roi.intensity_array
    return out


def _observed_mz_range(ms1_scans) -> tuple[float, float]:
    lo, hi = np.inf, -np.inf
    for s in ms1_scans:
        mz = s.mz_array
        if mz is None or len(mz) == 0:
            continue
        intens = np.asarray(s.intensity_array)
        mask = intens > 0.0
        if not mask.any():
            continue
        m = np.asarray(mz)[mask]
        lo = min(lo, float(m.min()))
        hi = max(hi, float(m.max()))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return float("inf"), float("-inf")
    pad = max(hi * 1e-6, 0.001)
    return lo - pad, hi + pad
