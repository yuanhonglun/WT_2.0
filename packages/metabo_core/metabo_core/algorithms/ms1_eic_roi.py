"""基于 slice + 50% overlap + 后期合并的共享 MS1 EIC (ROI) 构建器。

算法参考 MS-DIAL ``PeakSpottingCore.cs`` 的 mass-slice + redundancy
removal 思路, 在本项目中作为 LC-MS / GC-MS 共用的 EIC 提取层。

核心流程
--------
1. **Slice 划分**

   - LC-MS (``mode="lc_ppm"``): slice 宽度按 ppm 自适应, 即
     ``width(mz) = mz * ppm / 1e6``; 相邻 slice 中心间距 = ``width * overlap_fraction``,
     默认 50% overlap。
   - GC-MS (``mode="gc_da"``): slice 固定宽度 (默认 0.5 Da), step 0.25 Da。

2. **每个 slice 内的 EIC 提取**

   在该 slice [mz_lo, mz_hi] 内, 对每个 MS1 scan 取该 slice 内 m/z
   强度的 **max** (不是 sum, 避免噪声点叠加导致假阳)。同时记录贡献
   该 max 的 m/z, 用于后续 centroid。

3. **EIC centroid**

   ROI 的代表 m/z 用强度加权平均
   ``Σ(mz_i * intensity_i) / Σ(intensity_i)``。

4. **冗余合并**

   两个 ROI 满足下列条件 A 或 B 则保留强度顶点更高者:

   - 条件 A (典型 MS-DIAL 风格, 同一峰被 slice 切两半):
     m/z 中心差 <= slice_width * overlap_fraction (在重叠范围内) **且**
     RT 范围有交集 **且**
     apex RT 距离 <= ``min(HWHM_avg, rt_merge_max)``,
     HWHM ≈ (rt_right - rt_left) / 2
   - 条件 B (慢漂场景, 一个离子在 slice 边界处被拆成多段):
     m/z 中心差 <= slice_width * overlap_fraction **且**
     一个 ROI 的 RT 范围 >= overlap_ratio 比例被另一个 ROI 包含。
     这一条主要为了修复 legacy ``_build_mz_traces`` 把 m/z 慢漂离子
     切成多段 ROI 的缺陷。

5. **最小持续 scan 门**

   每个 ROI 至少跨 ``min_eic_points`` 个不同 scan。

性能策略
--------
为了在 ~3000 MS1 scan 上保持秒级响应:

- 将所有 (mz, intensity, scan_idx) 一次性铺平成全局数组并按 m/z 排序;
- 对每个 slice 用 ``np.searchsorted`` 在排序数组里 O(log n) 定位区间;
- slice 内部用 ``np.maximum.at`` 做 per-scan max 聚合 (向量化);
- 没有 Python 双层 for, 复杂度近似 O(N_points + N_slices * log N_points)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Literal, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# 配置和返回数据类
# ---------------------------------------------------------------------------


@dataclass
class ROIConfig:
    """ROI EIC 构建的统一配置。

    Parameters
    ----------
    mode : {"lc_ppm", "gc_da"}
        LC-MS 高分辨用 ppm 自适应; GC-MS 单位质量用固定 Da 宽度。
    ppm_tolerance : float
        LC 模式下的 slice 宽度系数 (ppm)。
    da_slice_width : float
        GC 模式下的 slice 宽度 (Da)。
    overlap_fraction : float
        相邻 slice 中心距离占 slice 宽度的比例; 0.5 = 50% overlap。
    min_eic_points : int
        ROI 至少跨越的不同 scan 数 (非零强度)。LC 默认 5, GC 默认 4。
    min_intensity : float
        参与构建的最小点强度阈值; 默认 0 表示不过滤。
    rt_merge_max : float
        后期合并 (条件 A) 时 apex RT 距离的上限 (min); MS-DIAL 默认 0.03。
    rt_range_overlap_ratio : float
        后期合并 (条件 B, 慢漂) 时, 较小 ROI 的 RT 范围被较大 ROI 覆盖
        的最低比例; 默认 0.5。
    start_mz, end_mz : float
        slice 扫描的 m/z 范围。
    """

    mode: Literal["lc_ppm", "gc_da"] = "lc_ppm"
    ppm_tolerance: float = 15.0
    da_slice_width: float = 0.5
    overlap_fraction: float = 0.5
    min_eic_points: int = 5
    min_intensity: float = 0.0
    rt_merge_max: float = 0.03  # min
    rt_range_overlap_ratio: float = 0.5
    start_mz: float = 50.0
    end_mz: float = 1500.0


@dataclass
class ROIEIC:
    """单个 ROI 的 EIC 表示。

    Parameters
    ----------
    mz_centroid : float
        ROI 的代表 m/z, 强度加权平均。
    rt_array : np.ndarray
        ROI 覆盖的 scan 对应 RT (min)。
    intensity_array : np.ndarray
        每个 scan 的强度 (slice 内 max), 与 rt_array 同长度。
    scan_indices : np.ndarray
        每个点对应的原始 MS1 scan index, 与 rt_array 同长度。
    n_points : int
        非零强度的 scan 数 (等同于 rt_array 长度, 因为零点不入 ROI)。
    """

    mz_centroid: float
    rt_array: np.ndarray
    intensity_array: np.ndarray
    scan_indices: np.ndarray
    n_points: int


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------


def build_eics_roi(
    scans: Sequence[Any],
    config: ROIConfig,
) -> List[ROIEIC]:
    """从一组 MS1 scan 构建 ROI 风格的 EIC 列表。

    Parameters
    ----------
    scans : list
        MS1 Scan 列表; 每个对象需要至少有 ``rt``, ``mz_array``,
        ``intensity_array`` 属性 (兼容 ``metabo_core.models.Scan``)。
        会被原样视作 MS1, 调用方负责预过滤 ``ms_level == 1``。
    config : ROIConfig
        参数配置。

    Returns
    -------
    list[ROIEIC]
        合并去冗余后的 ROI 列表; 顺序按 mz_centroid 升序。
    """
    if not scans:
        return []

    # ---- 步骤 1: 铺平所有点并按 m/z 排序 ------------------------------------
    flat_mz, flat_int, flat_scan = _flatten_scans(scans, config.min_intensity)
    if flat_mz.size == 0:
        return []

    # 按 m/z 排序, 便于 slice 用 searchsorted 切片
    order = np.argsort(flat_mz, kind="quicksort")
    sorted_mz = flat_mz[order]
    sorted_int = flat_int[order]
    sorted_scan = flat_scan[order]

    rt_per_scan = np.asarray([float(s.rt) for s in scans], dtype=np.float64)
    n_scans = len(scans)

    # ---- 步骤 2: 生成 slice 中心列表 ----------------------------------------
    slice_centers, slice_widths = _generate_slice_centers(config)
    if slice_centers.size == 0:
        return []

    # ---- 步骤 3: 每个 slice 提取 EIC ----------------------------------------
    raw_rois: list[ROIEIC] = []
    for center, width in zip(slice_centers, slice_widths):
        half = width * 0.5
        lo = center - half
        hi = center + half
        # searchsorted 在排好序的 sorted_mz 中找区间端点
        i_lo = np.searchsorted(sorted_mz, lo, side="left")
        i_hi = np.searchsorted(sorted_mz, hi, side="right")
        if i_hi - i_lo < config.min_eic_points:
            continue

        seg_mz = sorted_mz[i_lo:i_hi]
        seg_int = sorted_int[i_lo:i_hi]
        seg_scan = sorted_scan[i_lo:i_hi]

        roi = _build_roi_from_segment(
            seg_mz, seg_int, seg_scan,
            n_scans=n_scans,
            rt_per_scan=rt_per_scan,
            min_eic_points=config.min_eic_points,
        )
        if roi is not None:
            raw_rois.append(roi)

    if not raw_rois:
        return []

    # ---- 步骤 4: 后期冗余合并 -----------------------------------------------
    merged = _merge_redundant_rois(
        raw_rois,
        config=config,
    )
    return merged


# ---------------------------------------------------------------------------
# 内部: 铺平 / slice 生成
# ---------------------------------------------------------------------------


def _flatten_scans(
    scans: Iterable[Any],
    min_intensity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把每个 scan 的 (mz, intensity) 铺平成三个长一维数组。

    返回 (mz, intensity, scan_idx); scan_idx 是在传入 scans 序列中的位置
    (从 0 开始), 不是 Scan.scan_id, 这样保证后续可以直接做
    rt_per_scan[scan_idx]。
    """
    mz_chunks: list[np.ndarray] = []
    int_chunks: list[np.ndarray] = []
    scan_chunks: list[np.ndarray] = []
    for idx, scan in enumerate(scans):
        mz = np.asarray(scan.mz_array, dtype=np.float64)
        intens = np.asarray(scan.intensity_array, dtype=np.float64)
        if mz.size == 0:
            continue
        if min_intensity > 0.0:
            mask = intens > min_intensity
        else:
            mask = intens > 0.0
        if not mask.any():
            continue
        m = mz[mask]
        i = intens[mask]
        mz_chunks.append(m)
        int_chunks.append(i)
        scan_chunks.append(np.full(m.size, idx, dtype=np.int64))

    if not mz_chunks:
        empty_f = np.array([], dtype=np.float64)
        empty_i = np.array([], dtype=np.int64)
        return empty_f, empty_f, empty_i
    return (
        np.concatenate(mz_chunks),
        np.concatenate(int_chunks),
        np.concatenate(scan_chunks),
    )


def _generate_slice_centers(
    config: ROIConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """根据模式生成所有 slice 的中心 m/z 和宽度数组。

    LC 模式下宽度随 m/z 变化, 因此中心需要迭代; GC 模式宽度固定可直接
    arange。两种模式下都保证相邻 slice 重叠 ``overlap_fraction``。
    """
    if config.mode == "gc_da":
        width = float(config.da_slice_width)
        step = width * float(config.overlap_fraction)
        if step <= 0:
            raise ValueError("overlap_fraction must be > 0")
        centers = np.arange(
            config.start_mz + width * 0.5,
            config.end_mz + step,
            step,
            dtype=np.float64,
        )
        widths = np.full_like(centers, width)
        return centers, widths

    # LC mode: 按 ppm 自适应宽度, 迭代步进
    ppm = float(config.ppm_tolerance)
    if ppm <= 0:
        raise ValueError("ppm_tolerance must be > 0")
    overlap = float(config.overlap_fraction)
    if overlap <= 0:
        raise ValueError("overlap_fraction must be > 0")

    centers: list[float] = []
    widths: list[float] = []
    cur = float(config.start_mz)
    end = float(config.end_mz)
    # 安全上限, 防御参数极端值
    max_iter = 2_000_000
    n_iter = 0
    while cur <= end and n_iter < max_iter:
        w = cur * ppm * 1e-6
        if w <= 0:
            break
        centers.append(cur)
        widths.append(w)
        cur = cur + w * overlap
        n_iter += 1
    return np.asarray(centers, dtype=np.float64), np.asarray(widths, dtype=np.float64)


# ---------------------------------------------------------------------------
# 内部: 单 slice 内构建 ROI
# ---------------------------------------------------------------------------


def _build_roi_from_segment(
    seg_mz: np.ndarray,
    seg_int: np.ndarray,
    seg_scan: np.ndarray,
    n_scans: int,
    rt_per_scan: np.ndarray,
    min_eic_points: int,
) -> ROIEIC | None:
    """在一个 slice 的点集合里, 对每个 scan 取强度 max 形成 EIC。

    用 ``np.maximum.at`` 做 unbuffered 散点聚合, 等价于
    "for each (scan, intensity): eic[scan] = max(eic[scan], intensity)"
    但是向量化执行。同时维护贡献 max 的 m/z (用于 centroid)。
    """
    # 用 -inf 初始化 EIC, 之后比较得到 max
    eic_int = np.full(n_scans, -np.inf, dtype=np.float64)
    # 关键: np.maximum.at 是 unbuffered 的散点最大化
    np.maximum.at(eic_int, seg_scan, seg_int)
    # 哪些 scan 被赋值过
    mask = eic_int > -np.inf
    if not mask.any():
        return None
    n_pts = int(mask.sum())
    if n_pts < min_eic_points:
        return None

    scan_indices = np.flatnonzero(mask)
    intensities = eic_int[scan_indices]
    rts = rt_per_scan[scan_indices]

    # 还原每个被选中 scan 对应的 m/z: 在该 slice 的点集中, 找到与
    # eic_int[scan] 同强度的点。常见情况下每个 scan 只有 1-2 个点, 用
    # 字典法即可避免 O(n_scans * n_seg) 嵌套。
    mz_per_scan = _mz_at_max_per_scan(seg_mz, seg_int, seg_scan, scan_indices)

    # centroid 强度加权
    total_int = intensities.sum()
    if total_int <= 0:
        return None
    centroid = float((mz_per_scan * intensities).sum() / total_int)

    return ROIEIC(
        mz_centroid=centroid,
        rt_array=rts.astype(np.float64),
        intensity_array=intensities.astype(np.float64),
        scan_indices=scan_indices.astype(np.int64),
        n_points=n_pts,
    )


def _mz_at_max_per_scan(
    seg_mz: np.ndarray,
    seg_int: np.ndarray,
    seg_scan: np.ndarray,
    target_scans: np.ndarray,
) -> np.ndarray:
    """对每个 target_scan, 返回该 scan 在 segment 内贡献最大强度的 m/z。

    向量化做法: 先按 (scan, intensity) lexsort, 取每个 scan 块的最后一行
    (= 该 scan 的 max-intensity 行), 再用 searchsorted 把 target_scan 映射
    到这些行上。
    """
    if seg_scan.size == 0:
        return np.zeros(target_scans.shape, dtype=np.float64)

    # lexsort: 主键 seg_scan 升序, 次键 seg_int 升序
    order = np.lexsort((seg_int, seg_scan))
    sorted_scan = seg_scan[order]
    sorted_mz = seg_mz[order]
    # 每个 scan 块的最后一个 index 就是该 scan 的最大强度行
    # np.unique 的 return_index 给出每个唯一值在排序后的首次出现位置
    uniq_scans, first_idx = np.unique(sorted_scan, return_index=True)
    # 每个 scan 块的最后一个位置 = 下一个块的 first_idx - 1; 最后一块到末尾
    last_idx = np.empty_like(first_idx)
    last_idx[:-1] = first_idx[1:] - 1
    last_idx[-1] = sorted_scan.size - 1
    max_mz_for_uniq = sorted_mz[last_idx]

    # 把 target_scans 映射到 uniq_scans 的对应位置
    # uniq_scans 已经升序, target_scans 也升序 (由 flatnonzero 保证)
    locator = np.searchsorted(uniq_scans, target_scans)
    return max_mz_for_uniq[locator]


# ---------------------------------------------------------------------------
# 内部: 后期冗余合并
# ---------------------------------------------------------------------------


def _merge_redundant_rois(
    rois: List[ROIEIC],
    config: ROIConfig,
) -> List[ROIEIC]:
    """合并相邻 slice 因 overlap 产生的重复 ROI, 保留 apex 更高者。

    合并条件 (同时满足):
      - m/z 中心距离 <= slice_width(mz) * overlap_fraction
      - RT 范围有交集 [rt_left, rt_right]
      - apex RT 距离 <= min(HWHM_avg, rt_merge_max)
    """
    if len(rois) <= 1:
        return list(rois)

    # 预计算每个 ROI 的 rt_left/rt_right/apex/HWHM/apex_int
    summaries = [_summarize_roi(r) for r in rois]
    centroids = np.asarray([r.mz_centroid for r in rois], dtype=np.float64)

    # 排序: 按 mz_centroid 升序
    order = np.argsort(centroids, kind="quicksort")
    sorted_centroids = centroids[order]
    sorted_rois = [rois[i] for i in order]
    sorted_summaries = [summaries[i] for i in order]

    # 标记哪些被合并掉
    alive = np.ones(len(sorted_rois), dtype=bool)

    # 对每个 ROI 只与"右邻居窗口"内的其他 ROI 比较 (因为按 mz 排序了)
    n = len(sorted_rois)
    for i in range(n):
        if not alive[i]:
            continue
        ci = sorted_centroids[i]
        # 该 ROI 的 slice 宽度 (用于判定合并距离)
        slice_w_i = _slice_width_at(ci, config)
        # 合并距离上限
        mz_thresh = slice_w_i * config.overlap_fraction

        # 右侧窗口: ci + mz_thresh
        j_end = np.searchsorted(sorted_centroids, ci + mz_thresh, side="right")
        for j in range(i + 1, j_end):
            if not alive[j]:
                continue
            sj = sorted_summaries[j]
            si = sorted_summaries[i]

            # 1) m/z 距离已经天然满足 (由 searchsorted 窗口控制)
            # 2) RT 范围有交集?
            if si.rt_right < sj.rt_left or sj.rt_right < si.rt_left:
                continue

            # 条件 A: apex RT 距离够近 (同一峰被切两半)
            hwhm_avg = 0.5 * (si.hwhm + sj.hwhm)
            apex_thresh = min(hwhm_avg, config.rt_merge_max)
            if apex_thresh <= 0:
                apex_thresh = config.rt_merge_max
            cond_A = abs(si.rt_apex - sj.rt_apex) <= apex_thresh

            # 条件 B: 较小 ROI 的 RT 范围被另一方 >= ratio 覆盖
            # (用于捕获慢漂导致的多段切分)
            cond_B = _rt_range_overlap_ratio(si, sj) >= config.rt_range_overlap_ratio

            if not (cond_A or cond_B):
                continue

            # 满足合并条件, 保留 apex 更高者
            if si.apex_int >= sj.apex_int:
                alive[j] = False
            else:
                alive[i] = False
                break  # i 已死, 不需要再继续 i 的循环
    out = [sorted_rois[k] for k in range(n) if alive[k]]
    return out


@dataclass
class _ROISummary:
    rt_left: float
    rt_right: float
    rt_apex: float
    apex_int: float
    hwhm: float


def _summarize_roi(roi: ROIEIC) -> _ROISummary:
    """从 ROIEIC 提取 apex / 边界 / HWHM 用于合并判定。

    这里只取 ROI 已经存在的点的边界, 不重新做峰检测 (峰检测是下游的事)。
    HWHM 用 (rt_right - rt_left) / 2 作为粗近似, 与 spec 一致。
    """
    if roi.rt_array.size == 0:
        return _ROISummary(0.0, 0.0, 0.0, 0.0, 0.0)
    apex_idx = int(np.argmax(roi.intensity_array))
    rt_left = float(roi.rt_array.min())
    rt_right = float(roi.rt_array.max())
    rt_apex = float(roi.rt_array[apex_idx])
    apex_int = float(roi.intensity_array[apex_idx])
    hwhm = max(0.0, (rt_right - rt_left) * 0.5)
    return _ROISummary(
        rt_left=rt_left,
        rt_right=rt_right,
        rt_apex=rt_apex,
        apex_int=apex_int,
        hwhm=hwhm,
    )


def _rt_range_overlap_ratio(si: "_ROISummary", sj: "_ROISummary") -> float:
    """返回两个 ROI RT 范围中 ``min`` 一方被另一方覆盖的比例。

    用于慢漂场景的合并判定: 一个慢漂离子被多个 slice 切分时, 各段
    RT 范围互相高度覆盖, 即便 m/z 中心略偏 / apex 位置不同, 仍应合并。
    """
    overlap = min(si.rt_right, sj.rt_right) - max(si.rt_left, sj.rt_left)
    if overlap <= 0:
        return 0.0
    span_i = si.rt_right - si.rt_left
    span_j = sj.rt_right - sj.rt_left
    smaller = min(span_i, span_j)
    if smaller <= 0:
        return 1.0  # 退化为单点的 ROI, 视作完全被覆盖
    return overlap / smaller


def _slice_width_at(mz: float, config: ROIConfig) -> float:
    """返回某个 m/z 处的 slice 宽度, 与 _generate_slice_centers 保持一致。"""
    if config.mode == "gc_da":
        return float(config.da_slice_width)
    return float(mz) * float(config.ppm_tolerance) * 1e-6


__all__ = ["ROIConfig", "ROIEIC", "build_eics_roi"]
