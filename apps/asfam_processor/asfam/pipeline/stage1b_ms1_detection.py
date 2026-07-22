"""Stage 1b: MS1-driven feature detection (complementary to MS2-driven Stage 1).

PR-C: 在全分辨 MS1 survey 上做精细 m/z mass-slice 找峰 (平台共享的
``find_lc_ms1_features``), 取代旧的"每整数通道一条 1 Da EIC, 取 max 塌缩"
内核。每个 MS1 feature 从对应 1 Da 隔离窗用既有的色谱相关性反卷积
(``_collect_ms2_at_peak``) 取 MS2; 不再有 ≥2 MS2 强制门 (Q3), 弱/无 MS2
的 feature 保留并用 ``ms2_quality`` 标注 ("correlated"/"sparse"/"none")。

对每个不与 MS2 驱动 feature 重合的 MS1 峰, 创建一个新的 CandidateFeature。
这样能捕获 MS1 可见但被 MS2 驱动方法漏掉的化合物。
"""
from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import Optional, Callable

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import RawSegmentData, CandidateFeature
from asfam.core.eic import extract_ms1_eic, merge_close_ions
from asfam.core.ms1_scan_adapter import ms1_survey_scans
from metabo_core.algorithms.dedup_relations import eic_coelution_ok
from metabo_core.algorithms.lc_ms1_features import find_lc_ms1_features
from metabo_core.algorithms.msdial_ms1_features import find_lc_ms1_features_msdial
from metabo_core.algorithms.ms1_eic_roi import ROIConfig
from metabo_core.algorithms.msdec import deconvolute_ms2

logger = logging.getLogger(__name__)


def run_stage1b(
    data_by_replicate: dict[str, list[RawSegmentData]],
    existing_features: dict[str, list[CandidateFeature]],
    config: ProcessingConfig,
    progress_callback: Optional[Callable] = None,
) -> dict[str, list[CandidateFeature]]:
    """MS1-driven feature detection, complementary to MS2-driven Stage 1.

    For each replicate, runs the shared mass-slice MS1 finder
    (``find_lc_ms1_features``) over the full-resolution MS1 survey to detect
    fine-m/z MS1 peaks; skips peaks already covered by MS2-driven features,
    then creates CandidateFeature objects with detection_source="ms1_driven"
    and merges them into the existing list.

    Returns the updated features_by_replicate dict.
    """
    logger.info("Stage 1b: MS1-driven feature detection...")

    total_new = 0
    total_files = sum(len(segs) for segs in data_by_replicate.values())
    file_count = 0

    for rep_id, segments in data_by_replicate.items():
        existing = existing_features.get(rep_id, [])

        # Build index of existing features: channel -> list of RT apexes
        existing_index: dict[int, list[float]] = {}
        for f in existing:
            ch = f.precursor_mz_nominal
            if ch not in existing_index:
                existing_index[ch] = []
            existing_index[ch].append(f.rt_apex)

        new_features: list[CandidateFeature] = []

        for raw_data in segments:
            seg_new = _process_segment_ms1(
                raw_data, existing_index, config,
            )
            new_features.extend(seg_new)
            file_count += 1

            if progress_callback:
                progress_callback(
                    "stage1b", file_count, total_files,
                    f"Rep {rep_id} {raw_data.segment_name}: {len(seg_new)} new MS1-driven",
                )

        if new_features:
            existing_features[rep_id] = existing + new_features
            total_new += len(new_features)
            logger.info("  Replicate %s: %d new MS1-driven features", rep_id, len(new_features))
        else:
            logger.info("  Replicate %s: 0 new MS1-driven features", rep_id)

    logger.info("  Total new MS1-driven features: %d", total_new)
    return existing_features


def _assigned_window(
    raw_data: RawSegmentData,
    precise_mz: float,
    tol: float,
) -> Optional[int]:
    """Return the acquired MS2 window floor-key nearest ``precise_mz``, or None.

    Picks the isolation window whose recorded target m/z is closest to the
    feature's precise precursor m/z, provided the gap is within ``tol`` (half a
    1-Da window plus a small MS1↔MS2 calibration margin). Returns None when no
    acquired window covers this m/z — the feature was never isolated for MS2,
    so it correctly gets an empty MS2 spectrum.

    Replaces the old ``int(round(precise_mz))`` channel guess, which silently
    missed every X.5-target window in the high-m/z region: those windows are
    keyed by ``floor(target)`` (see ``mzml_reader``) and a rounded centroid
    lands on the wrong integer.
    """
    targets = raw_data.precursor_targets
    if not targets:
        # Fallback for a RawSegmentData built without recorded targets (legacy
        # project loads / synthetic test fixtures): assume integer-centred
        # windows at each acquired channel — the pre-fix behaviour. The live
        # reader always populates precursor_targets, so this never runs on real
        # data (where the X.5-target windows need the recorded values).
        targets = {c: float(c) for c in raw_data.precursor_list}
    if not targets:
        return None
    best_key: Optional[int] = None
    best_d = tol
    for key, tgt in targets.items():
        d = abs(tgt - precise_mz)
        if d <= best_d:
            best_d = d
            best_key = key
    return best_key


def _process_segment_ms1(
    raw_data: RawSegmentData,
    existing_index: dict[int, list[float]],
    config: ProcessingConfig,
) -> list[CandidateFeature]:
    """对一个 segment 做 MS1 驱动找峰, 为不与 MS2 驱动 feature 重合的峰建 feature。

    PR-C: 在全分辨 MS1 survey 上跑 mass-slice 找峰 (``find_lc_ms1_features``),
    每个命中峰从对应 1 Da 隔离窗用色谱相关性反卷积取 MS2; 不再丢弃弱/无 MS2
    的峰, 仅用 ``ms2_quality`` 标注。
    """
    features: list[CandidateFeature] = []

    # 把每个 cycle 的 MS1 谱适配成 Scan-like 序列 (scan_idx == cycle_idx,
    # 见 ms1_scan_adapter); 这保证 hit 的 *_scan_idx 可以直接当 cycle 索引
    # 喂给 _collect_ms2_at_peak。
    scans = ms1_survey_scans(raw_data.cycles)

    if config.peak_detector == "msdial":
        # MS-DIAL faithful detector: fixed-0.1Da SUM mass-slice EIC +
        # derivative engine + coarse→fine recalc. Honour stage1b's amplitude
        # floor (ms1_min_height) the same way the metra branch does, so the
        # A/B compares algorithm, not threshold.
        msdial_cfg = replace(config.msdial_peak, min_amplitude=config.ms1_min_height)
        hits = find_lc_ms1_features_msdial(scans, config=msdial_cfg)
    else:
        roi_cfg = ROIConfig(
            mode="lc_ppm",
            ppm_tolerance=config.ms1_massslice_ppm,
            min_eic_points=config.ms1_peak.min_data_points,
            overlap_fraction=0.5,
            rt_merge_max=0.03,
        )
        # stage1b 用更低的振幅地板 (ms1_min_height) 去捞主 MS1 阶段漏掉的弱真峰,
        # 覆盖掉 ms1_peak.min_amplitude; 其余质量门 (gaussian/prominence/S-N/
        # min_data_points) 沿用 ms1_peak。
        pd_cfg = replace(config.ms1_peak, min_amplitude=config.ms1_min_height)
        hits = find_lc_ms1_features(scans, roi_config=roi_cfg, peak_config=pd_cfg)

    counter = 0
    for hit in hits:
        # Assign this MS1 peak to the acquired MS2 isolation window whose target
        # m/z is nearest the precise centroid (within ms2_isolation_window_tol).
        # ASFAM windows are 1-Da wide at a target that drifts with mass, so
        # int(round(centroid)) does NOT reliably land on the window's floor-key —
        # above m/z ~800 the targets sit at X.5 and round() misses every one.
        # ``win_key`` is None when no acquired window covers this m/z (the
        # feature was never isolated for MS2); ``channel`` still gets a stable
        # nominal for identity/dedup.
        win_key = _assigned_window(
            raw_data, hit.mz_centroid, config.ms2_isolation_window_tol,
        )
        channel = win_key if win_key is not None else int(math.floor(hit.mz_centroid))

        # 去重: 跳过已被 MS2 驱动 feature 覆盖的 (channel, RT)
        if any(
            abs(hit.rt_apex - ert) <= config.ms1_rt_tolerance
            for ert in existing_index.get(channel, [])
        ):
            continue

        # 精确 m/z 的 MS1 EIC (不再用 0.5 Da; QTOF 高 m/z 漂移用 ppm 自适应)
        tol = max(
            config.eic_mz_tolerance,
            hit.mz_centroid * config.ms1_precise_eic_ppm * 1e-6,
        )
        rt_arr, ms1_eic = extract_ms1_eic(raw_data, hit.mz_centroid, mz_tolerance=tol)

        # MS2 反卷积 (复用既有色谱相关性反卷积; 无对应隔离窗则没有 MS2)。
        # hit 的 apex/left/right_scan_idx 是 scans 序列的 0-based 位置, 因
        # ms1_survey_scans 与 cycles 1:1 对齐, 等同 cycle 索引, 可直接传入。
        if win_key is not None:
            ms2_mz, ms2_intensity = _collect_ms2_at_peak(
                raw_data, win_key,
                hit.apex_scan_idx, hit.left_scan_idx, hit.right_scan_idx,
                config,
                ms1_eic=ms1_eic,
                rt_array=rt_arr,
            )
        else:
            ms2_mz = np.array([], dtype=np.float64)
            ms2_intensity = np.array([], dtype=np.float64)

        n_frags = len(ms2_mz)
        # Q3: 不再在 n_frags < 门限 时丢弃; 仅标注 MS2 质量。
        if n_frags >= 2:
            ms2_quality = "correlated"
        elif n_frags == 1:
            ms2_quality = "sparse"
        else:
            ms2_quality = "none"

        feature_id = f"{raw_data.segment_name}_{channel}_ms1_{counter}"
        counter += 1

        feat = CandidateFeature(
            feature_id=feature_id,
            segment_name=raw_data.segment_name,
            replicate_id=raw_data.replicate_id,
            precursor_mz_nominal=channel,
            rt_apex=hit.rt_apex,
            rt_left=hit.rt_left,
            rt_right=hit.rt_right,
            ms2_mz=ms2_mz,
            ms2_intensity=ms2_intensity,
            n_fragments=n_frags,
            ms1_precursor_mz=hit.mz_centroid,
            # The mass-slice finder recalculates height/area on a
            # +/-centroid_ms1_tolerance SUM chromatogram around this very
            # centroid, so precursor and quant ion coincide here.
            ms1_quant_mz=hit.mz_centroid,
            ms1_height=hit.height,
            ms1_area=hit.area,
            ms1_sn=hit.sn_ratio,
            ms1_gaussian=hit.gaussian_similarity,
            signal_type="ms1_detected",
            detection_source="ms1_driven",
            mz_source="ms1_peak",
            source_file=raw_data.file_path,
            ms2_quality=ms2_quality,
            n_correlated_ms2=n_frags,
        )
        features.append(feat)

    return features


def _centroid_seed(
    mz: np.ndarray, inten: np.ndarray, tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """0.025 Da 质心归并 (对齐 MS-DIAL Ms2Dec ``curatedSpectra``)。

    QTOF 会把同一真峰吐成间距 < tol 的相邻 vendor-centroid 点。若不归并, MSDec
    会把它们当成两个离子、各抽同一条强色谱、各发一峰 → base 峰被劈成重复峰 →
    与参考谱 cosine 被拉低。MS-DIAL 在建种子时先做本归并。

    规则 (与 CLAUDE.md 不变量一致):
      - 相邻 Δm/z ≤ ``tol`` 归为一峰 (链式:每步 ≤ tol 即同簇);
      - 输出 m/z = **峰内 intensity 加权质心** (绝非算术均值, 绝不跨离子平均);
      - 输出 intensity = 峰内 max;
      - 输出按 m/z 升序。
    """
    if mz.size == 0:
        return mz, inten
    order = np.argsort(mz)
    mz = mz[order]
    inten = inten[order]
    out_mz: list[float] = []
    out_int: list[float] = []
    start = 0
    for i in range(1, mz.size + 1):
        # 段边界: 到末尾, 或与前一点间距超过 tol (新簇开始)
        if i == mz.size or mz[i] - mz[i - 1] > tol:
            seg_mz = mz[start:i]
            seg_int = inten[start:i]
            tot = float(seg_int.sum())
            if tot > 0:
                out_mz.append(float((seg_mz * seg_int).sum() / tot))
            else:
                # 全零强度: 退回峰内最大强度点的 m/z (等价 argmax, 避免除零)
                out_mz.append(float(seg_mz[int(np.argmax(seg_int))]))
            out_int.append(float(seg_int.max()))
            start = i
    return (
        np.array(out_mz, dtype=np.float64),
        np.array(out_int, dtype=np.float64),
    )


def _collect_ms2_msdec(
    raw_data: RawSegmentData,
    channel: int,
    apex_idx: int,
    eic_left: int,
    eic_right: int,
    rt_array: np.ndarray,
    config: ProcessingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """档 C: MS-DIAL MSDec 最小二乘反卷积 (Ms2Dec.cs GetMS2DecResult)。

    产品离子集 = 前体 apex **单 scan** 的 MS2 谱 (阈值过滤 + RemoveAfterPrecursor),
    **不是** 档 B 的区间并集——后者把整个 peak 区间所有 scan 的所有 m/z 并起来,
    含大量跨 scan 噪声 (实测 ASFAM 单段每 feature 区间并集 ~200-400 离子, 而
    apex 单 scan 仅 ~10-25), 会让最小二乘谱爆炸到几百碎片。MS-DIAL 只取 apex
    单 scan 的 centroid 谱 (Ms2Dec.cs:81-92), 故照搬之。

    种子再经 ``_centroid_seed`` 做 0.025 Da 质心归并 (MS-DIAL curatedSpectra),
    去重同一真峰的相邻 vendor-centroid 点 (R1)。对每个 (归并后) 产品 m/z 在窗
    [eic_left, eic_right] 按 centroid_ms2_tolerance 抽色谱 (GetMs2ValuePeaks),
    缺失补 0, 交给 metabo_core 的 deconvolute_ms2 做模型峰分解。输出 m/z = 归并后
    的种子 m/z (不对反卷积**输出**重新质心, handoff 坑 #11)。
    """
    empty = (np.array([], dtype=np.float64), np.array([], dtype=np.float64))
    n_cycles = len(raw_data.cycles)
    if apex_idx < 0 or apex_idx >= n_cycles:
        return empty
    apex_cycle = raw_data.cycles[apex_idx]
    if channel not in apex_cycle.ms2_scans:
        return empty
    ap_mz, ap_int = apex_cycle.ms2_scans[channel]
    ap_mz = np.asarray(ap_mz, dtype=np.float64)
    ap_int = np.asarray(ap_int, dtype=np.float64)
    if ap_mz.size == 0:
        return empty

    mcfg = config.msdec_view()
    # Ms2Dec.cs:81-92: 阈值 max(ampTop*rel, abs, 0.1) + RemoveAfterPrecursor。
    amp_top = float(ap_int.max())
    threshold = max(
        amp_top * mcfg.relative_amplitude_cutoff, mcfg.amplitude_cutoff, 0.1
    )
    keep = ap_int > threshold
    if mcfg.remove_after_precursor:
        keep = keep & (ap_mz <= channel + mcfg.kept_isotope_range)
    prod_mzs = ap_mz[keep]
    prod_ints = ap_int[keep]
    if prod_mzs.size == 0:
        return empty

    # R1: MS-DIAL Ms2Dec curatedSpectra —— 0.025 Da 峰内加权质心归并, 去重同一
    # 真峰的相邻 vendor-centroid 点 (否则被劈成重复 base 峰, 拉低定性 cosine)。
    prod_mzs, prod_ints = _centroid_seed(
        prod_mzs, prod_ints, mcfg.centroid_ms2_tolerance
    )

    # 每个产品 m/z 在窗内抽色谱 (GetMs2ValuePeaks): centroid_ms2_tolerance 匹配,
    # 取该 cycle 内匹配峰的最大强度, 缺失补 0。
    n_pts = eic_right - eic_left + 1
    tol = mcfg.centroid_ms2_tolerance
    ion_eic_mat = np.zeros((prod_mzs.size, n_pts), dtype=np.float64)
    for col, ci in enumerate(range(eic_left, eic_right + 1)):
        cyc = raw_data.cycles[ci]
        if channel not in cyc.ms2_scans:
            continue
        pm, pi = cyc.ms2_scans[channel]
        pm = np.asarray(pm, dtype=np.float64)
        pi = np.asarray(pi, dtype=np.float64)
        if pm.size == 0:
            continue
        for i, mz in enumerate(prod_mzs):
            m = np.abs(pm - mz) <= tol
            if np.any(m):
                ion_eic_mat[i, col] = float(pi[m].max())

    rt_win = np.asarray(rt_array, dtype=np.float64)[eic_left:eic_right + 1]
    out_mz, out_int = deconvolute_ms2(
        prod_mzs,
        ion_eic_mat,
        precursor_apex_scan=apex_idx - eic_left,
        config=mcfg,
        rt_array=rt_win,
    )
    if out_mz.size == 0:
        # Ms2Dec.cs: MSDec 返回 null → GetDefaultMSDecResult, 退回 apex scan
        # 的 curatedSpectra (已 0.025 归并的种子, 与 prod_mzs 同长), 而非丢掉
        # MS2 (无模型/apex 不关联时)。用 prod_ints 而非旧的 ap_int[keep]——后者
        # 是归并前长度, 与归并后的 prod_mzs 形状不匹配。
        return prod_mzs.copy(), prod_ints.copy()
    return out_mz, out_int


def _collect_ms2_at_peak(
    raw_data: RawSegmentData,
    channel: int,
    apex_idx: int,
    left_idx: int,
    right_idx: int,
    config: ProcessingConfig,
    ms1_eic: np.ndarray,
    rt_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """收集与某个 MS1 峰共流出的 MS2 产品离子 (问题 B: MS-DIAL DIA MSDec 对齐)。

    判别器从"per-ion 色谱相关性硬门"换成 MS-DIAL 的"apex 时间对齐"结构判据
    (precursor↔product apex ≤2 scan), 让旧门砍掉的真实窄/带噪产品离子回来:

    1. 在 peak 区间 [left_idx, right_idx] 中收集所有候选产品离子 (沿用
       自适应 m/z 分箱)。
    2. 在扩展窗口 [left_idx-3, right_idx+3] 中为每个候选构造 EIC, 缺失
       的 cycle 补 0。
    3. 1% 基峰预过滤 (近 0, 对齐 MS-DIAL RelativeAmplitudeCutoff=0): 强度低于
       ``base_max × ms1b_chrom_corr_prefilter_pct`` 的候选直接丢弃。
    4. apex 对齐门 (主判别器): 候选 EIC 的 apex 落在 MS1 峰 apex
       ±``ms1b_apex_scan_tolerance`` scan 内才保留; apex 远离者 (噪声/邻近
       共流出物) 被剔。
    5. 轻量共流出门: 调 ``eic_coelution_ok`` 校验共流出 scan 数 ≥
       ``adaptive_n_correlated_threshold`` (= max(n_floor, n_fraction × min(peak_width)),
       默认 floor 2) 挡单点尖刺噪声。``ms1b_chrom_corr_threshold`` 默认 0.0 即
       per-ion Pearson 门关闭 (只挡严格反相关); 设回 0.70/0.90 可逆复现旧门。

    Parameters
    ----------
    ms1_eic : np.ndarray
        与 ``rt_array`` 同长度的 MS1 EIC 强度向量, 调用方在 stage1b
        主循环已构造, 不需要重复 ``extract_ms1_eic``。
    rt_array : np.ndarray
        所有 MS1 周期的 RT (与 ms1_eic 同长度)。
    """
    n_cycles = len(raw_data.cycles)
    # 扩展窗口给 EIC 一些上下文 (跟旧版一致, ±3 cycle)
    eic_left = max(0, left_idx - 3)
    eic_right = min(n_cycles - 1, right_idx + 3)
    n_pts = eic_right - eic_left + 1
    if n_pts <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # ms2_deconv dispatch (镜像 peak_detector "metra"/"msdial"): "msdec" (档 C)
    # 走 MS-DIAL MSDec, 产品离子集 = 前体 apex 单 scan 谱 (见 _collect_ms2_msdec),
    # 与档 B 的区间并集路径正交; "apex" (档 B 默认) 继续走下方 Pass1/2/3。
    if config.ms2_deconv == "msdec":
        return _collect_ms2_msdec(
            raw_data, channel, apex_idx, eic_left, eic_right, rt_array, config
        )

    # ------------------------------------------------------------------
    # 第一遍: 发现 peak 区间内的所有候选产品离子 (自适应 m/z 分箱)
    # ------------------------------------------------------------------
    adaptive_tol = max(config.eic_mz_tolerance, channel * 100e-6)
    bin_scale = max(1, int(1.0 / adaptive_tol))
    ion_bins: dict[int, float] = {}  # bin -> representative mz
    for ci in range(left_idx, right_idx + 1):
        if ci >= n_cycles:
            break
        cycle = raw_data.cycles[ci]
        if channel not in cycle.ms2_scans:
            continue
        prod_mz, prod_int = cycle.ms2_scans[channel]
        for mz_val, int_val in zip(prod_mz, prod_int):
            if int_val > 0:
                key = round(mz_val * bin_scale)
                if key not in ion_bins:
                    ion_bins[key] = float(mz_val)

    if not ion_bins:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # ------------------------------------------------------------------
    # 第二遍: 为每个候选构造 EIC (按时间序列, 缺失 scan 补 0)
    # ------------------------------------------------------------------
    match_tol = adaptive_tol  # 同一公式, 显式重命名增加可读性
    ion_keys: list[int] = []
    ion_mzs: list[float] = []
    ion_eics: list[np.ndarray] = []  # 每条 EIC 长度 == n_pts
    for key, rep_mz in ion_bins.items():
        eic = np.zeros(n_pts, dtype=np.float64)
        for i, ci in enumerate(range(eic_left, eic_right + 1)):
            if ci >= n_cycles:
                break
            cycle = raw_data.cycles[ci]
            if channel not in cycle.ms2_scans:
                continue
            prod_mz, prod_int = cycle.ms2_scans[channel]
            if len(prod_mz) > 0:
                mask = np.abs(prod_mz - rep_mz) <= match_tol
                if np.any(mask):
                    eic[i] = float(np.max(prod_int[mask]))
        ion_keys.append(key)
        ion_mzs.append(rep_mz)
        ion_eics.append(eic)

    # ------------------------------------------------------------------
    # 10% 基峰预过滤: 候选 EIC 最大强度 < base_max × pct 直接丢弃
    # base_max 取所有候选 EIC 的最大强度 (= 局部基峰), 跟 MS-DIAL
    # PeakCharacterEstimator 的预过滤规则一致。
    # ------------------------------------------------------------------
    max_per_ion = np.array(
        [float(np.max(e)) if e.size else 0.0 for e in ion_eics],
        dtype=np.float64,
    )
    base_max = float(np.max(max_per_ion)) if max_per_ion.size else 0.0
    prefilter_pct = float(config.ms1b_chrom_corr_prefilter_pct)
    if base_max <= 0.0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    prefilter_floor = base_max * prefilter_pct
    keep_pref = max_per_ion >= prefilter_floor
    if not np.any(keep_pref):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # ------------------------------------------------------------------
    # 准备色谱相关性所需的 MS1 EIC 窗口 + 各种宽度参数
    # ------------------------------------------------------------------
    # eic_coelution_ok 接受全长 EIC + (rt_start, rt_end), 内部用
    # rt_array 截窗。所以我们把整段 MS1 EIC + 候选 EIC 各自垫成等长向量,
    # 然后传扩展窗口的 RT 边界即可。
    ms1_eic_arr = np.asarray(ms1_eic, dtype=np.float64)
    rt_arr_full = np.asarray(rt_array, dtype=np.float64)
    if (
        ms1_eic_arr.size != rt_arr_full.size
        or ms1_eic_arr.size < n_cycles
    ):
        # 数据形状异常: 安全退路, 跟"没有候选"一致
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # RT 截窗用扩展窗口边界 (包含上下文 padding); eic_coelution_ok 内部
    # 会再按"两条 EIC 都 > 0"过滤, 所以 padding 不会污染相关性。
    rt_start = float(rt_arr_full[eic_left])
    rt_end = float(rt_arr_full[eic_right])

    # MS1 主峰宽度 = peak.right_index - peak.left_index + 1 (scan 数)
    ms1_peak_width = int(right_idx - left_idx + 1)
    # 产品离子默认峰宽: 扩展窗口的 scan 数 (但封顶 20, 跟一般 LC 峰宽相当)
    prod_peak_width = int(min(20, n_pts))
    pearson_threshold = float(config.ms1b_chrom_corr_threshold)
    # 共流出 scan 数自适应门的地板/比例 (stage1b 专属, 比 dedup/ISF 宽松, 见
    # config 问题 B 注释)。
    n_floor = int(getattr(config, "ms1b_chrom_corr_n_floor", 2))
    n_fraction = float(getattr(config, "ms1b_chrom_corr_n_fraction", 0.3))
    # apex 对齐门容差 (主判别器, MS-DIAL precursor↔model apex ≤2 scan)。
    apex_scan_tol = int(getattr(config, "ms1b_apex_scan_tolerance", 2))

    # ------------------------------------------------------------------
    # 第三遍: 色谱相关性 + n_correlated 门
    # ------------------------------------------------------------------
    result_mz: list[float] = []
    result_int: list[float] = []
    for j, (rep_mz, eic) in enumerate(zip(ion_mzs, ion_eics)):
        if not keep_pref[j]:
            continue
        # apex 对齐门 (MS-DIAL 结构判据, 主判别器): 候选 EIC 的 apex 落在
        # MS1 峰 apex ±apex_scan_tol scan 内才保留。eic 索引 0..n_pts-1 对应
        # cycle eic_left..eic_right, 故绝对 apex cycle = eic_left + argmax(eic)。
        # 取代旧的 per-ion Pearson 硬门做主噪声护栏: 真实碎片与前体同出峰,
        # apex 应重合; apex 远离者 (噪声/邻近共流出物) 被剔。
        if eic.size:
            cand_apex_idx = eic_left + int(np.argmax(eic))
            if abs(cand_apex_idx - apex_idx) > apex_scan_tol:
                continue
        # 把候选 EIC 嵌回长度 == rt_arr_full 的零向量, eic_coelution_ok
        # 在窗口外不会用到这些 0, 它会按 rt_start/rt_end 截窗。
        prod_eic_full = np.zeros_like(ms1_eic_arr)
        prod_eic_full[eic_left:eic_left + eic.size] = eic
        ok = eic_coelution_ok(
            prod_eic_full,
            ms1_eic_arr,
            rt_arr_full,
            rt_start,
            rt_end,
            peak_width_a_scans=prod_peak_width,
            peak_width_b_scans=ms1_peak_width,
            pearson_threshold=pearson_threshold,
            floor=n_floor,
            fraction=n_fraction,
        )
        if not ok:
            continue
        result_mz.append(float(rep_mz))
        result_int.append(float(max_per_ion[j]))

    if not result_mz:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # 按 m/z 排序
    order = np.argsort(result_mz)
    out_mz = np.array(result_mz, dtype=np.float64)[order]
    out_int = np.array(result_int, dtype=np.float64)[order]

    # 合并近邻离子 (自适应公差处理高 m/z 漂移)
    out_mz, out_int = merge_close_ions(
        out_mz, out_int,
        precursor_mz_nominal=channel,
        base_tolerance=config.eic_mz_tolerance,
    )

    # 末端二次相对强度阈值: 在已通过相关性筛选的离子上再去掉
    # 远低于本特征基峰的"陪跑"碎片 (沿用旧行为)
    if config.msms_relative_threshold > 0 and len(out_int) > 0:
        base_peak = float(np.max(out_int))
        keep = out_int >= base_peak * config.msms_relative_threshold
        out_mz = out_mz[keep]
        out_int = out_int[keep]

    return out_mz, out_int
