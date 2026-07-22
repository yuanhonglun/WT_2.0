"""Processing configuration with all tunable parameters."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Literal

from metabo_core.config import (
    SmoothingConfig,
    PeakDetectionConfig,
    SimilarityConfig,
    AnnotationConfig,
    ConfidenceConfig,
    AlignmentConfig,
    GapFillConfig,
    JoinerConfig,
    RefinerConfig,
    lc_ms1_peak_config,
    lc_ms2_peak_config,
    MsdialPeakSpottingConfig,
    lc_msdial_config,
    MsdecConfig,
    lc_msdec_config,
)


@dataclass
class Ms2AmdisConfig:
    """AMDIS component-perception clustering params for ASFAM MS2 (opt-in).

    Defaults mirror GC-MS; calibrate on the benchmark (AMDIS plan T6). Only the
    sharpness-domain perception knobs — NOT the _ion_peaks jump/height/fall/
    max_window (the ASFAM path does not run AMDIS peak identification; it feeds
    externally-built IonPeaks from ASFAM detect_peaks).
    """
    sharpness_bins_per_scan: int = 10
    sharpness_range_factor: float = 50.0
    sharpness_cutoff_ratio: float = 0.75
    inclusion_cutoff_ratio: float = 0.3
    min_range_scans: int = 3
    match_max_window: int = 12          # component->cluster apex-match cap; NOT _ion_peaks window
    noise_bin_size: int = 50            # B1 short-EIC fallback knob (calibrate to MS2 EIC length, T6)
    min_noise_windows: int = 10
    aref_floor: float = 1.0


@dataclass
class ProcessingConfig:
    """All processing parameters with sensible defaults."""

    # -- General --
    ionization_mode: str = "positive"  # "positive" or "negative"
    n_workers: int = 4                 # multiprocessing pool size

    # -- Stage 1: MS2 peak detection --
    eic_mz_tolerance: float = 0.02     # Da, product ion EIC extraction
    eic_smoothing_method: str = "savgol"
    eic_smoothing_window: int = 7      # Savitzky-Golay window length (legacy; new peak detector uses MS-DIAL LWMA internally)
    eic_smoothing_polyorder: int = 3
    # ------------------------------------------------------------------
    # Peak detection — split into MS1 / MS2 nested configs.
    #
    # Old flat fields (``peak_height_threshold`` / ``peak_width_min`` /
    # ``peak_gaussian_threshold``) were removed in W4 (peak-detection
    # unification). MS1 and MS2 share a unified MS-DIAL three-gate filter
    # but the user-tunable amplitude floor differs: MS2 product-ion EICs
    # carry markedly less response than MS1, so MS2 uses a lower default
    # (200) and MS1 uses 500. Both nested configs are sourced from
    # ``metabo_core.config.lc_ms{1,2}_peak_config`` so DDA / DIA / cSIM
    # can reuse the same defaults.
    # ------------------------------------------------------------------
    ms1_peak: PeakDetectionConfig = field(default_factory=lc_ms1_peak_config)
    ms2_peak: PeakDetectionConfig = field(default_factory=lc_ms2_peak_config)
    # ------------------------------------------------------------------
    # Peak-detector mode (Phase 3, MS-DIAL port). ``builtin`` keeps the
    # current mass-slice ROI + detect_peaks behaviour byte-for-byte;
    # ``msdial`` (now the DEFAULT, Phase 4 validated) routes stage1b/stage1/stage2 through the faithful
    # MS-DIAL derivative engine + fixed-0.1Da SUM mass-slice orchestrator.
    # ``msdial_peak`` is the MS-DIAL parameter surface
    # (metabo_core.config.MsdialPeakSpottingConfig), sourced from
    # ``lc_msdial_config`` so DDA / DIA can reuse the same defaults.
    # ------------------------------------------------------------------
    peak_detector: Literal["builtin", "msdial"] = "msdial"
    msdial_peak: MsdialPeakSpottingConfig = field(default_factory=lc_msdial_config)
    rt_cluster_tolerance: float = 0.02  # min, for grouping co-eluting ions
    cluster_max_apex_span: float = 0.05  # min, max span of apex RTs within a cluster
    min_fragments_per_feature: int = 2
    # Stage 1 cross-EIC shape coherence: after RT-proximity clustering,
    # require the median pairwise Pearson correlation of the cluster's
    # fragment-ion EIC segments (over the cluster's RT window) to clear
    # this threshold. Catches "continuous baseline noise at different
    # m/z values" features that pass per-EIC Gaussian gates but whose
    # fragments are uncorrelated. Real metabolite clusters typically
    # have median ≥ 0.85. Single-peak clusters trivially pass (1.0).
    shape_corr_threshold: float = 0.7

    # -- Stage 1: MS2 ion recall (second pass) --
    recall_enabled: bool = True
    recall_min_intensity: float = 200.0  # minimum raw intensity at apex for recalled ion
    recall_min_consecutive: int = 4      # minimum consecutive nonzero cycles around apex
    recall_apex_window: int = 2          # +/- cycles around consensus apex to search

    # -- Stage 2: MS1 assignment --
    ms1_mz_tolerance: float = 0.01     # Da
    ms1_rt_tolerance: float = 0.05      # min (3s, tighter to prevent cross-assignment)
    # MS1-driven detection amplitude floor — a per-stage override of
    # ``ms1_peak.min_amplitude``（stage1b mass-slice 抽峰用；Stage 2 的
    # MS1 指派/晋升门 ``stage2_ms1_assignment`` 也复用本字段）。
    # （只落在 ASFAM 配置面，不动共用的 ``lc_ms1_peak_config``）。
    #
    # PR-E 验证调参（val_260624，3 段 ASFAM DIA 对标 MS-DIAL 5.5）：
    #   - 旧值 100 让含副本总数冲到 2490（>> MS-DIAL 1436），但独立质量
    #     抽检（raw-EIC 复检 795-824 弱尾）发现 height<300 的弱 feature
    #     约 50–70% 是连续基线 "grass" / 大峰拖尾 / 亚基线起伏等噪声——
    #     gaussian 门与检测器内部 S/N 对"骑在连续背景上的低幅度峰"不具
    #     判别力（csv S/N 可达 15 而 raw-EIC S/N 仅 0.3）。违反 spec
    #     "高质量是硬约束、不充噪音"。
    #   - height-vs-count 曲线（3 段合计）：floor 100→2490、200→2035、
    #     300→1617（实跑；1688 为按高度过滤 floor-100 集的预估）、
    #     500→1275（<1436）。300 是"仍 > 1436 且移除最噪声
    #     主导的 <300 弱尾"的最高可行档（spec D1 amplitude 区间 ~100–300
    #     的上界）。
    #   - 故定 300：在保住总数 > 1436 的前提下尽量收紧弱尾噪声。
    #   注：这是钝化处理（同时丢掉部分真·弱峰）。彻底解法是"噪声感知门"
    #   （raw-EIC S/N + 连续背景 grass 拒绝器，保留真弱峰、只杀背景通道），
    #   属算法层改造，见 内部验证记录 §6.4
    #   的升级建议（PR-C 后续）。
    ms1_min_height: float = 500.0  # MS-DIAL-parity floor: MS-DIAL's ASFAM export keeps Height>=500. Lowered from Phase-4 "floor1000" on 2026-06-26 by user decision — the full-range recall check (docs/validation/2026-06-26-recall-vs-msdial.md) found 55.6% of un-recalled MS-DIAL peaks sit in the 500–1000 band that floor1000 excluded (raw recall 78.7% vs 87.1% at floor parity), almost all at low m/z; 500 recovers that weak low-m/z tail to match MS-DIAL. Trade-off: re-admits some low-amplitude features (the <300 "grass" audit above still argues against going below ~300; 500 stays clear of it). NOTE: not yet re-validated on a fresh floor500 run — val_260626 was floor1000.
    # Stage 1b MS2 product-ion deconvolution (问题 B, MS-DIAL DIA MSDec-aligned).
    # 把"真正属于该 MS1 峰"的 MS2 产品离子从同窗口的噪声碎片中分离出来。
    #
    # 问题 B 改写（2026-06-26）：旧版用 per-ion 色谱相关性硬门（Pearson≥0.70，
    # 更早是 0.90）做主判别器，把大量"真实但窄/带噪"的产品离子滤掉 →
    # ms1_driven feature 常 n_frags=0 (ms2_quality="none") → 无法谱注释（打分
    # 低的下游主因）。核对 MS-DIAL C# 源码（MsdialCore/MSDec/MSDecHandler.cs,
    # MSDecProcess.cs）：MS-DIAL 的 MS2 反卷积**根本不用 per-ion 相关性**，其
    # 真正的结构判据是 precursor↔product 的 **apex 时间对齐（≤2 scan）**，叠加
    # 模型峰最小二乘分离重叠（本档不移植后者）。故档 B 把判别器从"相关性"换成
    # "apex 对齐"：
    #   - apex_scan_tolerance: 候选 EIC 的 apex 落在 MS1 apex ±N scan 内才保留
    #     （主判别器, MS-DIAL precursor↔model apex ≤2 scan）。
    #   - chrom_corr_threshold: per-ion Pearson 门。默认 0.0 = 关闭（只挡严格
    #     反相关）；设回 0.70 复现问题 A 行为、0.90 复现最初行为（可逆）。
    #   - n_floor / n_fraction: 轻量共流出 scan 数门 =
    #     max(n_floor, ceil(n_fraction × min(peak_width)))，挡单点尖刺噪声。
    #   - prefilter_pct: 候选 EIC 最大强度 < base_max × pct 即丢弃。降到 1%
    #     （近 0，对齐 MS-DIAL RelativeAmplitudeCutoff=0%），放真实弱 MS2 进来。
    # 这些门**只作用于 stage1b 反卷积**，不改 dedup/ISF 共用的
    # ``eic_coelution_ok`` 默认 (floor 5 / fraction 0.5)。
    ms1b_chrom_corr_prefilter_pct: float = 0.01
    ms1b_chrom_corr_threshold: float = 0.0
    ms1b_chrom_corr_n_floor: int = 2
    ms1b_chrom_corr_n_fraction: float = 0.3
    ms1b_apex_scan_tolerance: int = 2   # 候选 product-ion apex 与 MS1 apex 的最大 scan 距离
    # ------------------------------------------------------------------
    # Stage 1b MS2 deconvolution mode (档 C, MS-DIAL MSDec port).
    #   ``msdec`` — 档 C（DEFAULT, 2026-06-29 验证后由用户设默认）：MS-DIAL 最小
    #               二乘模型峰反卷积（分离重叠共流出物），走
    #               metabo_core.algorithms.msdec.deconvolute_ms2，输出谱不重新
    #               质心（遵循 MS-DIAL），用 getRefinedMsDecSpectrum 收尾。产品
    #               离子集 = 前体 apex 单 scan 谱（见 stage1b _collect_ms2_msdec），
    #               无法建模时退回 apex scan 原始谱。基准 3 段全 pipeline 注释
    #               55→176（+220%，真实增益），见 docs/validation/2026-06-29-*。
    #   ``apex``  — 档 B：apex 对齐启发式门（可回滚），上面 ms1b_* 门生效。
    # ``msdec`` 参数面 = metabo_core MsdecConfig；``msdec_view`` 把模型峰
    # 振幅门 min_amplitude 覆盖为 ASFAM 弱信号门 ms1_min_height（保住弱 MS2；
    # A/B 时设 msdec_use_weak_floor=False 用 MS-DIAL 默认 1000）。
    # ------------------------------------------------------------------
    ms2_deconv: Literal["apex", "msdec"] = "msdec"
    msdec: MsdecConfig = field(default_factory=lc_msdec_config)
    # When True (default, user decision), msdec_view() lowers the model-peak
    # amplitude floor to the weaker-signal MS1 floor (ms1_min_height) so the
    # weak MS2 that 档 B recovered are not cut. Set False to A/B against
    # MS-DIAL's faithful floor (MsdecConfig.min_amplitude, default 1000)
    # WITHOUT touching ms1_min_height (which would change MS1 peak finding).
    msdec_use_weak_floor: bool = True
    # -- Stage 1b: MS1 mass-slice 找峰（PR-C，取代每通道 1 Da 塌缩）--
    ms1_massslice_ppm: float = 15.0          # ROI lc_ppm 切片公差
    ms1_precise_eic_ppm: float = 50.0        # 精确 m/z 抽 MS1 EIC 的公差（QTOF 高 m/z 漂移）
    # MS1-driven feature → MS2 隔离窗指派容差 (Da)。ASFAM DIA 的 1-Da 隔离窗
    # target 随质量漂移 (~X.0 低段 → ~X.5 高段)，不能用 round(precursor) 对齐
    # 整数通道 (银行家舍入在 X.5 高段丢半数窗)。改为「取 target 最接近 feature
    # 精确前体 m/z 的实采窗」，容差 = 半窗 0.5 + MS1↔MS2 标定余量。仅 ASFAM 使用
    # (DDA/GC-MS 各自的 config 不含本字段)。
    ms2_isolation_window_tol: float = 0.6
    # 注：≥2 MS2 强制门已移除（Q3, 对齐 MS-DIAL）；MS2 质量改为标注
    ms1_isotope_mz_tol: float = 0.01   # Da, for isotope pattern extraction
    ms1_shape_weight: float = 0.3      # weight for peak shape in MS1 assignment scoring

    # -- Stage 2.5: MS2-only m/z inference --
    min_fragments_for_inference: int = 3
    enable_library_mz_inference: bool = False   # Library-matching step for ms2_only features.
                                                # Very slow and typically assigns only a handful of
                                                # features, so OFF by default. Neutral-loss consensus
                                                # still runs regardless.
    # W7：Stage 2.5 库匹配反推前体 m/z 的专属分数门槛（默认 0.8）。
    # 与 GUI 显示阈值 ``matchms_similarity_threshold`` 解耦——反推一旦
    # 命中就直接写入 ``inferred_mz`` 并影响下游全部阶段，比单纯显示
    # 更敏感，必须用更严格的阈值保证只在高置信命中时落库。历史值 0.3
    # 留下过太多噪声反推，W7 后改为 0.8。
    # ⚠ PR-B(0.7.260624.3)：综合分已改为 MS-DIAL TotalScore，量程
    # [0,1]→[0,2]，0.8 在新量程下不再"更严"；按新量程重调留 PR-E。
    library_mz_inference_threshold: float = 0.8
    # Pipeline emit threshold (Plan F-followup-5 mirror): lowered from
    # 0.8 to 0.3 so low-score matches enter ``annotation_matches`` and
    # show up in the feature table — matching GC-MS, where top-N hits
    # are emitted regardless of score and the GUI applies a separate
    # display-time threshold via the "Annotated only" filter.
    # The single, user-tunable library-match threshold (GUI "Library
    # Match Thr"). Does NOT gate the pipeline emit list — every hit
    # always lands in ``annotation_matches``. Used as the sole gate for:
    #   - the scatter "Annotated" / table "Annotated only" filter
    #   - the export CSV's ``annotated`` boolean column
    # Stage 2.5 has its own dedicated ``library_mz_inference_threshold``
    # (above) — it does NOT read this field.
    matchms_similarity_threshold: float = 0.7
    # MS-DIAL default: 3. NOTE (2026-07-05): this is now the HIGH-CONFIDENCE
    # threshold — the minimum matched peaks for the export/GUI ``annotated``
    # flag (mirrors MS-DIAL IsReferenceMatched). It is NOT the emit gate:
    # annotation emits suggestions down to 1 matched peak (see
    # ``annotation_view``), so sparse matches keep a name + score but come out
    # ``annotated=False``. Still consumed as-is by Stage 2.5 MS2-only inference
    # (stage2b), which is unchanged.
    matchms_min_matched_peaks: int = 3
    matchms_min_matched_pct: float = 0.25       # MS-DIAL: 25% of ref peaks
    # Minimum weighted dot product to accept a library hit. Removes
    # inflated-fallback-query false positives: an MS1-driven query bloated to
    # 100–240 peaks trivially covers a small reference's ~3 significant peaks
    # (matched_pct=1.0, high rdp) and clears the high-confidence line while the
    # true m/z-weighted spectral shape (wdp) is ~0. On selected-3 the wdp<0.10
    # band is 89% clear false-positive signature (inflated npk, n_matched 3–8,
    # sdp≈0, total压线 1.4–1.5); the true-match wdp band starts at ~0.103
    # (median 0.459), so 0.10 is the empirical boundary. Honest count effect on
    # selected-3: high-confidence 144 → ~110 (still ≫ MS-DIAL 75). Exposed by
    # T1 R1 clean spectra; see docs/validation/2026-07-02-t5-min-wdp-gate-*.
    # ASFAM-only opt-in — AnnotationConfig.min_wdp defaults 0.0 (GC-MS/DDA
    # unaffected).
    matchms_min_wdp: float = 0.10
    matchms_use_rt: bool = False                # use RT in scoring (user toggle)

    # -- Stage 3: Segment merge --
    merge_rt_tolerance: float = 0.05   # min
    merge_mz_tolerance: float = 0.02   # Da
    merge_ms2_cosine_threshold: float = 0.8

    # -- Stage 4: Isotope deduplication --
    isotope_rt_tolerance: float = 0.1           # min (search window; overlap ratio is primary criterion)
    isotope_apex_rt_n_cycles: int = 2           # hard max apex RT diff in #scan cycles (computed from data)
    isotope_apex_rt_fallback: float = 0.04      # min, fallback if cycle time unavailable
    isotope_overlap_ratio: float = 0.70
    isotope_mz_tolerance: float = 0.01          # Da, classic gaps
    isotope_integer_step_tolerance: float = 0.02  # Da, relaxed gaps
    isotope_fragment_mz_tolerance: float = 0.02
    isotope_precursor_exclusion: float = 1.5    # Da
    isotope_modified_cos_threshold: float = 0.85
    isotope_modified_cos_relaxed: float = 0.90
    isotope_min_matches: int = 3
    isotope_min_matches_relaxed: int = 4
    isotope_nl_cos_threshold: float = 0.85
    isotope_min_nl_matches: int = 3
    isotope_max_step: int = 4
    # MS2 step-pattern evidence (primary detector for isotope pairs).
    # True isotope partners share the property that the heavier feature's
    # MS2 "echoes" the lighter feature's MS2 shifted by ~+1.003355 Da
    # (when the fragment retains the heavy atom). This is robust to low
    # cosine similarity caused by intensity differences.
    isotope_step_top_n: int = 6                # # of high-response ions to inspect
    isotope_step_min_ratio: float = 0.5        # min fraction of top-N ions that must show the +delta echo
    isotope_step_mz_tolerance: float = 0.01    # Da, around mz + isotope_delta
    # Tier 2.5 fallback: sparse MS2-only isotope pair detection via fragment-set Jaccard
    # Disabled by default in v0.4.1 — replaced by step-pattern evidence (more accurate;
    # Jaccard caused false positives like F01082/F01123).
    isotope_ms2only_jaccard_fallback: bool = False
    isotope_fragment_jaccard_threshold: float = 0.50
    isotope_ms2only_apex_tight: float = 0.02   # min, tight apex gate for Tier 2.5

    # -- Stage 5: Adduct deduplication --
    adduct_rt_tolerance: float = 0.05  # min
    adduct_mw_tolerance: float = 0.02  # Da
    adduct_eic_pearson_threshold: float = 0.9

    # -- Stage 5b: Duplicate detection --
    duplicate_rt_n_cycles: int = 4         # hard max RT diff in #scan cycles (computed from data)
    duplicate_rt_fallback: float = 0.07    # min, fallback if cycle time unavailable (~4 cycles @ 1s)
    duplicate_mz_tolerance: float = 0.5    # Da
    duplicate_cosine_threshold: float = 0.85
    duplicate_min_matched: int = 3

    # -- Stage 6: ISF detection --
    isf_eic_pearson_threshold: float = 0.9
    isf_min_correlated_scans: int = 10
    # Dedicated ISF apex-RT gate (~2 MS1 cycles). Previously stage6 reused
    # adduct_rt_tolerance (0.05); decoupled + tightened to reduce ISF false
    # positives where the child/parent apex RT are actually separable.
    isf_rt_tolerance: float = 0.035
    isf_ms2_mz_tolerance: float = 0.02  # Da

    # -- Stage 7: Cross-replicate alignment --
    # Peaks are matched on the ion they were quantified on (CandidateFeature.
    # align_mz), never on the precursor -- ASFAM fragments everything, so the
    # precursor m/z is a 1-Da isolation window's centroid and drifts with
    # whatever co-isolated in that sample.
    #
    # 0.2 rather than 0.1: measured cross-sample RT drift has a p90 of 0.13-0.19
    # min, so +/-0.1 covered only 85-89% of same-compound pairs. It also widens
    # the gap filler's peak-top search, which is deliberate and is what MS-DIAL
    # does (LcmsGapFiller._rtTol = RetentionTimeAlignmentTolerance). It does NOT
    # widen the refiner's redundancy gate: RefinerConfig.rt_tolerance_cap pins
    # that at 0.1 -> a 0.05 min gate. Do not remove that cap.
    alignment_rt_tolerance: float = 0.2  # min
    # Not tightened to 0.01: measured, that makes the row count go *up*
    # (rice +11.4%, cancer +8.3%). The key was the problem, not the window.
    alignment_mz_tolerance: float = 0.02  # Da
    # The three claim-score weights sum to 1. 0.3/0.3/0.4 reproduces the blend
    # the Hungarian-era scorer used for PRODUCT peaks. MS1-route matching is
    # geometry-only because its AIF spectrum is not precursor-specific; there
    # the m/z/RT terms renormalize to 0.5/0.5.
    alignment_mz_weight: float = 0.3
    alignment_rt_weight: float = 0.3
    alignment_ms2_weight: float = 0.4
    # PRODUCT-route MS2 identity gate on the alignment box. Two product peaks
    # share a box only if their spectra cannot tell them apart. Without it the
    # box is pure proximity, and
    # proximity is what the build step used to forbid a peak from founding a
    # master of its own -- while a sample may put only one peak in a master. The
    # surplus peaks ended up in no row and no cell: 1,242 (rice) / 2,026 (cancer)
    # of them a *different compound* from the peak that beat them. 0 disables it.
    alignment_ms2_identity_threshold: float = 0.5
    # Fewer fragments than this and the cosine is not evidence; the gate abstains
    # and the pair stays in one box. Abstaining must suppress, not split, or every
    # fragment-poor peak gets a row of its own.
    alignment_ms2_identity_min_fragments: int = 3
    # Total fragment counts alone are insufficient: both spectra may be rich yet
    # share only one ubiquitous ion.  Identity requires this many actual pairs.
    alignment_ms2_identity_min_matched_fragments: int = 3
    # Sample that seeds the master list. None = pick automatically by S/N quality.
    # After the union master list this only decides who claims a bucket first.
    alignment_reference_sample: Optional[str] = None
    # MS2 cosine at or above which a visible product-route (ms2_only) row and a
    # visible MS1-route row at the same RT are one compound -> the product row is
    # marked "ms1_covered" and stops counting as an MS2-only detection. The two
    # rows have no comparable m/z (precursor vs fragment), so MS2 is the only
    # route-independent identity signal there is. Read only by stage 7: it must
    # stay in spill._FINGERPRINT_EXCLUDED or every _work/ checkpoint is voided.
    #
    # Not the joiner's alignment_ms2_identity_threshold above: that one asks
    # whether two PRODUCT peaks in one box are the same compound, this one
    # whether an MS1 row already covers a product row. Same instrument,
    # different questions, and they were tuned on different evidence.
    alignment_ms1_covered_threshold: float = 0.7
    gap_fill_enabled: bool = True
    gap_fill_rt_expansion: float = 1.5
    # Fragment chromatograms stored per spot in alignment.eic, taken from the
    # representative sample. Only the MS2 panel of the EIC viewer reads them.
    eic_store_top_fragments: int = 10

    # -- Annotation reranker (optional, default disabled) --
    reranker_enabled: bool = False
    reranker_mode: str = "default"
    reranker_alpha: float = 0.7
    reranker_ri_sigma: float = 10.0
    reranker_top_k_explained: int = 3
    reranker_student_module: Optional[str] = None

    # -- Stage 8: Export --
    export_mgf: bool = True
    export_msp: bool = True
    export_report: bool = True
    export_include_duplicates: bool = False

    # -- Quality filtering --
    # ``final_*`` thresholds were dead config in the legacy code path; the
    # MS-DIAL three-gate filter inside ``detect_peaks`` now subsumes them.
    #
    # ms2_driven (stage1) MS2 cleanup floors. Every fragment reaching the
    # stage1 cleanup has already passed the full MS-DIAL 3-gate detector
    # (min_amplitude=200 + gaussian>=0.85 + S/N + prominence) AND the cross-EIC
    # shape gate (>=0.7), so the legacy absolute floor of 1000 only re-cut
    # validated real peaks that MS-DIAL keeps — the direct cause of "missing
    # low-response ions" (diagnosis 2026-07-02, spec §2.3/§2.6). Lower the
    # absolute floor to the detection amplitude gate (200) so cleanup no longer
    # second-guesses the detector, and use a dedicated low relative floor for the
    # ms2_driven path. NOTE: ``msms_relative_threshold`` (below) is deliberately
    # left at 0.02 because stage1b (ms1_driven/MSDec, T1) also reads it — do NOT
    # couple T2's floor to T1's path (see stage1b_ms1_detection.py:489-491).
    #
    # DECOUPLED ADMISSION GUARD (option B, user decision 2026-07-02). Lowering the
    # cleanup floor alone also lowered the stage1 feature-admission guard
    # (stage1_ms2_detection.py:250 "reject if brightest fragment < floor"), which
    # on selected-3 exploded ms2_driven features ~3x / ms2_only +2029 with
    # recall-dominated, annotation-unverified weak clusters (validation
    # 2026-07-02 §2/§4). ``ms2_driven_feature_floor`` restores the base-peak
    # quality bar for feature ADMISSION (1000) while ``msms_intensity_threshold``
    # (200) keeps the [200,1000) FRAGMENTS of admitted features. So a feature is
    # created only when its brightest fragment reaches 1000, but once admitted it
    # retains its weaker real fragments (the primary fix). Raising this to 200
    # reverts to the aggressive "recover MS2-only" behavior of option A.
    msms_intensity_threshold: float = 200.0      # stage1 (ms2_driven) cleanup absolute floor = detection gate
    msms_relative_threshold: float = 0.02        # stage1b (T1/MSDec) relative floor — unchanged
    ms2_driven_rel_floor: float = 0.01           # stage1 (ms2_driven) cleanup relative floor
    ms2_driven_feature_floor: float = 1000.0     # stage1 (ms2_driven) feature-admission base-peak bar
    msms_min_ions: int = 1

    # -- Stage 1: MS2 clustering path (opt-in AMDIS) --
    # "rt" = 1D RT-proximity cluster_peaks_by_rt (current default, unchanged).
    # "amdis" = AMDIS component-perception (sharpness NMS) for co-eluting
    # compound splitting (e.g. F01368). Opt-in; amdis path does NOT call
    # _resolve_peak_ownership (perception does component-level ownership).
    ms2_clustering: str = "rt"                    # "rt" | "amdis"
    ms2_amdis: Ms2AmdisConfig = field(default_factory=Ms2AmdisConfig)

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save config to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "ProcessingConfig":
        """Load config from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Rebuild nested config dataclasses from dicts — ``json.load``
        # returns plain dicts. Each nested key maps to its OWN dataclass
        # type: ``msdial_peak`` must be rebuilt as MsdialPeakSpottingConfig,
        # NOT PeakDetectionConfig, or its MS-DIAL-only fields are silently
        # dropped (→ AttributeError in msdial mode after any save/load).
        nested_pd_types = {
            "ms1_peak": PeakDetectionConfig,
            "ms2_peak": PeakDetectionConfig,
            "msdial_peak": MsdialPeakSpottingConfig,
            "msdec": MsdecConfig,
            "ms2_amdis": Ms2AmdisConfig,   # rebuild nested dataclass on load
        }
        nested = {}
        for key, klass in nested_pd_types.items():
            if key in data and isinstance(data[key], dict):
                nested[key] = klass(
                    **{
                        k: v
                        for k, v in data[key].items()
                        if k in klass.__dataclass_fields__
                    }
                )
        flat = {
            k: v
            for k, v in data.items()
            if k in cls.__dataclass_fields__ and k not in nested_pd_types
        }
        return cls(**flat, **nested)

    # -----------------------------------------------------------------------
    # Core config views
    #
    # These compose small reusable configs from metabo_core.config from the
    # current flat ASFAM parameter surface. They let WTV and other apps reuse
    # the same algorithm-level config shape without inheriting ASFAM's full
    # ``ProcessingConfig`` namespace.
    # -----------------------------------------------------------------------
    def smoothing_view(self) -> SmoothingConfig:
        return SmoothingConfig(
            method=self.eic_smoothing_method,
            window_length=self.eic_smoothing_window,
            polyorder=self.eic_smoothing_polyorder,
        )

    def peak_detection_view(self) -> PeakDetectionConfig:
        """Return the MS2 peak-detection config.

        Historically ``peak_detection_view`` exposed a single flat
        view; now MS1 / MS2 each have their own nested config. The
        legacy view returns the MS2 config because stage1 (the original
        consumer) builds product-ion EICs. Callers needing the MS1
        config should read ``self.ms1_peak`` directly.
        """
        return self.ms2_peak

    def similarity_view(self) -> SimilarityConfig:
        return SimilarityConfig(
            mz_tolerance=self.eic_mz_tolerance,
            ms1_tolerance=self.ms1_mz_tolerance,
            use_rt=self.matchms_use_rt,
        )

    def annotation_view(self) -> AnnotationConfig:
        # Pipeline emits every plausible match (similarity_threshold=0)
        # so the GUI can show "any annotation" without the pipeline
        # silently dropping low-score hits. The display gate lives in
        # ``matchms_annotated_display_threshold``.
        #
        # Two-tier confidence (mirrors MS-DIAL IsAnnotationSuggested vs
        # IsReferenceMatched). The ``min_matched_peaks>=3`` count requirement is
        # NOT applied as an emit gate here: we emit down to 1 matched peak so
        # sparse matches (e.g. precursor-only [M+Na]+ references, MS1-dominant
        # queries) still get a name + score as *suggestions*. The high-
        # confidence ``>=matchms_min_matched_peaks`` requirement is enforced
        # later at the export / GUI ``annotated`` flag — by tiering, never by
        # dropping the hit. ``min_peaks_to_match=1`` likewise lets 1-peak
        # spectra participate on both the query and the reference side. The
        # spectral-quality gates (min_matched_pct / min_wdp) are unchanged, so a
        # suggestion still has to be a genuine partial match.
        return AnnotationConfig(
            similarity_threshold=0.0,
            min_matched_peaks=1,
            min_peaks_to_match=1,
            min_matched_pct=self.matchms_min_matched_pct,
            min_wdp=self.matchms_min_wdp,
            use_rt=self.matchms_use_rt,
        )

    def reranker_view(self) -> "RerankerConfig":
        from metabo_core.config.reranker import RerankerConfig
        return RerankerConfig(
            enabled=self.reranker_enabled,
            mode=self.reranker_mode,
            alpha=self.reranker_alpha,
            ri_sigma=self.reranker_ri_sigma,
            top_k_explained=self.reranker_top_k_explained,
            student_module=self.reranker_student_module,
        )

    def msdec_view(self) -> MsdecConfig:
        """Return the MSDec config with the ASFAM weak-signal amplitude floor.

        MS-DIAL's faithful model-peak amplitude floor is 1000, but ASFAM
        carries many weak MS2 signals, so the model-peak gate is lowered to
        the same floor used for MS1-driven detection (``ms1_min_height``),
        retaining the weak MS2 that 档 B recovered. A/B against MS-DIAL's
        faithful floor is done by setting ``msdec_use_weak_floor = False``
        (then ``MsdecConfig.min_amplitude``, default 1000, is used directly).
        """
        from dataclasses import replace

        if self.msdec_use_weak_floor:
            return replace(self.msdec, min_amplitude=self.ms1_min_height)
        return replace(self.msdec)

    def alignment_view(self) -> AlignmentConfig:
        return AlignmentConfig(
            rt_tolerance=self.alignment_rt_tolerance,
            mz_tolerance=self.alignment_mz_tolerance,
            mz_weight=self.alignment_mz_weight,
            rt_weight=self.alignment_rt_weight,
            ms2_mz_tolerance=self.eic_mz_tolerance,
        )

    def joiner_view(self) -> JoinerConfig:
        return JoinerConfig(
            rt_tolerance=self.alignment_rt_tolerance,
            mz_tolerance=self.alignment_mz_tolerance,
            mz_weight=self.alignment_mz_weight,
            rt_weight=self.alignment_rt_weight,
            ms2_weight=self.alignment_ms2_weight,
            ms2_mz_tolerance=self.eic_mz_tolerance,
            ms2_identity_threshold=self.alignment_ms2_identity_threshold,
            ms2_identity_min_fragments=max(
                3, int(self.alignment_ms2_identity_min_fragments),
            ),
            ms2_identity_min_matched_fragments=(
                max(3, int(self.alignment_ms2_identity_min_matched_fragments))
            ),
            # Correctness invariants, deliberately not GUI-disableable.  These
            # opt ASFAM into new shared-core behaviour while DDA keeps defaults.
            use_reliable_ms2_identity=True,
            conserve_detected_peaks=True,
            # AIF spectra belong to a whole co-eluting segment, so their
            # cross-sample variation cannot veto an MS1 chromatographic peak.
            # PRODUCT spectra remain identity-gated in the shared joiner.
            use_ms2_identity_for_ms1=False,
            reference_sample=self.alignment_reference_sample,
        )

    def gap_fill_view(self) -> GapFillConfig:
        """Tolerances of the chromatograms the *detected* peaks were measured on.

        ``ms1_mz_tolerance`` and ``product_mz_tolerance`` are not free knobs: the
        first is the window ``msdial_ms1_features._recalculate`` re-integrates a
        detected MS1 peak in, the second is the slice ``build_slice_eics_sum``
        summed to give a fragment its intensity. A filled value has to come off
        the same window as the detected values it will sit beside.
        """
        return GapFillConfig(
            rt_tolerance=self.alignment_rt_tolerance,
            ms1_mz_tolerance=self.msdial_peak.centroid_ms1_tolerance,
            product_mz_tolerance=self.msdial_peak.mass_slice_width,
            smoothing_level=self.msdial_peak.smoothing_level,
            rt_expansion=self.gap_fill_rt_expansion,
        )

    def refiner_view(self) -> RefinerConfig:
        """Gates for the post-alignment redundancy pass, narrowed from the join's.

        The refiner caps and halves the RT tolerance itself, so passing the raw
        alignment tolerance is correct — see :class:`RefinerConfig`.

        ``ms2_mz_tolerance`` is the joiner's, deliberately: the two passes score
        the same pair of spectra and must not disagree about what a matched
        fragment is.
        """
        return RefinerConfig(
            rt_tolerance=self.alignment_rt_tolerance,
            mz_tolerance=self.alignment_mz_tolerance,
            ms2_identity_threshold=self.alignment_ms1_covered_threshold,
            ms2_mz_tolerance=self.eic_mz_tolerance,
            same_route_redundancy=False,
            use_reliable_ms2_identity=True,
            ms2_identity_min_fragments=max(
                3, int(self.alignment_ms2_identity_min_fragments),
            ),
            ms2_identity_min_matched_fragments=(
                max(3, int(self.alignment_ms2_identity_min_matched_fragments))
            ),
            same_route_ms2_identity_threshold=(
                self.alignment_ms2_identity_threshold
            ),
            visible_keepers_only=True,
            require_product_window_match=True,
            require_cross_route_window_match=True,
            preserve_cross_route_unique_detections=True,
        )

    def confidence_view(self) -> ConfidenceConfig:
        """What "annotated" means, for the one predicate that decides it.

        Read by the ``annotated`` column of ``features.csv`` and by the refiner's
        placement order. Both go through
        :func:`metabo_core.annotation.is_high_confidence`; nothing else may
        re-derive this.
        """
        return ConfidenceConfig(
            score_threshold=float(
                getattr(self, "matchms_similarity_threshold", 0.3) or 0.0
            ),
            min_matched_peaks=int(
                getattr(self, "matchms_min_matched_peaks", 3) or 0
            ),
        )
