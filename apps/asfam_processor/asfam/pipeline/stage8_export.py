"""Stage 8: Export results.

W10 改造点:
  * 不再按 ``export_include_duplicates`` 过滤; 三模式统一 "保留+打标",
    GUI 通过 "Show duplicates" 切换。``is_duplicate`` / ``duplicate_type``
    随每行一起导出。
  * CSV 头加 ``# mode=asfam`` + ``# version=`` + ``# chromatographic_mode=lc``。
  * 前 16 列固定为公共 schema: ``mode_local, feature_id, mz, rt, height,
    area, n_fragments, score, name, formula, adduct, is_duplicate,
    duplicate_type, isotope_index, isotope_group_id, adduct_group_id``; 之后
    保留原有 ASFAM 字段。``isotope_index`` / ``isotope_group_id`` /
    ``adduct_group_id`` 为 PR-D 新增 (每个同位素峰 M/M+1/M+2 及每个加合物拷贝
    作独立 feature 计数, MS-DIAL 风格)。
  * 缺样样品的 height/area 列填空字符串 (非 0), 以便 ``pd.isna()`` 区分
    "该样品没匹配到" 和 "匹配到但强度 0"。
  * ``processing_report.txt`` 移到 ``output_dir/_debug/processing_report.txt``。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Callable

import pandas as pd
import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import Feature
from metabo_core import __version__ as METRA_VERSION

logger = logging.getLogger(__name__)


def run_stage8(
    features: list[Feature],
    output_dir: str,
    config: ProcessingConfig,
    stage_stats: Optional[dict] = None,
    progress_callback: Optional[Callable] = None,
    feedback_store: "Optional[object]" = None,
) -> dict[str, str]:
    """Export all results to files.

    Returns dict of output file paths.
    """
    logger.info("Stage 8: Exporting results...")
    output_dir = Path(output_dir)

    # W10: 不再 filter duplicates; 一律保留+打标。
    n_dup = sum(1 for f in features if f.is_duplicate)
    logger.info(
        "  Exporting all %d features (%d marked as duplicate)",
        len(features), n_dup,
    )
    export_features = features
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # 1. CSV feature table
    csv_path = output_dir / "features.csv"
    _export_csv(export_features, csv_path, config)
    outputs["csv"] = str(csv_path)
    logger.info("  Feature table: %s (%d features)", csv_path.name, len(export_features))

    # 2. MGF export (always on — Export tab was removed).
    mgf_path = output_dir / "ms2_spectra.mgf"
    _export_mgf(export_features, mgf_path)
    outputs["mgf"] = str(mgf_path)
    logger.info("  MGF: %s", mgf_path.name)

    # 3. MSP export (always on).
    msp_path = output_dir / "ms2_spectra.msp"
    _export_msp(export_features, msp_path)
    outputs["msp"] = str(msp_path)
    logger.info("  MSP: %s", msp_path.name)

    # 4. Processing report → _debug/ 子目录 (W10)。
    if stage_stats:
        report_path = output_dir / "_debug" / "processing_report.txt"
        _export_report(stage_stats, config, report_path)
        outputs["report"] = str(report_path)

    # Dump feedback bundle alongside CSV (purely additive; existing export
    # is unchanged).
    if feedback_store is not None:
        try:
            from metabo_gui.feedback import dump_feedback_to_export_dir
            feedback_store.run_context.export_dir = str(output_dir)
            dump_feedback_to_export_dir(output_dir, feedback_store)
        except Exception:
            logger.warning("Failed to dump feedback (non-fatal)", exc_info=True)

    if progress_callback:
        progress_callback("stage8", 1, 1, "Export complete")

    return outputs


def _export_csv(features: list[Feature], path: Path, config: ProcessingConfig) -> None:
    """Export feature table as CSV.

    Every feature with a library hit gets its annotation cells filled
    in (name, score, etc.); the ``annotated`` boolean column says
    whether that hit clears the single user-tunable
    ``matchms_similarity_threshold`` (= GUI "Library Match Thr"). W10:
    ``annotated`` 不再被用作导出过滤; 仅作 GUI 标记。
    """
    rows = []
    display_thr = float(
        getattr(config, "matchms_similarity_threshold", 0.3) or 0.0
    )
    # High-confidence (MS-DIAL IsReferenceMatched) also requires >= this many
    # matched peaks. Annotation now emits sub-threshold hits as suggestions
    # (name + score present) which must come out annotated=False.
    high_conf_min = int(getattr(config, "matchms_min_matched_peaks", 3) or 0)

    # 全集样品 ID (跨 features 求并集), 缺样 -> 空字符串 (不是 0)。
    rep_keys: list = []
    seen: set = set()
    for feat in features:
        for k in feat.heights.keys():
            if k not in seen:
                seen.add(k)
                rep_keys.append(k)
    rep_keys = sorted(rep_keys)

    for f in features:
        sel = f.selected_annotation
        # Two-tier: score must clear the display floor AND the match must have
        # >= high_conf_min matched peaks. A sparse "suggested" hit (fewer peaks)
        # keeps its name/score cells but is annotated=False.
        annotated = (
            sel is not None
            and sel.score is not None
            and float(sel.score) >= display_thr
            and int(getattr(sel, "n_matched", 0) or 0) >= high_conf_min
        )

        # W10 公共列前缀
        row = {
            "mode_local": "asfam",
            "feature_id": f.feature_id,
            "mz": round(f.precursor_mz, 5),
            "rt": round(f.rt, 3),
            "height": round(f.mean_height, 1),
            "area": round(f.mean_area, 1),
            "n_fragments": f.n_fragments,
            "score": round(sel.score, 4) if sel else "",
            "name": f.name or "",
            "formula": f.formula or "",
            "adduct": f.adduct or "",
            "is_duplicate": bool(f.is_duplicate),
            "duplicate_type": f.duplicate_type or "",
            "isotope_index": int(f.isotope_index),
            "isotope_group_id": f.isotope_group_id if f.isotope_group_id is not None else "",
            "adduct_group_id": f.adduct_group_id if f.adduct_group_id is not None else "",
        }

        # ASFAM 专有字段 (保留 mean_*/cv/sn_ratio/score 细分等)
        row["precursor_mz"] = round(f.precursor_mz, 5)
        row["rt_min"] = round(f.rt, 3)
        row["rt_left"] = round(f.rt_left, 3)
        row["rt_right"] = round(f.rt_right, 3)
        row["signal_type"] = f.signal_type
        # PR-E: detection provenance (ms1_driven / ms2_driven / both) — consumed
        # by the MS-DIAL comparison tool's "METRA 特色净增" (net-add) metric.
        row["detection_source"] = f.detection_source
        row["mean_height"] = round(f.mean_height, 1)
        row["mean_area"] = round(f.mean_area, 1)
        row["cv"] = round(f.cv, 3)
        row["sn_ratio"] = round(f.sn_ratio, 1)
        row["gaussian_similarity"] = round(f.gaussian_similarity, 3)
        # MS-DIAL 对齐的分项列（便于与 MS-DIAL 导出列逐项对比）。
        # composite_score == total_score（都 = AnnotationMatch.score）；composite_score
        # 保留作向后兼容，total_score 为与 MS-DIAL "Total score" 对齐的显式列名。
        row["composite_score"] = round(sel.score, 4) if sel else ""
        row["wdp_score"] = round(sel.wdp, 4) if sel else ""
        row["sdp_score"] = round(sel.sdp, 4) if sel else ""
        row["rdp_score"] = round(sel.rdp, 4) if sel else ""
        row["matched_pct"] = round(sel.matched_pct, 4) if sel else ""
        # Matched-peak count — the tier discriminator: annotated=True requires
        # n_matched >= matchms_min_matched_peaks; lower = suggested only.
        row["n_matched"] = int(sel.n_matched) if (sel and sel.n_matched is not None) else ""
        row["total_score"] = round(sel.total_score, 4) if sel else ""
        row["annotated"] = bool(annotated)
        row["ms2_spectrum"] = f.ms2_as_str()

        # 缺样填空字符串 (非 0), 列结构在所有行间一致。
        for rep_id in rep_keys:
            if rep_id in f.heights:
                row[f"height_rep{rep_id}"] = round(f.heights[rep_id], 1)
            else:
                row[f"height_rep{rep_id}"] = ""
            if rep_id in f.areas:
                row[f"area_rep{rep_id}"] = round(f.areas[rep_id], 1)
            else:
                row[f"area_rep{rep_id}"] = ""

        rows.append(row)

    df = pd.DataFrame(rows)
    # 头注释 + 写出。
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("# mode=asfam\n")
        fh.write(f"# version={METRA_VERSION}\n")
        fh.write("# chromatographic_mode=lc\n")
        df.to_csv(fh, index=False, lineterminator="\n")


def _export_mgf(features: list[Feature], path: Path) -> None:
    """Export MS2 spectra in MGF format."""
    with open(path, "w", encoding="utf-8") as f:
        for feat in features:
            if feat.n_fragments == 0:
                continue
            f.write("BEGIN IONS\n")
            f.write(f"FEATURE_ID={feat.feature_id}\n")
            f.write(f"PEPMASS={feat.precursor_mz:.5f}\n")
            f.write(f"RTINSECONDS={feat.rt * 60:.1f}\n")
            f.write(f"CHARGE=1+\n")
            if feat.name:
                f.write(f"NAME={feat.name}\n")
            if feat.formula:
                f.write(f"FORMULA={feat.formula}\n")
            for mz, intensity in zip(feat.ms2_mz, feat.ms2_intensity):
                f.write(f"{mz:.5f}\t{intensity:.1f}\n")
            f.write("END IONS\n\n")


def _export_msp(features: list[Feature], path: Path) -> None:
    """Export MS2 spectra in MSP/NIST format."""
    with open(path, "w", encoding="utf-8") as f:
        for feat in features:
            if feat.n_fragments == 0:
                continue
            f.write(f"NAME: {feat.name or feat.feature_id}\n")
            f.write(f"PRECURSORMZ: {feat.precursor_mz:.5f}\n")
            f.write(f"RETENTIONTIME: {feat.rt:.3f}\n")
            if feat.formula:
                f.write(f"FORMULA: {feat.formula}\n")
            f.write(f"Num Peaks: {feat.n_fragments}\n")
            for mz, intensity in zip(feat.ms2_mz, feat.ms2_intensity):
                f.write(f"{mz:.5f}\t{intensity:.0f}\n")
            f.write("\n")


def _export_report(
    stage_stats: dict, config: ProcessingConfig, path: Path,
) -> None:
    """Export processing summary report (W10 writes to ``_debug/`` subdir)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("METRA — ASFAM Processing Report\n")
        f.write("=" * 60 + "\n\n")
        for stage, stats in sorted(stage_stats.items()):
            f.write(f"--- {stage} ---\n")
            for key, val in stats.items():
                f.write(f"  {key}: {val}\n")
            f.write("\n")
