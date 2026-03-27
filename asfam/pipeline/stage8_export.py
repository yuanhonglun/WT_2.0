"""Stage 8: Export results."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Callable

import pandas as pd
import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import Feature

logger = logging.getLogger(__name__)


def run_stage8(
    features: list[Feature],
    output_dir: str,
    config: ProcessingConfig,
    stage_stats: Optional[dict] = None,
    progress_callback: Optional[Callable] = None,
) -> dict[str, str]:
    """Export all results to files.

    Returns dict of output file paths.
    """
    logger.info("Stage 8: Exporting results...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # 1. CSV feature table
    csv_path = output_dir / "features.csv"
    _export_csv(features, csv_path, config)
    outputs["csv"] = str(csv_path)
    logger.info("  Feature table: %s (%d features)", csv_path.name, len(features))

    # 2. MGF export
    if config.export_mgf:
        mgf_path = output_dir / "ms2_spectra.mgf"
        _export_mgf(features, mgf_path)
        outputs["mgf"] = str(mgf_path)
        logger.info("  MGF: %s", mgf_path.name)

    # 3. MSP export
    if config.export_msp:
        msp_path = output_dir / "ms2_spectra.msp"
        _export_msp(features, msp_path)
        outputs["msp"] = str(msp_path)
        logger.info("  MSP: %s", msp_path.name)

    # 4. Processing report
    if config.export_report and stage_stats:
        report_path = output_dir / "processing_report.txt"
        _export_report(stage_stats, config, report_path)
        outputs["report"] = str(report_path)

    if progress_callback:
        progress_callback("stage8", 1, 1, "Export complete")

    return outputs


def _export_csv(features: list[Feature], path: Path, config: ProcessingConfig) -> None:
    """Export feature table as CSV."""
    rows = []
    for f in features:
        row = {
            "feature_id": f.feature_id,
            "precursor_mz": round(f.precursor_mz, 5),
            "rt_min": round(f.rt, 3),
            "rt_left": round(f.rt_left, 3),
            "rt_right": round(f.rt_right, 3),
            "signal_type": f.signal_type,
            "n_fragments": f.n_fragments,
            "mean_height": round(f.mean_height, 1),
            "mean_area": round(f.mean_area, 1),
            "cv": round(f.cv, 3),
            "sn_ratio": round(f.sn_ratio, 1),
            "formula": f.formula or "",
            "adduct": f.adduct or "",
            "name": f.name or "",
            "ms2_spectrum": f.ms2_as_str(),
        }
        # Add per-replicate columns
        for rep_id in sorted(f.heights.keys()):
            row[f"height_rep{rep_id}"] = round(f.heights.get(rep_id, 0), 1)
            row[f"area_rep{rep_id}"] = round(f.areas.get(rep_id, 0), 1)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


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
    """Export processing summary report."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("ASFAMProcessor Processing Report\n")
        f.write("=" * 60 + "\n\n")
        for stage, stats in sorted(stage_stats.items()):
            f.write(f"--- {stage} ---\n")
            for key, val in stats.items():
                f.write(f"  {key}: {val}\n")
            f.write("\n")
