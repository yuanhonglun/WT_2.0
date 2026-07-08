"""把 ASFAM ScanCycle 的 MS1 survey 适配成 find_lc_ms1_features 需要的
Scan-like（.rt / .mz_array / .intensity_array / .ms_level）。"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class _MS1ScanView:
    rt: float
    mz_array: np.ndarray
    intensity_array: np.ndarray
    ms_level: int = 1


def ms1_survey_scans(cycles) -> list[_MS1ScanView]:
    """每个 cycle 的 MS1 谱 → 一个 Scan-like view（scan_idx == cycle_idx）。"""
    out = []
    for cyc in cycles:
        mz = cyc.ms1_mz if cyc.ms1_mz is not None else np.array([], dtype=np.float64)
        it = cyc.ms1_intensity if cyc.ms1_intensity is not None else np.array([], dtype=np.float64)
        out.append(_MS1ScanView(rt=float(cyc.rt), mz_array=np.asarray(mz, dtype=np.float64),
                                intensity_array=np.asarray(it, dtype=np.float64)))
    return out
