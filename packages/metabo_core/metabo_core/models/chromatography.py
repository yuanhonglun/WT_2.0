"""Chromatography intermediates shared across pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ProductIonEIC:
    """EIC for one product ion in one MRM-HR channel."""
    precursor_mz_nominal: int
    product_mz: float
    rt_array: np.ndarray
    intensity_array: np.ndarray
    smoothed_intensity: Optional[np.ndarray] = None
    # Dense per-scan basePeakMz (same length as intensity_array); populated only
    # by mass-slice EICs. Other construction paths default to None so existing
    # constructors keep working.
    basepeak_mz: Optional[np.ndarray] = None


@dataclass
class DetectedPeak:
    """A chromatographic peak detected in an EIC."""
    precursor_mz_nominal: int
    product_mz: float
    rt_apex: float
    rt_left: float
    rt_right: float
    apex_index: int
    left_index: int
    right_index: int
    height: float
    area: float
    sn_ratio: float = 0.0
    gaussian_similarity: float = 0.0
