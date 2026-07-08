"""Mass spectrometry scan dataclass shared across LC-MS pipelines.

This is a generic container for one MS scan (MS1 or MS2) coming out of an
mzML reader. Higher-level pipelines convert these scans into EICs, features,
or candidate matches. The DDA-specific precursor metadata (``precursor_mz``,
``precursor_intensity``, ``isolation_window_lower``, ``isolation_window_upper``)
is general LC-MS information and lives on this shared dataclass rather than
in any one app's models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Scan:
    """One MS scan from an mzML file.

    Parameters
    ----------
    scan_id : int
        Sequential index of the scan within the run (0- or 1-based per reader).
    ms_level : int
        1 for MS1 survey scans, 2 for MS2 product-ion scans.
    rt : float
        Retention time in minutes.
    mz_array : np.ndarray
        m/z values for the scan's centroided peaks.
    intensity_array : np.ndarray
        Intensities aligned with ``mz_array``.
    precursor_mz : Optional[float]
        For MS2 scans, the selected precursor m/z. None for MS1.
    precursor_intensity : Optional[float]
        For MS2 scans, the precursor intensity if reported by the instrument.
    isolation_window_lower : Optional[float]
        Lower bound of the isolation window (m/z). For MS2 with a window of
        +/- 0.5 Da centred at 270.0 the lower bound is 269.5.
    isolation_window_upper : Optional[float]
        Upper bound of the isolation window (m/z).
    """

    scan_id: int
    ms_level: int
    rt: float
    mz_array: np.ndarray
    intensity_array: np.ndarray
    precursor_mz: Optional[float] = None
    precursor_intensity: Optional[float] = None
    isolation_window_lower: Optional[float] = None
    isolation_window_upper: Optional[float] = None
