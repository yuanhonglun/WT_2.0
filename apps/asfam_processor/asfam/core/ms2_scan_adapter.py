"""Adapt one MRM-HR channel's MS2 spectra (across all cycles) into the
Scan-like shape (.rt / .mz_array / .intensity_array) that build_slice_eics_sum
expects. Mirrors ms1_survey_scans: every cycle is covered (cycles missing the
channel get an empty-spectrum placeholder) so that scan_idx == cycle_idx. The
intensity floor is deferred to the slicer's >0 filter (D3: no >=10 floor here)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _MS2ScanView:
    rt: float
    mz_array: np.ndarray
    intensity_array: np.ndarray
    ms_level: int = 2


def ms2_channel_scans(raw_data, channel: int) -> list[_MS2ScanView]:
    """Each cycle's MS2 spectrum for this channel -> one Scan-like view (scan_idx == cycle_idx)."""
    out: list[_MS2ScanView] = []
    for cyc in raw_data.cycles:
        scan = cyc.ms2_scans.get(channel)
        if scan is None:
            mz = np.array([], dtype=np.float64)
            it = np.array([], dtype=np.float64)
        else:
            mz = np.asarray(scan[0], dtype=np.float64)
            it = np.asarray(scan[1], dtype=np.float64)
        out.append(_MS2ScanView(rt=float(cyc.rt), mz_array=mz, intensity_array=it))
    return out
