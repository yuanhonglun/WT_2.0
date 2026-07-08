"""Data models for the ASFAM processing pipeline.

ASFAM-specific raw-data containers stay here; feature-oriented models are
re-exported from metabo_core so app code keeps using ``asfam.models.Feature``
while extraction work proceeds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from metabo_core.models import (  # noqa: F401
    ProductIonEIC,
    DetectedPeak,
    AnnotationMatch,
    CandidateFeature,
    Feature,
)


@dataclass
class ScanCycle:
    """One scan cycle: 1 MS1 scan + N MRM-HR scans."""
    cycle_index: int
    rt: float
    ms1_mz: np.ndarray
    ms1_intensity: np.ndarray
    ms2_scans: dict


@dataclass
class RawSegmentData:
    """All data from one mzML file = one segment + one replicate."""
    file_path: str
    segment_name: str
    segment_low: int
    segment_high: int
    replicate_id: int
    n_cycles: int
    rt_array: np.ndarray
    precursor_list: list
    cycles: list
    collision_energy: float = 0.0
    # Maps each MS2 window's floor-key (int) to its actual acquired isolation
    # target m/z. ASFAM DIA acquires 1-Da windows whose target sits at a
    # fractional m/z that drifts with mass (~X.0 low → ~X.5 high). ms2_scans is
    # keyed by floor(target) so adjacent windows never collide — int(round())
    # collapses X.5 targets via banker's rounding, silently overwriting ~half
    # the windows above m/z ~800. Feature→window assignment reads this map to
    # pick the window whose target is nearest a feature's precise precursor m/z.
    precursor_targets: dict = field(default_factory=dict)
