"""Cross-replicate alignment configuration shared by core and apps."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlignmentConfig:
    """Parameters for cross-replicate alignment."""
    rt_tolerance: float = 0.1
    mz_tolerance: float = 0.02
    mz_weight: float = 0.5
    rt_weight: float = 0.5
    ms2_mz_tolerance: float = 0.02
    match_threshold: float = 0.5
