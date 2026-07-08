"""Smoothing configuration shared by core and apps."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SmoothingConfig:
    """Parameters for chromatogram smoothing."""
    method: str = "savgol"
    window_length: int = 7
    polyorder: int = 3
    sigma: float = 1.0
