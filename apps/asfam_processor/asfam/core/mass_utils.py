"""Compatibility shim: re-export shared mass utilities from metabo_core."""
from metabo_core.algorithms.mass_utils import (  # noqa: F401
    neutral_mass_from_adduct,
    classify_isotope_gap,
    peak_overlap_ratio,
    check_adduct_pair,
    max_c13_m1_ratio,
)
