"""GC-MS-specific reusable algorithms living in metabo_core.

This subpackage hosts logic that is GC-MS-specific but still shared across
GC-MS apps and modes (notably the data-derived acquired-ion-set logic for
cSIM library matching). It must not import any app-level modules.
"""
from metabo_core.gcms.library_matching import (
    acquired_ion_set,
    csim_intersected_cosine,
    fullscan_cosine,
)

__all__ = [
    "acquired_ion_set",
    "csim_intersected_cosine",
    "fullscan_cosine",
]
