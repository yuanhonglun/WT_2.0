"""Compatibility shim: re-export shared clustering helpers from metabo_core."""
from metabo_core.algorithms.clustering import (  # noqa: F401
    cluster_peaks_by_rt,
    connected_components,
    select_representative,
)
