"""Regression tests for shared clustering helpers."""
import numpy as np

from metabo_core.algorithms.clustering import (
    cluster_peaks_by_rt,
    connected_components,
    select_representative,
)
from metabo_core.models.chromatography import DetectedPeak


def _peak(rt_apex: float, product_mz: float = 100.0, height: float = 1000.0) -> DetectedPeak:
    return DetectedPeak(
        precursor_mz_nominal=100,
        product_mz=product_mz,
        rt_apex=rt_apex,
        rt_left=rt_apex - 0.05,
        rt_right=rt_apex + 0.05,
        apex_index=0,
        left_index=0,
        right_index=0,
        height=height,
        area=height,
    )


def test_cluster_peaks_by_rt_groups_close_apexes():
    peaks = [_peak(1.00, 100.0), _peak(1.005, 110.0), _peak(2.0, 120.0)]
    clusters = cluster_peaks_by_rt(peaks, rt_tolerance=0.02, max_apex_span=0.05)
    rt_groups = sorted(sorted(round(p.rt_apex, 3) for p in c) for c in clusters)
    assert rt_groups == [[1.0, 1.005], [2.0]]


def test_cluster_peaks_by_rt_handles_empty():
    assert cluster_peaks_by_rt([], rt_tolerance=0.02, max_apex_span=0.05) == []


def test_connected_components_basic_graph():
    adjacency = {0: {1}, 1: {0}, 2: {3}, 3: {2}, 4: set()}
    comps = connected_components(adjacency)
    assert sorted(map(tuple, comps)) == [(0, 1), (2, 3), (4,)]


def test_select_representative_picks_lowest_mz_then_highest_intensity():
    indices = [0, 1, 2]
    mz = {0: 200.0, 1: 100.0, 2: 100.0}
    intensity = {0: 999.0, 1: 50.0, 2: 100.0}
    chosen = select_representative(indices, mz.__getitem__, intensity.__getitem__)
    assert chosen == 2
