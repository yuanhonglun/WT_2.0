"""Clustering algorithms: RT grouping, graph connected components."""
from __future__ import annotations

from typing import Callable
import numpy as np

from asfam.models import DetectedPeak


# ---------------------------------------------------------------------------
# RT-based peak clustering
# ---------------------------------------------------------------------------

def cluster_peaks_by_rt(
    peaks: list[DetectedPeak],
    rt_tolerance: float = 0.02,
) -> list[list[DetectedPeak]]:
    """Cluster detected peaks by retention time proximity.

    Uses TWO criteria (either triggers merge):
    1. Apex RT within rt_tolerance of cluster median (original)
    2. Peak boundaries overlap with the cluster's RT range

    After initial clustering, adjacent clusters are merged if their
    peak ranges overlap, catching edge cases where greedy assignment
    splits a compound due to median drift.

    Within each cluster, duplicate product m/z (within 0.005 Da)
    are deduplicated, keeping the highest-intensity peak.
    """
    if not peaks:
        return []

    sorted_peaks = sorted(peaks, key=lambda p: p.rt_apex)
    clusters: list[list[DetectedPeak]] = []
    current: list[DetectedPeak] = [sorted_peaks[0]]

    for peak in sorted_peaks[1:]:
        cluster_rt = np.median([p.rt_apex for p in current])
        c_left = min(p.rt_left for p in current)
        c_right = max(p.rt_right for p in current)

        apex_diff = abs(peak.rt_apex - cluster_rt)
        within_tol = apex_diff <= rt_tolerance
        # Only use boundary overlap if apexes are also reasonably close (< 3x tolerance).
        # This prevents merging different compounds that merely overlap chromatographically.
        overlaps = (peak.rt_left <= c_right and peak.rt_right >= c_left
                    and apex_diff <= rt_tolerance * 3)

        if within_tol or overlaps:
            current.append(peak)
        else:
            clusters.append(current)
            current = [peak]
    clusters.append(current)

    # Post-clustering merge: merge adjacent clusters whose peak ranges overlap
    # AND whose median apex RTs are close enough (< 3x tolerance)
    merged = [clusters[0]]
    for cluster in clusters[1:]:
        prev = merged[-1]
        prev_right = max(p.rt_right for p in prev)
        cur_left = min(p.rt_left for p in cluster)
        prev_median = np.median([p.rt_apex for p in prev])
        cur_median = np.median([p.rt_apex for p in cluster])

        if cur_left <= prev_right and abs(cur_median - prev_median) <= rt_tolerance * 3:
            merged[-1] = prev + cluster
        else:
            merged.append(cluster)

    # Deduplicate same product m/z within each cluster
    deduped_clusters = []
    for cluster in merged:
        by_mz: dict[int, DetectedPeak] = {}
        for peak in cluster:
            key = round(peak.product_mz * 200)  # bin to ~0.005 Da
            if key not in by_mz or peak.height > by_mz[key].height:
                by_mz[key] = peak
        deduped_clusters.append(list(by_mz.values()))

    return deduped_clusters


# ---------------------------------------------------------------------------
# Graph connected components (iterative DFS)
# ---------------------------------------------------------------------------

def connected_components(adjacency: dict[int, set[int]]) -> list[list[int]]:
    """Find connected components in an undirected graph.

    Parameters
    ----------
    adjacency : dict mapping node_id -> set of neighbor node_ids

    Returns
    -------
    List of components, each a sorted list of node IDs.
    """
    visited: set[int] = set()
    components: list[list[int]] = []

    all_nodes = set(adjacency.keys())
    for neighbors in adjacency.values():
        all_nodes.update(neighbors)

    for node in sorted(all_nodes):
        if node in visited:
            continue
        component = []
        stack = [node]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    stack.append(neighbor)
        component.sort()
        components.append(component)

    return components


def select_representative(
    component_indices: list[int],
    get_mz: Callable[[int], float],
    get_intensity: Callable[[int], float],
) -> int:
    """Select the representative feature from a component.

    Strategy: lowest m/z first, then highest intensity as tiebreaker.
    This selects the monoisotopic peak.
    """
    return min(
        component_indices,
        key=lambda idx: (get_mz(idx), -get_intensity(idx)),
    )
