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
    max_apex_span: float = 0.05,
) -> list[list[DetectedPeak]]:
    """Cluster detected peaks by retention time proximity.

    Uses TWO criteria (either triggers merge), BOTH with a max-apex-span cap:
    1. Apex RT within rt_tolerance of cluster median
    2. Peak boundaries overlap AND apex RT within 2x tolerance
       AND admitting the peak does not push the cluster's
       (max_apex - min_apex) above max_apex_span.

    After initial clustering, adjacent clusters are merged only if their
    median apex RTs are within 2x tolerance AND the merged apex span stays
    within max_apex_span.

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
        c_apex_min = min(p.rt_apex for p in current)
        c_apex_max = max(p.rt_apex for p in current)
        new_apex_min = min(c_apex_min, peak.rt_apex)
        new_apex_max = max(c_apex_max, peak.rt_apex)
        new_span = new_apex_max - new_apex_min

        apex_diff = abs(peak.rt_apex - cluster_rt)
        within_tol = apex_diff <= rt_tolerance
        # Boundary overlap shortcut: apex must still be within 2x tolerance
        # AND the resulting cluster apex-span must not exceed max_apex_span.
        # This prevents co-elution smearing from merging chromatographically
        # distinct compounds into one feature.
        overlaps = (peak.rt_left <= c_right and peak.rt_right >= c_left
                    and apex_diff <= rt_tolerance * 2
                    and new_span <= max_apex_span)

        if within_tol and new_span <= max_apex_span:
            current.append(peak)
        elif overlaps:
            current.append(peak)
        else:
            clusters.append(current)
            current = [peak]
    clusters.append(current)

    # Post-clustering merge: merge adjacent clusters whose peak ranges overlap
    # AND whose median apex RTs are close AND the merged apex span is bounded.
    merged = [clusters[0]]
    for cluster in clusters[1:]:
        prev = merged[-1]
        prev_right = max(p.rt_right for p in prev)
        cur_left = min(p.rt_left for p in cluster)
        prev_median = np.median([p.rt_apex for p in prev])
        cur_median = np.median([p.rt_apex for p in cluster])
        combined_apex_min = min(p.rt_apex for p in prev + cluster)
        combined_apex_max = max(p.rt_apex for p in prev + cluster)
        combined_span = combined_apex_max - combined_apex_min

        if (cur_left <= prev_right
                and abs(cur_median - prev_median) <= rt_tolerance * 2
                and combined_span <= max_apex_span):
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
