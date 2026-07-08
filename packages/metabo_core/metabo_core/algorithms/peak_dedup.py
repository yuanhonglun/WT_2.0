"""Post-detection feature dedup.

Why this exists: m/z trace builders that cluster raw centroid points by
ppm boundary frequently split a single physical chromatographic peak
into two or more "traces" when the centroid drifts across the ppm
boundary mid-trace. Downstream peak picking then emits one feature per
trace, producing near-identical features with the same RT apex / peak
shape and m/z values 1–2× ppm tolerance apart. ASFAM does not see this
because its MS1 peaks are picked given a *known* precursor m/z; DDA has
to discover the m/z and therefore hits the boundary issue.

This module dedups *after* peak picking by grouping features whose
(m/z, rt_apex) fall within a configurable tolerance and the highest
``ms1_height`` representative wins. The losers are dropped from the
feature list (they would clutter alignment and library annotation).

The dedup ignores apex intensity ratio on purpose: if the trace
splitter divides the ion unevenly the two halves can have very different
heights yet still be the same physical peak.
"""
from __future__ import annotations

from typing import Iterable, List

from metabo_core.models import CandidateFeature


def dedup_features_by_proximity(
    features: Iterable[CandidateFeature],
    ppm_tolerance: float = 20.0,
    rt_tolerance: float = 0.05,
) -> List[CandidateFeature]:
    """Drop features that share (m/z within ppm, rt_apex within minutes)
    with another feature; keep the one with the highest ``ms1_height``.

    Parameters
    ----------
    features : iterable of CandidateFeature
        Per-replicate candidate feature list straight from peak picking.
    ppm_tolerance : float, default 20.0
        Two features are candidates for merging if their m/z values are
        within this many ppm. Default is 2× the typical 10 ppm trace
        tolerance so split-by-boundary cases are caught.
    rt_tolerance : float, default 0.05
        Two features must additionally have their RT apexes within this
        many minutes (~10 MS1 cycles at 285 ms per cycle) to be
        considered duplicates.

    Returns
    -------
    list[CandidateFeature]
        New list with duplicates removed, preserving the original
        ordering of survivors.
    """
    feats = list(features)
    if len(feats) <= 1:
        return feats

    # Sort indices by m/z so a single forward scan finds all neighbours
    # within the ppm window. ``order_by_mz[k]`` is the original index of
    # the kth-smallest-m/z feature.
    order_by_mz = sorted(range(len(feats)), key=lambda i: feats[i].precursor_mz)
    keep = [True] * len(feats)

    for pos, i in enumerate(order_by_mz):
        if not keep[i]:
            continue
        fi = feats[i]
        mz_i = fi.precursor_mz
        mz_tol = max(mz_i * ppm_tolerance * 1e-6, 0.005)
        for pos2 in range(pos + 1, len(order_by_mz)):
            j = order_by_mz[pos2]
            fj = feats[j]
            if fj.precursor_mz - mz_i > mz_tol:
                break
            if not keep[j]:
                continue
            if abs(fj.rt_apex - fi.rt_apex) > rt_tolerance:
                continue
            # Same physical peak — keep the brighter one.
            hi = fi.ms1_height or 0.0
            hj = fj.ms1_height or 0.0
            if hj > hi:
                keep[i] = False
                break  # i is gone; nothing more to compare from i
            else:
                keep[j] = False

    # Preserve the original order (not the m/z-sorted order).
    return [f for f, k in zip(feats, keep) if k]
