"""Per-feature aggregation of per-peak shape metrics.

Each chromatographic peak that contributes to a feature has its own
``DetectedPeak.gaussian_similarity`` score (computed in
``peak_detection._gaussian_similarity``). When several peaks are merged
into one feature (e.g. an ASFAM feature built from several co-eluting
product-ion peaks, or a GC-MS feature built from several m/z bins),
the per-feature shape quality is the **minimum** of its constituents:
a feature is only as shape-confident as its worst contributing peak.

The same helper is used by all three apps (ASFAM, DDA, GC-MS) so that
``Feature.gaussian_similarity`` has a uniform meaning across modes —
only the underlying parameter values (gates, tolerances) differ.
"""
from __future__ import annotations

import math
from typing import Iterable, Union

import numpy as np

Number = Union[int, float, np.floating, np.integer]


def aggregate_feature_gaussian(*sources: Union[Number, Iterable[Number], None]) -> float:
    """Aggregate per-peak gaussian similarities into a per-feature score.

    Accepts any mix of scalars, arrays, lists, or None. ``None`` /
    NaN / non-positive values are ignored (they represent peaks where
    gaussian was not computed or the trace was degenerate). Returns the
    **minimum** of all valid values, or ``0.0`` if no valid values are
    supplied.

    Examples
    --------
    >>> aggregate_feature_gaussian(0.95, 0.82, 0.97)
    0.82
    >>> aggregate_feature_gaussian([0.9, 0.85], 0.7)
    0.7
    >>> aggregate_feature_gaussian(None, [0.0, 0.0])
    0.0
    """
    flat: list[float] = []
    for src in sources:
        if src is None:
            continue
        if isinstance(src, np.ndarray):
            flat.extend(float(v) for v in src.tolist())
        elif isinstance(src, (list, tuple)):
            flat.extend(float(v) for v in src)
        else:
            flat.append(float(src))
    valid = [v for v in flat if v > 0.0 and not math.isnan(v)]
    return min(valid) if valid else 0.0
