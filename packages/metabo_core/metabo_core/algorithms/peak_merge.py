"""Shared near-duplicate ion-list merger.

Two-phase merge used to collapse near-duplicate product (or precursor)
ions that appear in centroid spectra because of EIC binning boundaries
or sub-ppm m/z drift.

Originally lived in ``asfam.core.eic``; promoted to ``metabo_core`` so
the DDA MS2 cleanup path can reuse the same logic. The algorithm is
unchanged; only the location moved.
"""
from __future__ import annotations

import numpy as np


def merge_close_ions(
    mz: np.ndarray,
    intensity: np.ndarray,
    precursor_mz_nominal: int,
    base_tolerance: float = 0.02,
    extra_arrays: list[np.ndarray] | None = None,
) -> tuple:
    """Two-phase merge of near-duplicate ions.

    Phase 1 — Adaptive-tolerance merge:
        Groups consecutive ions within ``max(base_tolerance,
        precursor_mz_nominal * 100 ppm)``. Within a group, the merged
        m/z is the intensity-weighted mean (matches MS-DIAL's centroid
        convention) and the merged intensity is the group maximum.

    Phase 2 — Identical-response merge:
        After Phase 1, adjacent ions with *exactly* the same intensity
        within 3× the adaptive tolerance are merged. Identical
        intensity is a very strong signal that two entries are the
        same physical ion split by EIC-binning boundary effects.
        Merged m/z is the simple mean (intensities are equal so the
        weighted and unweighted means coincide).

    ``extra_arrays`` are aligned per-ion arrays (e.g. S/N) that get
    carried along: for each merged group the value of the strongest
    member is kept.

    Returns ``(mz, intensity[, *extra_arrays])`` with duplicates merged.
    """
    if len(mz) <= 1:
        if extra_arrays:
            return (mz, intensity) + tuple(extra_arrays)
        return mz, intensity

    tolerance = max(base_tolerance, float(precursor_mz_nominal) * 100e-6)

    order = np.argsort(mz)
    sorted_mz = mz[order]
    sorted_int = intensity[order]
    sorted_extras = [arr[order] for arr in (extra_arrays or [])]

    p1_mz: list[float] = []
    p1_int: list[float] = []
    p1_extras: list[list] = [[] for _ in sorted_extras]

    n = len(sorted_mz)
    i = 0
    while i < n:
        weighted_sum = sorted_mz[i] * sorted_int[i]
        weight_total = sorted_int[i]
        best_int = sorted_int[i]
        best_idx = i

        j = i + 1
        while j < n:
            current_mean = weighted_sum / max(weight_total, 1e-12)
            if sorted_mz[j] - current_mean > tolerance:
                break
            weighted_sum += sorted_mz[j] * sorted_int[j]
            weight_total += sorted_int[j]
            if sorted_int[j] > best_int:
                best_int = sorted_int[j]
                best_idx = j
            j += 1

        p1_mz.append(weighted_sum / max(weight_total, 1e-12))
        p1_int.append(float(best_int))
        for k, arr in enumerate(sorted_extras):
            p1_extras[k].append(arr[best_idx])

        i = j

    ident_limit = tolerance * 3

    final_mz: list[float] = []
    final_int: list[float] = []
    final_extras: list[list] = [[] for _ in sorted_extras]

    i = 0
    n2 = len(p1_mz)
    while i < n2:
        grp_mz = [p1_mz[i]]
        grp_int = p1_int[i]
        grp_first = i

        j = i + 1
        while j < n2:
            if p1_int[j] == grp_int and p1_mz[j] - grp_mz[-1] <= ident_limit:
                grp_mz.append(p1_mz[j])
                j += 1
            else:
                break

        final_mz.append(float(np.mean(grp_mz)))
        final_int.append(grp_int)
        for k in range(len(sorted_extras)):
            final_extras[k].append(p1_extras[k][grp_first])

        i = j

    result_mz = np.array(final_mz, dtype=np.float64)
    result_int = np.array(final_int, dtype=np.float64)
    result = [result_mz, result_int]
    for arr_list in final_extras:
        result.append(np.array(arr_list))
    return tuple(result)
