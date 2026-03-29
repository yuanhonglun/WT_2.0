"""Extracted Ion Chromatogram (EIC) construction from raw ASFAM data.

Optimized: uses numpy vectorized operations and searchsorted for fast
m/z bin assignment instead of per-bin Python loops.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from asfam.models import RawSegmentData, ProductIonEIC


# ---------------------------------------------------------------------------
# Product ion EIC extraction (MS2 channels) — OPTIMIZED
# ---------------------------------------------------------------------------

def extract_product_ion_eics(
    raw_data: RawSegmentData,
    precursor_mz_nominal: int,
    mz_tolerance: float = 0.02,
    min_nonzero_scans: int = 3,
    min_intensity: float = 10.0,
) -> list[ProductIonEIC]:
    """Extract EICs for all product ions in one MRM-HR channel.

    Optimized algorithm:
    1. Quick signal check: skip channel if total signal too low.
    2. Single-pass m/z binning from all cycles (adaptive tolerance for high mz).
    3. Vectorized EIC matrix construction using searchsorted.
    """
    n_cycles = raw_data.n_cycles
    rt_array = raw_data.rt_array

    # Quick check: does this channel have data?
    first_valid = None
    for cycle in raw_data.cycles:
        if precursor_mz_nominal in cycle.ms2_scans:
            first_valid = cycle
            break
    if first_valid is None:
        return []

    # Phase 1: Collect all m/z values for binning (single pass, numpy)
    all_mz_list = []
    all_int_list = []
    for cycle in raw_data.cycles:
        if precursor_mz_nominal not in cycle.ms2_scans:
            continue
        prod_mz, prod_int = cycle.ms2_scans[precursor_mz_nominal]
        if len(prod_mz) == 0:
            continue
        mask = prod_int >= min_intensity
        if np.any(mask):
            all_mz_list.append(prod_mz[mask])
            all_int_list.append(prod_int[mask])

    if not all_mz_list:
        return []

    all_mz = np.concatenate(all_mz_list)
    all_int = np.concatenate(all_int_list)

    # Phase 2: Adaptive m/z binning - wider tolerance at higher m/z
    # Use max(fixed_tolerance, ppm_based) to handle both low and high mz
    # 100 ppm handles QTOF resolution degradation at high mz
    adaptive_tol = max(mz_tolerance, float(precursor_mz_nominal) * 100e-6)
    mz_bins = _fast_tolerance_binning(all_mz, all_int, adaptive_tol)
    if len(mz_bins) == 0:
        return []

    # Phase 3: Build EIC matrix using searchsorted (vectorized)
    n_bins = len(mz_bins)
    eic_matrix = np.zeros((n_bins, n_cycles), dtype=np.float64)
    bin_arr = np.array(mz_bins, dtype=np.float64)

    for i, cycle in enumerate(raw_data.cycles):
        if precursor_mz_nominal not in cycle.ms2_scans:
            continue
        prod_mz, prod_int = cycle.ms2_scans[precursor_mz_nominal]
        if len(prod_mz) == 0:
            continue

        indices = np.searchsorted(bin_arr, prod_mz)
        indices = np.clip(indices, 0, n_bins - 1)

        for alt in [indices, np.clip(indices - 1, 0, n_bins - 1)]:
            diffs = np.abs(prod_mz - bin_arr[alt])
            valid = diffs <= adaptive_tol
            for j in np.where(valid)[0]:
                bi = alt[j]
                if prod_int[j] > eic_matrix[bi, i]:
                    eic_matrix[bi, i] = prod_int[j]

    # Phase 4: Filter EICs by minimum nonzero scans
    nonzero_counts = np.count_nonzero(eic_matrix, axis=1)
    valid_mask = nonzero_counts >= min_nonzero_scans

    eics = []
    for idx in np.where(valid_mask)[0]:
        eics.append(ProductIonEIC(
            precursor_mz_nominal=precursor_mz_nominal,
            product_mz=float(bin_arr[idx]),
            rt_array=rt_array,
            intensity_array=eic_matrix[idx],
        ))

    return eics


def merge_close_ions(
    mz: np.ndarray,
    intensity: np.ndarray,
    precursor_mz_nominal: int,
    base_tolerance: float = 0.02,
    extra_arrays: list[np.ndarray] | None = None,
) -> tuple:
    """Two-phase merge of near-duplicate product ions.

    Phase 1 – Adaptive-tolerance merge:
        Groups consecutive ions within max(base_tolerance, precursor × 100 ppm).
        Merged m/z = intensity-weighted mean (consistent with MS-DIAL centroid
        approach).  Intensity = group maximum.

    Phase 2 – Identical-response merge:
        After Phase 1, adjacent ions with *exactly* the same intensity that are
        within 3× the adaptive tolerance are merged.  Identical intensity is an
        extremely strong signal that two entries originate from the same physical
        ion (distinct fragments virtually never share the exact same peak
        height), so the wider window is safe.
        Merged m/z = simple mean (equal weights);  intensity unchanged.

    Returns (mz, intensity[, *extra_arrays]) with duplicates merged.
    """
    if len(mz) <= 1:
        if extra_arrays:
            return (mz, intensity) + tuple(extra_arrays)
        return mz, intensity

    tolerance = max(base_tolerance, float(precursor_mz_nominal) * 100e-6)

    # ------------------------------------------------------------------
    # Phase 1: adaptive-tolerance merge
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Phase 2: identical-response merge
    # Adjacent ions with the exact same intensity within a wider window
    # (3× adaptive tolerance) are almost certainly the same physical ion
    # split by EIC-binning boundary effects.
    # ------------------------------------------------------------------
    ident_limit = tolerance * 3

    final_mz: list[float] = []
    final_int: list[float] = []
    final_extras: list[list] = [[] for _ in sorted_extras]

    i = 0
    n2 = len(p1_mz)
    while i < n2:
        grp_mz = [p1_mz[i]]
        grp_int = p1_int[i]
        grp_first = i           # keep extra-arrays from first member

        j = i + 1
        while j < n2:
            if p1_int[j] == grp_int and p1_mz[j] - grp_mz[-1] <= ident_limit:
                grp_mz.append(p1_mz[j])
                j += 1
            else:
                break

        # m/z: simple mean when intensities are equal (reduces to
        # intensity-weighted mean), matching MS-DIAL centroid logic.
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


def _fast_tolerance_binning(
    mz_array: np.ndarray,
    int_array: np.ndarray,
    tolerance: float,
) -> list[float]:
    """Fast m/z binning using sorted arrays.

    Groups consecutive m/z values within tolerance, returns
    intensity-weighted mean for each bin.
    """
    order = np.argsort(mz_array)
    sorted_mz = mz_array[order]
    sorted_int = int_array[order]

    bins = []
    i = 0
    n = len(sorted_mz)

    while i < n:
        # Start a new bin
        bin_start = i
        weighted_sum = sorted_mz[i] * sorted_int[i]
        weight_total = sorted_int[i]

        i += 1
        while i < n:
            # Check against intensity-weighted mean so far
            current_mean = weighted_sum / max(weight_total, 1e-12)
            if sorted_mz[i] - current_mean > tolerance:
                break
            weighted_sum += sorted_mz[i] * sorted_int[i]
            weight_total += sorted_int[i]
            i += 1

        bins.append(weighted_sum / max(weight_total, 1e-12))

    return bins


# ---------------------------------------------------------------------------
# MS1 EIC extraction
# ---------------------------------------------------------------------------

def extract_ms1_eic(
    raw_data: RawSegmentData,
    target_mz: float,
    mz_tolerance: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract MS1 EIC for a specific m/z across all cycles."""
    n_cycles = raw_data.n_cycles
    rt_array = raw_data.rt_array
    intensities = np.zeros(n_cycles, dtype=np.float64)

    for i, cycle in enumerate(raw_data.cycles):
        ms1_mz = cycle.ms1_mz
        ms1_int = cycle.ms1_intensity
        if ms1_mz is None or len(ms1_mz) == 0:
            continue
        mask = np.abs(ms1_mz - target_mz) <= mz_tolerance
        if np.any(mask):
            intensities[i] = float(np.max(ms1_int[mask]))

    return rt_array, intensities


def extract_ms1_precise_mz(
    raw_data: RawSegmentData,
    cycle_index: int,
    mz_low: float,
    mz_high: float,
) -> Optional[float]:
    """Get intensity-weighted centroid m/z from MS1 spectrum at a specific cycle."""
    cycle = raw_data.cycles[cycle_index]
    ms1_mz = cycle.ms1_mz
    ms1_int = cycle.ms1_intensity

    if ms1_mz is None or len(ms1_mz) == 0:
        return None

    mask = (ms1_mz >= mz_low) & (ms1_mz <= mz_high)
    if not np.any(mask):
        return None

    local_mz = ms1_mz[mask]
    local_int = ms1_int[mask]
    total_int = np.sum(local_int)
    if total_int <= 0:
        return None

    return float(np.sum(local_mz * local_int) / total_int)
