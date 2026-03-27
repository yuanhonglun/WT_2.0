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
