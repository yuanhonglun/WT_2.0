"""Extracted Ion Chromatogram (EIC) construction from raw ASFAM data.

Optimized: uses numpy vectorized operations and searchsorted for fast
m/z bin assignment instead of per-bin Python loops.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace as _dc_replace
from typing import TYPE_CHECKING, Optional

import numpy as np

from asfam.core.ms2_scan_adapter import ms2_channel_scans
from asfam.models import RawSegmentData, ProductIonEIC

# Reusable two-phase ion-list merger lives in metabo_core now so the DDA
# MS2 cleanup path can call the same algorithm. Re-exported here so
# legacy ``from asfam.core.eic import merge_close_ions`` callers keep
# working without changes.
from metabo_core.algorithms.msdial_ms1_features import build_slice_eics_sum
from metabo_core.algorithms.peak_merge import merge_close_ions  # noqa: F401

if TYPE_CHECKING:
    from asfam.config import ProcessingConfig


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


def extract_product_ion_eics_massslice(
    raw_data: RawSegmentData,
    channel: int,
    config: "ProcessingConfig",
) -> list[ProductIonEIC]:
    """Build MS2 product-ion EICs with the mass-slice strategy.

    Fixed-width 0.1 Da SUM slices (reusing the MS1 ``build_slice_eics_sum``
    kernel) plus per-scan basePeakMz. The slice m/z upper bound is capped at
    ``channel + 2`` (D4: covers the precursor +2 isotope). A minimum-nonzero-scan
    gate is kept because MS2 EICs are mostly zeros. This is pure EIC construction
    only -- peak finding and dedup happen in Stage 1 (Option A). Returns a
    ``list[ProductIonEIC]`` (the old ``all_eics`` contract: overlapping slices are
    not deduped here, and weak / peakless EICs are kept for recall).
    """
    scans = ms2_channel_scans(raw_data, channel)
    if not scans:
        return []
    # D1: 0.1 Da slice width comes from config.msdial_peak; D4: cap upper bound at channel + 2.
    slice_cfg = _dc_replace(config.msdial_peak, mass_range_end=float(channel) + 2.0)
    slices = build_slice_eics_sum(scans, slice_cfg)
    if not slices:
        return []

    n_cycles = raw_data.n_cycles
    rt_array = raw_data.rt_array
    # MS2 EIC quality gate: minimum number of nonzero scans (MS2 EICs are mostly
    # zeros). Reuses the ms2_peak.min_data_points knob (=3) as the nonzero-scan
    # floor, matching the old extractor's min_nonzero_scans default.
    min_nonzero = config.ms2_peak.min_data_points
    eics = []
    for center, basepeak_sparse, eic_sparse, scan_idx in slices:
        dense_eic = np.zeros(n_cycles, dtype=np.float64)
        dense_eic[scan_idx] = eic_sparse
        dense_bpmz = np.full(n_cycles, center, dtype=np.float64)
        dense_bpmz[scan_idx] = basepeak_sparse
        if np.count_nonzero(dense_eic) < min_nonzero:
            continue
        # Temporary product_mz = basePeakMz at the EIC maximum; Stage 1 overwrites
        # it with the basePeakMz at each detected peak's apex.
        max_idx = int(np.argmax(dense_eic))
        eics.append(ProductIonEIC(
            precursor_mz_nominal=channel,
            product_mz=float(dense_bpmz[max_idx]),
            rt_array=rt_array,
            intensity_array=dense_eic,
            basepeak_mz=dense_bpmz,
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


# ---------------------------------------------------------------------------
# Random-access SUM chromatograms (gap filling, EIC spill)
# ---------------------------------------------------------------------------

class SegmentEicIndex:
    """m/z-sorted centroid index over one segment: SUM chromatograms in O(log n).

    ``extract_ms1_eic`` walks every cycle in Python and costs ~5 ms per call.
    Gap filling and the EIC spill want one chromatogram per (spot, sample) plus
    the representative's fragments — ~50k calls on the 3-sample benchmark and
    ~1M on a 6-sample production run — so the per-call cost has to be
    microseconds, not milliseconds. Sorting each spectrum type's centroids by
    m/z once turns every extraction into two ``searchsorted`` calls and a
    ``bincount``: measured 0.008 ms/call against 5.18 ms, for a 23 ms / 5.2 MiB
    build per segment.

    SUM, not base peak. That is what ``LcmsGapFiller`` integrates, and it is
    what produced the heights of every feature this index is asked about:
    ``build_slice_eics_sum`` for the MS1-driven and product-ion peaks, and, for
    the MS2-driven MS1 peaks, ``extract_ms1_eic``'s base peak — which coincides
    with the sum over ``ms1_quant_mz +/- 0.01 Da`` because that window holds
    exactly the one centroid the base peak *is*.

    Product-ion indices are built on first use and dropped by an LRU: a segment
    carries ~30 MS2 channels and gap-fill work is walked channel-major, so a
    depth of two is enough to never rebuild one twice in a row.
    """

    __slots__ = ("_segment", "_ms1", "_ms2", "_ms2_maxsize", "_channels")

    def __init__(self, segment: RawSegmentData, ms2_cache_size: int = 2):
        self._segment = segment
        self._ms1 = self._build(
            (c.ms1_mz, c.ms1_intensity) for c in segment.cycles
        )
        self._channels = {int(p) for p in segment.precursor_list}
        self._ms2: "OrderedDict[int, tuple]" = OrderedDict()
        self._ms2_maxsize = max(1, ms2_cache_size)

    @staticmethod
    def _build(spectra) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mz_chunks, int_chunks, cycle_chunks = [], [], []
        for i, pair in enumerate(spectra):
            if pair is None:
                continue
            mz, intensity = pair
            if mz is None or len(mz) == 0:
                continue
            mz_chunks.append(np.asarray(mz, dtype=np.float64))
            int_chunks.append(np.asarray(intensity, dtype=np.float64))
            cycle_chunks.append(np.full(len(mz), i, dtype=np.int32))
        if not mz_chunks:
            return (np.empty(0), np.empty(0), np.empty(0, dtype=np.int32))
        mz = np.concatenate(mz_chunks)
        order = np.argsort(mz, kind="stable")
        return (mz[order],
                np.concatenate(int_chunks)[order],
                np.concatenate(cycle_chunks)[order])

    @property
    def rt_array(self) -> np.ndarray:
        return self._segment.rt_array

    def _channel(self, channel: int):
        cached = self._ms2.get(channel)
        if cached is not None:
            self._ms2.move_to_end(channel)
            return cached
        built = self._build(
            c.ms2_scans.get(channel) for c in self._segment.cycles
        )
        self._ms2[channel] = built
        while len(self._ms2) > self._ms2_maxsize:
            self._ms2.popitem(last=False)
        return built

    def _eic_sum(self, index, target_mz: float, tolerance: float) -> np.ndarray:
        mz, intensity, cycle = index
        n_cycles = self._segment.n_cycles
        lo = int(np.searchsorted(mz, target_mz - tolerance, side="left"))
        hi = int(np.searchsorted(mz, target_mz + tolerance, side="right"))
        if hi <= lo:
            return np.zeros(n_cycles, dtype=np.float64)
        return np.bincount(cycle[lo:hi], weights=intensity[lo:hi],
                           minlength=n_cycles).astype(np.float64, copy=False)

    def ms1_eic_sum(self, target_mz: float, tolerance: float) -> np.ndarray:
        """Summed MS1 intensity within ``+/-tolerance``, one value per cycle."""
        return self._eic_sum(self._ms1, target_mz, tolerance)

    def product_eic_sum(
        self, channel: int, target_mz: float, tolerance: float,
    ) -> Optional[np.ndarray]:
        """Same for one product-ion channel; ``None`` if it was never acquired."""
        if int(channel) not in self._channels:
            return None
        return self._eic_sum(self._channel(int(channel)), target_mz, tolerance)


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
