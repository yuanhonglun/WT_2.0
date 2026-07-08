"""ASFAM MS2 AMDIS component-perception clustering (B1 per-EIC noise factor).

AMDIS contributes ONLY the sharpness-domain component perception. Chromatographic
peak detection stays ASFAM's detect_peaks; the per-ion sharpness for the perception
is computed here from those already-detected peaks, using a per-EIC Nf derived from
ASFAM's existing per-EIC noise (B1): Nf_i = amplitude_noise_i / sqrt(A_ref_i).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from metabo_core.algorithms.baseline import estimate_baseline_and_noise
from metabo_core.gcms.deconvolution import (
    IonPeak,
    parabola_apex,
    peak_sharpness,
    perceive_components,
)


def compute_eic_nf(
    eic_intensity: np.ndarray,
    *,
    noise_bin_size: int,
    min_noise_windows: int,
    aref_floor: float = 1.0,
) -> float:
    """B1 per-EIC AMDIS noise factor: amplitude_noise / sqrt(A_ref).

    A_ref = median of the wide-LWMA baseline (abundance level at which noise was
    measured), floored. Independent of which detector found the peaks.

    NOTE (short-EIC fallback): estimate_baseline_and_noise only yields a binned
    noise when len >= noise_bin_size * min_noise_windows; otherwise amplitude_noise
    falls back to 1.0 and Nf collapses to 1/sqrt(A_ref). Callers must size
    noise_bin_size / min_noise_windows to actual MS2 EIC length (see plan T6).
    """
    bl = estimate_baseline_and_noise(
        eic_intensity,
        smooth_window=1,
        baseline_window=20,
        noise_bin_size=noise_bin_size,
        noise_factor=3.0,
        min_noise_windows=min_noise_windows,
    )
    a_ref = max(float(np.median(bl.baseline)), float(aref_floor))
    return float(bl.amplitude_noise) / float(np.sqrt(a_ref))


def build_channel_chroms(filtered_eics):
    """Stack a channel's product-ion EICs into a dense 2D (n_ions, n_scans) matrix.

    Rows sorted by product_mz ascending. Zero-pads to the longest EIC so all rows
    align on scan index. Returns (chroms, row_to_mz, mz_to_row).
    """
    eics = sorted(filtered_eics, key=lambda e: float(e.product_mz))
    n_ions = len(eics)
    n_scans = max((len(e.intensity_array) for e in eics), default=0)
    chroms = np.zeros((n_ions, n_scans), dtype=np.float64)
    row_to_mz = []
    mz_to_row = {}
    for i, e in enumerate(eics):
        arr = np.asarray(e.intensity_array, dtype=np.float64)
        chroms[i, : arr.size] = arr
        row_to_mz.append(float(e.product_mz))
        mz_to_row[float(e.product_mz)] = i
    return chroms, row_to_mz, mz_to_row


def build_ion_peaks(chroms, peaks_by_row, *, noise_bin_size, min_noise_windows, aref_floor=1.0):
    """Wrap each ASFAM DetectedPeak as an AMDIS IonPeak (sharpness via B1 Nf_i).

    NO peak detection happens here: apex/window come from the DetectedPeak; sharpness
    is AMDIS eq.(2) on the existing apex; one IonPeak per input DetectedPeak.
    """
    ion_peaks = []
    for row, peaks in peaks_by_row.items():
        if not peaks:
            continue
        eic_i = chroms[row]
        nf_i = compute_eic_nf(
            eic_i, noise_bin_size=noise_bin_size,
            min_noise_windows=min_noise_windows, aref_floor=aref_floor,
        )
        for p in peaks:
            lo = int(p.left_index)
            hi = int(p.right_index) + 1   # AMDIS half-open window
            ion_peaks.append(IonPeak(
                ion_index=int(row),
                apex_scan_int=int(p.apex_index),
                apex_scan_precise=float(parabola_apex(eic_i, int(p.apex_index))),
                apex_intensity=float(p.height),
                sharpness=float(peak_sharpness(eic_i, int(p.apex_index), lo, hi, nf_i)),
                window_lo=lo,
                window_hi=hi,
                baseline=np.zeros(0, dtype=np.float64),
            ))
    return ion_peaks


def cluster_peaks_amdis(filtered_eics, all_peaks, config):
    """AMDIS component-perception clustering — drop-in replacement for
    cluster_peaks_by_rt. Returns list[list[DetectedPeak]] (one list per component).
    Peak detection is upstream (ASFAM); this only groups co-eluting peaks.
    """
    if not filtered_eics or not all_peaks:
        return []
    amd = config.ms2_amdis

    chroms, row_to_mz, mz_to_row = build_channel_chroms(filtered_eics)
    if chroms.size == 0:
        return []

    # Group detected peaks by their source EIC row (matched via product_mz).
    peaks_by_row = defaultdict(list)
    for p in all_peaks:
        row = mz_to_row.get(float(p.product_mz))
        if row is not None:
            peaks_by_row[row].append(p)
    if not peaks_by_row:
        return []

    ion_peaks = build_ion_peaks(
        chroms, peaks_by_row,
        noise_bin_size=amd.noise_bin_size,
        min_noise_windows=amd.min_noise_windows,
        aref_floor=amd.aref_floor,
    )
    if not ion_peaks:
        return []

    components = perceive_components(
        chroms,
        external_ion_peaks=ion_peaks,
        use_tic_path=False,
        sharpness_bins_per_scan=amd.sharpness_bins_per_scan,
        sharpness_range_factor=amd.sharpness_range_factor,
        sharpness_cutoff_ratio=amd.sharpness_cutoff_ratio,
        min_range_scans=amd.min_range_scans,
        inclusion_cutoff_ratio=amd.inclusion_cutoff_ratio,
    )

    clusters = []
    for comp in components:
        if comp.sharpness > 0:
            raw = int(round(amd.sharpness_range_factor / max(comp.sharpness, 1e-9)))
        else:
            raw = 5
        range_scans = max(5, min(raw, int(amd.match_max_window)))

        by_mz = {}  # round(mz*200) -> DetectedPeak (intra-cluster m/z dedup, keep tallest)
        for ion_idx in comp.contributing_ions:
            cand = peaks_by_row.get(ion_idx, [])
            if not cand:
                continue
            best = min(cand, key=lambda pk: abs(int(pk.apex_index) - int(comp.apex_scan_int)))
            if abs(int(best.apex_index) - int(comp.apex_scan_int)) > range_scans:
                continue
            key = round(float(best.product_mz) * 200)
            if key not in by_mz or best.height > by_mz[key].height:
                by_mz[key] = best
        cluster = list(by_mz.values())
        if cluster:
            clusters.append(cluster)
    return clusters
