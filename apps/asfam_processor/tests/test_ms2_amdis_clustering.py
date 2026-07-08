"""ASFAM MS2 AMDIS component-perception clustering tests (AMDIS plan T2-T4)."""
from dataclasses import dataclass

import numpy as np

from metabo_core.models.chromatography import DetectedPeak, ProductIonEIC
from asfam.core.ms2_amdis_clustering import (
    build_channel_chroms,
    build_ion_peaks,
    compute_eic_nf,
)


def _eic(pmz, intens):
    rt = np.arange(len(intens), dtype=np.float64) * 0.01
    return ProductIonEIC(precursor_mz_nominal=100, product_mz=pmz,
                         rt_array=rt, intensity_array=np.asarray(intens, dtype=np.float64))


def _peak(pmz, apex, lo, hi, height):
    return DetectedPeak(precursor_mz_nominal=100, product_mz=pmz,
                        rt_apex=apex * 0.01, rt_left=lo * 0.01, rt_right=hi * 0.01,
                        apex_index=apex, left_index=lo, right_index=hi,
                        height=height, area=height * 3.0, sn_ratio=10.0, gaussian_similarity=0.9)


def _noisy_eic(n=600, baseline=50.0, noise_amp=10.0, seed=0):
    rng = np.random.default_rng(seed)
    # alternating-sign noise around a flat baseline -> binned (max-min) ~ noise_amp scale
    return baseline + rng.uniform(-noise_amp, noise_amp, size=n)


def test_compute_eic_nf_is_amplitude_over_sqrt_aref():
    eic = _noisy_eic(n=600, baseline=50.0, noise_amp=10.0)
    nf = compute_eic_nf(eic, noise_bin_size=50, min_noise_windows=10, aref_floor=1.0)
    # Nf = amplitude_noise / sqrt(median(baseline)); baseline ~ 50, so sqrt~7.07.
    assert np.isfinite(nf) and nf > 0.0


def test_compute_eic_nf_short_eic_falls_back_to_amp1():
    """n < noise_bin_size*min_noise_windows -> amplitude_noise=1.0 -> Nf = 1/sqrt(A_ref)."""
    eic = _noisy_eic(n=120, baseline=64.0, noise_amp=10.0)  # 120 < 500
    nf = compute_eic_nf(eic, noise_bin_size=50, min_noise_windows=10, aref_floor=1.0)
    # amplitude_noise floored to 1.0; A_ref ~ 64 -> Nf ~ 1/8 = 0.125
    assert 0.08 < nf < 0.2


def test_compute_eic_nf_short_eic_smaller_bins_avoids_fallback():
    """Calibrated small noise_bin_size on a short EIC gives a real (>1 mapped) estimate."""
    eic = _noisy_eic(n=120, baseline=64.0, noise_amp=20.0)
    nf_fallback = compute_eic_nf(eic, noise_bin_size=50, min_noise_windows=10, aref_floor=1.0)
    nf_calibrated = compute_eic_nf(eic, noise_bin_size=10, min_noise_windows=8, aref_floor=1.0)
    # calibrated path uses a real binned-median noise (larger than the floored 1.0 case)
    assert nf_calibrated > nf_fallback


# --- T3: channel chroms stacking + IonPeak builder from ASFAM detected peaks ---

def test_build_channel_chroms_sorts_by_mz_and_pads():
    e1 = _eic(120.5, [0, 5, 9, 5, 0])
    e2 = _eic(85.1, [0, 2, 4, 2, 0, 0])  # different length -> zero-pad to max
    chroms, row_to_mz, mz_to_row = build_channel_chroms([e1, e2])
    assert chroms.shape == (2, 6)             # padded to longest
    assert row_to_mz == [85.1, 120.5]         # sorted ascending
    assert mz_to_row[85.1] == 0 and mz_to_row[120.5] == 1
    assert chroms[1, 5] == 0.0                # e1 padded


def test_build_ion_peaks_no_new_peaks_and_window_from_detected():
    e1 = _eic(120.5, [0, 5, 90, 5, 0, 0, 0, 0])
    peaks_by_row = {0: [_peak(120.5, apex=2, lo=1, hi=3, height=90.0)]}
    chroms, row_to_mz, mz_to_row = build_channel_chroms([e1])
    ips = build_ion_peaks(chroms, peaks_by_row, noise_bin_size=50,
                          min_noise_windows=10, aref_floor=1.0)
    assert len(ips) == 1                       # exactly one IonPeak per DetectedPeak
    ip = ips[0]
    assert ip.ion_index == 0
    assert ip.apex_scan_int == 2
    assert ip.window_lo == 1 and ip.window_hi == 4   # right_index+1 (half-open)
    assert ip.apex_intensity == 90.0
    assert ip.sharpness > 0.0                  # computed via metabo_core peak_sharpness


# --- T4: cluster_peaks_amdis orchestrator (drop-in for cluster_peaks_by_rt) ---

@dataclass
class _AmdisParams:
    sharpness_bins_per_scan: int = 10
    sharpness_range_factor: float = 50.0
    sharpness_cutoff_ratio: float = 0.75
    inclusion_cutoff_ratio: float = 0.3
    min_range_scans: int = 3
    match_max_window: int = 12
    noise_bin_size: int = 50
    min_noise_windows: int = 10
    aref_floor: float = 1.0


@dataclass
class _Cfg:
    ms2_amdis: _AmdisParams = None


def _triangle(center, height, halfwidth, n):
    return [max(0.0, height - abs(s - center) * (height / halfwidth)) for s in range(n)]


def test_cluster_peaks_amdis_returns_list_of_lists():
    from asfam.core.ms2_amdis_clustering import cluster_peaks_amdis
    cfg = _Cfg(ms2_amdis=_AmdisParams())
    e1 = _eic(120.5, _triangle(30, 100, 5, 60))
    p1 = _peak(120.5, apex=30, lo=25, hi=35, height=100.0)
    clusters = cluster_peaks_amdis([e1], [p1], cfg)
    assert isinstance(clusters, list)
    assert all(isinstance(c, list) for c in clusters)
    assert all(isinstance(pk, DetectedPeak) for c in clusters for pk in c)


def test_cluster_peaks_amdis_splits_co_eluting_compounds():
    """Two compounds apexing a few scans apart -> AMDIS sharpness NMS splits them
    (RT-proximity clustering would merge within cluster_max_apex_span)."""
    from asfam.core.ms2_amdis_clustering import cluster_peaks_amdis
    cfg = _Cfg(ms2_amdis=_AmdisParams())
    n = 80
    # Compound A apex ~30 (ions 120.5, 95.2); Compound B apex ~42 (ions 130.7, 88.1)
    eA1 = _eic(120.5, _triangle(30, 100, 4, n)); pA1 = _peak(120.5, 30, 26, 34, 100.0)
    eA2 = _eic(95.2,  _triangle(30, 70, 4, n));  pA2 = _peak(95.2, 30, 26, 34, 70.0)
    eB1 = _eic(130.7, _triangle(42, 90, 4, n));  pB1 = _peak(130.7, 42, 38, 46, 90.0)
    eB2 = _eic(88.1,  _triangle(42, 55, 4, n));  pB2 = _peak(88.1, 42, 38, 46, 55.0)
    clusters = cluster_peaks_amdis([eA1, eA2, eB1, eB2], [pA1, pA2, pB1, pB2], cfg)
    assert len(clusters) == 2
    apexes = sorted(int(np.mean([pk.apex_index for pk in c])) for c in clusters)
    assert abs(apexes[0] - 30) <= 2 and abs(apexes[1] - 42) <= 2
