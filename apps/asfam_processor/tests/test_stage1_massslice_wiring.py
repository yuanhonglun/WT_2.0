import numpy as np
from asfam.models import ScanCycle, RawSegmentData
from asfam.config import ProcessingConfig
from asfam.pipeline.stage1_ms2_detection import run_stage1, _dedup_peaks_global
from metabo_core.models.chromatography import DetectedPeak


def _seg(channel=285, n=24):
    # A gaussian-like product ion 100.02 (rising/falling with RT) plus a
    # co-eluting 150.07 (0.6x) make up one MS2 spectrum. Amplitudes are kept
    # far above clean_ms2_spectrum's absolute floor msms_intensity_threshold
    # (default 200 since the T2 fix); otherwise both fragments get stripped below
    # min_fragments_per_feature (2) and the feature is rejected -> false failure.
    cyc = []
    amp = [0, 0, 0, 400, 1000, 2500, 5000, 8000, 9000, 8000, 5000, 2500, 1000, 400,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0][:n]
    for i in range(n):
        scans = {}
        if amp[i] > 0:
            scans[channel] = (np.array([100.02, 150.07]),
                              np.array([float(amp[i]), float(amp[i]) * 0.6]))
        cyc.append(ScanCycle(i, 1.0 + 0.05 * i, np.array([]), np.array([]), scans))
    return RawSegmentData("x", "seg", channel, channel + 29, 1, n,
                          np.array([1.0 + 0.05 * i for i in range(n)]),
                          [channel], cyc)


def test_stage1_massslice_no_duplicate_fragments_and_basepeak_mz():
    cfg = ProcessingConfig()
    by_rep = run_stage1({1: [_seg()]}, cfg)
    feats = by_rep[1]
    assert feats, "expected >=1 MS2-driven feature"
    f = max(feats, key=lambda x: x.n_fragments)
    # 50% overlap must not make 100.02 appear twice (Option-A dedup works)
    near = [m for m in f.ms2_mz if abs(m - 100.02) < 0.05]
    assert len(near) == 1, f"duplicate 100.02 fragments: {f.ms2_mz}"
    # product m/z comes from basePeakMz@apex (~injected value, no refine needed)
    assert min(abs(m - 100.02) for m in f.ms2_mz) < 0.02
    assert min(abs(m - 150.07) for m in f.ms2_mz) < 0.02
    # clean_ms2_spectrum is now the sole sorter (refine + resort deleted in T4);
    # output must stay m/z-ascending.
    assert np.all(np.diff(f.ms2_mz) >= 0), f"ms2_mz not sorted: {f.ms2_mz}"


def test_stage1_basepeak_mz_tracks_apex_scan():
    # The fragment's measured centroid is 100.00 on the peak flanks but 100.05
    # exactly at the apex scan. The emitted product_mz must come from the APEX
    # scan's basePeakMz (~100.05), proving the relabel reads
    # eic.basepeak_mz[apex_index] rather than a flank/global value.
    ch = 285
    amp = [0, 0, 0, 1000, 3000, 6000, 9000, 6000, 3000, 1000, 0, 0]
    apex_i = 6
    cyc = []
    for i, a in enumerate(amp):
        scans = {}
        if a > 0:
            mz = 100.05 if i == apex_i else 100.00
            scans[ch] = (np.array([mz, 150.07]),
                         np.array([float(a), float(a) * 0.6]))
        cyc.append(ScanCycle(i, 1.0 + 0.05 * i, np.array([]), np.array([]), scans))
    raw = RawSegmentData("x", "seg", ch, ch + 29, 1, len(amp),
                         np.array([1.0 + 0.05 * i for i in range(len(amp))]),
                         [ch], cyc)
    feats = run_stage1({1: [raw]}, ProcessingConfig())[1]
    assert feats, "expected >=1 MS2-driven feature"
    f = max(feats, key=lambda x: x.n_fragments)
    near = [m for m in f.ms2_mz if abs(m - 100.0) < 0.1]
    assert near, f"no ~100 fragment in {f.ms2_mz}"
    # apex-scan m/z is 100.05; must not be the flank value 100.00
    assert min(abs(m - 100.05) for m in near) < 0.02, f"product_mz not from apex scan: {near}"


# --- _dedup_peaks_global (Option-A global near-duplicate dedup) unit tests ---

def _pk(product_mz, rt_apex, height, apex_index=0):
    return DetectedPeak(
        precursor_mz_nominal=285, product_mz=product_mz, rt_apex=rt_apex,
        rt_left=rt_apex - 0.1, rt_right=rt_apex + 0.1, apex_index=apex_index,
        left_index=max(0, apex_index - 1), right_index=apex_index + 1,
        height=height, area=height,
    )


def test_dedup_collapses_overlapping_same_ion_keep_taller():
    out = _dedup_peaks_global([_pk(100.02, 5.0, 800.0), _pk(100.0205, 5.0, 1200.0)],
                              mass_tol=0.005, rt_tol=0.03)
    assert len(out) == 1 and out[0].height == 1200.0


def test_dedup_keeps_distinct_rt_same_mz():
    out = _dedup_peaks_global([_pk(100.02, 5.0, 800.0), _pk(100.02, 6.0, 900.0)],
                              mass_tol=0.005, rt_tol=0.03)
    assert len(out) == 2


def test_dedup_keeps_distinct_mz_same_rt():
    out = _dedup_peaks_global([_pk(100.02, 5.0, 800.0), _pk(100.20, 5.0, 900.0)],
                              mass_tol=0.005, rt_tol=0.03)
    assert len(out) == 2


def test_dedup_three_peak_chain_keeps_tallest():
    out = _dedup_peaks_global(
        [_pk(100.020, 5.0, 500.0), _pk(100.022, 5.0, 1500.0), _pk(100.024, 5.0, 900.0)],
        mass_tol=0.005, rt_tol=0.03)
    assert len(out) == 1 and out[0].height == 1500.0


def test_dedup_equal_height_keeps_later_mz():
    # tie: (target - searched) > 0 is False -> exclude the earlier (lower-mz) one.
    out = _dedup_peaks_global([_pk(100.020, 5.0, 1000.0), _pk(100.022, 5.0, 1000.0)],
                              mass_tol=0.005, rt_tol=0.03)
    assert len(out) == 1
    assert abs(out[0].product_mz - 100.022) < 1e-9


def test_dedup_rt_boundary_keeps_both():
    # rt difference (0.04) exceeds rt_tol (0.03) -> distinct co-eluting peaks kept.
    out = _dedup_peaks_global([_pk(100.02, 5.00, 800.0), _pk(100.02, 5.04, 900.0)],
                              mass_tol=0.005, rt_tol=0.03)
    assert len(out) == 2


def test_dedup_empty():
    assert _dedup_peaks_global([], 0.005, 0.03) == []
