"""T2 fix — ms2_driven (stage1) MS2 cleanup intensity floor aligned to the detector.

Fragments reaching ``clean_ms2_spectrum`` in stage1 have already passed the full
MS-DIAL 3-gate detector (min_amplitude=200 + gaussian>=0.85 + S/N + prominence)
AND the cross-EIC shape gate (>=0.7). The legacy absolute floor of 1000 only
re-cut those validated real peaks that MS-DIAL keeps (diagnosis 2026-07-02,
spec §2.3/§2.6). These tests pin: (1) the config defaults after the fix, (2) that
the [200,1000) band survives the new floor, and (3) that ions below the detection
floor are still dropped (the floor is not turned off, only aligned to 200).
"""
import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import ScanCycle, RawSegmentData
from asfam.pipeline.stage1_ms2_detection import run_stage1
from metabo_core.algorithms.ms2_cleanup import MS2CleanupConfig, clean_ms2_spectrum


def test_config_defaults_ms2_driven_floor():
    cfg = ProcessingConfig()
    # ms2_driven absolute floor lowered to the detection amplitude gate.
    assert cfg.msms_intensity_threshold == 200.0
    # dedicated ms2_driven relative floor, below stage1b's msms_relative_threshold.
    assert cfg.ms2_driven_rel_floor == 0.01
    # stage1b (T1/MSDec) relative floor MUST stay untouched (decoupled from T2).
    assert cfg.msms_relative_threshold == 0.02


def test_cleanup_keeps_200_1000_band_under_ms2_driven_floor():
    """[200,1000) real fragments must survive the new floor (INOSINE 119@897)."""
    cfg = ProcessingConfig()
    mz = np.array([110.067, 119.031, 137.053], dtype=np.float64)  # 119@897 < 1000
    it = np.array([1176.0, 897.0, 22167.0], dtype=np.float64)
    cc = MS2CleanupConfig(
        merge_absolute_tol=cfg.eic_mz_tolerance,
        absolute_intensity_threshold=cfg.msms_intensity_threshold,   # 200
        relative_intensity_threshold=cfg.ms2_driven_rel_floor,       # 0.01 -> cut 221.67
        remove_after_precursor=False,
    )
    out_mz, out_int = clean_ms2_spectrum(mz, it, precursor_mz=269.0, config=cc)
    # the old floor of 1000 would have dropped 119@897; the aligned floor keeps it.
    assert 119.031 in [round(float(m), 3) for m in out_mz]
    assert len(out_mz) == 3


def test_cleanup_still_drops_below_detection_floor():
    """Ions < the detection floor (200) are still cut — the floor is aligned, not removed."""
    cfg = ProcessingConfig()
    mz = np.array([60.0, 100.0, 200.0], dtype=np.float64)
    it = np.array([150.0, 500.0, 30000.0], dtype=np.float64)   # 60@150 < 200
    cc = MS2CleanupConfig(
        merge_absolute_tol=cfg.eic_mz_tolerance,
        absolute_intensity_threshold=cfg.msms_intensity_threshold,   # 200
        relative_intensity_threshold=cfg.ms2_driven_rel_floor,       # 0.01 -> cut 300
        remove_after_precursor=False,
    )
    out_mz, out_int = clean_ms2_spectrum(mz, it, precursor_mz=300.0, config=cc)
    # cut = max(200, 0.01*30000=300) = 300 -> 60(150) and 100(500)? 500>300 kept; 150<300 cut.
    assert 60.0 not in [round(float(m), 3) for m in out_mz]
    assert 200.0 in [round(float(m), 3) for m in out_mz]


# --- Option B: decouple the feature-ADMISSION guard from the fragment floor.
# The cleanup absolute floor stays at 200 (retain [200,1000) fragments = primary
# fix), but a whole ms2_driven cluster is admitted only if its brightest fragment
# reaches ms2_driven_feature_floor (1000). This keeps the fragment-recovery win
# while suppressing the 3x weak-feature explosion (validation 2026-07-02 §2/§4).


def _seg_scaled(channel, base_a, base_b_ratio=0.6, n=24):
    """A gaussian-like co-eluting pair (ions 100.02 + 150.07) whose apex heights
    scale to ``base_a`` and ``base_a*base_b_ratio``. Shape matches the passing
    massslice wiring fixture so detection + shape gate pass; only the amplitude
    scale changes, letting us place the base peak above/below the admission floor.
    """
    shape = [0, 0, 0, 400, 1000, 2500, 5000, 8000, 9000, 8000, 5000, 2500, 1000, 400,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0][:n]
    scale = base_a / 9000.0
    cyc = []
    for i in range(n):
        scans = {}
        a = shape[i] * scale
        if a > 0:
            scans[channel] = (np.array([100.02, 150.07]),
                              np.array([float(a), float(a) * base_b_ratio]))
        cyc.append(ScanCycle(i, 1.0 + 0.05 * i, np.array([]), np.array([]), scans))
    return RawSegmentData("x", "seg", channel, channel + 29, 1, n,
                          np.array([1.0 + 0.05 * i for i in range(n)]),
                          [channel], cyc)


def test_config_default_feature_admission_floor():
    cfg = ProcessingConfig()
    # admission bar (restored to the legacy base-peak quality gate)
    assert cfg.ms2_driven_feature_floor == 1000.0
    # cleanup fragment floor stays aligned to the detector (Phase A)
    assert cfg.msms_intensity_threshold == 200.0


def test_stage1_rejects_ms2_feature_with_base_below_admission_floor():
    """A co-eluting pair whose brightest fragment sits in [200,1000) survives the
    cleanup floor (200) but MUST be rejected by the admission guard (1000)."""
    cfg = ProcessingConfig()
    feats = run_stage1({1: [_seg_scaled(285, base_a=600.0)]}, cfg)[1]
    ms2d = [f for f in feats if f.detection_source == "ms2_driven"]
    assert ms2d == [], f"weak-base cluster must not become a feature: {ms2d}"


def test_stage1_admits_strong_feature_and_retains_200_1000_fragment():
    """A feature with a strong base (>=1000) keeps its co-eluting [200,1000)
    fragment — the primary fix survives the decoupled admission guard."""
    cfg = ProcessingConfig()
    # base 5000 (100.02); co-eluting 150.07 at 0.12x = 600 -> inside [200,1000)
    feats = run_stage1({1: [_seg_scaled(285, base_a=5000.0, base_b_ratio=0.12)]}, cfg)[1]
    ms2d = [f for f in feats if f.detection_source == "ms2_driven"]
    assert ms2d, "strong-base feature must be admitted"
    f = max(ms2d, key=lambda x: x.n_fragments)
    assert any(abs(m - 150.07) < 0.05 for m in f.ms2_mz), \
        f"weak co-eluting fragment dropped: {f.ms2_mz}"
    assert float(np.min(f.ms2_intensity)) < 1000.0, \
        "a retained sub-1000 fragment is expected (primary fix)"
