"""Task 3.2: Stage 1b peak_detector dispatch — metra vs msdial A/B tests.

Verifies that ``ProcessingConfig.peak_detector`` correctly routes stage1b's
MS1-driven finder to either the existing metra mass-slice ROI path or the
MS-DIAL faithful derivative-engine path, and that BOTH paths can find the same
clean Gaussian MS1 ion on a shared synthetic fixture.

Fixture:
  - 40 cycles at 0.005 min each (RT 0–0.195 min).
  - Single MS1 ion at m/z 301.10, Gaussian peak centered at cycle 20 (RT 0.1),
    sigma = 3 cycles (0.015 min), amplitude = 20000.
  - No MS2 signals (ms2_quality will be "none"; irrelevant for this test).
  - cfg.msms_relative_threshold = 0.0 to avoid any late-stage MS2 trimming.
"""
from __future__ import annotations

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import RawSegmentData, ScanCycle
from asfam.pipeline.stage1b_ms1_detection import run_stage1b


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian(rt: np.ndarray, center: float, sigma: float, amp: float) -> np.ndarray:
    """Simple noiseless Gaussian peak generator for deterministic tests."""
    return amp * np.exp(-0.5 * ((rt - center) / sigma) ** 2)


def _make_single_ion_segment(
    rt: np.ndarray,
    ion_mz: float,
    ms1_eic: np.ndarray,
) -> RawSegmentData:
    """Build a RawSegmentData with one MS1 ion and no MS2 fragments.

    Parameters
    ----------
    rt : 1-D float array
        RT values in minutes, one per cycle.
    ion_mz : float
        Exact m/z of the MS1 ion.
    ms1_eic : 1-D float array, same length as rt
        Intensity of the MS1 ion at each cycle.
    """
    channel = int(round(ion_mz))
    n_cycles = len(rt)
    cycles: list[ScanCycle] = []
    for ci in range(n_cycles):
        val = float(ms1_eic[ci])
        if val > 0:
            ms1_mz_arr = np.array([ion_mz], dtype=np.float64)
            ms1_int_arr = np.array([val], dtype=np.float64)
        else:
            ms1_mz_arr = np.array([], dtype=np.float64)
            ms1_int_arr = np.array([], dtype=np.float64)
        # MS2 channel present but empty (so _collect_ms2_at_peak can run
        # without KeyError, yet returns empty arrays)
        cycles.append(
            ScanCycle(
                cycle_index=ci,
                rt=float(rt[ci]),
                ms1_mz=ms1_mz_arr,
                ms1_intensity=ms1_int_arr,
                ms2_scans={
                    channel: (
                        np.array([], dtype=np.float64),
                        np.array([], dtype=np.float64),
                    )
                },
            )
        )
    return RawSegmentData(
        file_path="synthetic.mzML",
        segment_name="synthetic_seg",
        segment_low=channel - 1,
        segment_high=channel + 1,
        replicate_id=1,
        n_cycles=n_cycles,
        rt_array=rt.copy(),
        precursor_list=[channel],
        cycles=cycles,
    )


def _run_stage1b(raw: RawSegmentData, cfg: ProcessingConfig) -> list:
    """Run stage1b through the real entry point with empty existing features."""
    rep = str(raw.replicate_id)
    out = run_stage1b(
        data_by_replicate={rep: [raw]},
        existing_features={rep: []},
        config=cfg,
    )
    return out.get(rep, [])


def _make_fixture():
    """Return (raw_segment, base_config) for the shared Gaussian fixture."""
    rt = np.arange(40) * 0.005          # 40 cycles, 0.005 min each → 0–0.195 min
    center_rt = float(rt[20])           # cycle 20 → 0.100 min
    sigma_rt = 3 * 0.005                # 3 cycles → 0.015 min
    ion_mz = 301.10
    ms1_eic = _gaussian(rt, center=center_rt, sigma=sigma_rt, amp=20000.0)
    raw = _make_single_ion_segment(rt, ion_mz, ms1_eic)
    cfg = ProcessingConfig()
    cfg.msms_relative_threshold = 0.0   # disable late-stage MS2 trimming
    return raw, cfg


# ---------------------------------------------------------------------------
# Test 1: metra default mode finds the feature
# ---------------------------------------------------------------------------

def test_metra_mode_default_finds_feature():
    """metra branch (explicitly selected) finds ≥1 feature.

    Pins the metra mass-slice-ROI behaviour on this fixture so any future
    regression in the metra branch is caught. (Default is now msdial, so the
    metra branch is selected explicitly here.)
    """
    raw, cfg = _make_fixture()
    cfg.peak_detector = "metra"  # explicit: exercise the metra branch (default is now msdial)

    feats = _run_stage1b(raw, cfg)
    ms1_driven = [f for f in feats if f.detection_source == "ms1_driven"]

    assert len(ms1_driven) >= 1, (
        f"metra mode expected ≥1 ms1_driven feature, got 0; feats={feats}"
    )
    mzs = [f.ms1_precursor_mz for f in ms1_driven]
    assert any(abs(mz - 301.10) < 0.1 for mz in mzs), (
        f"metra: expected mz_centroid ≈ 301.10 (±0.1), got {mzs}"
    )


# ---------------------------------------------------------------------------
# Test 2: msdial mode finds the feature
# ---------------------------------------------------------------------------

def test_msdial_mode_finds_feature():
    """peak_detector='msdial' routes to find_lc_ms1_features_msdial; must find ≥1.

    MS-DIAL's fixed-0.1 Da SUM-slice + derivative-engine path; the centroid
    (basePeakMz) may differ slightly from metra's, so the m/z tolerance is
    kept generous (±0.2 Da).  The point is that the dispatch reaches the new
    detector without error AND it finds the clean Gaussian ion.
    """
    raw, cfg = _make_fixture()
    cfg.peak_detector = "msdial"

    feats = _run_stage1b(raw, cfg)
    ms1_driven = [f for f in feats if f.detection_source == "ms1_driven"]

    assert len(ms1_driven) >= 1, (
        f"msdial mode expected ≥1 ms1_driven feature, got 0; feats={feats}"
    )
    mzs = [f.ms1_precursor_mz for f in ms1_driven]
    assert any(abs(mz - 301.10) < 0.2 for mz in mzs), (
        f"msdial: expected mz_centroid ≈ 301.10 (±0.2 Da), got {mzs}"
    )


# ---------------------------------------------------------------------------
# Test 3: both modes run without error and each returns ≥1 feature
# ---------------------------------------------------------------------------

def test_metra_and_msdial_both_run():
    """Sanity check: running both modes on identical fixture each returns ≥1 feature.

    Proves the dispatch branch works in both directions and neither code path
    crashes or silently returns empty.
    """
    raw, _ = _make_fixture()

    cfg_metra = ProcessingConfig()
    cfg_metra.msms_relative_threshold = 0.0
    cfg_metra.peak_detector = "metra"  # explicit: default is now msdial
    feats_metra = _run_stage1b(raw, cfg_metra)
    ms1_metra = [f for f in feats_metra if f.detection_source == "ms1_driven"]

    cfg_msdial = ProcessingConfig()
    cfg_msdial.msms_relative_threshold = 0.0
    cfg_msdial.peak_detector = "msdial"
    feats_msdial = _run_stage1b(raw, cfg_msdial)
    ms1_msdial = [f for f in feats_msdial if f.detection_source == "ms1_driven"]

    assert len(ms1_metra) >= 1, (
        f"metra mode returned 0 ms1_driven features (sanity); all feats: {feats_metra}"
    )
    assert len(ms1_msdial) >= 1, (
        f"msdial mode returned 0 ms1_driven features (sanity); all feats: {feats_msdial}"
    )
