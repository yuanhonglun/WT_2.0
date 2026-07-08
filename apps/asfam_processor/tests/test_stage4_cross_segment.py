"""Regression tests for Stage 4 isotope dedup across m/z segments.

When ASFAM data is split across many segment mzML files (e.g. 31 SIM
windows from 075-110 to 975-1010), two features in the same replicate
can come from *different* segments. Each segment file has its own
``rt_array`` whose length matches that file's MS1 cycle count, and those
counts can differ by ±1 cycle between segments.

Stage 4 used to assume "same replicate => same rt_array", which broke
``eic_pearson_in_range`` with::

    IndexError: boolean index did not match indexed array along
    dimension 0; dimension is 827 but corresponding boolean dimension
    is 826

The fix below: when the two compared features come from different
segments, re-extract the second EIC against the first feature's raw so
both EIC vectors share the same rt_array length.
"""
from __future__ import annotations

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import CandidateFeature, RawSegmentData, ScanCycle
from asfam.pipeline.stage4_isotope_dedup import run_stage4


def _gaussian(rt: np.ndarray, center: float, sigma: float, height: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((rt - center) / sigma) ** 2)


def _make_raw(seg_name: str, n_cycles: int, mz_center: float,
              co_apex_rt: float = 4.0) -> RawSegmentData:
    """Build a minimal RawSegmentData with one MS1 ion centered at mz_center
    that elutes as a gaussian at ``co_apex_rt``.
    """
    rt = np.linspace(0.0, 10.0, n_cycles)
    intensity = _gaussian(rt, center=co_apex_rt, sigma=0.05, height=5000.0)
    cycles = []
    for i in range(n_cycles):
        cycles.append(
            ScanCycle(
                cycle_index=i,
                rt=float(rt[i]),
                ms1_mz=np.array([mz_center], dtype=np.float64),
                ms1_intensity=np.array([float(intensity[i])], dtype=np.float64),
                ms2_scans={},
            )
        )
    seg_lo = int(np.floor(mz_center - 5))
    seg_hi = int(np.ceil(mz_center + 5))
    return RawSegmentData(
        file_path=f"/fake/{seg_name}.mzML",
        segment_name=seg_name,
        segment_low=seg_lo,
        segment_high=seg_hi,
        replicate_id=1,
        n_cycles=n_cycles,
        rt_array=rt.astype(np.float64),
        precursor_list=[],
        cycles=cycles,
        collision_energy=20.0,
    )


def _make_feature(feature_id: str, seg_name: str, mz: float,
                  rt_apex: float, height: float) -> CandidateFeature:
    return CandidateFeature(
        feature_id=feature_id,
        segment_name=seg_name,
        replicate_id=1,
        precursor_mz_nominal=int(round(mz)),
        rt_apex=rt_apex,
        rt_left=rt_apex - 0.1,
        rt_right=rt_apex + 0.1,
        ms2_mz=np.array([50.0, 80.0, 120.0], dtype=np.float64),
        ms2_intensity=np.array([1000.0, 500.0, 200.0], dtype=np.float64),
        n_fragments=3,
        ms1_precursor_mz=mz,
        ms1_height=height,
    )


def test_stage4_cross_segment_does_not_crash():
    """Two features from different segments must not crash Stage 4.

    Reproduces the IndexError observed when processing CK1_075-110_P.mzML
    through CK1_975-1010_P.mzML (segments differing by one MS1 cycle).
    """
    cfg = ProcessingConfig()

    # Segment A has 826 MS1 cycles, segment B has 827 - reproduces the
    # exact length mismatch from the crash log (826 vs 827).
    raw_a = _make_raw("109-114", n_cycles=826, mz_center=110.05)
    raw_b = _make_raw("114-119", n_cycles=827, mz_center=111.05)

    # Boundary isotope pair: M+0 in segment A, M+1 (C13) in segment B,
    # both co-eluting at 4.00 min.
    f_light = _make_feature("rep1_00000", "109-114", mz=110.0500,
                            rt_apex=4.00, height=10000.0)
    f_heavy = _make_feature("rep1_00001", "114-119", mz=111.0533,
                            rt_apex=4.00, height=1000.0)

    features_by_rep = {"1": [f_light, f_heavy]}
    data_by_rep = {"1": [raw_a, raw_b]}

    # Pre-fix this raised IndexError; after the fix it must return cleanly.
    out = run_stage4(features_by_rep, cfg, data_by_replicate=data_by_rep)

    assert "1" in out
    assert len(out["1"]) == 2
    # We don't assert dedup outcome here - the synthetic MS2 spectra don't
    # carry isotope-step echoes, so Tier 0 may or may not fire. The point
    # is the EIC gate must run without IndexError when rt_arrays differ.


def test_stage4_same_segment_still_works():
    """Sanity: same-segment pairs continue to use the EIC coelution gate."""
    cfg = ProcessingConfig()
    raw = _make_raw("109-114", n_cycles=826, mz_center=110.05)
    # Force both features into the same segment so we hit the original code
    # path (no re-extraction needed).
    f_light = _make_feature("rep1_00000", "109-114", mz=110.0500,
                            rt_apex=4.00, height=10000.0)
    f_heavy = _make_feature("rep1_00001", "109-114", mz=111.0533,
                            rt_apex=4.00, height=1000.0)
    features_by_rep = {"1": [f_light, f_heavy]}
    data_by_rep = {"1": [raw]}
    out = run_stage4(features_by_rep, cfg, data_by_replicate=data_by_rep)
    assert "1" in out
    assert len(out["1"]) == 2
