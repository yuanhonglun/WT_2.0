"""T4: stage6 uses its own isf_rt_tolerance (0.035), tighter than the borrowed
adduct_rt_tolerance (0.05). A child/parent pair 0.04 min apart (< 0.05 but
> 0.035) must NO LONGER be called an ISF false positive.
"""
from __future__ import annotations

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import CandidateFeature, RawSegmentData, ScanCycle
from asfam.pipeline.stage6_isf_detection import run_stage6

_SEG = "100-600"
_CHILD_MZ = 149.02
_PARENT_MZ = 577.13


def _gaussian(rt, center, sigma, height):
    return height * np.exp(-0.5 * ((rt - center) / sigma) ** 2)


def _make_raw(n_cycles=800, co_apex=9.2, sigma=0.05):
    rt = np.linspace(0.0, 15.0, n_cycles)
    mz = np.asarray([_CHILD_MZ, _PARENT_MZ], dtype=np.float64)
    profiles = [_gaussian(rt, co_apex, sigma, h) for h in (8000.0, 12000.0)]
    cycles = [ScanCycle(cycle_index=i, rt=float(rt[i]), ms1_mz=mz.copy(),
                        ms1_intensity=np.asarray([p[i] for p in profiles]),
                        ms2_scans={}) for i in range(n_cycles)]
    return RawSegmentData(file_path="/fake.mzML", segment_name=_SEG,
                          segment_low=95, segment_high=605, replicate_id=1,
                          n_cycles=n_cycles, rt_array=rt.astype(np.float64),
                          precursor_list=[], cycles=cycles, collision_energy=20.0)


def _feat(fid, mz, rt_apex, height, ms2_mz):
    return CandidateFeature(
        feature_id=fid, segment_name=_SEG, replicate_id=1,
        precursor_mz_nominal=int(round(mz)), rt_apex=rt_apex,
        rt_left=rt_apex - 0.1, rt_right=rt_apex + 0.1,
        ms2_mz=np.asarray(ms2_mz, dtype=np.float64),
        ms2_intensity=np.ones(len(ms2_mz)) * 500.0,
        n_fragments=len(ms2_mz), ms1_precursor_mz=mz, ms1_height=height)


def test_isf_not_called_when_rt_beyond_isf_tolerance():
    cfg = ProcessingConfig()  # isf_rt_tolerance=0.035, adduct_rt_tolerance=0.05
    raw = _make_raw()
    # Parent MS2 contains the child m/z (ISF criterion 1 satisfied);
    # apex RT differ by 0.04 (within old adduct gate, beyond new isf gate).
    child = _feat("child", _CHILD_MZ, rt_apex=9.20, height=8000.0,
                  ms2_mz=[70.0, 90.0, 110.0])
    parent = _feat("parent", _PARENT_MZ, rt_apex=9.24, height=12000.0,
                   ms2_mz=[_CHILD_MZ, 193.0, 300.0])
    run_stage6({"1": [child, parent]}, {"1": [raw]}, cfg)
    assert child.status == "active"       # not removed as ISF
    assert child.duplicate_type != "isf"


def test_isf_still_called_when_co_apex():
    cfg = ProcessingConfig()
    raw = _make_raw()
    # apex RT differ by 0.01 (< 0.035) -> genuine co-eluting ISF still removed.
    child = _feat("child", _CHILD_MZ, rt_apex=9.20, height=8000.0,
                  ms2_mz=[70.0, 90.0, 110.0])
    parent = _feat("parent", _PARENT_MZ, rt_apex=9.21, height=12000.0,
                   ms2_mz=[_CHILD_MZ, 193.0, 300.0])
    run_stage6({"1": [child, parent]}, {"1": [raw]}, cfg)
    assert child.status == "isf_removed"
    assert child.duplicate_type == "isf"
