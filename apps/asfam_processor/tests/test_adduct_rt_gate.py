"""T4: stage5 must have a direct pairwise apex-RT gate (previously absent).

Spec §2.8: stage5 had no direct apex-RT gate — it relied only on the
running-median RT bucketing + EIC-coelution Pearson. The running-median
*chaining* lets a bucket span more than ``adduct_rt_tolerance`` when an
intermediate feature bridges the gap, so an [M+H]+/[M+Na]+ pair whose apexes
are >0.05 min apart can still be compared pairwise and merged (EIC tails
overlap → Pearson passes). A direct pairwise apex-RT gate rejects those.

We make every m/z co-elute as identical Gaussians (so EIC-coelution Pearson=1,
NOT the thing under test), then use the features' ``rt_apex`` attribute — which
is exactly what the new gate checks — to isolate the gate from coelution. The
RT-far case needs a THIRD "spacer" feature at the midpoint so the running-median
bucketing chains all three into one group (a bare 2-feature 0.06-apart pair
lands in separate buckets and is never compared — see probe in commit msg).
"""
from __future__ import annotations

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import CandidateFeature, RawSegmentData, ScanCycle
from asfam.pipeline.stage5_adduct_dedup import run_stage5

# Neutral mass M = 285.0, positive mode: [M+H]+ and [M+Na]+ co-eluting pair.
_MZ_H = 285.0 + 1.00727646677
_MZ_NA = 285.0 + 22.989218
_MID = 295.5   # spacer m/z: does NOT adduct-pair with either, only chains bucket
_SEG = "280-310"


def _gaussian(rt, center, sigma, height):
    return height * np.exp(-0.5 * ((rt - center) / sigma) ** 2)


def _make_raw(mz_list, height_list, n_cycles=800, co_apex=4.0, sigma=0.05):
    """MS1 survey carrying every m/z as an identical co-eluting Gaussian, so
    all EICs are perfectly correlated (coelution Pearson=1, not under test)."""
    rt = np.linspace(0.0, 10.0, n_cycles)
    profiles = [_gaussian(rt, co_apex, sigma, h) for h in height_list]
    mz = np.asarray(mz_list, dtype=np.float64)
    cycles = [ScanCycle(cycle_index=i, rt=float(rt[i]), ms1_mz=mz.copy(),
                        ms1_intensity=np.asarray([p[i] for p in profiles]),
                        ms2_scans={}) for i in range(n_cycles)]
    return RawSegmentData(file_path="/fake.mzML", segment_name=_SEG,
                          segment_low=275, segment_high=315, replicate_id=1,
                          n_cycles=n_cycles, rt_array=rt.astype(np.float64),
                          precursor_list=[], cycles=cycles, collision_energy=20.0)


def _feat(fid, mz, rt_apex, height):
    return CandidateFeature(
        feature_id=fid, segment_name=_SEG, replicate_id=1,
        precursor_mz_nominal=int(round(mz)), rt_apex=rt_apex,
        rt_left=rt_apex - 0.1, rt_right=rt_apex + 0.1,
        ms2_mz=np.array([50.0, 80.0, 120.0], dtype=np.float64),
        ms2_intensity=np.array([1000.0, 500.0, 200.0], dtype=np.float64),
        n_fragments=3, ms1_precursor_mz=mz, ms1_height=height)


def test_adduct_pair_rejected_when_apex_rt_far():
    """[M+H]+ (rt 4.00) and [M+Na]+ (rt 4.06) are chained into one RT bucket by
    a spacer at 4.03, and their EICs are identical (coelution passes) — but the
    apexes are 0.06 min apart (> 0.05). The new pairwise gate must reject."""
    cfg = ProcessingConfig()  # adduct_rt_tolerance = 0.05
    raw = _make_raw([_MZ_H, _MID, _MZ_NA], [10000.0, 5000.0, 3000.0])
    fh = _feat("A", _MZ_H, rt_apex=4.00, height=10000.0)
    fmid = _feat("M", _MID, rt_apex=4.03, height=5000.0)
    fna = _feat("B", _MZ_NA, rt_apex=4.06, height=3000.0)
    run_stage5({"1": [fh, fmid, fna]}, {"1": [raw]}, cfg)
    assert fh.is_duplicate is False
    assert fna.is_duplicate is False           # NOT merged (apex RT too far)
    assert fna.adduct_group_id is None
    assert fh.adduct_group_id is None


def test_adduct_pair_still_merges_when_co_apex():
    """apex RT differ by 0.02 (< 0.05) -> still a valid co-apex adduct merge."""
    cfg = ProcessingConfig()
    raw = _make_raw([_MZ_H, _MZ_NA], [10000.0, 3000.0])
    fh = _feat("A", _MZ_H, rt_apex=4.00, height=10000.0)
    fna = _feat("B", _MZ_NA, rt_apex=4.02, height=3000.0)
    run_stage5({"1": [fh, fna]}, {"1": [raw]}, cfg)
    assert fh.adduct_group_id == fna.adduct_group_id
    assert fh.adduct_group_id is not None
    assert fna.is_duplicate is True            # correctly merged
