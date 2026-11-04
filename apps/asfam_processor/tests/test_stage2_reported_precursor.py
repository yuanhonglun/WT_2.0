"""An MS1-detected, MS2-driven feature reports the ion it was quantified on.

Until v1.4.2610xx stage 2 reported the intensity-weighted centroid of the whole
1-Da isolation window at the MS1 peak apex. Measured on the validation data set
that centroid lies more than 0.01 Da from the quantified (base-peak) ion for
48 % of MS2-driven, MS1-detected features (p95 0.16 Da), because it mixes every
co-isolated ion — so a 0.01 Da library-match window for "measured MS1
precursors" was only true for the MS1-driven branch. The default now reports
the base peak; ``ms1_precursor_report="window_centroid"`` restores the old
value, and the centroid is kept on ``ms1_window_mz`` either way.
"""
from __future__ import annotations

import numpy as np
import pytest

from asfam.config import ProcessingConfig
from asfam.models import CandidateFeature, RawSegmentData, ScanCycle
from asfam.pipeline.stage2_ms1_assignment import run_stage2

STRONG, WEAK = 285.05, 285.42          # two co-isolated ions, 10000 : 3000
CENTROID = (STRONG * 10000 + WEAK * 3000) / 13000     # ≈ 285.1354


def _gaussian(rt, center, sigma, amp):
    return amp * np.exp(-0.5 * ((rt - center) / sigma) ** 2)


def _scene():
    rt = np.arange(30) * 0.005
    apex_rt = rt[14]
    ions = [(STRONG, _gaussian(rt, apex_rt, 0.02, 10000.0)),
            (WEAK, _gaussian(rt, apex_rt, 0.02, 3000.0))]
    cycles = []
    for ci in range(len(rt)):
        mzs, ints = [], []
        for mz, eic in ions:
            if eic[ci] > 0:
                mzs.append(mz)
                ints.append(float(eic[ci]))
        cycles.append(ScanCycle(cycle_index=ci, rt=float(rt[ci]),
                                ms1_mz=np.array(mzs), ms1_intensity=np.array(ints),
                                ms2_scans={285: (np.empty(0), np.empty(0))}))
    raw = RawSegmentData(file_path="synthetic.mzML", segment_name="seg_285",
                         segment_low=284, segment_high=286, replicate_id=1,
                         n_cycles=len(rt), rt_array=rt.copy(), precursor_list=[285],
                         cycles=cycles)
    feat = CandidateFeature(
        feature_id="F_ms2driven", segment_name="seg_285", replicate_id=1,
        precursor_mz_nominal=285, rt_apex=apex_rt, rt_left=apex_rt - 0.03,
        rt_right=apex_rt + 0.03, ms2_mz=np.array([120.0]),
        ms2_intensity=np.array([3000.0]), n_fragments=1,
        signal_type="ms1_detected", detection_source="ms2_driven",
    )
    return raw, feat


def test_default_reports_the_base_peak_and_keeps_the_centroid():
    raw, feat = _scene()
    run_stage2({"1": [raw]}, {"1": [feat]}, ProcessingConfig())
    assert feat.signal_type == "ms1_detected" and feat.mz_source == "ms1_peak"
    assert feat.ms1_precursor_mz == pytest.approx(STRONG, abs=1e-6)
    assert feat.ms1_quant_mz == pytest.approx(STRONG, abs=1e-6)
    assert feat.ms1_window_mz == pytest.approx(CENTROID, abs=1e-3)
    assert feat.precursor_mz == pytest.approx(STRONG, abs=1e-6)


def test_window_centroid_mode_restores_the_old_value():
    raw, feat = _scene()
    cfg = ProcessingConfig()
    cfg.ms1_precursor_report = "window_centroid"
    run_stage2({"1": [raw]}, {"1": [feat]}, cfg)
    assert feat.signal_type == "ms1_detected"
    assert feat.ms1_precursor_mz == pytest.approx(CENTROID, abs=1e-3)
    assert feat.ms1_quant_mz == pytest.approx(STRONG, abs=1e-6)     # unchanged
    assert feat.ms1_window_mz == pytest.approx(CENTROID, abs=1e-3)


def test_unknown_mode_is_rejected():
    raw, feat = _scene()
    cfg = ProcessingConfig()
    cfg.ms1_precursor_report = "whatever"
    with pytest.raises(ValueError):
        run_stage2({"1": [raw]}, {"1": [feat]}, cfg)


def test_the_mode_is_part_of_the_checkpoint_fingerprint():
    from asfam.io.spill import config_fingerprint
    a, b = ProcessingConfig(), ProcessingConfig()
    b.ms1_precursor_report = "window_centroid"
    assert config_fingerprint(a, None, ["x_285-286_1.mzML"]) != \
        config_fingerprint(b, None, ["x_285-286_1.mzML"])
