"""Stage 2 and stage 2.5 centre every m/z window on the *recorded* isolation target.

ASFAM acquires 1-Da MS2 isolation windows whose targets do not sit on
integers: the fractional part of the centre rises with mass (X.0 below m/z
~135, X.1 to X.4 in between, X.5 above ~825, X.6 in the last segment). The
reader keys each window by ``floor(target)`` and records the true target in
``RawSegmentData.precursor_targets``; the range the instrument actually
isolated is ``[target - 0.5, target + 0.5]``.

Until v1.4.261103 stage 2 built every m/z window as ``[channel - 0.5,
channel + 0.5]`` -- centred on the floor -- and stage 2.5 bounded its
neutral-loss candidates the same way. The upper 10-60 % of every window above
m/z ~135 was therefore never searched: an ion sitting there got no MS1 peak,
no correct neutral-loss precursor and a window centroid 0.5-1 Da off, so the
whole compound was reported as MS2-only (measured on the 2026-11 blank
experiment: 43 % of the MS2-only features of the 915-944 segment had a
co-eluting precursor in the unsearched half). The stage-2 part follows the
unmerged July branch ``f5-isolation-window``; this version also stamps the
centre on the feature (``isolation_target_mz``) so that stage 2.5 and the
alignment, which never see raw data, judge the same window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asfam.config import ProcessingConfig
from asfam.io.project_file import (
    _candidates_to_dicts,
    _dicts_to_candidates,
    _dicts_to_features,
    _features_to_dicts,
)
from asfam.models import CandidateFeature, Feature, RawSegmentData, ScanCycle
from asfam.pipeline.stage2_ms1_assignment import run_stage2, window_centroid_mz
from asfam.pipeline.stage2b_inference import run_stage2b
from asfam.pipeline.stage8_export import _export_csv


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _gaussian(rt: np.ndarray, center: float, sigma: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((rt - center) / sigma) ** 2)


def _build_raw_segment(
    rt: np.ndarray,
    ms1_ions: list[tuple[float, np.ndarray]],
    channel: int,
    precursor_targets: dict[int, float] | None = None,
    segment_name: str = "795-824",
) -> RawSegmentData:
    """A segment carrying an MS1 survey and (optionally) recorded targets.

    ``precursor_targets=None`` reproduces a legacy project load / old synthetic
    fixture: the map is empty and the window falls back to the integer channel.
    """
    cycles: list[ScanCycle] = []
    for ci in range(len(rt)):
        mzs: list[float] = []
        ints: list[float] = []
        for mz, eic in ms1_ions:
            val = float(eic[ci])
            if val > 0.0:
                mzs.append(mz)
                ints.append(val)
        cycles.append(ScanCycle(
            cycle_index=ci,
            rt=float(rt[ci]),
            ms1_mz=np.array(mzs, dtype=np.float64),
            ms1_intensity=np.array(ints, dtype=np.float64),
            ms2_scans={channel: (np.empty(0), np.empty(0))},
        ))
    return RawSegmentData(
        file_path="synthetic.mzML",
        segment_name=segment_name,
        segment_low=channel - 1,
        segment_high=channel + 1,
        replicate_id=1,
        n_cycles=len(rt),
        rt_array=rt.copy(),
        precursor_list=[channel],
        cycles=cycles,
        precursor_targets=precursor_targets or {},
    )


def _ms2_driven_feature(
    channel: int,
    rt_apex: float,
    fragments=(120.0, 180.0),
    feature_id: str = "F_ms2driven",
    segment_name: str = "795-824",
) -> CandidateFeature:
    half = 0.03
    frags = np.array(fragments, dtype=np.float64)
    return CandidateFeature(
        feature_id=feature_id,
        segment_name=segment_name,
        replicate_id=1,
        precursor_mz_nominal=channel,
        rt_apex=rt_apex,
        rt_left=rt_apex - half,
        rt_right=rt_apex + half,
        ms2_mz=frags,
        ms2_intensity=np.linspace(3000.0, 1500.0, len(frags)),
        n_fragments=len(frags),
        signal_type="ms1_detected",   # placeholder; stage 2 rewrites it
        detection_source="ms2_driven",
    )


# The worst case: above m/z ~825 the targets sit at X.5, so the true window is
# [N, N+1] and the floor-centred window [N-0.5, N+0.5] overlaps it by only 50%.
CHANNEL = 811
TARGET = 811.5           # true window: [811.0, 812.0]
TRUE_ION = 811.8         # isolated: inside [811, 812], outside [810.5, 811.5]
DECOY_ION = 810.7        # NOT isolated: inside [810.5, 811.5], outside [811, 812]


def _two_ion_scene(true_amp: float, decoy_amp: float, targets):
    rt = np.arange(30) * 0.005
    apex_rt = float(rt[14])
    raw = _build_raw_segment(
        rt,
        ms1_ions=[(TRUE_ION, _gaussian(rt, apex_rt, 0.02, true_amp)),
                  (DECOY_ION, _gaussian(rt, apex_rt, 0.02, decoy_amp))],
        channel=CHANNEL,
        precursor_targets=targets,
    )
    return raw, _ms2_driven_feature(CHANNEL, apex_rt)


# ---------------------------------------------------------------------------
# The window centre itself
# ---------------------------------------------------------------------------

def test_isolation_target_is_the_recorded_centre_not_the_floor():
    raw = _build_raw_segment(np.arange(5) * 0.005, [], CHANNEL,
                             precursor_targets={CHANNEL: TARGET})
    centre = raw.isolation_target(CHANNEL)
    assert centre == pytest.approx(TARGET)
    lo, hi = centre - 0.5, centre + 0.5
    assert (lo, hi) == pytest.approx((811.0, 812.0)), "window must span [N, N+1]"
    assert lo <= TRUE_ION <= hi
    assert not (lo <= DECOY_ION <= hi)


def test_isolation_target_falls_back_to_the_integer_channel():
    no_targets = _build_raw_segment(np.arange(5) * 0.005, [], CHANNEL,
                                    precursor_targets=None)
    assert no_targets.isolation_target(CHANNEL) == pytest.approx(float(CHANNEL))
    with_targets = _build_raw_segment(np.arange(5) * 0.005, [], CHANNEL,
                                      precursor_targets={CHANNEL: TARGET})
    # An unknown channel falls back rather than raising.
    assert with_targets.isolation_target(999) == pytest.approx(999.0)


# ---------------------------------------------------------------------------
# Stage 2 end to end
# ---------------------------------------------------------------------------

def test_stage2_quantifies_on_the_isolated_ion_not_the_neighbours():
    """The headline regression.

    Two MS1 ions straddle channel 811, whose target is 811.5: the isolated ion
    811.8 (amplitude 5,000) and the taller, never-isolated neighbour 810.7
    (amplitude 20,000) that belongs to channel 810. Pre-fix the chromatogram was
    built on [810.5, 811.5], so the neighbour won every time and the feature was
    quantified, aligned and gap-filled on an ion the instrument never isolated
    for it. Post-fix only 811.8 is in range.
    """
    raw, feat = _two_ion_scene(5000.0, 20000.0, {CHANNEL: TARGET})
    run_stage2({"1": [raw]}, {"1": [feat]}, ProcessingConfig())

    assert feat.signal_type == "ms1_detected"
    assert feat.mz_source == "ms1_peak"
    assert feat.isolation_target_mz == pytest.approx(TARGET)
    assert feat.ms1_quant_mz == pytest.approx(TRUE_ION, abs=1e-6)
    assert feat.ms1_precursor_mz == pytest.approx(TRUE_ION, abs=1e-6)
    assert feat.ms1_window_mz == pytest.approx(TRUE_ION, abs=1e-6)
    # Height on the isolated ion's 5,000 scale, not the neighbour's 20,000.
    assert 3000.0 < feat.ms1_height < 10000.0


def test_stage2_without_recorded_targets_keeps_the_integer_window():
    """No recorded targets (legacy load / old fixture): the pre-fix behaviour."""
    raw, feat = _two_ion_scene(5000.0, 20000.0, None)
    run_stage2({"1": [raw]}, {"1": [feat]}, ProcessingConfig())

    assert feat.signal_type == "ms1_detected"
    assert feat.isolation_target_mz == pytest.approx(float(CHANNEL))
    assert feat.ms1_quant_mz == pytest.approx(DECOY_ION, abs=1e-6)


def test_stage2_window_centroid_and_lowres_fallback_use_the_target_window():
    """Signal too weak for the strict (300) and relaxed (50) peak floors.

    Stage 2 then only stores the apex-cycle window centroid; stage 2.5 makes it
    the reported precursor (``ms1_lowres``) once neutral-loss inference is not
    applicable (two fragments < three). Both must come from the isolated range,
    so the neighbour at 810.7 must not contribute.
    """
    raw, feat = _two_ion_scene(35.0, 40.0, {CHANNEL: TARGET})
    run_stage2({"1": [raw]}, {"1": [feat]}, ProcessingConfig())

    assert feat.signal_type == "ms2_only"
    assert feat.isolation_target_mz == pytest.approx(TARGET)
    assert feat.ms1_window_mz == pytest.approx(TRUE_ION, abs=1e-6)
    assert window_centroid_mz(feat, raw) == pytest.approx(TRUE_ION, abs=1e-6)

    stats: dict = {}
    run_stage2b({"1": [feat]}, ProcessingConfig(), stats=stats)
    assert feat.mz_source == "ms1_lowres"
    assert feat.precursor_mz == pytest.approx(TRUE_ION, abs=1e-6)


def test_stage2_stamps_the_window_centre_on_every_feature_of_the_window():
    """MS2-only features get the centre too: stage 2.5 and alignment read it."""
    rt = np.arange(60) * 0.005
    apex_rt = float(rt[14])
    raw = _build_raw_segment(
        rt, ms1_ions=[(TRUE_ION, _gaussian(rt, apex_rt, 0.02, 5000.0))],
        channel=CHANNEL, precursor_targets={CHANNEL: TARGET},
    )
    with_ms1 = _ms2_driven_feature(CHANNEL, apex_rt, feature_id="F_a")
    without_ms1 = _ms2_driven_feature(CHANNEL, float(rt[50]), feature_id="F_b")
    run_stage2({"1": [raw]}, {"1": [with_ms1, without_ms1]}, ProcessingConfig())

    assert with_ms1.signal_type == "ms1_detected"
    assert without_ms1.signal_type == "ms2_only"
    assert with_ms1.isolation_target_mz == pytest.approx(TARGET)
    assert without_ms1.isolation_target_mz == pytest.approx(TARGET)


# ---------------------------------------------------------------------------
# Stage 2.5: neutral-loss candidates are bounded by the recorded window
# ---------------------------------------------------------------------------

NL_CHANNEL = 916
NL_TARGET = 916.5            # true window [916.0, 917.0]
NL_PRECURSOR = 916.70        # in the isolated range, outside [915.5, 916.5]
#: Three fragments one common neutral loss (H2O, NH3, CO) below the precursor.
NL_FRAGMENTS = (NL_PRECURSOR - 18.01056, NL_PRECURSOR - 17.02655, NL_PRECURSOR - 27.99491)


def _ms2_only_for_nl(target):
    feat = CandidateFeature(
        feature_id="F_nl", segment_name="915-944", replicate_id=1,
        precursor_mz_nominal=NL_CHANNEL, rt_apex=5.0, rt_left=4.9, rt_right=5.1,
        ms2_mz=np.array(NL_FRAGMENTS, dtype=np.float64),
        ms2_intensity=np.linspace(1000.0, 500.0, 3),
        n_fragments=3, signal_type="ms2_only", ms2_rep_ion_mz=float(NL_FRAGMENTS[0]),
        isolation_target_mz=target,
    )
    return feat


def test_neutral_loss_consensus_searches_the_target_centred_window():
    feat = _ms2_only_for_nl(NL_TARGET)
    stats: dict = {}
    run_stage2b({"1": [feat]}, ProcessingConfig(), stats=stats)
    assert feat.mz_source == "nl_consensus"
    assert feat.mz_confidence == "high"
    assert feat.inferred_mz == pytest.approx(NL_PRECURSOR, abs=1e-4)
    assert stats["nl_assigned"] == 1


def test_neutral_loss_consensus_without_a_recorded_centre_keeps_the_integer_window():
    """Pre-fix geometry: the true precursor at 916.70 lies outside [915.5, 916.5].

    The consensus then cannot land on it: it either finds nothing or settles
    on a wrong value inside the integer window (a fragment plus some other
    listed loss), which is exactly the error the recorded centre removes.
    """
    feat = _ms2_only_for_nl(None)
    stats: dict = {}
    run_stage2b({"1": [feat]}, ProcessingConfig(), stats=stats)
    if feat.mz_source == "nl_consensus":
        assert abs(feat.inferred_mz - NL_PRECURSOR) > 0.4
        assert feat.inferred_mz <= NL_CHANNEL + 0.5
    else:
        assert stats["nl_assigned"] == 0


# ---------------------------------------------------------------------------
# The centre survives export and project files
# ---------------------------------------------------------------------------

def _aligned_feature(fid: str, **overrides) -> Feature:
    feat = Feature(
        feature_id=fid, precursor_mz=916.7, rt=5.0, rt_left=4.9, rt_right=5.1,
        signal_type="ms2_only",
        ms2_mz=np.linspace(100.0, 180.0, 4),
        ms2_intensity=np.linspace(1000.0, 500.0, 4),
        n_fragments=4,
    )
    for key, value in overrides.items():
        setattr(feat, key, value)
    return feat


def test_export_carries_the_isolation_target_column(tmp_path):
    stamped = _aligned_feature("F1", isolation_target_mz=916.5)
    blank = _aligned_feature("F2")
    out = tmp_path / "features.csv"
    _export_csv([stamped, blank], out, ProcessingConfig())
    df = pd.read_csv(out, comment="#", keep_default_na=False)
    assert "isolation_target_mz" in df.columns
    assert float(df.iloc[0]["isolation_target_mz"]) == pytest.approx(916.5)
    assert df.iloc[1]["isolation_target_mz"] == ""


def test_project_file_round_trips_the_isolation_target():
    cand = _ms2_driven_feature(CHANNEL, 1.0)
    cand.isolation_target_mz = TARGET
    back = _dicts_to_candidates(_candidates_to_dicts([cand]))[0]
    assert back.isolation_target_mz == pytest.approx(TARGET)
    # Older project files have no such key: default None, no error.
    old = _candidates_to_dicts([cand])
    old[0].pop("isolation_target_mz")
    assert _dicts_to_candidates(old)[0].isolation_target_mz is None

    feat = _aligned_feature("F1", isolation_target_mz=916.5)
    back_f = _dicts_to_features(_features_to_dicts([feat]))[0]
    assert back_f.isolation_target_mz == pytest.approx(916.5)
