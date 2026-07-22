"""Gap filler: port fidelity against LcmsGapFiller.cs, and the participation rules."""
from __future__ import annotations

import numpy as np
import pytest

from metabo_core.alignment.gap_filler import (
    FILLED,
    MS1,
    NO_SIGNAL,
    PRODUCT,
    GapFillConfig,
    GapFillResult,
    GapFillTarget,
    QuantIon,
    _is_broad_peak_top,
    _is_peak_top,
    _search_left_edge,
    _search_right_edge,
    fill_from_chromatogram,
    fill_spot,
    gap_fill_target,
    make_filled_peak,
    quant_ion,
)
from metabo_core.alignment.joiner import AlignmentSpot, build_feature
from metabo_core.models import CandidateFeature


def _feature(**kw) -> CandidateFeature:
    base = dict(
        feature_id="F1", segment_name="100-129", replicate_id=1,
        precursor_mz_nominal=110, rt_apex=5.0, rt_left=4.9, rt_right=5.1,
        ms2_mz=np.array([50.0, 70.0]), ms2_intensity=np.array([100.0, 900.0]),
        n_fragments=2, ms1_height=1000.0, ms1_area=500.0, ms1_sn=10.0,
        signal_type="ms1_detected", ms1_precursor_mz=110.05, ms1_quant_mz=110.0512,
    )
    base.update(kw)
    return CandidateFeature(**base)


def _gaussian(center: float, width: float = 0.05, height: float = 1000.0,
              lo: float = 4.5, hi: float = 5.5, n: int = 61):
    rt = np.linspace(lo, hi, n)
    return rt, height * np.exp(-0.5 * ((rt - center) / width) ** 2)


CFG = GapFillConfig()


# ---------------------------------------------------------------------------
# Chromatogram.cs predicates
# ---------------------------------------------------------------------------

def test_is_peak_top_rejects_a_flat_plateau():
    # IsPeakTop demands a strict change on at least one side (Chromatogram.cs:575).
    assert _is_peak_top(np.array([1.0, 2.0, 3.0, 2.0, 1.0]), 2)
    assert not _is_peak_top(np.array([5.0, 5.0, 5.0, 5.0, 5.0]), 2)
    assert not _is_peak_top(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 2)


def test_is_broad_peak_top_needs_one_supporting_shoulder():
    # A local max whose *outer* neighbours both rise away from it is a notch in a
    # valley, not a broad top: neither disjunct of Chromatogram.cs:606-607 holds.
    y = np.array([9.0, 1.0, 2.0, 1.0, 9.0])
    assert _is_peak_top(y, 2)
    assert not _is_broad_peak_top(y, 2)

    assert _is_broad_peak_top(np.array([1.0, 2.0, 3.0, 2.0, 1.0]), 2)


def test_edge_search_hard_stops_on_a_plateau_soft_does_not():
    y = np.array([5.0, 3.0, 3.0, 9.0, 3.0, 3.0, 5.0])
    # walking left from index 2: soft keeps going (3 is not > 3), hard stops.
    assert _search_left_edge(y, 2, 0, hard=False) == 1
    assert _search_left_edge(y, 2, 0, hard=True) == 2
    # walking right from index 4: mirror image.
    assert _search_right_edge(y, 4, 6, hard=False) == 5
    assert _search_right_edge(y, 4, 6, hard=True) == 4


def test_edge_search_respects_its_limit():
    y = np.linspace(10.0, 1.0, 10)          # strictly decreasing
    assert _search_right_edge(y, 1, 4, hard=True) == 4
    assert _search_left_edge(np.linspace(1.0, 10.0, 10), 8, 5, hard=True) == 5


# ---------------------------------------------------------------------------
# fill_from_chromatogram
# ---------------------------------------------------------------------------

def _target(rt_center=5.0, noise=100.0, **kw) -> GapFillTarget:
    return GapFillTarget(
        quant=QuantIon(mz=110.05, kind=MS1, tolerance=0.01),
        rt_center=rt_center, rt_lo=rt_center - 0.5, rt_hi=rt_center + 0.5,
        peak_width=0.2, estimated_noise=noise, **kw,
    )


def test_fills_a_peak_at_the_centre():
    rt, intensity = _gaussian(5.0)
    result = fill_from_chromatogram(rt, intensity, _target(), CFG)

    assert result.status == FILLED
    assert result.rt_apex == pytest.approx(5.0, abs=0.02)
    assert result.rt_left < result.rt_apex < result.rt_right
    # LWMA level 3 lowers the apex a little; area is above baseline, in seconds.
    assert 800.0 < result.height <= 1000.0
    assert result.area > 0.0
    assert result.sn_ratio == pytest.approx(result.height / 100.0)


def test_empty_chromatogram_is_no_signal():
    empty = np.empty(0)
    assert fill_from_chromatogram(empty, empty, _target(), CFG).status == NO_SIGNAL


def test_all_zero_chromatogram_force_inserts_a_zero_rather_than_a_hole():
    # MS-DIAL's IsForceInsertForGapFilling default: a sample with no signal gets
    # a peak of height 0 at the centre, not a missing cell.
    rt = np.linspace(4.5, 5.5, 61)
    result = fill_from_chromatogram(rt, np.zeros_like(rt), _target(), CFG)

    assert result.status == NO_SIGNAL
    assert result.height == 0.0
    assert result.area == 0.0
    assert result.sn_ratio == 0.0
    # Not force-inserted: IsBroadPeakTop has no strict-change guard (unlike
    # IsPeakTop, Chromatogram.cs:575 vs :602-607), so on a flat trace every
    # interior point qualifies and the search finds the one nearest the centre.
    assert not result.forced


def test_forced_flags_a_fill_with_no_peak_top_in_range():
    """``forced`` is what separates "the picker missed a peak" from "no peak here".

    Weak on its own: only a monotone trace has no broad top at all, and smoothed
    noise usually offers one. It rules background *in*, never out.
    """
    rt = np.linspace(4.5, 5.5, 61)
    ramp = np.linspace(100.0, 900.0, 61)          # monotone: no peak top anywhere
    background = fill_from_chromatogram(rt, ramp, _target(), CFG)
    assert background.status == FILLED and background.height > 0.0
    assert background.forced

    _, peak = _gaussian(5.0)
    assert not fill_from_chromatogram(rt, peak, _target(), CFG).forced


def test_without_force_insert_a_flat_trace_yields_nothing():
    rt = np.linspace(4.5, 5.5, 61)
    result = fill_from_chromatogram(
        rt, np.zeros_like(rt), _target(), GapFillConfig(force_insert=False),
    )
    assert result.status == NO_SIGNAL


def test_peak_outside_the_rt_tolerance_is_not_claimed():
    # A tall peak 0.4 min away is beyond rt_tolerance=0.1: GetNearestPeak walks
    # past it, force-insert frames the (empty) centre instead.
    rt, intensity = _gaussian(5.4, width=0.02, height=5000.0)
    result = fill_from_chromatogram(rt, intensity, _target(rt_center=5.0), CFG)

    assert result.rt_apex == pytest.approx(5.0, abs=0.06)
    assert result.height < 100.0


def test_nearest_of_two_resolved_peaks_wins_over_the_taller_one():
    # Both tops sit inside rt_tolerance. GetNearestPeak picks by |t - centre|,
    # not by height, and the hard edge search then stops in the valley — so the
    # `IsPeakTop` upgrade loop, which only scans (left, right), never reaches the
    # 10x taller neighbour.
    rt = np.linspace(4.5, 5.5, 401)
    near = 500.0 * np.exp(-0.5 * ((rt - 5.03) / 0.012) ** 2)
    far = 5000.0 * np.exp(-0.5 * ((rt - 5.09) / 0.012) ** 2)
    result = fill_from_chromatogram(rt, near + far, _target(rt_center=5.0), CFG)

    assert result.rt_apex == pytest.approx(5.03, abs=0.005)
    assert result.height < 600.0
    assert result.rt_right < 5.06


def test_a_taller_top_inside_the_bracket_upgrades_the_apex():
    # Unresolved shoulder: 3 sigma apart, no valley between them, so the hard
    # edge search brackets both and the upgrade loop takes the taller top
    # (LcmsGapFiller.cs:128-132).
    rt = np.linspace(4.5, 5.5, 401)
    near = 500.0 * np.exp(-0.5 * ((rt - 5.03) / 0.02) ** 2)
    far = 5000.0 * np.exp(-0.5 * ((rt - 5.09) / 0.02) ** 2)
    result = fill_from_chromatogram(rt, near + far, _target(rt_center=5.0), CFG)

    assert result.rt_apex == pytest.approx(5.09, abs=0.005)


# ---------------------------------------------------------------------------
# quantitation ion
# ---------------------------------------------------------------------------

def test_ms2_only_is_quantified_on_its_representative_fragment():
    feat = _feature(signal_type="ms2_only", ms2_rep_ion_mz=70.0123,
                    ms1_quant_mz=None)
    ion = quant_ion(feat, CFG)

    assert ion == QuantIon(mz=70.0123, kind=PRODUCT,
                           tolerance=CFG.product_mz_tolerance, channel=110)


def test_ms1_feature_is_quantified_on_ms1_quant_mz_not_the_precursor():
    # ms1_precursor_mz is the isolation-window centroid and can sit on empty m/z;
    # ms1_quant_mz is the ion the stored height was read off.
    ion = quant_ion(_feature(), CFG)
    assert ion.mz == 110.0512
    assert ion.kind == MS1
    assert ion.tolerance == CFG.ms1_mz_tolerance


def test_a_feature_with_no_quantitation_ion_is_unfillable():
    assert quant_ion(_feature(ms1_quant_mz=None), CFG) is None
    assert quant_ion(_feature(signal_type="ms2_only", ms2_rep_ion_mz=None), CFG) is None


def test_the_route_gap_fill_integrates_is_the_route_the_joiner_groups_by():
    """``quant_ion`` and ``CandidateFeature.quant_route`` must be one decision.

    The joiner refuses to put two routes in one spot; the gap filler then picks
    the spot's chromatogram from its tallest detected peak. If these two ever
    disagreed about what route a feature is on, a spot the joiner certified as
    pure could still be filled on the wrong kind of trace — the exact bug the
    guard exists to prevent, reintroduced one layer down.
    """
    ms1 = _feature()
    product = _feature(signal_type="ms2_only", ms2_rep_ion_mz=70.0123,
                       ms1_quant_mz=None)

    assert (ms1.quant_route, product.quant_route) == (MS1, PRODUCT)
    assert quant_ion(ms1, CFG).kind == ms1.quant_route
    assert quant_ion(product, CFG).kind == product.quant_route

    # And the joiner keys each on the ion its height was read off — never on the
    # precursor, which on the product route is a noise centroid 40 Da away.
    assert ms1.align_mz == ms1.ms1_quant_mz == 110.0512
    assert product.align_mz == product.ms2_rep_ion_mz == 70.0123
    assert product.precursor_mz == 110.05      # what it would have been keyed on


# ---------------------------------------------------------------------------
# gap_fill_target
# ---------------------------------------------------------------------------

def test_target_averages_rt_but_takes_mz_from_the_tallest_peak():
    peaks = [
        _feature(feature_id="a", rt_apex=5.0, ms1_height=100.0, ms1_quant_mz=110.01,
                 rt_left=4.95, rt_right=5.05),
        _feature(feature_id="b", rt_apex=5.2, ms1_height=900.0, ms1_quant_mz=110.09,
                 rt_left=5.0, rt_right=5.4, ms1_sn=9.0),
    ]
    target = gap_fill_target(peaks, CFG)

    assert target.rt_center == pytest.approx(5.1)          # arithmetic mean
    assert target.quant.mz == 110.09                       # tallest, not averaged
    assert target.peak_width == pytest.approx(0.4)         # widest
    # noise = max(height / sn) = max(100/10, 900/9) = 100
    assert target.estimated_noise == pytest.approx(100.0)
    # +/- 1.5 * max(peak_width, 0.2)
    assert target.rt_lo == pytest.approx(5.1 - 0.6)
    assert target.rt_hi == pytest.approx(5.1 + 0.6)


def test_target_floors_a_narrow_peak_width_at_min_peak_width():
    target = gap_fill_target([_feature(rt_left=4.99, rt_right=5.01)], CFG)
    assert target.rt_hi - target.rt_center == pytest.approx(1.5 * 0.2)


def test_no_detected_peaks_means_no_target():
    assert gap_fill_target([], CFG) is None


# ---------------------------------------------------------------------------
# participation rules (Step 5.4)
# ---------------------------------------------------------------------------

def _spot_with(peaks: dict) -> AlignmentSpot:
    spot = AlignmentSpot(
        index=0, master_mz=110.05, master_rt=5.0, origin_sample_id="s1",
        peaks=peaks, representative_sample_id="s1",
        representative_ms2=(np.array([70.0]), np.array([900.0])),
    )
    return spot


def test_a_nonzero_gap_filled_peak_never_moves_mean_height_or_cv():
    """The acceptance test for Step 5.4's second trap.

    ``build_feature`` used to filter ``v > 0``, which looks like it excludes gap
    fills and does not: it drops only the no_signal zeros. A filled peak that
    lands on real intensity would walk straight into the mean and the CV.
    """
    detected = {
        "s1": _feature(feature_id="a", ms1_height=1000.0, ms1_area=500.0),
        "s2": _feature(feature_id="b", ms1_height=1200.0, ms1_area=600.0),
    }
    before = build_feature(_spot_with(dict(detected)), "F00000")

    with_fill = dict(detected)
    with_fill["s3"] = _feature(
        feature_id="c", ms1_height=400.0, ms1_area=200.0,
        gap_fill_status=FILLED,     # nonzero, and NOT detected
    )
    after = build_feature(_spot_with(with_fill), "F00000")

    assert after.mean_height == pytest.approx(before.mean_height)
    assert after.mean_area == pytest.approx(before.mean_area)
    assert after.cv == pytest.approx(before.cv)
    assert after.sn_ratio == pytest.approx(before.sn_ratio)
    # ...but the quantitation matrix does carry it.
    assert after.heights["s3"] == 400.0
    assert after.areas["s3"] == 200.0
    assert after.gap_fill_status == {"s1": "detected", "s2": "detected", "s3": FILLED}


def test_a_gap_filled_peak_never_represents_a_spot():
    # Even a filled peak that outscores every detected one: it carries no MS2 and
    # no annotation, and detected_peaks gates it out first.
    spot = _spot_with({
        "s1": _feature(feature_id="a", ms1_height=10.0),
        "s2": _feature(feature_id="b", ms1_height=99999.0, gap_fill_status=FILLED),
    })
    assert [p.feature_id for p in spot.detected_peaks] == ["a"]
    assert spot.n_detected == 1


def test_make_filled_peak_strips_identity_but_keeps_quantitation():
    template = _feature(matchms_name="caffeine", matchms_score=0.9)
    template.annotation_matches = [object()]
    result = GapFillResult(status=FILLED, height=42.0, area=7.0, rt_apex=5.0,
                           rt_left=4.9, rt_right=5.1, sn_ratio=2.0)
    filled = make_filled_peak(template, "s3", _target(), result)

    assert filled.gap_fill_status == FILLED
    assert filled.ms1_height == 42.0 and filled.ms1_area == 7.0
    assert filled.n_fragments == 0 and len(filled.ms2_mz) == 0
    assert filled.annotation_matches == [] and filled.matchms_name is None
    assert filled.segment_name == template.segment_name   # identity that is safe
    assert template.n_fragments == 2                      # template untouched


def test_fill_spot_only_touches_the_missing_samples():
    spot = _spot_with({"s1": _feature(feature_id="a")})
    rt, intensity = _gaussian(5.0)
    calls = []

    def provider(sample_id, target):
        calls.append(sample_id)
        return (rt, intensity) if sample_id == "s2" else None

    results = fill_spot(spot, ["s1", "s2", "s3"], provider, CFG)

    assert calls == ["s2", "s3"]
    assert results["s2"].status == FILLED
    assert results["s3"].status == NO_SIGNAL       # provider had no chromatogram
    assert spot.peaks["s1"].gap_fill_status == "detected"
    assert set(spot.peaks) == {"s1", "s2", "s3"}
