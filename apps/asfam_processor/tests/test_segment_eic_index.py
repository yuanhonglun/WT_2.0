"""``SegmentEicIndex``: the searchsorted SUM must equal the obvious per-cycle sum."""
from __future__ import annotations

import numpy as np
import pytest

from asfam.core.eic import SegmentEicIndex
from asfam.models import RawSegmentData


class _Cycle:
    def __init__(self, ms1_mz, ms1_intensity, ms2_scans=None):
        self.ms1_mz = None if ms1_mz is None else np.asarray(ms1_mz, dtype=float)
        self.ms1_intensity = (None if ms1_intensity is None
                              else np.asarray(ms1_intensity, dtype=float))
        self.ms2_scans = ms2_scans or {}


def _segment(cycles, precursors=(200,)) -> RawSegmentData:
    return RawSegmentData(
        file_path="x.mzML", segment_name="195-224", segment_low=195, segment_high=224,
        replicate_id=1, n_cycles=len(cycles),
        rt_array=np.arange(len(cycles), dtype=float) * 0.1,
        precursor_list=list(precursors), cycles=cycles,
    )


def _brute_ms1_sum(segment, mz, tol):
    out = np.zeros(segment.n_cycles)
    for i, c in enumerate(segment.cycles):
        if c.ms1_mz is None or len(c.ms1_mz) == 0:
            continue
        mask = np.abs(c.ms1_mz - mz) <= tol
        out[i] = c.ms1_intensity[mask].sum()
    return out


def test_ms1_sum_matches_a_per_cycle_loop_on_random_data():
    rng = np.random.default_rng(7)
    cycles = []
    for _ in range(40):
        mz = np.sort(rng.uniform(195.0, 225.0, 200))
        cycles.append(_Cycle(mz, rng.uniform(1.0, 1e4, 200)))
    segment = _segment(cycles)
    index = SegmentEicIndex(segment)

    for target in rng.uniform(196.0, 224.0, 25):
        for tol in (0.01, 0.1):
            np.testing.assert_allclose(
                index.ms1_eic_sum(target, tol),
                _brute_ms1_sum(segment, target, tol),
                rtol=1e-12, atol=1e-9,
            )


def test_sum_adds_co_isolated_centroids_rather_than_taking_the_base_peak():
    # The whole point of SUM over MAX: two centroids inside the window count once
    # each. LcmsGapFiller integrates Spectrum.RetrieveBin, which sums.
    cycles = [_Cycle([110.001, 110.007], [300.0, 700.0])]
    index = SegmentEicIndex(_segment(cycles))

    assert index.ms1_eic_sum(110.004, 0.01)[0] == pytest.approx(1000.0)
    assert index.ms1_eic_sum(110.001, 0.002)[0] == pytest.approx(300.0)


def test_window_bounds_are_inclusive():
    index = SegmentEicIndex(_segment([_Cycle([100.0, 100.02], [5.0, 7.0])]))
    assert index.ms1_eic_sum(100.01, 0.01)[0] == pytest.approx(12.0)
    assert index.ms1_eic_sum(100.01, 0.009)[0] == pytest.approx(0.0)


def test_cycles_with_no_ms1_stay_zero_and_do_not_shift_the_axis():
    cycles = [
        _Cycle([100.0], [5.0]),
        _Cycle(None, None),
        _Cycle([], []),
        _Cycle([100.0], [9.0]),
    ]
    eic = SegmentEicIndex(_segment(cycles)).ms1_eic_sum(100.0, 0.01)
    np.testing.assert_allclose(eic, [5.0, 0.0, 0.0, 9.0])


def test_product_eic_sums_only_its_own_channel():
    cycles = [
        _Cycle([200.0], [1.0], {200: (np.array([70.0]), np.array([11.0])),
                                201: (np.array([70.0]), np.array([99.0]))}),
        _Cycle([200.0], [1.0], {200: (np.array([70.0]), np.array([13.0]))}),
    ]
    index = SegmentEicIndex(_segment(cycles, precursors=(200, 201)))

    np.testing.assert_allclose(index.product_eic_sum(200, 70.0, 0.1), [11.0, 13.0])
    # Cycle 1 never acquired channel 201, so its slot is zero, not dropped.
    np.testing.assert_allclose(index.product_eic_sum(201, 70.0, 0.1), [99.0, 0.0])


def test_an_unacquired_channel_is_none_not_a_zero_trace():
    # A zero trace would gap-fill to "no_signal"; None means "we never looked",
    # which routes to a different segment or gives up.
    index = SegmentEicIndex(_segment([_Cycle([200.0], [1.0])], precursors=(200,)))
    assert index.product_eic_sum(999, 70.0, 0.1) is None


def test_the_product_channel_cache_evicts_but_keeps_answering():
    cycles = [
        _Cycle([200.0], [1.0], {ch: (np.array([70.0]), np.array([float(ch)]))
                                for ch in (200, 201, 202)}),
    ]
    index = SegmentEicIndex(_segment(cycles, precursors=(200, 201, 202)),
                            ms2_cache_size=1)
    for ch in (200, 201, 202, 200):
        assert index.product_eic_sum(ch, 70.0, 0.1)[0] == pytest.approx(float(ch))
