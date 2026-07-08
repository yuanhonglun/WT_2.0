"""Stage 1b ``ms2_deconv`` dispatch tests (档 C MSDec vs 档 B apex).

Mirrors the ``peak_detector`` dispatch: ``apex`` (default, 档 B, rollback-able)
keeps the apex-alignment + eic_coelution_ok + merge_close_ions path;
``msdec`` (档 C) routes the product-ion EIC matrix through
``metabo_core.algorithms.msdec.deconvolute_ms2`` and does NOT re-centroid the
output (no merge_close_ions — handoff pitfall #11).
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import RawSegmentData, ScanCycle
from asfam.pipeline.stage1b_ms1_detection import _collect_ms2_at_peak


def _gaussian(rt, center, sigma, amp):
    return amp * np.exp(-0.5 * ((rt - center) / sigma) ** 2)


def _build_raw_segment(rt, channel, ms1_intensity, ms2_ion_table):
    n_cycles = len(rt)
    cycles = []
    for ci in range(n_cycles):
        ms1_mz_arr = np.array([channel + 0.0001], dtype=np.float64)
        ms1_int_arr = np.array([float(ms1_intensity[ci])], dtype=np.float64)
        prod_mzs, prod_ints = [], []
        for ion_mz, ion_eic in ms2_ion_table.items():
            val = float(ion_eic[ci])
            if val > 0:
                prod_mzs.append(ion_mz)
                prod_ints.append(val)
        cycles.append(
            ScanCycle(
                cycle_index=ci,
                rt=float(rt[ci]),
                ms1_mz=ms1_mz_arr,
                ms1_intensity=ms1_int_arr,
                ms2_scans={channel: (
                    np.array(prod_mzs, dtype=np.float64),
                    np.array(prod_ints, dtype=np.float64),
                )},
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


def _coeluting_segment():
    rt = np.arange(60) * 0.005
    channel = 200
    ms1 = _gaussian(rt, center=0.15, sigma=0.04, amp=10000.0)  # apex cycle 30
    ion1 = _gaussian(rt, 0.15, 0.04, 6000.0)
    ion2 = _gaussian(rt, 0.15, 0.04, 4000.0)
    ion3 = _gaussian(rt, 0.15, 0.04, 2000.0)
    raw = _build_raw_segment(
        rt, channel, ms1, {100.05: ion1, 150.10: ion2, 200.15: ion3}
    )
    return raw, channel, ms1, rt


def test_msdec_mode_routes_to_deconvolute_ms2():
    raw, channel, ms1, rt = _coeluting_segment()
    cfg = ProcessingConfig()
    cfg.ms2_deconv = "msdec"
    sentinel = (np.array([100.05, 200.15]), np.array([6000.0, 2000.0]))
    with patch(
        "asfam.pipeline.stage1b_ms1_detection.deconvolute_ms2",
        return_value=sentinel,
    ) as m:
        out_mz, out_int = _collect_ms2_at_peak(
            raw, channel, 30, 10, 50, cfg, ms1_eic=ms1, rt_array=rt
        )
    m.assert_called_once()
    # The dispatch returns the deconvolution output as-is (no merge_close_ions).
    np.testing.assert_array_equal(out_mz, sentinel[0])
    np.testing.assert_array_equal(out_int, sentinel[1])


def test_apex_mode_does_not_call_deconvolute_ms2():
    raw, channel, ms1, rt = _coeluting_segment()
    cfg = ProcessingConfig()
    cfg.ms2_deconv = "apex"  # explicit 档 B (default is now msdec)
    with patch(
        "asfam.pipeline.stage1b_ms1_detection.deconvolute_ms2"
    ) as m:
        _collect_ms2_at_peak(
            raw, channel, 30, 10, 50, cfg, ms1_eic=ms1, rt_array=rt
        )
    m.assert_not_called()


def test_msdec_mode_recovers_coeluting_fragments_end_to_end():
    raw, channel, ms1, rt = _coeluting_segment()
    cfg = ProcessingConfig()
    cfg.ms2_deconv = "msdec"
    out_mz, out_int = _collect_ms2_at_peak(
        raw, channel, 30, 10, 50, cfg, ms1_eic=ms1, rt_array=rt
    )
    # Multiple co-eluting fragments are recovered by the real MSDec path.
    assert out_mz.size >= 2
    assert out_int.size == out_mz.size
    assert np.all(out_int > 0)


def test_msdec_product_ion_set_is_apex_scan_not_interval_union():
    # MS-DIAL Ms2Dec uses the precursor apex *single-scan* centroid spectrum
    # as the product-ion set, NOT the whole peak-interval union (which is
    # dominated by cross-scan noise and would explode the spectrum). Extra
    # ions that appear only OFF the apex cycle must not enter the product set.
    rt = np.arange(60) * 0.005
    channel = 200
    ms1 = _gaussian(rt, 0.15, 0.04, 10000.0)  # apex cycle 30
    ion1 = _gaussian(rt, 0.15, 0.04, 6000.0)
    ion2 = _gaussian(rt, 0.15, 0.04, 4000.0)
    ion3 = _gaussian(rt, 0.15, 0.04, 2000.0)
    extra_left = np.zeros_like(rt)
    extra_left[18:23] = 5000.0   # 300.30, present only at cycles 18-22
    extra_right = np.zeros_like(rt)
    extra_right[38:43] = 5000.0  # 400.40, present only at cycles 38-42
    raw = _build_raw_segment(rt, channel, ms1, {
        100.05: ion1, 150.10: ion2, 200.15: ion3,
        300.30: extra_left, 400.40: extra_right,
    })
    cfg = ProcessingConfig()
    cfg.ms2_deconv = "msdec"
    with patch(
        "asfam.pipeline.stage1b_ms1_detection.deconvolute_ms2",
        return_value=(np.zeros(0), np.zeros(0)),
    ) as m:
        _collect_ms2_at_peak(
            raw, channel, 30, 10, 50, cfg, ms1_eic=ms1, rt_array=rt
        )
    m.assert_called_once()
    fed_mzs = sorted(round(float(x), 2) for x in m.call_args.args[0])
    # Only the 3 ions present at the apex cycle 30 — not 300.30 / 400.40.
    assert fed_mzs == [100.05, 150.10, 200.15], fed_mzs


def test_msdec_falls_back_to_apex_scan_spectrum_when_deconvolution_fails():
    # MS-DIAL Ms2Dec returns GetDefaultMSDecResult (the apex-scan centroid
    # spectrum) when MSDec yields null (no model peak / apex not linked),
    # NOT an empty spectrum. Here the product ions are single spikes at the
    # apex cycle only — VS1 finds no model peak → deconvolution fails → the
    # apex-scan spectrum must still be returned.
    rt = np.arange(40) * 0.005
    channel = 200
    ms1 = _gaussian(rt, 0.10, 0.03, 10000.0)  # apex ~cycle 20
    spike1 = np.zeros_like(rt); spike1[20] = 5000.0
    spike2 = np.zeros_like(rt); spike2[20] = 3000.0
    spike3 = np.zeros_like(rt); spike3[20] = 1000.0
    raw = _build_raw_segment(
        rt, channel, ms1, {100.05: spike1, 150.10: spike2, 200.15: spike3}
    )
    cfg = ProcessingConfig()
    cfg.ms2_deconv = "msdec"
    out_mz, out_int = _collect_ms2_at_peak(
        raw, channel, 20, 14, 26, cfg, ms1_eic=ms1, rt_array=rt
    )
    got = sorted(round(m, 2) for m in out_mz.tolist())
    assert got == [100.05, 150.10, 200.15]
    assert out_int.size == out_mz.size and np.all(out_int > 0)


def test_apex_mode_still_reachable_and_returns_fragments():
    # 档 B path must remain reachable / unchanged (rollback mode).
    raw, channel, ms1, rt = _coeluting_segment()
    cfg = ProcessingConfig()
    cfg.ms2_deconv = "apex"
    cfg.msms_relative_threshold = 0.0
    out_mz, out_int = _collect_ms2_at_peak(
        raw, channel, 30, 10, 50, cfg, ms1_eic=ms1, rt_array=rt
    )
    assert out_mz.size >= 1
