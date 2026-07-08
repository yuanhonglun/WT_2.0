import numpy as np
from asfam.models import ScanCycle, RawSegmentData
from asfam.config import ProcessingConfig
from asfam.core.eic import extract_product_ion_eics_massslice


def _raw(cycles, channel=285):
    return RawSegmentData(
        file_path="x", segment_name="seg", segment_low=channel, segment_high=channel + 29,
        replicate_id=1, n_cycles=len(cycles),
        rt_array=np.array([c.rt for c in cycles]), precursor_list=[channel],
        cycles=cycles,
    )


def _peak_cycles(channel, ions_per_cycle):
    """ions_per_cycle: list of (mz_array, int_array) per cycle."""
    cyc = []
    for i, (mz, it) in enumerate(ions_per_cycle):
        cyc.append(ScanCycle(i, 1.0 + 0.05 * i, np.array([]), np.array([]),
                             {channel: (np.array(mz, float), np.array(it, float))}))
    return cyc


def test_massslice_builds_eics_with_basepeak_and_sum():
    # one fragment at 100.02 elutes over 6 cycles (rises/falls), plus a noisy
    # co-bin centroid at 100.05 in 2 cycles -> SUM in the +/-0.1 window
    ch = 285
    ions = [
        ([100.02], [50]),
        ([100.02, 100.05], [300, 40]),
        ([100.02, 100.05], [800, 60]),
        ([100.02], [500]),
        ([100.02], [120]),
        ([100.02], [30]),
    ]
    eics = extract_product_ion_eics_massslice(_raw(_peak_cycles(ch, ions), ch), ch,
                                              ProcessingConfig())
    assert eics, "expected at least one EIC"
    e = max(eics, key=lambda x: float(np.max(x.intensity_array)))
    assert e.basepeak_mz is not None and e.basepeak_mz.shape == e.intensity_array.shape
    assert abs(float(e.basepeak_mz[int(np.argmax(e.intensity_array))]) - 100.02) < 0.02
    # SUM: cycle index 2 window includes 800+60 = 860
    assert float(np.max(e.intensity_array)) >= 860.0 - 1e-6


def test_massslice_skips_above_precursor_plus_2():
    ch = 285
    # 150.0 is in range and must yield an EIC; 288.0 ( > channel+2 = 287 ) must not.
    # Putting both in the same channel proves the cap discriminates (not just
    # "an isolated out-of-range ion yields nothing").
    ions = [([150.0, 288.0], [900, 900])] * 5
    eics = extract_product_ion_eics_massslice(_raw(_peak_cycles(ch, ions), ch), ch,
                                              ProcessingConfig())
    assert all(e.product_mz <= ch + 2 + 0.1 for e in eics)
    assert any(abs(e.product_mz - 150.0) < 0.15 for e in eics), "in-range ion should survive"
    assert not any(abs(e.product_mz - 288.0) < 0.15 for e in eics)


def test_massslice_min_nonzero_gate():
    ch = 285
    # 120.0 present in only 2 cycles (< min_data_points=3) -> dropped;
    # 200.0 present in 3 cycles (>= 3) -> kept. Proves the gate discriminates.
    ions = [([120.0, 200.0], [900, 900]),
            ([120.0, 200.0], [900, 900]),
            ([200.0], [900]),
            ([], []),
            ([], [])]
    eics = extract_product_ion_eics_massslice(_raw(_peak_cycles(ch, ions), ch),
                                              ch, ProcessingConfig())
    assert any(abs(e.product_mz - 200.0) < 0.1 for e in eics), "3-scan ion should survive"
    assert not any(abs(e.product_mz - 120.0) < 0.1 for e in eics)
