import numpy as np
from asfam.models import ScanCycle, RawSegmentData
from asfam.core.ms2_scan_adapter import ms2_channel_scans


def _raw(cycles):
    return RawSegmentData(
        file_path="x", segment_name="seg", segment_low=285, segment_high=314,
        replicate_id=1, n_cycles=len(cycles),
        rt_array=np.array([c.rt for c in cycles]), precursor_list=[285],
        cycles=cycles,
    )


def test_channel_scans_align_with_cycles_and_fill_empty():
    cycles = [
        ScanCycle(0, 1.0, np.array([]), np.array([]),
                  {285: (np.array([100.0, 150.0]), np.array([500.0, 5.0]))}),
        ScanCycle(1, 1.1, np.array([]), np.array([]), {}),  # channel missing
        ScanCycle(2, 1.2, np.array([]), np.array([]),
                  {285: (np.array([100.0]), np.array([600.0]))}),
    ]
    scans = ms2_channel_scans(_raw(cycles), 285)
    assert len(scans) == 3                       # one per cycle (scan_idx == cycle_idx)
    assert scans[0].rt == 1.0
    np.testing.assert_allclose(scans[0].mz_array, [100.0, 150.0])
    assert scans[1].mz_array.size == 0           # empty placeholder for missing channel
    np.testing.assert_allclose(scans[2].intensity_array, [600.0])
