import numpy as np
from asfam.models import ScanCycle
from asfam.core.ms1_scan_adapter import ms1_survey_scans

def test_adapter_exposes_scan_like():
    cycles = [
        ScanCycle(0, 1.0, np.array([285.05, 286.1]), np.array([100.0, 50.0]), {}),
        ScanCycle(1, 1.1, np.array([285.05]), np.array([200.0]), {}),
    ]
    scans = ms1_survey_scans(cycles)
    assert len(scans) == 2
    assert scans[0].rt == 1.0
    assert scans[0].ms_level == 1
    np.testing.assert_allclose(scans[1].mz_array, [285.05])
    np.testing.assert_allclose(scans[0].intensity_array, [100.0, 50.0])

def test_adapter_tolerates_none_arrays():
    cycles = [ScanCycle(0, 1.0, None, None, {})]
    scans = ms1_survey_scans(cycles)
    assert scans[0].mz_array.dtype == np.float64
    assert len(scans[0].mz_array) == 0
    assert scans[0].intensity_array.dtype == np.float64
    assert len(scans[0].intensity_array) == 0
