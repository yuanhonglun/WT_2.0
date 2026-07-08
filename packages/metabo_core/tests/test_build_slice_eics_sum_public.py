import numpy as np
from dataclasses import dataclass

from metabo_core.algorithms.msdial_ms1_features import build_slice_eics_sum
from metabo_core.config.msdial_peak_spotting import MsdialPeakSpottingConfig
from metabo_core.models.chromatography import ProductIonEIC


@dataclass
class _Scan:
    rt: float
    mz_array: np.ndarray
    intensity_array: np.ndarray


def test_build_slice_eics_sum_is_public_and_sums():
    # two cycles, one ion at 100.00 present in both, SUM per scan
    scans = [
        _Scan(1.0, np.array([100.00, 100.04]), np.array([10.0, 5.0])),
        _Scan(1.1, np.array([100.00]), np.array([20.0])),
    ]
    cfg = MsdialPeakSpottingConfig()  # mass_slice_width=0.1
    slices = build_slice_eics_sum(scans, cfg)
    # there is at least one slice covering ~100.0 with SUM (10+5=15) at scan0
    assert slices, "expected non-empty slices"
    centers = [s[0] for s in slices]
    near = [s for s in slices if abs(s[0] - 100.0) <= 0.1]
    assert near, f"no slice near 100.0 in {centers}"
    center, basepeak, eic, scan_idx = near[0]
    # scan0 SUM = 10+5 = 15 (both centroids within +/-0.1 of 100.0)
    assert 0 in scan_idx
    assert eic[list(scan_idx).index(0)] == 15.0


def test_product_ion_eic_has_basepeak_field():
    e = ProductIonEIC(precursor_mz_nominal=285, product_mz=100.0,
                      rt_array=np.array([1.0]), intensity_array=np.array([5.0]))
    assert e.basepeak_mz is None  # default
    e2 = ProductIonEIC(precursor_mz_nominal=285, product_mz=100.0,
                       rt_array=np.array([1.0]), intensity_array=np.array([5.0]),
                       basepeak_mz=np.array([100.0]))
    np.testing.assert_allclose(e2.basepeak_mz, [100.0])
