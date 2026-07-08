"""Compatibility shim + peak-detector router.

Re-exports the shared metabo_core ``detect_peaks`` and adds
``detect_chrom_peaks``, a thin router that dispatches a single-EIC
chromatogram peak detection to either the metra detector (``detect_peaks``)
or the faithful MS-DIAL derivative engine, based on ``config.peak_detector``.
"""
from dataclasses import replace

from metabo_core.algorithms.peak_detection import detect_peaks  # noqa: F401
from metabo_core.algorithms.msdial_peak_spotting import (
    msdial_detect_peaks_in_chromatogram,
)


def detect_chrom_peaks(rt_array, intensity_array, *, config, **detect_kwargs):
    """Route a single-EIC chromatogram peak detection by config.peak_detector.

    The msdial branch HONOURS the caller's ``min_amplitude`` / ``min_data_points``
    so the per-site floor (stage1 MS2 = ms2_peak.min_amplitude=200, stage2-relaxed
    = 50, stage1b = ms1_min_height) is the SAME for both detectors — keeping the
    A/B about algorithm, not threshold. (Without this override the msdial branch
    would use MsdialPeakSpottingConfig's default 1000 and suppress MS2 peaks for a
    threshold reason, muddying the comparison.) The metra-only kwargs
    (gaussian_threshold / sn_fold / min_prominence_ratio / rt_window_*) are
    ignored by the msdial branch.

    The metra branch forwards ALL kwargs to ``detect_peaks`` unchanged, so metra
    output is byte-for-byte identical to calling ``detect_peaks`` directly.
    """
    if config.peak_detector == "msdial":
        mcfg = config.msdial_peak
        if "min_amplitude" in detect_kwargs:
            mcfg = replace(mcfg, min_amplitude=detect_kwargs["min_amplitude"])
        if "min_data_points" in detect_kwargs:
            mcfg = replace(mcfg, min_data_points=detect_kwargs["min_data_points"])
        return msdial_detect_peaks_in_chromatogram(
            rt_array, intensity_array, config=mcfg,
            precursor_mz_nominal=detect_kwargs.get("precursor_mz_nominal", 0),
            product_mz=detect_kwargs.get("product_mz", 0.0),
        )
    return detect_peaks(rt_array, intensity_array, **detect_kwargs)
