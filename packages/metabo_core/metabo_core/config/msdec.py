"""MS-DIAL MSDec parameter surface for MS2 model-peak deconvolution.

Faithful port of the MS-DIAL ``ChromDecBaseParameter`` constants
(ParameterBase.cs ~L1207) plus the MSDec engine hardcoded values
(MSDecHandler.cs / MSDecProcess.cs / Ms2Dec.cs). The dataclass is consumed
by :func:`metabo_core.algorithms.msdec.deconvolute_ms2`.

All defaults reproduce MS-DIAL behaviour so the engine produces results
directly comparable to MS-DIAL without user tuning. ASFAM carries many weak
MS2 signals, so the app view lowers ``min_amplitude`` from the MS-DIAL
faithful floor (1000) to a weaker-signal floor; this is a deliberate,
tunable A/B knob, not a change to the core default.

The dataclass mirrors :class:`MsdialPeakSpottingConfig` (not frozen) so the
ASFAM view can ``replace()`` / assign fields per run.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MsdecConfig:
    """Parameter surface for the MS-DIAL-style MSDec deconvolution engine.

    Field groups
    ------------
    Matched filter (model-peak grouping)
        sigma_window, matched_filter_half_point, region_margin

    MS2 product-ion centroiding / curation
        centroid_ms2_tolerance, amplitude_cutoff, relative_amplitude_cutoff,
        kept_isotope_range, remove_after_precursor

    Model-peak detection (VS1)
        min_data_points, min_amplitude, smoothing_level, average_peak_width

    Model selection / least-squares association
        sharpness_inclusion_fraction, ideal_slope_high, ideal_slope_middle,
        apex_model_tolerance, max_neighbor_models, model_min_edge_points
    """

    # ----- Matched filter (Mexican-hat, model-peak-top grouping) -----
    # Mexican-hat sigma controlling how close peak tops merge into one
    # component (MS-DIAL SigmaWindowValue).
    sigma_window: float = 0.5
    # Half-length of the fixed matched-filter kernel (kernel length =
    # 2*half+1 = 21); MS-DIAL hardcodes this independently of sigma.
    matched_filter_half_point: int = 10
    # Region detection ignores the first/last ``region_margin`` scans.
    region_margin: int = 5

    # ----- MS2 product-ion centroiding / curation -----
    # Product-ion m/z binning tolerance in Da (MS-DIAL CentroidMs2Tolerance).
    centroid_ms2_tolerance: float = 0.025
    # Absolute product-ion intensity floor (MS-DIAL AmplitudeCutoff).
    amplitude_cutoff: float = 0.0
    # Relative product-ion intensity floor (MS-DIAL RelativeAmplitudeCutoff).
    relative_amplitude_cutoff: float = 0.0
    # RemoveAfterPrecursor cut window in Da: drop product m/z greater than
    # precursorMz + kept_isotope_range (MS-DIAL KeptIsotopeRange).
    kept_isotope_range: float = 5.0
    # Whether to drop product ions above the precursor + kept_isotope_range.
    remove_after_precursor: bool = True

    # ----- Model-peak detection (VS1) -----
    # Minimum model-peak width in scans (MS-DIAL MinimumDatapoints).
    min_data_points: int = 5
    # Minimum model-peak height (MS-DIAL MinimumAmplitude). 1000 is the
    # MS-DIAL faithful floor; the ASFAM view lowers this to retain weak MS2.
    min_amplitude: float = 1000.0
    # LWMA smoothing level for product-ion chromatograms (MS-DIAL level 3).
    smoothing_level: int = 3
    # Edge-walk half-width uses average_peak_width * 0.5 (=15 by default)
    # when refining a model chromatogram (MS-DIAL AveragePeakWidth).
    average_peak_width: int = 30

    # ----- Model selection / least-squares association -----
    # An ion joins a model chromatogram if its sharpness >= this fraction of
    # the region's max sharpness (MS-DIAL 0.9 * max).
    sharpness_inclusion_fraction: float = 0.9
    # IdealSlope threshold above which a model peak is High quality.
    ideal_slope_high: float = 0.999
    # IdealSlope threshold above which a model peak is Middle quality.
    ideal_slope_middle: float = 0.9
    # Max |precursor apex - model apex| (in scans) allowed to run
    # deconvolution; otherwise fall back to the raw spectrum (MS-DIAL <=2).
    apex_model_tolerance: int = 2
    # Maximum number of overlapping neighbour models in the design matrix
    # (MS-DIAL allows 2 left + 2 right = 4); the engine currently implements
    # the fixed 2L2R pattern set.
    max_neighbor_models: int = 4
    # A refined model chromatogram is rejected if either side has fewer than
    # this many points around the apex (MS-DIAL <3).
    model_min_edge_points: int = 3


def lc_msdec_config() -> MsdecConfig:
    """Return a fresh MsdecConfig with MS-DIAL MSDec defaults."""
    return MsdecConfig()
