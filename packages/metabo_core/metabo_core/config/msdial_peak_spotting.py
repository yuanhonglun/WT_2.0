"""MS-DIAL peak-spotting parameter surface for LC-MS feature extraction.

Mirrors the parameter surface described in §6.8 of the MS-DIAL peak detection
port spec (2026-06-25). The dataclass is consumed by:

  - Task 1.2 — the derivative-based peak detection engine (TrackB)
  - Task 2.2 — the MS1 mass-slice orchestrator

All defaults reproduce MS-DIAL LC-mode hardcoded values so that the engine
produces results comparable to MS-DIAL without any user tuning.

The LWMA smoothing *method* is fixed (7-point kernel {1,2,3,4,3,2,1}/16);
no field is needed for it. The area ×60 min→sec conversion is engine
behaviour, not a config field.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MsdialPeakSpottingConfig:
    """Parameter surface for the MS-DIAL-style LC-MS peak-spotting pipeline.

    Field groups
    ------------
    Mass slicing
        mass_slice_width, mass_range_begin, mass_range_end

    Smoothing
        smoothing_level

    Peak criteria (absolute)
        min_data_points, min_amplitude

    Centroiding
        centroid_ms1_tolerance

    Noise estimation
        noise_factor, noise_bin_size, min_noise_windows, baseline_window,
        slope_noise_small_diff_frac

    Gate thresholds (relative to noise)
        amplitude_noise_fold, slope_noise_fold

    Peak-shape / artefact rejection
        average_peak_width, background_spike_threshold

    Redundancy / deduplication
        adjacent_redundancy_mass_tol, adjacent_redundancy_apex_tol,
        global_dedup_mass_tol, global_dedup_rt_tol
    """

    # ----- Mass slicing -----
    # EIC half-window (±) AND slice step in Da (MS-DIAL MassSliceWidth).
    mass_slice_width: float = 0.1
    # Lower m/z bound for slicing (MS-DIAL MassRangeBegin).
    mass_range_begin: float = 0.0
    # Upper m/z bound for slicing (MS-DIAL MassRangeEnd).
    mass_range_end: float = 2000.0

    # ----- Smoothing -----
    # LWMA smoothing level — kernel size = 2*level+1 (7-point at level=3,
    # weights {1,2,3,4,3,2,1}/16).
    smoothing_level: int = 3

    # ----- Peak criteria (absolute) -----
    # Minimum peak width in scans (MS-DIAL MinimumDatapoints).
    min_data_points: int = 5
    # Minimum peak height above local baseline (MS-DIAL MinimumAmplitude).
    min_amplitude: float = 1000.0

    # ----- Centroiding -----
    # Coarse→fine m/z recalculation tolerance in Da (MS-DIAL centroid recalc
    # step uses this window around the slice centre).
    centroid_ms1_tolerance: float = 0.01

    # ----- Noise estimation -----
    # Multiplier applied to the per-bin noise estimate: Noise = binNoise × NoiseFactor.
    noise_factor: float = 3.0
    # Noise-estimate bin size in scans (number of points per noise window).
    noise_bin_size: int = 50
    # Minimum number of bins required for a valid noise estimate
    # (MS-DIAL MinimumNoiseWindowSize).
    min_noise_windows: int = 10
    # LWMA window for the baseline-subtraction trace used in noise estimation only.
    baseline_window: int = 20
    # Slope-noise candidate threshold: nonzero diffs < this fraction of the
    # maximum absolute diff are used as the slope-noise sample.
    slope_noise_small_diff_frac: float = 0.05

    # ----- Gate thresholds (relative to noise) -----
    # Amplitude gate: minPeakHeight ≥ amplitude_noise_fold × amplitudeNoise.
    amplitude_noise_fold: float = 4.0
    # Slope gate: start/stop slope ≥ slope_noise_fold × slopeNoise.
    slope_noise_fold: float = 2.0

    # ----- Peak-shape / artefact rejection -----
    # ShrinkPeakRange tail-trim window in scans (MS-DIAL AveragePeakWidth).
    average_peak_width: int = 20
    # Reject a peak if the count of side spikes ≥ this threshold
    # (MS-DIAL BackgroundSubtract spike gate).
    background_spike_threshold: int = 15

    # ----- Redundancy / deduplication -----
    # Adjacent-slice same-peak mass tolerance in Da (= massStep × 0.5).
    adjacent_redundancy_mass_tol: float = 0.05
    # Adjacent-slice apex-RT cap in minutes: min(hwhm, adjacent_redundancy_apex_tol).
    adjacent_redundancy_apex_tol: float = 0.03
    # Global near-duplicate mass tolerance in Da.
    global_dedup_mass_tol: float = 0.005
    # Global near-duplicate RT tolerance in minutes.
    global_dedup_rt_tol: float = 0.03


def lc_msdial_config() -> MsdialPeakSpottingConfig:
    """Return a fresh MsdialPeakSpottingConfig with MS-DIAL LC defaults."""
    return MsdialPeakSpottingConfig()
