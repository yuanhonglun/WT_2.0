"""mzML file parser for ASFAM data."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pymzml

from asfam.models import ScanCycle, RawSegmentData
from asfam.config import ProcessingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_filename(filepath: str, pattern: Optional[str] = None) -> dict:
    """Parse mzML filename to extract segment range and replicate ID.

    Tries multiple common naming conventions. If none match, falls back
    to extracting any N-M number range from the filename.

    Returns dict with keys: sample, seg_low, seg_high, step, rep
    """
    name = Path(filepath).stem

    # If user provided a custom pattern, try it first
    if pattern:
        m = re.match(pattern, name)
        if m:
            return _extract_groups(m)

    # Common patterns to try (ordered from most to least specific)
    patterns = [
        # MIX_100-129_1_30_1  (sample_seglow-seghigh_X_step_rep)
        r"^(?P<sample>.+?)_(?P<seg_low>\d+)-(?P<seg_high>\d+)_\d+_(?P<step>\d+)_(?P<rep>\d+)$",
        # CK1_075-110_1  (sample_seglow-seghigh_rep)
        r"^(?P<sample>.+?)_(?P<seg_low>\d+)-(?P<seg_high>\d+)_(?P<rep>\d+)$",
        # Sample_100-129  (sample_seglow-seghigh, no replicate)
        r"^(?P<sample>.+?)_(?P<seg_low>\d+)-(?P<seg_high>\d+)$",
    ]

    for pat in patterns:
        m = re.match(pat, name)
        if m:
            return _extract_groups(m)

    # Fallback: find any N-M range in the filename
    m = re.search(r"(\d+)-(\d+)", name)
    if m:
        seg_low = int(m.group(1))
        seg_high = int(m.group(2))
        # Extract sample name as everything before the range
        prefix = name[:m.start()].rstrip("_- ")
        # Try to find replicate number after the range
        suffix = name[m.end():]
        rep_match = re.search(r"(\d+)", suffix)
        rep = int(rep_match.group(1)) if rep_match else 1
        return {
            "sample": prefix or "Sample",
            "seg_low": seg_low,
            "seg_high": seg_high,
            "step": 0,
            "rep": rep,
        }

    raise ValueError(
        f"Cannot parse filename: {name}\n"
        f"Expected pattern like: SAMPLE_100-129_1.mzML or SAMPLE_100-129_1_30_1.mzML\n"
        f"The filename must contain a number range (e.g. 100-129) indicating the m/z segment."
    )


def _extract_groups(m: re.Match) -> dict:
    """Extract named groups from a regex match, with defaults."""
    d = m.groupdict()
    return {
        "sample": d.get("sample", "Sample"),
        "seg_low": int(d["seg_low"]),
        "seg_high": int(d["seg_high"]),
        "step": int(d.get("step", 0) or 0),
        "rep": int(d.get("rep", 1) or 1),
    }


# ---------------------------------------------------------------------------
# mzML loading
# ---------------------------------------------------------------------------

def load_mzml(filepath: str, config: Optional[ProcessingConfig] = None) -> RawSegmentData:
    """Load a single mzML file into a RawSegmentData object.

    Streams through spectra sequentially. MS2 spectra are centroided
    on the fly.
    """
    filepath = str(filepath)
    info = parse_filename(filepath)
    seg_low = info["seg_low"]
    seg_high = info["seg_high"]
    n_precursors = seg_high - seg_low + 1  # e.g. 30 for 100-129
    # Note: the actual precursor range may differ. In the demo data,
    # 100-129 means channels 100..129 -> 30 channels (but segment_high
    # might be inclusive giving 30 channels).
    precursor_list = list(range(seg_low, seg_high + 1))

    # Parse all spectra sequentially.
    # The IDs are simple integers 1..31 repeating per cycle.
    # Strategy: count sequential spectra. Each cycle = 1 MS1 + N_precursors MS2.
    spectra_per_cycle = 1 + n_precursors  # e.g. 31

    cycles_dict: dict[int, dict] = {}
    collision_energy = 0.0
    spec_counter = 0

    run = pymzml.run.Reader(filepath)
    for spectrum in run:
        ms_level = spectrum.ms_level
        if ms_level is None:
            spec_counter += 1
            continue

        cycle_idx = spec_counter // spectra_per_cycle
        pos_in_cycle = spec_counter % spectra_per_cycle

        if cycle_idx not in cycles_dict:
            cycles_dict[cycle_idx] = {"ms1_mz": None, "ms1_int": None,
                                       "rt": 0.0, "ms2": {}}

        # Extract RT from first scan of cycle (regardless of MS level)
        if pos_in_cycle == 0:
            try:
                cycles_dict[cycle_idx]["rt"] = float(spectrum.scan_time_in_minutes())
            except Exception:
                pass

        # MS1: accept at ANY position in the cycle (not just pos=0)
        if ms_level == 1:
            mz_arr, int_arr = _get_spectrum_arrays(spectrum)
            # Keep the MS1 with the most data points (some cycles have multiple MS1)
            if cycles_dict[cycle_idx]["ms1_mz"] is None or len(mz_arr) > len(cycles_dict[cycle_idx]["ms1_mz"]):
                cycles_dict[cycle_idx]["ms1_mz"] = mz_arr
                cycles_dict[cycle_idx]["ms1_int"] = int_arr
                if cycles_dict[cycle_idx]["rt"] == 0.0:
                    try:
                        cycles_dict[cycle_idx]["rt"] = float(spectrum.scan_time_in_minutes())
                    except Exception:
                        pass

        if ms_level == 2:
            # Get precursor m/z directly from spectrum metadata
            prec_mz_val = _get_precursor_mz(spectrum)
            if prec_mz_val is not None:
                prec_mz = int(round(prec_mz_val))
                mz_arr, int_arr = _get_spectrum_arrays(spectrum)
                mz_arr, int_arr = _centroid_if_needed(mz_arr, int_arr)
                cycles_dict[cycle_idx]["ms2"][prec_mz] = (mz_arr, int_arr)

                if collision_energy == 0.0:
                    ce = _get_collision_energy(spectrum)
                    if ce > 0:
                        collision_energy = ce

        spec_counter += 1

    # Build ScanCycle list sorted by cycle index
    sorted_indices = sorted(cycles_dict.keys())
    cycles = []
    rt_values = []
    for idx in sorted_indices:
        cd = cycles_dict[idx]
        ms1_mz = cd["ms1_mz"] if cd["ms1_mz"] is not None else np.array([], dtype=np.float64)
        ms1_int = cd["ms1_int"] if cd["ms1_int"] is not None else np.array([], dtype=np.float64)
        cycles.append(ScanCycle(
            cycle_index=idx,
            rt=cd["rt"],
            ms1_mz=ms1_mz,
            ms1_intensity=ms1_int,
            ms2_scans=cd["ms2"],
        ))
        rt_values.append(cd["rt"])

    segment_name = f"{seg_low}-{seg_high}"
    logger.info(
        "Loaded %s: %d cycles, %d channels, CE=%.1f eV",
        Path(filepath).name, len(cycles), len(precursor_list), collision_energy,
    )

    return RawSegmentData(
        file_path=filepath,
        segment_name=segment_name,
        segment_low=seg_low,
        segment_high=seg_high,
        replicate_id=info["rep"],
        n_cycles=len(cycles),
        rt_array=np.array(rt_values, dtype=np.float64),
        precursor_list=precursor_list,
        cycles=cycles,
        collision_energy=collision_energy,
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_precursor_mz(spectrum) -> Optional[float]:
    """Extract precursor m/z from MS2 spectrum metadata."""
    try:
        precursors = spectrum.selected_precursors
        if precursors and len(precursors) > 0:
            return float(precursors[0]["mz"])
    except (KeyError, TypeError, IndexError):
        pass
    try:
        iso_target = spectrum.get("MS:1000827")  # isolation window target m/z
        if iso_target is not None:
            return float(iso_target)
    except (KeyError, TypeError):
        pass
    return None


def _get_spectrum_arrays(spectrum) -> tuple[np.ndarray, np.ndarray]:
    """Extract m/z and intensity arrays from a pymzml spectrum."""
    peaks = spectrum.peaks("raw")
    if peaks is None or len(peaks) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    mz_arr = np.asarray(peaks[:, 0], dtype=np.float64)
    int_arr = np.asarray(peaks[:, 1], dtype=np.float64)
    return mz_arr, int_arr


def _centroid_if_needed(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    min_intensity: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Centroid profile data using local maximum detection.

    For each local maximum, compute intensity-weighted mean m/z
    using the 3 points around the apex.
    """
    if len(mz_array) < 3:
        return mz_array, intensity_array

    # Check if already centroided (few points or large gaps)
    if len(mz_array) < 20:
        mask = intensity_array >= min_intensity
        return mz_array[mask], intensity_array[mask]

    # Find local maxima
    centroid_mz = []
    centroid_int = []
    for i in range(1, len(intensity_array) - 1):
        if (intensity_array[i] > intensity_array[i - 1] and
                intensity_array[i] >= intensity_array[i + 1] and
                intensity_array[i] >= min_intensity):
            # Intensity-weighted centroid of 3 points
            local_mz = mz_array[i - 1:i + 2]
            local_int = intensity_array[i - 1:i + 2]
            total = np.sum(local_int)
            if total > 0:
                centroid_mz.append(float(np.sum(local_mz * local_int) / total))
                centroid_int.append(float(intensity_array[i]))

    if not centroid_mz:
        mask = intensity_array >= min_intensity
        return mz_array[mask], intensity_array[mask]

    return np.array(centroid_mz, dtype=np.float64), np.array(centroid_int, dtype=np.float64)


def _get_collision_energy(spectrum) -> float:
    """Extract collision energy from spectrum metadata."""
    try:
        ce = spectrum.get("MS:1000045")  # collision energy CV param
        if ce is not None:
            return float(ce)
    except (KeyError, TypeError, ValueError):
        pass
    return 0.0
