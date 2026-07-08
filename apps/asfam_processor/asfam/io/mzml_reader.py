"""mzML file parser for ASFAM data.

This reader keeps two ASFAM-specific behaviours that the shared
``metabo_core.io.mzml`` does not provide:

1. **周期切片**：ASFAM 把 mzML 看成 ``1 MS1 + N MS2`` 的周期循环，
   每个周期切成一个 :class:`ScanCycle`。
2. **MS1 选数据点最多的**：同一周期内可能出现多张 MS1 spectrum，本
   reader 保留数据点最多的那张作为该周期的 MS1。

底层的 ``RT 提取``、``m/z + intensity 数组提取`` 复用
:mod:`metabo_core.io.mzml` 中的辅助函数；centroid 与 collision-energy
解析仍保留为 ASFAM 本地实现，因为它们是 ASFAM 流程特有的。
"""
from __future__ import annotations

import logging
import math
import re
import urllib.request  # noqa: F401 — must be imported before pymzml for PyInstaller
from pathlib import Path
from typing import Optional

import numpy as np
import pymzml

from asfam.models import ScanCycle, RawSegmentData
from asfam.config import ProcessingConfig
from metabo_core.io.mzml import _scan_rt_minutes, _spectrum_arrays

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
    # Nominal channel range from the filename; used only as a fallback for
    # ``precursor_list`` when a file carries no MS2. The authoritative list is
    # rebuilt from the actually-acquired isolation windows after parsing (see
    # ``precursor_targets`` below).
    precursor_list = list(range(seg_low, seg_high + 1))

    # Parse all spectra sequentially.
    # The IDs are simple integers 1..31 repeating per cycle.
    # Strategy: count sequential spectra. Each cycle = 1 MS1 + N_precursors MS2.
    spectra_per_cycle = 1 + n_precursors  # e.g. 31

    # floor-key -> actual acquired isolation-target m/z (first occurrence;
    # the target is method-defined and stable across cycles). Rebuilds the
    # true acquired-window list and lets stage1b assign each MS1-driven
    # feature to the window whose target is nearest its precursor m/z.
    precursor_targets: dict[int, float] = {}

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
            rt_val = _scan_rt_minutes(spectrum)
            if rt_val:
                cycles_dict[cycle_idx]["rt"] = rt_val

        # MS1: accept at ANY position in the cycle (not just pos=0)
        if ms_level == 1:
            mz_arr, int_arr = _spectrum_arrays(spectrum)
            # Keep the MS1 with the most data points (some cycles have multiple MS1)
            if cycles_dict[cycle_idx]["ms1_mz"] is None or len(mz_arr) > len(cycles_dict[cycle_idx]["ms1_mz"]):
                cycles_dict[cycle_idx]["ms1_mz"] = mz_arr
                cycles_dict[cycle_idx]["ms1_int"] = int_arr
                if cycles_dict[cycle_idx]["rt"] == 0.0:
                    rt_val = _scan_rt_minutes(spectrum)
                    if rt_val:
                        cycles_dict[cycle_idx]["rt"] = rt_val

        if ms_level == 2:
            # Get precursor m/z directly from spectrum metadata
            prec_mz_val = _get_precursor_mz(spectrum)
            if prec_mz_val is not None:
                # Key each 1-Da isolation window by floor(target), not
                # round(target). ASFAM targets sit at a fractional m/z (X.5 in
                # the high-m/z region); int(round()) collapses adjacent X.5
                # windows onto the same even integer via banker's rounding and
                # the assignment below then silently overwrites one of them
                # (measured: 83/930 windows, 8.9%, lost across the 31-segment
                # benchmark — up to half per segment above m/z 800). floor() is
                # injective for 1-Da-spaced windows, so every window survives
                # with key == its lower integer bound.
                prec_mz = int(math.floor(prec_mz_val))
                mz_arr, int_arr = _spectrum_arrays(spectrum)
                mz_arr, int_arr = _centroid_if_needed(mz_arr, int_arr)
                cycles_dict[cycle_idx]["ms2"][prec_mz] = (mz_arr, int_arr)
                precursor_targets.setdefault(prec_mz, float(prec_mz_val))

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

    # Authoritative precursor list = the floor-keys actually acquired (== the
    # ms2_scans keys). Falls back to the nominal filename range only if the
    # file carried no MS2 at all.
    if precursor_targets:
        precursor_list = sorted(precursor_targets)

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
        precursor_targets=precursor_targets,
    )


# ---------------------------------------------------------------------------
# ASFAM-specific helpers
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


def _looks_centroided(mz_array: np.ndarray) -> bool:
    """判定谱是否已是 centroid（稀疏、不该再质心化）。

    Profile 谱把每个峰密集采样，相邻 m/z 间距 ~0.001-0.02 Da；vendor-centroid
    谱是稀疏孤立峰，相邻间距通常 ~0.5-3 Da。用相邻间距**中位数**判定，对少量
    近同位素对稳健。ASFAM（ProteoWizard/ABI peak-picking）实测 MS2 中位间距
    ~1-3 Da，远大于阈值。
    """
    if mz_array.size < 3:
        return True
    gaps = np.diff(np.sort(mz_array))
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return True
    return float(np.median(gaps)) >= 0.1


def _centroid_profile(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    min_intensity: float,
    gap_threshold: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """对真 profile 数据做逐峰质心：按 m/z 间距 > gap_threshold 切分成峰，
    每个峰取**峰内**强度加权质心（不跨峰、不混不同离子）。峰内加权符合
    CLAUDE.md「单个真离子 profile 用强度加权」的不变量。
    """
    order = np.argsort(mz_array)
    smz = mz_array[order]
    sint = intensity_array[order]
    out_mz: list[float] = []
    out_int: list[float] = []
    n = smz.size
    start = 0
    for i in range(1, n + 1):
        if i == n or (smz[i] - smz[i - 1] > gap_threshold):
            seg_mz = smz[start:i]
            seg_int = sint[start:i]
            peak_int = float(seg_int.max())
            if peak_int >= min_intensity:
                total = float(seg_int.sum())
                if total > 0:
                    out_mz.append(float(np.sum(seg_mz * seg_int) / total))
                else:
                    out_mz.append(float(seg_mz[int(np.argmax(seg_int))]))
                out_int.append(peak_int)
            start = i
    return (
        np.array(out_mz, dtype=np.float64),
        np.array(out_int, dtype=np.float64),
    )


def _centroid_if_needed(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    min_intensity: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """按真实 profile/centroid 判定处理 MS2 谱。

    - 已 centroid（多数 vendor 数据，如 ASFAM 的 ProteoWizard 输出）：原样直通，
      仅做强度 >= min_intensity 掩码（与 MS1 装载对称，保留数据点 m/z）。
    - 真 profile：逐峰、峰内加权质心（见 ``_centroid_profile``）。

    历史 bug：旧实现用「点数 >= 20 即当 profile」的判据，对稀疏 centroid 谱做
    跨离子 3 点加权均值，导致碎片 m/z 漂移最多 ~0.6 Da（见 spec 2026-07-01
    asfam-ms2-mz-accuracy-fix-design）。
    """
    if len(mz_array) < 3:
        return mz_array, intensity_array

    if _looks_centroided(mz_array):
        mask = intensity_array >= min_intensity
        return mz_array[mask], intensity_array[mask]

    return _centroid_profile(mz_array, intensity_array, min_intensity)


def _get_collision_energy(spectrum) -> float:
    """Extract collision energy from spectrum metadata."""
    try:
        ce = spectrum.get("MS:1000045")  # collision energy CV param
        if ce is not None:
            return float(ce)
    except (KeyError, TypeError, ValueError):
        pass
    return 0.0
