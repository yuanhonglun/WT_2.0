"""MS-DIAL-style fixed-Da SUM mass-slice EIC builder for LC-MS MS1 feature extraction.

Task 2.1: ``build_slice_eics_sum`` (formerly private ``_build_slice_eics_sum``;
a private alias is retained for backward compatibility) — faithful port of
``RetentionTimeTypedSpectra.GetMs1ExtractedChromatogram_temp2`` from::

    src/MSDIAL5/MsdialCore/DataObj/Spectrum.cs  (lines 26-42, RetrieveBin)

Key semantics that differ from the existing ppm ROI builder
(``ms1_eic_roi.py``):
  * Window width is a fixed Da half-width (``mass_slice_width``), not ppm.
  * Slice step equals ``mass_slice_width`` (NOT half-width) — adjacent slices
    overlap 50%.
  * Intensity aggregation is **SUM** of all centroids in the window, not MAX.
  * basePeakMz is the centroid with highest intensity (lowest m/z on ties,
    reproducing MS-DIAL's strict-``<`` update over m/z-ascending centroids).
  * Output is **sparse**: only scans whose summed intensity > 0 are stored;
    slices with no nonzero scan are omitted entirely.

Task 2.2 will add ``find_lc_ms1_features_msdial`` to this module.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence, Any

import numpy as np

from metabo_core.config.msdial_peak_spotting import MsdialPeakSpottingConfig
from metabo_core.algorithms.lc_ms1_features import MS1FeatureHit
from metabo_core.algorithms.msdial_peak_spotting import (
    msdial_detect_peaks_in_chromatogram,
    _chromatogram_estimated_noise,
    _lwma_msdial,
)


# ---------------------------------------------------------------------------
# Private builder — Task 2.1
# ---------------------------------------------------------------------------


def build_slice_eics_sum(
    scans: Sequence[Any],
    cfg: MsdialPeakSpottingConfig,
) -> list[tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
    """Build sparse SUM-based slice EICs mirroring MS-DIAL's fixed-Da approach.

    Parameters
    ----------
    scans:
        MS1 Scan-like sequence; each element must expose ``.rt``,
        ``.mz_array``, ``.intensity_array`` (same duck-type as
        ``find_lc_ms1_features``; caller pre-filters MS1 scans).
    cfg:
        ``MsdialPeakSpottingConfig`` driving the slice parameters.

    Returns
    -------
    list of ``(center_mz, basepeak_mz_per_scan, eic_per_scan, scan_indices)``
        Each tuple represents one non-empty slice:

        * ``center_mz`` — slice centre in Da.
        * ``basepeak_mz_per_scan`` — 1-D float64 array; the m/z of the
          highest-intensity centroid in the window for each retained scan
          (lowest m/z wins on intensity ties).
        * ``eic_per_scan`` — 1-D float64 array; summed intensity for each
          retained scan.
        * ``scan_indices`` — 1-D int64 array; 0-based positions into the
          input ``scans`` sequence, in ascending order.

        All three arrays are co-indexed and length == number of retained
        scans for that slice.  Slices with no scan having summed intensity
        > 0 are omitted from the list.
    """
    # Guard: empty input
    if not scans:
        return []

    # ------------------------------------------------------------------
    # Step 1: Flatten all (mz, intensity, scan_idx), filter zero intensity
    # ------------------------------------------------------------------
    mz_chunks: list[np.ndarray] = []
    int_chunks: list[np.ndarray] = []
    scan_chunks: list[np.ndarray] = []

    for idx, scan in enumerate(scans):
        mz = np.asarray(scan.mz_array, dtype=np.float64)
        intensity = np.asarray(scan.intensity_array, dtype=np.float64)
        if mz.size == 0:
            continue
        mask = intensity > 0.0
        if not mask.any():
            continue
        m = mz[mask]
        i = intensity[mask]
        mz_chunks.append(m)
        int_chunks.append(i)
        scan_chunks.append(np.full(m.size, idx, dtype=np.int64))

    if not mz_chunks:
        return []

    flat_mz = np.concatenate(mz_chunks)
    flat_int = np.concatenate(int_chunks)
    flat_scan = np.concatenate(scan_chunks)

    # Sort by m/z for O(log N) searchsorted slicing
    order = np.argsort(flat_mz, kind="quicksort")
    sorted_mz = flat_mz[order]
    sorted_int = flat_int[order]
    sorted_scan = flat_scan[order]

    n_scans = len(scans)

    # ------------------------------------------------------------------
    # Step 2: Determine slice centres
    # Intersection of observed m/z range with [mass_range_begin, mass_range_end].
    # Step = mass_slice_width (MS-DIAL massStep), starting from the intersection
    # lower bound — matches MS-DIAL's focusedMass stepping pattern.
    # ------------------------------------------------------------------
    w = float(cfg.mass_slice_width)
    obs_min = float(sorted_mz[0])
    obs_max = float(sorted_mz[-1])
    global_start = max(obs_min, float(cfg.mass_range_begin))
    global_end = min(obs_max, float(cfg.mass_range_end))

    if global_end < global_start or w <= 0.0:
        return []

    # np.arange with w*0.5 epsilon so the last centre at global_end is included
    centers = np.arange(global_start, global_end + w * 0.5, w, dtype=np.float64)
    if centers.size == 0:
        return []

    # ------------------------------------------------------------------
    # Step 3: Per-slice extraction (vectorised, no Python loop over points)
    # ------------------------------------------------------------------
    result: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []

    for center in centers:
        lo = center - w
        hi = center + w

        # Inclusive window: searchsorted "left" keeps == lo; "right" keeps == hi
        i_lo = int(np.searchsorted(sorted_mz, lo, side="left"))
        i_hi = int(np.searchsorted(sorted_mz, hi, side="right"))

        if i_hi <= i_lo:
            # No centroids in this window → empty slice, omit
            continue

        seg_mz = sorted_mz[i_lo:i_hi]
        seg_int = sorted_int[i_lo:i_hi]
        seg_scan = sorted_scan[i_lo:i_hi]

        # SUM per scan via unbuffered scatter-add
        eic = np.zeros(n_scans, dtype=np.float64)
        np.add.at(eic, seg_scan, seg_int)

        # Sparse: keep only scans with nonzero sum
        nonzero_mask = eic > 0.0
        if not nonzero_mask.any():
            continue

        scan_indices = np.flatnonzero(nonzero_mask).astype(np.int64)
        eic_values = eic[scan_indices]

        # basePeakMz per retained scan:
        #   lexsort primary=scan (asc), secondary=intensity (desc), tertiary=mz (asc)
        #   → first element per scan block = highest intensity, lowest m/z on ties
        #   This reproduces MS-DIAL's strict-< update over m/z-ascending centroids.
        bp_order = np.lexsort((seg_mz, -seg_int, seg_scan))
        sorted_seg_scan = seg_scan[bp_order]
        sorted_seg_mz = seg_mz[bp_order]

        # np.unique returns first occurrence per unique scan in the sorted order
        uniq_scans, first_idx = np.unique(sorted_seg_scan, return_index=True)
        max_mz_for_uniq = sorted_seg_mz[first_idx]

        # Map scan_indices (ascending) → index in uniq_scans (also ascending)
        loc = np.searchsorted(uniq_scans, scan_indices)
        basepeak_mz = max_mz_for_uniq[loc]

        result.append((float(center), basepeak_mz, eic_values, scan_indices))

    return result


_build_slice_eics_sum = build_slice_eics_sum


# ---------------------------------------------------------------------------
# Task 2.2 — MS1 mass-slice orchestrator
#
# Faithful port of MS-DIAL's full MS1 feature pipeline:
#   PeakSpottingCore.Execute3DFeatureDetectionNormalModeBySingleThread (67-107)
#   + GetCombinedChromPeakFeatures (55-65). The stages below mirror that flow:
#     A. per-slice detect (engine) + adjacent-slice redundancy (709-756)
#     B. flatten kept slices (758-768)
#     C. coarse->fine recalc by CentroidMs1Tolerance (770-940)
#     D. sort (Mass, RT) + global near-dup cleanup (678-707)
#     E. map survivors to MS1FeatureHit
# All literal tolerances are taken from MsdialPeakSpottingConfig (the parameter
# surface), which carries the MS-DIAL defaults (massStep*0.5=0.05, 0.03 apex
# cap, CentroidMs1Tolerance*0.5=0.005 global mass tol, 0.03 global RT tol).
# ---------------------------------------------------------------------------


@dataclass
class _SliceFeature:
    """Intermediate per-slice feature carrying everything later stages need.

    Mirrors the subset of ``ChromatogramPeakFeature`` fields the MS-DIAL
    pipeline reads after coarse detection. ``mass`` is the coarse basePeakMz
    at the apex scan (``smoothedChromatogram.Mz(ScanNumAtPeakTop)``);
    ``height`` is ``PeakHeightTop`` (apex intensity on the LWMA trace). The
    scan indices are 0-based positions into ``ms1_scans`` (the engine's dense
    trace indices, which equal ``ms1_scans`` positions). ``estimated_noise`` is
    the COARSE slice's EstimatedNoise (constant within a slice), kept here so
    the recalc can reproduce ``PeakShape.EstimatedNoise`` (line 926).
    """
    mass: float
    rt_apex: float
    rt_left: float
    rt_right: float
    apex_scan_idx: int
    left_scan_idx: int
    right_scan_idx: int
    height: float
    area: float
    sn_ratio: float
    gaussian_similarity: float
    estimated_noise: float


# ---------------------------------------------------------------------------
# Stage A helpers — adjacent-slice redundancy
# ---------------------------------------------------------------------------


def _is_overlapped(prev: _SliceFeature, cur: _SliceFeature) -> bool:
    """Port of ``PeakSpottingCore.isOverlapedChecker`` (lines 748-756).

    True when the later-apex peak's left edge falls before the earlier-apex
    peak's apex (``peak1`` = ``prev``, ``peak2`` = ``cur``).
    """
    if prev.rt_apex > cur.rt_apex:
        return prev.rt_left < cur.rt_apex
    return cur.rt_left < prev.rt_apex


def _remove_peak_area_redundancy(
    parent_list: list[_SliceFeature],
    cur_list: list[_SliceFeature],
    mass_tol: float,
    apex_cap: float,
) -> list[_SliceFeature] | None:
    """Port of ``PeakSpottingCore.RemovePeakAreaBeanRedundancy`` (709-746).

    Compares ``cur_list`` (this slice) against ``parent_list`` (the immediately
    previous KEPT slice) and removes the SHORTER of any pair that is (a) within
    ``mass_tol`` in mass, (b) RT-overlapping per :func:`_is_overlapped`, and
    (c) within ``min(hwhm, apex_cap)`` in apex RT (``hwhm`` = quarter of the two
    peak widths). The taller survives. Both lists are MUTATED in place exactly as
    the C# mutates ``parentPeakAreaBeanList`` and ``chromPeakFeatures`` (the
    ``j--``/``i--``/``break`` index bookkeeping is reproduced faithfully).

    Returns ``cur_list`` (possibly shortened) or ``None`` when ``cur_list`` is
    emptied — matching the C# ``return null`` (line 743). The early ``return
    chromPeakFeatures`` when ``parentPeakAreaBeanList`` empties (line 742) is
    likewise reproduced.
    """
    i = 0
    while i < len(cur_list):
        broke = False
        j = 0
        while j < len(parent_list):
            prev = parent_list[j]
            cur = cur_list[i]
            if abs(prev.mass - cur.mass) <= mass_tol:
                if not _is_overlapped(prev, cur):
                    j += 1
                    continue
                hwhm = (
                    (prev.rt_right - prev.rt_left)
                    + (cur.rt_right - cur.rt_left)
                ) * 0.25
                tolerance = min(hwhm, apex_cap)
                if abs(prev.rt_apex - cur.rt_apex) <= tolerance:
                    if cur.height > prev.height:
                        # cur taller -> drop prev; C# RemoveAt(j); j--; continue
                        parent_list.pop(j)
                        continue  # keep j (C# j-- then loop j++ == same index)
                    else:
                        # cur shorter (or equal) -> drop cur; RemoveAt(i); i--; break
                        cur_list.pop(i)
                        broke = True
                        break  # keep i (C# i-- then loop i++ == same index)
            j += 1
        # post-inner-loop early returns (C# lines 742-743)
        if not parent_list:
            return cur_list
        if not cur_list:
            return None
        if not broke:
            i += 1
    return cur_list


# ---------------------------------------------------------------------------
# Stage C helper — coarse->fine recalc by CentroidMs1Tolerance
# ---------------------------------------------------------------------------


def _recalculate(
    feat: _SliceFeature,
    ms1_scans: Sequence[Any],
    rt_all: np.ndarray,
    cfg: MsdialPeakSpottingConfig,
) -> MS1FeatureHit | None:
    """Port of ``GetRecalculatedChromPeakFeaturesByMs1MsTolerance`` body (770-940).

    Re-extracts a TIGHT SUM EIC at ``feat.mass ± CentroidMs1Tolerance`` over the
    RT window ``[rt_left - margin, rt_right + margin]`` (``margin = peakWidth *
    0.5``), smooths it (LWMA), re-finds the apex/edges and re-measures
    height/area/RT/scan-indices/S/N. ``mass`` and ``gaussian_similarity`` are NOT
    changed by the recalc. Returns the refined :class:`MS1FeatureHit`, or
    ``None`` when the feature is dropped (the two ``continue`` guards at lines
    885-886, or a degenerate too-short window).

    Index mapping: the tight EIC is built over the contiguous RT-windowed scans
    ``[start_index, end_index)`` of ``ms1_scans`` (all MS1, caller-prefiltered),
    so tight-EIC position ``k`` is absolute scan ``start_index + k`` — the
    MS1FeatureHit scan-index contract. This is the faithful equivalent of
    ``sPeaklist[...].ID`` (the C# ValuePeak.Id is the scan index).
    """
    smoothing_level = int(cfg.smoothing_level)
    min_datapoint = 3  # C# local `minDatapoint = 3` (line 773)
    tol = float(cfg.centroid_ms1_tolerance)
    min_amplitude = float(cfg.min_amplitude)
    mass = feat.mass

    peak_width = feat.rt_right - feat.rt_left            # peak.PeakWidth()
    margin = peak_width * 0.5
    begin = feat.rt_left - margin
    end = feat.rt_right + margin

    # Scans in [begin, end] inclusive -> LowerBound(begin)/UpperBound(end)
    # (RetentionTimeTypedSpectra.GetMs1ExtractedChromatogram_temp2:78-79).
    start_index = int(np.searchsorted(rt_all, begin, side="left"))
    end_index = int(np.searchsorted(rt_all, end, side="right"))
    if end_index <= start_index:
        return None

    n_pts = end_index - start_index
    # Dense SUM EIC over the windowed scans (one point per scan, 0 where empty),
    # value = sum of intensities in [mass - tol, mass + tol] (Spectrum.RetrieveBin).
    tight_int = np.zeros(n_pts, dtype=np.float64)
    lo = mass - tol
    hi = mass + tol
    for k in range(n_pts):
        scan = ms1_scans[start_index + k]
        mz = np.asarray(scan.mz_array, dtype=np.float64)
        if mz.size == 0:
            continue
        it = np.asarray(scan.intensity_array, dtype=np.float64)
        m = (mz >= lo) & (mz <= hi)
        if m.any():
            tight_int[k] = float(it[m].sum())
    tight_rt = np.asarray(rt_all[start_index:end_index], dtype=np.float64)

    # ChromatogramSmoothing(LWMA, SmoothingLevel) (line 783).
    s_int = _lwma_msdial(tight_int, smoothing_level)
    count = s_int.size

    # minRtId: index nearest feat.rt_apex; C# loops i < count-1 (line 790).
    min_rt_id = -1
    min_rt_val = np.inf
    for i in range(count - 1):
        d = abs(tight_rt[i] - feat.rt_apex)
        if d < min_rt_val:
            min_rt_val = d
            min_rt_id = i
    if min_rt_id == -1:
        # Degenerate (count <= 1): C# would index sPeaklist[-1] and throw;
        # we skip the feature instead. Normal windows have many points.
        return None

    # Local maximum within minRtId +/- 2 (lines 798-807).
    max_id = -1
    max_int = -np.inf  # C# double.MinValue
    for i in range(min_rt_id - 2, min_rt_id + 3):
        if i - 1 < 0:
            continue
        if i > count - 2:
            break
        if (
            s_int[i] > max_int
            and s_int[i - 1] <= s_int[i]
            and s_int[i] >= s_int[i + 1]
        ):
            max_int = s_int[i]
            max_id = i
    if max_id == -1:  # lines 818-821
        max_id = min_rt_id

    # Left edge walk (lines 824-853).
    min_left_int = s_int[max_id]
    min_left_id = -1
    i = max_id - min_datapoint
    while i >= 0:
        if i < max_id and min_left_int < s_int[i]:
            break
        if tight_rt[max_id] - peak_width > tight_rt[i]:
            break
        if min_left_int >= s_int[i]:
            min_left_int = s_int[i]
            min_left_id = i
        i -= 1
    if min_left_id == -1:  # fallback to nearest-to-original-rt_left (839-853)
        min_orig_left_diff = np.inf
        min_orig_left_id = max_id - min_datapoint
        if min_orig_left_id < 0:
            min_orig_left_id = 0
        for ii in range(max_id, -1, -1):
            diff = abs(tight_rt[ii] - feat.rt_left)
            if diff < min_orig_left_diff:
                min_orig_left_diff = diff
                min_orig_left_id = ii
        min_left_id = min_orig_left_id

    # Right edge walk (lines 856-883).
    min_right_int = s_int[max_id]
    min_right_id = -1
    i = max_id + min_datapoint
    while i < count - 1:
        if i > max_id and min_right_int < s_int[i]:
            break
        if tight_rt[max_id] + peak_width < tight_rt[i]:
            break
        if min_right_int >= s_int[i]:
            min_right_int = s_int[i]
            min_right_id = i
        i += 1
    if min_right_id == -1:  # fallback to nearest-to-original-rt_right (869-883)
        min_orig_right_diff = np.inf
        min_orig_right_id = max_id + min_datapoint
        if min_orig_right_id > count - 1:
            min_orig_right_id = count - 1
        for ii in range(max_id, count):
            diff = abs(tight_rt[ii] - feat.rt_right)
            if diff < min_orig_right_diff:
                min_orig_right_diff = diff
                min_orig_right_id = ii
        min_right_id = min_orig_right_id

    # Drop conditions (lines 885-886).
    if max(s_int[min_left_id], s_int[min_right_id]) >= s_int[max_id]:
        return None
    if s_int[max_id] - min(s_int[min_left_id], s_int[min_right_id]) < min_amplitude:
        return None

    # realMaxID + area over [minLeftId, minRightId) (lines 889-898).
    real_max_int = -np.inf
    real_max_id = max_id
    area_above_zero = 0.0
    for i in range(min_left_id, min_right_id):  # C# i <= minRightId - 1
        if real_max_int < s_int[i]:
            real_max_int = s_int[i]
            real_max_id = i
        area_above_zero += (
            (s_int[i] + s_int[i + 1]) * (tight_rt[i + 1] - tight_rt[i]) * 0.5
        )
    max_id = real_max_id  # line 901

    # areaAboveBaseline (line 903-904); x60 for the RT axis -> seconds, the same
    # convention the engine uses for DetectedPeak.area (line 907 + engine line 799).
    area_above_baseline = area_above_zero - (
        (s_int[min_left_id] + s_int[min_right_id])
        * (tight_rt[min_right_id] - tight_rt[min_left_id])
        / 2.0
    )
    hit_area = area_above_baseline * 60.0

    # SignalToNoise = peakHeightFromBaseline / EstimatedNoise (lines 925-926),
    # EstimatedNoise = the COARSE slice's value (carried on the feature).
    peak_height_from_baseline = max(
        s_int[max_id] - s_int[min_left_id],
        s_int[max_id] - s_int[min_right_id],
    )
    sn = peak_height_from_baseline / feat.estimated_noise

    return MS1FeatureHit(
        mz_centroid=mass,
        rt_apex=float(tight_rt[max_id]),
        rt_left=float(tight_rt[min_left_id]),
        rt_right=float(tight_rt[min_right_id]),
        height=float(s_int[max_id]),
        area=float(hit_area),
        sn_ratio=float(sn),
        gaussian_similarity=float(feat.gaussian_similarity),
        apex_scan_idx=int(start_index + max_id),
        left_scan_idx=int(start_index + min_left_id),
        right_scan_idx=int(start_index + min_right_id),
    )


# ---------------------------------------------------------------------------
# Stage D helper — global near-duplicate cleanup
# ---------------------------------------------------------------------------


def _further_cleanup(
    features: list[MS1FeatureHit], mass_tol: float, rt_tol: float
) -> list[MS1FeatureHit]:
    """Port of ``PeakSpottingCore.GetFurtherCleanupedChromPeakFeatures`` (678-707).

    ``features`` MUST be sorted by ``mz_centroid`` ascending (then RT), matching
    the C# ``OrderBy(Mass).ThenBy(RT)`` that precedes this call (line 60). For
    each pair ``(i, j>i)`` it breaks once ``mass[j] - mass[i] > mass_tol`` and,
    when ``|rt[i] - rt[j]| < rt_tol``, marks the SHORTER for exclusion (the
    taller survives; on an exact height tie the EARLIER ``i`` is excluded, exactly
    as ``(target - searched) > 0`` evaluates). Both ``i`` and ``j`` may be
    excluded across different pairings (the ``excludeList`` semantics).
    """
    n = len(features)
    exclude: set[int] = set()
    for i in range(n):
        target = features[i]
        for j in range(i + 1, n):
            searched = features[j]
            if (searched.mz_centroid - target.mz_centroid) > mass_tol:
                break
            if abs(target.rt_apex - searched.rt_apex) < rt_tol:
                if (target.height - searched.height) > 0:
                    exclude.add(j)
                else:
                    exclude.add(i)
    return [features[i] for i in range(n) if i not in exclude]


# ---------------------------------------------------------------------------
# Public orchestrator — Task 2.2
# ---------------------------------------------------------------------------


def find_lc_ms1_features_msdial(
    ms1_scans,
    *,
    config: MsdialPeakSpottingConfig,
    mz_range: tuple[float, float] | None = None,
) -> list[MS1FeatureHit]:
    """MS-DIAL-faithful MS1 mass-slice feature extraction (drop-in for
    :func:`metabo_core.algorithms.lc_ms1_features.find_lc_ms1_features`).

    Ports the full MS-DIAL MS1 pipeline (``PeakSpottingCore`` normal LC path):
    fixed-Da SUM slices -> derivative engine per slice -> adjacent-slice
    redundancy -> flatten -> coarse->fine recalc by CentroidMs1Tolerance ->
    global near-duplicate cleanup. Returns the same :class:`MS1FeatureHit`
    contract: ``apex_scan_idx`` / ``left_scan_idx`` / ``right_scan_idx`` are
    0-based positions into the input ``ms1_scans`` sequence.

    Parameters
    ----------
    ms1_scans : Scan-like sequence
        Each element exposes ``.rt`` / ``.mz_array`` / ``.intensity_array``;
        caller pre-filters to MS1, RT ascending.
    config : MsdialPeakSpottingConfig
        Drives slicing, the engine, and all redundancy/dedup tolerances.
    mz_range : (lo, hi) or None
        Restricts slice centres to ``[lo, hi]`` intersected with the config mass
        range; ``None`` uses the observed m/z range (same as
        :func:`build_slice_eics_sum`).

    Returns
    -------
    list[MS1FeatureHit]
        Sorted by ``mz_centroid`` then ``rt_apex``.
    """
    if not ms1_scans:
        return []

    rt_all = np.array([float(s.rt) for s in ms1_scans], dtype=np.float64)
    n_scans = len(ms1_scans)

    # mz_range restriction: intersect with the config mass range and let the
    # slice builder additionally intersect with the observed range.
    if mz_range is not None:
        eff_begin = max(float(config.mass_range_begin), float(mz_range[0]))
        eff_end = min(float(config.mass_range_end), float(mz_range[1]))
        slice_cfg = replace(
            config, mass_range_begin=eff_begin, mass_range_end=eff_end
        )
    else:
        slice_cfg = config

    slices = build_slice_eics_sum(ms1_scans, slice_cfg)
    if not slices:
        return []

    mass_tol = float(config.adjacent_redundancy_mass_tol)
    apex_cap = float(config.adjacent_redundancy_apex_tol)

    # --- Stage A: per-slice detect + adjacent-slice redundancy --------------
    kept_slices: list[list[_SliceFeature]] = []
    for center, basepeak_mz, eic_vals, scan_idx in slices:
        # Densify to a dense trace over ALL scans: 0-fill EIC gaps (zeros feed
        # smoothing/derivatives/noise) and center-fill basePeakMz gaps (MS-DIAL's
        # RetrieveBin returns the slice centre as basePeakMz for empty bins).
        dense_eic = np.zeros(n_scans, dtype=np.float64)
        dense_eic[scan_idx] = eic_vals
        dense_bpmz = np.full(n_scans, center, dtype=np.float64)
        dense_bpmz[scan_idx] = basepeak_mz

        peaks = msdial_detect_peaks_in_chromatogram(rt_all, dense_eic, config=config)
        if not peaks:
            continue

        # Coarse slice EstimatedNoise (constant within the slice), reused by the
        # recalc S/N (line 926). Same dense EIC the engine just consumed.
        estimated_noise = _chromatogram_estimated_noise(dense_eic, config)

        cur: list[_SliceFeature] = [
            _SliceFeature(
                mass=float(dense_bpmz[p.apex_index]),
                rt_apex=float(p.rt_apex),
                rt_left=float(p.rt_left),
                rt_right=float(p.rt_right),
                apex_scan_idx=int(p.apex_index),
                left_scan_idx=int(p.left_index),
                right_scan_idx=int(p.right_index),
                height=float(p.height),
                area=float(p.area),
                sn_ratio=float(p.sn_ratio),
                gaussian_similarity=float(p.gaussian_similarity),
                estimated_noise=float(estimated_noise),
            )
            for p in peaks
        ]

        if kept_slices:  # compare against the immediately previous KEPT slice
            reduced = _remove_peak_area_redundancy(
                kept_slices[-1], cur, mass_tol, apex_cap
            )
        else:
            reduced = cur
        if reduced:  # only keep non-empty slices
            kept_slices.append(reduced)

    # --- Stage B: flatten kept slices (skip emptied ones) -------------------
    combined: list[_SliceFeature] = []
    for slc in kept_slices:
        if slc:
            combined.extend(slc)
    if not combined:
        return []

    # --- Stage C: coarse->fine recalc ---------------------------------------
    hits: list[MS1FeatureHit] = []
    for feat in combined:
        hit = _recalculate(feat, ms1_scans, rt_all, config)
        if hit is not None:
            hits.append(hit)
    if not hits:
        return []

    # --- Stage D: sort (Mass, RT) + global near-dup cleanup -----------------
    hits.sort(key=lambda h: (h.mz_centroid, h.rt_apex))
    hits = _further_cleanup(
        hits, float(config.global_dedup_mass_tol), float(config.global_dedup_rt_tol)
    )
    return hits


__all__: list[str] = ["find_lc_ms1_features_msdial", "build_slice_eics_sum"]
