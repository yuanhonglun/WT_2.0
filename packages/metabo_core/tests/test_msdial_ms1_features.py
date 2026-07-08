"""Unit tests for _build_slice_eics_sum (Task 2.1).

Tests:
1. Canonical sum + basePeakMz (spec fixture: scan0={800.95:100,801.05:200},
   scan1={801.0:500}, slice near 801.0)
2. Tie-break: equal-intensity centroids → lowest m/z wins (faithful MS-DIAL
   strict-< update)
3. Window boundary inclusivity: centroids exactly at center ± mass_slice_width
   are included
4. Sparse output: scan with no centroid in a slice is excluded from scan_indices
5. Zero-signal slices are omitted from the returned list
6. Empty scans → []
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from metabo_core.algorithms.lc_ms1_features import MS1FeatureHit
from metabo_core.algorithms.msdial_ms1_features import (
    _build_slice_eics_sum,
    _further_cleanup,
    _is_overlapped,
    _remove_peak_area_redundancy,
    _SliceFeature,
    find_lc_ms1_features_msdial,
)
from metabo_core.config.msdial_peak_spotting import MsdialPeakSpottingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scan(rt: float, mz, intensity):
    """Minimal Scan-like duck type (SimpleNamespace)."""
    return SimpleNamespace(
        rt=float(rt),
        mz_array=np.asarray(mz, dtype=np.float64),
        intensity_array=np.asarray(intensity, dtype=np.float64),
    )


def _default_cfg(**overrides) -> MsdialPeakSpottingConfig:
    return MsdialPeakSpottingConfig(**overrides)


def _find_slice_near(result, target_mz, half_step):
    """Return all (center, basepeak_mz, eic, scan_indices) within half_step of target_mz."""
    return [
        (c, bp, eic, si)
        for c, bp, eic, si in result
        if abs(c - target_mz) <= half_step
    ]


# ---------------------------------------------------------------------------
# Test 1: canonical SUM + basePeakMz
# ---------------------------------------------------------------------------


def test_canonical_sum_and_basepeak_mz():
    """Spec canonical fixture: two scans, slice near 801.0 must sum correctly.

    scan0 centroids {800.95:100, 801.05:200}; scan1 {801.0:500}
    Window ±0.1 around ~801.0 → [800.9, 801.1] includes all three.
    Expected: eic = [300, 500]; basePeakMz scan0 = 801.05 (highest int=200);
    basePeakMz scan1 = 801.0; scan_indices = [0, 1].
    """
    scans = [
        _scan(1.0, [800.95, 801.05], [100.0, 200.0]),
        _scan(1.1, [801.0], [500.0]),
    ]
    cfg = _default_cfg()
    result = _build_slice_eics_sum(scans, cfg)

    assert result, "Expected at least one slice"

    half_step = cfg.mass_slice_width / 2  # 0.05
    hits = _find_slice_near(result, 801.0, half_step)
    assert hits, f"No slice found within {half_step} Da of 801.0"

    # All matching slices must satisfy the canonical assertions
    # (any center within half-step of 801.0 covers all three centroids)
    c, bp, eic, si = hits[0]
    np.testing.assert_array_equal(si, [0, 1], err_msg="scan_indices must be [0, 1]")
    np.testing.assert_allclose(eic, [300.0, 500.0],
                               err_msg="SUM intensities must be [300, 500]")
    np.testing.assert_allclose(bp[0], 801.05,
                               err_msg="scan0 basePeakMz must be 801.05 (int=200>100)")
    np.testing.assert_allclose(bp[1], 801.0,
                               err_msg="scan1 basePeakMz must be 801.0 (only centroid)")


# ---------------------------------------------------------------------------
# Test 2: tie-break — equal intensity → lowest m/z wins
# ---------------------------------------------------------------------------


def test_basepeak_tiebreak_lowest_mz_wins():
    """MS-DIAL strict-< update: on intensity tie, first (lowest-m/z) centroid wins.

    Two centroids at m/z=100.05 and m/z=100.10 with EQUAL intensity=500.
    The slice window that covers both must report basePeakMz=100.05.
    """
    scans = [
        _scan(1.0, [100.05, 100.10], [500.0, 500.0]),
    ]
    cfg = _default_cfg()
    result = _build_slice_eics_sum(scans, cfg)

    w = cfg.mass_slice_width  # 0.1
    # Find any slice whose window [c-w, c+w] covers both 100.05 and 100.10
    covering = []
    for c, bp, eic, si in result:
        lo, hi = c - w, c + w
        if lo <= 100.05 and 100.10 <= hi:
            covering.append((c, bp, eic, si))

    assert covering, "Expected at least one slice covering both 100.05 and 100.10"
    c, bp, eic, si = covering[0]
    np.testing.assert_allclose(
        bp[0], 100.05,
        err_msg="Tie-break must favour lowest m/z (100.05 not 100.10)",
    )


# ---------------------------------------------------------------------------
# Test 3: window boundary inclusivity
# ---------------------------------------------------------------------------


def test_window_boundary_inclusive():
    """Centroids exactly at center - w and center + w must be included.

    MS-DIAL: break when peak.Mz > mz+tol, so == is kept.
    searchsorted(..., 'right') on the upper end captures == elements.
    """
    cfg = _default_cfg()
    w = cfg.mass_slice_width   # 0.1
    center = 100.0
    lo_mz = center - w         # 99.9 — exactly on lower boundary
    hi_mz = center + w         # 100.1 — exactly on upper boundary

    scans = [
        _scan(1.0, [lo_mz, hi_mz], [300.0, 400.0]),
    ]
    result = _build_slice_eics_sum(scans, cfg)

    half_step = w / 2
    hits = _find_slice_near(result, center, half_step)
    assert hits, f"No slice found near {center}"

    # The slice whose center == 100.0 (or nearest) must sum both boundary points
    # Any center within 0.05 of 100.0 has window spanning ±0.1 → [c-0.1, c+0.1]
    # For c=99.9..100.0..100.1, the window [c-0.1, c+0.1] covers 99.9 and 100.1
    found_inclusive = False
    for c, bp, eic, si in hits:
        lo, hi = c - w, c + w
        if lo <= lo_mz and hi_mz <= hi:
            # Both boundary centroids are in this window
            np.testing.assert_allclose(
                eic[0], 700.0,
                err_msg=f"Boundary centroids at {lo_mz} and {hi_mz} should both be included; "
                        f"slice center={c}, window=[{lo},{hi}]",
            )
            found_inclusive = True
            break
    assert found_inclusive, "No slice whose window actually covers both boundary centroids"


# ---------------------------------------------------------------------------
# Test 4: sparse — scan with no centroid in slice is excluded
# ---------------------------------------------------------------------------


def test_sparse_zero_scan_excluded_from_indices():
    """Scans with no centroid in a given slice are absent from scan_indices.

    scan0 has centroid at m/z=500 (far away); scan1 has centroid at 100.0.
    A slice near 100.0 must have scan_indices=[1] only.
    """
    scans = [
        _scan(1.0, [500.0], [1000.0]),   # scan0: far from 100
        _scan(1.1, [100.0], [800.0]),    # scan1: in range
    ]
    cfg = _default_cfg()
    result = _build_slice_eics_sum(scans, cfg)

    w = cfg.mass_slice_width
    hits = _find_slice_near(result, 100.0, w / 2)
    assert hits, "Expected a slice near 100.0"

    c, bp, eic, si = hits[0]
    assert 0 not in si, "scan0 has no centroid in this slice; it must be excluded (sparse)"
    assert 1 in si, "scan1 has the centroid; it must be present"


# ---------------------------------------------------------------------------
# Test 5: slice with zero total signal is omitted
# ---------------------------------------------------------------------------


def test_zero_signal_slice_omitted():
    """Every slice in the output must have at least one nonzero scan."""
    cfg = _default_cfg()
    scans = [
        _scan(1.0, [100.0], [1000.0]),
    ]
    result = _build_slice_eics_sum(scans, cfg)

    for c, bp, eic, si in result:
        assert eic.sum() > 0, (
            f"Slice at center={c} has zero total signal and should have been omitted"
        )
    # Also: the slice near 100.0 must be present
    hits = _find_slice_near(result, 100.0, cfg.mass_slice_width / 2)
    assert hits, "The only nonzero slice (near 100.0) must appear in output"


# ---------------------------------------------------------------------------
# Test 6: empty scans → []
# ---------------------------------------------------------------------------


def test_empty_scans_returns_empty_list():
    cfg = _default_cfg()
    assert _build_slice_eics_sum([], cfg) == []


# ---------------------------------------------------------------------------
# Test 7: all-zero-intensity scans → []
# ---------------------------------------------------------------------------


def test_all_zero_intensity_scans_returns_empty():
    scans = [
        _scan(1.0, [100.0], [0.0]),
        _scan(1.1, [100.5], [0.0]),
    ]
    cfg = _default_cfg()
    result = _build_slice_eics_sum(scans, cfg)
    assert result == [], "Scans with only zero intensities should yield no slices"


# ===========================================================================
# Task 2.2 — find_lc_ms1_features_msdial (orchestrator)
# ===========================================================================


def _gaussian_scans(n_scans, ion_specs, decoys=None, rt_step=0.1, seed=12345):
    """Build ``n_scans`` MS1 Scan-like objects.

    ion_specs : list of dict(center_mz, apex_scan, sigma_scans, apex_int,
                             scan_lo, scan_hi, jitter)
        Each ion contributes a Gaussian-in-RT centroid per scan in
        ``[scan_lo, scan_hi]`` at ``center_mz (+/- jitter)``.
    decoys : list of (scan_idx, mz, intensity)
    """
    rng = np.random.default_rng(seed)
    per_mz = [[] for _ in range(n_scans)]
    per_int = [[] for _ in range(n_scans)]
    for spec in ion_specs:
        jit_amp = float(spec.get("jitter", 0.0))
        for s in range(spec["scan_lo"], spec["scan_hi"] + 1):
            val = spec["apex_int"] * np.exp(
                -0.5 * ((s - spec["apex_scan"]) / spec["sigma_scans"]) ** 2
            )
            jit = rng.uniform(-jit_amp, jit_amp) if jit_amp > 0 else 0.0
            per_mz[s].append(spec["center_mz"] + jit)
            per_int[s].append(val)
    for (s, mz, it) in (decoys or []):
        per_mz[s].append(mz)
        per_int[s].append(it)
    scans = []
    for i in range(n_scans):
        if per_mz[i]:
            order = np.argsort(per_mz[i])
            mz = np.asarray(per_mz[i], dtype=np.float64)[order]
            it = np.asarray(per_int[i], dtype=np.float64)[order]
        else:
            mz = np.zeros(0, dtype=np.float64)
            it = np.zeros(0, dtype=np.float64)
        scans.append(_scan(i * rt_step, mz, it))
    return scans


def _sf(mass, apex, left, right, h):
    return _SliceFeature(
        mass=mass, rt_apex=apex, rt_left=left, rt_right=right,
        apex_scan_idx=0, left_scan_idx=0, right_scan_idx=0,
        height=h, area=0.0, sn_ratio=0.0, gaussian_similarity=0.0,
        estimated_noise=1.0,
    )


def _hit(mz, rt, h):
    return MS1FeatureHit(
        mz_centroid=mz, rt_apex=rt, rt_left=rt - 0.05, rt_right=rt + 0.05,
        height=h, area=0.0, sn_ratio=0.0, gaussian_similarity=0.0,
        apex_scan_idx=0, left_scan_idx=0, right_scan_idx=0,
    )


# ---------------------------------------------------------------------------
# Main: a clean high-intensity MS1 ion is found
# ---------------------------------------------------------------------------


def test_msdial_finds_high_intensity_ms1_feature():
    """One clean Gaussian ion at m/z ~801.2096 -> exactly one hit; decoys ignored."""
    scans = _gaussian_scans(
        n_scans=80,
        ion_specs=[dict(center_mz=801.2096, apex_scan=37, sigma_scans=3.0,
                        apex_int=1.7e6, scan_lo=30, scan_hi=45, jitter=0.004)],
        decoys=[(10, 300.0, 200.0), (60, 500.0, 200.0)],  # sub-threshold spikes
        seed=12345,
    )
    cfg = _default_cfg()
    result = find_lc_ms1_features_msdial(scans, config=cfg)

    near = [h for h in result if abs(h.mz_centroid - 801.2096) < 0.02]
    assert len(near) == 1, (
        f"expected exactly one hit near 801.2096; got "
        f"{[round(h.mz_centroid, 4) for h in result]}"
    )
    assert len(result) == 1, f"low decoys must not produce hits; got {len(result)}"
    hit = near[0]
    assert 3.0 <= hit.rt_apex <= 4.5, hit.rt_apex            # scans 30..45 -> rt 3.0..4.5
    assert hit.height > 1e6, hit.height
    assert 30 <= hit.apex_scan_idx <= 45, hit.apex_scan_idx


# ---------------------------------------------------------------------------
# (a) two coeluting near-isobars within 0.05 Da -> collapse to one feature
# ---------------------------------------------------------------------------


def test_coeluting_near_isobars_collapse_to_one():
    scans = _gaussian_scans(
        n_scans=80,
        ion_specs=[
            dict(center_mz=600.100, apex_scan=37, sigma_scans=3.0,
                 apex_int=8.0e5, scan_lo=30, scan_hi=45, jitter=0.0),
            dict(center_mz=600.140, apex_scan=37, sigma_scans=3.0,
                 apex_int=1.5e6, scan_lo=30, scan_hi=45, jitter=0.0),
        ],
        seed=7,
    )
    cfg = _default_cfg()
    result = find_lc_ms1_features_msdial(scans, config=cfg)

    region = [h for h in result if 600.0 <= h.mz_centroid <= 600.25]
    assert len(region) == 1, (
        "coeluting near-isobars (0.04 Da apart) must resolve to one feature; "
        f"got {[round(h.mz_centroid, 4) for h in region]}"
    )
    # The survivor carries the taller ion's basePeakMz (600.140).
    assert abs(region[0].mz_centroid - 600.140) < 0.02, region[0].mz_centroid


# ---------------------------------------------------------------------------
# (b) recalc drop-condition fires -> feature dropped
# ---------------------------------------------------------------------------


def test_recalc_drops_low_tight_amplitude_feature():
    """Coarse +/-0.1 SUM peak passes (3 centroids summed), but each single
    +/-0.01 sub-window has amplitude < min_amplitude -> recalc drops it."""
    scans = _gaussian_scans(
        n_scans=80,
        ion_specs=[
            dict(center_mz=700.00, apex_scan=37, sigma_scans=3.0,
                 apex_int=700.0, scan_lo=30, scan_hi=44, jitter=0.0),
            dict(center_mz=700.05, apex_scan=37, sigma_scans=3.0,
                 apex_int=700.0, scan_lo=30, scan_hi=44, jitter=0.0),
            dict(center_mz=700.10, apex_scan=37, sigma_scans=3.0,
                 apex_int=700.0, scan_lo=30, scan_hi=44, jitter=0.0),
        ],
        seed=3,
    )
    cfg = _default_cfg()
    result = find_lc_ms1_features_msdial(scans, config=cfg)

    region = [h for h in result if 699.8 <= h.mz_centroid <= 700.3]
    assert region == [], (
        "recalc must drop the feature whose tight +/-0.01 EIC amplitude is below "
        f"min_amplitude; got {[(round(h.mz_centroid, 3), h.height) for h in region]}"
    )


# ---------------------------------------------------------------------------
# (c) global near-duplicate cleanup keeps the taller (Stage D, direct)
# ---------------------------------------------------------------------------


def test_global_near_dup_keeps_taller():
    feats = [
        _hit(mz=500.0000, rt=3.00, h=1.0e5),
        _hit(mz=500.0030, rt=3.01, h=3.0e5),  # within 0.005 Da & 0.03 min of [0]
    ]
    out = _further_cleanup(feats, mass_tol=0.005, rt_tol=0.03)
    assert len(out) == 1
    assert out[0].height == 3.0e5, "taller of the near-duplicate pair must survive"


def test_global_near_dup_tie_keeps_later():
    feats = [
        _hit(mz=500.000, rt=3.00, h=2.0e5),
        _hit(mz=500.002, rt=3.01, h=2.0e5),  # equal height -> exclude the earlier i
    ]
    out = _further_cleanup(feats, mass_tol=0.005, rt_tol=0.03)
    assert len(out) == 1
    assert out[0].mz_centroid == 500.002, "on a height tie the later (j) survives"


def test_global_near_dup_mass_break_keeps_separated():
    feats = [
        _hit(mz=500.000, rt=3.00, h=1.0e5),
        _hit(mz=500.020, rt=3.00, h=1.0e5),  # 0.02 Da apart (> 0.005) -> both kept
    ]
    out = _further_cleanup(feats, mass_tol=0.005, rt_tol=0.03)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# Adjacent-slice redundancy (Stage A, direct)
# ---------------------------------------------------------------------------


def test_is_overlapped_checker():
    a = _sf(mass=1.0, apex=3.00, left=2.90, right=3.10, h=1.0)
    b = _sf(mass=1.0, apex=3.05, left=2.95, right=3.15, h=1.0)  # b.left < a.apex
    assert _is_overlapped(a, b) is True
    c = _sf(mass=1.0, apex=5.00, left=4.90, right=5.10, h=1.0)  # disjoint
    assert _is_overlapped(a, c) is False


def test_adjacent_redundancy_taller_cur_survives():
    prev = [_sf(mass=400.00, apex=3.00, left=2.90, right=3.10, h=5.0e5)]
    cur = [_sf(mass=400.02, apex=3.00, left=2.90, right=3.10, h=8.0e5)]
    out = _remove_peak_area_redundancy(prev, cur, mass_tol=0.05, apex_cap=0.03)
    assert out is not None and len(out) == 1 and out[0].height == 8.0e5
    assert prev == [], "the shorter previous-slice peak is removed in place"


def test_adjacent_redundancy_shorter_cur_removed():
    prev = [_sf(mass=400.00, apex=3.00, left=2.90, right=3.10, h=9.0e5)]
    cur = [_sf(mass=400.02, apex=3.00, left=2.90, right=3.10, h=4.0e5)]
    out = _remove_peak_area_redundancy(prev, cur, mass_tol=0.05, apex_cap=0.03)
    assert out is None, "cur emptied -> None (mirrors C# return null)"
    assert len(prev) == 1 and prev[0].height == 9.0e5, "taller prev untouched"


def test_adjacent_redundancy_mass_too_far_keeps_both():
    prev = [_sf(mass=400.00, apex=3.00, left=2.90, right=3.10, h=5.0e5)]
    cur = [_sf(mass=400.20, apex=3.00, left=2.90, right=3.10, h=8.0e5)]  # 0.2 > 0.05
    out = _remove_peak_area_redundancy(prev, cur, mass_tol=0.05, apex_cap=0.03)
    assert out is not None and len(out) == 1
    assert len(prev) == 1, "mass too far apart -> no dedup"


# ---------------------------------------------------------------------------
# (d) empty scans -> []
# ---------------------------------------------------------------------------


def test_msdial_empty_scans_returns_empty():
    cfg = _default_cfg()
    assert find_lc_ms1_features_msdial([], config=cfg) == []


# ---------------------------------------------------------------------------
# mz_range restriction
# ---------------------------------------------------------------------------


def test_msdial_mz_range_restricts_slices():
    scans = _gaussian_scans(
        n_scans=80,
        ion_specs=[dict(center_mz=801.2096, apex_scan=37, sigma_scans=3.0,
                        apex_int=1.7e6, scan_lo=30, scan_hi=45, jitter=0.004)],
        seed=12345,
    )
    cfg = _default_cfg()
    # Band excluding 801 -> no slices -> []
    assert find_lc_ms1_features_msdial(scans, config=cfg, mz_range=(100.0, 500.0)) == []
    # Band including 801 -> exactly one hit
    res = find_lc_ms1_features_msdial(scans, config=cfg, mz_range=(750.0, 850.0))
    assert len(res) == 1 and abs(res[0].mz_centroid - 801.2096) < 0.02
