"""Task D1: Stage 4 must label isotope-cluster members with ``isotope_index``.

PR-D makes every isotope peak (M / M+1 / M+2 ...) count as an independent
feature, per the MS-DIAL convention. Stage 4 already *marks-and-keeps* the
non-monoisotopic members (``status="isotope_excluded"``, ``is_duplicate=True``,
shared ``isotope_group_id``) instead of deleting them. Task D1 adds the
``isotope_index`` label: ``0`` for the monoisotopic representative, ``n`` for
the M+n member (``round(delta_mz / C13_DELTA)``).

Fixture note: ``run_stage4`` only forms an isotope edge when the EIC
co-elution gate (``base_ok``) or the MS2 isotope-step echo (``tier0_ok``)
holds — a bare feature list does NOT cluster. Here we drive ``base_ok`` by
supplying a ``RawSegmentData`` whose MS1 survey carries all three m/z as
co-eluting Gaussians at one RT, so the three peaks form a single cluster.
This mirrors ``test_stage4_cross_segment.py``'s ``_make_raw`` helper.
"""
from __future__ import annotations

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import CandidateFeature, RawSegmentData, ScanCycle
from asfam.pipeline.stage4_isotope_dedup import run_stage4


# A clean C13 ladder: 285.0500 (M), +1.003355 (M+1), +2x (M+2). Heights are
# strictly decreasing and stay well under the C13 M+1 physical limit
# (~0.39 at this m/z) so Stage 4's intensity-ratio sanity gate accepts the
# M+1 and M+2 edges.
_MZS = [285.0500, 286.0533, 287.0566]
_HEIGHTS = [10000.0, 1100.0, 200.0]
_CO_APEX_RT = 4.0
_SEG = "280-292"


def _gaussian(rt: np.ndarray, center: float, sigma: float, height: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((rt - center) / sigma) ** 2)


def _make_raw_multi(seg_name: str, n_cycles: int, mz_list: list[float],
                    height_list: list[float], co_apex_rt: float = _CO_APEX_RT,
                    sigma: float = 0.05, replicate_id: int = 1) -> RawSegmentData:
    """RawSegmentData whose MS1 survey carries every m/z in ``mz_list`` as a
    co-eluting Gaussian (so all extract as perfectly-correlated MS1 EICs).
    """
    rt = np.linspace(0.0, 10.0, n_cycles)
    profiles = [_gaussian(rt, co_apex_rt, sigma, h) for h in height_list]
    mz_arr = np.asarray(mz_list, dtype=np.float64)
    cycles = []
    for i in range(n_cycles):
        ms1_int = np.asarray([p[i] for p in profiles], dtype=np.float64)
        cycles.append(
            ScanCycle(
                cycle_index=i,
                rt=float(rt[i]),
                ms1_mz=mz_arr.copy(),
                ms1_intensity=ms1_int,
                ms2_scans={},
            )
        )
    seg_lo = int(np.floor(min(mz_list) - 5))
    seg_hi = int(np.ceil(max(mz_list) + 5))
    return RawSegmentData(
        file_path=f"/fake/{seg_name}.mzML",
        segment_name=seg_name,
        segment_low=seg_lo,
        segment_high=seg_hi,
        replicate_id=replicate_id,
        n_cycles=n_cycles,
        rt_array=rt.astype(np.float64),
        precursor_list=[],
        cycles=cycles,
        collision_energy=20.0,
    )


def _make_feature(feature_id: str, seg_name: str, mz: float,
                  rt_apex: float, height: float) -> CandidateFeature:
    return CandidateFeature(
        feature_id=feature_id,
        segment_name=seg_name,
        replicate_id=1,
        precursor_mz_nominal=int(round(mz)),
        rt_apex=rt_apex,
        rt_left=rt_apex - 0.1,
        rt_right=rt_apex + 0.1,
        ms2_mz=np.array([50.0, 80.0, 120.0], dtype=np.float64),
        ms2_intensity=np.array([1000.0, 500.0, 200.0], dtype=np.float64),
        n_fragments=3,
        ms1_precursor_mz=mz,
        ms1_height=height,
    )


def _build_and_run():
    """Build the M / M+1 / M+2 trio co-eluting at one RT and run Stage 4.

    Returns ``(f_m, f_m1, f_m2, out)``. The three features are mutated in
    place by ``run_stage4`` and are the same objects returned in ``out``.
    """
    cfg = ProcessingConfig()
    raw = _make_raw_multi(_SEG, n_cycles=800, mz_list=_MZS, height_list=_HEIGHTS)
    f_m = _make_feature("rep1_00000", _SEG, mz=_MZS[0], rt_apex=_CO_APEX_RT,
                        height=_HEIGHTS[0])
    f_m1 = _make_feature("rep1_00001", _SEG, mz=_MZS[1], rt_apex=_CO_APEX_RT,
                         height=_HEIGHTS[1])
    f_m2 = _make_feature("rep1_00002", _SEG, mz=_MZS[2], rt_apex=_CO_APEX_RT,
                         height=_HEIGHTS[2])
    features_by_rep = {"1": [f_m, f_m1, f_m2]}
    data_by_rep = {"1": [raw]}
    out = run_stage4(features_by_rep, cfg, data_by_replicate=data_by_rep)
    return f_m, f_m1, f_m2, out


def test_isotope_trio_assigns_mn_index():
    """M / M+1 / M+2 receive isotope_index 0 / 1 / 2."""
    f_m, f_m1, f_m2, _out = _build_and_run()

    # Guard: the three peaks must actually form ONE isotope cluster, else the
    # index assertions below would be vacuous. (Existing Stage-4 behavior.)
    assert f_m.isotope_group_id is not None
    assert f_m.isotope_group_id == f_m1.isotope_group_id == f_m2.isotope_group_id

    # New behavior (Task D1): sequential M+n index relative to the monoisotope.
    assert f_m.isotope_index == 0
    assert f_m1.isotope_index == 1
    assert f_m2.isotope_index == 2


def test_isotope_representative_kept_others_removed_but_retained():
    """Representative stays active (index 0); M+n members are flagged removed
    yet remain present in the returned list."""
    f_m, f_m1, f_m2, out = _build_and_run()

    # Guard: cluster formed.
    assert f_m.isotope_group_id == f_m1.isotope_group_id == f_m2.isotope_group_id

    # Representative = monoisotopic 285 peak: index 0, kept active.
    assert f_m.isotope_index == 0
    assert f_m.is_duplicate is False
    assert f_m.status == "active"

    # Non-representatives: flagged duplicate + isotope_excluded, with M+n index.
    for feat, expected_index in ((f_m1, 1), (f_m2, 2)):
        assert feat.is_duplicate is True
        assert feat.status == "isotope_excluded"
        assert feat.isotope_index == expected_index

    # All three survive in the returned replicate list (mark-and-keep, not
    # physical deletion). Compare by feature_id — CandidateFeature holds numpy
    # arrays, so set()/`in` membership would trip array-truthiness.
    returned_ids = sorted(f.feature_id for f in out["1"])
    assert returned_ids == ["rep1_00000", "rep1_00001", "rep1_00002"]
