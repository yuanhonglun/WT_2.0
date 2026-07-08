"""Regression test for isotope + adduct ``duplicate_group_id`` namespacing.

A previous DDA run showed RT-far features (e.g. RT 0.6 min and RT 5.9 min)
collapsed into one duplicate group because the isotope estimator and the
adduct grouper both wrote into ``duplicate_group_id`` with independent
0-based counters. The fix offsets the adduct namespace by
``DUP_GID_ADDUCT_OFFSET`` so the GUI's group-by-dup_gid view cannot fuse
unrelated groups.
"""
from __future__ import annotations

import numpy as np

from metabo_core.algorithms.adduct_grouping import (
    DUP_GID_ADDUCT_OFFSET,
    group_adducts,
)
from metabo_core.algorithms.isotope_estimator import estimate_isotopes
from metabo_core.models import CandidateFeature


def _make_feature(fid, mz, rt, intensity) -> CandidateFeature:
    return CandidateFeature(
        feature_id=fid,
        segment_name="rep1",
        replicate_id=0,
        precursor_mz_nominal=int(round(mz)),
        rt_apex=rt,
        rt_left=rt - 0.05,
        rt_right=rt + 0.05,
        ms2_mz=np.array([], dtype=np.float64),
        ms2_intensity=np.array([], dtype=np.float64),
        n_fragments=0,
        ms1_precursor_mz=mz,
        ms1_height=intensity,
    )


def test_dup_gid_namespaces_are_disjoint():
    # Isotope cluster at RT 0.6 (M, M+1, M+2 with [M+H]+ at 263 Da).
    iso_mono = _make_feature("iso0", 263.04, 0.61, 10_000.0)
    iso_m1 = _make_feature("iso1", 263.04 + 1.003355, 0.61, 1_000.0)
    iso_m2 = _make_feature("iso2", 263.04 + 2 * 1.003355, 0.61, 100.0)

    # Adduct pair at RT 5.9 ([M+H]+ 274.27 and [M+Na]+ 296.25).
    add_h = _make_feature("addH", 274.2739, 5.90, 8_000.0)
    add_na = _make_feature("addNa", 296.2558, 5.90, 6_000.0)

    features = [iso_mono, iso_m1, iso_m2, add_h, add_na]

    iso_next = estimate_isotopes(
        features,
        base_mz_tolerance=0.01,
        rt_tolerance=0.05,
        max_charge=1,
        max_isotope_step=3,
        group_id_start=0,
    )
    add_next = group_adducts(
        features,
        ionization_mode="positive",
        mw_tolerance=0.02,
        rt_tolerance=0.06,
        group_id_start=0,
    )

    iso_group_ids = {f.duplicate_group_id for f in (iso_mono, iso_m1, iso_m2)}
    add_group_ids = {f.duplicate_group_id for f in (add_h, add_na)}

    # Both functions started counting from 0, so a buggy implementation
    # writes the same numeric id into both groups. The fix offsets the
    # adduct namespace, so the two id sets must be disjoint.
    assert iso_group_ids & add_group_ids == set(), (
        f"isotope and adduct duplicate_group_id collide: "
        f"iso={iso_group_ids}, add={add_group_ids}"
    )

    # Adduct ids must live above the offset.
    for gid in add_group_ids:
        assert gid is not None
        assert gid >= DUP_GID_ADDUCT_OFFSET

    # And counters return the next-available id, sanity check.
    assert iso_next >= 1
    assert add_next >= 1


def test_two_rt_distant_adduct_pairs_get_distinct_gids():
    """Two RT-isolated adduct pairs must end up in separate groups."""
    # Pair A around RT 1.0
    a_h = _make_feature("a_h", 201.0073, 1.00, 1_000.0)
    a_na = _make_feature("a_na", 222.9893, 1.00, 800.0)
    # Pair B around RT 5.0 (far past rt_tolerance)
    b_h = _make_feature("b_h", 301.0073, 5.00, 600.0)
    b_na = _make_feature("b_na", 322.9893, 5.00, 400.0)

    features = [a_h, a_na, b_h, b_na]
    group_adducts(
        features,
        ionization_mode="positive",
        mw_tolerance=0.02,
        rt_tolerance=0.06,
        group_id_start=0,
    )

    # Both pairs should have been recognised but with different ids.
    assert a_h.duplicate_group_id is not None
    assert b_h.duplicate_group_id is not None
    assert a_h.duplicate_group_id != b_h.duplicate_group_id
