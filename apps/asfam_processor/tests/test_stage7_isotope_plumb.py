"""Task D2: Stage 7 plumbs ``isotope_index`` / ``isotope_group_id`` from the
representative ``CandidateFeature`` onto the produced ``Feature``.

PR-D makes every isotope peak (M / M+1 / M+2 ...) an independent feature, per
the MS-DIAL convention. Task D1 labels ``CandidateFeature.isotope_index`` in
Stage 4, but ``features.csv`` is written from the post-alignment ``Feature``
object, not from ``CandidateFeature``. Stage 7 is the single place ``Feature``
objects are built, so it must copy the two isotope fields across.

Single-replicate alignment passes the reference ``CandidateFeature`` straight
through (no Hungarian matching against other replicates), so it is the minimal
case to assert the plumbing. ``data_by_replicate`` is accepted but unused by the
single-replicate path, so an empty list is supplied.

The ``CandidateFeature`` construction mirrors ``test_isotope_counting.py``'s
``_make_feature`` (a valid minimal candidate), plus the two isotope kwargs.
"""
from __future__ import annotations

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import CandidateFeature
from asfam.pipeline.stage7_alignment import run_stage7


def _make_feature(feature_id: str, mz: float, rt_apex: float, height: float,
                  isotope_index: int, isotope_group_id: int) -> CandidateFeature:
    return CandidateFeature(
        feature_id=feature_id,
        segment_name="280-292",
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
        isotope_index=isotope_index,
        isotope_group_id=isotope_group_id,
    )


def test_stage7_plumbs_isotope_index_and_group_single_replicate():
    """Single-replicate alignment copies isotope_index / group onto Feature."""
    cf = _make_feature("rep1_00000", mz=286.0533, rt_apex=4.0, height=1100.0,
                       isotope_index=2, isotope_group_id=7)

    out = run_stage7({"1": [cf]}, {"1": []}, ProcessingConfig())

    assert len(out) == 1
    assert out[0].isotope_index == 2
    assert out[0].isotope_group_id == 7
