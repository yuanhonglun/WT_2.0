"""PR-6: stage 7 calls the refiner *after* it plumbs ``isotope_index``.

``joiner.build_feature`` deliberately leaves ``isotope_index`` unset — it needs
ASFAM's dedup-stage bookkeeping — and ``run_stage7`` copies it off the
representative peak afterwards. The refiner's second placement loop skips
unidentified isotope satellites by reading exactly that field.

Run the refiner one line too early and every ``isotope_index`` is still 0. A
tall M+1 satellite then wins the height sort, claims the master slot, and marks
some *other* compound's monoisotopic peak — a few mDa away, co-eluting — as
``cross_sample_redundant``. The output is wrong and every unit test still
passes. Hence this test, which pins the call order rather than the refiner.

The two peaks below are 5 mDa apart, inside the 0.02 Da gate, and 0.01 min
apart, inside the capped 0.05 min one. Neither is annotated, so both fall to the
height-ordered loop, where the satellite sorts first.
"""
from __future__ import annotations

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import CandidateFeature
from asfam.pipeline.stage7_alignment import run_stage7
from metabo_core.alignment.refiner import CROSS_SAMPLE_REDUNDANT


def _peak(feature_id: str, mz: float, rt_apex: float, height: float,
          isotope_index: int, fragment_mz: float) -> CandidateFeature:
    return CandidateFeature(
        feature_id=feature_id,
        segment_name="280-292",
        replicate_id=1,
        precursor_mz_nominal=int(round(mz)),
        rt_apex=rt_apex,
        rt_left=rt_apex - 0.1,
        rt_right=rt_apex + 0.1,
        # Distinct fragments so the joiner's MS2 cosine term makes each peak
        # score highest against the master it seeded, and both survive the claim.
        ms2_mz=np.array([fragment_mz, fragment_mz + 30.0], dtype=np.float64),
        ms2_intensity=np.array([1000.0, 500.0], dtype=np.float64),
        n_fragments=2,
        ms1_precursor_mz=mz,
        ms1_quant_mz=mz,
        ms1_height=height,
        isotope_index=isotope_index,
    )


def test_refiner_sees_isotope_index_and_spares_the_monoisotopic_peak():
    satellite = _peak("rep1_00000", 300.0000, 5.00, 9000.0,
                      isotope_index=1, fragment_mz=80.0)
    monoisotopic = _peak("rep1_00001", 300.0050, 5.01, 100.0,
                         isotope_index=0, fragment_mz=140.0)

    out = run_stage7({"1": [satellite, monoisotopic]}, ProcessingConfig())

    assert len(out) == 2, "the refiner marks; it never drops a row"
    by_mz = {round(f.precursor_mz, 4): f for f in out}
    mono, sat = by_mz[300.0050], by_mz[300.0000]

    assert sat.isotope_index == 1, "isotope_index must reach the Feature at all"
    # The satellite was skipped, so the real peak kept the slot. Run the refiner
    # before the plumbing above and this flips: the taller satellite masters and
    # the monoisotopic peak comes out cross_sample_redundant.
    assert not mono.is_duplicate
    assert mono.duplicate_type != CROSS_SAMPLE_REDUNDANT
    assert not sat.is_duplicate


def test_stage7_numbers_features_by_ascending_mz():
    """Step 6.4. The ids must already be m/z-ordered when alignment.eic is keyed."""
    peaks = [
        _peak("rep1_00000", 500.0, 3.0, 1000.0, isotope_index=0, fragment_mz=80.0),
        _peak("rep1_00001", 100.0, 9.0, 1000.0, isotope_index=0, fragment_mz=140.0),
        _peak("rep1_00002", 300.0, 1.0, 1000.0, isotope_index=0, fragment_mz=200.0),
    ]

    out = run_stage7({"1": peaks}, ProcessingConfig())

    assert [f.feature_id for f in out] == ["F00000", "F00001", "F00002"]
    assert [f.precursor_mz for f in out] == [100.0, 300.0, 500.0]
