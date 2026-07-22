"""Dedup group ids are namespaced per sample before they reach alignment.

Stages 4/5/5b/6 restart their group counter at zero for every sample, so two
samples both hold a group 1006. Alignment copies the id off the representative
without remapping, and the GUI groups the aligned table by equality on
``duplicate_group_id`` — fusing the two into one displayed group. In the cancer
field test that put an m/z 426 feature, 2 min away, in an m/z 609 feature's
isotope group.

The remap lives at the orchestrator's read-back rather than at the four write
sites: the checkpoint fingerprint covers the config, not the stage output, so a
``_work/`` spill written before the fix is still reused — with its colliding ids
intact.
"""
from __future__ import annotations

import numpy as np
import pytest

from asfam.config import ProcessingConfig
from asfam.io import spill
from asfam.models import CandidateFeature
from asfam.pipeline.group_ids import SAMPLE_STRIDE, namespace_group_ids
from asfam.pipeline.orchestrator import PipelineOrchestrator
from asfam.pipeline.stage7_alignment import run_stage7


def _candidate(fid: str, mz: float, *, isotope_index: int = 0,
               group_id: int | None = 1006, rt: float = 4.0,
               height: float = 1000.0) -> CandidateFeature:
    """A minimal valid candidate carrying one dedup group's ids."""
    return CandidateFeature(
        feature_id=fid,
        segment_name=f"{int(mz)}-{int(mz) + 1}",
        replicate_id=1,
        precursor_mz_nominal=int(round(mz)),
        rt_apex=rt, rt_left=rt - 0.1, rt_right=rt + 0.1,
        ms2_mz=np.array([50.0, 80.0, 120.0], dtype=np.float64),
        ms2_intensity=np.array([1000.0, 500.0, 200.0], dtype=np.float64),
        n_fragments=3,
        ms1_precursor_mz=mz, ms1_height=height,
        isotope_index=isotope_index,
        isotope_group_id=group_id,
        adduct_group_id=group_id,
        duplicate_group_id=group_id,
    )


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------

def test_same_group_id_in_two_samples_stops_colliding():
    """The bug itself: both samples number a group 1006; they must not stay equal."""
    features = {
        "1": [_candidate("a", 426.223)],
        "2": [_candidate("b", 609.277)],
    }

    namespace_group_ids(features)

    a, b = features["1"][0], features["2"][0]
    for field in ("isotope_group_id", "adduct_group_id", "duplicate_group_id"):
        assert getattr(a, field) != getattr(b, field), field


def test_members_of_one_group_stay_together():
    """Namespacing must not shear a real group apart: one offset per sample."""
    features = {
        "1": [_candidate("mono", 426.223, isotope_index=0),
              _candidate("m+1", 427.226, isotope_index=1),
              _candidate("m+2", 428.229, isotope_index=2)],
    }

    namespace_group_ids(features)

    ids = {f.isotope_group_id for f in features["1"]}
    assert len(ids) == 1


def test_isotope_index_is_not_offset():
    """isotope_index is a position (0 = monoisotopic), not an id."""
    features = {
        "1": [_candidate("mono", 426.223, isotope_index=0)],
        "2": [_candidate("m+1", 609.277, isotope_index=1)],
    }

    namespace_group_ids(features)

    assert features["1"][0].isotope_index == 0
    assert features["2"][0].isotope_index == 1


def test_unassigned_group_id_stays_none():
    """A feature no dedup stage claimed has no group to namespace."""
    features = {"1": [_candidate("solo", 426.223, group_id=None)]}

    namespace_group_ids(features)

    f = features["1"][0]
    assert f.isotope_group_id is None
    assert f.adduct_group_id is None
    assert f.duplicate_group_id is None


def test_offset_does_not_depend_on_iteration_order():
    """Ids key on the sorted sample list, so a re-run cannot renumber them.

    Feature ids are only reproducible if the offsets are. A dict built in a
    different order — a resumed run scanning ``_work/`` rather than the input
    list — has to land on the same numbers.
    """
    forward = {"1": [_candidate("a", 426.223)], "2": [_candidate("b", 609.277)]}
    reversed_ = {"2": [_candidate("b", 609.277)], "1": [_candidate("a", 426.223)]}

    namespace_group_ids(forward)
    namespace_group_ids(reversed_)

    assert (forward["1"][0].duplicate_group_id
            == reversed_["1"][0].duplicate_group_id)
    assert (forward["2"][0].duplicate_group_id
            == reversed_["2"][0].duplicate_group_id)


def test_id_beyond_the_stride_is_refused():
    """A sample overflowing its range would collide again — silently. Fail loudly."""
    features = {
        "1": [_candidate("a", 426.223)],
        "2": [_candidate("b", 609.277, group_id=SAMPLE_STRIDE)],
    }

    with pytest.raises(ValueError, match="SAMPLE_STRIDE"):
        namespace_group_ids(features)


# ---------------------------------------------------------------------------
# End to end: spill -> orchestrator read-back -> stage 7
# ---------------------------------------------------------------------------

def _spill_two_colliding_samples(work_dir):
    """Two samples, each holding its own group 1006, far apart in m/z.

    The m/z gap (426 vs 609) is far wider than the joiner's tolerance, so the
    two never align into one spot: every aligned feature traces back to exactly
    one sample, and any shared group id across them is a collision.
    """
    spill.write_sample(
        work_dir / "1",
        [_candidate("s1_mono", 426.223, isotope_index=0),
         _candidate("s1_m+1", 427.226, isotope_index=1)],
        fingerprint=None, sample_id="1",
    )
    spill.write_sample(
        work_dir / "2",
        [_candidate("s2_mono", 609.277, isotope_index=0),
         _candidate("s2_m+1", 610.280, isotope_index=1)],
        fingerprint=None, sample_id="2",
    )


def test_aligned_features_never_share_a_group_across_samples(tmp_path):
    """The invariant that matters: equal group id => same sample.

    This is what the GUI's ``[f for f in features if f.duplicate_group_id == gid]``
    relies on. Fails on the unpatched pipeline, where both samples' groups are
    1006 and all four features come back as one group.
    """
    work_dir = tmp_path / "_work"
    _spill_two_colliding_samples(work_dir)

    orch = PipelineOrchestrator(ProcessingConfig())
    orch.work_dir = work_dir
    orch.sample_ids = ["1", "2"]

    features = run_stage7(orch._read_spilled_features(), ProcessingConfig())

    assert len(features) == 4
    # Without gap fill a spot's heights hold only the samples it was detected
    # in, and these spots are one sample each.
    by_sample: dict[str, set] = {}
    for f in features:
        (sample_id,) = f.heights
        by_sample.setdefault(sample_id, set()).add(f.duplicate_group_id)

    assert set(by_sample) == {"1", "2"}
    assert not by_sample["1"] & by_sample["2"]
    # ...and each sample's own group survived as one group, not two.
    assert len(by_sample["1"]) == 1
    assert len(by_sample["2"]) == 1


def test_reannotate_read_back_is_not_namespaced(tmp_path):
    """The other read-back writes the sample *back* to disk.

    Namespacing there would bake the offset into ``_work/`` and then apply it a
    second time when alignment reads the spill. Guards the comment at that call
    site: the ids on disk must survive a re-annotation unchanged.
    """
    work_dir = tmp_path / "_work"
    _spill_two_colliding_samples(work_dir)

    orch = PipelineOrchestrator(ProcessingConfig())
    orch.sample_ids = ["1", "2"]
    orch.run_reannotate(str(tmp_path), spectral_library_path=None,
                        work_dir=str(work_dir))

    for sample_id in ("1", "2"):
        for feat in spill.read_sample_features(work_dir / sample_id):
            assert feat.duplicate_group_id == 1006, (
                "the spill was rewritten with a namespaced id; alignment would "
                "offset it a second time"
            )
