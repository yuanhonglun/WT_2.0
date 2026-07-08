"""Regression tests for the metabo_core cross-replicate aligner."""
import numpy as np

from metabo_core.alignment import align_features_across_replicates
from metabo_core.config import AlignmentConfig
from metabo_core.models import CandidateFeature


def _candidate(feature_id: str, replicate_id: int, mz: float, rt: float, height: float = 1000.0) -> CandidateFeature:
    feat = CandidateFeature(
        feature_id=feature_id,
        segment_name="100-200",
        replicate_id=replicate_id,
        precursor_mz_nominal=int(round(mz)),
        rt_apex=rt,
        rt_left=rt - 0.05,
        rt_right=rt + 0.05,
        ms2_mz=np.array([100.0, 150.0]),
        ms2_intensity=np.array([1000.0, 500.0]),
        n_fragments=2,
    )
    feat.ms1_precursor_mz = mz
    feat.ms1_height = height
    feat.ms1_area = height * 2
    return feat


def test_align_features_uses_largest_replicate_as_reference():
    rep_a = [_candidate("A1", 1, 100.0, 5.0), _candidate("A2", 1, 200.0, 6.0)]
    rep_b = [_candidate("B1", 2, 100.001, 5.0), _candidate("B2", 2, 200.001, 6.0), _candidate("B3", 2, 300.0, 7.0)]
    result = align_features_across_replicates(
        {"a": rep_a, "b": rep_b},
        AlignmentConfig(rt_tolerance=0.1, mz_tolerance=0.02, match_threshold=0.5),
    )
    assert len(result) == 3
    rt_values = sorted(round(f.rt, 2) for f in result)
    assert rt_values == [5.0, 6.0, 7.0]


def test_align_features_quantifies_per_replicate_heights():
    rep_a = [_candidate("A1", 1, 100.0, 5.0, height=800.0)]
    rep_b = [_candidate("B1", 2, 100.001, 5.0, height=1200.0)]
    result = align_features_across_replicates(
        {"a": rep_a, "b": rep_b},
        AlignmentConfig(rt_tolerance=0.1, mz_tolerance=0.02, match_threshold=0.5),
    )
    assert len(result) == 1
    feat = result[0]
    assert set(feat.heights.keys()) == {"a", "b"}
    assert feat.mean_height == 1000.0
    assert feat.cv > 0.0


def test_align_features_returns_empty_for_no_replicates():
    assert align_features_across_replicates({}, AlignmentConfig()) == []
