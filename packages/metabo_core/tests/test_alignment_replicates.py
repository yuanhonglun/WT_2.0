"""Regression tests for reference-replicate selection.

``align_features_across_replicates`` used to be tested here too. It was deleted
when DDA moved to ``metabo_core.alignment.joiner``; what remains in the module is
the quality score the joiner still uses to pick which sample seeds its buckets.
"""
import numpy as np

from metabo_core.alignment import reference_replicate_quality
from metabo_core.models import CandidateFeature


def _candidate(feature_id: str, sn: float | None, height: float = 1000.0) -> CandidateFeature:
    feat = CandidateFeature(
        feature_id=feature_id,
        segment_name="100-200",
        replicate_id=1,
        precursor_mz_nominal=100,
        rt_apex=5.0,
        rt_left=4.95,
        rt_right=5.05,
        ms2_mz=np.array([100.0, 150.0]),
        ms2_intensity=np.array([1000.0, 500.0]),
        n_fragments=2,
    )
    feat.ms1_height = height
    feat.ms1_sn = sn
    return feat


def test_quality_counts_and_averages_only_features_over_the_sn_gate():
    feats = [_candidate("A", 10.0), _candidate("B", 2.0), _candidate("C", 20.0)]
    n_passes, mean_sn, n_total = reference_replicate_quality(feats)
    assert (n_passes, mean_sn, n_total) == (2, 15.0, 3)


def test_more_passing_features_outranks_a_higher_mean_sn():
    """The key is compared lexicographically, so the count dominates."""
    many = [_candidate(f"m{i}", 5.0) for i in range(10)]
    pristine = [_candidate("p", 500.0)]
    assert reference_replicate_quality(many) > reference_replicate_quality(pristine)


def test_falls_back_to_feature_count_when_nothing_reports_sn():
    """No S/N anywhere degrades to the old "most features wins" heuristic."""
    small = [_candidate(f"s{i}", None) for i in range(3)]
    large = [_candidate(f"l{i}", None) for i in range(7)]
    assert reference_replicate_quality(small) == (0, 0.0, 3)
    assert reference_replicate_quality(large) > reference_replicate_quality(small)


def test_reads_gc_ms_dicts_and_their_apex_fallback():
    """GC-MS hands in dicts and often has no S/N at all; a tall apex stands in."""
    assert reference_replicate_quality([{"sn_ratio": 9.0}]) == (1, 9.0, 1)
    assert reference_replicate_quality([{"apex_intensity": 5000.0}]) == (1, 3.0, 1)
    assert reference_replicate_quality([{"apex_intensity": 10.0}]) == (0, 0.0, 1)


def test_empty_replicate_scores_zero():
    assert reference_replicate_quality([]) == (0, 0.0, 0)
    assert reference_replicate_quality(None) == (0, 0.0, 0)
