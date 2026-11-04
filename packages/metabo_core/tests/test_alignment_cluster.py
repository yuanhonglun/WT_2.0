"""One-pass, route-agnostic clustering (SPEC 2026-08-02 §7, §8.1).

The properties that matter here are structural rather than numeric: that the
candidate key carries no m/z, that acceptance is complete-link and not
single-link, that a cluster may hold two observations of one sample, and that
every observation lands in exactly one cluster.
"""
from __future__ import annotations

import numpy as np
import pytest

from metabo_core.alignment.cluster import (
    Cluster, cluster_observations, observation_of,
)
from metabo_core.config import ClusterConfig, IdentityConfig
from metabo_core.models import CandidateFeature

RICH = [(100.0, 1000.0), (150.0, 700.0), (200.0, 300.0), (250.0, 120.0)]
#: Cosine 0.19 against RICH -- below theta_veto, so it contradicts even a shared
#: precursor.
CONTRADICT = [(100.0, 2.0), (150.0, 20.0), (200.0, 400.0), (250.0, 1000.0)]
#: Cosine 0.77 against RICH *and* 0.77 against CONTRADICT, which score 0.19
#: against each other: SAME with both ends of a pair that is DIFFERENT. This is
#: the fixture that separates complete-link from single-link.
MIDDLE = [(100.0, 775.7), (150.0, 559.9), (200.0, 595.0), (250.0, 1000.0)]


def _ms1(
    fid: str, mz: float, rt: float, *, spectrum=RICH, window: int = 300,
    segment: str = "seg", height: float = 1000.0,
) -> CandidateFeature:
    return CandidateFeature(
        feature_id=fid,
        segment_name=segment,
        replicate_id=1,
        precursor_mz_nominal=window,
        rt_apex=rt,
        rt_left=rt - 0.05,
        rt_right=rt + 0.05,
        ms2_mz=np.asarray([p[0] for p in spectrum], dtype=np.float64),
        ms2_intensity=np.asarray([p[1] for p in spectrum], dtype=np.float64),
        n_fragments=len(spectrum),
        ms1_precursor_mz=float(window) + 0.5,
        ms1_quant_mz=mz,
        ms1_height=height,
        ms1_area=height * 2,
        ms1_sn=10.0,
    )


def _ms2_only(
    fid: str, rep_mz: float, rt: float, *, spectrum=RICH, window: int = 300,
    segment: str = "seg", height: float = 1000.0, lowres: float | None = None,
) -> CandidateFeature:
    return CandidateFeature(
        feature_id=fid,
        segment_name=segment,
        replicate_id=1,
        precursor_mz_nominal=window,
        rt_apex=rt,
        rt_left=rt - 0.05,
        rt_right=rt + 0.05,
        ms2_mz=np.asarray([p[0] for p in spectrum], dtype=np.float64),
        ms2_intensity=np.asarray([p[1] for p in spectrum], dtype=np.float64),
        n_fragments=len(spectrum),
        ms1_precursor_mz=lowres,
        mz_source="ms1_lowres" if lowres is not None else "",
        ms1_quant_mz=None,
        ms2_rep_ion_mz=rep_mz,
        signal_type="ms2_only",
        ms1_height=height,
        ms1_area=height * 2,
        ms1_sn=10.0,
    )


def _run(features_by_sample, **kw) -> tuple[list[Cluster], object]:
    return cluster_observations(features_by_sample, ClusterConfig(**kw))


def _membership(clusters) -> set[frozenset[str]]:
    return {
        frozenset(f"{sample}:{peak.feature_id}" for sample, peak in c.members)
        for c in clusters
    }


# --- the point of the exercise ---------------------------------------------

def test_a_precursor_view_and_a_fragment_view_reach_one_cluster():
    """G1. Impossible under route-partitioned grouping, by construction.

    Sample 1 saw the compound's MS1 peak; sample 2 only ever saw its fragments.
    There is no m/z these two share -- one is a precursor, the other a fragment --
    which is exactly why the candidate key has no m/z in it.
    """
    clusters, stats = _run({
        "1": [_ms1("a", 300.20, 5.00)],
        "2": [_ms2_only("b", 150.0, 5.30)],
    })

    assert len(clusters) == 1
    assert clusters[0].is_cross_route
    assert clusters[0].tiers == {3}
    assert stats.n_cross_route_clusters == 1


def test_the_candidate_key_carries_no_mz():
    # Two observations 90 Da apart in quantitation ion, same window and RT, with
    # identical spectra. Any m/z-keyed index puts them in different buckets and
    # they never meet; here the spectra decide.
    clusters, _stats = _run({
        "1": [_ms2_only("a", 100.0, 5.00)],
        "2": [_ms2_only("b", 190.0, 5.02)],
    })
    assert len(clusters) == 1


def test_clustering_is_complete_link_not_single_link():
    """Cosine is not transitive, so a chain of pairwise SAME is not a compound.

    MIDDLE is SAME with both RICH and CONTRADICT; those two are DIFFERENT from
    each other. Single-link would chain all three into one row.
    """
    from metabo_core.algorithms.similarity import cosine_similarity
    theta = IdentityConfig().theta_same
    assert cosine_similarity(RICH, MIDDLE, 0.02)[0] >= theta
    assert cosine_similarity(CONTRADICT, MIDDLE, 0.02)[0] >= theta
    assert cosine_similarity(RICH, CONTRADICT, 0.02)[0] < theta

    clusters, _stats = _run({
        "1": [_ms2_only("rich", 150.0, 5.0, spectrum=RICH)],
        "2": [_ms2_only("middle", 150.0, 5.0, spectrum=MIDDLE)],
        "3": [_ms2_only("contra", 150.0, 5.0, spectrum=CONTRADICT)],
    })

    sizes = sorted(len(c.members) for c in clusters)
    assert sizes == [1, 2], "complete-link must refuse the third member"


def test_a_cluster_may_hold_two_observations_of_one_sample():
    """"One cell per sample" is a quantitation rule, not a grouping rule.

    Enforcing it during grouping is what forced the old joiner into a matching
    algorithm plus a conservation loop to rescue the losers it created.
    """
    clusters, stats = _run({
        "1": [
            _ms1("first", 300.20, 5.00, height=1000.0),
            _ms1("second", 300.21, 5.02, height=900.0),
        ],
        "2": [_ms1("other", 300.20, 5.01)],
    })

    assert len(clusters) == 1
    assert len(clusters[0].members) == 3
    assert stats.n_clusters_with_repeat_samples == 1


def test_every_observation_lands_in_exactly_one_cluster():
    """INV-5. Conservation is structural here rather than something to restore."""
    features = {
        "1": [_ms1("a", 300.20, 5.0), _ms2_only("b", 150.0, 5.3),
              _ms1("c", 410.10, 7.0, window=410, segment="two")],
        "2": [_ms1("d", 300.20, 5.1), _ms2_only("e", 190.0, 9.9),
              _ms1("f", 410.11, 7.1, window=410, segment="two")],
    }
    clusters, stats = _run(features)

    placed = [
        (sample, peak.feature_id)
        for cluster in clusters for sample, peak in cluster.members
    ]
    expected = [
        (sample, peak.feature_id)
        for sample, peaks in features.items() for peak in peaks
    ]
    assert sorted(placed) == sorted(expected)
    assert len(placed) == len(set(placed))
    assert stats.n_observations == len(expected)


# --- hard constraints the index and the rule share --------------------------

def test_segments_never_share_a_cluster():
    clusters, _stats = _run({
        "1": [_ms1("a", 300.20, 5.0, segment="one")],
        "2": [_ms1("b", 300.20, 5.0, segment="two")],
    })
    assert len(clusters) == 2


def test_windows_reach_one_apart_and_no_further():
    adjacent, _ = _run({
        "1": [_ms1("a", 300.20, 5.0, window=300)],
        "2": [_ms1("b", 300.20, 5.0, window=301)],
    })
    distant, _ = _run({
        "1": [_ms1("a", 300.20, 5.0, window=300)],
        "2": [_ms1("b", 300.20, 5.0, window=302)],
    })
    assert len(adjacent) == 1
    assert len(distant) == 2


def test_rt_reach_spans_buckets_rather_than_stopping_at_a_boundary():
    # A bucket edge is an artefact of the index, not of chromatography. Two
    # observations 0.02 min apart that straddle one must still meet.
    config = ClusterConfig()
    edge = config.rt_bucket
    clusters, _stats = _run({
        "1": [_ms1("a", 300.20, edge - 0.01)],
        "2": [_ms1("b", 300.20, edge + 0.01)],
    })
    assert len(clusters) == 1


def test_cross_route_pairs_get_the_wider_rt_window():
    # Tier 3 reaches 0.5 min, tiers 1 and 2 only 0.3 -- a fragment chromatogram
    # and its precursor's do not peak at the same time.
    config = IdentityConfig()
    drt = 0.5 * (config.rt_window + config.rt_window_cross)

    same_route, _ = _run({
        "1": [_ms1("a", 300.20, 5.0)],
        "2": [_ms1("b", 300.20, 5.0 + drt)],
    })
    cross_route, _ = _run({
        "1": [_ms1("a", 300.20, 5.0)],
        "2": [_ms2_only("b", 150.0, 5.0 + drt)],
    })
    assert len(same_route) == 2
    assert len(cross_route) == 1


# --- determinism ------------------------------------------------------------

def test_membership_does_not_depend_on_input_order():
    peaks = {
        "1": [_ms1("a", 300.20, 5.00), _ms2_only("b", 150.0, 5.30)],
        "2": [_ms1("c", 300.21, 5.02), _ms2_only("d", 190.0, 9.90)],
        "3": [_ms1("e", 410.10, 7.00, window=410, segment="two")],
    }
    forward, _ = _run(peaks)
    reversed_input, _ = _run({k: list(reversed(v)) for k, v in peaks.items()})
    assert _membership(forward) == _membership(reversed_input)


def test_unkeyed_observations_are_counted_not_silently_dropped():
    orphan = _ms2_only("orphan", 150.0, 5.0)
    orphan.ms2_rep_ion_mz = None            # no quantitation ion at all
    clusters, stats = _run({"1": [_ms1("a", 300.20, 5.0), orphan]})

    assert stats.n_unkeyed == 1
    assert stats.n_observations == 2
    assert _membership(clusters) == {frozenset({"1:a"})}


# --- the index must not silently outrun the rule ----------------------------

def test_a_bucket_narrower_than_the_widest_window_is_rejected_outright():
    """The failure it prevents is silent: the run completes, having quietly
    dropped every tier-3 candidate before the rule was asked."""
    with pytest.raises(ValueError, match="rt_bucket"):
        ClusterConfig(rt_bucket=0.3)
    with pytest.raises(ValueError, match="rt_bucket"):
        ClusterConfig(
            identity=IdentityConfig(rt_window_cross=0.9), rt_bucket=0.5,
        )


def test_observation_adapter_reports_the_ion_the_row_is_keyed_on():
    # ``align_mz`` falls back to ``ms1_precursor_mz`` when ``ms1_quant_mz`` is
    # unset, which is the ordinary DDA shape. Reading ``ms1_quant_mz`` alone
    # would call such a peak fragment-only and send it to the wrong tier.
    peak = _ms1("a", 300.20, 5.0)
    peak.ms1_quant_mz = None
    peak.ms1_precursor_mz = 300.20

    observation = observation_of(peak, lambda: RICH)

    assert observation.has_precursor
    assert observation.ms1_quant_mz == pytest.approx(300.20)
