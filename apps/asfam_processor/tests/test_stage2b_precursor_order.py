"""Stage 2.5 assigns MS2-only precursors in the design order.

strict MS1 -> relaxed MS1 (both stage 2, never seen here) -> neutral-loss
consensus -> isolation-window centroid stored by stage 2 -> integer channel.

Between 2026-06-25 and v1.4.260803 the window centroid was written into
``ms1_precursor_mz`` *inside stage 2*, and stage 2.5's entry filter ("no source
yet") then skipped neutral-loss inference for nearly every MS2-only feature
(49,168 of 50,032 on the rice data set carried ``ms1_lowres``). These tests pin
the restored order and the two things that come with it: the stored centroid is
the *last* fallback, and fragment-poor features are kept (tier D) by default.
"""
from __future__ import annotations

import numpy as np
import pytest

from asfam.config import ProcessingConfig
from asfam.models import CandidateFeature
from asfam.pipeline.stage2b_inference import STAGE2B_COUNTERS, run_stage2b

CAFFEINE = 195.0877
#: Three fragments that each sit one common neutral loss below the precursor
#: (H2O, NH3, CO) -> a 3-vote consensus at the true m/z.
NL_FRAGMENTS = (CAFFEINE - 18.01056, CAFFEINE - 17.02655, CAFFEINE - 27.99491)
#: Three fragments so light that no listed neutral loss reaches the window.
NO_NL_FRAGMENTS = (50.0, 60.0, 70.0)


def _ms2_only(fid, fragments, window_mz=195.11, **overrides) -> CandidateFeature:
    feat = CandidateFeature(
        feature_id=fid, segment_name="190-200", replicate_id=1,
        precursor_mz_nominal=195, rt_apex=5.0, rt_left=4.9, rt_right=5.1,
        ms2_mz=np.array(fragments, dtype=np.float64),
        ms2_intensity=np.linspace(1000.0, 500.0, len(fragments)),
        n_fragments=len(fragments),
        signal_type="ms2_only", ms2_rep_ion_mz=float(fragments[0]),
        ms1_window_mz=window_mz,
    )
    for key, value in overrides.items():
        setattr(feat, key, value)
    return feat


def _run(feats, **config_overrides):
    cfg = ProcessingConfig()
    for key, value in config_overrides.items():
        setattr(cfg, key, value)
    stats: dict = {}
    out = run_stage2b({"1": feats}, cfg, stats=stats)
    return out["1"], stats


def test_neutral_loss_consensus_comes_before_the_window_centroid():
    feat = _ms2_only("F1", NL_FRAGMENTS)
    kept, stats = _run([feat])

    assert kept == [feat]
    assert feat.mz_source == "nl_consensus"
    assert feat.mz_confidence == "high"
    assert feat.nl_n_support == 3
    assert abs(feat.inferred_mz - CAFFEINE) < 0.01
    assert abs(feat.precursor_mz - CAFFEINE) < 0.01
    # The centroid stays stored for the cross-check column; it was NOT used as
    # the precursor even though it was available.
    assert feat.ms1_precursor_mz is None
    assert feat.ms1_window_mz == pytest.approx(195.11)
    assert stats["nl_assigned"] == 1 and stats["window_fallback"] == 0


def test_window_centroid_is_the_fallback_when_inference_fails():
    feat = _ms2_only("F2", NO_NL_FRAGMENTS)
    kept, stats = _run([feat])

    assert kept == [feat]
    assert feat.mz_source == "ms1_lowres"
    assert feat.mz_confidence == "low"
    assert feat.nl_n_support is None
    assert feat.inferred_mz is None
    assert feat.ms1_precursor_mz == pytest.approx(195.11)
    assert feat.precursor_mz == pytest.approx(195.11)
    assert feat.signal_type == "ms2_only"
    assert stats["window_fallback"] == 1 and stats["nl_assigned"] == 0


def test_no_centroid_and_no_consensus_leaves_the_integer_channel():
    feat = _ms2_only("F3", NO_NL_FRAGMENTS, window_mz=None)
    kept, stats = _run([feat])

    assert kept == [feat]
    assert feat.mz_source == "" and feat.mz_confidence == ""
    assert feat.precursor_mz == 195.0
    assert stats["no_precursor"] == 1


def test_fragment_poor_features_are_retained_by_default_and_take_the_fallback():
    poor = _ms2_only("F4", NL_FRAGMENTS[:2])     # 2 fragments < min 3
    kept, stats = _run([poor])

    assert kept == [poor]
    assert poor.status == "active"
    # No neutral-loss inference below the fragment floor, however well the two
    # fragments would have voted: the floor is the rule, not a heuristic.
    assert poor.mz_source == "ms1_lowres"
    assert poor.nl_n_support is None
    assert stats["retained_fragment_poor"] == 1
    assert stats["discarded_fragment_poor"] == 0
    assert stats["window_fallback"] == 1


def test_fragment_poor_features_are_dropped_when_retention_is_off():
    poor = _ms2_only("F5", NL_FRAGMENTS[:2])
    rich = _ms2_only("F6", NL_FRAGMENTS)
    kept, stats = _run([poor, rich], retain_fragment_poor_ms2_only=False)

    assert kept == [rich]
    assert poor.status == "discarded_few_fragments"
    assert poor.mz_source == ""          # dropped before any fallback
    assert stats["discarded_fragment_poor"] == 1
    assert stats["retained_fragment_poor"] == 0


def test_features_that_already_have_a_source_are_left_alone():
    strict = _ms2_only("F7", NO_NL_FRAGMENTS, ms1_precursor_mz=195.0901,
                       mz_source="ms1_relaxed")
    kept, stats = _run([strict])

    assert kept == [strict]
    assert strict.mz_source == "ms1_relaxed"
    assert strict.ms1_precursor_mz == pytest.approx(195.0901)
    assert stats["ms2_only_in"] == 0


def test_counters_are_summed_over_replicates_and_cover_every_feature():
    feats = [
        _ms2_only("A", NL_FRAGMENTS),
        _ms2_only("B", NO_NL_FRAGMENTS),
        _ms2_only("C", NO_NL_FRAGMENTS, window_mz=None),
        _ms2_only("D", NL_FRAGMENTS[:1]),
    ]
    kept, stats = _run(feats)

    assert set(stats) == set(STAGE2B_COUNTERS)
    assert stats["ms2_only_in"] == 4
    assert stats["nl_assigned"] == 1
    assert stats["window_fallback"] == 2       # B and the fragment-poor D
    assert stats["no_precursor"] == 1
    assert stats["retained_fragment_poor"] == 1
    assert (stats["library_assigned"] + stats["nl_assigned"]
            + stats["window_fallback"] + stats["no_precursor"]) == stats["ms2_only_in"]
    assert len(kept) == 4
