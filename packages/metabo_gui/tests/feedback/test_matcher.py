"""Cross-run feature matching for FeedbackEntry."""
from __future__ import annotations

from dataclasses import dataclass

from metabo_gui.feedback.matcher import (
    MatchResult,
    match_entries_to_features,
    LC_MZ_PPM,
    LC_MZ_MIN_DA,
    GC_MZ_DA,
    RT_TOLERANCE_MIN,
)
from metabo_gui.feedback.models import FeatureSignature, FeedbackEntry


@dataclass(frozen=True)
class _Feat:
    """Test stand-in for a feature exposing the (id, signature) interface."""
    feature_id: str
    signature: FeatureSignature


def _entry(fid: str, mz: float, rt: float, mode: str = "dda") -> FeedbackEntry:
    return FeedbackEntry(
        feature_id_at_run=fid,
        feature_signature=FeatureSignature(mz=mz, rt=rt, mode=mode),
        tags=[], verified_good=False, comment="",
        created_at="t", updated_at="t", run_timestamp_created="t",
    )


def test_exact_id_match():
    e = _entry("F1", 280.0, 5.0)
    feats = [_Feat("F1", FeatureSignature(mz=280.0, rt=5.0, mode="dda"))]
    res = match_entries_to_features([e], feats)
    assert res.exact == {e.feature_id_at_run: feats[0]}
    assert res.auto_matched == {}
    assert res.ambiguous == {}
    assert res.orphan == []


def test_unique_signature_match_lc():
    e = _entry("F_old", 280.0, 5.0)
    feats = [_Feat("F_new", FeatureSignature(mz=280.0, rt=5.0, mode="dda"))]
    res = match_entries_to_features([e], feats)
    assert res.exact == {}
    assert res.auto_matched == {e.feature_id_at_run: feats[0]}


def test_multiple_signature_matches_ambiguous():
    e = _entry("F_old", 280.0, 5.0)
    f1 = _Feat("F_a", FeatureSignature(mz=280.0001, rt=5.0, mode="dda"))
    f2 = _Feat("F_b", FeatureSignature(mz=279.9999, rt=5.0, mode="dda"))
    res = match_entries_to_features([e], [f1, f2])
    assert e.feature_id_at_run in res.ambiguous
    assert set(res.ambiguous[e.feature_id_at_run]) == {f1, f2}


def test_no_match_is_orphan():
    e = _entry("F_old", 280.0, 5.0)
    feats = [_Feat("F_new", FeatureSignature(mz=900.0, rt=20.0, mode="dda"))]
    res = match_entries_to_features([e], feats)
    assert res.orphan == [e]


def test_mode_isolation():
    e = _entry("F_old", 280.0, 5.0, mode="asfam")
    feats = [_Feat("F_new", FeatureSignature(mz=280.0, rt=5.0, mode="dda"))]
    res = match_entries_to_features([e], feats)
    assert res.orphan == [e]
    assert res.auto_matched == {}


def test_gc_mode_uses_unit_mass_tolerance():
    e = _entry("F_old", 158.0, 7.5, mode="gcms:fullscan")
    feats = [_Feat("F_new", FeatureSignature(mz=158.4, rt=7.5, mode="gcms:fullscan"))]
    res = match_entries_to_features([e], feats)
    assert res.auto_matched == {e.feature_id_at_run: feats[0]}


def test_rt_outside_tolerance_is_orphan():
    e = _entry("F_old", 280.0, 5.0)
    feats = [_Feat("F_new", FeatureSignature(mz=280.0, rt=6.0, mode="dda"))]
    res = match_entries_to_features([e], feats)
    assert res.orphan == [e]


def test_lc_ppm_floor():
    e = _entry("F_old", 100.0, 5.0)
    # 100 * 5e-6 = 0.0005 Da, floor is 0.001 Da → 0.0008 within tolerance
    feats = [_Feat("F_new", FeatureSignature(mz=100.0008, rt=5.0, mode="dda"))]
    res = match_entries_to_features([e], feats)
    assert res.auto_matched == {e.feature_id_at_run: feats[0]}


def test_tags_mutation_does_not_break_lookup():
    """Regression: mutating tags on an entry must not affect MatchResult lookup.

    With the old frozen+__hash__ approach, appending to entry.tags after
    insertion would change the hash and make the entry silently unfindable.
    With the new str-key approach, this hazard is gone entirely.
    """
    e = _entry("F1", 280.0, 5.0)
    feats = [_Feat("F1", FeatureSignature(mz=280.0, rt=5.0, mode="dda"))]
    res = match_entries_to_features([e], feats)
    # mutate tags after matching — must not affect lookup
    e.tags.append("mutated")
    assert e.feature_id_at_run in res.exact
    assert res.exact[e.feature_id_at_run] is feats[0]
