"""Tests for the MS-DIAL-style peak joiner (metabo_core.alignment.joiner)."""
from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest

from metabo_core.alignment.joiner import (
    _build_master_list,
    _Ms2Cache,
    _score,
    build_feature,
    default_ms2_reader,
    join_features,
    join_spots,
    select_reference_sample,
    JoinStats,
)
from metabo_core.config import JoinerConfig
from metabo_core.models import MS1, PRODUCT, AnnotationMatch, CandidateFeature


def _cache(config: JoinerConfig | None = None) -> _Ms2Cache:
    """Spectra straight off the feature objects — what the joiner does by default."""
    size = config.ms2_cache_size if config else 4096
    return _Ms2Cache(default_ms2_reader, size)


def _feat(
    fid: str,
    mz: float,
    rt: float,
    *,
    height: float = 1000.0,
    sn: float = 10.0,
    ms2: list[tuple[float, float]] | None = None,
    total_score: float | None = None,
    quant_mz: float | None = None,
) -> CandidateFeature:
    """An ``ms1_detected`` feature. ``mz`` is the *reported* precursor.

    ``quant_mz`` is the ion the height was actually read off — what the joiner
    matches on. Left unset it defaults to ``mz``, which is the DDA shape (there
    the precursor *is* a picked MS1 ion). Set them apart to get the ASFAM shape,
    where the precursor is a 1-Da isolation window's centroid and only
    ``ms1_quant_mz`` names a real ion.
    """
    peaks = ms2 or []
    matches = []
    name = None
    if total_score is not None:
        name = f"cmp_{fid}"
        matches = [AnnotationMatch(rank=1, name=name, total_score=total_score,
                                   score=total_score)]
    return CandidateFeature(
        matchms_name=name,
        feature_id=fid,
        segment_name="seg",
        replicate_id=1,
        precursor_mz_nominal=int(mz),
        rt_apex=rt,
        rt_left=rt - 0.05,
        rt_right=rt + 0.05,
        ms2_mz=np.asarray([p[0] for p in peaks], dtype=np.float64),
        ms2_intensity=np.asarray([p[1] for p in peaks], dtype=np.float64),
        n_fragments=len(peaks),
        ms1_precursor_mz=mz,
        ms1_quant_mz=quant_mz,
        ms1_height=height,
        ms1_area=height * 2,
        ms1_sn=sn,
        annotation_matches=matches,
    )


def _ms2_only(
    fid: str,
    rep_ion_mz: float | None,
    rt: float,
    *,
    window: int,
    precursor_mz: float | None = None,
    height: float = 1000.0,
    ms2: list[tuple[float, float]] | None = None,
) -> CandidateFeature:
    """An ``ms2_only`` feature: quantified on a fragment, not on any MS1 ion.

    ``precursor_mz`` is the low-res centroid of the whole isolation window that
    stage 2 falls back to — noise, near enough to an MS1 feature's own window
    centroid (median 0.005 Da apart) to be claimed into its spot. That is the
    mixed-route bug the joiner's guard exists to stop.
    """
    peaks = ms2 or []
    return CandidateFeature(
        feature_id=fid,
        segment_name="seg",
        replicate_id=1,
        precursor_mz_nominal=window,
        rt_apex=rt,
        rt_left=rt - 0.05,
        rt_right=rt + 0.05,
        ms2_mz=np.asarray([p[0] for p in peaks], dtype=np.float64),
        ms2_intensity=np.asarray([p[1] for p in peaks], dtype=np.float64),
        n_fragments=len(peaks),
        ms1_precursor_mz=precursor_mz,
        ms1_quant_mz=None,
        ms2_rep_ion_mz=rep_ion_mz,
        signal_type="ms2_only",
        ms1_height=height,
        ms1_area=height * 2,
        ms1_sn=10.0,
    )


def _config(**kw) -> JoinerConfig:
    base = dict(mz_tolerance=0.015, rt_tolerance=0.1)
    base.update(kw)
    return JoinerConfig(**base)


# ---------------------------------------------------------------------------
# Union master list
# ---------------------------------------------------------------------------

def test_union_master_keeps_features_absent_from_the_reference():
    """A feature only the non-reference sample saw must survive.

    This is the 34% silent loss the reference-seeded master list caused: the
    old core built its master list from the reference replicate alone, so
    ``only_in_b`` had nothing to match against and vanished.
    """
    shared = _feat("a1", 200.0, 1.0)
    features = {
        # 'a' wins the automatic reference: more peaks above the S/N gate.
        "a": [shared, _feat("a2", 300.0, 2.0)],
        "b": [_feat("b1", 200.0, 1.0), _feat("only_in_b", 400.1234, 3.0)],
    }
    spots, stats = join_spots(features, _config())

    assert stats.reference_sample_id == "a"
    assert stats.n_added_by_sample["b"] == 1
    mzs = sorted(round(s.master_mz, 4) for s in spots)
    assert mzs == [200.0, 300.0, 400.1234]

    only = next(s for s in spots if s.master_mz == pytest.approx(400.1234))
    assert set(only.peaks) == {"b"}
    assert only.origin_sample_id == "b"


def test_master_list_is_order_independent_of_dict_insertion():
    """Samples merge in sorted(sample_id) order, not dict order."""
    a = [_feat("a1", 100.0, 1.0)]
    b = [_feat("b1", 100.02, 1.0)]   # 0.02 Da away: outside a 0.015 tol
    cfg = _config()

    forward = _build_master_list({"a": a, "b": b}, "a", cfg, JoinStats(), _cache(cfg))
    reverse = _build_master_list({"b": b, "a": a}, "a", cfg, JoinStats(), _cache(cfg))

    assert [(m.mz, m.sample_id) for m in forward] == [(m.mz, m.sample_id) for m in reverse]
    assert len(forward) == 2


def test_bucketing_finds_every_pair_brute_force_finds():
    """The +/-1 bucket scan must never miss an is_similar_to pair.

    A miss would silently duplicate a master peak. Bucket width is 2*mz_tol, so
    a peak at the very edge of its bucket still sees everything within mz_tol.
    """
    rng = random.Random(20260709)
    cfg = _config(mz_tolerance=0.015, rt_tolerance=0.1)

    ref = [_feat(f"r{i}", rng.uniform(100.0, 110.0), rng.uniform(0.0, 5.0))
           for i in range(500)]
    other = [_feat(f"o{i}", rng.uniform(100.0, 110.0), rng.uniform(0.0, 5.0))
             for i in range(500)]

    master = _build_master_list({"a": ref, "b": other}, "a", cfg, JoinStats(), _cache(cfg))
    kept_from_b = {m.feature.feature_id for m in master if m.sample_id == "b"}

    # Brute force: replay the same greedy merge, comparing against every master
    # peak instead of only the neighbouring buckets.
    seen = [(f.precursor_mz, f.rt_apex) for f in ref]
    expected_kept = set()
    for f in other:
        similar = any(
            abs(mz - f.precursor_mz) <= cfg.mz_tolerance
            and abs(rt - f.rt_apex) <= cfg.rt_tolerance
            for mz, rt in seen
        )
        if not similar:
            expected_kept.add(f.feature_id)
            seen.append((f.precursor_mz, f.rt_apex))

    assert kept_from_b == expected_kept


# ---------------------------------------------------------------------------
# Greedy claim
# ---------------------------------------------------------------------------

def test_a_losing_peak_lands_on_its_best_available_master():
    """Outbid for its favourite, a peak takes its best *available* master.

    MS-DIAL folds "can I outbid the holder" into the argmax itself
    (``LcmsPeakJoiner.cs:161``), so a master already held by a better bid is
    skipped while the peak is still choosing. ``AlignPeaksToMasterOverlapTest``
    pins it.

    We used to take the plain argmax and *then* bid, so a peak that lost fell
    out of the run entirely — it had no master of its own either, because the
    build step had suppressed it as a duplicate of the very master it just lost.
    3.1% of all detected peaks disappeared this way, and the spot that suppressed
    them gap-filled their sample instead. The docstring that used to sit here
    claimed this was "exactly as LcmsPeakJoiner.cs:148-166 does". It was not; it
    misread the C#.
    """
    # Two masters 0.15 min apart: further than rt_tol, so both survive the build
    # step, and 'weak' at 2.07 is the only peak whose window covers both.
    features = {
        "a": [_feat("a1", 500.0, 2.0), _feat("a2", 500.0, 2.15)],
        "b": [
            _feat("strong", 500.0, 2.001, height=1.0),    # only master@2.0 is in range
            _feat("weak", 500.0, 2.07, height=9999.0),    # prefers 2.0, can reach 2.15
        ],
    }
    spots, stats = join_spots(features, _config(ms2_weight=0.0, reference_sample="a"))

    assert len(spots) == 2
    at_2_00 = next(s for s in spots if s.master_rt == pytest.approx(2.0))
    at_2_15 = next(s for s in spots if s.master_rt == pytest.approx(2.15))

    # 'strong' outbids 'weak' for master@2.0 ...
    assert at_2_00.peaks["b"].feature_id == "strong"
    # ... and 'weak' falls to master@2.15 rather than out of the run. Under the
    # old rule it vanished and this spot gap-filled sample b.
    assert at_2_15.peaks["b"].feature_id == "weak"
    assert stats.matched_by_sample["b"] == (2, 0)      # both peaks placed
    # Height still decides nothing: 'weak' is 10^4 x taller and lost anyway.


def test_ms2_cosine_breaks_a_tie_that_mz_and_rt_cannot():
    """Two co-eluting isobars: the MS2 term must pick the right partner."""
    mz, rt = 350.0, 1.5
    spec_x = [(100.0, 100.0), (150.0, 50.0)]
    spec_y = [(120.0, 100.0), (170.0, 50.0)]

    features = {
        "a": [_feat("ax", mz, rt, ms2=spec_x)],
        "b": [
            _feat("by", mz, rt, ms2=spec_y, height=5000.0),   # identical m/z & RT
            _feat("bx", mz, rt, ms2=spec_x, height=10.0),
        ],
    }
    spots, _ = join_spots(features, _config(reference_sample="a"))
    assert len(spots) == 1
    assert spots[0].peaks["b"].feature_id == "bx"

    # With the MS2 term switched off the score is a pure m/z+RT tie, and the
    # first bidder keeps the master — spectra no longer matter.
    spots_no_ms2, _ = join_spots(features, _config(ms2_weight=0.0, reference_sample="a"))
    assert spots_no_ms2[0].peaks["b"].feature_id == "by"


def test_score_without_ms2_renormalizes_to_the_mz_rt_pair():
    """A no-MS2 pair scores on the same [0,1] scale as an MS2-bearing one.

    Otherwise match_threshold would mean two different things and every no-MS2
    pair would score at most mz_weight + rt_weight = 0.6.
    """
    cfg = _config()
    perfect_no_ms2 = _score(0.0, 0.0, [], [], cfg)
    perfect_with_ms2 = _score(0.0, 0.0, [(100.0, 1.0)], [(100.0, 1.0)], cfg)
    assert perfect_no_ms2 == pytest.approx(1.0)
    assert perfect_with_ms2 == pytest.approx(1.0)

    # And it reproduces the pre-PR-4 blend: 0.6 * gauss + 0.4 * ms2_cos.
    peaks_a, peaks_b = [(100.0, 1.0), (200.0, 1.0)], [(100.0, 1.0), (300.0, 1.0)]
    dmz, drt = 0.005, 0.02
    gauss = 0.5 * np.exp(-0.5 * (dmz / cfg.mz_tolerance) ** 2) \
        + 0.5 * np.exp(-0.5 * (drt / cfg.rt_tolerance) ** 2)
    ms2_cos = 0.5   # one of two unit peaks matches
    assert _score(dmz, drt, peaks_a, peaks_b, cfg) == pytest.approx(
        0.6 * gauss + 0.4 * ms2_cos,
    )


def test_ms2_reader_is_called_once_per_spectrum_not_per_edge():
    """Reads are O(peaks touched), not O(candidate edges): the LRU absorbs repeats.

    Guards the memory fix. Reading a spectrum per candidate edge is what made
    the dense-matrix scorer need every replicate's spectra resident at once.
    """
    calls: list[tuple[str, str]] = []

    def reader(sample_id, feature):
        calls.append((sample_id, feature.feature_id))
        if feature.ms2_mz is None or len(feature.ms2_mz) == 0:
            return None
        return feature.ms2_mz, feature.ms2_intensity

    spec = [(100.0, 1.0)]
    # Each of b's peaks bids on a's single master, plus a1 self-matches, so the
    # master's spectrum is *asked for* four times. It must be read once.
    features = {
        "a": [_feat("a1", 500.0, 2.0, ms2=spec)],
        "b": [_feat(f"b{i}", 500.0 + 0.001 * i, 2.0 + 0.001 * i, ms2=spec)
              for i in range(3)],
    }
    spots, stats = join_spots(features, _config(reference_sample="a"), ms2_reader=reader)

    assert len(spots) == 1
    assert stats.n_candidate_edges == 4          # a1 + b0..b2, one master each
    assert stats.n_ms2_cache_hits > 0

    # The master's spectrum is *asked for* on all four edges and read once — that
    # is what the LRU is for. a1 is read once even though it is both a master and
    # a target of its own.
    assert calls.count(("a", "a1")) == 1
    # Reads are O(peaks), never O(edges). Two of b's three peaks lose the claim,
    # and the lost-peak accounting weighs each loser against the winner *after*
    # the claim, so those spectra are read a second time: O(lost), still not
    # O(edges). Nothing is read more than twice.
    assert stats.n_lost_peaks == 2
    assert max(Counter(calls).values()) <= 2, f"a spectrum was read 3+ times: {calls}"
    assert stats.n_ms2_reads <= len(features["a"]) + len(features["b"]) + 2 * stats.n_lost_peaks


# ---------------------------------------------------------------------------
# The quantitation key: align_mz, and the route guard
# ---------------------------------------------------------------------------

def test_peaks_match_on_the_quantitation_ion_not_the_precursor():
    """One compound, one real ion, two samples whose *precursors* disagree.

    This is the ASFAM shape. Both samples measured the height off m/z 200.00, but
    sample b co-isolated something else in the same 1-Da window, dragging its
    intensity-weighted precursor centroid to 200.05. Matching on the precursor
    puts the two peaks 0.05 Da apart — more than three times the tolerance — and
    splits one compound into two rows, each detected in one sample, each
    gap-filling the other. Matching on the ion the height came off puts them
    0.001 Da apart.
    """
    features = {
        "a": [_feat("a1", 200.000, 1.0, quant_mz=200.000)],
        "b": [_feat("b1", 200.050, 1.0, quant_mz=200.001)],
    }
    spots, stats = join_spots(features, _config(reference_sample="a"))

    assert len(spots) == 1, "the precursor drift split one compound in two"
    assert set(spots[0].peaks) == {"a", "b"}
    assert stats.n_added_by_sample["b"] == 0
    # The spot is keyed on the real ion; the *reported* precursor is untouched.
    assert spots[0].master_mz == pytest.approx(200.000)
    assert build_feature(spots[0], "F00000").precursor_mz == pytest.approx(200.000)


def test_peaks_of_different_quantitation_routes_never_share_a_spot():
    """Same m/z, same RT, different chromatogram: two rows, not one.

    An ms1_detected height is read off a +/-0.01 Da MS1 window; an ms2_only
    height off a +/-0.1 Da product-ion slice. Merging them puts two numbers
    measured on different things in one row, and its CV and fold change stop
    meaning anything. 11% of exported rows were like this.

    Here the two even agree on ``align_mz``, so the m/z and RT boxes cannot tell
    them apart — only the route guard can.
    """
    features = {
        "a": [_feat("ms1", 100.05, 5.0, quant_mz=100.05)],
        "b": [_ms2_only("frag", 100.05, 5.0, window=300)],
    }
    spots, stats = join_spots(features, _config(reference_sample="a"))

    assert len(spots) == 2
    assert {s.quant_route for s in spots} == {MS1, PRODUCT}
    for spot in spots:
        routes = {p.quant_route for p in spot.detected_peaks}
        assert len(routes) == 1, f"spot {spot.index} mixes routes: {routes}"
    assert stats.spots_by_route == {MS1: 1, PRODUCT: 1}


def test_ms2_only_peaks_need_the_same_isolation_window_to_merge():
    """A shared fragment is not a shared compound.

    ms2_only peaks are keyed on a fragment, and common fragments (85.03, 69.03)
    are thrown by many different precursors. Two of them co-eluting would merge
    on m/z + RT alone, so the box also demands they came out of the same
    isolation window.
    """
    apart = {
        "a": [_ms2_only("a1", 85.03, 5.0, window=200)],
        "b": [_ms2_only("b1", 85.03, 5.0, window=300)],
    }
    spots, _ = join_spots(apart, _config(reference_sample="a"))
    assert len(spots) == 2, "two precursors sharing one fragment were merged"

    # Control: same window, same fragment, same RT — that *is* one compound.
    together = {
        "a": [_ms2_only("a1", 85.03, 5.0, window=200)],
        "b": [_ms2_only("b1", 85.03, 5.0, window=200)],
    }
    spots, _ = join_spots(together, _config(reference_sample="a"))
    assert len(spots) == 1
    assert set(spots[0].peaks) == {"a", "b"}


def test_ms1_peaks_may_merge_across_isolation_windows():
    """The window is a condition on the product-ion route only.

    A precursor sitting near a window edge lands in window 199 in one sample and
    200 in the next, but its base peak — the ion both heights were read off — is
    the same. Demanding the same window here would re-split exactly what this
    change exists to join.
    """
    features = {
        "a": [_feat("a1", 199.98, 1.0, quant_mz=200.001)],
        "b": [_feat("b1", 200.02, 1.0, quant_mz=200.002)],
    }
    features["a"][0].precursor_mz_nominal = 199
    features["b"][0].precursor_mz_nominal = 200

    spots, _ = join_spots(features, _config(reference_sample="a"))
    assert len(spots) == 1
    assert set(spots[0].peaks) == {"a", "b"}


def test_a_peak_with_no_quantitation_ion_is_counted_not_silently_dropped():
    """``align_mz is None`` -> it cannot be aligned onto an ion or filled on one.

    Measured incidence on both ASFAM benchmarks: zero. The joiner still has to
    say so out loud rather than key the peak on something it was never measured
    on.
    """
    features = {
        "a": [_feat("a1", 250.0, 1.0, quant_mz=250.0),
              _ms2_only("bad", None, 3.0, window=250)],       # no rep ion at all
        "b": [_feat("b1", 250.0, 1.0, quant_mz=250.0)],
    }
    spots, stats = join_spots(features, _config(reference_sample="a"))

    assert stats.n_unkeyed_peaks == 1
    assert len(spots) == 1                       # the good pair, joined
    assert set(spots[0].peaks) == {"a", "b"}
    assert all("bad" != p.feature_id
               for s in spots for p in s.peaks.values())


# ---------------------------------------------------------------------------
# Per-spot representative
# ---------------------------------------------------------------------------

def test_representative_comes_from_the_best_annotated_sample_not_the_reference():
    """The reference's poor spectrum must not drag the spot's annotation down."""
    good_spec = [(100.0, 100.0), (150.0, 80.0), (200.0, 60.0)]
    poor_spec = [(100.0, 100.0)]

    features = {
        # 'a' is the reference (higher S/N), but its spectrum scored badly.
        "a": [_feat("a1", 250.0, 1.0, sn=50.0, ms2=poor_spec, total_score=0.30)],
        "b": [_feat("b1", 250.0, 1.0, sn=4.0, ms2=good_spec, total_score=0.91)],
    }
    spots, stats = join_spots(features, _config())

    assert stats.reference_sample_id == "a"
    assert len(spots) == 1
    assert spots[0].representative_sample_id == "b"

    feature = build_feature(spots[0], "F00000")
    assert feature.name == "cmp_b1"
    assert feature.annotation_matches[0].total_score == pytest.approx(0.91)
    assert feature.n_fragments == 3
    np.testing.assert_allclose(feature.ms2_mz, [100.0, 150.0, 200.0])
    # Quantitation still spans every sample that matched.
    assert set(feature.heights) == {"a", "b"}


def test_representative_prefers_ms2_over_a_higher_scoring_bare_peak():
    """MS2 presence outranks (TotalScore, height) — it is the first sort key."""
    features = {
        "a": [_feat("a1", 250.0, 1.0, height=9e6, total_score=0.99)],   # no MS2
        "b": [_feat("b1", 250.0, 1.0, height=1.0, ms2=[(50.0, 1.0)], total_score=0.10)],
    }
    spots, _ = join_spots(features, _config())
    assert spots[0].representative_sample_id == "b"


def test_representative_ties_break_on_height_then_sample_id():
    features = {
        "a": [_feat("a1", 250.0, 1.0, ms2=[(50.0, 1.0)], total_score=0.5, height=10.0)],
        "b": [_feat("b1", 250.0, 1.0, ms2=[(50.0, 1.0)], total_score=0.5, height=99.0)],
    }
    spots, _ = join_spots(features, _config())
    assert spots[0].representative_sample_id == "b"    # taller peak wins

    same = {
        "a": [_feat("a1", 250.0, 1.0, ms2=[(50.0, 1.0)], total_score=0.5, height=10.0)],
        "b": [_feat("b1", 250.0, 1.0, ms2=[(50.0, 1.0)], total_score=0.5, height=10.0)],
    }
    spots, _ = join_spots(same, _config())
    assert spots[0].representative_sample_id == "a"    # fully tied: lowest id


# ---------------------------------------------------------------------------
# Assembly + reference selection
# ---------------------------------------------------------------------------

def test_means_and_cv_ignore_zero_and_missing_samples():
    features = {
        "a": [_feat("a1", 250.0, 1.0, height=100.0)],
        "b": [_feat("b1", 250.0, 1.0, height=300.0)],
        "c": [_feat("c1", 900.0, 4.0)],       # unrelated spot
    }
    feats = join_features(features, _config())
    spot = next(f for f in feats if f.precursor_mz == pytest.approx(250.0))

    assert set(spot.heights) == {"a", "b"}    # 'c' never matched; no zero filled in
    assert spot.mean_height == pytest.approx(200.0)
    assert spot.cv == pytest.approx(np.std([100.0, 300.0]) / 200.0)


def test_reference_sample_can_be_pinned_and_a_stale_name_falls_back():
    features = {
        "a": [_feat("a1", 200.0, 1.0, sn=50.0), _feat("a2", 300.0, 2.0, sn=50.0)],
        "b": [_feat("b1", 200.0, 1.0, sn=4.0)],
    }
    assert select_reference_sample(features) == "a"
    assert select_reference_sample(features, "b") == "b"
    # A saved config naming a sample that is not in this run must not raise.
    assert select_reference_sample(features, "gone") == "a"


def test_pinned_reference_only_changes_who_seeds_not_what_survives():
    """The union means the reference choice no longer decides which peaks exist."""
    features = {
        "a": [_feat("a1", 200.0, 1.0), _feat("a2", 300.0, 2.0)],
        "b": [_feat("b1", 200.0, 1.0), _feat("b2", 400.0, 3.0)],
    }
    from_a = join_features(features, _config(reference_sample="a"))
    from_b = join_features(features, _config(reference_sample="b"))

    assert sorted(round(f.precursor_mz, 4) for f in from_a) == [200.0, 300.0, 400.0]
    assert sorted(round(f.precursor_mz, 4) for f in from_b) == [200.0, 300.0, 400.0]


def test_spots_are_counted_by_quantitation_route():
    """The log has to say how many spots each route produced.

    Replaces a nominal-vs-exact m/z tally that only ever counted the ms2_only
    spots keyed on an isolation-window floor — an artefact of matching on the
    precursor, which is exactly what ``align_mz`` removes.
    """
    features = {
        "a": [_feat("a1", 250.0, 1.0), _ms2_only("a2", 85.03, 3.0, window=250)],
        "b": [_feat("b1", 250.0, 1.0)],
    }
    _spots, stats = join_spots(features, _config())
    assert stats.spots_by_route == {MS1: 1, PRODUCT: 1}


def test_empty_input_returns_no_spots():
    assert join_features({}, _config()) == []
    assert join_features({"a": []}, _config()) == []


# ---------------------------------------------------------------------------
# The MS2 identity gate on the box
# ---------------------------------------------------------------------------

# Two spectra with no fragment in common: cosine 0. Three fragments each, which
# is the default ms2_identity_min_fragments — enough for the gate to rule.
_SPEC_X = [(100.0, 100.0), (150.0, 80.0), (200.0, 60.0)]
_SPEC_Y = [(110.0, 100.0), (160.0, 80.0), (210.0, 60.0)]


def test_a_different_compound_founds_its_own_master_instead_of_vanishing():
    """Proximity is not identity, and the build step may only suppress identity.

    ``b1`` sits inside ``a1``'s box on every geometric axis, so the build step
    used to forbid it from founding a master — while the claim step lets one
    sample put one peak in a master. Its MS2 says it is a different compound. It
    must get a master of its own; the alternative is the peak being deleted from
    the run, which is what this gate exists to stop (1,242 rice / 2,026 cancer).
    """
    features = {
        "a": [_feat("a1", 200.0, 1.0, ms2=_SPEC_X)],
        "b": [_feat("b1", 200.005, 1.03, ms2=_SPEC_Y)],
    }
    spots, stats = join_spots(features, _config())

    assert len(spots) == 2
    assert stats.n_added_by_sample["b"] == 1
    # Neither peak is claimed into the other's spot: the gate is in the box the
    # *claim* step uses too, not only the build step's.
    assert [set(s.peaks) for s in spots] == [{"a"}, {"b"}]
    assert stats.n_lost_peaks == 0
    assert stats.n_empty_spots == 0

    # And with the gate off, the same pair is one master again — so it is the
    # gate doing this, not the geometry.
    ungated, _ = join_spots(features, _config(ms2_identity_threshold=0.0))
    assert len(ungated) == 1


def test_ms1_can_opt_out_of_aif_identity_without_changing_product_identity():
    """ASFAM MS1 alignment is geometric; PRODUCT still needs MS2 identity."""
    left = [(100.0, 1000.0), (150.0, 1.0), (200.0, 1.0)]
    right = [(100.0, 1.0), (150.0, 1000.0), (200.0, 1.0)]
    ms1_features = {
        "a": [_feat("a1", 200.0, 1.0, ms2=left)],
        "b": [_feat("b1", 200.005, 1.03, ms2=right)],
    }
    config = _config(
        use_reliable_ms2_identity=True,
        conserve_detected_peaks=True,
        use_ms2_identity_for_ms1=False,
    )

    ms1_spots, ms1_stats = join_spots(ms1_features, config)

    assert len(ms1_spots) == 1
    assert set(ms1_spots[0].peaks) == {"a", "b"}
    assert ms1_stats.n_unexplained_lost == 0

    product_features = {
        "a": [_ms2_only("a1", 85.03, 1.0, window=300, ms2=left)],
        "b": [_ms2_only("b1", 85.035, 1.03, window=300, ms2=right)],
    }
    product_spots, product_stats = join_spots(product_features, config)

    assert len(product_spots) == 2
    assert [set(spot.peaks) for spot in product_spots] == [{"a"}, {"b"}]
    assert product_stats.n_unexplained_lost == 0


def test_ms1_opt_out_does_not_fold_an_ms1_driven_loser_on_aif_cosine():
    spectrum = [(100.0, 100.0), (150.0, 80.0), (200.0, 60.0)]
    loser = _feat("B2", 200.002, 5.002, ms2=spectrum)
    loser.detection_source = "ms1_driven"
    features = {
        "a": [_feat("A", 200.0, 5.0, ms2=spectrum)],
        "b": [
            _feat("B1", 200.001, 5.001, ms2=spectrum),
            loser,
        ],
    }
    config = _config(
        reference_sample="a",
        use_reliable_ms2_identity=True,
        conserve_detected_peaks=True,
        use_ms2_identity_for_ms1=False,
    )

    spots, stats = join_spots(features, config)

    assert len(spots) == 2
    assert stats.n_collapsed_same_compound == 0
    assert stats.n_promoted_unjudgeable == 1
    assert stats.n_unexplained_lost == 0


def test_a_split_peak_is_still_suppressed_by_the_gate():
    """Same compound, twice: still one master. The gate must not un-merge splits.

    A widened gate that let every near-duplicate found a row would trade one bug
    for a worse one. Identical spectra score 1.0, so ``b1`` stays suppressed,
    loses nothing, and joins ``a1``'s spot.
    """
    features = {
        "a": [_feat("a1", 200.0, 1.0, ms2=_SPEC_X)],
        "b": [_feat("b1", 200.005, 1.03, ms2=_SPEC_X)],
    }
    spots, stats = join_spots(features, _config())

    assert len(spots) == 1
    assert set(spots[0].peaks) == {"a", "b"}
    assert stats.n_added_by_sample["b"] == 0


@pytest.mark.parametrize("ref_ms2, other_ms2", [
    (_SPEC_X, None),                       # target carries nothing
    (None, _SPEC_Y),                       # master carries nothing
    (_SPEC_X, [(110.0, 100.0)]),           # 1 fragment: below min_fragments
])
def test_the_gate_abstains_when_it_cannot_judge(ref_ms2, other_ms2):
    """Too little MS2 to rule on => stay in one box.

    Abstaining has to mean "suppress", not "split". The opposite would give every
    fragment-poor peak — and ASFAM has many — a master and a row of its own,
    which is a row-count explosion dressed up as a fix.
    """
    features = {
        "a": [_feat("a1", 200.0, 1.0, ms2=ref_ms2)],
        "b": [_feat("b1", 200.005, 1.03, ms2=other_ms2)],
    }
    spots, _stats = join_spots(features, _config())

    assert len(spots) == 1
    assert set(spots[0].peaks) == {"a", "b"}


def test_a_peak_the_build_step_suppressed_can_claim_the_master_that_suppressed_it():
    """PR-3's invariant, and the reason the gate lives in the shared ``_matches``.

    The build step and the claim step must ask the *same* question. If the build
    step suppresses a peak with a box the claim step would not honour, that peak
    founds no master and claims none: it vanishes from the run and its sample is
    gap-filled in the spot that suppressed it. Both steps go through
    :func:`_matches`, gate included, so an uncontested suppressed peak always
    lands in the master that suppressed it.
    """
    features = {
        "a": [_feat("a1", 200.0, 1.0, ms2=_SPEC_X)],
        "b": [_feat("b1", 200.008, 1.05, ms2=_SPEC_X)],
    }
    spots, stats = join_spots(features, _config())

    assert stats.n_added_by_sample["b"] == 0          # build suppressed it ...
    assert set(spots[0].peaks) == {"a", "b"}          # ... and claim took it.
    assert stats.n_lost_peaks == 0


def test_lost_peaks_are_counted_and_split_by_what_they_actually_were():
    """The acceptance metric of this PR, on a two-peak reproduction of the bug.

    Sample ``b`` puts two peaks in ``a1``'s box: one is the same compound as
    ``a1`` (a split), the other is not. Ungated — today's behaviour — the build
    step suppresses both and the claim step seats only one, so the other is
    deleted, and the counter must say it was a *different* compound. Gated,
    nothing is lost: the different compound founds its own master.
    """
    features = {
        # 'a' pinned as reference: 'b' has more peaks and would win the automatic
        # choice, seeding both masters and leaving nothing for the claim to drop.
        "a": [_feat("a1", 200.0, 1.0, ms2=_SPEC_X)],
        "b": [_feat("b1", 200.004, 1.02, ms2=_SPEC_X),
              _feat("b2", 200.006, 1.04, ms2=_SPEC_Y)],
    }

    _spots, ungated = join_spots(
        features, _config(reference_sample="a", ms2_identity_threshold=0.0))
    assert ungated.n_lost_peaks == 1
    assert ungated.n_lost_different_compound == 1
    assert ungated.n_lost_same_compound == 0

    spots, gated = join_spots(features, _config(reference_sample="a"))
    assert gated.n_lost_peaks == 0
    assert gated.n_lost_different_compound == 0
    assert len(spots) == 2
    # Every peak of every sample reached a spot.
    assert sum(len(s.peaks) for s in spots) == 3


def test_no_peak_is_deleted_for_being_a_different_compound():
    """The property, on a randomized population — not just the hand-built case.

    Distinct compounds get disjoint spectra; each is split into two peaks in some
    samples to give the claim step something legitimate to drop. Whatever the
    joiner does with the splits, no peak may be dropped whose MS2 says it is a
    different compound from the peak that beat it. That number is the whole
    reason this PR exists, and it is measured through the *geometric* box, so the
    gate cannot flatter itself.
    """
    rng = random.Random(20260713)
    compounds = [
        (100.0 + 5.0 * i, 1.0 + 0.3 * i,
         [(50.0 + 7.0 * i, 100.0), (90.0 + 7.0 * i, 70.0), (130.0 + 7.0 * i, 40.0)])
        for i in range(40)
    ]

    features: dict[str, list[CandidateFeature]] = {}
    for sample in ("s1", "s2", "s3"):
        peaks = []
        for i, (mz, rt, spec) in enumerate(compounds):
            # Real cross-sample jitter, inside the tolerances.
            peaks.append(_feat(f"{sample}_c{i}", mz + rng.uniform(-0.004, 0.004),
                               rt + rng.uniform(-0.04, 0.04), ms2=spec))
            if rng.random() < 0.4:      # the peak picker split this one in two
                peaks.append(_feat(f"{sample}_c{i}b", mz + rng.uniform(-0.004, 0.004),
                                   rt + rng.uniform(-0.04, 0.04), ms2=spec))
        features[sample] = peaks

    _spots, stats = join_spots(features, _config())

    assert stats.n_lost_different_compound == 0
    # The splits are still dropped — the gate did not achieve this by refusing to
    # drop anything, which would be a row explosion, not a fix.
    assert stats.n_lost_same_compound > 0


# ---------------------------------------------------------------------------
# ASFAM opt-in: reliable identity + detected-peak conservation
# ---------------------------------------------------------------------------

def _conserving_config(**kwargs) -> JoinerConfig:
    return _config(
        reference_sample="a",
        use_reliable_ms2_identity=True,
        conserve_detected_peaks=True,
        ms2_identity_min_fragments=3,
        ms2_identity_min_matched_fragments=3,
        **kwargs,
    )


def _nontransitive_features(order=("P", "Q")):
    # All three comparisons match three fragments.  M is a mixed spectrum that
    # clears the identity threshold against both pure compounds, while P and Q
    # are mutually orthogonal.  This is the residual real-data loss F2 missed.
    p = [(100.0, 1000.0), (150.0, 1.0), (200.0, 1.0)]
    q = [(100.0, 1.0), (150.0, 1000.0), (200.0, 1.0)]
    m = [(100.0, 1001.0), (150.0, 1001.0), (200.0, 2.0)]
    by_name = {
        "P": _feat("P", 200.001, 5.001, ms2=p),
        "Q": _feat("Q", 200.002, 5.002, ms2=q),
    }
    return {
        "a": [_feat("M", 200.000, 5.000, ms2=m)],
        "b": [by_name[name] for name in order],
    }


def _membership(spots):
    return sorted(
        tuple(sorted((sid, peak.feature_id) for sid, peak in spot.peaks.items()))
        for spot in spots
    )


def test_nontransitive_identity_overlap_promotes_each_unassigned_compound():
    spots, stats = join_spots(_nontransitive_features(), _conserving_config())

    assert _membership(spots) == [(('a', 'M'), ('b', 'P')), (('b', 'Q'),)]
    assert stats.n_promoted_different == 1
    assert stats.n_unexplained_lost == 0
    assert stats.n_input_keyed_peaks == 3
    assert stats.n_input_keyed_peaks == (
        stats.n_assigned_peaks + stats.n_collapsed_same_compound
    )


def test_conservation_is_invariant_to_candidate_permutation():
    forward, forward_stats = join_spots(
        _nontransitive_features(("P", "Q")), _conserving_config(),
    )
    reverse, reverse_stats = join_spots(
        _nontransitive_features(("Q", "P")), _conserving_config(),
    )

    assert _membership(forward) == _membership(reverse)
    keys = (
        "n_input_keyed_peaks", "n_assigned_peaks",
        "n_collapsed_same_compound", "n_promoted_different",
        "n_promoted_unjudgeable", "n_unexplained_lost",
    )
    assert tuple(getattr(forward_stats, k) for k in keys) == tuple(
        getattr(reverse_stats, k) for k in keys
    )


def test_visible_holder_prevents_nontransitive_hidden_fold_chain():
    features = _nontransitive_features(("P", "Q"))
    features["b"][0].is_duplicate = True
    features["b"][0].duplicate_type = ""

    spots, stats = join_spots(features, _conserving_config())

    # Q is visible and therefore gets the contested cell first. P is DIFFERENT
    # from that final holder and must be promoted, even though both are SAME to M.
    assert _membership(spots) == [(('a', 'M'), ('b', 'Q')), (('b', 'P'),)]
    assert stats.n_promoted_different == 1
    assert stats.n_collapsed_same_compound == 0
    assert stats.n_unexplained_lost == 0


def test_reliable_same_loser_is_folded_with_evidence_and_mapping():
    spectrum = [(100.0, 100.0), (150.0, 80.0), (200.0, 60.0)]
    features = {
        "a": [_feat("A", 200.0, 5.0, ms2=spectrum)],
        "b": [
            _feat("B1", 200.001, 5.001, ms2=spectrum),
            _feat("B2", 200.002, 5.002, ms2=spectrum),
        ],
    }

    spots, stats = join_spots(features, _conserving_config())

    assert len(spots) == 1
    assert stats.n_assigned_peaks == 2
    assert stats.n_collapsed_same_compound == 1
    assert stats.n_unexplained_lost == 0
    assert [fold.peak.feature_id for fold in spots[0].folded_peaks] == ["B2"]
    fold = spots[0].folded_peaks[0]
    assert fold.target_peak_id == "B1"
    assert fold.n_matched_fragments == 3
    assert fold.cosine == pytest.approx(1.0)


def test_unjudgeable_loser_is_promoted_not_silently_folded():
    sparse = [(100.0, 1.0)]
    features = {
        "a": [_feat("A", 200.0, 5.0, ms2=sparse)],
        "b": [
            _feat("B1", 200.001, 5.001, ms2=sparse),
            _feat("B2", 200.002, 5.002, ms2=sparse),
        ],
    }

    spots, stats = join_spots(features, _conserving_config())

    assert len(spots) == 2
    assert stats.n_promoted_unjudgeable == 1
    assert stats.n_collapsed_same_compound == 0
    assert stats.n_assigned_peaks == 3
    assert stats.n_unexplained_lost == 0


def test_unjudgeable_ms2_does_not_receive_claim_score_weight():
    config = _conserving_config()
    geometry_only = _score(0.01, 0.08, [], [], config)
    one_shared_fragment = _score(
        0.01, 0.08, [(100.0, 1.0)], [(100.0, 1.0)], config,
    )

    assert one_shared_fragment == pytest.approx(geometry_only)


def test_reliable_lost_classification_requires_enough_matched_fragments():
    left = [(100.0, 1000.0), (150.0, 1.0), (200.0, 1.0), (250.0, 1.0)]
    right = [(100.0, 1000.0), (350.0, 1.0), (400.0, 1.0), (450.0, 1.0)]
    features = {
        "a": [_feat("A", 200.0, 5.0, ms2=[])],
        "b": [
            _feat("B1", 200.001, 5.001, ms2=left),
            _feat("B2", 200.002, 5.002, ms2=right),
        ],
    }
    config = _config(
        reference_sample="a",
        use_reliable_ms2_identity=True,
        conserve_detected_peaks=False,
        ms2_identity_min_fragments=3,
        ms2_identity_min_matched_fragments=3,
    )

    _spots, stats = join_spots(features, config)

    assert stats.n_lost_peaks == 1
    assert stats.n_lost_unjudgeable == 1
    assert stats.n_lost_same_compound == 0
    assert stats.n_lost_different_compound == 0
