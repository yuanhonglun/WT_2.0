"""PR-6: the refiner marks and renames. It never deletes a row."""
from __future__ import annotations

import numpy as np
import pytest

from metabo_core.alignment.joiner import AlignmentSpot, build_feature
from metabo_core.alignment.refiner import (
    CROSS_SAMPLE_REDUNDANT,
    MS1_COVERED,
    MS1_COVERED_PARTIAL,
    PUTATIVE_PREFIX,
    deduplicate_names,
    order_spots_by_mz,
    refine_features,
)
from metabo_core.annotation import is_high_confidence
from metabo_core.config import ConfidenceConfig, RefinerConfig
from metabo_core.models import AnnotationMatch, CandidateFeature, Feature

CONFIDENCE = ConfidenceConfig(score_threshold=0.3, min_matched_peaks=3)


def _annotated(f) -> bool:
    return is_high_confidence(f, CONFIDENCE)


def _match(name: str, total_score: float) -> AnnotationMatch:
    return AnnotationMatch(
        rank=1, name=name, score=total_score, n_matched=5, total_score=total_score,
    )


def _feature(
    feature_id: str,
    mz: float,
    rt: float,
    *,
    height: float = 1000.0,
    name: str | None = None,
    total_score: float = 0.0,
    isotope_index: int = 0,
    is_duplicate: bool = False,
    duplicate_type: str = "",
    signal_type: str = "ms1_detected",
    align_mz: float | None = None,
    ms2: list[tuple[float, float]] | None = None,
) -> Feature:
    """``mz`` is the *reported* precursor m/z. ``align_mz`` is what the row was
    quantified on, and what the refiner keys redundancy off — they are the same
    number only on a peak that MS1 actually picked. Left ``None``, the refiner
    falls back to ``mz``, which is what the tests written before F1 assume."""
    peaks = [(100.0, 1000.0), (150.0, 500.0)] if ms2 is None else ms2
    feature = Feature(
        feature_id=feature_id,
        precursor_mz=mz,
        rt=rt,
        rt_left=rt - 0.1,
        rt_right=rt + 0.1,
        signal_type=signal_type,
        ms2_mz=np.array([m for m, _ in peaks], dtype=np.float64),
        ms2_intensity=np.array([i for _, i in peaks], dtype=np.float64),
        n_fragments=len(peaks),
        mean_height=height,
        name=name,
        align_mz=align_mz,
        isotope_index=isotope_index,
        is_duplicate=is_duplicate,
        duplicate_type=duplicate_type,
    )
    if name is not None:
        feature.annotation_matches = [_match(name, total_score)]
    return feature


def _refine(features, **config_kwargs):
    return refine_features(features, RefinerConfig(**config_kwargs), _annotated)


# ---------------------------------------------------------------------------
# Step 6.2 — cross-sample redundancy
# ---------------------------------------------------------------------------

def test_near_neighbour_is_marked_not_dropped():
    """Both spots in m/z + RT range: the weaker one is marked, the row survives."""
    strong = _feature("F00000", 300.0000, 5.00, name="Caffeine", total_score=0.90)
    weak = _feature("F00001", 300.0050, 5.02, name="Caffeine", total_score=0.60)

    features = [strong, weak]
    stats = _refine(features)

    assert len(features) == 2                    # nothing deleted
    assert stats.n_marked_redundant == 1
    assert not strong.is_duplicate
    assert weak.is_duplicate
    assert weak.duplicate_type == CROSS_SAMPLE_REDUNDANT
    assert weak.mean_height == 1000.0            # quantitation untouched


def test_spot_outside_the_rt_gate_keeps_its_own_slot():
    a = _feature("F00000", 300.0000, 5.00, name="Caffeine", total_score=0.90)
    b = _feature("F00001", 300.0050, 5.30, name="Caffeine", total_score=0.60)

    stats = _refine([a, b])

    assert stats.n_marked_redundant == 0
    assert stats.n_masters == 2
    assert not b.is_duplicate


def test_existing_duplicate_type_is_not_overwritten():
    """A per-sample stage already claimed this spot. "isotope" says more."""
    master = _feature("F00000", 300.0000, 5.00, height=9000.0)
    satellite = _feature(
        "F00001", 300.0050, 5.01, height=100.0,
        is_duplicate=True, duplicate_type="isotope",
    )

    stats = _refine([master, satellite])

    assert satellite.is_duplicate
    assert satellite.duplicate_type == "isotope"       # not clobbered
    assert stats.n_marked_redundant == 0
    assert stats.n_redundant_already_marked == 1


def test_already_marked_spot_can_still_hold_a_master_slot():
    """"Still competes for a slot, and can hold one" — it just keeps its mark."""
    spectral = _feature(
        "F00000", 300.0000, 5.00, height=9000.0,
        is_duplicate=True, duplicate_type="spectral",
    )
    other = _feature("F00001", 300.0050, 5.01, height=100.0)

    stats = _refine([spectral, other])

    assert stats.n_masters == 1
    assert spectral.duplicate_type == "spectral"
    assert other.duplicate_type == CROSS_SAMPLE_REDUNDANT


def test_unidentified_isotope_satellite_never_claims_a_slot():
    """``LcmsAlignmentRefiner.cs:49``. The taller satellite must not evict a real peak.

    The satellite would win the height sort, take the slot, and mark the
    monoisotopic peak of a *different* compound 5 mDa away as redundant.
    """
    satellite = _feature("F00000", 300.0000, 5.00, height=9000.0, isotope_index=1)
    monoisotopic = _feature("F00001", 300.0050, 5.01, height=100.0, isotope_index=0)

    stats = _refine([satellite, monoisotopic])

    assert stats.n_isotope_satellites_skipped == 1
    assert not monoisotopic.is_duplicate            # it got the slot
    assert not satellite.is_duplicate               # skipped, not marked
    assert stats.n_masters == 1


def test_identified_isotope_satellite_still_competes():
    """Loops 1-3 of ``GetCleanedSpots`` do not check ``IsotopeWeightNumber``."""
    satellite = _feature(
        "F00000", 300.0000, 5.00, name="Caffeine", total_score=0.9, isotope_index=1,
    )
    other = _feature("F00001", 300.0050, 5.01, name="Theine", total_score=0.5)

    stats = _refine([satellite, other])

    assert stats.n_isotope_satellites_skipped == 0
    assert not satellite.is_duplicate
    assert other.duplicate_type == CROSS_SAMPLE_REDUNDANT


def test_rt_gate_is_capped_before_it_is_halved():
    """A user widening the GUI's alignment RT tolerance must not widen this gate.

    ``rtTol = min(RetentionTimeAlignmentTolerance, 0.1)``, then ``< rtTol * 0.5``
    (``LcmsAlignmentRefiner.cs:92,102``). Without the cap, 0.3 min would give a
    0.15 min gate — 3x MS-DIAL's — and merge distinct compounds.
    """
    assert RefinerConfig(rt_tolerance=0.3).rt_gate == pytest.approx(0.05)
    assert RefinerConfig(rt_tolerance=0.02).rt_gate == pytest.approx(0.01)

    a = _feature("F00000", 300.0000, 5.00, name="Caffeine", total_score=0.9)
    b = _feature("F00001", 300.0050, 5.07, name="Caffeine", total_score=0.5)

    stats = _refine([a, b], rt_tolerance=0.3)

    # 0.07 min apart: inside a naive 0.15 gate, outside the capped 0.05 one.
    assert stats.n_marked_redundant == 0
    assert not b.is_duplicate


def test_mz_gate_widens_above_the_ppm_pivot():
    """``TryMergeToMaster:96-98`` re-evaluates the m/z tolerance as ppm above 500."""
    config = RefinerConfig(mz_tolerance=0.02)
    assert config.mz_gate(300.0) == pytest.approx(0.02)      # flat below the pivot
    assert config.mz_gate(1000.0) == pytest.approx(0.04)     # 40 ppm at 1000 Da

    a = _feature("F00000", 1000.000, 5.00, name="Big", total_score=0.9)
    b = _feature("F00001", 1000.030, 5.01, name="Big", total_score=0.5)

    # 0.030 Da apart: outside the flat 0.02 gate, inside the widened 0.04 one.
    assert _refine([a, b]).n_marked_redundant == 1
    assert b.duplicate_type == CROSS_SAMPLE_REDUNDANT


def test_bucketing_agrees_with_brute_force():
    """The m/z buckets are an index, not a second gate. Including at a wide tolerance.

    Replays the placement loop with an O(n^2) scan over every master and asserts
    the same spots come out marked. A tolerance of 3 Da makes the gate wider than
    a bucket, which is where a fixed +/-1 bucket scan starts silently missing
    masters on the far side of a boundary.
    """
    rng = np.random.default_rng(6)
    for mz_tol in (0.02, 0.4, 3.0):
        config = RefinerConfig(mz_tolerance=mz_tol)
        made = [
            _feature(f"F{i:05d}", float(mz), float(rt), height=float(h))
            for i, (mz, rt, h) in enumerate(zip(
                rng.uniform(100.0, 1000.0, 300),
                rng.uniform(0.0, 12.0, 300),
                rng.uniform(100.0, 9000.0, 300),
            ))
        ]
        expected: list[str] = []
        masters: list[Feature] = []
        for f in sorted(made, key=lambda f: (-f.mean_height, f.feature_id)):
            gate = config.mz_gate(f.precursor_mz)
            if any(abs(m.precursor_mz - f.precursor_mz) < gate
                   and abs(m.rt - f.rt) < config.rt_gate for m in masters):
                expected.append(f.feature_id)
            else:
                masters.append(f)

        refine_features(made, config, _annotated)
        marked = sorted(f.feature_id for f in made if f.is_duplicate)
        assert marked == sorted(expected), f"mz_tolerance={mz_tol}"


# ---------------------------------------------------------------------------
# F1 — the key is the quantitation ion, and routes are not compared on m/z
# ---------------------------------------------------------------------------

#: Two spectra of two compounds: no fragment in common, cosine 0.
_MS2_A = [(80.0, 1000.0), (120.0, 700.0), (160.0, 300.0)]
_MS2_B = [(91.0, 1000.0), (137.0, 800.0), (205.0, 400.0)]


def _cross_route_pair(product_ms2, rt: float = 5.00):
    """One MS1 row and one product row: same RT, reported m/z 1 mDa apart.

    That near-identity is the trap. Both numbers are the intensity-weighted
    centroid of the *same* 1-Da isolation window, so they agree whether or not the
    compounds do — and on the product row the window was centroided on a cycle
    where nothing was detectable, making it noise. Meanwhile the ions the two rows
    were really quantified on are a precursor and a fragment: not comparable at
    all, at any tolerance.
    """
    ms1 = _feature("F00000", 300.5000, rt, height=9000.0,
                   align_mz=300.5000, ms2=_MS2_A)
    product = _feature("F00001", 300.5010, rt, height=100.0, signal_type="ms2_only",
                       align_mz=137.0400, ms2=product_ms2)
    return ms1, product


def test_same_route_redundancy_keys_on_the_quantitation_ion():
    """Two MS1 rows of one compound, 0.5 Da apart in the m/z they *report*.

    Which is outside every gate — because that number is a window centroid and
    drifts with whatever co-isolated in each sample. The ion both were measured
    on is 2 mDa apart. Same compound, so the weaker row is redundant, and keying
    on the reported m/z is why it used not to be.
    """
    strong = _feature("F00000", 300.4000, 5.00, height=9000.0, align_mz=300.1000)
    weak = _feature("F00001", 300.9000, 5.01, height=100.0, align_mz=300.1020)

    stats = _refine([strong, weak])

    assert stats.n_marked_redundant == 1
    assert weak.duplicate_type == CROSS_SAMPLE_REDUNDANT
    assert not strong.is_duplicate


def test_two_compounds_whose_window_centroids_agree_are_not_merged():
    """The mirror image, and the one that hid real detections.

    Reported m/z 1 mDa apart — the noise value agreeing by chance — but quantified
    on ions half a Dalton apart. Two compounds. Neither may be marked.
    """
    a = _feature("F00000", 300.0000, 5.00, height=9000.0, align_mz=300.0000)
    b = _feature("F00001", 300.0010, 5.01, height=100.0, align_mz=300.5000)

    stats = _refine([a, b])

    assert stats.n_marked_redundant == 0
    assert stats.n_masters == 2
    assert not b.is_duplicate


def test_cross_route_rows_are_never_compared_on_mz():
    """The false kill this pass exists to stop: an unrelated MS2-only detection.

    1 mDa from the MS1 row in reported m/z, same RT — inside any m/z gate. Its MS2
    says it is a different compound, so it stays visible and keeps counting toward
    the MS2-only claim. Under the old key it did not.
    """
    ms1, product = _cross_route_pair(_MS2_B)

    stats = _refine([ms1, product])

    assert stats.n_marked_ms1_covered == 0
    assert stats.n_marked_redundant == 0
    assert not product.is_duplicate
    assert stats.n_masters == 2          # one slot per route; neither evicted


def test_cross_route_redundancy_is_decided_on_ms2():
    """Same pair, one compound's MS2 on both rows. MS1 saw it: not an MS2-only row."""
    ms1, product = _cross_route_pair(_MS2_A)

    stats = _refine([ms1, product])

    assert stats.n_marked_ms1_covered == 1
    assert product.duplicate_type == MS1_COVERED
    assert not ms1.is_duplicate                  # the MS1 row survives
    assert product.mean_height == 100.0          # quantitation untouched
    assert stats.ms1_covered_pairs[0][:2] == ("F00001", "F00000")


def test_the_ms1_row_survives_even_when_the_product_row_outranks_it():
    """Priority order decides who claims a slot. It must not decide who survives.

    Marked symmetrically, the compound's quantitation would move to a product-ion
    chromatogram whenever the product row happened to be taller or better-scoring,
    and a visible ms2_only row would come to mean "MS1 lost a sort" rather than
    "no sample's MS1 ever saw this".
    """
    ms1 = _feature("F00000", 300.5000, 5.00, height=100.0,
                   align_mz=300.5000, ms2=_MS2_A)
    product = _feature("F00001", 300.5010, 5.00, height=9000.0, name="Caffeine",
                       total_score=0.99, signal_type="ms2_only",
                       align_mz=137.0400, ms2=_MS2_A)

    _refine([ms1, product])

    assert product.duplicate_type == MS1_COVERED
    assert not ms1.is_duplicate


def test_the_ms2_identity_threshold_is_the_gate():
    """Two thirds of one spectrum in common: covered at 0.6, a detection at 0.9."""
    shared_two_of_three = [(80.0, 1000.0), (120.0, 700.0), (999.0, 900.0)]

    def covered(threshold: float) -> bool:
        ms1, product = _cross_route_pair(shared_two_of_three)
        _refine([ms1, product], ms2_identity_threshold=threshold)
        return product.is_duplicate

    assert not covered(0.9)
    assert covered(0.6)


def _reliable_refiner_config(**kwargs) -> RefinerConfig:
    base = dict(
        use_reliable_ms2_identity=True,
        ms2_identity_min_fragments=3,
        ms2_identity_min_matched_fragments=3,
        preserve_cross_route_unique_detections=True,
    )
    base.update(kwargs)
    return RefinerConfig(**base)


def test_one_matched_fragment_cannot_mark_ms1_covered():
    one = [(100.0, 1000.0)]
    ms1 = _feature("F00000", 300.0, 5.0, ms2=one)
    product = _feature(
        "F00001", 85.03, 5.0, signal_type="ms2_only", ms2=one,
    )

    stats = refine_features(
        [ms1, product], _reliable_refiner_config(), _annotated,
    )

    assert stats.n_marked_ms1_covered == 0
    assert not product.is_duplicate


def test_rich_spectra_with_one_match_cannot_mark_ms1_covered():
    ms1 = _feature(
        "F00000", 300.0, 5.0,
        ms2=[(100.0, 1000.0), (150.0, 1.0), (200.0, 1.0), (250.0, 1.0)],
    )
    product = _feature(
        "F00001", 85.03, 5.0, signal_type="ms2_only",
        ms2=[(100.0, 1000.0), (350.0, 1.0), (400.0, 1.0), (450.0, 1.0)],
    )

    refine_features([ms1, product], _reliable_refiner_config(), _annotated)

    assert not product.is_duplicate


def test_product_with_unique_detected_sample_stays_visible_as_partial_coverage():
    ms1, product = _cross_route_pair(_MS2_A)
    ms1.gap_fill_status = {"s1": "detected", "s2": "filled"}
    product.gap_fill_status = {"s1": "detected", "s2": "detected"}

    stats = refine_features(
        [ms1, product], _reliable_refiner_config(), _annotated,
    )

    assert stats.n_marked_ms1_covered == 0
    assert stats.n_ms1_covered_partial == 1
    assert not product.is_duplicate
    assert product.alignment_relation == MS1_COVERED_PARTIAL
    assert product.alignment_related_feature_id == ms1.feature_id


def test_product_without_unique_detected_sample_can_be_fully_covered():
    ms1, product = _cross_route_pair(_MS2_A)
    ms1.gap_fill_status = {"s1": "detected", "s2": "detected"}
    product.gap_fill_status = {"s1": "detected", "s2": "filled"}

    stats = refine_features(
        [ms1, product], _reliable_refiner_config(), _annotated,
    )

    assert stats.n_marked_ms1_covered == 1
    assert product.duplicate_type == MS1_COVERED
    assert product.alignment_relation == MS1_COVERED


def test_unique_detection_preservation_is_opt_in_for_shared_core():
    ms1, product = _cross_route_pair(_MS2_A)
    ms1.gap_fill_status = {"s1": "detected", "s2": "filled"}
    product.gap_fill_status = {"s1": "detected", "s2": "detected"}

    stats = refine_features([ms1, product], RefinerConfig(), _annotated)

    assert stats.n_marked_ms1_covered == 1
    assert stats.n_ms1_covered_partial == 0
    assert product.duplicate_type == MS1_COVERED
    assert product.alignment_relation == ""
    assert product.alignment_related_feature_id == ""


def test_cross_route_coverage_requires_compatible_window_when_opted_in():
    ms1, product = _cross_route_pair(_MS2_A)
    ms1.alignment_window = 300
    product.alignment_window = 700

    refine_features(
        [ms1, product],
        _reliable_refiner_config(require_cross_route_window_match=True),
        _annotated,
    )

    assert not product.is_duplicate


def test_a_product_row_with_no_ms2_is_never_covered():
    """No spectrum, no route-independent identity signal. Leave it visible."""
    ms1 = _feature("F00000", 300.5000, 5.00, height=9000.0,
                   align_mz=300.5000, ms2=_MS2_A)
    product = _feature("F00001", 300.5010, 5.00, height=100.0,
                       signal_type="ms2_only", align_mz=137.0400, ms2=[])

    stats = _refine([ms1, product])

    assert stats.n_marked_ms1_covered == 0
    assert not product.is_duplicate


def test_a_hidden_ms1_row_covers_nothing():
    """An isotope satellite is a row no reader counts, so it cannot hide a detection."""
    satellite = _feature("F00000", 300.5000, 5.00, height=9000.0, align_mz=300.5000,
                         ms2=_MS2_A, is_duplicate=True, duplicate_type="isotope")
    product = _feature("F00001", 300.5010, 5.00, height=100.0,
                       signal_type="ms2_only", align_mz=137.0400, ms2=_MS2_A)

    stats = _refine([satellite, product])

    assert stats.n_marked_ms1_covered == 0
    assert not product.is_duplicate


def test_an_already_hidden_product_row_keeps_its_duplicate_type():
    """"isf" says more than "ms1_covered", and the row is hidden either way."""
    ms1 = _feature("F00000", 300.5000, 5.00, height=9000.0,
                   align_mz=300.5000, ms2=_MS2_A)
    product = _feature("F00001", 300.5010, 5.00, height=100.0, signal_type="ms2_only",
                       align_mz=137.0400, ms2=_MS2_A,
                       is_duplicate=True, duplicate_type="isf")

    stats = _refine([ms1, product])

    assert product.duplicate_type == "isf"
    assert stats.n_marked_ms1_covered == 0
    assert stats.n_cross_route_candidates == 0    # it was never visible


def test_visible_keeper_only_opt_in_excludes_hidden_master():
    hidden = _feature(
        "F00000", 300.000, 5.00, height=9000.0,
        is_duplicate=True, duplicate_type="",
    )
    visible = _feature("F00001", 300.005, 5.01, height=100.0)

    stats = refine_features(
        [hidden, visible], RefinerConfig(visible_keepers_only=True), _annotated,
    )

    assert hidden.is_duplicate
    assert hidden.duplicate_type == ""
    assert not visible.is_duplicate
    assert stats.n_masters == 1


def test_the_rt_gate_still_bounds_ms1_coverage():
    """Identical MS2, but 0.07 min apart: outside the capped gate, so not covered."""
    ms1, product = _cross_route_pair(_MS2_A)
    product.rt = ms1.rt + 0.07

    stats = _refine([ms1, product], rt_tolerance=0.5)

    assert stats.n_marked_ms1_covered == 0
    assert not product.is_duplicate


def test_an_ms1_covered_row_cannot_claim_a_compound_name():
    """Invariant #8: only a row a reader sees may hold the plain name."""
    ms1 = _feature("F00000", 300.5000, 5.00, height=100.0, name="Caffeine",
                   total_score=0.50, align_mz=300.5000, ms2=_MS2_A)
    product = _feature("F00001", 300.5010, 5.00, height=9000.0, name="Caffeine",
                       total_score=0.99, signal_type="ms2_only",
                       align_mz=137.0400, ms2=_MS2_A)

    stats = _refine([ms1, product])

    # The product row scores higher, so under the old key it took the name and
    # renamed the row that survives. It is hidden now, so it takes part in neither.
    assert product.duplicate_type == MS1_COVERED
    assert ms1.name == "Caffeine"
    assert stats.n_renamed == 0


def test_duplicate_type_counts_cover_every_marked_spot():
    features = [
        _feature("F00000", 300.0, 5.0, height=9000.0),
        _feature("F00001", 300.005, 5.01, height=800.0),
        _feature("F00002", 400.0, 6.0, height=500.0,
                 is_duplicate=True, duplicate_type="adduct"),
    ]
    stats = _refine(features)

    assert stats.duplicate_type_counts == {CROSS_SAMPLE_REDUNDANT: 1, "adduct": 1}
    assert sum(stats.duplicate_type_counts.values()) == sum(
        1 for f in features if f.is_duplicate
    )


# ---------------------------------------------------------------------------
# Step 6.1 — annotation name deduplication
# ---------------------------------------------------------------------------

def test_name_dedup_renames_only_the_name():
    strong = _feature("F00000", 300.0, 5.0, name="Caffeine", total_score=0.90)
    weak = _feature("F00001", 400.0, 9.0, name="Caffeine", total_score=0.60)
    before = (list(weak.annotation_matches), weak.mean_height,
              is_high_confidence(weak, CONFIDENCE))

    assert deduplicate_names([strong, weak], _annotated) == 1

    assert strong.name == "Caffeine"
    assert weak.name == PUTATIVE_PREFIX + "Caffeine"
    assert (list(weak.annotation_matches), weak.mean_height,
            is_high_confidence(weak, CONFIDENCE)) == before
    # The `annotated` column reads selected_annotation, never `name`, so the
    # high-confidence count cannot fall because of a rename.
    assert is_high_confidence(weak, CONFIDENCE)


def test_name_dedup_ignores_unannotated_spots():
    """``SpotAction.cs:33`` filters on IsReferenceMatched before grouping."""
    suggested = _feature("F00000", 300.0, 5.0, name="Caffeine", total_score=0.9)
    suggested.annotation_matches[0].n_matched = 1        # sparse => not annotated
    identified = _feature("F00001", 400.0, 9.0, name="Caffeine", total_score=0.1)

    assert not is_high_confidence(suggested, CONFIDENCE)
    assert deduplicate_names([suggested, identified], _annotated) == 0
    assert suggested.name == "Caffeine"
    assert identified.name == "Caffeine"


def test_a_redundant_spot_never_keeps_the_plain_name():
    """The keeper of a compound name must be a row that survives an is_duplicate filter.

    ``strong_x`` takes the slot; ``best_y`` is a *different* compound that lands
    inside the gate and is marked redundant — even though it is the best-scoring
    ``Caffeine`` row. Deduplicate names first and ``best_y`` keeps "Caffeine"
    while ``other_y``, the row a reader will actually see, becomes "Putative:".

    MS-DIAL cannot reach this state: ``GetCleanedSpots`` deletes ``best_y``
    before ``MatchResultAnnotationDeduplicator`` runs.
    """
    strong_x = _feature("F00000", 300.0000, 5.00, name="Theine", total_score=0.95)
    best_y = _feature("F00001", 300.0050, 5.01, name="Caffeine", total_score=0.90)
    other_y = _feature("F00002", 700.0000, 9.00, name="Caffeine", total_score=0.50)

    features = [strong_x, best_y, other_y]
    stats = _refine(features)

    assert best_y.duplicate_type == CROSS_SAMPLE_REDUNDANT
    assert not other_y.is_duplicate
    # The surviving row is the one that carries the compound.
    assert other_y.name == "Caffeine"
    assert stats.n_renamed == 0                      # best_y was never eligible
    assert best_y.name == "Caffeine"                 # its own name is left alone

    survivors = [f for f in features if not f.is_duplicate and f.name == "Caffeine"]
    assert len(survivors) == 1


def test_a_per_sample_duplicate_never_keeps_the_plain_name():
    """Same rule, other source: an isotope copy must not take the compound's name.

    Stage 4 marks the M+1 satellite ``is_duplicate``. Nothing stops it scoring
    above the monoisotopic peak, and it holds a master slot of its own (it is a
    Dalton away, so it collides with nothing). Let it into the pass and the only
    plain-named "Caffeine" row is one no reader ever sees.
    """
    satellite = _feature("F00000", 301.0067, 5.00, name="Caffeine", total_score=0.90,
                         isotope_index=1, is_duplicate=True, duplicate_type="isotope")
    mono = _feature("F00001", 300.0000, 5.00, name="Caffeine", total_score=0.50)

    features = [satellite, mono]
    stats = _refine(features)

    assert stats.n_marked_redundant == 0             # a Dalton apart; no collision
    assert satellite.duplicate_type == "isotope"     # untouched
    assert satellite.name == "Caffeine"              # not eligible, so not renamed
    assert mono.name == "Caffeine"                   # the visible row keeps it
    assert stats.n_renamed == 0

    visible = [f for f in features if not f.is_duplicate and f.name == "Caffeine"]
    assert len(visible) == 1


def test_name_dedup_is_case_insensitive_and_keeps_the_best():
    low = _feature("F00000", 300.0, 5.0, name="caffeine", total_score=0.4)
    high = _feature("F00001", 400.0, 9.0, name="Caffeine", total_score=0.8)
    mid = _feature("F00002", 500.0, 2.0, name="CAFFEINE", total_score=0.6)

    assert deduplicate_names([low, high, mid], _annotated) == 2

    assert high.name == "Caffeine"
    assert low.name.startswith(PUTATIVE_PREFIX)
    assert mid.name.startswith(PUTATIVE_PREFIX)


# ---------------------------------------------------------------------------
# Step 6.3 — detection counts
# ---------------------------------------------------------------------------

def _peak(sample: str, height: float, status: str) -> CandidateFeature:
    return CandidateFeature(
        feature_id=f"{sample}_0", segment_name="s", replicate_id=1,
        precursor_mz_nominal=300, rt_apex=5.0, rt_left=4.9, rt_right=5.1,
        ms2_mz=np.array([100.0]), ms2_intensity=np.array([10.0]), n_fragments=1,
        ms1_precursor_mz=300.0, ms1_height=height, ms1_area=height * 2,
        gap_fill_status=status,
    )


def test_n_detected_excludes_gap_filled_and_divides_by_all_samples():
    """Every filled cell now carries a height. Only ``gap_fill_status`` can tell."""
    spot = AlignmentSpot(
        index=0, master_mz=300.0, master_rt=5.0, origin_sample_id="A",
        peaks={
            "A": _peak("A", 5000.0, "detected"),
            "B": _peak("B", 4000.0, "filled"),      # nonzero, but not a detection
            "C": _peak("C", 0.0, "no_signal"),
        },
        representative_sample_id="A",
    )

    feature = build_feature(spot, "F00000", n_samples=4)   # a 4th sample had no cell

    assert feature.n_detected == 1
    assert feature.detection_rate == pytest.approx(0.25)   # denominator = the run
    assert all(h > 0 for h in list(feature.heights.values())[:2])   # the trap
    assert feature.mean_height == pytest.approx(5000.0)    # fill kept out of the mean


def test_detection_histogram_reports_every_spot():
    features = [
        _feature("F00000", 300.0, 5.0),
        _feature("F00001", 400.0, 6.0),
        _feature("F00002", 500.0, 7.0),
    ]
    features[0].n_detected = 3
    features[1].n_detected = 3
    features[2].n_detected = 1

    stats = _refine(features)

    assert stats.detection_histogram == {3: 2, 1: 1}
    assert sum(stats.detection_histogram.values()) == len(features)


# ---------------------------------------------------------------------------
# Step 6.4 — renumbering
# ---------------------------------------------------------------------------

def _spot(index: int, mz: float, rt: float) -> AlignmentSpot:
    peak = _peak("A", 1000.0, "detected")
    peak.ms1_precursor_mz = mz
    peak.rt_apex = rt
    return AlignmentSpot(
        index=index, master_mz=mz, master_rt=rt, origin_sample_id="A",
        peaks={"A": peak}, representative_sample_id="A",
    )


def test_order_spots_by_mz_rewrites_the_index():
    spots = [_spot(0, 500.0, 3.0), _spot(1, 100.0, 9.0), _spot(2, 300.0, 1.0)]

    ordered = order_spots_by_mz(spots)

    assert [s.representative.precursor_mz for s in ordered] == [100.0, 300.0, 500.0]
    assert [s.index for s in ordered] == [0, 1, 2]
    # The list positions and the indices agree, which is what lets stage 7 mint
    # feature ids off spot.index and hand the same order to the EIC spill writer.
    assert all(s.index == i for i, s in enumerate(ordered))


def test_order_spots_by_mz_breaks_ties_deterministically():
    """Equal m/z: RT decides, so a re-run numbers the two features the same way."""
    spots = [_spot(0, 300.0, 9.0), _spot(1, 300.0, 1.0)]

    ordered = order_spots_by_mz(spots)

    assert [s.representative.rt_apex for s in ordered] == [1.0, 9.0]
    assert [s.master_rt for s in ordered] == [1.0, 9.0]


def test_every_visible_compound_keeps_exactly_one_plain_named_row():
    """The invariant the whole name pass exists to establish.

    Filter the exported table on ``is_duplicate == False``, group the identified
    rows by compound, and each group must hold exactly one row whose name is not
    prefixed. Neither a cross-sample redundant row nor a per-sample duplicate may
    be the row that satisfies it, because neither is in the table.
    """
    features = [
        # "Caffeine": best row is redundant (collides with Theine's slot).
        _feature("F00000", 300.0000, 5.00, name="Theine", total_score=0.95),
        _feature("F00001", 300.0050, 5.01, name="Caffeine", total_score=0.90),
        _feature("F00002", 700.0000, 9.00, name="Caffeine", total_score=0.50),
        _feature("F00003", 720.0000, 3.00, name="Caffeine", total_score=0.40),
        # "Rutin": best row is an isotope copy a per-sample stage marked.
        _feature("F00004", 611.1607, 4.00, name="Rutin", total_score=0.88,
                 isotope_index=1, is_duplicate=True, duplicate_type="isotope"),
        _feature("F00005", 610.1533, 4.00, name="Rutin", total_score=0.70),
        _feature("F00006", 900.0000, 8.00, name="Rutin", total_score=0.60),
    ]
    _refine(features)

    groups: dict[str, list[Feature]] = {}
    for f in features:
        if f.is_duplicate or not _annotated(f):
            continue
        base = f.name.removeprefix(PUTATIVE_PREFIX).lower()
        groups.setdefault(base, []).append(f)

    assert set(groups) == {"theine", "caffeine", "rutin"}
    for base, group in groups.items():
        plain = [f for f in group if not f.name.startswith(PUTATIVE_PREFIX)]
        assert len(plain) == 1, f"{base}: {[f.name for f in group]}"

    # And specifically: the visible keeper is the best *visible* row, not the
    # better row that got hidden.
    assert features[2].name == "Caffeine"       # F00002, score 0.50
    assert features[5].name == "Rutin"          # F00005, score 0.70


def test_refine_features_on_an_empty_list():
    assert refine_features([], RefinerConfig(), _annotated).n_features == 0
