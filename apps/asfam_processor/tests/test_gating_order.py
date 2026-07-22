"""Dedup marks features; it never removes them.

``status`` gates the *dedup* stages against each other: once stage 4/5/6 claims
a feature it becomes ``"*_excluded"`` so a later dedup stage will not consider
it as a candidate again. Annotation, alignment and export see every feature.

This used to be expressed as a hack — the dedup stages wrote ``"*_removed"``
and ``_restore_duplicate_status()`` flipped them all back to ``"active"`` right
before stage 6.5, which also backfilled the ``duplicate_type`` that stage 5b
never set. Both jobs now live where they belong, so the tests below pin the two
invariants the restore sweep used to guarantee:

1. stage 6.5 annotates all features, excluded or not (T4);
2. every ``is_duplicate`` feature carries a ``duplicate_type``.

Merged T5 requirement: annotating a duplicate carries a score-floor guard so
weak/wrong names do not flood back in — only duplicates whose best match clears
``matchms_similarity_threshold`` (0.7) are annotated. Normal (non-duplicate)
features keep the "emit any hit" behavior (display gate is applied later at
export/GUI time).
"""
from __future__ import annotations

import numpy as np

from asfam.config import ProcessingConfig
from asfam.pipeline.stage5b_duplicate_detection import run_stage5b
from asfam.pipeline.stage6b_annotation import run_stage6b_annotation
from metabo_core.models.features import CandidateFeature


def _feat(fid, status="active", is_dup=False,
          ms2_mz=(110.07, 138.07, 195.09), ms2_int=(0.3, 0.4, 1.0)):
    f = CandidateFeature(
        feature_id=fid, segment_name="190-200", replicate_id=1,
        precursor_mz_nominal=195, rt_apex=5.0, rt_left=4.9, rt_right=5.1,
        ms2_mz=np.array(ms2_mz, dtype=np.float64),
        ms2_intensity=np.array(ms2_int, dtype=np.float64),
        n_fragments=len(ms2_mz),
    )
    f.ms1_precursor_mz = 195.09
    f.status = status
    f.is_duplicate = is_dup
    return f


def _library():
    """Exact caffeine self-match -> composite score ~1.97 (well above 0.7)."""
    return [{
        "mz": [110.07, 138.07, 195.09], "intensity": [0.3, 0.4, 1.0],
        "metadata": {"name": "caffeine", "precursor_mz": 195.09,
                     "formula": "C8H10N4O2"},
    }]


def _weak_library():
    """Rich reference; a 3-peak query with mismatched intensities matches only
    3/12 significant peaks -> passes the emit gate (n_matched=3, matched_pct
    =0.25) but scores ~0.59 (< 0.7 floor). Used to exercise the guard."""
    return [{
        "mz": [110.07, 138.07, 195.09, 55.0, 66.0, 77.0, 88.0, 99.0,
               120.0, 150.0, 170.0, 180.0],
        "intensity": [0.3, 0.4, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5,
                      0.85, 0.75, 0.65, 0.55],
        "metadata": {"name": "richcpd", "precursor_mz": 195.09, "formula": "C1"},
    }]


def test_dedup_excluded_feature_is_annotated():
    """A feature the dedup stages excluded still gets its library hit (T4)."""
    a = _feat("A", status="active", is_dup=False)
    b = _feat("B", status="isotope_excluded", is_dup=True)
    run_stage6b_annotation({"rep1": [a, b]}, ProcessingConfig(),
                           preloaded_library=_library())
    assert a.matchms_name == "caffeine"
    assert b.matchms_name == "caffeine"   # score ~1.97 >= 0.7 floor
    assert b.is_duplicate is True          # still flagged as duplicate
    assert b.status == "isotope_excluded"  # and still excluded from dedup


def test_score_floor_rejects_weak_duplicate_but_keeps_normal():
    """Merged T5 guard: a duplicate whose best match is below the 0.7 floor must
    NOT be annotated (no wrong name), while a normal feature with the SAME weak
    match keeps its hit (emit-any-hit preserved)."""
    cfg = ProcessingConfig()  # matchms_similarity_threshold = 0.7
    # Mismatched intensities vs the ref's [0.3,0.4,1.0] first three peaks -> the
    # 3-peak query matches on m/z (passes n_matched/matched_pct gate) but scores
    # ~0.59 (< 0.7 floor). A matching-intensity query would score above 0.7.
    weak_int = (1.0, 0.05, 0.05)
    dup = _feat("D", status="active", is_dup=True, ms2_int=weak_int)
    normal = _feat("N", status="active", is_dup=False, ms2_int=weak_int)
    run_stage6b_annotation({"rep1": [dup, normal]}, cfg,
                           preloaded_library=_weak_library())
    # Normal feature: weak hit (~0.59, below floor) still emitted for display —
    # the guard is duplicate-only, not a general floor on annotation.
    assert normal.matchms_name == "richcpd"
    assert normal.annotation_matches  # non-empty
    assert normal.matchms_score < cfg.matchms_similarity_threshold
    # Duplicate feature: same weak hit gated out -> no name, no matches.
    assert dup.matchms_name is None
    assert dup.annotation_matches == []


def test_score_floor_admits_strong_duplicate():
    """A duplicate whose best match clears the floor IS annotated."""
    cfg = ProcessingConfig()
    dup = _feat("D", status="active", is_dup=True)
    run_stage6b_annotation({"rep1": [dup]}, cfg, preloaded_library=_library())
    assert dup.matchms_name == "caffeine"   # score ~1.97 >= 0.7
    assert dup.is_duplicate is True


# ---------------------------------------------------------------------------
# duplicate_type is named where the flag is set, not backfilled later
# ---------------------------------------------------------------------------

def _dup_pair():
    """Two identical spectra at the same RT -> stage 5b groups them."""
    peaks = (110.07, 138.07, 195.09, 210.1)
    ints = (0.3, 0.4, 1.0, 0.6)
    a = _feat("A", ms2_mz=peaks, ms2_int=ints)
    b = _feat("B", ms2_mz=peaks, ms2_int=ints)
    a.ms1_height, b.ms1_height = 1e5, 1e3   # A is the representative
    return a, b


def test_stage5b_names_the_spectral_duplicate():
    a, b = _dup_pair()
    run_stage5b({"rep1": [a, b]}, ProcessingConfig())
    assert b.is_duplicate is True
    assert b.duplicate_type == "spectral"
    # Stage 5b flags but does not exclude: stage 6 must still see this feature.
    assert b.status == "active"
    assert a.is_duplicate is False


def test_stage5b_keeps_an_existing_duplicate_type():
    """An isotope/adduct *representative* is still active and already typed.

    If stage 5b flags it, the earlier, more specific label must survive — the
    old backfill only wrote "spectral" when ``duplicate_type`` was empty.
    """
    a, b = _dup_pair()
    b.duplicate_type = "isotope"   # representative of an isotope group
    run_stage5b({"rep1": [a, b]}, ProcessingConfig())
    assert b.is_duplicate is True
    assert b.duplicate_type == "isotope"


def test_every_duplicate_is_typed_and_every_excluded_is_a_duplicate():
    """The invariant that lets stage 6.5 replace the restore sweep with ``list``.

    Post-restore, ``[f for f in features if f.status == "active"]`` returned
    *every* feature, because the only way to leave "active" was to be flagged a
    duplicate — and the sweep guaranteed each of those carried a type. Both
    halves must keep holding.
    """
    a, b = _dup_pair()
    c = _feat("C", status="isotope_excluded", is_dup=True)
    c.duplicate_type = "isotope"
    run_stage5b({"rep1": [a, b, c]}, ProcessingConfig())

    for f in (a, b, c):
        if f.status != "active":
            assert f.is_duplicate, f"{f.feature_id}: excluded but not a duplicate"
        if f.is_duplicate:
            assert f.duplicate_type, f"{f.feature_id}: duplicate without a type"
