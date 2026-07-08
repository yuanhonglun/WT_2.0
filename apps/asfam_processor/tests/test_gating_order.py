"""T4 gating-order fix: dedup-removed features must still get annotated.

Root cause was: orchestrator ran stage6b annotation (which gates on
status=="active") BEFORE restoring is_duplicate features to active, so
isotope/adduct/isf-removed features permanently lost their annotation.

Merged T5 requirement: the restore path must carry a score-floor guard so
weak/wrong names do not flood back in — only duplicates whose best match
clears ``matchms_similarity_threshold`` (0.7) are re-annotated. Normal
(non-duplicate) features keep the "emit any hit" behavior (display gate is
applied later at export/GUI time).
"""
from __future__ import annotations

import numpy as np

from asfam.config import ProcessingConfig
from asfam.pipeline.stage6b_annotation import run_stage6b_annotation
from asfam.pipeline.orchestrator import _restore_duplicate_status
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


def test_restore_helper_reactivates_duplicates():
    a = _feat("A", status="active", is_dup=False)
    b = _feat("B", status="isotope_removed", is_dup=True)
    _restore_duplicate_status({"rep1": [a, b]})
    assert a.status == "active"
    assert b.status == "active"          # restored
    assert b.duplicate_type == "spectral"  # backfilled default


def test_dedup_removed_feature_annotated_after_restore():
    # Reproduce the bug: annotate WITHOUT restoring -> removed feature skipped.
    a = _feat("A", status="active", is_dup=False)
    b = _feat("B", status="isotope_removed", is_dup=True)
    run_stage6b_annotation({"rep1": [a, b]}, ProcessingConfig(),
                           preloaded_library=_library())
    assert a.matchms_name == "caffeine"
    assert b.matchms_name is None        # BUG: removed feature got no annotation

    # Fixed order: restore first, then annotate -> removed feature annotated.
    a2 = _feat("A", status="active", is_dup=False)
    b2 = _feat("B", status="isotope_removed", is_dup=True)
    _restore_duplicate_status({"rep1": [a2, b2]})
    run_stage6b_annotation({"rep1": [a2, b2]}, ProcessingConfig(),
                           preloaded_library=_library())
    assert b2.matchms_name == "caffeine"  # RECOVERED (score ~1.97 >= 0.7 floor)
    assert b2.is_duplicate is True         # still flagged as duplicate


def test_score_floor_rejects_weak_duplicate_but_keeps_normal():
    """Merged T5 guard: a restored duplicate whose best match is below the
    0.7 floor must NOT be annotated (no wrong name), while a normal feature
    with the SAME weak match keeps its hit (emit-any-hit preserved)."""
    cfg = ProcessingConfig()  # matchms_similarity_threshold = 0.7
    # Mismatched intensities vs the ref's [0.3,0.4,1.0] first three peaks -> the
    # 3-peak query matches on m/z (passes n_matched/matched_pct gate) but scores
    # ~0.59 (< 0.7 floor). A matching-intensity query would score above 0.7.
    weak_int = (1.0, 0.05, 0.05)
    dup = _feat("D", status="active", is_dup=True, ms2_int=weak_int)   # restored
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
    """A restored duplicate whose best match clears the floor IS annotated."""
    cfg = ProcessingConfig()
    dup = _feat("D", status="active", is_dup=True)
    run_stage6b_annotation({"rep1": [dup]}, cfg, preloaded_library=_library())
    assert dup.matchms_name == "caffeine"   # score ~1.97 >= 0.7
    assert dup.is_duplicate is True
