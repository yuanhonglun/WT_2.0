"""Task D3: project save/reload must preserve the PR-D isotope / adduct labels.

``_features_to_dicts`` / ``_dicts_to_features`` previously dropped
``isotope_index`` / ``isotope_group_id`` / ``adduct_group_id``, so
save -> reload -> re-export silently lost the PR-D labels (they came back as
the dataclass defaults 0 / None / None). These round-trip tests pin them, plus
backward compatibility with old project files that predate the fields.

PR-2 adds the same guarantee for every ``AnnotationMatch`` field: the archival
helpers used to enumerate 11 of the 13, dropping ``matched_pct`` and
``total_score``.
"""
from __future__ import annotations

import dataclasses
import pickle

import numpy as np

from asfam.config import ProcessingConfig
from asfam.models import AnnotationMatch, Feature, CandidateFeature
from asfam.io.project_file import (
    save_project, load_project, _ANNOTATION_MATCH_FIELDS,
    _features_to_dicts, _dicts_to_features,
    _candidates_to_dicts, _dicts_to_candidates,
)


def _make_annotation_match() -> AnnotationMatch:
    """An AnnotationMatch with every field set to a distinctive non-default."""
    return AnnotationMatch(
        rank=1,
        name="Quercetin",
        formula="C15H10O7",
        score=0.87,
        n_matched=7,
        ref_peaks=[[151.0, 100.0], [179.0, 42.0]],
        ref_precursor_mz=303.0499,
        adduct="[M+H]+",
        wdp=0.81,
        sdp=0.74,
        rdp=0.69,
        matched_pct=0.55,
        total_score=0.87,
    )


def _make_feature() -> Feature:
    return Feature(
        feature_id="F0001",
        precursor_mz=285.05,
        rt=4.0,
        rt_left=3.9,
        rt_right=4.1,
        signal_type="ms1_detected",
        ms2_mz=np.array([100.0, 150.0], dtype=np.float64),
        ms2_intensity=np.array([1000.0, 500.0], dtype=np.float64),
        n_fragments=2,
        align_mz=285.0512,
        representative_rt=3.98,
        alignment_window=285,
        alignment_segment="285-314",
        alignment_relation="ms1_covered_partial",
        alignment_related_feature_id="F0002",
        isotope_index=1,
        isotope_group_id=3,
        adduct_group_id=5,
    )


def test_isotope_adduct_labels_survive_roundtrip():
    f = _make_feature()
    out = _dicts_to_features(_features_to_dicts([f]))[0]
    assert out.isotope_index == 1
    assert out.isotope_group_id == 3
    assert out.adduct_group_id == 5


def test_alignment_identity_fields_survive_roundtrip():
    out = _dicts_to_features(_features_to_dicts([_make_feature()]))[0]

    assert out.align_mz == 285.0512
    assert out.representative_rt == 3.98
    assert out.alignment_window == 285
    assert out.alignment_segment == "285-314"
    assert out.alignment_relation == "ms1_covered_partial"
    assert out.alignment_related_feature_id == "F0002"


def test_old_project_without_labels_defaults_gracefully():
    """Backward compat: an old project dict lacking the PR-D keys round-trips
    to the dataclass defaults (index 0, group ids None) rather than raising."""
    f = _make_feature()
    d = _features_to_dicts([f])[0]
    # Simulate an old project file written before the PR-D fields existed.
    d.pop("isotope_index", None)
    d.pop("isotope_group_id", None)
    d.pop("adduct_group_id", None)
    out = _dicts_to_features([d])[0]
    assert out.isotope_index == 0
    assert out.isotope_group_id is None
    assert out.adduct_group_id is None


def _make_candidate() -> CandidateFeature:
    return CandidateFeature(
        feature_id="rep1_00000",
        segment_name="280-292",
        replicate_id=1,
        precursor_mz_nominal=286,
        rt_apex=4.0,
        rt_left=3.9,
        rt_right=4.1,
        ms2_mz=np.array([100.0, 150.0], dtype=np.float64),
        ms2_intensity=np.array([1000.0, 500.0], dtype=np.float64),
        n_fragments=2,
        isotope_index=2,
        isotope_group_id=4,
        adduct_group_id=6,
    )


def test_candidate_isotope_labels_survive_roundtrip():
    """The ``candidates_by_rep`` archival payload already kept the group ids but
    dropped ``isotope_index``; reload -> re-annotate -> re-export then reset the
    M+n ordinal to 0. Pin all three so candidate archival matches feature
    archival."""
    c = _make_candidate()
    out = _dicts_to_candidates(_candidates_to_dicts([c]))[0]
    assert out.isotope_index == 2
    assert out.isotope_group_id == 4
    assert out.adduct_group_id == 6


# ---------------------------------------------------------------------------
# PR-2: every AnnotationMatch field survives save_project -> load_project
# ---------------------------------------------------------------------------

def test_annotation_match_all_fields_survive_save_load(tmp_path):
    """``matched_pct`` and ``total_score`` were dropped on the way to disk, so
    a saved-then-reopened project silently lost those two columns. PR-3's
    on-disk feature store reuses these helpers, so pin all 13 fields."""
    feature = _make_feature()
    feature.annotation_matches = [_make_annotation_match()]
    feature.selected_annotation_idx = 0

    candidate = _make_candidate()
    candidate.annotation_matches = [_make_annotation_match()]
    candidate.selected_annotation_idx = 0

    path = str(tmp_path / "proj.asfam")
    save_project(path, ProcessingConfig(), [feature], mzml_paths=["a.mzML"],
                 candidates_by_rep={"1": [candidate]})
    loaded = load_project(path)

    expected = dataclasses.asdict(_make_annotation_match())
    for match in (loaded["features"][0].annotation_matches[0],
                  loaded["candidates_by_rep"]["1"][0].annotation_matches[0]):
        assert dataclasses.asdict(match) == expected


def test_archival_covers_every_annotation_match_field():
    """The regression this guards: the archival helpers enumerate fields by
    hand, so a field added to AnnotationMatch is dropped on save until this
    tuple is updated. Fail loudly at that moment instead of at load time."""
    declared = {f.name for f in dataclasses.fields(AnnotationMatch)}
    assert declared == set(_ANNOTATION_MATCH_FIELDS)


def test_archival_covers_every_candidate_feature_field():
    """Same hazard, higher stakes: ``_candidates_to_dicts`` is also the
    serializer behind the per-sample ``.mfeat`` spill, so a field it forgets is
    a field the pipeline silently loses between stage 6.5 and alignment."""
    declared = {f.name for f in dataclasses.fields(CandidateFeature)}
    archived = set(_candidates_to_dicts([_make_candidate()])[0])
    assert declared - archived == set(), f"dropped on save: {declared - archived}"


def test_archival_covers_every_feature_field():
    """The same hand-enumeration hazard on the aligned ``Feature``.

    Caught ``charge_state`` and ``gap_fill_status`` already missing when it was
    written; without it, a saved project quietly loses whatever the last PR
    added.
    """
    declared = {f.name for f in dataclasses.fields(Feature)}
    archived = set(_features_to_dicts([_make_feature()])[0])
    assert declared - archived == set(), f"dropped on save: {declared - archived}"


# ---------------------------------------------------------------------------
# PR-7: the .asfam points at _work/ instead of embedding every candidate
# ---------------------------------------------------------------------------

def test_new_project_stores_a_work_dir_and_no_inline_candidates(tmp_path):
    """Since the spill landed, candidates live in ``_work/``. A fresh run saves a
    pointer to that directory and nothing else — embedding them would put the
    whole per-sample feature set back into the pickle the spill removed it
    from."""
    path = str(tmp_path / "new.asfam")
    save_project(path, ProcessingConfig(), [_make_feature()],
                 mzml_paths=["a.mzML"], work_dir=str(tmp_path / "_work"))

    with open(path, "rb") as fh:
        raw = pickle.load(fh)
    assert "candidates_by_rep" not in raw, "a fresh run re-embedded the candidates"
    assert raw["work_dir"] == str(tmp_path / "_work")

    loaded = load_project(path)
    assert loaded["work_dir"] == str(tmp_path / "_work")
    assert "candidates_by_rep" not in loaded
    assert loaded["features"][0].feature_id == "F0001"


def test_legacy_project_with_inline_candidates_still_opens(tmp_path):
    """Projects written before the spill carry ``candidates_by_rep`` and no
    ``work_dir``. They must keep opening: the GUI falls back to the inline copy
    for its single-sample view when there is no ``_work/`` to read."""
    candidate = _make_candidate()
    path = str(tmp_path / "legacy.asfam")
    save_project(path, ProcessingConfig(), [_make_feature()],
                 mzml_paths=["a.mzML"], candidates_by_rep={"1": [candidate]})

    loaded = load_project(path)
    assert loaded["work_dir"] is None
    assert loaded["candidates_by_rep"]["1"][0].feature_id == "rep1_00000"
    assert loaded["candidates_by_rep"]["1"][0].isotope_index == 2


def test_a_project_can_carry_both_and_neither(tmp_path):
    """The two keys are independent. A legacy project that is re-saved after a
    fresh run carries the new ``work_dir``; one saved with neither still loads."""
    both = str(tmp_path / "both.asfam")
    save_project(both, ProcessingConfig(), [_make_feature()], mzml_paths=["a.mzML"],
                 candidates_by_rep={"1": [_make_candidate()]},
                 work_dir=str(tmp_path / "_work"))
    loaded = load_project(both)
    assert loaded["work_dir"] == str(tmp_path / "_work")
    assert "candidates_by_rep" in loaded

    neither = str(tmp_path / "neither.asfam")
    save_project(neither, ProcessingConfig(), [_make_feature()], mzml_paths=["a.mzML"])
    loaded = load_project(neither)
    assert loaded["work_dir"] is None
    assert "candidates_by_rep" not in loaded


def test_old_project_without_matched_pct_defaults_to_zero():
    """Backward compat: a .asfam written before PR-2 has no ``matched_pct`` /
    ``total_score`` keys. They must reload as 0.0, not raise."""
    f = _make_feature()
    f.annotation_matches = [_make_annotation_match()]
    d = _features_to_dicts([f])[0]
    for m in d["annotation_matches"]:
        m.pop("matched_pct", None)
        m.pop("total_score", None)

    out = _dicts_to_features([d])[0].annotation_matches[0]
    assert out.matched_pct == 0.0
    assert out.total_score == 0.0
    # Untouched fields still round-trip.
    assert out.name == "Quercetin"
    assert out.wdp == 0.81
