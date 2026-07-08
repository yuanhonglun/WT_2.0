"""Task D3: project save/reload must preserve the PR-D isotope / adduct labels.

``_features_to_dicts`` / ``_dicts_to_features`` previously dropped
``isotope_index`` / ``isotope_group_id`` / ``adduct_group_id``, so
save -> reload -> re-export silently lost the PR-D labels (they came back as
the dataclass defaults 0 / None / None). These round-trip tests pin them, plus
backward compatibility with old project files that predate the fields.
"""
from __future__ import annotations

import numpy as np

from asfam.models import Feature, CandidateFeature
from asfam.io.project_file import (
    _features_to_dicts, _dicts_to_features,
    _candidates_to_dicts, _dicts_to_candidates,
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
