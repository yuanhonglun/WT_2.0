"""The per-sample spill must round-trip a CandidateFeature losslessly.

Everything stage 7 and stage 8 read off a feature travels through
``.mfeat`` + ``.mspec``, so anything this layer drops or coerces changes
``features.csv``. Two coercions in particular are load-bearing:

* MS2 arrays stay float64. The annotation scorer identifies a peak by
  ``round(mz, 4)``, and ``round(np.float32(x), 4)`` disagrees with
  ``round(float(x), 4)`` often enough to move ``wdp`` / ``rdp``.
* ``ms2_mz.tolist()`` must yield Python floats, which it does for a real
  ``float64`` ndarray — the callers rely on that.
"""
from __future__ import annotations

import numpy as np
import pytest

from asfam.io import spill
from asfam.models import AnnotationMatch, CandidateFeature


def _feature(fid: str, n_peaks: int, seed: int = 0, **overrides) -> CandidateFeature:
    rng = np.random.default_rng(seed)
    mz = np.sort(rng.uniform(50.0, 500.0, n_peaks))
    intensity = rng.uniform(1.0, 1e5, n_peaks)
    feat = CandidateFeature(
        feature_id=fid, segment_name="100-129", replicate_id=1,
        precursor_mz_nominal=200, rt_apex=1.5, rt_left=1.4, rt_right=1.6,
        ms2_mz=mz, ms2_intensity=intensity, n_fragments=n_peaks,
        ms2_sn=rng.uniform(1.0, 10.0, n_peaks),
        ms2_gaussian=rng.uniform(0.0, 1.0, n_peaks),
        ms1_precursor_mz=200.12345, ms1_height=1e5, ms1_area=2e5,
        ms1_sn=12.5, ms1_gaussian=0.93, ms2_rep_ion_mz=123.4567,
        ms2_quality="correlated", n_correlated_ms2=3, charge_state=1,
    )
    feat.annotation_matches = [AnnotationMatch(
        rank=1, name="caffeine", formula="C8H10N4O2", score=0.91, n_matched=3,
        ref_peaks=[(110.07, 0.3)], ref_precursor_mz=195.09, adduct="[M+H]+",
        wdp=0.5, sdp=0.6, rdp=0.7, matched_pct=0.25, total_score=0.91,
    )]
    for key, value in overrides.items():
        setattr(feat, key, value)
    return feat


@pytest.fixture()
def sample(tmp_path):
    feats = [
        _feature("F000", 5, seed=1),
        _feature("F001", 3, seed=2, status="isotope_excluded",
                 is_duplicate=True, duplicate_type="isotope",
                 duplicate_group_id=7, isotope_index=1, isotope_group_id=7),
        _feature("F002", 0, seed=3),                       # no MS2 at all
        _feature("F003", 4, seed=4, ms2_sn=None, ms2_gaussian=None),
    ]
    stem = tmp_path / "1"
    spill.write_sample(stem, feats, fingerprint="fp-abc", sample_id="1")
    return stem, feats


def test_read_back_preserves_every_field(sample):
    stem, original = sample
    reloaded = spill.read_sample_features(stem, load_ms2=True)

    assert [f.feature_id for f in reloaded] == [f.feature_id for f in original]
    for got, want in zip(reloaded, original):
        assert np.array_equal(got.ms2_mz, want.ms2_mz)
        assert np.array_equal(got.ms2_intensity, want.ms2_intensity)
        assert got.ms2_mz.dtype == np.float64
        for optional in ("ms2_sn", "ms2_gaussian"):
            a, b = getattr(got, optional), getattr(want, optional)
            assert (a is None) == (b is None)
            if b is not None:
                assert np.array_equal(a, b)
        assert got.precursor_mz == want.precursor_mz
        assert got.status == want.status
        assert (got.is_duplicate, got.duplicate_type, got.duplicate_group_id) == (
            want.is_duplicate, want.duplicate_type, want.duplicate_group_id)
        assert (got.isotope_index, got.isotope_group_id) == (
            want.isotope_index, want.isotope_group_id)
        assert (got.ms2_quality, got.n_correlated_ms2, got.charge_state) == (
            want.ms2_quality, want.n_correlated_ms2, want.charge_state)
        assert got.ms2_rep_ion_mz == want.ms2_rep_ion_mz
        match, want_match = got.annotation_matches[0], want.annotation_matches[0]
        assert (match.matched_pct, match.total_score) == (
            want_match.matched_pct, want_match.total_score)
        assert (match.wdp, match.sdp, match.rdp) == (
            want_match.wdp, want_match.sdp, want_match.rdp)


def test_lean_read_leaves_ms2_on_disk_behind_a_pointer(sample):
    stem, original = sample
    lean = spill.read_sample_features(stem)

    for got, want in zip(lean, original):
        assert got.ms2_mz is None and got.ms2_intensity is None
        mz, intensity = spill.read_ms2(stem, got.ms2_seek_ptr)
        assert np.array_equal(mz, want.ms2_mz)
        assert np.array_equal(intensity, want.ms2_intensity)


def test_ms2_mz_round_trips_as_python_floats(sample):
    """``_three_scores_core`` keys peaks on ``round(mz, 4)``; the reference and
    the query must both round the Python way (see .baselines-260708.md §1.2)."""
    stem, _ = sample
    feat = spill.read_sample_features(stem, load_ms2=True)[0]
    assert all(type(v) is float for v in feat.ms2_mz.tolist())


def test_manifest_gates_reuse_on_the_fingerprint(sample):
    stem, _ = sample
    assert spill.sample_is_complete(stem, "fp-abc")
    assert not spill.sample_is_complete(stem, "fp-changed")
    assert not spill.sample_is_complete(stem.parent / "missing", "fp-abc")


def test_missing_manifest_means_the_sample_never_finished(sample):
    """A process killed between .mspec and the manifest must not look complete."""
    stem, _ = sample
    (stem.parent / "1.json").unlink()
    assert not spill.sample_is_complete(stem, "fp-abc")
    assert spill.scan_checkpoints(stem.parent) == []


def test_version_mismatch_raises_rather_than_degrading(sample):
    stem, _ = sample
    path = stem.parent / "1.mspec"
    raw = bytearray(path.read_bytes())
    raw[2] = 99                       # first byte of the version int32
    path.write_bytes(bytes(raw))
    with pytest.raises(spill.SpillFormatError):
        spill.read_sample_features(stem, load_ms2=True)


def test_dotted_sample_id_does_not_lose_its_suffix(tmp_path):
    """``ungrouped_<stem>`` ids carry dots; ``with_suffix`` would eat them."""
    stem = tmp_path / "ungrouped_MIX_1.0_100-129"
    spill.write_sample(stem, [_feature("F000", 2)], fingerprint="fp", sample_id="s")
    assert (tmp_path / "ungrouped_MIX_1.0_100-129.mspec").is_file()
    assert spill.sample_is_complete(stem, "fp")


def test_scan_and_clear(sample):
    stem, _ = sample
    found = spill.scan_checkpoints(stem.parent)
    assert [sid for sid, _ in found] == ["1"]
    assert spill.clear_work_dir(stem.parent) == 3
    assert spill.scan_checkpoints(stem.parent) == []
