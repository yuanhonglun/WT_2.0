"""Three-state MS2 identity evidence used by ASFAM Stage 7."""
from __future__ import annotations

from metabo_core.alignment.identity import IdentityState, ms2_identity_evidence


def _identity(left, right, **kwargs):
    return ms2_identity_evidence(
        left,
        right,
        mz_tolerance=0.02,
        same_threshold=0.7,
        min_fragments=3,
        min_matched_fragments=3,
        **kwargs,
    )


def test_one_shared_fragment_with_cosine_one_is_unjudgeable():
    evidence = _identity([(100.0, 1.0)], [(100.0, 1.0)])

    assert evidence.state is IdentityState.UNJUDGEABLE
    assert evidence.n_matched_fragments == 0  # total-count gate avoids scoring


def test_rich_spectra_with_only_one_actual_match_are_unjudgeable():
    evidence = _identity(
        [(100.0, 100.0), (150.0, 1.0), (200.0, 1.0), (250.0, 1.0)],
        [(100.0, 100.0), (350.0, 1.0), (400.0, 1.0), (450.0, 1.0)],
    )

    assert evidence.cosine > 0.99
    assert evidence.n_matched_fragments == 1
    assert evidence.state is IdentityState.UNJUDGEABLE


def test_three_reliable_matches_above_threshold_are_same():
    spectrum = [(100.0, 100.0), (150.0, 50.0), (200.0, 25.0)]
    evidence = _identity(spectrum, spectrum)

    assert evidence.state is IdentityState.SAME
    assert evidence.n_matched_fragments == 3
    assert evidence.cosine == 1.0


def test_three_reliable_matches_below_threshold_are_different():
    evidence = _identity(
        [(100.0, 100.0), (150.0, 10.0), (200.0, 10.0)],
        [(100.0, 10.0), (150.0, 100.0), (200.0, 10.0)],
    )

    assert evidence.n_matched_fragments == 3
    assert evidence.cosine < 0.7
    assert evidence.state is IdentityState.DIFFERENT
