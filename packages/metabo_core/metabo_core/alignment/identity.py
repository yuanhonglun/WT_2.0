"""Reliable, three-state MS2 identity evidence for alignment.

Geometric proximity says that two features *may* be the same compound.  MS2
can either support that identity, contradict it, or be too sparse to say.  The
last case must stay distinct from both answers: treating one shared fragment as
identity is a false merge, while treating missing spectra as different compounds
needlessly splits every sparse feature.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from metabo_core.algorithms.similarity import cosine_similarity


class IdentityState(str, Enum):
    """The only three scientifically meaningful identity outcomes."""

    SAME = "same"
    DIFFERENT = "different"
    UNJUDGEABLE = "unjudgeable"


@dataclass(frozen=True)
class IdentityEvidence:
    """Identity decision plus the measurements that produced it."""

    state: IdentityState
    cosine: float
    n_matched_fragments: int
    n_left_fragments: int
    n_right_fragments: int

    @property
    def is_reliable(self) -> bool:
        return self.state is not IdentityState.UNJUDGEABLE


def ms2_identity_evidence(
    left: Iterable[tuple[float, float]],
    right: Iterable[tuple[float, float]],
    *,
    mz_tolerance: float,
    same_threshold: float,
    min_fragments: int = 3,
    min_matched_fragments: int = 3,
) -> IdentityEvidence:
    """Return reliable three-state MS2 identity evidence.

    A cosine is identity evidence only when both spectra contain enough total
    fragments *and* the comparison actually matches enough fragments.  The
    latter guard matters when two rich spectra share only one ubiquitous ion:
    their cosine may still be high, but one match is not a molecular identity.
    """
    left_peaks = list(left)
    right_peaks = list(right)
    n_left, n_right = len(left_peaks), len(right_peaks)
    min_total = max(0, int(min_fragments))
    min_matched = max(0, int(min_matched_fragments))

    if n_left < min_total or n_right < min_total:
        return IdentityEvidence(
            IdentityState.UNJUDGEABLE, 0.0, 0, n_left, n_right,
        )

    cosine, n_matched = cosine_similarity(
        left_peaks, right_peaks, float(mz_tolerance),
    )
    if n_matched < min_matched:
        return IdentityEvidence(
            IdentityState.UNJUDGEABLE,
            float(cosine),
            int(n_matched),
            n_left,
            n_right,
        )

    state = (
        IdentityState.SAME
        if cosine >= float(same_threshold)
        else IdentityState.DIFFERENT
    )
    return IdentityEvidence(
        state, float(cosine), int(n_matched), n_left, n_right,
    )
