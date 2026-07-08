"""Cross-run feature matching, pure Python.

Given a list of FeedbackEntry (from a previously-saved sidecar) and a
list of feature stand-ins exposing ``feature_id`` and ``signature``,
classify each entry into one of: exact id match, auto-matched (unique
signature match), ambiguous (multiple signature matches), orphan (no
match in tolerance window).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .models import FeatureSignature, FeedbackEntry

LC_MZ_PPM = 5.0
LC_MZ_MIN_DA = 0.001
GC_MZ_DA = 0.5
RT_TOLERANCE_MIN = 0.05


class _HasSignature(Protocol):
    feature_id: str
    signature: FeatureSignature


@dataclass
class MatchResult:
    exact: dict = field(default_factory=dict)         # feature_id_at_run -> feature
    auto_matched: dict = field(default_factory=dict)  # feature_id_at_run -> feature
    ambiguous: dict = field(default_factory=dict)     # feature_id_at_run -> list[feature]
    orphan: list = field(default_factory=list)        # list[entry]


def _is_gc_mode(mode: str) -> bool:
    return mode.startswith("gcms")


def _mz_within(target_mz: float, candidate_mz: float, mode: str) -> bool:
    if _is_gc_mode(mode):
        tol = GC_MZ_DA
    else:
        tol = max(LC_MZ_PPM * 1e-6 * target_mz, LC_MZ_MIN_DA)
    return abs(target_mz - candidate_mz) <= tol


def _rt_within(target_rt: float, candidate_rt: float) -> bool:
    return abs(target_rt - candidate_rt) <= RT_TOLERANCE_MIN


def match_entries_to_features(
    entries: list[FeedbackEntry],
    features: list[_HasSignature],
) -> MatchResult:
    """Classify entries against the current run's features."""
    by_id = {f.feature_id: f for f in features}
    result = MatchResult()
    for entry in entries:
        # 1. exact id match
        if entry.feature_id_at_run in by_id:
            result.exact[entry.feature_id_at_run] = by_id[entry.feature_id_at_run]
            continue
        # 2. signature match within tolerance, mode-isolated
        sig = entry.feature_signature
        candidates = [
            f for f in features
            if f.signature.mode == sig.mode
            and _mz_within(sig.mz, f.signature.mz, sig.mode)
            and _rt_within(sig.rt, f.signature.rt)
        ]
        if len(candidates) == 1:
            result.auto_matched[entry.feature_id_at_run] = candidates[0]
        elif len(candidates) >= 2:
            result.ambiguous[entry.feature_id_at_run] = candidates
        else:
            result.orphan.append(entry)
    return result
