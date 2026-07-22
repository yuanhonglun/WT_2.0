"""The one definition of "this feature is confidently identified".

MS-DIAL calls it ``IsReferenceMatched``. Two consumers need it and they must
agree: the ``annotated`` column of ``features.csv``, and the alignment refiner,
whose first placement loop iterates the identified spots before the rest.

They live on opposite sides of the ``metabo_core`` boundary — the refiner may
not import app code — so a second copy of the rule inside the refiner would be
invisible to every test and would silently regroup the refiner's two loops the
day either threshold moved.
"""
from __future__ import annotations

from typing import Any

from metabo_core.config.annotation import ConfidenceConfig


def is_high_confidence(feature: Any, config: ConfidenceConfig) -> bool:
    """Does ``feature``'s selected annotation clear both confidence gates?

    Annotation emits sub-threshold hits as *suggestions* — they carry a name and
    a score but must not count as identified. A hit is high-confidence only when
    its composite score clears ``score_threshold`` **and** it matched at least
    ``min_matched_peaks`` reference peaks; a sparse match that happens to score
    well (a precursor-only reference, say) stays a suggestion.

    ``feature`` is anything exposing ``selected_annotation``.
    """
    selected = getattr(feature, "selected_annotation", None)
    if selected is None or selected.score is None:
        return False
    return (
        float(selected.score) >= config.score_threshold
        and int(getattr(selected, "n_matched", 0) or 0) >= config.min_matched_peaks
    )
