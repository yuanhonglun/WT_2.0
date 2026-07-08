"""Display-only spectrum transforms.

DISPLAY / EXPORT ONLY. These helpers MUST NOT be used in the matching path
(``metabo_core.annotation.library`` / ``composite_similarity_breakdown``),
which consumes raw, unnormalized reference peaks. Kept in ``metabo_gui`` (not
``metabo_core``) so it is physically out of reach of the scorer.
"""
from __future__ import annotations


def normalize_for_display(
    peaks: list[tuple[float, float]],
    target: float = 100.0,
) -> list[tuple[float, float]]:
    """Base-peak normalize a COPY of ``peaks`` to ``target`` (MS-DIAL Relative).

    Returns a NEW list; never mutates the input. No m/z binning (MS-DIAL does
    not bin the displayed reference spectrum). Zero/empty inputs are returned
    unscaled (no division by zero).
    """
    if not peaks:
        return []
    mx = max(v for _, v in peaks)
    if mx <= 0:
        return [(float(m), float(v)) for m, v in peaks]
    scale = target / mx
    return [(float(m), float(v) * scale) for m, v in peaks]


def _display_cosine(
    a: list[tuple[float, float]],
    b: list[tuple[float, float]],
    tol: float = 0.02,
) -> float:
    """Cheap display-only cosine for picking the closest CE variant.

    NOT the annotation scorer — this never feeds matching/counts. Greedy
    nearest-m/z pairing within ``tol``; base-peak normalized so absolute
    intensity units do not matter.
    """
    if not a or not b:
        return 0.0
    amax = max(v for _, v in a) or 1.0
    bmax = max(v for _, v in b) or 1.0
    an = [(m, v / amax) for m, v in a]
    bn = sorted(((m, v / bmax) for m, v in b), key=lambda p: p[0])
    dot = na = 0.0
    used = [False] * len(bn)
    for m, v in an:
        na += v * v
        best_j, best_d = -1, tol
        for j, (mb, _vb) in enumerate(bn):
            if used[j]:
                continue
            d = abs(mb - m)
            if d <= best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            used[best_j] = True
            dot += v * bn[best_j][1]
    nb = sum(v * v for _, v in bn)
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na ** 0.5 * nb ** 0.5)


def variant_best_matching_query(
    query_peaks: list[tuple[float, float]],
    matches: list,
    top_name: str,
) -> int:
    """Index into ``matches`` of the same-``top_name`` entry whose ref peaks
    best match ``query_peaks`` (display cosine). Returns 0 if no same-name
    candidate or ``matches`` is empty. DISPLAY ONLY — does not change which
    match is the annotation/selected result.
    """
    if not matches:
        return 0
    best_idx, best_cos = 0, -1.0
    for i, m in enumerate(matches):
        if getattr(m, "name", "") != top_name:
            continue
        cos = _display_cosine(query_peaks, list(getattr(m, "ref_peaks", None) or []))
        if cos > best_cos:
            best_cos, best_idx = cos, i
    return best_idx
