"""Annotation configuration shared by core and apps."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnnotationConfig:
    """Parameters for library annotation.

    ``similarity_threshold`` is the **display-time** threshold used to
    decide whether a feature counts as "annotated" in the GUI / export
    (``annotated`` column = True iff top-1 score ≥ this). The pipeline
    itself always emits every plausible hit — apps override
    ``similarity_threshold`` to 0.0 inside their annotate stage so the
    feature-level ``annotation_matches`` list keeps all hits regardless
    of score.
    """
    similarity_threshold: float = 0.7
    min_matched_peaks: int = 3
    # Minimum number of peaks a spectrum must carry — on *each* side — to be
    # considered for matching. A query with fewer peaks returns no hits; a
    # reference with fewer peaks is skipped as a candidate. Default 2 keeps
    # GC-MS / DDA byte-identical. An app opts down to 1 (via its config view)
    # to let sparse spectra participate — notably precursor-only reference
    # spectra (1 peak, common for [M+Na]+ adducts) and MS1-dominant queries.
    # This is a per-side *count* floor only; the spectral-quality gates
    # (min_matched_pct / min_wdp) still apply, so relaxing it never bypasses
    # them.
    min_peaks_to_match: int = 2
    min_matched_pct: float = 0.25
    # Minimum weighted dot product (``CompositeSimilarityResult.wdp``) required
    # to accept a hit. Guards against inflated-query false positives: a query
    # with many noise peaks can trivially cover a small reference's few
    # significant peaks (matched_pct=1.0, high rdp) and clear the
    # high-confidence line while the true m/z-weighted spectral shape (wdp) is
    # near zero. Defaults to 0.0 (disabled) so GC-MS / DDA annotation paths are
    # byte-identical unless an app opts in via its config view.
    min_wdp: float = 0.0
    use_rt: bool = False
    top_n: int = 5
