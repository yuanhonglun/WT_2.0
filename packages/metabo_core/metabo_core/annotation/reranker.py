"""Annotation reranker contract.

Defines the data types and Protocol the student annotation re-ranker
fills in. See docs/student/annotation-reranker-contract.md for the
end-to-end contract.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class AnnotationCandidate:
    """Normalized candidate across LC-MS and GC-MS paths.

    LC-MS callers populate {name, formula, score, n_matched, wdp, sdp, rdp,
    ref_peaks, ref_precursor_mz, adduct}. GC-MS callers additionally populate
    {inchikey, ref_ri, ref_rt} as first-class fields, plus
    {chrom_score, spectral_score, matched_pct, acquired_ion_count} via the
    ``extras`` dict. The student's reranker must preserve ``extras``
    round-trip.
    """
    rank: int                              # 1-indexed; reranker MAY rewrite this
    name: str = ""
    formula: str = ""
    inchikey: str = ""
    adduct: str = ""
    score: float = 0.0                     # library composite (input score)
    n_matched: int = 0
    wdp: float = 0.0
    sdp: float = 0.0
    rdp: float = 0.0
    ref_peaks: list[tuple[float, float]] | None = None
    ref_precursor_mz: float | None = None
    ref_ri: float | None = None
    ref_rt: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class RankingInput:
    """Evidence bundle passed to a reranker for one feature."""
    feature_id: str
    mode: str                              # "asfam" | "dda" | "gcms_fullscan" | "gcms_csim"
    measured_precursor_mz: float | None
    measured_rt: float | None              # in minutes
    measured_ri: float | None              # GCMS only; None elsewhere
    measured_spectrum: list[tuple[float, float]]
    candidates: list[AnnotationCandidate]
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class RankingResult:
    """Reranker output for one feature."""
    feature_id: str
    candidates: list[AnnotationCandidate]  # re-ordered; rank field MUST be re-assigned
    explanations: dict[int, str] = field(default_factory=dict)  # candidate.rank -> NL reason
    reranker_name: str = ""
    reranker_version: str = ""


@runtime_checkable
class CandidateReranker(Protocol):
    """Contract the student fills in. See docs/student/annotation-reranker-contract.md."""
    name: str
    version: str

    def rerank(self, request: RankingInput) -> RankingResult: ...


class IdentityReranker:
    """No-op reranker. Used when reranker is disabled or as a test baseline."""

    name = "identity"
    version = "1.0"

    def rerank(self, request: RankingInput) -> RankingResult:
        # Preserve input order; re-assert rank field for consistency.
        out: list[AnnotationCandidate] = []
        for i, cand in enumerate(request.candidates):
            cand.rank = i + 1
            out.append(cand)
        return RankingResult(
            feature_id=request.feature_id,
            candidates=out,
            explanations={},
            reranker_name=self.name,
            reranker_version=self.version,
        )


class CosineRiReranker:
    """Default fallback reranker.

    Composite score: ``final = alpha * lib_score + (1 - alpha) * ri_term``
    where ``ri_term = exp(-((measured_ri - ref_ri) / ri_sigma) ** 2)`` when
    both ``measured_ri`` and ``ref_ri`` are present, else ``0``. With LC-MS
    requests (``measured_ri is None``) the RI term is zero for all
    candidates, so the order reduces to pure library score — i.e. identity
    behavior with re-asserted ranks.
    """

    name = "cosine_ri"
    version = "1.0"

    def __init__(self, alpha: float = 0.7, ri_sigma: float = 10.0):
        self.alpha = alpha
        self.ri_sigma = ri_sigma

    def rerank(self, request: RankingInput) -> RankingResult:
        scored: list[tuple[float, AnnotationCandidate]] = []
        for cand in request.candidates:
            ri_term = 0.0
            if request.measured_ri is not None and cand.ref_ri is not None:
                delta = (request.measured_ri - cand.ref_ri) / self.ri_sigma
                ri_term = math.exp(-(delta * delta))
            final = self.alpha * cand.score + (1.0 - self.alpha) * ri_term
            scored.append((final, cand))

        scored.sort(key=lambda t: t[0], reverse=True)
        ordered: list[AnnotationCandidate] = []
        for i, (_score, cand) in enumerate(scored):
            cand.rank = i + 1
            ordered.append(cand)
        return RankingResult(
            feature_id=request.feature_id,
            candidates=ordered,
            explanations={},
            reranker_name=self.name,
            reranker_version=self.version,
        )


@runtime_checkable
class LlmExplainer(Protocol):
    """Minimal explainer contract.

    Plan 2's ``LlmProvider`` (in ``metabo_core.copilot``) will satisfy
    this via a thin adapter when it lands. Until then, ``StubLlmExplainer``
    is the only implementation — used in tests and as a no-network default.
    """
    name: str

    def explain(
        self,
        request: RankingInput,
        candidate: AnnotationCandidate,
    ) -> str: ...


class StubLlmExplainer:
    """Deterministic offline explainer. Returns a templated string."""

    name = "stub"

    def explain(
        self,
        request: RankingInput,
        candidate: AnnotationCandidate,
    ) -> str:
        bits = [
            f"Candidate {candidate.name!r}",
            f"ranked {candidate.rank}",
            f"with library score {candidate.score:.2f}",
        ]
        if candidate.ref_ri is not None and request.measured_ri is not None:
            bits.append(
                f"and RI delta {request.measured_ri - candidate.ref_ri:+.1f}"
            )
        return "; ".join(bits) + "."


class LlmExplanationWrapper:
    """Wrap any reranker, add a one-line explanation for the top-K candidates."""

    version = "1.0"

    def __init__(
        self,
        inner: CandidateReranker,
        explainer: LlmExplainer,
        top_k_explained: int = 3,
    ):
        self.inner = inner
        self.explainer = explainer
        self.top_k_explained = top_k_explained
        explainer_name = getattr(explainer, "name", explainer.__class__.__name__)
        self.name = f"{inner.name}+{explainer_name}"

    def rerank(self, request: RankingInput) -> RankingResult:
        result = self.inner.rerank(request)
        if not result.candidates:
            return result
        explanations: dict[int, str] = {}
        for cand in result.candidates[: self.top_k_explained]:
            explanations[cand.rank] = self.explainer.explain(request, cand)
        result.explanations = explanations
        result.reranker_name = self.name
        result.reranker_version = self.version
        return result


def build_reranker(cfg) -> "CandidateReranker | None":
    """Build a reranker from RerankerConfig, or return None if disabled.

    Imports RerankerConfig lazily to keep reranker.py free of config-package
    dependencies (avoids circular imports).
    """
    if not cfg.enabled:
        return None

    mode = cfg.mode
    if mode == "identity":
        return IdentityReranker()
    if mode == "default":
        return CosineRiReranker(alpha=cfg.alpha, ri_sigma=cfg.ri_sigma)
    if mode == "llm_explain":
        inner = CosineRiReranker(alpha=cfg.alpha, ri_sigma=cfg.ri_sigma)
        return LlmExplanationWrapper(
            inner=inner,
            explainer=StubLlmExplainer(),
            top_k_explained=cfg.top_k_explained,
        )
    if mode == "student":
        if not cfg.student_module:
            raise ValueError(
                "student_module must be set (e.g. 'mypkg.reranker:build') "
                "when reranker mode is 'student'"
            )
        return _load_student_reranker(cfg.student_module)

    raise ValueError(f"unknown reranker mode {mode!r}")


def _load_student_reranker(dotted: str):
    """Load a CandidateReranker from a 'module.path:callable' string."""
    import importlib
    if ":" not in dotted:
        raise ValueError(
            f"student_module must be 'module:callable' form, got {dotted!r}"
        )
    mod_path, attr = dotted.split(":", 1)
    mod = importlib.import_module(mod_path)
    factory = getattr(mod, attr)
    return factory()
