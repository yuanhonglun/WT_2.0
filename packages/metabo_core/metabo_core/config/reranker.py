"""Reranker config (off by default).

Used by ASFAM / DDA / GCMS annotation stages to optionally run the
candidate reranker. Mode strings map to:
  - "identity": IdentityReranker (no-op)
  - "default":  CosineRiReranker (RI Gaussian + library score)
  - "student":  student module loaded via student_module dotted path
  - "llm_explain": wraps the selected mode with LlmExplanationWrapper
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RerankerConfig:
    enabled: bool = False
    mode: str = "default"
    top_k_explained: int = 3
    alpha: float = 0.7
    ri_sigma: float = 10.0
    # Dotted path to a callable returning a CandidateReranker, e.g.
    # "student_pkg.reranker:build". Only used when mode == "student".
    student_module: str | None = None
