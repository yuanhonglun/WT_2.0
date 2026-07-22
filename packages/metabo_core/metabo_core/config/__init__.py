"""Reusable algorithm-level configuration objects.

These dataclasses are meant to be composed by app-level configs (such as the
ASFAM ``ProcessingConfig``) and consumed by core algorithms or future apps
that do not want to inherit ASFAM's full parameter surface.
"""
from metabo_core.config.smoothing import SmoothingConfig
from metabo_core.config.peak_detection import (
    PeakDetectionConfig,
    lc_ms1_peak_config,
    lc_ms2_peak_config,
    gc_peak_config,
)
from metabo_core.config.similarity import SimilarityConfig
from metabo_core.config.annotation import AnnotationConfig, ConfidenceConfig
from metabo_core.config.alignment import (
    AlignmentConfig,
    GapFillConfig,
    JoinerConfig,
    RefinerConfig,
)
from metabo_core.config.reranker import RerankerConfig
from metabo_core.config.msdial_peak_spotting import (
    MsdialPeakSpottingConfig,
    lc_msdial_config,
)
from metabo_core.config.msdec import (
    MsdecConfig,
    lc_msdec_config,
)

__all__ = [
    "SmoothingConfig",
    "PeakDetectionConfig",
    "lc_ms1_peak_config",
    "lc_ms2_peak_config",
    "gc_peak_config",
    "SimilarityConfig",
    "AnnotationConfig",
    "ConfidenceConfig",
    "AlignmentConfig",
    "GapFillConfig",
    "JoinerConfig",
    "RefinerConfig",
    "RerankerConfig",
    "MsdialPeakSpottingConfig",
    "lc_msdial_config",
    "MsdecConfig",
    "lc_msdec_config",
]
