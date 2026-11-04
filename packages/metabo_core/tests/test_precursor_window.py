"""The library-match precursor candidate window is a parameter, not 0.5 Da.

Reviewer 4 (2026-10) showed that a fixed +/-0.5 Da window admits other-formula
high-scoring hits for 3.8 % of MassBank queries where 0.01 Da admits 0.5 %.
``match_feature_topn`` therefore takes ``precursor_window_da``; the default
(``AnnotationConfig.precursor_window_da`` = 0.5) keeps every existing caller
byte-identical, and a caller that knows how good a feature's precursor is passes
a window to match.
"""
from __future__ import annotations

import numpy as np

from metabo_core.annotation import build_index_from_list, match_feature_topn
from metabo_core.config import AnnotationConfig, SimilarityConfig
from metabo_core.models import CandidateFeature

PEAKS = [(100.0, 1000.0), (150.0, 800.0), (200.0, 600.0)]


def _feature(precursor_mz: float) -> CandidateFeature:
    feat = CandidateFeature(
        feature_id="F1", segment_name="180-182", replicate_id=1,
        precursor_mz_nominal=int(round(precursor_mz)),
        rt_apex=5.0, rt_left=4.9, rt_right=5.1,
        ms2_mz=np.array([m for m, _ in PEAKS]),
        ms2_intensity=np.array([i for _, i in PEAKS]),
        n_fragments=len(PEAKS),
    )
    feat.ms1_precursor_mz = precursor_mz
    return feat


def _library():
    # Identical spectra, precursors 0.3 Da apart: only the window tells them apart.
    return build_index_from_list([
        {"mz": [m for m, _ in PEAKS], "intensity": [i for _, i in PEAKS],
         "metadata": {"precursor_mz": 181.0707, "name": "Near"}},
        {"mz": [m for m, _ in PEAKS], "intensity": [i for _, i in PEAKS],
         "metadata": {"precursor_mz": 181.3707, "name": "Far"}},
    ])


def _match(window=None, annotation=None):
    lib = _library()
    return match_feature_topn(
        _feature(181.0707), lib["spectra"], lib["index"],
        annotation or AnnotationConfig(similarity_threshold=0.5, min_matched_peaks=2,
                                       min_matched_pct=0.5),
        SimilarityConfig(mz_tolerance=0.02, ms1_tolerance=0.01),
        top_n=5, csr=lib.get("csr"), precursor_window_da=window,
    )


def test_default_window_is_the_historical_half_dalton():
    assert AnnotationConfig().precursor_window_da == 0.5
    names = {m.name for m in _match()}
    assert names == {"Near", "Far"}


def test_a_tight_window_excludes_the_reference_a_third_of_a_dalton_away():
    names = {m.name for m in _match(window=0.01)}
    assert names == {"Near"}


def test_the_config_default_applies_when_no_per_call_window_is_given():
    cfg = AnnotationConfig(similarity_threshold=0.5, min_matched_peaks=2,
                           min_matched_pct=0.5, precursor_window_da=0.05)
    names = {m.name for m in _match(annotation=cfg)}
    assert names == {"Near"}


def test_the_per_call_window_overrides_the_config_default():
    cfg = AnnotationConfig(similarity_threshold=0.5, min_matched_peaks=2,
                           min_matched_pct=0.5, precursor_window_da=0.05)
    names = {m.name for m in _match(window=0.5, annotation=cfg)}
    assert names == {"Near", "Far"}


def test_the_window_only_filters_candidates_and_never_changes_a_score():
    wide = {m.name: m.score for m in _match(window=0.5)}
    tight = {m.name: m.score for m in _match(window=0.01)}
    assert tight["Near"] == wide["Near"]
