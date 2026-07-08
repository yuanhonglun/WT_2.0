"""Regression tests for shared feature/annotation models."""
import numpy as np

from metabo_core.models import (
    AnnotationMatch,
    CandidateFeature,
    Feature,
    DetectedPeak,
    ProductIonEIC,
    Scan,
)


def _candidate(precursor_nominal: int = 181) -> CandidateFeature:
    return CandidateFeature(
        feature_id="F00001",
        segment_name="100-200",
        replicate_id=1,
        precursor_mz_nominal=precursor_nominal,
        rt_apex=5.0,
        rt_left=4.9,
        rt_right=5.1,
        ms2_mz=np.array([100.0, 150.0]),
        ms2_intensity=np.array([1000.0, 500.0]),
        n_fragments=2,
    )


def test_candidate_precursor_falls_back_to_nominal():
    feat = _candidate()
    assert feat.precursor_mz == 181.0
    feat.inferred_mz = 181.07
    assert feat.precursor_mz == 181.07
    feat.ms1_precursor_mz = 181.0707
    assert feat.precursor_mz == 181.0707


def test_candidate_ms2_as_list_returns_python_floats():
    feat = _candidate()
    pairs = feat.ms2_as_list()
    assert pairs == [(100.0, 1000.0), (150.0, 500.0)]


def test_feature_selected_annotation_returns_match_or_none():
    feat = Feature(
        feature_id="F00001",
        precursor_mz=181.0707,
        rt=5.0,
        rt_left=4.9,
        rt_right=5.1,
        signal_type="ms1_detected",
        ms2_mz=np.array([100.0]),
        ms2_intensity=np.array([1000.0]),
        n_fragments=1,
    )
    assert feat.selected_annotation is None
    match = AnnotationMatch(rank=1, name="Test", score=0.95, n_matched=2)
    feat.annotation_matches = [match]
    assert feat.selected_annotation is match


def test_scan_ms1_construct_minimal_fields():
    scan = Scan(
        scan_id=1,
        ms_level=1,
        rt=0.5,
        mz_array=np.array([100.0, 200.0]),
        intensity_array=np.array([500.0, 1000.0]),
    )
    assert scan.ms_level == 1
    assert scan.precursor_mz is None
    assert scan.precursor_intensity is None
    assert scan.isolation_window_lower is None
    assert scan.isolation_window_upper is None


def test_scan_ms2_carries_precursor_and_isolation_window():
    scan = Scan(
        scan_id=42,
        ms_level=2,
        rt=1.5,
        mz_array=np.array([50.0, 75.0]),
        intensity_array=np.array([100.0, 200.0]),
        precursor_mz=270.123,
        precursor_intensity=12345.0,
        isolation_window_lower=269.123,
        isolation_window_upper=271.123,
    )
    assert scan.ms_level == 2
    assert scan.precursor_mz == 270.123
    assert scan.precursor_intensity == 12345.0
    assert scan.isolation_window_lower == 269.123
    assert scan.isolation_window_upper == 271.123


def test_detected_peak_and_eic_construct_with_required_fields():
    peak = DetectedPeak(
        precursor_mz_nominal=100,
        product_mz=120.0,
        rt_apex=5.0,
        rt_left=4.9,
        rt_right=5.1,
        apex_index=10,
        left_index=8,
        right_index=12,
        height=1000.0,
        area=2000.0,
    )
    assert peak.sn_ratio == 0.0
    eic = ProductIonEIC(
        precursor_mz_nominal=100,
        product_mz=120.0,
        rt_array=np.array([0.0, 1.0]),
        intensity_array=np.array([0.0, 1000.0]),
    )
    assert eic.smoothed_intensity is None
