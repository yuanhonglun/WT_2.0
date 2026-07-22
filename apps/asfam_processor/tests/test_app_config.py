"""Regression tests for shared and ASFAM-side config composition."""
import json

from asfam.config import ProcessingConfig
from metabo_core.config import (
    AlignmentConfig,
    AnnotationConfig,
    PeakDetectionConfig,
    SimilarityConfig,
    SmoothingConfig,
)


def test_smoothing_view_reflects_flat_fields():
    cfg = ProcessingConfig()
    view = cfg.smoothing_view()
    assert isinstance(view, SmoothingConfig)
    assert view.method == cfg.eic_smoothing_method
    assert view.window_length == cfg.eic_smoothing_window
    assert view.polyorder == cfg.eic_smoothing_polyorder


def test_peak_detection_view_carries_user_knobs():
    cfg = ProcessingConfig()
    view = cfg.peak_detection_view()
    assert isinstance(view, PeakDetectionConfig)
    # ``peak_detection_view`` is the legacy MS2-side view, now backed
    # by the nested ``ms2_peak`` config.
    assert view is cfg.ms2_peak
    assert view.min_amplitude == cfg.ms2_peak.min_amplitude
    assert view.min_data_points == cfg.ms2_peak.min_data_points
    assert view.gaussian_threshold == cfg.ms2_peak.gaussian_threshold


def test_processing_config_has_split_ms1_ms2_peak_configs():
    cfg = ProcessingConfig()
    assert isinstance(cfg.ms1_peak, PeakDetectionConfig)
    assert isinstance(cfg.ms2_peak, PeakDetectionConfig)
    # Distinct instances (factories return fresh objects).
    assert cfg.ms1_peak is not cfg.ms2_peak
    # MS1 / MS2 amplitude defaults differ on purpose (MS2 weaker).
    assert cfg.ms1_peak.min_amplitude == 500.0
    assert cfg.ms2_peak.min_amplitude == 200.0


def test_similarity_view_uses_eic_tolerances():
    cfg = ProcessingConfig()
    view = cfg.similarity_view()
    assert isinstance(view, SimilarityConfig)
    assert view.mz_tolerance == cfg.eic_mz_tolerance
    assert view.ms1_tolerance == cfg.ms1_mz_tolerance
    assert view.use_rt == cfg.matchms_use_rt


def test_annotation_view_carries_matchms_thresholds():
    cfg = ProcessingConfig()
    view = cfg.annotation_view()
    assert isinstance(view, AnnotationConfig)
    # Pipeline emits every plausible match; the score gate is the
    # single user-tunable ``matchms_similarity_threshold`` (= GUI
    # "Library Match Thr"), applied only at display / export time.
    assert view.similarity_threshold == 0.0
    # Two-tier confidence (mirrors MS-DIAL suggested vs reference-matched):
    # the emit floor is 1 matched peak and 1-peak spectra participate on both
    # sides, so sparse hits are kept as suggestions. The >=3 high-confidence
    # requirement lives on ``matchms_min_matched_peaks`` and is applied at the
    # export / GUI ``annotated`` flag, NOT as an emit gate.
    assert view.min_matched_peaks == 1
    assert view.min_peaks_to_match == 1
    assert cfg.matchms_min_matched_peaks == 3
    assert view.min_matched_pct == cfg.matchms_min_matched_pct


def test_alignment_view_carries_alignment_fields():
    cfg = ProcessingConfig()
    view = cfg.alignment_view()
    assert isinstance(view, AlignmentConfig)
    assert view.rt_tolerance == cfg.alignment_rt_tolerance
    assert view.mz_tolerance == cfg.alignment_mz_tolerance


def test_processing_config_save_load_round_trip(tmp_path):
    cfg = ProcessingConfig()
    cfg.ms2_peak.min_amplitude = 1234.0
    cfg.ms1_peak.min_amplitude = 777.0
    cfg.matchms_similarity_threshold = 0.42
    path = tmp_path / "cfg.json"
    cfg.save(path)
    restored = ProcessingConfig.load(path)
    assert isinstance(restored.ms1_peak, PeakDetectionConfig)
    assert isinstance(restored.ms2_peak, PeakDetectionConfig)
    assert restored.ms2_peak.min_amplitude == 1234.0
    assert restored.ms1_peak.min_amplitude == 777.0
    assert restored.matchms_similarity_threshold == 0.42


def test_peak_detector_default_and_msdial_view():
    from asfam.config import ProcessingConfig
    from metabo_core.config.msdial_peak_spotting import MsdialPeakSpottingConfig
    cfg = ProcessingConfig()
    assert cfg.peak_detector == "msdial"  # default flipped to msdial (0.7.260626.9)
    assert isinstance(cfg.msdial_peak, MsdialPeakSpottingConfig)
    cfg2 = ProcessingConfig(peak_detector="builtin")
    assert cfg2.peak_detector == "builtin"


def test_save_load_preserves_msdial_peak_type(tmp_path):
    from asfam.config import ProcessingConfig
    from metabo_core.config.msdial_peak_spotting import MsdialPeakSpottingConfig
    p = tmp_path / "cfg.json"
    ProcessingConfig(peak_detector="msdial").save(p)
    restored = ProcessingConfig.load(p)
    assert restored.peak_detector == "msdial"
    # MUST be reconstructed as the MS-DIAL dataclass, not PeakDetectionConfig,
    # and MS-DIAL-only fields must survive the round-trip.
    assert isinstance(restored.msdial_peak, MsdialPeakSpottingConfig)
    assert restored.msdial_peak.mass_slice_width == 0.1
    assert restored.msdial_peak.centroid_ms1_tolerance == 0.01


def test_ms2_deconv_default_and_msdec_config():
    from asfam.config import ProcessingConfig
    from metabo_core.config.msdec import MsdecConfig
    cfg = ProcessingConfig()
    # 档 C (msdec) is the default since 2026-06-29 (user decision after
    # validation); 档 B (apex) stays reachable / rollback-able.
    assert cfg.ms2_deconv == "msdec"
    assert isinstance(cfg.msdec, MsdecConfig)
    cfg2 = ProcessingConfig(ms2_deconv="apex")
    assert cfg2.ms2_deconv == "apex"


def test_save_load_preserves_msdec_type(tmp_path):
    from asfam.config import ProcessingConfig
    from metabo_core.config.msdec import MsdecConfig
    p = tmp_path / "cfg.json"
    cfg = ProcessingConfig(ms2_deconv="msdec")
    cfg.msdec.sigma_window = 0.7
    cfg.save(p)
    restored = ProcessingConfig.load(p)
    assert restored.ms2_deconv == "msdec"
    # Must be reconstructed as MsdecConfig, not silently dropped.
    assert isinstance(restored.msdec, MsdecConfig)
    assert restored.msdec.sigma_window == 0.7
    assert restored.msdec.centroid_ms2_tolerance == 0.025


def test_msdec_view_overrides_min_amplitude_with_weak_floor():
    from asfam.config import ProcessingConfig
    from metabo_core.config.msdec import MsdecConfig
    cfg = ProcessingConfig()
    # Default: weak-signal floor on (user decision).
    assert cfg.msdec_use_weak_floor is True
    view = cfg.msdec_view()
    assert isinstance(view, MsdecConfig)
    # ASFAM lowers the model-peak amplitude floor to the weaker-signal
    # MS1 floor (ms1_min_height) so 档 B-recovered weak MS2 survive.
    assert view.min_amplitude == cfg.ms1_min_height
    # Other MS-DIAL defaults are untouched.
    assert view.sigma_window == 0.5
    assert view.apex_model_tolerance == 2


def test_msdec_view_high_floor_ab_uses_msdec_min_amplitude():
    # A/B against MS-DIAL's faithful 1000 floor: turn off the weak-signal
    # override and the view uses MsdecConfig.min_amplitude directly, WITHOUT
    # touching ms1_min_height (which would contaminate MS1 peak finding).
    from asfam.config import ProcessingConfig
    cfg = ProcessingConfig()
    cfg.msdec_use_weak_floor = False
    cfg.msdec.min_amplitude = 1000.0
    view = cfg.msdec_view()
    assert view.min_amplitude == 1000.0
    assert cfg.ms1_min_height == 500.0  # unchanged


def test_isf_rt_tolerance_default_and_round_trip(tmp_path):
    """ISF gets its own RT tolerance (no longer reuses adduct's), tighter."""
    cfg = ProcessingConfig()
    # New dedicated key exists, default tighter than the borrowed adduct value.
    assert hasattr(cfg, "isf_rt_tolerance")
    assert cfg.isf_rt_tolerance == 0.035
    assert cfg.isf_rt_tolerance < cfg.adduct_rt_tolerance  # 0.035 < 0.05
    # Survives save/load.
    cfg.isf_rt_tolerance = 0.03
    p = tmp_path / "cfg.json"
    cfg.save(p)
    restored = ProcessingConfig.load(p)
    assert restored.isf_rt_tolerance == 0.03


def test_refiner_view_caps_the_rt_gate_however_wide_the_join_is():
    """The redundancy gate is 0.05 min and a user cannot widen it from the GUI.

    ``alignment_rt_tolerance`` is the *join* window, and it is already 0.2 —
    double MS-DIAL's — because the real cross-sample RT jitter needs it. The
    refiner is a different question: two spots have to be much closer than the
    match window to be one compound split in two. ``rt_tolerance_cap`` is the only
    thing holding those apart, so widening the join must not drag the redundancy
    gate along with it and start merging distinct compounds.
    """
    cfg = ProcessingConfig()
    cfg.alignment_rt_tolerance = 0.5
    assert cfg.refiner_view().rt_gate == 0.05

    cfg.alignment_rt_tolerance = 0.02      # below the cap, it still halves
    assert cfg.refiner_view().rt_gate == 0.01


def test_refiner_view_carries_the_ms2_identity_gate(tmp_path):
    """The cross-route gate, and the fragment tolerance it scores with.

    ``ms2_mz_tolerance`` is the joiner's on purpose: both passes score the same
    pair of spectra, and they must not disagree about what a matched fragment is.
    """
    cfg = ProcessingConfig()
    assert cfg.refiner_view().ms2_identity_threshold == cfg.alignment_ms1_covered_threshold
    assert cfg.refiner_view().ms2_mz_tolerance == cfg.eic_mz_tolerance

    cfg.alignment_ms1_covered_threshold = 0.8
    p = tmp_path / "cfg.json"
    cfg.save(p)
    assert ProcessingConfig.load(p).refiner_view().ms2_identity_threshold == 0.8


def test_asfam_explicitly_enables_alignment_correctness_mechanisms():
    cfg = ProcessingConfig()
    joiner = cfg.joiner_view()
    refiner = cfg.refiner_view()

    assert joiner.use_reliable_ms2_identity
    assert joiner.conserve_detected_peaks
    assert not joiner.use_ms2_identity_for_ms1
    assert not refiner.same_route_redundancy  # handled pre-gap-fill by ASFAM
    assert refiner.use_reliable_ms2_identity
    assert refiner.visible_keepers_only
    assert refiner.require_product_window_match
    assert refiner.require_cross_route_window_match
    assert refiner.preserve_cross_route_unique_detections


def test_shared_alignment_correctness_defaults_remain_legacy():
    from metabo_core.config import JoinerConfig, RefinerConfig

    joiner = JoinerConfig()
    refiner = RefinerConfig()
    assert not joiner.use_reliable_ms2_identity
    assert not joiner.conserve_detected_peaks
    assert joiner.use_ms2_identity_for_ms1
    assert refiner.same_route_redundancy
    assert not refiner.use_reliable_ms2_identity
    assert not refiner.visible_keepers_only
    assert not refiner.require_product_window_match
    assert not refiner.require_cross_route_window_match
    assert not refiner.preserve_cross_route_unique_detections


def test_asfam_identity_fragment_floor_cannot_be_loaded_below_three(tmp_path):
    cfg = ProcessingConfig()
    cfg.alignment_ms2_identity_min_fragments = 1
    cfg.alignment_ms2_identity_min_matched_fragments = 1
    path = tmp_path / "unsafe.json"
    cfg.save(path)

    restored = ProcessingConfig.load(path)

    assert restored.joiner_view().ms2_identity_min_fragments == 3
    assert restored.joiner_view().ms2_identity_min_matched_fragments == 3
    assert restored.refiner_view().ms2_identity_min_fragments == 3
    assert restored.refiner_view().ms2_identity_min_matched_fragments == 3
