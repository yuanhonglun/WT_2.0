"""ASFAM Stage 7 conservation, reconciliation, and consensus invariants."""
from __future__ import annotations

from collections import Counter
import json
import numpy as np
import pytest

from asfam.config import ProcessingConfig
from asfam.io.eic_store import load_spot_map
from asfam.models import CandidateFeature
from asfam.pipeline import stage7_alignment
from asfam.pipeline.stage7_alignment import (
    GapFillContext,
    MERGE_MAP_NAME,
    _NaturalMs1PeakIndex,
    _candidate_is_high_confidence,
    _one_sample,
    run_stage7,
)
from asfam.pipeline.stage7_reconcile import (
    audit_spots,
    collapse_ms1_natural_peak_aliases,
    gap_fill_target_for_spot,
    reconcile_spots,
    recompute_alignment_center,
)
from metabo_core.alignment.joiner import AlignmentSpot, FoldedPeak
from metabo_core.alignment.gap_filler import FILLED, GapFillResult
from metabo_core.models import AnnotationMatch


_SAME = [(100.0, 1000.0), (150.0, 700.0), (200.0, 300.0)]
_DIFFERENT = [(100.0, 1.0), (150.0, 1000.0), (200.0, 1.0)]


def _peak(
    sample: str,
    name: str,
    mz: float,
    rt: float,
    *,
    height: float = 1000.0,
    spectrum=_SAME,
    signal_type: str = "ms1_detected",
    window: int = 300,
) -> CandidateFeature:
    return CandidateFeature(
        feature_id=f"rep{sample}_{name}",
        segment_name=f"{window}-{window + 1}",
        replicate_id=int(sample),
        precursor_mz_nominal=window,
        rt_apex=rt,
        rt_left=rt - 0.05,
        rt_right=rt + 0.05,
        ms2_mz=np.asarray([p[0] for p in spectrum], dtype=np.float64),
        ms2_intensity=np.asarray([p[1] for p in spectrum], dtype=np.float64),
        n_fragments=len(spectrum),
        ms1_precursor_mz=float(window) + 0.5,
        ms1_quant_mz=mz if signal_type != "ms2_only" else None,
        ms2_rep_ion_mz=mz if signal_type == "ms2_only" else None,
        signal_type=signal_type,
        ms1_height=height,
        ms1_area=height * 2,
        ms1_sn=10.0,
    )


def _gap_context(tmp_path) -> GapFillContext:
    return GapFillContext(
        sample_files={"1": ["one.mzML"], "2": ["two.mzML"]},
        output_dir=str(tmp_path),
    )


def _spot(index: int, sample_id: str, peak: CandidateFeature) -> AlignmentSpot:
    return AlignmentSpot(
        index=index,
        master_mz=float(peak.align_mz),
        master_rt=peak.rt_apex,
        origin_sample_id=sample_id,
        peaks={sample_id: peak},
        representative_sample_id=sample_id,
        representative_ms2=(peak.ms2_mz, peak.ms2_intensity),
    )


def test_same_route_reconciliation_happens_before_gap_fill(tmp_path, monkeypatch):
    features = {
        "1": [
            _peak("1", "keeper", 300.000, 5.000, height=9000.0),
            _peak("1", "loser", 300.005, 5.010, height=50.0),
        ],
        "2": [_peak("2", "detected", 300.005, 5.010, height=100.0)],
    }
    seen = {}

    def fake_gap_fill(spots, sample_ids, config, context, progress_callback=None):
        assert len(spots) == 1, "duplicate spots must reconcile before gap fill"
        seen["statuses"] = {
            sid: peak.gap_fill_status for sid, peak in spots[0].peaks.items()
        }
        # A legacy mark-only flow reaches here with sample 2 absent from the
        # keeper and would integrate a filled value over its natural detection.
        assert set(spots[0].peaks) == {"1", "2"}

    monkeypatch.setattr(stage7_alignment, "run_gap_fill", fake_gap_fill)
    stats = {}
    out = run_stage7(
        features,
        ProcessingConfig(),
        gap_fill=_gap_context(tmp_path),
        stats_out=stats,
    )

    assert len(out) == 1
    assert seen["statuses"] == {"1": "detected", "2": "detected"}
    assert out[0].gap_fill_status == {"1": "detected", "2": "detected"}
    assert out[0].heights["2"] == 100.0
    assert out[0].n_detected == 2
    assert stats["conservation"]["n_unexplained_lost"] == 0
    assert stats["conservation"]["n_input_keyed_peaks"] == 3
    assert stats["conservation"]["n_assigned_peaks"] == 2
    assert stats["conservation"]["n_collapsed_same_compound"] == 1


def test_reconciliation_spotmap_records_every_source_peak(tmp_path, monkeypatch):
    features = {
        "1": [
            _peak("1", "keeper", 300.000, 5.000, height=9000.0),
            _peak("1", "loser", 300.005, 5.010, height=50.0),
        ],
        "2": [_peak("2", "detected", 300.005, 5.010, height=100.0)],
    }
    monkeypatch.setattr(
        stage7_alignment, "run_gap_fill",
        lambda spots, sample_ids, config, context, progress_callback=None: None,
    )

    run_stage7(features, ProcessingConfig(), gap_fill=_gap_context(tmp_path))

    spot_map = load_spot_map(tmp_path)
    spot_of = spot_map.spot_of
    assert set(spot_of) == {
        "rep1_keeper", "rep1_loser", "rep2_detected",
    }
    assert len(set(spot_of.values())) == 1
    assert spot_map.fold_reason_of == {
        "rep1_loser": "same_route_redundant_cell",
    }
    assert spot_map.fold_evidence_of["rep1_loser"]["n_matched_fragments"] == 3


def test_merge_provenance_is_saved_without_raw_gap_fill_context(tmp_path):
    features = {
        "1": [
            _peak("1", "keeper", 300.000, 5.000, height=9000.0),
            _peak("1", "loser", 300.005, 5.010, height=50.0),
        ],
        "2": [_peak("2", "detected", 300.005, 5.010, height=100.0)],
    }

    run_stage7(
        features, ProcessingConfig(), mapping_output_dir=str(tmp_path),
    )

    raw = json.loads((tmp_path / MERGE_MAP_NAME).read_text(encoding="utf-8"))
    assert raw["fold_reason_of"]["rep1_loser"] == "same_route_redundant_cell"
    assert raw["spot_of"]["rep1_loser"] == raw["spot_of"]["rep1_keeper"]
    assert not (tmp_path / "spotmap.json").exists()


def test_product_rows_from_different_windows_remain_visible_after_refine():
    features = {
        "1": [
            _peak(
                "1", "w300", 85.03, 5.0, height=9000.0,
                signal_type="ms2_only", window=300, spectrum=_SAME,
            ),
            _peak(
                "1", "w700", 85.03, 5.0, height=100.0,
                signal_type="ms2_only", window=700, spectrum=_DIFFERENT,
            ),
        ],
    }

    out = run_stage7(features, ProcessingConfig())

    assert len(out) == 2
    assert {feature.alignment_window for feature in out} == {300, 700}
    assert all(not feature.is_duplicate for feature in out)


def test_same_window_product_rows_with_reliable_different_ms2_stay_visible():
    p = [(100.0, 1000.0), (150.0, 1.0), (200.0, 1.0)]
    q = [(100.0, 1.0), (150.0, 1000.0), (200.0, 1.0)]
    features = {
        "1": [
            _peak(
                "1", "p", 85.030, 5.000, height=9000.0,
                signal_type="ms2_only", window=300, spectrum=p,
            ),
            _peak(
                "1", "q", 85.031, 5.001, height=100.0,
                signal_type="ms2_only", window=300, spectrum=q,
            ),
        ],
    }

    out = run_stage7(features, ProcessingConfig())

    assert len(out) == 2
    assert all(not feature.is_duplicate for feature in out)


def test_disjoint_ms1_spots_ignore_aif_identity_during_reconciliation():
    """Old checkpoints must heal MS1 rows split by unstable AIF spectra."""
    p = [(100.0, 1000.0), (150.0, 1.0), (200.0, 1.0)]
    q = [(100.0, 1.0), (150.0, 1000.0), (200.0, 1.0)]
    left = _peak("1", "left", 300.000, 5.000, spectrum=p)
    right = _peak("2", "right", 300.001, 5.001, spectrum=q)
    cfg = ProcessingConfig()

    spots, stats = reconcile_spots(
        [_spot(0, "1", left), _spot(1, "2", right)],
        cfg.joiner_view(),
        cfg.refiner_view(),
        is_annotated=lambda _peak: False,
    )

    assert len(spots) == 1
    assert set(spots[0].peaks) == {"1", "2"}
    assert all(
        peak.gap_fill_status == "detected" for peak in spots[0].peaks.values()
    )
    assert stats.n_detected_cells_transferred == 1


def test_overlapping_ms1_spots_merge_on_direct_natural_peak_evidence():
    """The two ASFAM detection routes may report the same MS1 peak twice."""
    p = [(100.0, 1000.0), (150.0, 1.0), (200.0, 1.0)]
    q = [(100.0, 1.0), (150.0, 1000.0), (200.0, 1.0)]
    ms2_driven = _peak(
        "1", "ms2_driven", 300.000, 5.000, height=1000.0, spectrum=p,
    )
    ms1_driven = _peak(
        "1", "ms1_driven", 300.001, 5.001, height=995.0, spectrum=q,
    )
    ms2_driven.detection_source = "ms2_driven"
    ms1_driven.detection_source = "ms1_driven"
    # One MS1 ion may legitimately be assigned to adjacent AIF windows.
    ms1_driven.precursor_mz_nominal = 301
    left = _spot(0, "1", ms2_driven)
    left.peaks["2"] = _peak("2", "left", 300.000, 5.000, spectrum=p)
    right = _spot(1, "1", ms1_driven)
    right.peaks["3"] = _peak("3", "right", 300.001, 5.001, spectrum=q)
    cfg = ProcessingConfig()

    spots, stats = reconcile_spots(
        [left, right],
        cfg.joiner_view(),
        cfg.refiner_view(),
        is_annotated=lambda _peak: False,
    )

    assert len(spots) == 1
    assert set(spots[0].peaks) == {"1", "2", "3"}
    assert len(spots[0].folded_peaks) == 1
    fold = spots[0].folded_peaks[0]
    assert fold.evidence_kind == "ms1_natural_peak"
    assert fold.evidence_details["mz_delta"] == pytest.approx(0.001)
    assert fold.evidence_details["rt_delta"] == pytest.approx(0.001)
    assert fold.evidence_details["peak_overlap_ratio"] >= 0.5
    assert fold.evidence_details["height_ratio"] == pytest.approx(0.995)
    assert fold.n_matched_fragments == 0
    assert stats.n_detected_conflicts_folded == 1
    audit = audit_spots(spots, cfg.joiner_view())
    assert audit.n_same_folds_below_min_matched == 0
    assert audit.n_fold_identity_not_same == 0
    assert audit.n_fold_evidence_mismatch == 0
    assert audit.n_ms1_natural_peak_evidence_mismatch == 0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda peak: setattr(peak, "detection_source", "ms2_driven"),
        lambda peak: setattr(peak, "segment_name", "other-segment"),
        lambda peak: setattr(peak, "source_file", "other-file.mzML"),
        lambda peak: setattr(peak, "ms1_quant_mz", 300.011),
        lambda peak: setattr(peak, "rt_left", 5.06),
        lambda peak: setattr(peak, "ms1_height", 700.0),
    ],
)
def test_overlapping_ms1_spots_reject_weak_natural_peak_evidence(mutate):
    left_peak = _peak("1", "left", 300.000, 5.000, height=1000.0, spectrum=[])
    right_peak = _peak("1", "right", 300.001, 5.001, height=995.0, spectrum=[])
    left_peak.detection_source = "ms2_driven"
    right_peak.detection_source = "ms1_driven"
    mutate(right_peak)
    cfg = ProcessingConfig()

    spots, _stats = reconcile_spots(
        [_spot(0, "1", left_peak), _spot(1, "1", right_peak)],
        cfg.joiner_view(),
        cfg.refiner_view(),
        is_annotated=lambda _peak: False,
    )

    assert len(spots) == 2


def test_ms1_driven_aif_same_cannot_replace_natural_peak_evidence():
    left_peak = _peak("1", "left", 300.000, 5.000, height=1000.0)
    right_peak = _peak("1", "right", 300.001, 5.001, height=995.0)
    left_peak.detection_source = "ms1_driven"
    right_peak.detection_source = "ms1_driven"
    cfg = ProcessingConfig()

    spots, _stats = reconcile_spots(
        [_spot(0, "1", left_peak), _spot(1, "1", right_peak)],
        cfg.joiner_view(),
        cfg.refiner_view(),
        is_annotated=lambda _peak: False,
    )

    assert len(spots) == 2


def test_same_source_exact_ms1_integration_is_direct_natural_peak_evidence():
    left_peak = _peak("1", "left", 300.000, 5.000, height=1000.0)
    right_peak = _peak(
        "1", "right", 300.008, 5.000, height=1000.0 + 5e-10,
    )
    for peak in (left_peak, right_peak):
        peak.detection_source = "ms1_driven"
        peak.source_file = "same-raw.mzML"
    # Overlapping mass slices can integrate different shoulders/areas while the
    # apex and exact peak boundaries identify the duplicated natural peak.
    left_peak.ms1_area = 1000.0
    right_peak.ms1_area = 750.0
    cfg = ProcessingConfig()

    spots, stats = reconcile_spots(
        [_spot(0, "1", left_peak), _spot(1, "1", right_peak)],
        cfg.joiner_view(),
        cfg.refiner_view(),
        is_annotated=lambda _peak: False,
    )

    assert len(spots) == 1
    assert len(spots[0].folded_peaks) == 1
    assert spots[0].folded_peaks[0].evidence_kind == "ms1_natural_peak"
    assert stats.n_detected_conflicts_folded == 1
    audit = audit_spots(spots, cfg.joiner_view())
    assert audit.n_ms1_natural_peak_evidence_mismatch == 0


def test_same_source_exact_ms1_evidence_does_not_bypass_spot_geometry():
    left_common = _peak("1", "left_common", 300.000, 5.000, height=1000.0)
    right_common = _peak("1", "right_common", 300.008, 5.000, height=1000.0)
    for peak in (left_common, right_common):
        peak.detection_source = "ms1_driven"
        peak.source_file = "same-raw.mzML"
    left = _spot(0, "1", left_common)
    left.peaks["2"] = _peak("2", "early", 300.000, 4.800)
    right = _spot(1, "1", right_common)
    right.peaks["3"] = _peak("3", "late", 300.008, 5.100)
    cfg = ProcessingConfig()

    spots, _stats = reconcile_spots(
        [left, right],
        cfg.joiner_view(),
        cfg.refiner_view(),
        is_annotated=lambda _peak: False,
    )

    assert len(spots) == 2


def test_prejoin_does_not_fold_same_source_exact_ms1_candidates():
    left = _peak("1", "left", 300.000, 5.000, height=1000.0)
    right = _peak("1", "right", 300.008, 5.000, height=1000.0)
    for peak in (left, right):
        peak.detection_source = "ms1_driven"
        peak.source_file = "same-raw.mzML"

    filtered, folds, stats = collapse_ms1_natural_peak_aliases({
        "1": [left, right],
    })

    assert filtered == {"1": [left, right]}
    assert folds == {}
    assert stats.n_ms1_aliases_folded == 0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda peak: setattr(peak, "source_file", ""),
        lambda peak: setattr(peak, "source_file", "other-raw.mzML"),
        lambda peak: setattr(peak, "segment_name", "other-segment"),
        lambda peak: setattr(peak, "replicate_id", 2),
        lambda peak: setattr(peak, "ms1_quant_mz", 300.011),
        lambda peak: setattr(peak, "rt_apex", 5.000001),
        lambda peak: setattr(peak, "rt_left", 4.950001),
        lambda peak: setattr(peak, "rt_right", 5.050001),
        lambda peak: setattr(peak, "ms1_height", 999.999),
        lambda peak: setattr(peak, "isotope_index", 1),
    ],
)
def test_same_source_ms1_evidence_requires_an_exact_fingerprint(mutate):
    left_peak = _peak("1", "left", 300.000, 5.000, height=1000.0)
    right_peak = _peak("1", "right", 300.008, 5.000, height=1000.0)
    for peak in (left_peak, right_peak):
        peak.detection_source = "ms1_driven"
        peak.source_file = "same-raw.mzML"
    mutate(right_peak)
    cfg = ProcessingConfig()

    spots, _stats = reconcile_spots(
        [_spot(0, "1", left_peak), _spot(1, "1", right_peak)],
        cfg.joiner_view(),
        cfg.refiner_view(),
        is_annotated=lambda _peak: False,
    )

    assert len(spots) == 2


def test_same_source_ms2_driven_candidates_do_not_use_exact_ms1_evidence():
    left_peak = _peak(
        "1", "left", 300.000, 5.000, height=1000.0, spectrum=[],
    )
    right_peak = _peak(
        "1", "right", 300.008, 5.000, height=1000.0, spectrum=[],
    )
    for peak in (left_peak, right_peak):
        peak.detection_source = "ms2_driven"
        peak.source_file = "same-raw.mzML"
    cfg = ProcessingConfig()

    spots, _stats = reconcile_spots(
        [_spot(0, "1", left_peak), _spot(1, "1", right_peak)],
        cfg.joiner_view(),
        cfg.refiner_view(),
        is_annotated=lambda _peak: False,
    )

    assert len(spots) == 2


def test_ms1_natural_fold_audit_rejects_fake_ms2_counts():
    keeper = _peak("1", "keeper", 300.000, 5.000, height=1000.0)
    alias = _peak("1", "alias", 300.001, 5.001, height=995.0)
    keeper.detection_source = "ms2_driven"
    alias.detection_source = "ms1_driven"
    spot = _spot(0, "1", keeper)
    spot.folded_peaks = [FoldedPeak(
        sample_id="1",
        peak=alias,
        reason="ms1_natural_peak_alias",
        target_peak_id=keeper.feature_id,
        cosine=0.0,
        n_matched_fragments=999,
        evidence_kind="ms1_natural_peak",
        evidence_details={
            "mz_delta": 0.001,
            "rt_delta": 0.001,
            "peak_overlap_ratio": 0.99,
            "height_ratio": 0.995,
        },
    )]
    recompute_alignment_center(spot)

    audit = audit_spots([spot], ProcessingConfig().joiner_view())

    assert audit.n_fold_evidence_mismatch == 1
    assert audit.n_ms1_natural_peak_evidence_mismatch == 1


def test_every_overlapping_sample_must_prove_the_same_ms1_peak():
    left_one = _peak("1", "left_one", 300.000, 5.000, height=1000.0, spectrum=[])
    right_one = _peak("1", "right_one", 300.001, 5.001, height=995.0, spectrum=[])
    left_one.detection_source = "ms2_driven"
    right_one.detection_source = "ms1_driven"
    left_two = _peak("2", "left_two", 300.000, 5.000, height=1000.0, spectrum=[])
    right_two = _peak("2", "right_two", 300.001, 5.001, height=995.0, spectrum=[])
    # This common sample has two candidates from the same path, so it cannot
    # certify that the spots are duplicate views of one integrated peak.
    left_two.detection_source = "ms2_driven"
    right_two.detection_source = "ms2_driven"
    left = _spot(0, "1", left_one)
    left.peaks["2"] = left_two
    right = _spot(1, "1", right_one)
    right.peaks["2"] = right_two
    cfg = ProcessingConfig()

    spots, _stats = reconcile_spots(
        [left, right],
        cfg.joiner_view(),
        cfg.refiner_view(),
        is_annotated=lambda _peak: False,
    )

    assert len(spots) == 2


def test_dual_route_ms1_alias_occupies_only_one_master_before_join():
    """One natural peak cannot satisfy two neighbouring rows by aliasing."""
    early_reference = _peak(
        "1", "early", 300.000, 5.000, height=1000.0, spectrum=[],
    )
    late_reference = _peak(
        "1", "late", 300.000, 5.150, height=1000.0, spectrum=[],
    )
    dual_ms2 = _peak(
        "2", "dual_ms2", 300.000, 5.020, height=1000.0, spectrum=[],
    )
    dual_ms1 = _peak(
        "2", "dual_ms1", 300.001, 5.021, height=995.0, spectrum=[],
    )
    dual_ms2.detection_source = "ms2_driven"
    dual_ms1.detection_source = "ms1_driven"
    cfg = ProcessingConfig()
    cfg.alignment_reference_sample = "1"
    stats = {}

    result = run_stage7(
        {
            "1": [early_reference, late_reference],
            "2": [dual_ms2, dual_ms1],
        },
        cfg,
        stats_out=stats,
    )

    assert len(result) == 2
    assert sum(
        feature.gap_fill_status.get("2") == "detected" for feature in result
    ) == 1
    assert stats["prejoin"]["n_ms1_aliases_folded"] == 1
    assert stats["conservation"] == {
        "n_input_keyed_peaks": 4,
        "n_assigned_peaks": 3,
        "n_collapsed_same_compound": 1,
        "n_unexplained_lost": 0,
    }


def test_prejoin_ms1_alias_collapse_is_invariant_to_input_order():
    def run(order):
        peaks = {
            "ms2": _peak("1", "ms2", 300.000, 5.000, height=1000.0),
            "ms1": _peak("1", "ms1", 300.001, 5.001, height=995.0),
        }
        peaks["ms2"].detection_source = "ms2_driven"
        peaks["ms1"].detection_source = "ms1_driven"
        filtered, folds, stats = collapse_ms1_natural_peak_aliases({
            "1": [peaks[name] for name in order],
        })
        return (
            [peak.feature_id for peak in filtered["1"]],
            sorted(
                fold.peak.feature_id
                for values in folds.values()
                for fold in values
            ),
            stats,
        )

    assert run(("ms2", "ms1")) == run(("ms1", "ms2"))


def test_prejoin_ms1_alias_matching_is_one_to_one_when_one_path_is_ambiguous():
    ms1 = _peak("1", "ms1", 300.000, 5.000, height=1000.0)
    closest = _peak("1", "closest", 300.001, 5.001, height=995.0)
    other = _peak("1", "other", 300.002, 5.020, height=990.0)
    ms1.detection_source = "ms1_driven"
    closest.detection_source = "ms2_driven"
    other.detection_source = "ms2_driven"

    filtered, folds, stats = collapse_ms1_natural_peak_aliases({
        "1": [other, ms1, closest],
    })
    folded_ids = {
        fold.peak.feature_id
        for values in folds.values()
        for fold in values
    }
    matched_ids = {
        feature_id
        for values in folds.values()
        for fold in values
        for feature_id in (fold.peak.feature_id, fold.target_peak_id)
    }

    assert stats.n_ms1_aliases_folded == 1
    assert matched_ids == {ms1.feature_id, closest.feature_id}
    assert len(folded_ids) == 1
    kept_ids = {peak.feature_id for peak in filtered["1"]}
    assert other.feature_id in kept_ids
    assert kept_ids | folded_ids == {
        ms1.feature_id, closest.feature_id, other.feature_id,
    }


def test_prejoin_alias_rejects_wide_overlapping_peaks_with_distant_apices():
    early = _peak("1", "early", 300.000, 5.000, height=1000.0)
    late = _peak("1", "late", 300.001, 5.150, height=995.0)
    early.detection_source = "ms2_driven"
    late.detection_source = "ms1_driven"
    early.rt_left = late.rt_left = 4.7
    early.rt_right = late.rt_right = 5.4

    filtered, folds, stats = collapse_ms1_natural_peak_aliases({
        "1": [early, late],
    })

    assert filtered == {"1": [early, late]}
    assert folds == {}
    assert stats.n_ms1_aliases_folded == 0


def test_feature_and_gap_target_share_the_stored_consensus_center():
    # Sample 2 has the MS2-rich representative.  Sample 1 is tallest, so its
    # quant ion supplies alignment_mz; the mean of both apices supplies RT.
    no_ms2 = []
    features = {
        "1": [
            _peak("1", "tall", 300.010, 5.180, height=9000.0, spectrum=no_ms2),
        ],
        "2": [
            _peak("2", "rep", 300.000, 5.000, height=100.0, spectrum=_SAME),
        ],
    }

    out = run_stage7(features, ProcessingConfig())

    assert len(out) == 1
    feature = out[0]
    assert feature.rt == pytest.approx(5.09)
    assert feature.align_mz == pytest.approx(300.010)
    assert feature.representative_rt == pytest.approx(5.000)


def test_gap_target_reads_the_same_stored_consensus_center():
    tall = _peak("1", "tall", 300.010, 5.180, height=9000.0, spectrum=[])
    representative = _peak("2", "rep", 300.000, 5.000, height=100.0)
    spot = _spot(0, "2", representative)
    spot.peaks["1"] = tall
    recompute_alignment_center(spot)

    target = gap_fill_target_for_spot(
        spot, ProcessingConfig().gap_fill_view(),
    )

    assert target is not None
    assert target.rt_center == pytest.approx(5.09)
    assert target.quant.mz == pytest.approx(300.010)
    assert target.segment_name == tall.segment_name


def test_gap_fill_cannot_reuse_a_natural_ms1_peak_from_another_spot():
    natural = _peak("2", "natural", 300.000, 5.000, height=1000.0)
    natural.rt_left = 4.95
    natural.rt_right = 5.05
    natural_spot = _spot(0, "2", natural)
    target_peak = _peak("1", "target", 300.001, 5.001, height=900.0)
    target_spot = _spot(1, "1", target_peak)
    recompute_alignment_center(natural_spot)
    recompute_alignment_center(target_spot)
    cfg = ProcessingConfig()
    target = gap_fill_target_for_spot(target_spot, cfg.gap_fill_view())
    assert target is not None
    index = _NaturalMs1PeakIndex(
        [natural_spot, target_spot],
        rt_tolerance=cfg.gap_fill_view().rt_tolerance,
    )

    reused = index.reused_peak(
        "2",
        target_spot,
        target,
        GapFillResult(
            status=FILLED,
            height=1000.0,
            rt_apex=5.0,
            rt_left=4.96,
            rt_right=5.04,
        ),
    )
    distinct = index.reused_peak(
        "2",
        target_spot,
        target,
        GapFillResult(
            status=FILLED,
            height=100.0,
            rt_apex=5.15,
            rt_left=5.10,
            rt_right=5.20,
        ),
    )

    assert reused is not None
    assert reused.peak is natural
    assert distinct is None


def test_gap_fill_blocks_same_raw_height_when_detection_apex_is_offset():
    natural = _peak("2", "natural", 300.000, 5.000, height=1000.0)
    natural.rt_left = 4.90
    natural.rt_right = 5.20
    natural_spot = _spot(0, "2", natural)
    target_peak = _peak("1", "target", 300.001, 5.080, height=900.0)
    target_spot = _spot(1, "1", target_peak)
    recompute_alignment_center(natural_spot)
    recompute_alignment_center(target_spot)
    cfg = ProcessingConfig()
    target = gap_fill_target_for_spot(target_spot, cfg.gap_fill_view())
    assert target is not None
    index = _NaturalMs1PeakIndex(
        [natural_spot, target_spot],
        rt_tolerance=cfg.gap_fill_view().rt_tolerance,
    )

    reused = index.reused_peak(
        "2",
        target_spot,
        target,
        GapFillResult(
            status=FILLED,
            height=1000.0,
            rt_apex=5.08,
            rt_left=5.04,
            rt_right=5.12,
        ),
    )
    different_height = index.reused_peak(
        "2",
        target_spot,
        target,
        GapFillResult(
            status=FILLED,
            height=800.0,
            rt_apex=5.08,
            rt_left=5.04,
            rt_right=5.12,
        ),
    )
    narrow_disjoint_bounds = index.reused_peak(
        "2",
        target_spot,
        target,
        GapFillResult(
            status=FILLED,
            height=1000.0,
            rt_apex=5.195,
            rt_left=5.19,
            rt_right=5.25,
        ),
    )

    assert reused is not None
    assert reused.peak is natural
    assert different_height is None
    assert narrow_disjoint_bounds is not None
    assert narrow_disjoint_bounds.peak is natural


def test_gap_fill_blocks_exact_raw_peak_when_both_fitted_apices_are_outside_bounds():
    """Rice F45391: two fit models can bracket the same raw EIC top differently."""
    natural = _peak("3", "rep3_21111", 940.493286, 9.054104, height=1148.9106464823406)
    natural.rt_left = 8.826733
    natural.rt_right = 9.109800
    natural_spot = _spot(0, "3", natural)
    target_peak = _peak("1", "target", 940.495175, 9.153217, height=900.0)
    target_spot = _spot(1, "1", target_peak)
    recompute_alignment_center(natural_spot)
    recompute_alignment_center(target_spot)
    cfg = ProcessingConfig()
    target = gap_fill_target_for_spot(target_spot, cfg.gap_fill_view())
    assert target is not None
    index = _NaturalMs1PeakIndex(
        [natural_spot, target_spot],
        rt_tolerance=cfg.gap_fill_view().rt_tolerance,
    )

    reused = index.reused_peak(
        "3",
        target_spot,
        target,
        GapFillResult(
            status=FILLED,
            height=1148.9106464823406,
            rt_apex=9.125517,
            rt_left=9.062633,
            rt_right=9.440250,
        ),
    )

    assert 9.125517 > natural.rt_right
    assert natural.rt_apex < 9.062633
    assert reused is not None
    assert reused.peak is natural


def test_gap_fill_keeps_strong_overlap_when_source_apex_is_just_outside_search():
    """Cancer+ F30717: the stored source apex is 0.209 min from consensus."""
    natural = _peak("9", "rep9_12230", 816.6966801822485, 9.336666666667,
                    height=11008.535979020753)
    natural.rt_left = 9.138533333333
    natural.rt_right = 9.462066666667
    natural_spot = _spot(0, "9", natural)
    target_peak = _peak("1", "target", 816.6966223516466, 9.127666666667,
                        height=900.0)
    target_spot = _spot(1, "1", target_peak)
    recompute_alignment_center(natural_spot)
    recompute_alignment_center(target_spot)
    cfg = ProcessingConfig()
    target = gap_fill_target_for_spot(target_spot, cfg.gap_fill_view())
    assert target is not None
    index = _NaturalMs1PeakIndex(
        [natural_spot, target_spot],
        rt_tolerance=cfg.gap_fill_view().rt_tolerance,
    )

    reused = index.reused_peak(
        "9",
        target_spot,
        target,
        GapFillResult(
            status=FILLED,
            height=11008.535979020915,
            rt_apex=9.2647,
            rt_left=9.06545,
            rt_right=9.31865,
        ),
    )

    assert abs(natural.rt_apex - target.rt_center) > cfg.alignment_rt_tolerance
    assert reused is not None
    assert reused.peak is natural


def test_gap_fill_rejects_weak_overlap_outside_the_source_rt_search_window():
    natural = _peak("2", "natural", 300.000, 5.210, height=1000.0)
    natural.rt_left = 5.04
    natural.rt_right = 5.26
    natural_spot = _spot(0, "2", natural)
    target_peak = _peak("1", "target", 300.001, 5.000, height=900.0)
    target_spot = _spot(1, "1", target_peak)
    recompute_alignment_center(natural_spot)
    recompute_alignment_center(target_spot)
    cfg = ProcessingConfig()
    target = gap_fill_target_for_spot(target_spot, cfg.gap_fill_view())
    assert target is not None
    index = _NaturalMs1PeakIndex(
        [natural_spot, target_spot],
        rt_tolerance=cfg.gap_fill_view().rt_tolerance,
    )

    distinct = index.reused_peak(
        "2",
        target_spot,
        target,
        GapFillResult(
            status=FILLED,
            height=1000.0,
            rt_apex=5.100,
            rt_left=4.95,
            rt_right=5.12,
        ),
    )

    assert 5.12 > natural.rt_left
    assert distinct is None


def test_gap_fill_exact_height_does_not_block_outside_the_rt_search_window():
    natural = _peak("2", "natural", 300.000, 5.000, height=1000.0)
    natural.rt_left = 4.95
    natural.rt_right = 5.05
    natural_spot = _spot(0, "2", natural)
    target_peak = _peak("1", "target", 300.001, 5.300, height=900.0)
    target_spot = _spot(1, "1", target_peak)
    recompute_alignment_center(natural_spot)
    recompute_alignment_center(target_spot)
    cfg = ProcessingConfig()
    target = gap_fill_target_for_spot(target_spot, cfg.gap_fill_view())
    assert target is not None
    index = _NaturalMs1PeakIndex(
        [natural_spot, target_spot],
        rt_tolerance=cfg.gap_fill_view().rt_tolerance,
    )

    distinct = index.reused_peak(
        "2",
        target_spot,
        target,
        GapFillResult(
            status=FILLED,
            height=1000.0,
            rt_apex=5.30,
            rt_left=5.25,
            rt_right=5.35,
        ),
    )

    assert distinct is None


def test_gap_fill_exact_height_does_not_merge_disjoint_peaks_in_one_search_window():
    natural = _peak("2", "natural", 300.000, 4.910, height=1000.0)
    natural.rt_left = 4.86
    natural.rt_right = 4.96
    natural_spot = _spot(0, "2", natural)
    target_peak = _peak("1", "target", 300.001, 5.000, height=900.0)
    target_spot = _spot(1, "1", target_peak)
    recompute_alignment_center(natural_spot)
    recompute_alignment_center(target_spot)
    cfg = ProcessingConfig()
    target = gap_fill_target_for_spot(target_spot, cfg.gap_fill_view())
    assert target is not None
    index = _NaturalMs1PeakIndex(
        [natural_spot, target_spot],
        rt_tolerance=cfg.gap_fill_view().rt_tolerance,
    )

    distinct = index.reused_peak(
        "2",
        target_spot,
        target,
        GapFillResult(
            status=FILLED,
            height=1000.0,
            rt_apex=5.090,
            rt_left=5.04,
            rt_right=5.14,
        ),
    )

    assert abs(natural.rt_apex - target.rt_center) < cfg.alignment_rt_tolerance
    assert abs(5.090 - target.rt_center) < cfg.alignment_rt_tolerance
    assert distinct is None


def test_gap_fill_exact_height_does_not_join_opposite_ends_of_rt_search_window():
    natural = _peak("2", "natural", 300.000, 4.810, height=1000.0)
    natural.rt_left = 4.75
    natural.rt_right = 5.15
    natural_spot = _spot(0, "2", natural)
    target_peak = _peak("1", "target", 300.001, 5.000, height=900.0)
    target_spot = _spot(1, "1", target_peak)
    recompute_alignment_center(natural_spot)
    recompute_alignment_center(target_spot)
    cfg = ProcessingConfig()
    target = gap_fill_target_for_spot(target_spot, cfg.gap_fill_view())
    assert target is not None
    index = _NaturalMs1PeakIndex(
        [natural_spot, target_spot],
        rt_tolerance=cfg.gap_fill_view().rt_tolerance,
    )

    distinct = index.reused_peak(
        "2",
        target_spot,
        target,
        GapFillResult(
            status=FILLED,
            height=1000.0,
            rt_apex=5.180,
            rt_left=4.85,
            rt_right=5.25,
        ),
    )

    assert 4.85 < natural.rt_right  # intervals overlap substantially
    assert distinct is None


def test_gap_fill_checks_farther_exact_candidate_after_nearest_one_differs():
    near = _peak("2", "near", 300.0005, 4.850, height=800.0)
    near.rt_left = 4.80
    near.rt_right = 4.90
    exact = _peak("2", "exact", 300.0020, 5.020, height=1000.0)
    exact.rt_left = 4.97
    exact.rt_right = 5.07
    near_spot = _spot(0, "2", near)
    exact_spot = _spot(1, "2", exact)
    target_peak = _peak("1", "target", 300.000, 5.010, height=900.0)
    target_spot = _spot(2, "1", target_peak)
    for spot in (near_spot, exact_spot, target_spot):
        recompute_alignment_center(spot)
    cfg = ProcessingConfig()
    target = gap_fill_target_for_spot(target_spot, cfg.gap_fill_view())
    assert target is not None
    index = _NaturalMs1PeakIndex(
        [near_spot, exact_spot, target_spot],
        rt_tolerance=cfg.gap_fill_view().rt_tolerance,
    )

    reused = index.reused_peak(
        "2",
        target_spot,
        target,
        GapFillResult(
            status=FILLED,
            height=1000.0,
            rt_apex=5.020,
            rt_left=4.98,
            rt_right=5.06,
        ),
    )

    assert reused is not None
    assert reused.peak is exact


def test_gap_fill_index_never_blocks_from_the_target_spot_itself():
    natural = _peak("2", "natural", 300.000, 5.000, height=1000.0)
    spot = _spot(0, "2", natural)
    recompute_alignment_center(spot)
    cfg = ProcessingConfig()
    target = gap_fill_target_for_spot(spot, cfg.gap_fill_view())
    assert target is not None
    index = _NaturalMs1PeakIndex(
        [spot],
        rt_tolerance=cfg.gap_fill_view().rt_tolerance,
    )

    assert index.reused_peak(
        "2",
        spot,
        target,
        GapFillResult(
            status=FILLED,
            height=1000.0,
            rt_apex=5.0,
            rt_left=4.96,
            rt_right=5.04,
        ),
    ) is None


def test_gap_fill_blocker_changes_the_cell_and_records_auditable_details(monkeypatch):
    natural = _peak("2", "natural", 300.000, 4.940, height=1000.0)
    natural.rt_left = 4.85
    natural.rt_right = 5.01
    target_peak = _peak("1", "target", 300.001, 5.001, height=900.0)
    natural_spot = _spot(0, "2", natural)
    target_spot = _spot(1, "1", target_peak)
    for spot in (natural_spot, target_spot):
        recompute_alignment_center(spot)
    cfg = ProcessingConfig()
    targets = [
        gap_fill_target_for_spot(spot, cfg.gap_fill_view())
        for spot in (natural_spot, target_spot)
    ]

    class Source:
        @staticmethod
        def chromatogram(_target):
            return np.asarray([4.95, 5.0, 5.05]), np.asarray([1.0, 10.0, 1.0])

        @staticmethod
        def segment_for(_target):
            return None

    monkeypatch.setattr(
        stage7_alignment,
        "fill_from_chromatogram",
        lambda *_args: GapFillResult(
            status=FILLED,
            height=1000.0,
            rt_apex=5.0,
            rt_left=4.98,
            rt_right=5.05,
        ),
    )
    counts = Counter()
    blocks = []

    list(_one_sample(
        [natural_spot, target_spot],
        targets,
        [0, 1],
        "2",
        Source(),
        cfg,
        cfg.gap_fill_view(),
        _NaturalMs1PeakIndex(
            [natural_spot, target_spot],
            rt_tolerance=cfg.gap_fill_view().rt_tolerance,
        ),
        counts,
        blocks,
    ))

    assert target_spot.peaks["2"].gap_fill_status == "no_signal"
    assert counts["blocked_natural_peak_reuse"] == 1
    assert counts["no_signal"] == 1
    assert blocks[0].target_spot_id == "F00001"
    assert blocks[0].source_spot_id == "F00000"
    assert blocks[0].source_peak_id == natural.feature_id
    assert blocks[0].fill_height == natural.ms1_height
    assert not (4.98 <= natural.rt_apex <= 5.05)


def test_gap_fill_blocker_summary_is_preserved_in_stage7_stats(tmp_path, monkeypatch):
    marker = {
        "blocked_natural_peak_reuse": 1,
        "blocked_natural_peak_reuse_sha256": "abc",
        "blocked_natural_peak_reuse_examples": [{"target_spot_id": "F00001"}],
    }
    monkeypatch.setattr(
        stage7_alignment,
        "run_gap_fill",
        lambda spots, sample_ids, config, context, progress_callback=None: marker,
    )
    stats = {}

    run_stage7(
        {"1": [_peak("1", "target", 300.0, 5.0)]},
        ProcessingConfig(),
        gap_fill=GapFillContext(
            sample_files={"1": ["one.mzML"]}, output_dir=str(tmp_path),
        ),
        stats_out=stats,
    )

    assert stats["gap_fill"] == marker


def test_candidate_annotation_predicate_reads_selected_match_safely():
    peak = _peak("1", "annotated", 300.0, 5.0)
    peak.annotation_matches = [AnnotationMatch(
        rank=1, name="known", score=0.9, total_score=0.9, n_matched=4,
    )]
    confidence = ProcessingConfig().confidence_view()

    assert _candidate_is_high_confidence(peak, confidence)
    peak.selected_annotation_idx = 9
    assert not _candidate_is_high_confidence(peak, confidence)


def test_merge_reselects_representative_from_the_combined_candidate_pool():
    keeper = _peak("1", "keeper", 300.000, 5.000, height=100.0)
    keeper.annotation_matches = [AnnotationMatch(
        rank=1, name="keeper", score=0.8, total_score=0.4, n_matched=4,
    )]
    stronger = _peak("2", "stronger", 300.001, 5.001, height=9000.0)
    # Not high-confidence (one matched reference peak), so the annotated keeper
    # establishes the cluster; its higher total score must still win the
    # representative rule after the cells are combined.
    stronger.annotation_matches = [AnnotationMatch(
        rank=1, name="suggestion", score=0.9, total_score=0.9, n_matched=1,
    )]
    cfg = ProcessingConfig()

    spots, _ = reconcile_spots(
        [_spot(0, "1", keeper), _spot(1, "2", stronger)],
        cfg.joiner_view(),
        cfg.refiner_view(),
        is_annotated=lambda peak: peak is keeper,
    )

    assert len(spots) == 1
    assert spots[0].representative_sample_id == "2"


def test_hidden_peak_cannot_become_reconciled_representative():
    visible = _peak("1", "visible", 300.000, 5.000, height=100.0)
    hidden = _peak("2", "hidden", 300.001, 5.001, height=9000.0)
    hidden.is_duplicate = True
    hidden.duplicate_type = ""
    cfg = ProcessingConfig()

    spots, _ = reconcile_spots(
        [_spot(0, "1", visible), _spot(1, "2", hidden)],
        cfg.joiner_view(),
        cfg.refiner_view(),
        is_annotated=lambda _peak: False,
    )

    assert len(spots) == 1
    assert spots[0].representative is visible


def test_singleton_spot_reselects_a_visible_representative():
    hidden = _peak("1", "hidden", 300.000, 5.000, height=9000.0)
    hidden.is_duplicate = True
    hidden.duplicate_type = "spectral"
    visible = _peak("2", "visible", 300.001, 5.001, height=100.0)

    out = run_stage7({"1": [hidden], "2": [visible]}, ProcessingConfig())

    assert len(out) == 1
    assert not out[0].is_duplicate
    assert out[0].duplicate_type == ""


def test_visible_same_fold_replaces_a_hidden_joiner_cell():
    reference = _peak("1", "reference", 300.000, 5.000, height=9000.0)
    reference.is_duplicate = True
    reference.duplicate_type = "spectral"
    hidden = _peak("2", "a_hidden", 300.001, 5.001, height=8000.0)
    hidden.is_duplicate = True
    hidden.duplicate_type = "spectral"
    visible = _peak("2", "z_visible", 300.002, 5.002, height=100.0)
    cfg = ProcessingConfig()
    cfg.alignment_reference_sample = "1"
    stats = {}

    out = run_stage7(
        {"1": [reference], "2": [hidden, visible]}, cfg, stats_out=stats,
    )

    assert len(out) == 1
    assert not out[0].is_duplicate
    assert out[0].heights["2"] == visible.ms1_height
    assert stats["join"]["n_collapsed_same_compound"] == 1
    assert stats["spots"]["n_fold_identity_not_same"] == 0
    assert stats["spots"]["n_fold_evidence_mismatch"] == 0


def test_protected_isotope_singleton_promotes_its_visible_same_fold():
    hidden = _peak("1", "hidden_isotope", 815.206, 4.699, height=100.0)
    hidden.is_duplicate = True
    hidden.duplicate_type = "isotope"
    hidden.isotope_index = 1
    visible = _peak("1", "visible_mono", 815.207, 4.700, height=9000.0)
    spot = _spot(0, "1", hidden)
    spot.folded_peaks = [FoldedPeak(
        sample_id="1",
        peak=visible,
        reason="same_compound_claim_loser",
        target_peak_id=hidden.feature_id,
        cosine=1.0,
        n_matched_fragments=3,
    )]
    cfg = ProcessingConfig()

    spots, _ = reconcile_spots(
        [spot],
        cfg.joiner_view(),
        cfg.refiner_view(),
        is_annotated=lambda _peak: False,
    )

    assert len(spots) == 1
    assert spots[0].representative is visible
    assert spots[0].peaks["1"] is visible
    assert len(spots[0].folded_peaks) == 1
    reverse_fold = spots[0].folded_peaks[0]
    assert reverse_fold.peak is hidden
    assert reverse_fold.target_peak_id == visible.feature_id
    assert reverse_fold.n_matched_fragments == 3


def test_unidentified_isotope_satellite_cannot_claim_a_keeper_cluster():
    satellite = _peak("1", "satellite", 300.000, 5.000, height=9000.0)
    satellite.isotope_index = 1
    visible = _peak("2", "visible", 300.001, 5.001, height=100.0)
    cfg = ProcessingConfig()

    spots, _ = reconcile_spots(
        [_spot(0, "1", satellite), _spot(1, "2", visible)],
        cfg.joiner_view(),
        cfg.refiner_view(),
        is_annotated=lambda _peak: False,
    )

    assert len(spots) == 2


def test_disjoint_unidentified_isotopes_with_the_same_role_complete_one_row():
    left = _peak("1", "left_i1", 300.000, 5.000, height=9000.0)
    right = _peak("2", "right_i1", 300.001, 5.001, height=100.0)
    left.isotope_index = right.isotope_index = 1
    cfg = ProcessingConfig()

    spots, stats = reconcile_spots(
        [_spot(0, "1", left), _spot(1, "2", right)],
        cfg.joiner_view(),
        cfg.refiner_view(),
        is_annotated=lambda _peak: False,
    )

    assert len(spots) == 1
    assert set(spots[0].peaks) == {"1", "2"}
    assert all(
        peak.gap_fill_status == "detected" for peak in spots[0].peaks.values()
    )
    assert stats.n_disjoint_ms1_spots_merged == 1


def test_disjoint_completion_preserves_different_isotope_roles():
    mono = _peak("1", "mono", 300.000, 5.000, height=9000.0)
    isotope_one = _peak("2", "i1", 300.001, 5.001, height=100.0)
    isotope_two = _peak("3", "i2", 300.002, 5.002, height=90.0)
    isotope_one.isotope_index = 1
    isotope_two.isotope_index = 2
    cfg = ProcessingConfig()

    spots, stats = reconcile_spots(
        [
            _spot(0, "1", mono),
            _spot(1, "2", isotope_one),
            _spot(2, "3", isotope_two),
        ],
        cfg.joiner_view(),
        cfg.refiner_view(),
        is_annotated=lambda _peak: False,
    )

    assert len(spots) == 3
    assert stats.n_disjoint_ms1_spots_merged == 0


def test_disjoint_completion_rejects_a_same_role_sample_overlap():
    left = _peak("1", "left", 300.000, 5.000, height=9000.0)
    right = _peak("1", "right", 300.001, 5.001, height=100.0)
    left.isotope_index = right.isotope_index = 1
    cfg = ProcessingConfig()

    spots, stats = reconcile_spots(
        [_spot(0, "1", left), _spot(1, "1", right)],
        cfg.joiner_view(),
        cfg.refiner_view(),
        is_annotated=lambda _peak: False,
    )

    assert len(spots) == 2
    assert stats.n_disjoint_ms1_spots_merged == 0


def test_disjoint_completion_uses_final_centres_without_iterating():
    # First pass merges A+B because their original centres are within 0.05 min.
    # C cannot join that complete-link cluster because it is too far from A.
    # Once A+B has its final 5.0245 consensus, one completion pass can place C.
    a = _peak("1", "a", 300.000, 5.000, height=3000.0)
    b = _peak("2", "b", 300.001, 5.049, height=2000.0)
    c = _peak("3", "c", 300.002, 5.074, height=1000.0)
    cfg = ProcessingConfig()

    spots, stats = reconcile_spots(
        [_spot(0, "1", a), _spot(1, "2", b), _spot(2, "3", c)],
        cfg.joiner_view(),
        cfg.refiner_view(),
        is_annotated=lambda _peak: False,
    )

    assert len(spots) == 1
    assert set(spots[0].peaks) == {"1", "2", "3"}
    assert spots[0].alignment_rt == pytest.approx((5.000 + 5.049 + 5.074) / 3)
    assert stats.n_merged_spots == 2
    assert stats.n_disjoint_ms1_spots_merged == 1


def test_disjoint_completion_does_not_chain_through_its_new_centroid():
    a = _peak("1", "a", 300.000, 5.000, height=3000.0)
    b = _peak("2", "b", 300.001, 5.049, height=2000.0)
    c = _peak("3", "c", 300.002, 5.074, height=1000.0)
    for peak in (a, b, c):
        # Isotope protection leaves all three as immutable first-pass members.
        peak.isotope_index = 1
    cfg = ProcessingConfig()

    spots, stats = reconcile_spots(
        [_spot(0, "1", a), _spot(1, "2", b), _spot(2, "3", c)],
        cfg.joiner_view(),
        cfg.refiner_view(),
        is_annotated=lambda _peak: False,
    )

    assert len(spots) == 2
    assert sorted(len(spot.peaks) for spot in spots) == [1, 2]
    assert stats.n_disjoint_ms1_spots_merged == 1


def test_disjoint_completion_rejects_an_internally_mixed_isotope_role():
    isotope = _peak("1", "isotope", 300.000, 5.000, height=9000.0)
    isotope.isotope_index = 1
    isotope.detection_source = "ms2_driven"
    mono_fold = _peak("1", "mono", 300.001, 5.001, height=8995.0)
    mono_fold.detection_source = "ms1_driven"
    mixed = _spot(0, "1", isotope)
    mixed.folded_peaks = [FoldedPeak(
        sample_id="1",
        peak=mono_fold,
        reason="same_compound_claim_loser",
        target_peak_id=isotope.feature_id,
        cosine=1.0,
        n_matched_fragments=3,
    )]
    other = _peak("2", "other_i1", 300.002, 5.002, height=90.0)
    other.isotope_index = 1
    cfg = ProcessingConfig()

    spots, stats = reconcile_spots(
        [mixed, _spot(1, "2", other)],
        cfg.joiner_view(),
        cfg.refiner_view(),
        is_annotated=lambda _peak: False,
    )

    assert len(spots) == 2
    assert stats.n_disjoint_ms1_spots_merged == 0


def test_first_pass_rejects_mixed_isotope_role_with_mono_representative():
    mono = _peak("1", "mono", 300.000, 5.000, height=9000.0)
    mono.detection_source = "ms2_driven"
    isotope_fold = _peak(
        "1", "isotope_fold", 300.001, 5.001, height=8995.0,
    )
    isotope_fold.detection_source = "ms1_driven"
    isotope_fold.isotope_index = 1
    mixed = _spot(0, "1", mono)
    mixed.folded_peaks = [FoldedPeak(
        sample_id="1",
        peak=isotope_fold,
        reason="same_compound_claim_loser",
        target_peak_id=mono.feature_id,
        cosine=1.0,
        n_matched_fragments=3,
    )]
    other_mono = _peak("2", "other_mono", 300.002, 5.002, height=90.0)
    cfg = ProcessingConfig()

    spots, stats = reconcile_spots(
        [mixed, _spot(1, "2", other_mono)],
        cfg.joiner_view(),
        cfg.refiner_view(),
        is_annotated=lambda _peak: False,
    )

    assert len(spots) == 2
    assert stats.n_merged_spots == 0
    assert stats.n_disjoint_ms1_spots_merged == 0


def test_disjoint_completion_does_not_touch_product_rows():
    left = _peak(
        "1", "left", 85.030, 5.000, signal_type="ms2_only", spectrum=_SAME,
    )
    right = _peak(
        "2", "right", 85.031, 5.001, signal_type="ms2_only", spectrum=_SAME,
    )
    left.isotope_index = right.isotope_index = 1
    cfg = ProcessingConfig()

    spots, stats = reconcile_spots(
        [_spot(0, "1", left), _spot(1, "2", right)],
        cfg.joiner_view(),
        cfg.refiner_view(),
        is_annotated=lambda _peak: False,
    )

    # Isotope protection keeps these otherwise SAME rows out of the ordinary
    # pass; the disjoint completion must still remain strictly MS1-only.
    assert len(spots) == 2
    assert stats.n_disjoint_ms1_spots_merged == 0


def test_spot_merge_rejects_a_nontransitive_fold_chain():
    p_spec = [(100.0, 1000.0), (150.0, 1.0), (200.0, 1.0)]
    q_spec = [(100.0, 1.0), (150.0, 1000.0), (200.0, 1.0)]
    m_spec = [(100.0, 1001.0), (150.0, 1001.0), (200.0, 2.0)]
    central = _peak("1", "M", 300.000, 5.000, spectrum=m_spec)
    central.is_duplicate = True
    central.duplicate_type = "spectral"
    folded_p = _peak("1", "P", 300.001, 5.001, spectrum=p_spec)
    visible_q = _peak("1", "Q", 300.002, 5.002, spectrum=q_spec)
    left = _spot(0, "1", central)
    left.folded_peaks = [FoldedPeak(
        sample_id="1",
        peak=folded_p,
        reason="same_compound_claim_loser",
        target_peak_id=central.feature_id,
        cosine=0.7,
        n_matched_fragments=3,
    )]
    right = _spot(1, "1", visible_q)
    cfg = ProcessingConfig()

    spots, stats = reconcile_spots(
        [left, right],
        cfg.joiner_view(),
        cfg.refiner_view(),
        is_annotated=lambda _peak: False,
    )

    assert len(spots) == 2
    assert stats.n_identity_different_rejected > 0


def test_sparse_reconciliation_index_does_not_scan_the_full_table():
    cfg = ProcessingConfig()
    spots = [
        _spot(i, str(i + 1), _peak(str(i + 1), "p", 100.0 + i * 2, 5.0))
        for i in range(200)
    ]

    reconciled, stats = reconcile_spots(
        spots,
        cfg.joiner_view(),
        cfg.refiner_view(),
        is_annotated=lambda _peak: False,
    )

    assert len(reconciled) == 200
    assert stats.n_candidate_clusters_examined < 20


def test_stage7_output_is_invariant_to_candidate_permutation():
    p = [(100.0, 1000.0), (150.0, 1.0), (200.0, 1.0)]
    q = [(100.0, 1.0), (150.0, 1000.0), (200.0, 1.0)]
    m = [(100.0, 1001.0), (150.0, 1001.0), (200.0, 2.0)]

    def run(order):
        stats = {}
        peaks = {
            "P": _peak("2", "P", 300.001, 5.001, spectrum=p),
            "Q": _peak("2", "Q", 300.002, 5.002, spectrum=q),
        }
        result = run_stage7(
            {
                "1": [_peak("1", "M", 300.000, 5.000, spectrum=m)],
                "2": [peaks[name] for name in order],
            },
            ProcessingConfig(),
            stats_out=stats,
        )
        canonical = [
            (
                feature.feature_id,
                round(feature.rt, 6),
                round(feature.align_mz or 0.0, 6),
                tuple(sorted(feature.gap_fill_status.items())),
                tuple(sorted(feature.heights.items())),
                feature.is_duplicate,
                feature.duplicate_type,
            )
            for feature in result
        ]
        return canonical, stats["conservation"]

    assert run(("P", "Q")) == run(("Q", "P"))
