"""PR-B: features.csv exposes the MS-DIAL score breakdown columns."""
import numpy as np
import pandas as pd

from asfam.pipeline.stage8_export import _export_csv
from asfam.config import ProcessingConfig
from asfam.models import Feature, AnnotationMatch
from metabo_core.annotation import is_high_confidence


def _annotated_feature() -> Feature:
    ann = AnnotationMatch(
        rank=1, name="TestHit", formula="C6H12O6", score=1.85,
        n_matched=4, wdp=0.81, sdp=0.76, rdp=0.88,
        matched_pct=0.95, total_score=1.85,
    )
    return Feature(
        feature_id="F0001", precursor_mz=181.0707, rt=5.0,
        rt_left=4.9, rt_right=5.1, signal_type="ms1_detected",
        ms2_mz=np.array([100.0, 150.0]),
        ms2_intensity=np.array([1000.0, 500.0]),
        n_fragments=2, name="TestHit", formula="C6H12O6",
        annotation_matches=[ann], selected_annotation_idx=0,
    )


def test_export_csv_includes_msdial_breakdown_columns(tmp_path):
    out = tmp_path / "features.csv"
    _export_csv([_annotated_feature()], out, ProcessingConfig())
    df = pd.read_csv(out, comment="#")
    for col in ("wdp_score", "sdp_score", "rdp_score",
                "matched_pct", "total_score", "composite_score"):
        assert col in df.columns, f"missing column {col}"
    row = df.iloc[0]
    assert row["wdp_score"] == 0.81
    assert row["sdp_score"] == 0.76
    assert row["rdp_score"] == 0.88
    assert row["matched_pct"] == 0.95
    assert row["total_score"] == 1.85
    assert row["composite_score"] == 1.85   # composite_score == total_score


def test_export_csv_includes_alignment_center_and_relation_columns(tmp_path):
    feature = _annotated_feature()
    feature.align_mz = 181.0712
    feature.representative_rt = 4.98
    feature.alignment_window = 181
    feature.alignment_segment = "180-182"
    feature.alignment_relation = "ms1_covered_partial"
    feature.alignment_related_feature_id = "F0002"
    out = tmp_path / "features.csv"

    _export_csv([feature], out, ProcessingConfig())
    row = pd.read_csv(out, comment="#").iloc[0]

    assert row["alignment_mz"] == 181.0712
    assert row["representative_rt"] == 4.98
    assert row["alignment_window"] == 181
    assert row["alignment_segment"] == "180-182"
    assert row["alignment_relation"] == "ms1_covered_partial"
    assert row["alignment_related_feature_id"] == "F0002"


def test_export_csv_breakdown_blank_when_unannotated(tmp_path):
    feat = _annotated_feature()
    feat.annotation_matches = []   # no selected annotation
    out = tmp_path / "features.csv"
    _export_csv([feat], out, ProcessingConfig())
    df = pd.read_csv(out, comment="#")
    # 未注释行：分项列为空（NaN after pandas parse），但列仍存在
    for col in ("sdp_score", "matched_pct", "total_score"):
        assert col in df.columns
        assert pd.isna(df.iloc[0][col])


# --- PR-D Task D2: isotope_index / isotope_group_id export columns ---------


def _iso_feature(feature_id: str, mz: float, isotope_index: int,
                 isotope_group_id, *, is_duplicate: bool = False,
                 duplicate_type: str = "", adduct=None) -> Feature:
    """Minimal Feature carrying the PR-D isotope / duplicate labels."""
    return Feature(
        feature_id=feature_id, precursor_mz=mz, rt=5.0,
        rt_left=4.9, rt_right=5.1, signal_type="ms1_detected",
        ms2_mz=np.array([100.0, 150.0]),
        ms2_intensity=np.array([1000.0, 500.0]),
        n_fragments=2,
        isotope_index=isotope_index,
        isotope_group_id=isotope_group_id,
        is_duplicate=is_duplicate,
        duplicate_type=duplicate_type,
        adduct=adduct,
    )


def test_export_csv_counts_isotope_and_adduct_copies(tmp_path):
    """MS-DIAL convention: an isotope cluster + an adduct copy export as
    separate rows (copies are counted, not filtered), and the new
    ``isotope_index`` / ``isotope_group_id`` columns are present and
    round-trip."""
    feats = [
        # monoisotope (representative, index 0)
        _iso_feature("F0001", 285.0500, isotope_index=0, isotope_group_id=3),
        # M+1 / M+2 — marked duplicate (isotope), kept in export
        _iso_feature("F0002", 286.0533, isotope_index=1, isotope_group_id=3,
                     is_duplicate=True, duplicate_type="isotope"),
        _iso_feature("F0003", 287.0566, isotope_index=2, isotope_group_id=3,
                     is_duplicate=True, duplicate_type="isotope"),
        # adduct copy of a different ion — counted too
        _iso_feature("F0004", 308.0392, isotope_index=0, isotope_group_id=None,
                     is_duplicate=True, duplicate_type="adduct",
                     adduct="[M+Na]+"),
    ]
    out = tmp_path / "features.csv"
    _export_csv(feats, out, ProcessingConfig())
    df = pd.read_csv(out, comment="#")

    # All 4 copies counted — nothing filtered.
    assert len(df) == 4

    for col in ("isotope_index", "isotope_group_id", "duplicate_type"):
        assert col in df.columns, f"missing column {col}"

    # isotope_index round-trips 0 / 1 / 2 for the cluster members.
    cluster = df[df["feature_id"].isin(["F0001", "F0002", "F0003"])]
    assert sorted(int(v) for v in cluster["isotope_index"]) == [0, 1, 2]

    # duplicate_type carries both labels across the table.
    assert set(df["duplicate_type"].dropna()) >= {"isotope", "adduct"}


# --- 2026-07-05: two-tier annotated (suggested vs high-confidence) ----------


def test_export_csv_sparse_hit_is_suggested_not_annotated(tmp_path):
    """A sparse hit (n_matched < matchms_min_matched_peaks) keeps its
    name / score / n_matched cells but comes out annotated=False — the MS-DIAL
    "suggested" tier. A hit with enough matched peaks stays annotated=True."""
    suggested = AnnotationMatch(
        rank=1, name="SparseHit", formula="C6H12O6", score=1.85,
        n_matched=1, wdp=0.65, sdp=0.40, rdp=0.90,
        matched_pct=1.0, total_score=1.85,
    )
    feat = Feature(
        feature_id="F0009", precursor_mz=181.0707, rt=5.0,
        rt_left=4.9, rt_right=5.1, signal_type="ms2_driven",
        ms2_mz=np.array([100.0, 150.0]),
        ms2_intensity=np.array([1000.0, 500.0]),
        n_fragments=2, name="SparseHit", formula="C6H12O6",
        annotation_matches=[suggested], selected_annotation_idx=0,
    )
    out = tmp_path / "sparse.csv"
    _export_csv([feat], out, ProcessingConfig())   # matchms_min_matched_peaks=3
    row = pd.read_csv(out, comment="#").iloc[0]
    assert row["annotated"] == False               # suggestion, not high-confidence
    assert row["name"] == "SparseHit"              # but still named + scored
    assert row["total_score"] == 1.85
    assert int(row["n_matched"]) == 1

    # A hit with n_matched=4 clears the high-confidence tier.
    out2 = tmp_path / "hi.csv"
    _export_csv([_annotated_feature()], out2, ProcessingConfig())
    hi = pd.read_csv(out2, comment="#").iloc[0]
    assert hi["annotated"] == True
    assert int(hi["n_matched"]) == 4


# --- PR-6: detection counts + the shared high-confidence predicate ----------


def test_export_csv_has_detection_columns(tmp_path):
    """``height_rep*`` is non-empty for every cell after gap filling, so the only
    way to tell a detection from a fill is ``n_detected`` / ``detection_rate``."""
    feat = _annotated_feature()
    feat.heights = {"1": 5000.0, "2": 4000.0, "3": 0.0}
    feat.gap_fill_status = {"1": "detected", "2": "filled", "3": "no_signal"}
    feat.n_detected = 1
    feat.detection_rate = 1 / 3

    out = tmp_path / "features.csv"
    _export_csv([feat], out, ProcessingConfig())
    df = pd.read_csv(out, comment="#")

    assert df.iloc[0]["n_detected"] == 1
    assert df.iloc[0]["detection_rate"] == 0.333
    # The trap: two of three cells carry a positive height, and only one is a
    # detection.
    assert sum(df.iloc[0][f"height_rep{r}"] > 0 for r in ("1", "2")) == 2


# --- PR-7: the quantitation matrix has no holes, and says what each cell is --


def test_export_csv_carries_per_sample_gap_fill_status(tmp_path):
    """``gap_fill_status`` is a ``sample_id -> str`` map, so it exports one
    column per sample, alongside ``height_rep{i}`` / ``area_rep{i}``. Without it
    a reader cannot tell a detection from a fill: after gap filling every
    height cell is non-empty."""
    feat = _annotated_feature()
    feat.heights = {"1": 5000.0, "2": 4000.0, "3": 0.0}
    feat.areas = {"1": 500.0, "2": 400.0, "3": 0.0}
    feat.gap_fill_status = {"1": "detected", "2": "filled", "3": "no_signal"}

    out = tmp_path / "features.csv"
    _export_csv([feat], out, ProcessingConfig())
    df = pd.read_csv(out, comment="#")

    row = df.iloc[0]
    assert [row[f"gap_fill_status_rep{r}"] for r in ("1", "2", "3")] == [
        "detected", "filled", "no_signal",
    ]
    # The no_signal cell exports 0, not a blank: a blank is indistinguishable
    # from "sample missing" and makes pandas read the whole column as object.
    assert row["height_rep3"] == 0.0
    assert row["area_rep3"] == 0.0


def test_export_csv_fills_a_missing_sample_with_zero_not_blank(tmp_path):
    """A sample absent from ``heights`` (gap filling off, or the spot never saw
    it) exports 0 with a blank status — the blank status is what marks the 0 as
    a placeholder rather than a measured zero."""
    present = _annotated_feature()
    present.heights = {"1": 5000.0}
    present.areas = {"1": 500.0}
    present.gap_fill_status = {"1": "detected"}

    other = _annotated_feature()
    other.feature_id = "F0002"
    other.heights = {"2": 3000.0}
    other.areas = {"2": 300.0}
    other.gap_fill_status = {"2": "detected"}

    out = tmp_path / "features.csv"
    _export_csv([present, other], out, ProcessingConfig())
    df = pd.read_csv(out, comment="#", keep_default_na=False)

    # Union of sample ids -> both columns exist on both rows, no holes.
    assert df.iloc[0]["height_rep2"] == 0.0
    assert df.iloc[0]["area_rep2"] == 0.0
    assert df.iloc[0]["gap_fill_status_rep2"] == ""
    assert df.iloc[1]["height_rep1"] == 0.0
    assert df.iloc[1]["gap_fill_status_rep1"] == ""

    # And the numeric columns really parse as numbers, which a blank would break.
    assert pd.to_numeric(pd.read_csv(out, comment="#")["height_rep2"]).notna().all()


def test_export_annotated_column_uses_the_shared_predicate(tmp_path):
    """``annotated`` must be exactly what the stage-7 refiner grouped on.

    A sparse hit clears the score floor but not the matched-peak floor, so it
    keeps its name and score cells and comes out annotated=False — and a rename
    to "Putative: ..." must not change the answer either way.
    """
    config = ProcessingConfig()
    confidence = config.confidence_view()

    good = _annotated_feature()
    sparse = _annotated_feature()
    sparse.feature_id = "F0002"
    sparse.annotation_matches[0].n_matched = 1
    renamed = _annotated_feature()
    renamed.feature_id = "F0003"
    renamed.name = "Putative: TestHit"

    out = tmp_path / "features.csv"
    _export_csv([good, sparse, renamed], out, ProcessingConfig())
    df = pd.read_csv(out, comment="#")

    assert list(df["annotated"]) == [
        is_high_confidence(f, confidence) for f in (good, sparse, renamed)
    ] == [True, False, True]
    assert df.iloc[1]["name"] == "TestHit"          # suggestion keeps its cells
    assert df.iloc[2]["name"] == "Putative: TestHit"
