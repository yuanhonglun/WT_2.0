"""PR-B: features.csv exposes the MS-DIAL score breakdown columns."""
import numpy as np
import pandas as pd

from asfam.pipeline.stage8_export import _export_csv
from asfam.config import ProcessingConfig
from asfam.models import Feature, AnnotationMatch


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
