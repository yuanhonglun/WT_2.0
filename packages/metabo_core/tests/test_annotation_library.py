"""Regression tests for the metabo_core library annotator."""
import numpy as np

from metabo_core.annotation import (
    build_index_from_list,
    load_and_index_library,
    match_feature_topn,
)
from metabo_core.config import AnnotationConfig, SimilarityConfig
from metabo_core.models import CandidateFeature


def _candidate(precursor_mz: float, peaks: list[tuple[float, float]]) -> CandidateFeature:
    feat = CandidateFeature(
        feature_id="F00001",
        segment_name="100-200",
        replicate_id=1,
        precursor_mz_nominal=int(round(precursor_mz)),
        rt_apex=5.0,
        rt_left=4.9,
        rt_right=5.1,
        ms2_mz=np.array([m for m, _ in peaks]),
        ms2_intensity=np.array([i for _, i in peaks]),
        n_fragments=len(peaks),
    )
    feat.ms1_precursor_mz = precursor_mz
    return feat


def test_build_index_from_list_groups_by_integer_precursor():
    spectra = [
        {"mz": [100.0, 150.0], "intensity": [500.0, 250.0],
         "metadata": {"precursor_mz": 181.07, "name": "A"}},
        {"mz": [100.0, 150.0], "intensity": [500.0, 250.0],
         "metadata": {"precursor_mz": 181.40, "name": "B"}},
        {"mz": [50.0, 60.0], "intensity": [100.0, 200.0],
         "metadata": {"precursor_mz": 90.0, "name": "C"}},
    ]
    library = build_index_from_list(spectra)
    assert library is not None
    assert sorted(library["index"][181]) == [0, 1]
    assert library["index"][90] == [2]


def test_match_feature_topn_returns_ranked_match():
    spectra = [
        {"mz": [100.0, 150.0], "intensity": [1000.0, 500.0],
         "metadata": {"precursor_mz": 181.0707, "name": "Hit", "formula": "C9H12O4"}},
        {"mz": [60.0, 70.0], "intensity": [1000.0, 500.0],
         "metadata": {"precursor_mz": 181.0707, "name": "NotMatching"}},
    ]
    library = build_index_from_list(spectra)
    feature = _candidate(181.0707, [(100.0, 1000.0), (150.0, 500.0)])
    matches = match_feature_topn(
        feature,
        library["spectra"],
        library["index"],
        AnnotationConfig(similarity_threshold=0.5, min_matched_peaks=2, min_matched_pct=0.5),
        SimilarityConfig(mz_tolerance=0.02, ms1_tolerance=0.01),
        top_n=3,
    )
    assert matches
    assert matches[0].name == "Hit"
    assert matches[0].formula == "C9H12O4"
    assert matches[0].rank == 1


def test_match_feature_topn_exposes_score_breakdown():
    # 恒等 3-峰命中：分项全部透出，matched_pct=1.0，total_score==score 且 >1（[0,2] 不钳制）。
    spectra = [
        {"mz": [100.0, 150.0, 200.0], "intensity": [1000.0, 800.0, 600.0],
         "metadata": {"precursor_mz": 181.0707, "name": "Hit", "formula": "C9H12O4"}},
    ]
    library = build_index_from_list(spectra)
    feature = _candidate(181.0707, [(100.0, 1000.0), (150.0, 800.0), (200.0, 600.0)])
    matches = match_feature_topn(
        feature, library["spectra"], library["index"],
        AnnotationConfig(similarity_threshold=0.5, min_matched_peaks=2, min_matched_pct=0.5),
        SimilarityConfig(mz_tolerance=0.02, ms1_tolerance=0.01),
        top_n=1,
    )
    assert matches
    m = matches[0]
    assert 0.0 <= m.wdp <= 1.0
    assert 0.0 <= m.sdp <= 1.0
    assert 0.0 <= m.rdp <= 1.0
    assert 0.0 <= m.matched_pct <= 1.0
    assert abs(m.matched_pct - 1.0) < 1e-9      # 3 显著参考峰全部命中
    assert abs(m.total_score - m.score) < 1e-12  # total_score == composite score
    assert m.total_score > 1.0                   # 恒等谱落在 [0,2]，未被钳到 1


def test_precursor_only_reference_participates_when_relaxed():
    """A precursor-only (1-peak) reference — the sparse [M+Na]+ case — is
    skipped entirely under defaults (``min_peaks_to_match=2``) but participates
    and yields an ``n_matched == 1`` suggestion once an app opts the per-side
    floor AND the emit floor down to 1. Mirrors MS-DIAL keeping such sparse
    matches as suggestions instead of dropping them."""
    spectra = [
        {"mz": [181.0707], "intensity": [1000.0],
         "metadata": {"precursor_mz": 181.0707, "name": "SodiatedThing",
                      "adduct": "[M+Na]+"}},
    ]
    library = build_index_from_list(spectra)
    feature = _candidate(181.0707, [(181.0707, 1000.0), (100.0, 500.0)])
    sim = SimilarityConfig(mz_tolerance=0.02, ms1_tolerance=0.01)

    # Defaults: 1-peak reference is skipped (min_peaks_to_match=2) -> no hit.
    assert match_feature_topn(
        feature, library["spectra"], library["index"],
        AnnotationConfig(), sim,
    ) == []

    # Opt-in emit tier: reference participates, emitted with n_matched == 1.
    matches = match_feature_topn(
        feature, library["spectra"], library["index"],
        AnnotationConfig(similarity_threshold=0.0, min_matched_peaks=1,
                         min_peaks_to_match=1, min_matched_pct=0.25, min_wdp=0.0),
        sim,
    )
    assert matches and matches[0].name == "SodiatedThing"
    assert matches[0].n_matched == 1


def test_emit_floor_keeps_sub_three_matched_hits():
    """With a fragment-bearing reference the *emit floor* (not the per-side
    floor) is the discriminator: a query matching only one of two significant
    reference peaks is dropped at ``min_matched_peaks=3`` but kept — as a
    lower-confidence suggestion carrying its real ``n_matched`` — at
    ``min_matched_peaks=1``. The downstream ``annotated`` flag, not this
    matcher, then applies the high-confidence >=3 tier."""
    spectra = [
        {"mz": [100.0, 200.0], "intensity": [1000.0, 1000.0],
         "metadata": {"precursor_mz": 181.0707, "name": "PartialHit"}},
    ]
    library = build_index_from_list(spectra)
    feature = _candidate(181.0707, [(100.0, 1000.0), (55.0, 500.0)])
    sim = SimilarityConfig(mz_tolerance=0.02, ms1_tolerance=0.01)
    common = dict(similarity_threshold=0.0, min_matched_pct=0.25, min_wdp=0.0)

    # High-confidence emit gate (3) drops the single-matched-peak hit.
    assert match_feature_topn(
        feature, library["spectra"], library["index"],
        AnnotationConfig(min_matched_peaks=3, **common), sim,
    ) == []

    # Relaxed emit floor (1) keeps it, carrying the real n_matched.
    matches = match_feature_topn(
        feature, library["spectra"], library["index"],
        AnnotationConfig(min_matched_peaks=1, **common), sim,
    )
    assert matches and matches[0].n_matched == 1


def test_load_and_index_library_loads_lean(tmp_path):
    """The annotation loader trims each spectrum to the fields the matcher
    uses and stores peaks as numpy arrays (memory bound for multi-GB
    libraries), while matching still works end-to-end and the returned
    ref_peaks are JSON-safe Python floats (not numpy scalars).
    """
    text = (
        "NAME: Hit\nPRECURSORMZ: 181.0707\nFORMULA: C9H12O4\nADDUCT: [M+H]+\n"
        "INCHIKEY: XYZ\nSMILES: CCO\nCOMMENT: bulky unused metadata\n"
        "Num Peaks: 2\n100.0 1000\n150.0 500\n\n"
    )
    path = tmp_path / "lib.msp"
    path.write_text(text, encoding="utf-8")

    lib = load_and_index_library(str(path))
    assert lib is not None
    spec = lib["spectra"][0]
    assert isinstance(spec["mz"], np.ndarray)        # arrays -> low memory
    assert "comment" not in spec["metadata"]         # unused fields pruned
    assert "inchikey" not in spec["metadata"]
    assert spec["metadata"]["name"] == "Hit"         # used fields retained

    feature = _candidate(181.0707, [(100.0, 1000.0), (150.0, 500.0)])
    matches = match_feature_topn(
        feature, lib["spectra"], lib["index"],
        AnnotationConfig(similarity_threshold=0.5, min_matched_peaks=2, min_matched_pct=0.5),
        SimilarityConfig(mz_tolerance=0.02, ms1_tolerance=0.01), top_n=3,
    )
    assert matches and matches[0].name == "Hit"
    assert matches[0].ref_peaks
    m0, i0 = matches[0].ref_peaks[0]
    assert type(m0) is float and type(i0) is float


def test_match_feature_topn_empty_when_no_index_overlap():
    spectra = [
        {"mz": [10.0, 20.0], "intensity": [100.0, 100.0],
         "metadata": {"precursor_mz": 50.0, "name": "Other"}},
    ]
    library = build_index_from_list(spectra)
    feature = _candidate(181.0707, [(100.0, 1000.0), (150.0, 500.0)])
    matches = match_feature_topn(
        feature,
        library["spectra"],
        library["index"],
        AnnotationConfig(),
        SimilarityConfig(),
    )
    assert matches == []


# ---------------------------------------------------------------------------
# min_wdp accept-gate: reject inflated-query false positives whose weighted
# dot product (true spectral shape) is near zero but whose matched_pct=1.0 /
# high rdp still push total_score over the high-confidence line. See
# docs/validation/2026-07-02-t1-r1-ms1-driven-ms2-fix.md §4 and the T5
# matched_pct-inflation handoff. Real matches keep a substantial wdp and are
# unaffected; min_wdp defaults to 0.0 so GC-MS / DDA behaviour is unchanged.
# ---------------------------------------------------------------------------

def _inflated_query_fixture():
    """An inflated fallback-style query that trivially covers a small ref.

    The 3 reference significant peaks are all matched (matched_pct=1.0,
    n_matched=3), but ~40 unmatched high-intensity noise peaks dilute the
    weighted/simple dot products to near zero — the false-positive signature.
    Same precursor so no neutral-loss frame confounds the match.
    """
    ref_peaks = [(100.0, 999.0), (150.0, 800.0), (200.0, 600.0)]
    spectra = [
        {"mz": [m for m, _ in ref_peaks], "intensity": [i for _, i in ref_peaks],
         "metadata": {"precursor_mz": 300.0, "name": "InflatedFalsePositive"}},
    ]
    library = build_index_from_list(spectra)
    # 3 matched peaks at low intensity + 40 unmatched noise peaks at full scale.
    q = [(100.0, 50.0), (150.0, 40.0), (200.0, 30.0)]
    q += [(250.0 + k, 999.0) for k in range(40)]  # noise, no ref overlap
    feature = _candidate(300.0, q)
    return library, feature


def test_min_wdp_gate_rejects_inflated_low_wdp_false_positive():
    library, feature = _inflated_query_fixture()
    common = dict(min_matched_peaks=2, min_matched_pct=0.25, similarity_threshold=0.0)

    # Without the gate (min_wdp=0), the inflated match is accepted and its
    # breakdown shows the false-positive signature: matched_pct=1.0, wdp<0.10.
    before = match_feature_topn(
        feature, library["spectra"], library["index"],
        AnnotationConfig(min_wdp=0.0, **common),
        SimilarityConfig(mz_tolerance=0.02, ms1_tolerance=0.01),
    )
    assert before, "inflated match should pass the legacy gate"
    assert abs(before[0].matched_pct - 1.0) < 1e-9  # trivial full coverage
    assert before[0].wdp < 0.10                     # near-zero true spectral shape

    # With the gate at 0.10, the same match is rejected.
    after = match_feature_topn(
        feature, library["spectra"], library["index"],
        AnnotationConfig(min_wdp=0.10, **common),
        SimilarityConfig(mz_tolerance=0.02, ms1_tolerance=0.01),
    )
    assert after == []


def test_min_wdp_gate_keeps_true_high_wdp_match():
    ref_peaks = [(100.0, 999.0), (150.0, 800.0), (200.0, 600.0)]
    spectra = [
        {"mz": [m for m, _ in ref_peaks], "intensity": [i for _, i in ref_peaks],
         "metadata": {"precursor_mz": 300.0, "name": "TrueMatch"}},
    ]
    library = build_index_from_list(spectra)
    feature = _candidate(300.0, ref_peaks)  # clean identity query -> wdp high
    matches = match_feature_topn(
        feature, library["spectra"], library["index"],
        AnnotationConfig(min_wdp=0.10, min_matched_peaks=2,
                         min_matched_pct=0.25, similarity_threshold=0.0),
        SimilarityConfig(mz_tolerance=0.02, ms1_tolerance=0.01),
    )
    assert matches and matches[0].name == "TrueMatch"
    assert matches[0].wdp >= 0.10


def test_min_wdp_defaults_to_no_op():
    # Default AnnotationConfig leaves min_wdp disabled so shared GC-MS / DDA
    # annotation paths are byte-identical unless an app opts in.
    assert AnnotationConfig().min_wdp == 0.0
    library, feature = _inflated_query_fixture()
    matches = match_feature_topn(
        feature, library["spectra"], library["index"],
        AnnotationConfig(min_matched_peaks=2, min_matched_pct=0.25,
                         similarity_threshold=0.0),
        SimilarityConfig(mz_tolerance=0.02, ms1_tolerance=0.01),
    )
    assert matches  # no gate by default -> inflated match still accepted
