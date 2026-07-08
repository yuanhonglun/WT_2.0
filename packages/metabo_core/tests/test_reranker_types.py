from dataclasses import asdict
from metabo_core.annotation.reranker import (
    AnnotationCandidate, RankingInput, RankingResult,
)


def test_annotation_candidate_defaults():
    c = AnnotationCandidate(rank=1, name="caffeine")
    assert c.rank == 1
    assert c.name == "caffeine"
    assert c.formula == ""
    assert c.extras == {}
    assert c.ref_peaks is None


def test_annotation_candidate_round_trip_via_asdict():
    c = AnnotationCandidate(
        rank=2,
        name="quercetin",
        formula="C15H10O7",
        inchikey="REFJWTPEDVJJIY-UHFFFAOYSA-N",
        adduct="[M+H]+",
        score=0.91,
        n_matched=7,
        wdp=0.88, sdp=0.85, rdp=0.93,
        ref_peaks=[(303.05, 1.0), (151.0, 0.4)],
        ref_precursor_mz=303.05,
        ref_ri=1850.0,
        ref_rt=12.4,
        extras={"chrom_score": 0.77, "matched_pct": 0.42},
    )
    d = asdict(c)
    assert d["extras"]["chrom_score"] == 0.77
    assert d["ref_peaks"][0] == (303.05, 1.0)


def test_ranking_input_holds_evidence():
    r = RankingInput(
        feature_id="F00007",
        mode="dda",
        measured_precursor_mz=303.05,
        measured_rt=12.4,
        measured_ri=None,
        measured_spectrum=[(303.0, 1.0)],
        candidates=[AnnotationCandidate(rank=1, name="x")],
    )
    assert r.mode == "dda"
    assert r.measured_ri is None
    assert len(r.candidates) == 1


def test_ranking_result_explanations_keyed_by_rank():
    res = RankingResult(
        feature_id="F00007",
        candidates=[AnnotationCandidate(rank=1, name="a"), AnnotationCandidate(rank=2, name="b")],
        explanations={1: "Best library hit + RI within 5 units.", 2: "Lower spectral score."},
        reranker_name="CosineRiReranker",
        reranker_version="1.0",
    )
    assert res.explanations[1].startswith("Best")
    assert res.candidates[1].name == "b"
