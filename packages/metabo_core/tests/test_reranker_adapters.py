from metabo_core.annotation.adapters import (
    from_annotation_match, from_gcms_hit, to_annotation_match, to_gcms_hit,
)
from metabo_core.annotation.reranker import AnnotationCandidate
from metabo_core.models.features import AnnotationMatch


def test_annotation_match_round_trip():
    m = AnnotationMatch(
        rank=1, name="caffeine", formula="C8H10N4O2",
        score=0.91, n_matched=7, wdp=0.88, sdp=0.85, rdp=0.93,
        ref_peaks=[(195.09, 1.0), (138.07, 0.4)],
        ref_precursor_mz=195.09, adduct="[M+H]+",
    )
    cand = from_annotation_match(m)
    assert cand.name == "caffeine"
    assert cand.formula == "C8H10N4O2"
    assert cand.score == 0.91
    assert cand.ref_precursor_mz == 195.09
    assert cand.adduct == "[M+H]+"
    assert cand.ref_peaks == [(195.09, 1.0), (138.07, 0.4)]

    back = to_annotation_match(cand)
    assert back.name == "caffeine"
    assert back.rank == 1
    assert back.wdp == 0.88
    assert back.adduct == "[M+H]+"


def test_gcms_hit_round_trip_preserves_all_fields():
    hit = {
        "name": "limonene", "formula": "C10H16", "inchikey": "XMGQYMWWDOXHJM-UHFFFAOYSA-N",
        "adduct": "", "rt": 12.3, "ri": 1030.5,
        "total_score": 0.86, "spectral_score": 0.91, "chrom_score": 0.74,
        "wdp": 0.92, "rdp": 0.88, "sdp": 0.85,
        "matched_pct": 0.7,
        "n_adjacent_subtracted": 2,
        "ref_peaks": [(93.0, 1.0), (68.0, 0.7)],
        "score": 0.86, "n_matched": 9,
        "acquired_ion_count": 24,
    }
    cand = from_gcms_hit(hit, rank=1)
    assert cand.name == "limonene"
    assert cand.inchikey.startswith("XMGQ")
    assert cand.ref_ri == 1030.5
    assert cand.ref_rt == 12.3
    assert cand.score == 0.86  # mapped from total_score
    assert cand.extras["chrom_score"] == 0.74
    assert cand.extras["matched_pct"] == 0.7
    assert cand.extras["acquired_ion_count"] == 24

    back = to_gcms_hit(cand)
    # All original keys must come back unchanged
    for key in ("total_score", "spectral_score", "chrom_score",
                "matched_pct", "n_adjacent_subtracted", "acquired_ion_count",
                "ref_peaks", "ri", "rt", "name", "formula", "inchikey",
                "wdp", "rdp", "sdp", "n_matched", "score"):
        assert key in back, f"adapter lost field {key}"
    assert back["chrom_score"] == 0.74
    assert back["ri"] == 1030.5


def test_gcms_hit_missing_optional_fields_defaults_to_zero_or_none():
    hit = {
        "name": "x", "total_score": 0.5, "score": 0.5, "n_matched": 3,
        "ref_peaks": [], "wdp": 0.5, "rdp": 0.5, "sdp": 0.5,
        # all other fields absent
    }
    cand = from_gcms_hit(hit, rank=1)
    assert cand.ref_ri is None
    assert cand.ref_rt is None
    assert cand.inchikey == ""
