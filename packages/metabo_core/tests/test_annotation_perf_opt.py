"""Performance-optimization tests for the LC-MS library annotator.

Covers three behavior-level guarantees added when Stage 6.5 was sped up:

* S1 — the precursor candidate window is tightened to ±0.5 Da.
* S2 — a cheap binned shared-peak pre-filter skips candidates that cannot
  reach ``min_matched_peaks`` (in either the direct or the neutral-loss
  frame), and is *result-preserving*: it never drops a real hit.
* S3 — removing per-candidate redundancy keeps the composite score and the
  emitted matches byte-identical (golden regression below).
"""
import numpy as np

from metabo_core.annotation import (
    build_index_from_list,
    match_feature_topn,
)
from metabo_core.config import AnnotationConfig, SimilarityConfig
from metabo_core.models import CandidateFeature


# --------------------------------------------------------------------------
# Shared deterministic dataset for the golden regression
# --------------------------------------------------------------------------
QPEAKS = [(80.05, 1000.0), (120.07, 850.0), (160.10, 700.0),
          (200.12, 500.0), (255.15, 300.0), (300.20, 150.0)]


def _build_dataset():
    """2 planted hits + 58 noise spectra, all precursors within 0.4 Da."""
    rng = np.random.default_rng(20260626)
    spectra = [
        {  # Hit 1: exact identity copy of the query
            "mz": np.array([m for m, _ in QPEAKS]),
            "intensity": np.array([i for _, i in QPEAKS]),
            "metadata": {"precursor_mz": 400.02, "name": "Alpha", "formula": "C1"},
        },
        {  # Hit 2: first 4 query peaks (rescaled) + 2 off-target peaks
            "mz": np.array([80.05, 120.07, 160.10, 200.12, 60.0, 350.0]),
            "intensity": np.array([900.0, 800.0, 650.0, 480.0, 200.0, 100.0]),
            "metadata": {"precursor_mz": 399.95, "name": "Beta", "formula": "C2"},
        },
    ]
    for k in range(58):  # noise: peaks deliberately off the query m/z grid
        npk = int(rng.integers(4, 9))
        mz = np.sort(np.concatenate([
            rng.uniform(50.0, 75.0, npk // 2),
            rng.uniform(310.0, 399.0, npk - npk // 2),
        ]))
        inten = rng.uniform(100.0, 5000.0, mz.size)
        pmz = float(rng.uniform(399.6, 400.4))
        spectra.append({"mz": mz, "intensity": inten,
                        "metadata": {"precursor_mz": pmz, "name": f"N{k}"}})
    return spectra


def _feature(precursor_mz: float, peaks: list[tuple[float, float]]) -> CandidateFeature:
    feat = CandidateFeature(
        feature_id="F00001", segment_name="seg", replicate_id=1,
        precursor_mz_nominal=int(round(precursor_mz)),
        rt_apex=5.0, rt_left=4.9, rt_right=5.1,
        ms2_mz=np.array([m for m, _ in peaks]),
        ms2_intensity=np.array([i for _, i in peaks]),
        n_fragments=len(peaks),
    )
    feat.ms1_precursor_mz = precursor_mz
    return feat


# Captured from the pre-optimization implementation; the optimized path must
# reproduce these exactly. Fields: name, score, n_matched, wdp, sdp, rdp,
# matched_pct (floats rounded to 6 dp).
GOLDEN = [
    ("Alpha", 2.000000, 6, 1.000000, 1.000000, 1.000000, 1.000000),
    ("Beta", 1.550819, 4, 0.677708, 0.787797, 0.886699, 0.666667),
]


# --------------------------------------------------------------------------
# S1: precursor window tightened to ±0.5 Da
# --------------------------------------------------------------------------
def test_precursor_window_excludes_beyond_half_da():
    peaks = [(100.0, 1000.0), (150.0, 800.0), (200.0, 600.0)]
    spectra = [
        {"mz": np.array([100.0, 150.0, 200.0]),
         "intensity": np.array([1000.0, 800.0, 600.0]),
         "metadata": {"precursor_mz": 400.30, "name": "Near"}},  # 0.3 Da -> in
        {"mz": np.array([100.0, 150.0, 200.0]),
         "intensity": np.array([1000.0, 800.0, 600.0]),
         "metadata": {"precursor_mz": 400.70, "name": "Far"}},   # 0.7 Da -> out
    ]
    library = build_index_from_list(spectra)
    feature = _feature(400.0, peaks)
    matches = match_feature_topn(
        feature, library["spectra"], library["index"],
        AnnotationConfig(similarity_threshold=0.5, min_matched_peaks=2,
                         min_matched_pct=0.5),
        SimilarityConfig(mz_tolerance=0.02, ms1_tolerance=0.01),
        top_n=5,
    )
    names = {m.name for m in matches}
    assert "Near" in names
    assert "Far" not in names


# --------------------------------------------------------------------------
# S2: binned shared-peak pre-filter is a safe upper bound (no false skips)
# --------------------------------------------------------------------------
def _wide_query_bins(query_mz, inv_tol):
    bins = set()
    for q in query_mz:
        b = int(q * inv_tol)
        bins.update((b - 1, b, b + 1))
    return bins


def test_prefilter_passes_on_direct_overlap():
    from metabo_core.annotation.library import _passes_prefilter
    inv = 1.0 / 0.02
    qbins = _wide_query_bins((100.0, 150.0, 200.0), inv)
    ref = np.array([100.0, 150.0, 200.0])  # 3 direct matches
    assert _passes_prefilter(qbins, ref, 0.0, inv, 3) is True


def test_prefilter_skips_when_too_few_shared():
    from metabo_core.annotation.library import _passes_prefilter
    inv = 1.0 / 0.02
    qbins = _wide_query_bins((100.0, 150.0, 200.0), inv)
    ref = np.array([100.0, 999.0, 888.0])  # only 1 direct match
    assert _passes_prefilter(qbins, ref, 0.0, inv, 3) is False


def test_prefilter_keeps_neutral_loss_only_candidate():
    """Direct overlap < min, but the shifted (neutral-loss) frame aligns:
    must NOT be skipped, else a real NL hit would be lost."""
    from metabo_core.annotation.library import _passes_prefilter
    inv = 1.0 / 0.02
    qbins = _wide_query_bins((100.0, 150.0, 200.0), inv)
    # query_mz ~ ref_mz + shift ; ref = query - 5.0 aligns when shift = 5.0
    ref = np.array([95.0, 145.0, 195.0])
    assert _passes_prefilter(qbins, ref, 0.0, inv, 3) is False   # no shift -> skip
    assert _passes_prefilter(qbins, ref, 5.0, inv, 3) is True    # NL frame -> keep


# --------------------------------------------------------------------------
# S2 + S3: optimized matching reproduces the pre-optimization output exactly
# --------------------------------------------------------------------------
def test_optimized_matching_preserves_golden():
    lib = build_index_from_list(_build_dataset())
    feat = _feature(400.0, QPEAKS)
    matches = match_feature_topn(
        feat, lib["spectra"], lib["index"],
        AnnotationConfig(), SimilarityConfig(), top_n=5,
    )
    got = [
        (m.name, round(m.score, 6), m.n_matched, round(m.wdp, 6),
         round(m.sdp, 6), round(m.rdp, 6), round(m.matched_pct, 6))
        for m in matches
    ]
    assert got == GOLDEN
