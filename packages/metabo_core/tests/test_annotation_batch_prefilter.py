"""PR-1 / PR-1b: CSR index + the two batch upper-bound screens in ``match_feature_topn``.

Both screens drop candidates before the exact scorer ever sees them, so the whole
optimisation rests on a single property: neither bound may *under*estimate the true
value it bounds (``matched_pct`` for S2', ``wdp`` for S2"). If one did, a real library
hit would silently vanish from ``features.csv`` and no other test would notice.

``_oracle_match_feature_topn`` below is the matcher as of commit ``ce8600c``, kept
verbatim. Both PRs are pure speed changes: every emitted ``AnnotationMatch`` must still
equal the oracle's field for field, exactly (``==``, not ``approx`` — hits are ranked
by raw score and the baseline CSV is compared by hash).
"""
import numpy as np

from metabo_core.algorithms.similarity import (
    _greedy_match_arrays,
    composite_similarity_breakdown,
)
from metabo_core.annotation import build_index_from_list, match_feature_topn
from metabo_core.annotation import library as library_mod
from metabo_core.annotation.library import (
    _build_bucket_csr,
    _matched_pct_upper_bound,
    _passes_prefilter,
    _range_max,
    _sparse_range_max_table,
    _wdp_upper_bound,
)
from metabo_core.config import AnnotationConfig, SimilarityConfig
from metabo_core.models import AnnotationMatch, CandidateFeature


def _candidate(precursor_mz: float, peaks: list[tuple[float, float]]) -> CandidateFeature:
    feat = CandidateFeature(
        feature_id="F00001", segment_name="seg", replicate_id=1,
        precursor_mz_nominal=int(round(precursor_mz)),
        rt_apex=5.0, rt_left=4.9, rt_right=5.1,
        ms2_mz=np.array([m for m, _ in peaks], dtype=np.float64),
        ms2_intensity=np.array([i for _, i in peaks], dtype=np.float64),
        n_fragments=len(peaks),
    )
    feat.ms1_precursor_mz = precursor_mz
    return feat


# ---------------------------------------------------------------------------
# Oracle: match_feature_topn exactly as of ce8600c (pre-CSR)
# ---------------------------------------------------------------------------
def _oracle_match_feature_topn(feature, spectra, mz_index, annotation, similarity,
                               top_n=None):
    feat_mz = feature.precursor_mz
    feat_mz_int = int(round(feat_mz))

    candidate_indices: list[int] = []
    for offset in (-1, 0, 1):
        candidate_indices.extend(mz_index.get(feat_mz_int + offset, []))
    if not candidate_indices:
        return []

    query_peaks = list(zip(feature.ms2_mz.tolist(), feature.ms2_intensity.tolist()))
    if len(query_peaks) < annotation.min_peaks_to_match:
        return []

    limit = top_n if top_n is not None else annotation.top_n

    hits = []
    rt_query = feature.rt_apex
    mz_tol = similarity.mz_tolerance
    q_mz = np.asarray(feature.ms2_mz, dtype=np.float64)
    q_int = np.asarray(feature.ms2_intensity, dtype=np.float64)
    inv_tol = 1.0 / mz_tol if mz_tol > 0 else 0.0
    query_bins_wide: set[int] = set()
    if inv_tol > 0:
        for m in q_mz:
            b = int(m * inv_tol)
            query_bins_wide.update((b - 1, b, b + 1))

    for idx in candidate_indices:
        spec = spectra[idx]
        ref_mz_arr = spec.get("mz")
        ref_int_arr = spec.get("intensity")
        if ref_mz_arr is None or len(ref_mz_arr) < annotation.min_peaks_to_match:
            continue
        ref_pmz = spec.get("_precursor_mz")
        if ref_pmz is not None and abs(ref_pmz - feat_mz) > 0.5:
            continue
        if inv_tol > 0 and annotation.min_matched_peaks > 0:
            shift = (ref_pmz - feat_mz) if ref_pmz is not None else 0.0
            if not _passes_prefilter(query_bins_wide, ref_mz_arr, shift, inv_tol,
                                     annotation.min_matched_peaks):
                continue

        ref_peaks = list(zip(ref_mz_arr, ref_int_arr))
        breakdown = composite_similarity_breakdown(
            query_peaks, ref_peaks, mz_tol,
            precursor_query=feat_mz,
            precursor_ref=ref_pmz if ref_pmz is not None else 0.0,
            rt_query=rt_query, rt_ref=spec.get("_rt", 0.0),
            ms1_tolerance=similarity.ms1_tolerance, use_rt=similarity.use_rt,
            q_arrays=(q_mz, q_int),
            r_arrays=(np.asarray(ref_mz_arr, dtype=np.float64),
                      np.asarray(ref_int_arr, dtype=np.float64)),
        )
        if (breakdown.score >= annotation.similarity_threshold and
                breakdown.n_matched >= annotation.min_matched_peaks and
                breakdown.matched_pct >= annotation.min_matched_pct and
                breakdown.wdp >= annotation.min_wdp):
            meta = spec.get("metadata", {})
            hits.append({
                "name": meta.get("name", ""),
                "formula": meta.get("formula", ""),
                "score": breakdown.score,
                "n_matched": breakdown.n_matched,
                "ref_peaks": [(float(m), float(v)) for m, v in ref_peaks],
                "ref_precursor_mz": meta.get("precursor_mz"),
                "adduct": meta.get("adduct", ""),
                "wdp": breakdown.wdp, "sdp": breakdown.sdp, "rdp": breakdown.rdp,
                "matched_pct": breakdown.matched_pct,
                "total_score": breakdown.total_score,
            })

    hits.sort(key=lambda h: h["score"], reverse=True)
    return [
        AnnotationMatch(
            rank=i + 1, name=h["name"], formula=h.get("formula", ""),
            score=h["score"], n_matched=h["n_matched"],
            ref_peaks=h.get("ref_peaks"), ref_precursor_mz=h.get("ref_precursor_mz"),
            adduct=h.get("adduct", ""), wdp=h.get("wdp", 0.0), sdp=h.get("sdp", 0.0),
            rdp=h.get("rdp", 0.0), matched_pct=h.get("matched_pct", 0.0),
            total_score=h.get("total_score", 0.0),
        )
        for i, h in enumerate(hits[:limit])
    ]


# ---------------------------------------------------------------------------
# Randomised fixtures
# ---------------------------------------------------------------------------
def _random_library(rng, n_spectra, centre=400.0):
    """Spectra whose precursors straddle the three integer buckets around `centre`."""
    spectra = []
    for k in range(n_spectra):
        n_pk = int(rng.integers(1, 25))
        mz = np.sort(rng.uniform(50.0, centre, n_pk))
        # A wide intensity spread makes n_sig_ref (>= 1% of base peak) non-trivial,
        # which is what gives the Matched% bound something to be wrong about.
        inten = rng.uniform(1.0, 1000.0, n_pk) ** rng.uniform(1.0, 3.0)
        pmz = float(centre + rng.uniform(-1.4, 1.4))
        spectra.append({
            "mz": mz, "intensity": inten,
            "metadata": {"precursor_mz": pmz, "name": f"S{k}", "formula": f"F{k}"},
        })
    return spectra


def _random_query(rng, library_spectra, centre=400.0):
    """Copy peaks off a random reference so hits actually happen; sometimes shift
    the whole query into the neutral-loss frame."""
    donor = library_spectra[int(rng.integers(0, len(library_spectra)))]
    donor_mz = np.asarray(donor["mz"], dtype=np.float64)
    donor_int = np.asarray(donor["intensity"], dtype=np.float64)
    n_take = min(donor_mz.size, int(rng.integers(1, 10)))
    take = rng.choice(donor_mz.size, n_take, replace=False)
    nl_shift = float(rng.choice([0.0, 0.0, 18.0106, -2.5]))
    mz = list(donor_mz[take] + nl_shift)
    inten = list(donor_int[take])
    for _ in range(int(rng.integers(0, 12))):
        mz.append(float(rng.uniform(50.0, centre)))
        inten.append(float(rng.uniform(1.0, 1000.0)))
    order = np.argsort(mz)
    peaks = [(float(mz[i]), float(inten[i])) for i in order]
    donor_pmz = float(donor["metadata"]["precursor_mz"])
    return _candidate(donor_pmz - nl_shift, peaks)


def _assert_matches_equal(got, expected):
    assert len(got) == len(expected)
    for a, b in zip(got, expected):
        assert a.rank == b.rank
        assert a.name == b.name
        assert a.formula == b.formula
        assert a.score == b.score
        assert a.n_matched == b.n_matched
        assert a.ref_precursor_mz == b.ref_precursor_mz
        assert a.adduct == b.adduct
        assert a.wdp == b.wdp and a.sdp == b.sdp and a.rdp == b.rdp
        assert a.matched_pct == b.matched_pct
        assert a.total_score == b.total_score
        assert a.ref_peaks == b.ref_peaks


# ---------------------------------------------------------------------------
# Step 1.4.1 -- the bound never underestimates (the correctness core of PR-1)
# ---------------------------------------------------------------------------
def test_matched_pct_upper_bound_never_underestimates():
    """Brute-force the true ``matched_pct`` of *every* (query, candidate) pair and
    assert the vectorised bound dominates it, including when the neutral-loss frame
    is the one that wins."""
    rng = np.random.default_rng(20260708)
    spectra = _random_library(rng, 200)
    library = build_index_from_list(spectra)
    csr = library["csr"]
    mz_tol = 0.02

    n_pairs = 0
    n_nl_wins = 0
    for _ in range(100):
        feature = _random_query(rng, library["spectra"])
        feat_mz = feature.precursor_mz
        q_mz = np.asarray(feature.ms2_mz, dtype=np.float64)
        q_int = np.asarray(feature.ms2_intensity, dtype=np.float64)
        q_sorted = np.sort(q_mz)
        key0 = int(round(feat_mz))
        for key in (key0 - 1, key0, key0 + 1):
            bucket = csr.get(key)
            if bucket is None:
                continue
            ub_pct = _matched_pct_upper_bound(bucket, q_sorted, feat_mz, mz_tol)
            for k, idx in enumerate(bucket["spec_idx"].tolist()):
                spec = library["spectra"][idx]
                r_mz = np.asarray(spec["mz"], dtype=np.float64)
                r_int = np.asarray(spec["intensity"], dtype=np.float64)
                ref_pmz = spec["_precursor_mz"]
                breakdown = composite_similarity_breakdown(
                    None, None, mz_tol,
                    precursor_query=feat_mz, precursor_ref=ref_pmz,
                    q_arrays=(q_mz, q_int), r_arrays=(r_mz, r_int),
                )
                assert ub_pct[k] >= breakdown.matched_pct - 1e-12, (
                    f"UB {ub_pct[k]} < true {breakdown.matched_pct} "
                    f"(feat_mz={feat_mz}, ref_pmz={ref_pmz})"
                )
                n_pairs += 1
                direct = _greedy_match_arrays(q_mz, q_int, r_mz, r_int, mz_tol, 0.0)
                nl = _greedy_match_arrays(q_mz, q_int, r_mz, r_int, mz_tol,
                                          ref_pmz - feat_mz)
                if len(nl) > len(direct):
                    n_nl_wins += 1

    assert n_pairs > 1000, f"too few pairs exercised: {n_pairs}"
    assert n_nl_wins > 0, "no neutral-loss-winning pair exercised"


# ---------------------------------------------------------------------------
# Step 1.4.2 -- end-to-end equivalence with the pre-CSR implementation
# ---------------------------------------------------------------------------
def test_batch_screen_matches_pre_csr_oracle():
    """ASFAM gates (min_matched_peaks=1, min_wdp=0.10)."""
    rng = np.random.default_rng(4242)
    spectra = _random_library(rng, 300)
    library = build_index_from_list(spectra)
    ann = AnnotationConfig(similarity_threshold=0.0, min_matched_peaks=1,
                           min_peaks_to_match=1, min_matched_pct=0.25, min_wdp=0.10)
    sim = SimilarityConfig(mz_tolerance=0.02, ms1_tolerance=0.01)

    n_with_hits = 0
    for _ in range(200):
        feature = _random_query(rng, library["spectra"])
        got = match_feature_topn(feature, library["spectra"], library["index"],
                                 ann, sim, top_n=5, csr=library["csr"])
        expected = _oracle_match_feature_topn(
            feature, library["spectra"], library["index"], ann, sim, top_n=5)
        _assert_matches_equal(got, expected)
        n_with_hits += bool(got)
    assert n_with_hits > 20, f"oracle comparison was mostly vacuous: {n_with_hits}"


def test_batch_screen_equals_oracle_under_dda_gates():
    """DDA runs min_matched_peaks=3, the regime where _passes_prefilter still bites."""
    rng = np.random.default_rng(99)
    spectra = _random_library(rng, 200)
    library = build_index_from_list(spectra)
    ann = AnnotationConfig(similarity_threshold=0.0, min_matched_peaks=3,
                           min_peaks_to_match=2, min_matched_pct=0.25)
    sim = SimilarityConfig(mz_tolerance=0.02, ms1_tolerance=0.01)
    for _ in range(120):
        feature = _random_query(rng, library["spectra"])
        got = match_feature_topn(feature, library["spectra"], library["index"],
                                 ann, sim, top_n=5, csr=library["csr"])
        expected = _oracle_match_feature_topn(
            feature, library["spectra"], library["index"], ann, sim, top_n=5)
        _assert_matches_equal(got, expected)


def test_lazy_csr_equals_prebuilt_csr():
    """Omitting `csr` re-flattens the three touched buckets on the fly; the results
    must not move (the two paths share one builder)."""
    rng = np.random.default_rng(7)
    spectra = _random_library(rng, 120)
    library = build_index_from_list(spectra)
    ann = AnnotationConfig(similarity_threshold=0.0, min_matched_peaks=1,
                           min_peaks_to_match=1, min_matched_pct=0.25)
    sim = SimilarityConfig(mz_tolerance=0.02, ms1_tolerance=0.01)
    for _ in range(40):
        feature = _random_query(rng, library["spectra"])
        with_csr = match_feature_topn(feature, library["spectra"], library["index"],
                                      ann, sim, top_n=5, csr=library["csr"])
        without = match_feature_topn(feature, library["spectra"], library["index"],
                                     ann, sim, top_n=5)
        _assert_matches_equal(without, with_csr)


# ---------------------------------------------------------------------------
# Step 1.4.3 / 1.4.4 -- top-N ref_peaks, and tie order
# ---------------------------------------------------------------------------
def test_ref_peaks_built_only_for_top_n_survivors():
    """8 hits, top_n=3: exactly 3 come back, each with ref_peaks, score-descending."""
    peaks = [(100.0, 1000.0), (150.0, 800.0), (200.0, 600.0), (250.0, 400.0)]
    spectra = []
    for k in range(8):
        inten = [1000.0 - 40.0 * k, 800.0, 600.0, 400.0]
        spectra.append({
            "mz": np.array([m for m, _ in peaks]),
            "intensity": np.array(inten),
            "metadata": {"precursor_mz": 300.0 + 0.001 * k, "name": f"H{k}"},
        })
    library = build_index_from_list(spectra)
    feature = _candidate(300.0, peaks)
    matches = match_feature_topn(
        feature, library["spectra"], library["index"],
        AnnotationConfig(similarity_threshold=0.0, min_matched_peaks=1,
                         min_peaks_to_match=1, min_matched_pct=0.25),
        SimilarityConfig(mz_tolerance=0.02, ms1_tolerance=0.01),
        top_n=3, csr=library["csr"],
    )
    assert len(matches) == 3
    assert all(m.ref_peaks and len(m.ref_peaks) == 4 for m in matches)
    assert [m.rank for m in matches] == [1, 2, 3]
    scores = [m.score for m in matches]
    assert scores == sorted(scores, reverse=True)
    assert matches[0].name == "H0"  # the identity copy scores highest


def test_equal_score_ties_keep_candidate_enumeration_order():
    """Identical spectra in different buckets score identically. ``hits.sort`` is
    stable, so the winner is whichever the (-1, 0, +1) bucket walk enumerates
    first -- the CSR walk must preserve that."""
    peaks = [(100.0, 1000.0), (150.0, 800.0), (200.0, 600.0)]
    spectra = [
        {"mz": np.array([m for m, _ in peaks]),
         "intensity": np.array([i for _, i in peaks]),
         "metadata": {"precursor_mz": 300.0, "name": "InBucket300"}},
        {"mz": np.array([m for m, _ in peaks]),
         "intensity": np.array([i for _, i in peaks]),
         "metadata": {"precursor_mz": 299.6, "name": "InBucket300Lower"}},
        {"mz": np.array([m for m, _ in peaks]),
         "intensity": np.array([i for _, i in peaks]),
         "metadata": {"precursor_mz": 299.4, "name": "InBucket299"}},
    ]
    library = build_index_from_list(spectra)
    feature = _candidate(299.9, peaks)
    ann = AnnotationConfig(similarity_threshold=0.0, min_matched_peaks=1,
                           min_peaks_to_match=1, min_matched_pct=0.25)
    sim = SimilarityConfig(mz_tolerance=0.02, ms1_tolerance=0.01)
    got = match_feature_topn(feature, library["spectra"], library["index"],
                             ann, sim, top_n=5, csr=library["csr"])
    expected = _oracle_match_feature_topn(
        feature, library["spectra"], library["index"], ann, sim, top_n=5)
    assert len({m.score for m in got}) == 1, "fixture must produce a real tie"
    assert [m.name for m in got] == [m.name for m in expected]
    assert got[0].name == "InBucket299"  # bucket 299 is walked before bucket 300


def test_build_index_keeps_spectra_and_index_keys():
    """DDA's ctx.library_index and the annotate stages read these two; `csr` is
    additive, never a replacement."""
    rng = np.random.default_rng(1)
    library = build_index_from_list(_random_library(rng, 10))
    assert set(library) == {"spectra", "index", "csr"}
    assert isinstance(library["index"], dict)
    assert all(isinstance(v, list) for v in library["index"].values())


def test_shared_peak_views_preserve_values():
    """`build_index_from_list` re-points spec["mz"] into the bucket's flat array;
    the values must be bit-identical to what was loaded."""
    rng = np.random.default_rng(3)
    spectra = _random_library(rng, 30)
    originals = [np.array(s["mz"], dtype=np.float64, copy=True) for s in spectra]
    library = build_index_from_list(spectra)
    for orig, spec in zip(originals, library["spectra"]):
        assert np.array_equal(orig, np.asarray(spec["mz"], dtype=np.float64))


# ===========================================================================
# PR-1b -- the WDP upper bound (S2")
# ===========================================================================
def _wdp_inputs(feature):
    """The per-feature precompute ``match_feature_topn`` hands ``_wdp_upper_bound``."""
    q_mz = np.asarray(feature.ms2_mz, dtype=np.float64)
    q_int = np.asarray(feature.ms2_intensity, dtype=np.float64)
    order = np.argsort(q_mz, kind="stable")
    q_sorted = q_mz[order]
    nq = q_int / float(q_int.max())
    table = _sparse_range_max_table(np.maximum(nq[order], 0.0))
    return q_sorted, table, float((nq * q_mz).sum()), float(nq.sum())


def _wdp_library(rng, n_spectra, centre=400.0):
    """Like `_random_library`, but with peak counts up to 40 and an intensity
    spread wide enough that `n_sig_ref` ranges from 1 to ~30 -- the regime where
    the exact scorer's `min(.., 1.0)` clamp and the penalty brackets both bite."""
    spectra = []
    for k in range(n_spectra):
        n_pk = int(rng.integers(1, 41))
        mz = np.sort(rng.uniform(50.0, centre, n_pk))
        inten = rng.uniform(1.0, 1000.0, n_pk) ** rng.uniform(1.0, 3.5)
        pmz = float(centre + rng.uniform(-1.4, 1.4))
        spectra.append({
            "mz": mz, "intensity": inten,
            "metadata": {"precursor_mz": pmz, "name": f"S{k}", "formula": f"F{k}"},
        })
    return spectra


def test_wdp_upper_bound_never_underestimates():
    """The correctness core of PR-1b.

    Brute-force the true ``wdp`` of *every* (query, candidate) pair and assert the
    vectorised bound dominates it -- across both frames, both signs of the
    neutral-loss shift, and the pairs where the NL frame is the one the scorer picks.
    """
    rng = np.random.default_rng(20260709)
    spectra = _wdp_library(rng, 200)
    library = build_index_from_list(spectra)
    csr = library["csr"]
    mz_tol = 0.02

    nsig = np.concatenate([b["n_sig_ref"] for b in csr.values()])
    assert nsig.min() == 1 and nsig.max() >= 25, f"n_sig_ref spread too narrow: {nsig.min()}..{nsig.max()}"

    n_pairs = n_nl_wins = n_shift_pos = n_shift_neg = n_informative = 0
    for _ in range(100):
        feature = _random_query(rng, library["spectra"])
        feat_mz = feature.precursor_mz
        q_mz = np.asarray(feature.ms2_mz, dtype=np.float64)
        q_int = np.asarray(feature.ms2_intensity, dtype=np.float64)
        q_sorted, table, TQ, SQ0 = _wdp_inputs(feature)
        key0 = int(round(feat_mz))
        for key in (key0 - 1, key0, key0 + 1):
            bucket = csr.get(key)
            if bucket is None:
                continue
            sel = np.arange(bucket["spec_idx"].size)
            ub = _wdp_upper_bound(bucket, sel, q_sorted, table, TQ, SQ0,
                                  feat_mz, mz_tol)
            for k, idx in enumerate(bucket["spec_idx"].tolist()):
                spec = library["spectra"][idx]
                r_mz = np.asarray(spec["mz"], dtype=np.float64)
                r_int = np.asarray(spec["intensity"], dtype=np.float64)
                ref_pmz = spec["_precursor_mz"]
                breakdown = composite_similarity_breakdown(
                    None, None, mz_tol,
                    precursor_query=feat_mz, precursor_ref=ref_pmz,
                    q_arrays=(q_mz, q_int), r_arrays=(r_mz, r_int),
                )
                assert ub[k] >= breakdown.wdp - 1e-12, (
                    f"UB {ub[k]} < true wdp {breakdown.wdp} "
                    f"(feat_mz={feat_mz}, ref_pmz={ref_pmz})"
                )
                n_pairs += 1
                if np.isfinite(ub[k]) and breakdown.wdp > 0.0:
                    n_informative += 1
                shift = ref_pmz - feat_mz
                n_shift_pos += shift > 0
                n_shift_neg += shift < 0
                direct = _greedy_match_arrays(q_mz, q_int, r_mz, r_int, mz_tol, 0.0)
                nl = _greedy_match_arrays(q_mz, q_int, r_mz, r_int, mz_tol, shift)
                if len(nl) > len(direct):
                    n_nl_wins += 1

    assert n_pairs > 1000, f"too few pairs exercised: {n_pairs}"
    assert n_informative > 200, f"bound was vacuous (+inf or wdp=0) too often: {n_informative}"
    assert n_nl_wins > 0, "no neutral-loss-winning pair exercised"
    assert n_shift_pos > 0 and n_shift_neg > 0, (
        f"shift sign coverage: +{n_shift_pos} -{n_shift_neg}")


def test_wdp_upper_bound_keeps_candidate_when_denominator_bound_collapses():
    """A large neutral-loss shift drives ``SQ_lb`` non-positive: the bound stops
    being a bound and the candidate must survive to the exact scorer (+inf), not
    be scored against a clamped denominator."""
    spectra = [{
        "mz": np.array([50.0, 52.0, 54.0]),
        "intensity": np.array([1000.0, 500.0, 250.0]),
        "metadata": {"precursor_mz": 300.0, "name": "far"},
    }]
    library = build_index_from_list(spectra)
    bucket = library["csr"][300]
    feature = _candidate(10.0, [(50.0, 1000.0), (52.0, 500.0), (54.0, 250.0)])
    q_sorted, table, TQ, SQ0 = _wdp_inputs(feature)

    # SQ_lb = TQ - (s + tol)/2 * SQ0 with s = 290 and mean query m/z ~ 51 -> negative.
    assert TQ - 0.5 * (290.0 + 0.02) * SQ0 < 0.0
    ub = _wdp_upper_bound(bucket, np.array([0]), q_sorted, table, TQ, SQ0,
                          feature.precursor_mz, 0.02)
    assert np.isinf(ub[0])

    # ...and the direct frame alone would have thrown the (real) hit away.
    direct_only = _wdp_upper_bound(bucket, np.array([0]), q_sorted, table, TQ, SQ0,
                                   0.0, 0.02)  # feat_mz = 0 disables the NL frame
    assert np.isfinite(direct_only[0])


def test_flat_sqrt_nr_rounds_up():
    """float32 halves the per-peak cost, but only rounding *up* keeps ``ub_cov`` an
    upper bound: a value stored one ulp low silently drops real hits."""
    rng = np.random.default_rng(11)
    spectra = _wdp_library(rng, 50)
    library = build_index_from_list(spectra)
    n_checked = 0
    for bucket in library["csr"].values():
        stored = bucket["flat_sqrt_nr"]
        assert stored.dtype == np.float32
        for k, idx in enumerate(bucket["spec_idx"].tolist()):
            inten = np.asarray(library["spectra"][idx]["intensity"], dtype=np.float64)
            exact = np.sqrt(inten / inten.max())
            lo, hi = bucket["offsets"][k], bucket["offsets"][k + 1]
            assert np.all(stored[lo:hi].astype(np.float64) >= exact)
            n_checked += exact.size
    assert n_checked > 500


def test_round4_collision_disables_the_bound():
    """``_three_scores_core`` drops an unmatched peak whose ``round(mz, 4)`` equals a
    matched peak's, shrinking the true WDP denominator below the model the bound
    assumes. Such a spectrum must opt out (+inf) -- otherwise a genuine hit is lost."""
    # Two reference peaks 5e-5 apart: one matches the query, the other vanishes.
    spectra = [{
        "mz": np.array([300.00000, 300.00005, 500.0]),
        "intensity": np.array([1000.0, 1000.0, 1.0]),
        "metadata": {"precursor_mz": 600.0, "name": "collide"},
    }]
    library = build_index_from_list(spectra)
    bucket = library["csr"][600]
    assert not bucket["wdp_bound_ok"][0]

    feature = _candidate(600.0, [(300.0, 1000.0)])
    q_sorted, table, TQ, SQ0 = _wdp_inputs(feature)
    ub = _wdp_upper_bound(bucket, np.array([0]), q_sorted, table, TQ, SQ0, 600.0, 0.02)
    assert np.isinf(ub[0])

    # The guard is load-bearing: the true wdp really does exceed what the model
    # (which sums *every* reference peak into SR) can bound.
    breakdown = composite_similarity_breakdown(
        None, None, 0.02, precursor_query=600.0, precursor_ref=600.0,
        q_arrays=(np.array([300.0]), np.array([1000.0])),
        r_arrays=(np.asarray(spectra[0]["mz"]), np.asarray(spectra[0]["intensity"])),
    )
    modelled = 300.0 ** 2 / (300.0 * bucket["TR"][0])
    assert breakdown.wdp > modelled + 0.1

    # A spectrum with ordinary peak spacing keeps the bound switched on.
    ok_bucket = build_index_from_list([{
        "mz": np.array([300.0, 300.001, 500.0]),
        "intensity": np.array([1000.0, 1000.0, 1.0]),
        "metadata": {"precursor_mz": 600.0, "name": "fine"},
    }])["csr"][600]
    assert ok_bucket["wdp_bound_ok"][0]


def test_min_wdp_zero_is_a_no_op(monkeypatch):
    """``AnnotationConfig()`` (and GC-MS) leave ``min_wdp`` at 0.0. The screen must
    then be skipped outright, not merely pass everything."""
    rng = np.random.default_rng(555)
    spectra = _wdp_library(rng, 120)
    library = build_index_from_list(spectra)
    ann = AnnotationConfig(similarity_threshold=0.0, min_matched_peaks=1,
                           min_peaks_to_match=1, min_matched_pct=0.25)
    assert ann.min_wdp == 0.0
    sim = SimilarityConfig(mz_tolerance=0.02, ms1_tolerance=0.01)

    def _boom(*a, **k):
        raise AssertionError("_wdp_upper_bound must not run when min_wdp <= 0")

    monkeypatch.setattr(library_mod, "_wdp_upper_bound", _boom)
    for _ in range(40):
        feature = _random_query(rng, library["spectra"])
        got = match_feature_topn(feature, library["spectra"], library["index"],
                                 ann, sim, top_n=5, csr=library["csr"])
        expected = _oracle_match_feature_topn(
            feature, library["spectra"], library["index"], ann, sim, top_n=5)
        _assert_matches_equal(got, expected)


def test_wdp_screen_equals_oracle_on_wide_library():
    """End-to-end: with ASFAM's ``min_wdp = 0.10`` the screen kills most candidates.
    The surviving hits must still be the oracle's, field for field."""
    rng = np.random.default_rng(31337)
    spectra = _wdp_library(rng, 300)
    library = build_index_from_list(spectra)
    ann = AnnotationConfig(similarity_threshold=0.0, min_matched_peaks=1,
                           min_peaks_to_match=1, min_matched_pct=0.25, min_wdp=0.10)
    sim = SimilarityConfig(mz_tolerance=0.02, ms1_tolerance=0.01)

    n_with_hits = 0
    for _ in range(200):
        feature = _random_query(rng, library["spectra"])
        got = match_feature_topn(feature, library["spectra"], library["index"],
                                 ann, sim, top_n=5, csr=library["csr"])
        expected = _oracle_match_feature_topn(
            feature, library["spectra"], library["index"], ann, sim, top_n=5)
        _assert_matches_equal(got, expected)
        n_with_hits += bool(got)
    assert n_with_hits > 20, f"oracle comparison was mostly vacuous: {n_with_hits}"


def test_lazy_csr_carries_the_wdp_fields():
    """`csr=None` rebuilds the bucket through the same builder, so the WDP screen
    must behave identically -- including under DDA's gates."""
    rng = np.random.default_rng(808)
    spectra = _wdp_library(rng, 150)
    library = build_index_from_list(spectra)
    lazy = _build_bucket_csr(library["spectra"], library["index"][400])
    prebuilt = library["csr"][400]
    for key in ("flat_sqrt_nr", "TR", "SR0", "wdp_bound_ok"):
        assert np.array_equal(lazy[key], prebuilt[key]), key

    ann = AnnotationConfig(similarity_threshold=0.0, min_matched_peaks=3,
                           min_peaks_to_match=2, min_matched_pct=0.25, min_wdp=0.10)
    sim = SimilarityConfig(mz_tolerance=0.02, ms1_tolerance=0.01)
    for _ in range(40):
        feature = _random_query(rng, library["spectra"])
        with_csr = match_feature_topn(feature, library["spectra"], library["index"],
                                      ann, sim, top_n=5, csr=library["csr"])
        without = match_feature_topn(feature, library["spectra"], library["index"],
                                     ann, sim, top_n=5)
        _assert_matches_equal(without, with_csr)


def test_sparse_range_max_table_matches_brute_force():
    rng = np.random.default_rng(2)
    for n in (1, 2, 3, 7, 8, 9, 64, 100):
        a = rng.uniform(0.0, 1.0, n)
        rows = _sparse_range_max_table(a)
        lo = rng.integers(0, n, 50)
        hi = np.minimum(lo + rng.integers(1, n + 1, 50), n)
        got = _range_max(rows, lo, hi)
        want = np.array([a[i:j].max() for i, j in zip(lo, hi)])
        assert np.array_equal(got, want), n
