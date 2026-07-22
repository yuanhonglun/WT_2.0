"""``composite_similarity_breakdown``: array path == tuple path, bit for bit.

The library matcher (``match_feature_topn``) feeds the scorer float64 arrays and
passes ``None`` for the tuple lists, so it never materialises ``list(zip(...))``
per candidate. That shortcut is only legitimate if the two paths agree on every
field of ``CompositeSimilarityResult`` exactly -- not to within a tolerance,
because ``features.csv`` equivalence is checked by hash and hits are ranked by
raw ``score`` (a one-ulp drift can reorder a tie).
"""
import numpy as np
import pytest

from metabo_core.algorithms.similarity import composite_similarity_breakdown


def _both(peaks_q, peaks_r, **kw):
    """Score the same spectra through the tuple path and the array path."""
    q_mz = np.asarray([m for m, _ in peaks_q], dtype=np.float64)
    q_int = np.asarray([i for _, i in peaks_q], dtype=np.float64)
    r_mz = np.asarray([m for m, _ in peaks_r], dtype=np.float64)
    r_int = np.asarray([i for _, i in peaks_r], dtype=np.float64)
    tuple_res = composite_similarity_breakdown(peaks_q, peaks_r, **kw)
    # Lists deliberately omitted: the array path must not read them.
    array_res = composite_similarity_breakdown(
        None, None, q_arrays=(q_mz, q_int), r_arrays=(r_mz, r_int), **kw)
    return tuple_res, array_res


def _assert_identical(a, b):
    assert a == b, f"tuple={a} array={b}"
    # NamedTuple __eq__ is field-wise; spell the fields out so a future field
    # addition cannot silently escape the comparison.
    for field in ("score", "n_matched", "wdp", "sdp", "rdp",
                  "matched_pct", "total_score"):
        assert getattr(a, field) == getattr(b, field), field


def _random_spectrum(rng, n, mz_lo=50.0, mz_hi=500.0, zero_intensity=False):
    mz = np.sort(rng.uniform(mz_lo, mz_hi, n))
    inten = np.zeros(n) if zero_intensity else rng.uniform(1.0, 5000.0, n)
    return list(zip(mz.tolist(), inten.tolist()))


def test_random_pairs_bit_identical():
    """1000 random (query, reference) pairs, both frames live."""
    rng = np.random.default_rng(20260708)
    for _ in range(1000):
        n_q = int(rng.integers(1, 40))
        n_r = int(rng.integers(1, 40))
        peaks_q = _random_spectrum(rng, n_q)
        peaks_r = _random_spectrum(rng, n_r)
        # Plant shared peaks so matching is not always empty.
        n_shared = int(rng.integers(0, min(n_q, n_r) + 1))
        for k in range(n_shared):
            peaks_r[k] = (peaks_q[k][0] + float(rng.uniform(-0.01, 0.01)),
                          peaks_r[k][1])
        pq = float(rng.uniform(200.0, 600.0))
        pr = pq + float(rng.choice([0.0, 2.0, -3.5, 18.011]))
        t, a = _both(peaks_q, peaks_r, mz_tolerance=0.02,
                     precursor_query=pq, precursor_ref=pr,
                     rt_query=float(rng.uniform(0.5, 20.0)),
                     rt_ref=float(rng.uniform(0.5, 20.0)),
                     use_rt=bool(rng.integers(0, 2)))
        _assert_identical(t, a)


def test_neutral_loss_frame_wins():
    """The NL frame matches strictly more peaks; both paths must switch to it."""
    peaks_q = [(100.0, 900.0), (150.0, 700.0), (200.0, 500.0), (321.0, 100.0)]
    # ref + 18.0 == query for three peaks; direct frame shares only 321.0
    peaks_r = [(82.0, 900.0), (132.0, 700.0), (182.0, 500.0), (321.0, 100.0)]
    t, a = _both(peaks_q, peaks_r, mz_tolerance=0.02,
                 precursor_query=418.0, precursor_ref=436.0)
    _assert_identical(t, a)
    # NL frame lands 3 (100/150/200); the direct frame only shares 321.0.
    assert t.n_matched == 3


@pytest.mark.parametrize("peaks_q,peaks_r", [
    ([], [(100.0, 1.0)]),                       # empty query
    ([(100.0, 1.0)], []),                       # empty reference
    ([], []),                                   # both empty
    ([(100.0, 1.0)], [(100.0, 1.0)]),           # single peak, exact match
    ([(100.0, 1.0)], [(400.0, 1.0)]),           # single peak, no match
])
def test_degenerate_shapes(peaks_q, peaks_r):
    t, a = _both(peaks_q, peaks_r, mz_tolerance=0.02,
                 precursor_query=300.0, precursor_ref=300.0)
    _assert_identical(t, a)


@pytest.mark.parametrize("q_zero,r_zero", [(True, False), (False, True), (True, True)])
def test_all_zero_intensity(q_zero, r_zero):
    """max_q / max_r below 1e-12 short-circuits; both paths return the same zero."""
    rng = np.random.default_rng(7)
    peaks_q = _random_spectrum(rng, 6, zero_intensity=q_zero)
    peaks_r = _random_spectrum(rng, 6, zero_intensity=r_zero)
    t, a = _both(peaks_q, peaks_r, mz_tolerance=0.02,
                 precursor_query=300.0, precursor_ref=300.0)
    _assert_identical(t, a)
    assert t.score == 0.0 and t.n_matched == 0


def test_n_sig_ref_floor_of_one():
    """A reference whose only significant peak is its own base peak -> denom 1."""
    peaks_q = [(100.0, 1000.0)]
    # 100.0 is the base peak; the rest sit below 1% of it and are insignificant.
    peaks_r = [(100.0, 1000.0), (150.0, 5.0), (200.0, 3.0), (250.0, 1.0)]
    t, a = _both(peaks_q, peaks_r, mz_tolerance=0.02,
                 precursor_query=300.0, precursor_ref=300.0)
    _assert_identical(t, a)
    assert t.matched_pct == 1.0


def test_duplicate_mz_within_rounding_bucket():
    """Two query peaks that round to the same 4-dp m/z.

    ``_three_scores_core`` marks peaks matched by *rounded m/z*, not by index, so
    an unmatched twin of a matched peak is still treated as matched. Both paths
    must reproduce that quirk identically.
    """
    peaks_q = [(100.00001, 900.0), (100.00002, 400.0), (250.0, 100.0)]
    peaks_r = [(100.0, 1000.0), (250.0, 90.0)]
    t, a = _both(peaks_q, peaks_r, mz_tolerance=0.02,
                 precursor_query=300.0, precursor_ref=300.0)
    _assert_identical(t, a)


def test_arrays_ignored_unless_both_supplied():
    """One-sided ``q_arrays`` must fall back to the tuple path, not crash."""
    peaks_q = [(100.0, 1000.0), (150.0, 500.0)]
    peaks_r = [(100.0, 900.0), (150.0, 450.0)]
    q_mz = np.asarray([100.0, 150.0])
    q_int = np.asarray([1000.0, 500.0])
    only_q = composite_similarity_breakdown(
        peaks_q, peaks_r, 0.02, q_arrays=(q_mz, q_int))
    plain = composite_similarity_breakdown(peaks_q, peaks_r, 0.02)
    _assert_identical(plain, only_q)
