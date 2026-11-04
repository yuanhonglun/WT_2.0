"""Tests for :func:`metabo_core.algorithms.similarity.ei_similarity`.

Reference: MS-DIAL ``GetIntegratedSpectraSimilarity``
(``Common/CommonStandard/Algorithm/Scoring/MsScanMatching.cs:556-562``), the
spectral half of the GC-MS alignment gate.
"""
from __future__ import annotations

import math

import pytest

from metabo_core.algorithms.similarity import (
    composite_similarity_breakdown,
    ei_similarity,
)


# A plausible unit-mass EI spectrum. Six peaks so the MS-DIAL peak-count
# penalty (Stein 1999; 1/2/3/<=5 peaks -> 0.75/0.88/0.94/0.97) is 1.0 and the
# self-similarity lands exactly on 1.0. The penalty gets its own test below.
SPEC_A = [
    (43.0, 999.0), (58.0, 420.0), (71.0, 310.0),
    (91.0, 180.0), (105.0, 90.0), (121.0, 55.0),
]
# Same compound, different sample: intensities drift, m/z grid identical.
SPEC_A_PRIME = [
    (43.0, 980.0), (58.0, 455.0), (71.0, 290.0),
    (91.0, 205.0), (105.0, 78.0), (121.0, 61.0),
]
# A different compound sharing nothing.
SPEC_B = [(147.0, 999.0), (163.0, 500.0), (191.0, 260.0), (207.0, 120.0)]


def test_self_similarity_is_one() -> None:
    r = ei_similarity(SPEC_A, SPEC_A)
    assert r.ei == pytest.approx(1.0, abs=1e-9)
    assert r.matched_pct == pytest.approx(1.0)
    assert r.n_matched_sig == 6


def test_short_reference_spectrum_takes_the_peak_count_penalty() -> None:
    """MS-DIAL penalises dot products computed against few reference peaks, so
    even a perfect self-match on a 5-peak spectrum scores below 1.0. Worth
    pinning: it is the reason a sparse deconvolution result cannot reach the
    top of the EI scale no matter how well it matches."""
    short = SPEC_A[:5]
    r = ei_similarity(short, short)
    assert r.ei < 1.0
    # 0.97 penalty on both squared dot products, Matched% untouched.
    expected = (3.0 * math.sqrt(0.97) + 2.0 * math.sqrt(0.97) + 1.0) / 6.0
    assert r.ei == pytest.approx(expected)


def test_same_compound_across_samples_scores_high() -> None:
    """The case the GC-MS joiner must let through: same compound, drifted
    intensities. Must clear the 0.5 EI gate comfortably."""
    r = ei_similarity(SPEC_A, SPEC_A_PRIME)
    assert r.ei > 0.9


def test_disjoint_spectra_score_zero() -> None:
    r = ei_similarity(SPEC_A, SPEC_B)
    assert r.ei == pytest.approx(0.0, abs=1e-12)
    assert r.n_matched_sig == 0


def test_missing_or_empty_spectrum_returns_minus_one() -> None:
    """MS-DIAL ``IsComparedAvailable`` equivalent — the value must be negative
    so ``ei > threshold`` is False for every legal threshold, and must not be
    0.0 (that already means "compared, and nothing matched")."""
    for bad in (None, []):
        assert ei_similarity(bad, SPEC_A).ei == -1.0
        assert ei_similarity(SPEC_A, bad).ei == -1.0
    assert ei_similarity(None, None).ei == -1.0
    # Distinguishable from the disjoint case above.
    assert ei_similarity(SPEC_A, SPEC_B).ei == 0.0


def test_is_asymmetric_when_one_spectrum_is_a_subset() -> None:
    """RDP and Matched% are computed against the *reference* end, so swapping
    the arguments is a different question. Getting this backwards in the joiner
    silently changes the gate."""
    subset = SPEC_A[:2]                    # 43 and 58 only
    forward = ei_similarity(SPEC_A, subset)     # query = full, ref = subset
    backward = ei_similarity(subset, SPEC_A)    # query = subset, ref = full

    assert forward.ei != pytest.approx(backward.ei)
    # Every significant reference peak is present in the full query...
    assert forward.matched_pct == pytest.approx(1.0)
    # ...but the subset query only covers part of the full reference.
    assert backward.matched_pct < 1.0
    assert forward.ei > backward.ei


def test_formula_matches_its_components() -> None:
    """ei is exactly the MS-DIAL 3/2/1 weighting of sqrt(WDP), sqrt(RDP) and
    Matched% — not, say, the composite library-match score."""
    r = ei_similarity(SPEC_A, SPEC_A_PRIME)
    expected = (
        3.0 * math.sqrt(r.wdp) + 2.0 * math.sqrt(r.rdp) + r.matched_pct
    ) / 6.0
    assert r.ei == pytest.approx(expected)


def test_components_come_from_composite_breakdown() -> None:
    """ei_similarity must not re-derive the dot products; it reuses the same
    matching path as library annotation so the two never drift apart."""
    r = ei_similarity(SPEC_A, SPEC_A_PRIME, mz_tolerance=0.5)
    c = composite_similarity_breakdown(SPEC_A, SPEC_A_PRIME, mz_tolerance=0.5)
    assert (r.wdp, r.sdp, r.rdp) == (c.wdp, c.sdp, c.rdp)
    assert r.matched_pct == c.matched_pct
    assert r.n_matched_sig == c.n_matched_sig


def test_n_matched_sig_counts_significant_reference_peaks_only() -> None:
    """The strict branch of the alignment gate needs a MatchedPeaksCount >= 3
    check, so the count must exclude reference peaks below 1% of the ref base
    peak — those are noise and MS-DIAL does not count them."""
    ref = [(43.0, 1000.0), (58.0, 500.0), (71.0, 300.0), (300.0, 5.0)]
    query = [(43.0, 1000.0), (58.0, 500.0), (71.0, 300.0), (300.0, 900.0)]
    r = ei_similarity(query, ref)
    # 300.0 sits at 0.5% of the reference base peak: matched, but not counted.
    assert r.n_matched_sig == 3


def test_composite_similarity_result_gained_n_matched_sig() -> None:
    """Appending the field must not disturb the existing positional layout —
    ASFAM and DDA both unpack this NamedTuple."""
    c = composite_similarity_breakdown(SPEC_A, SPEC_A_PRIME, mz_tolerance=0.5)
    assert c[0] == c.score
    assert c[1] == c.n_matched
    assert c[2] == c.wdp
    assert c[3] == c.sdp
    assert c[4] == c.rdp
    assert c[5] == c.matched_pct
    assert c[6] == c.total_score
    assert c[7] == c.n_matched_sig
