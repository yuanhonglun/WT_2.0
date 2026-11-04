"""The four-tier, route-agnostic identity rule (SPEC 2026-08-02 §6).

One positive, one negative and one abstention per tier, plus the properties the
rule exists to guarantee: that the dispatch never looks at a quantitation route,
that an abstention never merges, and that the isolation-window centroid can only
ever relax a threshold.
"""
from __future__ import annotations

import pytest

from metabo_core.alignment.identity import (
    IdentityObservation,
    IdentityState,
    identity,
    static_spectrum,
)
from metabo_core.config import IdentityConfig

RICH = [(100.0, 1000.0), (150.0, 700.0), (200.0, 300.0), (250.0, 120.0)]
#: Same four ions as RICH, intensity order reversed: matchable, and cosine 0.19
#: -- below theta_veto, so it can contradict even a shared precursor.
CONTRADICT = [(100.0, 2.0), (150.0, 20.0), (200.0, 400.0), (250.0, 1000.0)]
#: Cosine 0.38: between theta_veto (0.2) and theta_same (0.7). Passes tier 1's
#: veto, fails tier 2's permit -- the pair that shows the two are different
#: questions.
LUKEWARM = [(100.0, 60.0), (150.0, 300.0), (200.0, 900.0), (250.0, 1000.0)]
#: Cosine 0.77: between theta_cross_supported (0.7) and theta_cross (0.8), so
#: only the window centroid's corroboration can carry it.
SUPPORTED_BAND = [(100.0, 300.0), (150.0, 1000.0), (200.0, 700.0), (250.0, 250.0)]
#: One fragment: below n_min, so no cosine is evidence however it scores.
SPARSE = [(100.0, 1000.0)]


def _obs(
    *,
    mz=None,
    rep=None,
    spectrum=RICH,
    rt=5.0,
    window=300,
    segment="300-301",
    lowres=None,
    lowres_source="",
    n_fragments=None,
) -> IdentityObservation:
    return IdentityObservation(
        rt_apex=rt,
        n_fragments=len(spectrum) if n_fragments is None else n_fragments,
        read_spectrum=static_spectrum(spectrum),
        ms1_quant_mz=mz,
        ms2_rep_ion_mz=rep,
        window=window,
        segment=segment,
        lowres_mz=lowres,
        lowres_source=lowres_source,
    )


def _cos(left, right) -> float:
    return identity(_obs(mz=300.5, spectrum=left), _obs(mz=300.5, spectrum=right)).cosine


def test_fixtures_sit_in_the_bands_the_tests_below_assume():
    # Every negative in this file is "the cosine landed in band X". If a change
    # to the similarity kernel moves a fixture out of its band, those tests would
    # keep passing while testing nothing -- so pin the bands here.
    config = IdentityConfig()
    assert _cos(RICH, CONTRADICT) < config.theta_veto
    assert config.theta_veto < _cos(RICH, LUKEWARM) < config.theta_same
    assert (
        config.theta_cross_supported
        <= _cos(RICH, SUPPORTED_BAND)
        < config.theta_cross
    )


# --- tier 1: both sides carry an exact precursor ---------------------------

def test_tier1_same_precursor_and_agreeing_spectra_is_same():
    verdict = identity(_obs(mz=300.5000), _obs(mz=300.5100))
    assert (verdict.state, verdict.tier) == (IdentityState.SAME, 1)


def test_tier1_vetoes_only_when_the_spectra_actively_disagree():
    # A shared quantitation ion is strong evidence, so the spectra get a veto
    # rather than a vote: they must fall below theta_veto, not merely below
    # theta_same.
    lukewarm = identity(_obs(mz=300.5, spectrum=RICH),
                        _obs(mz=300.5, spectrum=LUKEWARM))
    assert lukewarm.state is IdentityState.SAME

    contradicted = identity(_obs(mz=300.5, spectrum=RICH),
                            _obs(mz=300.5, spectrum=CONTRADICT))
    assert (contradicted.state, contradicted.tier) == (IdentityState.DIFFERENT, 1)


def test_tier1_different_precursors_is_unjudgeable_however_alike_the_spectra():
    # Different exact masses means different formulae; identical spectra then say
    # isomer or homologue, which is not a reason to merge quantitation.
    verdict = identity(_obs(mz=300.50, spectrum=RICH),
                       _obs(mz=300.62, spectrum=RICH))
    assert (verdict.state, verdict.tier) == (IdentityState.UNJUDGEABLE, 1)


def test_tier1_sparse_spectra_cannot_veto():
    verdict = identity(_obs(mz=300.5, spectrum=SPARSE),
                       _obs(mz=300.5, spectrum=CONTRADICT))
    assert verdict.state is IdentityState.SAME


# --- tier 2: neither side has a precursor ----------------------------------

def test_tier2_agreeing_spectra_is_same():
    verdict = identity(_obs(rep=150.0), _obs(rep=150.0))
    assert (verdict.state, verdict.tier) == (IdentityState.SAME, 2)


def test_tier2_is_a_permit_not_a_veto():
    # With no precursor the spectra are the entire case, so the bar is
    # theta_same (0.7) and a lukewarm cosine is not enough -- the opposite of
    # tier 1, where the same cosine passes.
    verdict = identity(_obs(rep=150.0, spectrum=RICH),
                       _obs(rep=150.0, spectrum=LUKEWARM))
    assert (verdict.state, verdict.tier) == (IdentityState.DIFFERENT, 2)


def test_tier2_falls_back_to_the_representative_fragment_only_when_blind():
    same_ion = identity(_obs(rep=150.0, spectrum=SPARSE),
                        _obs(rep=150.005, spectrum=SPARSE))
    assert (same_ion.state, same_ion.weak) == (IdentityState.SAME, True)

    other_ion = identity(_obs(rep=150.0, spectrum=SPARSE),
                         _obs(rep=180.0, spectrum=SPARSE))
    assert (other_ion.state, other_ion.tier) == (IdentityState.UNJUDGEABLE, 2)


def test_tier2_fragment_fallback_is_last_resort_not_a_key():
    # Judgeable spectra decide, even when the two rows were quantified on the
    # same fragment. The old joiner keyed on that fragment, so this pair used to
    # be one row unconditionally.
    verdict = identity(_obs(rep=150.0, spectrum=RICH),
                       _obs(rep=150.0, spectrum=CONTRADICT))
    assert verdict.state is IdentityState.DIFFERENT
    assert not verdict.weak


# --- tier 3: one side has a precursor, the other only fragments -------------

def test_tier3_precursor_and_fragment_views_of_one_compound_are_same():
    verdict = identity(
        _obs(mz=300.5, spectrum=RICH),
        _obs(rep=150.0, spectrum=RICH),
    )
    assert (verdict.state, verdict.tier) == (IdentityState.SAME, 3)


def test_tier3_dispatch_ignores_argument_order():
    left, right = _obs(mz=300.5, spectrum=RICH), _obs(rep=150.0, spectrum=RICH)
    assert identity(left, right) == identity(right, left)


def test_tier3_needs_more_than_a_high_cosine():
    # Two things in one isolation window at one RT deconvolve from the same
    # co-eluting mixture, so a high cosine alone can mean no more than "these
    # co-elute". The anchor is what tests "this fragment row IS that compound's
    # fragment view". Both pairs below score a *perfect* cosine -- identical
    # spectra -- and only the anchor separates them.
    with_trace = RICH + [(180.0, 20.0)]        # 180.0 at 2% of the base peak
    precursor = _obs(mz=300.5, spectrum=with_trace)

    anchored = identity(precursor, _obs(rep=250.0, spectrum=with_trace))
    assert anchored.state is IdentityState.SAME
    assert anchored.cosine == pytest.approx(1.0)

    unanchored = identity(precursor, _obs(rep=180.0, spectrum=with_trace))
    assert (unanchored.state, unanchored.tier) == (IdentityState.UNJUDGEABLE, 3)
    assert unanchored.cosine == pytest.approx(1.0)


def test_tier3_rejects_a_precursor_outside_the_fragment_sides_window():
    # Free and decisive: the isolation window is an acquisition fact.
    verdict = identity(
        _obs(mz=300.5, spectrum=RICH, window=300),
        _obs(rep=150.0, spectrum=RICH, window=302, segment=""),
    )
    assert verdict.state is IdentityState.UNJUDGEABLE


def test_tier3_rejects_a_fragment_heavier_than_the_precursor():
    heavy = [(400.0, 1000.0), (150.0, 700.0), (200.0, 300.0)]
    verdict = identity(
        _obs(mz=300.5, spectrum=heavy),
        _obs(rep=400.0, spectrum=heavy),
    )
    assert verdict.state is IdentityState.UNJUDGEABLE


def test_tier3_lukewarm_cosine_does_not_merge():
    verdict = identity(_obs(mz=300.5, spectrum=RICH),
                       _obs(rep=150.0, spectrum=LUKEWARM))
    assert verdict.state is IdentityState.UNJUDGEABLE


# --- the window centroid may relax, never reject ---------------------------

@pytest.mark.parametrize("source", ["ms1_lowres", "ms1_relaxed"])
def test_window_centroid_relaxes_the_cross_threshold(source):
    # SUPPORTED_BAND sits below theta_cross (0.8) but above
    # theta_cross_supported (0.7), so the centroid's corroboration is exactly
    # what decides it.
    precursor = _obs(mz=300.5, spectrum=RICH)
    unsupported = identity(precursor, _obs(rep=150.0, spectrum=SUPPORTED_BAND))
    supported = identity(
        precursor,
        _obs(rep=150.0, spectrum=SUPPORTED_BAND, lowres=300.4,
             lowres_source=source),
    )
    assert unsupported.state is IdentityState.UNJUDGEABLE
    assert supported.state is IdentityState.SAME
    assert supported.support["lowres_source"] == source


def test_a_distant_window_centroid_never_vetoes():
    # It is an intensity-weighted mixture centroid: on a low-abundance compound
    # it reports whatever co-isolated ion dominates the window. Making it a hard
    # gate would drop exactly the features all-ion acquisition exists to find.
    verdict = identity(
        _obs(mz=300.5, spectrum=RICH),
        _obs(rep=150.0, spectrum=RICH, lowres=300.9, lowres_source="ms1_lowres"),
    )
    assert verdict.state is IdentityState.SAME
    assert verdict.support["lowres_mz_delta"] > IdentityConfig().lowres_tol


def test_inferred_precursors_are_not_reachable_from_the_evidence_record():
    # A precursor inferred from a library hit would close the loop
    # annotate -> align -> pick a representative -> annotate. The record has no
    # field for it, and this is the test that says so on purpose.
    assert not any(
        "inferred" in name
        for name in IdentityObservation.__dataclass_fields__
    )


# --- geometry gates --------------------------------------------------------

def test_geometry_gates_reject_a_foreign_segment():
    verdict = identity(
        _obs(mz=300.5, segment="300-301"),
        _obs(mz=300.5, segment="400-401"),
    )
    assert (verdict.state, verdict.tier) == (IdentityState.UNJUDGEABLE, 4)


def test_geometry_gates_allow_one_window_of_slack():
    adjacent = identity(_obs(mz=300.5, window=300), _obs(mz=300.5, window=301))
    assert adjacent.state is IdentityState.SAME

    distant = identity(_obs(mz=300.5, window=300), _obs(mz=300.5, window=302))
    assert (distant.state, distant.tier) == (IdentityState.UNJUDGEABLE, 4)


def test_cross_tier_gets_the_wider_rt_window():
    # A fragment chromatogram and its precursor's do not peak at the same time,
    # so tier 3 reaches further in RT than tiers 1 and 2 do.
    config = IdentityConfig()
    drt = 0.5 * (config.rt_window + config.rt_window_cross)

    same_tier = identity(_obs(mz=300.5, rt=5.0), _obs(mz=300.5, rt=5.0 + drt))
    assert same_tier.state is IdentityState.UNJUDGEABLE

    cross_tier = identity(
        _obs(mz=300.5, rt=5.0, spectrum=RICH),
        _obs(rep=150.0, rt=5.0 + drt, spectrum=RICH),
    )
    assert cross_tier.state is IdentityState.SAME


def test_check_geometry_false_leaves_the_gates_to_the_caller():
    # The joiner's three RT paths are wider than rt_window; swapping the identity
    # rule underneath them must not silently narrow them.
    far = identity(
        _obs(mz=300.5, rt=5.0), _obs(mz=300.5, rt=9.0), check_geometry=False,
    )
    assert far.state is IdentityState.SAME
