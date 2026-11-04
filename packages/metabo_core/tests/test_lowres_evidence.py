"""Tier-3 corroboration reads the stored window centroid, whichever path
supplied the reported precursor.

Before v1.4.2610xx every fragment-only feature reported the isolation-window
centroid *as* its precursor (``mz_source="ms1_lowres"``), and tier 3 corroborated
with ``ms1_precursor_mz``. Now the neutral-loss consensus path reports its own
value and the centroid lives in ``ms1_window_mz``; the corroboration must keep
using the centroid — the same measurement as before — and must never use the
consensus m/z, which is computed from the very fragments the cosine weighs.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from metabo_core.alignment.cluster import observation_of
from metabo_core.alignment.identity import (
    LOWRES_SOURCES, WINDOW_SOURCE, lowres_evidence,
)
from metabo_core.models import CandidateFeature


def _peak(**fields):
    base = dict(mz_source="", ms1_precursor_mz=None, ms1_window_mz=None)
    base.update(fields)
    return SimpleNamespace(**base)


def test_window_source_is_an_accepted_lowres_source():
    assert WINDOW_SOURCE in LOWRES_SOURCES
    assert "ms1_lowres" in LOWRES_SOURCES and "ms1_relaxed" in LOWRES_SOURCES


def test_a_lowres_precursor_corroborates_as_before():
    mz, source = lowres_evidence(_peak(mz_source="ms1_lowres", ms1_precursor_mz=300.4,
                                       ms1_window_mz=300.4))
    assert (mz, source) == (300.4, "ms1_lowres")
    mz, source = lowres_evidence(_peak(mz_source="ms1_relaxed", ms1_precursor_mz=300.41))
    assert (mz, source) == (300.41, "ms1_relaxed")


def test_a_consensus_precursor_corroborates_through_the_stored_centroid_only():
    peak = _peak(mz_source="nl_consensus", inferred_mz=300.1234, ms1_window_mz=300.4)
    mz, source = lowres_evidence(peak)
    assert source == WINDOW_SOURCE
    assert mz == 300.4                      # the centroid, not 300.1234


def test_no_centroid_means_no_corroboration():
    assert lowres_evidence(_peak(mz_source="nl_consensus", inferred_mz=300.1)) == (None, "")
    assert lowres_evidence(_peak()) == (None, "")
    # A detected MS1 peak is the precursor side of tier 3, never the
    # corroborating side; its centroid (when stored) is still reported, harmlessly.
    assert lowres_evidence(_peak(mz_source="ms1_peak", ms1_precursor_mz=300.2)) == (None, "")


def test_the_cluster_observation_adapter_carries_the_window_centroid():
    peak = CandidateFeature(
        feature_id="F1", segment_name="300-301", replicate_id=1,
        precursor_mz_nominal=300, rt_apex=5.0, rt_left=4.9, rt_right=5.1,
        ms2_mz=np.array([100.0, 150.0, 200.0]),
        ms2_intensity=np.array([1000.0, 800.0, 600.0]), n_fragments=3,
        signal_type="ms2_only", ms2_rep_ion_mz=100.0,
        mz_source="nl_consensus", mz_confidence="high", inferred_mz=300.1234,
        nl_n_support=3, ms1_window_mz=300.4,
    )
    obs = observation_of(peak, lambda: [(100.0, 1000.0), (150.0, 800.0), (200.0, 600.0)])
    assert not obs.has_precursor                     # fragment side of tier 3
    assert obs.lowres_mz == 300.4
    assert obs.lowres_source == WINDOW_SOURCE
