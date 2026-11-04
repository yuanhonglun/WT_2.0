"""Tier 3's containment test judges against the recorded isolation-window centre.

An observation's integer ``window`` is floor(target). Above m/z ~800 the
instrument's targets sit at X.5, so the range it isolated is [N, N+1] and not
[N-0.5, N+0.5]. Until v1.4.261103 containment was judged on the integer plus
``eps_win``, which refused every precursor-vs-fragment pair whose precursor sat
in the upper 0.5 - eps_win of such a window and accepted precursors from the
lower neighbour's range. With ``window_mz`` recorded, the rule uses the true
centre; without it, the integer window stands as before.
"""
from __future__ import annotations

from metabo_core.alignment.identity import (
    IdentityObservation,
    IdentityState,
    identity,
    static_spectrum,
)

RICH = [(100.0, 1000.0), (150.0, 700.0), (200.0, 300.0), (250.0, 120.0)]
WINDOW = 916            # floor of the recorded target
TARGET = 916.5          # isolated range [916.0, 917.0]; with eps_win 0.2: [915.8, 917.2]


def _precursor_side(mz: float) -> IdentityObservation:
    return IdentityObservation(
        rt_apex=5.0, n_fragments=len(RICH), read_spectrum=static_spectrum(RICH),
        ms1_quant_mz=mz, ms2_rep_ion_mz=None, window=WINDOW, segment="915-944",
    )


def _fragment_side(window_mz) -> IdentityObservation:
    return IdentityObservation(
        rt_apex=5.0, n_fragments=len(RICH), read_spectrum=static_spectrum(RICH),
        ms1_quant_mz=None, ms2_rep_ion_mz=150.0, window=WINDOW,
        window_mz=window_mz, segment="915-944",
    )


def test_precursor_in_the_upper_half_of_a_half_integer_window_is_contained():
    # 916.85 is inside the isolated range but beyond the integer rule's reach
    # (916 + 0.5 + 0.2 = 916.7).
    verdict = identity(_precursor_side(916.85), _fragment_side(TARGET))
    assert verdict.tier == 3
    assert verdict.state is IdentityState.SAME


def test_without_a_recorded_centre_the_integer_window_still_rules():
    verdict = identity(_precursor_side(916.85), _fragment_side(None))
    assert verdict.tier == 3
    assert verdict.state is IdentityState.UNJUDGEABLE


def test_a_precursor_the_window_never_isolated_is_refused():
    # 915.4 lies in the lower neighbour's range: accepted by the integer rule
    # (916 - 0.7 = 915.3), refused once the recorded centre is known
    # (916.5 - 0.7 = 915.8).
    assert identity(_precursor_side(915.4), _fragment_side(None)).state is IdentityState.SAME
    assert identity(_precursor_side(915.4), _fragment_side(TARGET)).state is IdentityState.UNJUDGEABLE
