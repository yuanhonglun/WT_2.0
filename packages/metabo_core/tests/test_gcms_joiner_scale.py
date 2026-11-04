"""A10 — the join must stay usable at real batch scale.

The three fullscan samples in ``test_data`` carry 315-575 features each, which
never stressed the old ``linear_sum_assignment`` path enough to show its O(n^3)
cost. Spec A10 is stated at 8 samples x ~2000 features, so it needs synthetic
input at that size.

Marked ``slow`` — it builds 16k spectra and is not worth paying for on every
run. ``pytest -m slow`` or ``--deselect`` accordingly.
"""
from __future__ import annotations

import math
import time

import pytest

from metabo_core.alignment.gcms_joiner import (
    GcmsPeak,
    assign_to_master,
    build_master_list,
    promote_unclaimed,
    reset_quant_masses,
)
from metabo_core.config.gcms_alignment import GcmsAlignmentConfig

N_SAMPLES = 8
N_FEATURES = 2000
RUN_MINUTES = 60.0
BUDGET_SECONDS = 60.0


def _spectrum(seed: int) -> tuple[tuple[float, float], ...]:
    """A 12-fragment EI spectrum that is reproducible from an integer and
    distinct enough between compounds that most pairs fail the gate."""
    base = 40 + (seed * 7) % 200
    return tuple(
        (float(base + 13 * k), 1000.0 / (1 + k) * (1 + (seed + k) % 3 * 0.1))
        for k in range(12)
    )


def _peaks(sample: str, jitter: float) -> list[GcmsPeak]:
    out = []
    for i in range(N_FEATURES):
        rt = RUN_MINUTES * (i + 0.5) / N_FEATURES + jitter * math.sin(i)
        spec = _spectrum(i)
        out.append(GcmsPeak(
            sample=sample,
            feature_id=f"F{i:05d}",
            index=rt,
            mz=spec[0][0],
            height=1000.0 + i,
            area=100.0 + i,
            spectrum=spec,
        ))
    return out


@pytest.mark.slow
def test_join_scales_to_eight_samples_of_two_thousand() -> None:
    cfg = GcmsAlignmentConfig()
    peaks_by_sample = {
        f"s{j}": _peaks(f"s{j}", jitter=0.004 * j) for j in range(N_SAMPLES)
    }
    total = N_SAMPLES * N_FEATURES

    t0 = time.perf_counter()
    masters = build_master_list(peaks_by_sample, "s0", cfg)
    spots, unclaimed, stats = assign_to_master(masters, peaks_by_sample, cfg)
    spots = list(spots) + promote_unclaimed(unclaimed, sorted(peaks_by_sample), cfg, stats)
    spots, stats = reset_quant_masses(spots, cfg, stats)
    elapsed = time.perf_counter() - t0

    # The 8 samples hold the same 2000 compounds, so they must collapse to
    # ~2000 rows rather than 16000 — otherwise the timing means nothing.
    assert len(spots) < N_FEATURES * 1.2, len(spots)
    placed = sum(s.n_detected for s in spots)
    assert placed == total, f"{total - placed} peaks lost"
    assert elapsed < BUDGET_SECONDS, f"{elapsed:.1f}s for {total} peaks"
