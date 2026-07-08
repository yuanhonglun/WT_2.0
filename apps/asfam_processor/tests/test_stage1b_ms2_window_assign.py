"""Stage 1b MS1-driven feature → MS2 isolation-window assignment.

Regression coverage for R1_no_ms2 (2026-07-05): high-m/z ASFAM DIA windows
have 1-Da isolation targets sitting at X.5, so ``int(round(target))`` collapsed
adjacent windows via banker's rounding (silently dropping ~half the MS2 above
m/z ~800), and ``int(round(precursor))`` missed the surviving windows entirely.
The reader now keys windows by ``floor(target)`` (injective) and stage1b assigns
each feature to the acquired window whose target is nearest its precursor m/z.
"""
import math

import numpy as np

from asfam.models import RawSegmentData
from asfam.pipeline.stage1b_ms1_detection import _assigned_window


def _seg(targets: dict[int, float]) -> RawSegmentData:
    """Minimal RawSegmentData carrying only what _assigned_window reads."""
    return RawSegmentData(
        file_path="mem",
        segment_name="795-824",
        segment_low=795,
        segment_high=824,
        replicate_id=3,
        n_cycles=0,
        rt_array=np.array([], dtype=np.float64),
        precursor_list=sorted(targets),
        cycles=[],
        precursor_targets=targets,
    )


def test_floor_keying_avoids_banker_rounding_collision():
    """X.5 targets: int(round) collides (banker's), int(floor) stays injective."""
    targets = [915.5, 916.5, 917.5, 918.5, 919.5, 920.5]
    round_keys = {int(round(t)) for t in targets}
    floor_keys = {int(math.floor(t)) for t in targets}
    # round() maps 915.5→916, 916.5→916, 917.5→918, 918.5→918, ... — half collapse
    assert len(round_keys) < len(targets)
    # floor() gives one distinct key per window
    assert len(floor_keys) == len(targets)
    assert floor_keys == {915, 916, 917, 918, 919, 920}


def test_assigned_window_picks_nearest_target_for_odd_channel_feature():
    """R1 case: a feature whose round() lands on an unacquired odd channel is
    still assigned to its true window (target X.5, floor-key X)."""
    # High-m/z segment: 1-Da windows with targets at X.5 → floor-keys = X.
    targets = {ch: ch + 0.5 for ch in range(795, 825)}
    seg = _seg(targets)
    # F20101 (811.489): int(round)=811 previously found no ms2_scans[811] under
    # the collision; now maps to window 811 (target 811.5, covers [811,812]).
    assert _assigned_window(seg, 811.489, 0.6) == 811
    assert _assigned_window(seg, 809.194, 0.6) == 809
    assert _assigned_window(seg, 821.211, 0.6) == 821
    # A feature rounding UP to an odd channel (828.53→round 829) still lands in
    # its true window by nearest-target (nearest here would be 828 for [828,829)).
    tgt2 = {ch: ch + 0.5 for ch in range(825, 855)}
    seg2 = _seg(tgt2)
    assert _assigned_window(seg2, 828.534, 0.6) == 828


def test_assigned_window_low_segment_integer_targets():
    """Low-m/z segment with X.0 targets: window centred on the integer; a
    feature below the centre still maps to it (nearest-target, not floor)."""
    targets = {ch: float(ch) for ch in range(75, 105)}  # targets at X.0
    seg = _seg(targets)
    # 90.3 is inside window 90 (covers [89.5, 90.5]); nearest target is 90.0.
    assert _assigned_window(seg, 90.3, 0.6) == 90
    # 89.8 is inside window 90 as well (0.2 from 90.0 vs 0.8 from 89.0).
    assert _assigned_window(seg, 89.8, 0.6) == 90


def test_assigned_window_returns_none_when_no_window_in_tolerance():
    """No acquired window within tol → None (feature was never isolated)."""
    targets = {ch: ch + 0.5 for ch in range(795, 825)}
    seg = _seg(targets)
    # 850.0 is 25 Da away from any acquired target in this segment.
    assert _assigned_window(seg, 850.0, 0.6) is None
    # Empty targets (degenerate / no MS2) → None.
    assert _assigned_window(_seg({}), 811.5, 0.6) is None


def test_assigned_window_tolerance_boundary():
    """Gap just over tol is rejected; just under is accepted."""
    targets = {811: 811.5}
    seg = _seg(targets)
    assert _assigned_window(seg, 811.5 + 0.59, 0.6) == 811   # within tol
    assert _assigned_window(seg, 811.5 + 0.61, 0.6) is None  # beyond tol
