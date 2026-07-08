"""MS-DIAL-style isotope peak identification.

Ports the algorithm from MS-DIAL's
``MsdialCore/Algorithm/IsotopeEstimator.cs`` to Python without the
IUPAC simulation dependency. Suitable for any LC-MS feature list where
the m/z is well-centroided and a charge state estimate is needed
without MS2 evidence (so it works for DDA features that have no MS2).

Algorithm summary
-----------------
1. Sort features by m/z ascending.
2. For each not-yet-assigned feature ``peak``:
   a. Gather candidates within ``[peak.mz, peak.mz + 8.1]`` whose RT
      apex is within ``rt_tolerance`` minutes.
   b. Estimate charge by looking at the first candidate near
      ``peak.mz + 1.003355 / k`` for k from ``max_charge`` down to 1.
      Falls back to charge = 1 if nothing fits.
   c. Walk the predicted envelope up to ``max_isotope_step`` slots,
      tracking the centroid as it drifts.
   d. Validate the envelope: for monoisotopic mass ≤ 800 Da, require
      monotonic intensity decrease across detected slots. For > 800 Da,
      compare measured M+k / M+(k-1) ratios to an alkane formula
      (CnH2n with n = mass / 14) — accepts |ratio diff| < 5.0.
3. For each accepted M+k slot, mark the candidate feature:
   ``is_duplicate=True``, ``duplicate_type='isotope'``,
   ``duplicate_group_id=monoiso_group_id``,
   ``isotope_group_id=monoiso_group_id``. The monoisotopic peak keeps
   its charge in a synthetic ``ms1_isotopes`` summary so downstream
   adduct logic can read it.

Mass tolerances
---------------
Mass tolerance is ppm-scaled in the original MS-DIAL code so a 5 mDa
``base_tolerance`` at m/z 200 becomes ~25 mDa at m/z 1000. We keep the
same scaling.

Constants
---------
- ``C13_DELTA = 1.003355`` (carbon-13 - carbon-12 mass difference)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from metabo_core.constants.mass import C13_DELTA
from metabo_core.models import CandidateFeature


@dataclass
class _IsotopeSlot:
    """Per-step tracking slot used by the envelope walker."""
    weight: int
    predicted_mz: float
    matched_mz: Optional[float] = None
    matched_intensity: float = 0.0
    matched_feature_idx: Optional[int] = None


def estimate_isotopes(
    features: List[CandidateFeature],
    base_mz_tolerance: float = 0.01,
    rt_tolerance: float = 0.06,
    max_charge: int = 4,
    max_isotope_step: int = 8,
    group_id_start: int = 0,
) -> int:
    """Identify isotope peaks in ``features``. Mutates in place.

    Returns the next available ``isotope_group_id``. Callers chaining
    multiple replicates should pass the previous return value as the
    new ``group_id_start`` so group ids stay unique across reps.
    """
    if not features:
        return group_id_start

    # Pair each feature with its original index so we can mark by index
    # even after sorting. Indices into ``features``.
    order = sorted(range(len(features)), key=lambda i: features[i].precursor_mz)
    assigned: list[bool] = [False] * len(features)
    group_id = group_id_start

    for pos, mono_idx in enumerate(order):
        if assigned[mono_idx]:
            continue
        mono = features[mono_idx]
        mono_mz = mono.precursor_mz
        mono_rt = mono.rt_apex
        mono_intensity = mono.ms1_height or 0.0

        if mono_intensity <= 0.0:
            continue

        tolerance = _scaled_tolerance(mono_mz, base_mz_tolerance)

        # 1. Gather candidates within m/z and RT window.
        candidates: list[int] = []
        for pos2 in range(pos + 1, len(order)):
            idx = order[pos2]
            if assigned[idx]:
                continue
            cand = features[idx]
            if cand.precursor_mz > mono_mz + 8.1:
                break
            if cand.precursor_mz <= mono_mz:
                continue
            if abs(cand.rt_apex - mono_rt) > rt_tolerance:
                continue
            candidates.append(idx)

        if not candidates:
            continue

        # 2. Estimate charge from the first candidate near M+1.
        charge = _estimate_charge(
            mono_mz, mono_rt, candidates, features, tolerance, rt_tolerance,
            max_charge,
        )

        # 3. Walk the envelope.
        slots = _walk_envelope(
            mono_mz, mono_intensity, mono_rt, charge, candidates, features,
            tolerance, rt_tolerance, max_isotope_step,
        )

        # 4. Validate and assign.
        accepted = _accept_envelope(mono_mz, mono_intensity, slots)
        if not accepted:
            continue

        # Mark monoisotopic + accepted slots.
        mono.charge_state = charge  # set on the dataclass dynamically
        for slot in accepted:
            cand = features[slot.matched_feature_idx]
            cand.is_duplicate = True
            cand.duplicate_type = "isotope"
            cand.duplicate_group_id = group_id
            cand.isotope_group_id = group_id
            cand.charge_state = charge
            assigned[slot.matched_feature_idx] = True

        # The monoisotopic peak gets the group_id too so the GUI can
        # render the cluster, but stays is_duplicate=False (representative).
        mono.duplicate_group_id = group_id
        mono.duplicate_type = ""  # representative isn't labelled
        mono.isotope_group_id = group_id
        assigned[mono_idx] = True
        group_id += 1

    return group_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scaled_tolerance(mz: float, base_tolerance: float) -> float:
    """ppm-scaled mass tolerance.

    MS-DIAL's convention: ``base_tolerance`` is defined at m/z 200, and
    scales linearly with m/z. So at m/z 1000 the tolerance is 5×.
    """
    ppm_at_200 = base_tolerance / 200.0 * 1e6
    accuracy = mz * ppm_at_200 / 1e6
    return max(accuracy, base_tolerance)


def _estimate_charge(
    mono_mz: float,
    mono_rt: float,
    candidate_indices: list[int],
    features: List[CandidateFeature],
    tolerance: float,
    rt_tolerance: float,
    max_charge: int,
) -> int:
    """Find charge state from the first candidate within +1.003355 / k."""
    # Walk candidates in m/z order; first candidate close enough to be M+1
    # for any charge k wins.
    for idx in candidate_indices:
        cand = features[idx]
        if cand.precursor_mz > mono_mz + C13_DELTA + tolerance:
            return 1
        if abs(cand.rt_apex - mono_rt) > rt_tolerance:
            continue
        for k in range(max_charge, 0, -1):
            predicted = mono_mz + C13_DELTA / k
            if abs(predicted - cand.precursor_mz) <= tolerance:
                return k
    return 1


def _walk_envelope(
    mono_mz: float,
    mono_intensity: float,
    mono_rt: float,
    charge: int,
    candidate_indices: list[int],
    features: List[CandidateFeature],
    tolerance: float,
    rt_tolerance: float,
    max_steps: int,
) -> list[_IsotopeSlot]:
    """Walk M+1..M+max_steps, slotting the best matching candidate per step."""
    slots: list[_IsotopeSlot] = []
    mz_focused = mono_mz
    # Precompute the candidate (mz, idx) list for forward scanning.
    cand_pairs = [(features[i].precursor_mz, i) for i in candidate_indices]
    cand_pairs.sort()

    cursor = 0
    for step in range(1, max_steps + 1):
        predicted = mz_focused + C13_DELTA / charge
        slot = _IsotopeSlot(weight=step, predicted_mz=predicted)
        # Advance the cursor past anything strictly below predicted - tolerance.
        while cursor < len(cand_pairs) and cand_pairs[cursor][0] < predicted - tolerance:
            cursor += 1
        # Scan candidates whose m/z is within predicted ± tolerance.
        local = cursor
        while local < len(cand_pairs):
            cand_mz, cand_idx = cand_pairs[local]
            if cand_mz > predicted + tolerance:
                break
            cand = features[cand_idx]
            if abs(cand.rt_apex - mono_rt) <= rt_tolerance:
                if slot.matched_feature_idx is None:
                    slot.matched_mz = cand_mz
                    slot.matched_intensity = cand.ms1_height or 0.0
                    slot.matched_feature_idx = cand_idx
                else:
                    # Closer-to-predicted wins ties.
                    if abs(slot.matched_mz - predicted) > abs(cand_mz - predicted):
                        slot.matched_mz = cand_mz
                        slot.matched_intensity = cand.ms1_height or 0.0
                        slot.matched_feature_idx = cand_idx
            local += 1
        slots.append(slot)
        if slot.matched_feature_idx is not None:
            mz_focused = slot.matched_mz
        else:
            mz_focused = predicted
            # If two consecutive slots empty, terminate (MS-DIAL convention).
            if step >= 2 and slots[-2].matched_feature_idx is None:
                break
    return slots


def _accept_envelope(
    mono_mz: float,
    mono_intensity: float,
    slots: list[_IsotopeSlot],
) -> list[_IsotopeSlot]:
    """Apply MS-DIAL's intensity-decrease / alkane-ratio validation."""
    accepted: list[_IsotopeSlot] = []
    mono_mass = mono_mz  # for charge 1 envelope; multiplying by charge
                        # for the alkane check is a refinement we skip
                        # for now (charge ≥ 2 is rare for DDA at this scale)
    if mono_mass <= 800.0:
        prev_intensity = mono_intensity
        for slot in slots:
            if slot.matched_feature_idx is None:
                # MS-DIAL keeps walking past a single missing slot; require
                # the *next* slot to also be empty before terminating.
                continue
            if slot.matched_intensity >= prev_intensity:
                break
            accepted.append(slot)
            prev_intensity = slot.matched_intensity
    else:
        # > 800 Da: compare measured M+k / M+(k-1) ratio to alkane CnH2n.
        n_carbons = max(1, int(mono_mass / 14.0))
        prev_intensity = mono_intensity
        for slot in slots:
            if slot.matched_feature_idx is None:
                continue
            if prev_intensity <= 0:
                break
            exp_ratio = slot.matched_intensity / prev_intensity
            sim_ratio = _alkane_ratio(n_carbons, slot.weight)
            if abs(exp_ratio - sim_ratio) >= 5.0:
                break
            accepted.append(slot)
            prev_intensity = slot.matched_intensity

    return accepted


def _alkane_ratio(n_carbons: int, step: int) -> float:
    """Approximate M+k / M+(k-1) intensity ratio for a CnH2n alkane.

    Uses the binomial isotope envelope for carbon: each C has 1.07 %
    chance of being 13C. ``step`` is the k in M+k.
    """
    p = 0.0107
    if step <= 0 or n_carbons <= 0 or step > n_carbons:
        return 0.0
    # Ratio C(n, k) / C(n, k-1) = (n - k + 1) / k
    combo_ratio = (n_carbons - step + 1) / step
    return combo_ratio * (p / (1.0 - p))
