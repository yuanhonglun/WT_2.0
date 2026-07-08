"""Tests for metabo_core.gcms.library_matching.

Covers:
1. ``acquired_ion_set`` returns the union of m/z (rounded by tol) where
   intensity > 0 across apex ± window scans.
2. With a synthetic "non-acquired ion is exactly 0, acquired ion has noise > 0"
   setup, the function returns exactly the acquired set.
2b. Critical edge case: an acquired ion that happens to be 0 at one scan in the
   window must still appear in the set (the union covers it).
3. ``csim_intersected_cosine`` clamps the reference to the acquired set then
   computes cosine.
4. ``fullscan_cosine`` returns standard cosine without intersection.
5. An acquired ion missing entirely from the reference contributes nothing
   (it stays unmatched).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


@dataclass
class _Scan:
    """Minimal scan-like duck-typed object for testing."""
    mz_array: np.ndarray
    intensity_array: np.ndarray


def _scan(pairs: list[tuple[float, float]]) -> _Scan:
    if not pairs:
        return _Scan(np.array([], dtype=float), np.array([], dtype=float))
    mz, inten = zip(*pairs)
    return _Scan(np.asarray(mz, dtype=float), np.asarray(inten, dtype=float))


# ---------------------------------------------------------------------------
# acquired_ion_set
# ---------------------------------------------------------------------------

def test_acquired_ion_set_basic_union() -> None:
    from metabo_core.gcms.library_matching import acquired_ion_set

    scans = [
        _scan([(73.0, 5.0), (91.0, 0.0), (100.0, 12.0)]),
        _scan([(73.0, 4.0), (91.0, 1.0), (100.0, 14.0)]),
        _scan([(73.0, 6.0), (91.0, 0.0), (100.0, 16.0)]),
    ]
    result = acquired_ion_set(scans, apex_index=1, window=1, mz_tol=0.01)
    # 91.0 is > 0 only at scan index 1 (within the window), so it is acquired.
    assert pytest.approx(73.0, abs=0.005) in [round(m, 3) for m in result]
    assert pytest.approx(91.0, abs=0.005) in [round(m, 3) for m in result]
    assert pytest.approx(100.0, abs=0.005) in [round(m, 3) for m in result]


def test_acquired_ion_set_excludes_non_acquired_zero_channel() -> None:
    """Channel that is exactly 0 across the entire window is NOT acquired."""
    from metabo_core.gcms.library_matching import acquired_ion_set

    scans = []
    for _ in range(7):
        scans.append(_scan([
            (73.0, 5.0),    # acquired: noise > 0 everywhere
            (91.0, 4.0),    # acquired
            (200.0, 0.0),   # NOT acquired: exactly 0 across all scans
        ]))
    apex = 3
    result = acquired_ion_set(scans, apex_index=apex, window=3, mz_tol=0.01)

    assert any(abs(mz - 73.0) < 0.005 for mz in result)
    assert any(abs(mz - 91.0) < 0.005 for mz in result)
    assert not any(abs(mz - 200.0) < 0.005 for mz in result)


def test_acquired_ion_set_includes_ion_with_one_zero_in_window() -> None:
    """Critical edge case: ion X has intensity 0 at one scan within the window
    (because of noise/quantization, not because it wasn't acquired) and > 0
    at all other scans. It MUST still appear in the acquired set.
    """
    from metabo_core.gcms.library_matching import acquired_ion_set

    # window of 7 scans (apex ± 3).
    # ion X (m/z=120.0): [120, 95, 0, 110, 87, 92, 105] - acquired-but-zero at one scan
    # ion Y (m/z=200.0): [0, 0, 0, 0, 0, 0, 0]          - not acquired
    x_intensities = [120.0, 95.0, 0.0, 110.0, 87.0, 92.0, 105.0]
    y_intensities = [0.0] * 7
    scans = []
    for xi, yi in zip(x_intensities, y_intensities):
        scans.append(_scan([(120.0, xi), (200.0, yi)]))

    result = acquired_ion_set(scans, apex_index=3, window=3, mz_tol=0.01)
    assert any(abs(mz - 120.0) < 0.005 for mz in result), \
        "Ion with one zero-scan within the window must remain in acquired set"
    assert not any(abs(mz - 200.0) < 0.005 for mz in result), \
        "Ion with all-zero values must not appear in acquired set"


def test_acquired_ion_set_window_clamps_to_bounds() -> None:
    """Apex near the start/end of the scan list must still work without crash."""
    from metabo_core.gcms.library_matching import acquired_ion_set

    scans = [
        _scan([(50.0, 1.0)]),
        _scan([(50.0, 1.0)]),
        _scan([(50.0, 1.0)]),
    ]
    result_start = acquired_ion_set(scans, apex_index=0, window=3, mz_tol=0.01)
    result_end = acquired_ion_set(scans, apex_index=2, window=3, mz_tol=0.01)
    assert any(abs(mz - 50.0) < 0.005 for mz in result_start)
    assert any(abs(mz - 50.0) < 0.005 for mz in result_end)


def test_acquired_ion_set_dedupes_via_tolerance() -> None:
    """Two m/z within mz_tol collapse to one entry in the set."""
    from metabo_core.gcms.library_matching import acquired_ion_set

    scans = [
        _scan([(73.001, 5.0), (73.005, 6.0)]),
    ]
    result = acquired_ion_set(scans, apex_index=0, window=0, mz_tol=0.01)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# csim_intersected_cosine and fullscan_cosine
# ---------------------------------------------------------------------------

def test_fullscan_cosine_identical_spectra_score_one() -> None:
    from metabo_core.gcms.library_matching import fullscan_cosine

    spec = [(73.0, 100.0), (91.0, 50.0), (100.0, 80.0)]
    score, n = fullscan_cosine(spec, spec, mz_tol=0.01)
    assert score == pytest.approx(1.0, abs=1e-6)
    assert n == 3


def test_fullscan_cosine_disjoint_spectra_score_zero() -> None:
    from metabo_core.gcms.library_matching import fullscan_cosine

    a = [(73.0, 100.0)]
    b = [(150.0, 100.0)]
    score, n = fullscan_cosine(a, b, mz_tol=0.01)
    assert score == 0.0
    assert n == 0


def test_csim_intersected_cosine_drops_non_acquired_reference_ions() -> None:
    """Two reference ions are not acquired; cosine after intersection should be
    higher than fullscan cosine for the same data because the ref vector loses
    its non-acquired components.
    """
    from metabo_core.gcms.library_matching import (
        csim_intersected_cosine,
        fullscan_cosine,
    )

    measured = [(73.0, 100.0), (91.0, 50.0)]                  # 2 acquired ions
    reference = [(73.0, 100.0), (91.0, 50.0),
                 (150.0, 80.0), (200.0, 70.0)]                 # 4 ref ions; last 2 not acquired
    acquired = [73.0, 91.0]                                    # only the first two

    fs_score, _ = fullscan_cosine(measured, reference, mz_tol=0.01)
    cs_score, _ = csim_intersected_cosine(measured, reference, acquired, mz_tol=0.01)

    assert cs_score > fs_score
    assert cs_score == pytest.approx(1.0, abs=1e-6)


def test_csim_intersected_cosine_acquired_ion_missing_from_reference_unmatched() -> None:
    """If an acquired ion exists in measured but is missing from the reference,
    intersection cannot conjure a match. The cosine reflects the unmatched gap.
    """
    from metabo_core.gcms.library_matching import csim_intersected_cosine

    measured = [(73.0, 100.0), (91.0, 50.0), (130.0, 200.0)]   # 130 has no ref entry
    reference = [(73.0, 100.0), (91.0, 50.0)]
    acquired = [73.0, 91.0, 130.0]

    score, n = csim_intersected_cosine(measured, reference, acquired, mz_tol=0.01)
    assert 0.0 < score < 1.0
    assert n == 2  # only 73 and 91 matched


def test_csim_intersected_cosine_acquired_set_can_be_passed_as_iterable() -> None:
    from metabo_core.gcms.library_matching import csim_intersected_cosine

    measured = [(73.0, 100.0)]
    reference = [(73.0, 100.0), (91.0, 50.0)]
    score_list, _ = csim_intersected_cosine(measured, reference, [73.0], mz_tol=0.01)
    score_set, _ = csim_intersected_cosine(measured, reference, {73.0}, mz_tol=0.01)
    assert score_list == score_set
