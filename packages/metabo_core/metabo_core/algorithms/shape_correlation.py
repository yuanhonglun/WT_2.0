"""Cross-EIC shape coherence for fragment-ion grouping.

When several product-ion EICs are clustered into a single feature, the
peak detector's per-EIC Gaussian check only verifies that *each*
individual trace looks bell-shaped. It does NOT verify that all the
traces **rise and fall together** within the cluster's RT window.

Real metabolite features have all their fragments co-eluting: a single
underlying ionisation event produces every fragment, so each fragment's
EIC follows the same time course (with tiny intensity differences from
detector noise). The pairwise Pearson correlations between fragment
EICs over a shared window therefore cluster around 0.9-1.0.

Continuous-baseline-noise "features" do not have this property: each
m/z channel's noise is independent, so pairwise correlations are
centered near 0. A cluster of independent noise traces that happen to
each look bell-shaped will pass per-EIC Gaussian gates while still
having low median pairwise correlation.

This helper is shared by ASFAM (Stage 1 MS2 clustering, Stage 2
MS2-to-MS1 assignment) and GC-MS (deconvolution stage) so all three
apps' "ions belong to the same compound" judgments use the same
underlying criterion. Each app supplies its own threshold but the
algorithm is identical.
"""
from __future__ import annotations

import numpy as np


def median_pairwise_correlation(eic_segments: list[np.ndarray]) -> float:
    """Median pairwise Pearson correlation of a set of co-windowed EICs.

    Parameters
    ----------
    eic_segments : list of 1D numpy arrays
        Each entry is one fragment's intensity values sampled at the
        same RT grid (same length, same scan indices). Callers must
        slice every EIC to a common window before calling.

    Returns
    -------
    float
        Median of all C(n, 2) pairwise Pearson correlations. For
        ``n == 1`` returns 1.0 (a single EIC is trivially coherent).
        Pairs with degenerate (zero-variance) traces are excluded from
        the median. If every pair is degenerate, returns 0.0.

    Notes
    -----
    Real metabolite clusters typically have median ≥ 0.7 over a 5-15
    scan window. A median ≤ 0.3 strongly suggests the traces are
    independent (baseline-noise origin).
    """
    n = len(eic_segments)
    if n < 2:
        return 1.0
    valid: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            r = _pairwise_pearson(eic_segments[i], eic_segments[j])
            if r is not None:
                valid.append(r)
    return float(np.median(valid)) if valid else 0.0


def _pairwise_pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    """Return Pearson(a, b) or None if either array is constant.

    A zero-variance trace has undefined correlation; the caller treats
    such pairs as "cannot judge" rather than letting them poison the
    median with NaN.
    """
    if a.size != b.size or a.size < 2:
        return None
    if float(np.std(a)) < 1e-9 or float(np.std(b)) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])
