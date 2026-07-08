"""Unit tests for filter_ions_by_coelution.

The function is GC-MS / ASFAM / DDA agnostic. These tests use synthetic
EICs + DetectedPeak stand-ins and never import anything from app code.
"""
import numpy as np
import pytest

from metabo_core.algorithms.ion_coelution_filter import (
    IonFilterResult,
    ModelPeakWindow,
    filter_ions_by_coelution,
)
from metabo_core.models.chromatography import DetectedPeak


def _make_peak(apex_index: int, **kwargs) -> DetectedPeak:
    """Build a DetectedPeak with only the field the algorithm reads."""
    defaults = dict(
        precursor_mz_nominal=0,
        product_mz=0.0,
        rt_apex=float(apex_index) * 0.1,
        rt_left=float(apex_index - 1) * 0.1,
        rt_right=float(apex_index + 1) * 0.1,
        apex_index=apex_index,
        left_index=apex_index - 1,
        right_index=apex_index + 1,
        height=1000.0,
        area=100.0,
        sn_ratio=10.0,
        gaussian_similarity=0.9,
    )
    defaults.update(kwargs)
    return DetectedPeak(**defaults)


# --- Common fixtures -------------------------------------------------------

@pytest.fixture
def model_peak():
    return ModelPeakWindow(left_idx=10, right_idx=20, apex_idx=15)


@pytest.fixture
def gaussian_eic():
    """30-scan gaussian centered at scan 15, FWHM ~ 6 scans."""
    xs = np.arange(30)
    return 1000.0 * np.exp(-0.5 * ((xs - 15) / 3.0) ** 2)


@pytest.fixture
def flat_eic():
    return np.zeros(30)


# --- Tests -----------------------------------------------------------------

def test_apex_coincident_pass(model_peak, gaussian_eic):
    """High-response ion with detected peak apex == model apex -> kept."""
    spec = [(57.0, 800.0)]
    eics = {57.0: gaussian_eic}
    peaks = {57.0: [_make_peak(apex_index=15)]}
    result = filter_ions_by_coelution(
        spec, model_peak=model_peak,
        ion_eics=eics, ion_detected_peaks=peaks,
        apex_window_scans=3, pearson_threshold=0.7,
        pearson_min_correlated_scans=4, low_response_rel=0.005,
    )
    assert result.kept_mask == [True]
    assert result.verdicts[0].reason == "apex_coincident"


def test_apex_offset_within_window_pass(model_peak, gaussian_eic):
    """Detected peak apex within +/-apex_window_scans of model apex -> kept."""
    spec = [(57.0, 800.0)]
    eics = {57.0: gaussian_eic}
    peaks = {57.0: [_make_peak(apex_index=17)]}  # offset=2, window=3
    result = filter_ions_by_coelution(
        spec, model_peak=model_peak,
        ion_eics=eics, ion_detected_peaks=peaks,
        apex_window_scans=3, pearson_threshold=0.7,
        pearson_min_correlated_scans=4, low_response_rel=0.005,
    )
    assert result.kept_mask == [True]
    assert result.verdicts[0].reason == "apex_coincident"


def test_apex_offset_outside_window_drop(model_peak, flat_eic):
    """Detected peak apex outside window AND Pearson fail -> dropped."""
    spec = [(91.0, 800.0)]
    eics = {91.0: flat_eic}  # zero EIC -> Pearson degenerate
    peaks = {91.0: [_make_peak(apex_index=25)]}  # offset=10, window=3
    result = filter_ions_by_coelution(
        spec, model_peak=model_peak,
        ion_eics=eics, ion_detected_peaks=peaks,
        apex_window_scans=3, pearson_threshold=0.7,
        pearson_min_correlated_scans=4, low_response_rel=0.005,
    )
    assert result.kept_mask == [False]
    assert result.verdicts[0].reason == "no_peak_no_corr"


def test_no_detected_peak_pearson_pass(model_peak, gaussian_eic):
    """Ion not in detected_peaks but EIC correlates with model -> kept via Pearson.

    PR4: single-ion fixture -> base_peak == intensity, so the ion is
    inevitably classified as strong (intensity >= base * strong_ion_rel).
    The candidate is excluded from the top-K model, so the model EIC
    falls back to the synthetic gaussian, which still correlates well.
    Expected reason is therefore ``strong_pearson_pass``.
    """
    spec = [(43.0, 200.0)]
    eics = {43.0: gaussian_eic * 0.3}  # same shape, lower amplitude
    peaks = {}  # no detected peak on this channel
    result = filter_ions_by_coelution(
        spec, model_peak=model_peak,
        ion_eics=eics, ion_detected_peaks=peaks,
        apex_window_scans=3, pearson_threshold=0.7,
        pearson_min_correlated_scans=4, low_response_rel=0.005,
    )
    assert result.kept_mask == [True]
    assert result.verdicts[0].reason == "strong_pearson_pass"


def test_no_detected_peak_pearson_fail(model_peak, gaussian_eic):
    """No detected peak + uncorrelated EIC -> dropped."""
    spec = [(120.0, 200.0)]
    rng = np.random.default_rng(0)
    noise = rng.uniform(0, 100, 30)
    eics = {120.0: noise}
    peaks = {}
    result = filter_ions_by_coelution(
        spec, model_peak=model_peak,
        ion_eics=eics, ion_detected_peaks=peaks,
        apex_window_scans=3, pearson_threshold=0.7,
        pearson_min_correlated_scans=4, low_response_rel=0.005,
    )
    assert result.kept_mask == [False]
    assert result.verdicts[0].reason == "no_peak_no_corr"


def test_low_response_skip(model_peak, gaussian_eic):
    """intensity < base * low_response_rel -> kept unconditionally."""
    spec = [(57.0, 10000.0), (200.0, 1.0)]  # 200 is 0.01% of base
    eics = {57.0: gaussian_eic, 200.0: np.zeros(30)}
    peaks = {57.0: [_make_peak(apex_index=15)]}
    result = filter_ions_by_coelution(
        spec, model_peak=model_peak,
        ion_eics=eics, ion_detected_peaks=peaks,
        apex_window_scans=3, pearson_threshold=0.7,
        pearson_min_correlated_scans=4, low_response_rel=0.005,
    )
    assert result.kept_mask == [True, True]
    assert result.verdicts[1].reason == "low_response"


def test_model_ion_always_kept(model_peak, gaussian_eic):
    """Model ion (apex_index == model.apex_idx) must always pass."""
    spec = [(67.0, 10000.0)]
    eics = {67.0: gaussian_eic}
    peaks = {67.0: [_make_peak(apex_index=15)]}
    result = filter_ions_by_coelution(
        spec, model_peak=model_peak,
        ion_eics=eics, ion_detected_peaks=peaks,
        apex_window_scans=3, pearson_threshold=0.99,  # absurd threshold
        pearson_min_correlated_scans=4, low_response_rel=0.005,
    )
    assert result.kept_mask == [True]


def test_empty_spectrum(model_peak):
    """Empty spec -> empty mask + empty verdicts; no exception."""
    result = filter_ions_by_coelution(
        [], model_peak=model_peak,
        ion_eics={}, ion_detected_peaks={},
        apex_window_scans=3, pearson_threshold=0.7,
        pearson_min_correlated_scans=4, low_response_rel=0.005,
    )
    assert result.kept_mask == []
    assert result.verdicts == []


def test_mz_missing_from_eics(model_peak):
    """mz in spectrum but not in ion_eics -> drop (conservative)."""
    spec = [(999.0, 500.0)]
    result = filter_ions_by_coelution(
        spec, model_peak=model_peak,
        ion_eics={}, ion_detected_peaks={},
        apex_window_scans=3, pearson_threshold=0.7,
        pearson_min_correlated_scans=4, low_response_rel=0.005,
    )
    assert result.kept_mask == [False]


def test_constant_eic_no_pearson(model_peak):
    """Zero-variance EIC -> Pearson degenerate -> drop."""
    spec = [(100.0, 500.0)]
    eics = {100.0: np.full(30, 5.0)}  # constant non-zero
    result = filter_ions_by_coelution(
        spec, model_peak=model_peak,
        ion_eics=eics, ion_detected_peaks={},
        apex_window_scans=3, pearson_threshold=0.7,
        pearson_min_correlated_scans=4, low_response_rel=0.005,
    )
    assert result.kept_mask == [False]


def test_insufficient_overlap(model_peak, gaussian_eic):
    """Too few nonzero overlap scans -> drop even with high correlation."""
    spec = [(50.0, 200.0)]
    eic = np.zeros(30)
    # Only 2 nonzero scans, perfectly correlated with model in those scans
    eic[14] = 100.0
    eic[16] = 100.0
    eics = {50.0: eic}
    result = filter_ions_by_coelution(
        spec, model_peak=model_peak,
        ion_eics=eics, ion_detected_peaks={},
        apex_window_scans=3, pearson_threshold=0.5,
        pearson_min_correlated_scans=4,  # requires 4, we have 2
        low_response_rel=0.005,
    )
    assert result.kept_mask == [False]


def test_compose_topk_model_eic_averages_top_three_normalized():
    """Composed model EIC = mean of top-K ions' max-normalized EICs."""
    import numpy as np
    from metabo_core.algorithms.ion_coelution_filter import _compose_topk_model_eic

    # 5 ions; top 3 at apex=15 are 60.0, 70.0, 80.0 with apex intensities
    # 100, 80, 60. Bottom 2 are 90.0, 100.0 with apex intensities 10, 5.
    n_scans = 30
    ions = {}
    for mz, amp in [(60.0, 100), (70.0, 80), (80.0, 60),
                    (90.0, 10), (100.0, 5)]:
        ions[mz] = amp * np.exp(-0.5 * ((np.arange(n_scans) - 15) / 3.0) ** 2)
    composed = _compose_topk_model_eic(ions, apex=15, k=3)
    # composed value at apex should equal 1.0 (3 ions each normalized to 1)
    assert abs(composed[15] - 1.0) < 1e-6
    # The full shape should still be a single gaussian
    assert composed.argmax() == 15


def test_compose_topk_model_eic_returns_none_on_empty():
    import numpy as np
    from metabo_core.algorithms.ion_coelution_filter import _compose_topk_model_eic
    assert _compose_topk_model_eic({}, apex=10, k=3) is None


def test_topk_model_uses_three_ions():
    """Composed model uses top-3 (not loudest only)."""
    import numpy as np
    from metabo_core.algorithms.ion_coelution_filter import (
        ModelPeakWindow, filter_ions_by_coelution,
    )

    n_scans = 30
    apex = 15
    sigma = 3.0
    # 5 ions: top 3 have shape == gaussian at apex 15. The 4th and 5th
    # have a different shape (flat). The candidate is one of the top 3 —
    # it must pass because the composed model is the average of the top 3.
    ions = {}
    for mz, amp in [(60.0, 100), (70.0, 80), (80.0, 60),
                    (90.0, 10), (100.0, 5)]:
        ions[mz] = amp * np.exp(-0.5 * ((np.arange(n_scans) - apex) / sigma) ** 2)
    # Candidate = the 3rd ion (still strong, in top 3)
    spectrum = [(80.0, 60.0)]
    result = filter_ions_by_coelution(
        spectrum,
        model_peak=ModelPeakWindow(left_idx=5, right_idx=25, apex_idx=apex),
        ion_eics=ions,
        ion_detected_peaks={mz: [] for mz in ions},  # no detected peaks; force Pearson branch
        model_topk=3,
    )
    assert result.kept_mask == [True]
    assert result.verdicts[0].reason in {"pearson_pass", "strong_pearson_pass"}
    assert result.verdicts[0].pearson_value is not None
    assert result.verdicts[0].pearson_value > 0.95


def test_strong_ion_relaxed_pearson_pass():
    """Strong ion (intensity >= base * strong_ion_rel) uses relaxed threshold.

    Hand-crafted candidate EIC: gaussian shape on the broad side, model_topk=1
    so the model is just the 1000-intensity 60.0 ion. We deliberately make
    the candidate's sigma slightly different so Pearson lands in the
    [0.40, 0.60) band — verified deterministically without RNG.
    """
    import numpy as np
    from metabo_core.algorithms.ion_coelution_filter import (
        ModelPeakWindow, filter_ions_by_coelution,
    )

    n_scans = 30
    apex = 15
    base_model = np.exp(-0.5 * ((np.arange(n_scans) - apex) / 3.0) ** 2)
    # Candidate = wider gaussian + offset apex: deterministically gives
    # Pearson in the [0.4, 0.6) band when computed over [5, 25].
    candidate = np.exp(-0.5 * ((np.arange(n_scans) - (apex + 4)) / 6.0) ** 2)
    # base peak intensity = 1000; candidate intensity = 500 (>= 0.3 * 1000 = 300)
    ions = {60.0: 1000.0 * base_model, 70.0: 500.0 * candidate}
    spectrum = [(60.0, 1000.0), (70.0, 500.0)]
    result = filter_ions_by_coelution(
        spectrum,
        model_peak=ModelPeakWindow(left_idx=5, right_idx=25, apex_idx=apex),
        ion_eics=ions,
        ion_detected_peaks={60.0: [], 70.0: []},
        pearson_threshold=0.60,                # weak-ion threshold
        strong_ion_pearson_threshold=0.40,     # strong-ion relaxed
        strong_ion_rel=0.30,
        relaxed_pearson_threshold=0.30,
        apex_concentration_threshold=0.99,     # disable concentration branch
        model_topk=1,
    )
    # candidate is strong, Pearson should be in [0.40, 0.60) → kept via strong branch
    cand_verdict = result.verdicts[1]
    assert cand_verdict.is_strong_ion is True
    assert cand_verdict.kept is True
    assert cand_verdict.reason == "strong_pearson_pass"
    assert cand_verdict.pearson_value is not None
    assert 0.40 <= cand_verdict.pearson_value < 0.60, (
        f"Pearson {cand_verdict.pearson_value} outside intended band; "
        "tweak candidate sigma/offset until it lands."
    )


def test_weak_ion_strict_pearson_fail():
    """Weak ion below relaxed branch threshold drops."""
    import numpy as np
    from metabo_core.algorithms.ion_coelution_filter import (
        ModelPeakWindow, filter_ions_by_coelution,
    )

    n_scans = 30
    apex = 15
    base_model = np.exp(-0.5 * ((np.arange(n_scans) - apex) / 3.0) ** 2)
    # candidate weak (intensity = 50, base = 1000 → 5%, well below strong_rel 0.3)
    # and noisy enough that Pearson < 0.6, and apex_concentration low.
    rng = np.random.default_rng(0)
    candidate = 0.3 * base_model + rng.normal(0, 0.5, n_scans)
    ions = {60.0: 1000.0 * base_model, 70.0: 50.0 * candidate}
    spectrum = [(60.0, 1000.0), (70.0, 50.0)]
    result = filter_ions_by_coelution(
        spectrum,
        model_peak=ModelPeakWindow(left_idx=5, right_idx=25, apex_idx=apex),
        ion_eics=ions,
        ion_detected_peaks={60.0: [], 70.0: []},
        pearson_threshold=0.60,
        strong_ion_pearson_threshold=0.40,
        strong_ion_rel=0.30,
        relaxed_pearson_threshold=0.30,
        apex_concentration_threshold=0.99,
        model_topk=1,
    )
    cand = result.verdicts[1]
    assert cand.is_strong_ion is False
    # Allow either drop or low_response — what we want is "not kept on a strong branch"
    assert cand.reason in {"no_peak_no_corr", "low_response"}


def test_concentration_branch_pass():
    """Low Pearson but apex-concentrated → kept via concentration branch."""
    import numpy as np
    from metabo_core.algorithms.ion_coelution_filter import (
        ModelPeakWindow, filter_ions_by_coelution,
    )

    n_scans = 30
    apex = 15
    base_model = np.exp(-0.5 * ((np.arange(n_scans) - apex) / 3.0) ** 2)
    # Candidate ion = a sharp narrow spike right at apex (apex_concentration high)
    # but its full-window Pearson against the gaussian is moderate. Need at
    # least 4 nonzero scans inside [left, right] to satisfy
    # pearson_min_correlated_scans (default 4).
    cand_ion = np.zeros(n_scans)
    cand_ion[apex - 2: apex + 3] = [0.05, 0.3, 1.0, 0.3, 0.05]
    ions = {60.0: 1000.0 * base_model, 70.0: 200.0 * cand_ion}
    spectrum = [(60.0, 1000.0), (70.0, 200.0)]
    result = filter_ions_by_coelution(
        spectrum,
        model_peak=ModelPeakWindow(left_idx=5, right_idx=25, apex_idx=apex),
        ion_eics=ions,
        ion_detected_peaks={60.0: [], 70.0: []},
        pearson_threshold=0.99,                # force fail strict
        strong_ion_pearson_threshold=0.99,     # force fail strong
        strong_ion_rel=0.30,
        relaxed_pearson_threshold=0.20,
        apex_concentration_threshold=0.30,
        model_topk=1,
    )
    cand = result.verdicts[1]
    assert cand.kept is True
    assert cand.reason == "concentration_pass"
    assert cand.apex_concentration > 0.30


def test_concentration_branch_fail_flat_pollution():
    """Flat pollution (low concentration) drops even with moderate Pearson."""
    import numpy as np
    from metabo_core.algorithms.ion_coelution_filter import (
        ModelPeakWindow, filter_ions_by_coelution,
    )

    n_scans = 30
    apex = 15
    base_model = np.exp(-0.5 * ((np.arange(n_scans) - apex) / 3.0) ** 2)
    # Candidate = flat constant with tiny slope. Concentration = 1/n (very low).
    cand_ion = np.linspace(0.99, 1.01, n_scans)
    ions = {60.0: 1000.0 * base_model, 70.0: 200.0 * cand_ion}
    spectrum = [(60.0, 1000.0), (70.0, 200.0)]
    result = filter_ions_by_coelution(
        spectrum,
        model_peak=ModelPeakWindow(left_idx=5, right_idx=25, apex_idx=apex),
        ion_eics=ions,
        ion_detected_peaks={60.0: [], 70.0: []},
        pearson_threshold=0.99,
        strong_ion_pearson_threshold=0.99,
        strong_ion_rel=0.30,
        relaxed_pearson_threshold=0.20,
        apex_concentration_threshold=0.30,
        model_topk=1,
    )
    cand = result.verdicts[1]
    assert cand.kept is False
    assert cand.reason == "no_peak_no_corr"
