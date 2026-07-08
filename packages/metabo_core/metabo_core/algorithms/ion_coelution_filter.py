"""Per-ion coelution filter for reconstructed spectra.

For a given feature's model peak window, decide which ions in the
reconstructed spectrum actually belong to that feature versus being
neighbor-compound contamination pulled in by the spectrum-reconstruction
step.

Designed to be GC-MS / ASFAM / DDA agnostic: callers pass their own EIC
dict + detected-peaks dict + model peak window in scan-index space.
Currently wired into GC-MS ``stage_filter_spectrum_ions`` only;
ASFAM / DDA integration is deferred to follow-up PRs.

PR4 multi-criteria voting:

1. Weak ions (intensity < base_peak * low_response_rel) -> pass
   unconditionally — too weak to judge shape reliably, and they
   contribute negligibly to library matching anyway.

2. Stronger ions:
   2a. If the ion's detected-peaks list contains any peak whose
       apex_index is within +/-apex_window_scans of the model apex
       -> pass (``apex_coincident``).
   2b. Else compute Pearson r between the ion's EIC slice and a model
       EIC composed from the top-K other ions (excluding the candidate
       itself) over the model peak window.
         - Strong ion (intensity >= base * strong_ion_rel):
             r >= strong_ion_pearson_threshold -> pass
             (``strong_pearson_pass``)
         - Weak ion:
             r >= pearson_threshold -> pass (``pearson_pass``)
   2c. Concentration fallback (any ion):
         r >= relaxed_pearson_threshold
         AND apex_concentration >= apex_concentration_threshold
         -> pass (``concentration_pass``).
   2d. Else drop (``no_peak_no_corr``).

This algorithm is the dual of
``shape_correlation.median_pairwise_correlation``: the latter judges
cluster-level coherence; this one judges per-ion membership.

Model EIC for Pearson comparison
--------------------------------
The model EIC used in step 2b is composed PER CANDIDATE by
``_compose_topk_model_eic``: it takes the top-K ions at the apex scan
(excluding the candidate itself), max-normalizes each, and averages
them. Excluding the candidate prevents trivial self-correlation. When
the pool of other ions is empty (single-EIC fixtures), the algorithm
falls back to a synthetic gaussian implied by the model peak window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from metabo_core.models.chromatography import DetectedPeak


@dataclass(frozen=True)
class ModelPeakWindow:
    """A feature's model peak window in scan-index space."""
    left_idx: int
    right_idx: int
    apex_idx: int


@dataclass
class IonVerdict:
    """Per-ion judgment record for audit / debugging / QC."""
    mz: float
    intensity: float
    kept: bool
    # one of: "low_response" | "apex_coincident" | "pearson_pass" |
    #         "strong_pearson_pass" | "concentration_pass" | "no_peak_no_corr"
    reason: str
    # PR4: optional, populated only when computed
    pearson_value: float | None = None
    apex_concentration: float | None = None
    is_strong_ion: bool = False


@dataclass
class IonFilterResult:
    """Aggregate result of filter_ions_by_coelution."""
    kept_mask: list[bool]      # aligned to input spectrum order
    verdicts: list[IonVerdict] # full per-ion audit trail


def filter_ions_by_coelution(
    spectrum: list[tuple[float, float]],
    *,
    model_peak: ModelPeakWindow,
    ion_eics: Mapping[float, np.ndarray],
    ion_detected_peaks: Mapping[float, list[DetectedPeak]],
    apex_window_scans: int = 3,
    pearson_threshold: float = 0.6,
    low_response_rel: float = 0.005,
    pearson_min_correlated_scans: int = 4,
    # PR4 新增 ↓
    model_topk: int = 3,
    strong_ion_rel: float = 0.3,
    strong_ion_pearson_threshold: float = 0.4,
    relaxed_pearson_threshold: float = 0.3,
    apex_concentration_threshold: float = 0.3,
) -> IonFilterResult:
    """Decide which spectrum ions belong to the given feature.

    PR4 upgrade:
      - model EIC = top-K ions' max-normalized average, composed PER
        CANDIDATE (excluding candidate itself, see _compose_topk_model_eic
        docstring). Replaces PR1's single-channel "loudest at apex" model
        and the old self-correlation guard.
      - multi-criteria voting:
          Step 1: low_response (< base * low_response_rel) → pass
          Step 2a: apex_coincident detected peak → pass
          Step 2b: strong ion (intensity >= base*strong_ion_rel):
                     Pearson >= strong_ion_pearson_threshold → pass (strong_pearson_pass)
                 weak ion:
                     Pearson >= pearson_threshold → pass (pearson_pass)
          Step 2c: any ion:
                     Pearson >= relaxed_pearson_threshold
                     AND apex_concentration >= apex_concentration_threshold
                     → pass (concentration_pass)
          Step 3: drop (no_peak_no_corr)

      - apex_concentration = sum(ion_slice[apex-1 : apex+2]) / sum(ion_slice).
    """
    if not spectrum:
        return IonFilterResult(kept_mask=[], verdicts=[])

    base_int = max((float(intensity) for _, intensity in spectrum), default=0.0)
    threshold_low = base_int * float(low_response_rel)
    strong_threshold = base_int * float(strong_ion_rel)

    left = int(model_peak.left_idx)
    right = int(model_peak.right_idx)
    apex = int(model_peak.apex_idx)

    mask: list[bool] = []
    verdicts: list[IonVerdict] = []

    for mz, intensity in spectrum:
        intensity_f = float(intensity)
        is_strong = intensity_f >= strong_threshold

        # Step 1 — low response
        if intensity_f < threshold_low:
            mask.append(True)
            verdicts.append(IonVerdict(
                mz=mz, intensity=intensity_f, kept=True,
                reason="low_response", is_strong_ion=is_strong,
            ))
            continue

        # Step 2a — apex-coincident detected peak
        peaks = ion_detected_peaks.get(mz, [])
        if any(abs(int(p.apex_index) - apex) <= apex_window_scans for p in peaks):
            mask.append(True)
            verdicts.append(IonVerdict(
                mz=mz, intensity=intensity_f, kept=True,
                reason="apex_coincident", is_strong_ion=is_strong,
            ))
            continue

        # Steps 2b / 2c — Pearson + concentration.
        # PR4: compose model EIC PER CANDIDATE, excluding the candidate
        # itself. This replaces the old self-correlation guard
        # (model_eic_auto is ion_eic -> synthetic fallback).
        model_eic = _compose_topk_model_eic(
            ion_eics, apex, k=model_topk, exclude_mz=mz,
        )
        if model_eic is None:
            model_eic = _synthetic_model_eic(left, right, apex)

        ion_eic = ion_eics.get(mz)
        pearson: float | None = None
        concentration: float | None = None
        if ion_eic is not None and model_eic is not None:
            pearson = _pearson_within_window(
                ion_eic, model_eic, left, right,
                min_correlated_scans=pearson_min_correlated_scans,
            )
            concentration = _apex_concentration(ion_eic, apex)

        # 2b — strong vs weak Pearson threshold
        if pearson is not None:
            effective_thr = strong_ion_pearson_threshold if is_strong else pearson_threshold
            if pearson >= effective_thr:
                reason = "strong_pearson_pass" if is_strong else "pearson_pass"
                mask.append(True)
                verdicts.append(IonVerdict(
                    mz=mz, intensity=intensity_f, kept=True,
                    reason=reason, pearson_value=pearson,
                    apex_concentration=concentration, is_strong_ion=is_strong,
                ))
                continue
            # 2c — concentration fallback
            if (pearson >= relaxed_pearson_threshold
                    and concentration is not None
                    and concentration >= apex_concentration_threshold):
                mask.append(True)
                verdicts.append(IonVerdict(
                    mz=mz, intensity=intensity_f, kept=True,
                    reason="concentration_pass", pearson_value=pearson,
                    apex_concentration=concentration, is_strong_ion=is_strong,
                ))
                continue

        # Step 3 — drop
        mask.append(False)
        verdicts.append(IonVerdict(
            mz=mz, intensity=intensity_f, kept=False,
            reason="no_peak_no_corr", pearson_value=pearson,
            apex_concentration=concentration, is_strong_ion=is_strong,
        ))

    _assert_model_ion_kept(spectrum, ion_detected_peaks, apex, mask)
    return IonFilterResult(kept_mask=mask, verdicts=verdicts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compose_topk_model_eic(
    ion_eics: Mapping[float, np.ndarray],
    apex: int,
    *,
    k: int = 3,
    exclude_mz: float | None = None,
) -> np.ndarray | None:
    """Compose a model EIC from the top-K ions at the apex scan.

    Each top ion is max-normalized (so its own peak = 1), then the K
    normalized EICs are averaged. The result is a clean reference shape
    that's more robust than picking the single "loudest at apex" channel
    (PR1 behavior), which could be polluted by neighbor components.

    ``exclude_mz`` — when set, that mz is excluded from the top-K pool.
    Caller uses this to prevent the candidate ion from self-correlating
    against a model composed (in part) of itself, which would give
    trivially high Pearson. Critical when k=1 (single-EIC fixtures or
    pathological cases): without exclusion the candidate would BE the
    model.

    Returns None when no valid EIC is available (empty pool, all zeros,
    or only the excluded ion was available).
    """
    if not ion_eics:
        return None
    # Rank by intensity at apex
    sortable: list[tuple[float, np.ndarray, float]] = []
    for mz, arr in ion_eics.items():
        if exclude_mz is not None and float(mz) == float(exclude_mz):
            continue
        if apex >= len(arr):
            continue
        sortable.append((float(mz), arr, float(arr[apex])))
    sortable.sort(key=lambda t: t[2], reverse=True)
    top = sortable[: max(1, int(k))]
    if not top:
        return None
    normalized = []
    for _mz, arr, _v in top:
        peak_max = float(arr.max())
        if peak_max <= 0:
            continue
        normalized.append(arr / peak_max)
    if not normalized:
        return None
    stacked = np.stack(normalized, axis=0)
    return np.mean(stacked, axis=0)


def _synthetic_model_eic(left: int, right: int, apex: int) -> np.ndarray:
    """Build an implicit gaussian model EIC from the model peak window.

    Used as a fallback when the candidate ion is the only EIC available
    (i.e. the auto-picked "loudest at apex" model would BE the candidate
    itself, making Pearson trivially 1.0). The gaussian is centered at
    ``apex`` with sigma set to a quarter of the window width, which puts
    the FWHM roughly inside [left, right].
    """
    n = max(int(right) + 1, int(apex) + 1, 1)
    xs = np.arange(n, dtype=np.float64)
    width = max(float(right - left), 1.0)
    sigma = width / 4.0
    return np.exp(-0.5 * ((xs - float(apex)) / sigma) ** 2)


def _pearson_within_window(
    ion_eic: np.ndarray, model_eic: np.ndarray,
    left: int, right: int, *,
    min_correlated_scans: int,
) -> float | None:
    """Return Pearson r over [left, right], or None if not computable."""
    end = min(len(ion_eic), len(model_eic), right + 1)
    start = max(0, left)
    if end - start < min_correlated_scans:
        return None
    ion_slice = np.asarray(ion_eic[start:end], dtype=np.float64)
    model_slice = np.asarray(model_eic[start:end], dtype=np.float64)
    overlap = int(np.count_nonzero((ion_slice > 0) & (model_slice > 0)))
    if overlap < min_correlated_scans:
        return None
    if float(np.std(ion_slice)) < 1e-9 or float(np.std(model_slice)) < 1e-9:
        return None
    return float(np.corrcoef(ion_slice, model_slice)[0, 1])


def _apex_concentration(ion_eic: np.ndarray, apex: int) -> float | None:
    """sum(intensity[apex-1 : apex+2]) / sum(ion_eic).

    Returns None when total is zero (silent EIC).
    """
    total = float(np.sum(ion_eic))
    if total <= 0:
        return None
    lo = max(0, apex - 1)
    hi = min(len(ion_eic), apex + 2)
    apex_sum = float(np.sum(ion_eic[lo:hi]))
    return apex_sum / total


def _assert_model_ion_kept(
    spectrum: list[tuple[float, float]],
    ion_detected_peaks: Mapping[float, list[DetectedPeak]],
    apex: int,
    mask: list[bool],
) -> None:
    """If a spectrum mz has a detected peak with apex_index == apex EXACTLY,
    it is the model ion and MUST be kept. Else implementation bug.

    Uses an explicit ``raise AssertionError`` rather than ``assert ...``
    so the check survives under ``python -O`` (the spec calls this
    "implementation bug must fail loudly, never silently degrade").
    """
    for i, (mz, _intensity) in enumerate(spectrum):
        for p in ion_detected_peaks.get(mz, []):
            if int(p.apex_index) == apex:
                if not mask[i]:
                    raise AssertionError(
                        f"model ion mz={mz} (apex={apex}) was dropped - "
                        "internal bug in filter_ions_by_coelution"
                    )
                return
