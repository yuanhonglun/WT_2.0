"""MS-DIAL MSDec — least-squares model-peak deconvolution for MS2 spectra.

Faithful port of the MS-DIAL MSDec engine (the historically MS1-named engine
reused for MS2 deconvolution):

  - ``MSDIAL5/MsdialCore/MSDec/MSDecHandler.cs`` — model-peak grouping
    (LC-MS ``ExtractedIonChromatogram`` overload, L689-737).
  - ``MSDIAL5/MsdialCore/MSDec/MSDecProcess.cs`` — least-squares decomposition
    (``ValuePeak[]`` overload).
  - ``MSDIAL5/MsdialLcMsApi/Algorithm/Ms2Dec.cs`` — per-precursor entry.

The engine separates the fragments of overlapping co-eluters that share an
isolation window: for each precursor peak it builds model chromatograms from
the product-ion EICs, links the precursor apex to the nearest model
(``<=2`` scan), and regresses every product ion against the target model
(plus up to 2-left/2-right overlapping neighbour models, a linear term and a
constant baseline). Only the target regression weight is kept; an ion whose
target weight is positive contributes one spectrum peak at the model apex,
with the product ion's own m/z (NOT re-centroided — handoff pitfall #11) and
the deconvoluted apex height.

Reusable foundations: LWMA from :func:`..._lwma_msdial`, VS1 peak detection
from :func:`...msdial_detect_peaks_in_chromatogram`, shape scores from
:func:`...compute_peak_shape_scores`, and the singular-tolerant LU from
:mod:`metabo_core.algorithms.lu_solve`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from metabo_core.algorithms.lu_solve import (
    determinant_a,
    matrix_decompose,
    matrix_inverse,
)
from metabo_core.algorithms.peak_shape import compute_peak_shape_scores
from metabo_core.algorithms.msdial_peak_spotting import (
    _lwma_msdial,
    msdial_detect_peaks_in_chromatogram,
)
from metabo_core.config.msdec import MsdecConfig
from metabo_core.config.msdial_peak_spotting import lc_msdial_config


# ---------------------------------------------------------------------------
# Baseline correction — MSDecHandler.cs getBaselineCorrectedPeaklist (L1380)
# ---------------------------------------------------------------------------
def baseline_correct(
    intensity: np.ndarray, rt: np.ndarray, peak_top: int
) -> np.ndarray:
    """Subtract a straight baseline through the flanking local minima.

    The baseline endpoints are the local minima on ``[0, peak_top]`` and
    ``[peak_top, n-1]``; the baseline value at each scan is
    ``int(coeff * RT + intercept)`` (C-style truncation toward zero) and the
    corrected intensity is floored at 0. The ``int`` truncation and the use
    of RT (``Time``) as x — not the scan index — are load-bearing.
    """
    yi = np.asarray(intensity, dtype=np.float64)
    ti = np.asarray(rt, dtype=np.float64)
    n = yi.shape[0]

    # Left local minimum on [0, peak_top] (scan from peak_top down, strict <).
    min_left_id = 0
    min_val = math.inf
    for i in range(peak_top, -1, -1):
        if yi[i] < min_val:
            min_val = yi[i]
            min_left_id = i

    # Right local minimum on [peak_top, n-1] (scan from peak_top up, strict <).
    min_right_id = n - 1
    min_val = math.inf
    for i in range(peak_top, n):
        if yi[i] < min_val:
            min_val = yi[i]
            min_right_id = i

    dt = ti[min_right_id] - ti[min_left_id]
    if dt == 0:
        coeff = 0.0
        intercept = float(yi[min_left_id])
    else:
        coeff = (yi[min_right_id] - yi[min_left_id]) / dt
        intercept = (
            ti[min_right_id] * yi[min_left_id] - ti[min_left_id] * yi[min_right_id]
        ) / dt

    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        c = yi[i] - int(coeff * ti[i] + intercept)
        out[i] = c if c >= 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# Matched filter — MSDecHandler.cs getMatchedFileterArray (L1461)
# ---------------------------------------------------------------------------
def matched_filter(
    sharpness: np.ndarray, sigma: float, half_point: int = 10
) -> np.ndarray:
    """Mexican-hat matched filter over the per-scan sharpness array.

    Fixed kernel length ``2*half_point+1`` (21 by default), zero-padded at the
    array edges, independent of ``sigma`` — merges peak tops 1-2 scans apart
    into one component.
    """
    s = np.asarray(sharpness, dtype=np.float64)
    n = s.shape[0]
    klen = 2 * half_point + 1
    coef = np.empty(klen, dtype=np.float64)
    for i in range(klen):
        x = (-half_point + i) / sigma
        coef[i] = (1.0 - x * x) * math.exp(-0.5 * x * x)

    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        acc = 0.0
        for j in range(-half_point, half_point + 1):
            k = i + j
            if k < 0 or k > n - 1:
                continue
            acc += s[k] * coef[j + half_point]
        out[i] = acc
    return out


# ---------------------------------------------------------------------------
# Region markers — MSDecHandler.cs getRegionMarkers (L1414)
# ---------------------------------------------------------------------------
def region_markers(
    matched_filter_array: np.ndarray, margin: int = 5
) -> list[tuple[int, int]]:
    """Return inclusive ``(scan_begin, scan_end)`` regions of positive filter
    response. The first/last ``margin`` scans are ignored. A region opens on a
    rising positive value and closes on a non-positive value or a local
    minimum (where a new region immediately opens)."""
    mf = np.asarray(matched_filter_array, dtype=np.float64)
    n = mf.shape[0]
    regions: list[tuple[int, int]] = []
    scan_begin = 0
    flg = False
    i = margin
    while i < n - margin:
        if mf[i] > 0 and mf[i - 1] < mf[i] and not flg:
            scan_begin = i
            flg = True
        elif flg:
            if mf[i] <= 0:
                regions.append((scan_begin, i - 1))
                flg = False
            elif mf[i - 1] > mf[i] and mf[i] < mf[i + 1] and mf[i] >= 0:
                regions.append((scan_begin, i))
                scan_begin = i + 1
                flg = True
                i += 1
        i += 1
    return regions


# ---------------------------------------------------------------------------
# Least-squares target coefficient — MSDecProcess.cs ms1DecPattern* (L57-911)
# ---------------------------------------------------------------------------
def _gram_inverse_row0(basis: list[np.ndarray]) -> np.ndarray | None:
    """Build the Gram matrix of ``basis`` and return row 0 of its inverse.

    Returns ``None`` when the singular-tolerant LU rejects the matrix
    (all-zero row) or the determinant is exactly zero — the caller then drops
    to a lower-order pattern (MSDecProcess fallback cascade).
    """
    m = len(basis)
    gram = np.empty((m, m), dtype=np.float64)
    for a in range(m):
        for b in range(m):
            gram[a, b] = float(np.dot(basis[a], basis[b]))
    lu = matrix_decompose(gram)
    if lu is None:
        return None
    if determinant_a(lu) == 0.0:
        return None
    return matrix_inverse(lu)[0, :]


def solve_target_coefficient(
    target: np.ndarray, neighbors: list[np.ndarray], exp: np.ndarray
) -> float | None:
    """Regress ``exp`` onto ``[target, *neighbors, linear, const]`` and return
    only the target coefficient (``invMatrix[0, :] . z``).

    This is NOT NNLS: the neighbour / linear / constant coefficients are
    computed (they shape the inverse) but discarded. On a singular Gram the
    fallback drops the neighbours (single pattern); if that is still singular,
    returns ``None``.
    """
    tgt = np.asarray(target, dtype=np.float64)
    n = tgt.shape[0]
    linear = np.arange(n, dtype=np.float64)
    const = np.ones(n, dtype=np.float64)
    expv = np.asarray(exp, dtype=np.float64)

    basis = [tgt] + [np.asarray(x, dtype=np.float64) for x in neighbors] + [
        linear,
        const,
    ]
    row0 = _gram_inverse_row0(basis)
    if row0 is None:
        basis = [tgt, linear, const]  # fallback: single pattern
        row0 = _gram_inverse_row0(basis)
        if row0 is None:
            return None

    z = np.array([float(np.dot(b, expv)) for b in basis], dtype=np.float64)
    return float(np.dot(row0, z))


# ---------------------------------------------------------------------------
# Model peaks / model chromatograms — MSDecHandler.cs
# ---------------------------------------------------------------------------
@dataclass
class _PeakSpot:
    ion_idx: int
    mz: float
    left: int
    top: int
    right: int
    shapeness: float
    ideal_slope: float
    quality: str = "Low"  # High / Middle / Low


@dataclass
class _Model:
    scan_left: int
    scan_top: int
    scan_right: int
    peaks: np.ndarray
    model_mz_list: list[float] = field(default_factory=list)
    max_peak_top_value: float = 0.0


def _detect_model_peaks(
    ion_mzs: np.ndarray,
    raw_eics: np.ndarray,
    smoothed_eics: np.ndarray,
    rt: np.ndarray,
    config: MsdecConfig,
) -> list[_PeakSpot]:
    """VS1 peak detection per product-ion EIC + shape scores (getPeakSpots)."""
    det_cfg = replace(
        lc_msdial_config(),
        min_amplitude=float(config.min_amplitude),
        min_data_points=int(config.min_data_points),
        smoothing_level=int(config.smoothing_level),
        average_peak_width=int(config.average_peak_width),
    )
    spots: list[_PeakSpot] = []
    for ion_idx in range(raw_eics.shape[0]):
        peaks = msdial_detect_peaks_in_chromatogram(
            rt, raw_eics[ion_idx], config=det_cfg
        )
        if not peaks:
            continue
        smoothed = smoothed_eics[ion_idx]
        for pk in peaks:
            shape = compute_peak_shape_scores(
                smoothed, pk.left_index, pk.apex_index, pk.right_index
            )
            if shape is None:
                continue
            if shape.ideal_slope > config.ideal_slope_high:
                quality = "High"
            elif shape.ideal_slope > config.ideal_slope_middle:
                quality = "Middle"
            else:
                quality = "Low"
            spots.append(
                _PeakSpot(
                    ion_idx=ion_idx,
                    mz=float(ion_mzs[ion_idx]),
                    left=pk.left_index,
                    top=pk.apex_index,
                    right=pk.right_index,
                    shapeness=shape.shapeness,
                    ideal_slope=shape.ideal_slope,
                    quality=quality,
                )
            )
    return spots


def _model_intensity_at(model: _Model, scan: int) -> float:
    if model.scan_left > scan or model.scan_right < scan:
        return 0.0
    return float(model.peaks[scan - model.scan_left])


def _refine_model(model: _Model, config: MsdecConfig) -> _Model | None:
    """getRefinedModelChromatogram (L1252): re-locate apex, edge-walk with a
    ``>= average_peak_width*0.5`` minimum, reject if either side < min edge
    points."""
    peaks = model.peaks
    npx = peaks.shape[0]
    peak_top_id = int(np.argmax(peaks))
    model.max_peak_top_value = float(peaks[peak_top_id])
    half = config.average_peak_width * 0.5

    peak_left_id = -1
    for i in range(peak_top_id, 0, -1):
        if peak_top_id - i < half:
            continue
        if peaks[i - 1] >= peaks[i]:
            peak_left_id = i
            break
    if peak_left_id < 0:
        peak_left_id = 0

    peak_right_id = -1
    for i in range(peak_top_id, npx - 1):
        if i - peak_top_id < half:
            continue
        if peaks[i] <= peaks[i + 1]:
            peak_right_id = i
            break
    if peak_right_id < 0:
        peak_right_id = npx - 1

    # Update scan positions: top/right use the OLD scan_left, then left moves.
    new_top = peak_top_id + model.scan_left
    new_right = peak_right_id + model.scan_left
    new_left = peak_left_id + model.scan_left

    if peak_top_id - peak_left_id < config.model_min_edge_points:
        return None
    if peak_right_id - peak_top_id < config.model_min_edge_points:
        return None

    model.peaks = peaks[peak_left_id : peak_right_id + 1]
    model.scan_top = new_top
    model.scan_right = new_right
    model.scan_left = new_left
    return model


def _build_model_chromatogram(
    smoothed_eics: np.ndarray,
    areas: list[_PeakSpot],
    rt: np.ndarray,
    config: MsdecConfig,
) -> _Model | None:
    """getModelChromatogram (L874): the sharpest peak fixes the window; ions
    with sharpness >= 0.9*max join; model trace = mean of their baseline-
    corrected EICs (/ mzCount)."""
    max_sharp = max(a.shapeness for a in areas)
    participating = sorted(
        [a for a in areas if a.shapeness >= max_sharp * config.sharpness_inclusion_fraction],
        key=lambda a: -a.shapeness,
    )
    first = participating[0]
    scan_left, scan_top, scan_right = first.left, first.top, first.right
    rt_win = rt[scan_left : scan_right + 1]
    top_in_win = scan_top - scan_left

    model_mz_list: list[float] = []
    peaklists: list[np.ndarray] = []
    for a in participating:
        model_mz_list.append(a.mz)
        trace = smoothed_eics[a.ion_idx][scan_left : scan_right + 1]
        peaklists.append(baseline_correct(trace, rt_win, top_in_win))

    mz_count = float(len(model_mz_list))
    peaks = np.sum(peaklists, axis=0) / mz_count
    model = _Model(
        scan_left=scan_left,
        scan_top=scan_top,
        scan_right=scan_right,
        peaks=peaks,
        model_mz_list=model_mz_list,
    )
    return _refine_model(model, config)


def _refine_models(models: list[_Model]) -> list[_Model]:
    """getRefinedModelChromatograms (L1048): order by apex, drop duplicates at
    the same apex keeping the tallest."""
    ordered = sorted(models, key=lambda m: m.scan_top)
    out: list[_Model] = [ordered[0]]
    for m in ordered[1:]:
        if out[-1].scan_top == m.scan_top:
            if out[-1].max_peak_top_value < m.max_peak_top_value:
                out[-1] = m
        else:
            out.append(m)
    return out


@dataclass
class _ModelVector:
    chrom_scan_list: list[int]
    target_array: np.ndarray
    neighbor_arrays: list[np.ndarray]
    target_scan_left: int
    target_scan_top: int
    target_scan_right: int


def _build_model_vector(
    model_id: int, models: list[_Model]
) -> _ModelVector:
    """getModelChromatogramVector (L1129): include up to 2-left/2-right
    neighbour models by peak-range OVERLAP; window covers them."""
    target = models[model_id]
    left = target.scan_left
    right = target.scan_right
    cnt = len(models)

    is_2l = model_id > 1 and models[model_id - 2].scan_right > left
    is_1l = model_id > 0 and models[model_id - 1].scan_right > left
    is_2r = model_id < cnt - 2 and models[model_id + 2].scan_left < right
    is_1r = model_id < cnt - 1 and models[model_id + 1].scan_left < right

    if is_2l:
        left = min(target.scan_left, models[model_id - 2].scan_left, models[model_id - 1].scan_left)
    elif is_1l:
        left = min(target.scan_left, models[model_id - 1].scan_left)
    if is_2r:
        right = max(target.scan_right, models[model_id + 2].scan_right, models[model_id + 1].scan_right)
    elif is_1r:
        right = max(target.scan_right, models[model_id + 1].scan_right)

    chrom_scan_list = list(range(left, right + 1))
    target_arr = np.array([_model_intensity_at(target, i) for i in chrom_scan_list])

    # Neighbour basis order mirrors MS-DIAL {2L, 1L, 1R, 2R} (only present ones).
    neighbor_arrays: list[np.ndarray] = []
    if is_2l:
        neighbor_arrays.append(
            np.array([_model_intensity_at(models[model_id - 2], i) for i in chrom_scan_list])
        )
    if is_1l or is_2l:
        neighbor_arrays.append(
            np.array([_model_intensity_at(models[model_id - 1], i) for i in chrom_scan_list])
        )
    if is_1r or is_2r:
        neighbor_arrays.append(
            np.array([_model_intensity_at(models[model_id + 1], i) for i in chrom_scan_list])
        )
    if is_2r:
        neighbor_arrays.append(
            np.array([_model_intensity_at(models[model_id + 2], i) for i in chrom_scan_list])
        )

    target_scan_left = target.scan_left - left
    target_scan_top = target_scan_left + (target.scan_top - target.scan_left)
    target_scan_right = target_scan_left + (target.scan_right - target.scan_left)
    return _ModelVector(
        chrom_scan_list=chrom_scan_list,
        target_array=target_arr,
        neighbor_arrays=neighbor_arrays,
        target_scan_left=target_scan_left,
        target_scan_top=target_scan_top,
        target_scan_right=target_scan_right,
    )


def _adhoc_coefficient(
    exp: np.ndarray, target: np.ndarray, top: int
) -> float | None:
    """tryGetAdhocCoefficient (MSDecProcess.cs L932): rescue a non-positive
    target weight when the ion strictly rises 4 then falls 4 around the apex."""
    if top < 4:
        return None
    if top > exp.shape[0] - 5:
        return None
    if (
        exp[top - 1] > exp[top - 2] > exp[top - 3] > exp[top - 4]
        and exp[top + 1] > exp[top + 2] > exp[top + 3] > exp[top + 4]
    ):
        if target[top] == 0:
            return None
        return float(exp[top] / target[top])
    return None


def _refine_spectrum(
    spectrum: list[tuple[float, float]], config: MsdecConfig
) -> tuple[np.ndarray, np.ndarray]:
    """getRefinedMsDecSpectrum (L460): sort by m/z, dedupe to 4 decimals
    keeping max intensity, then cut off below the relative/absolute floor."""
    if not spectrum:
        return np.zeros(0), np.zeros(0)
    spectrum = sorted(spectrum, key=lambda p: p[0])
    deduped: list[list[float]] = [list(spectrum[0])]
    for mz, inten in spectrum[1:]:
        if round(deduped[-1][0], 4) != round(mz, 4):
            deduped.append([mz, inten])
        elif deduped[-1][1] < inten:
            deduped[-1] = [mz, inten]
    arr = np.array(deduped, dtype=np.float64)
    max_int = float(arr[:, 1].max())
    cutoff = max(
        max_int * config.relative_amplitude_cutoff, config.amplitude_cutoff
    )
    keep = arr[:, 1] > cutoff
    return arr[keep, 0], arr[keep, 1]


def deconvolute_ms2(
    ion_mzs: np.ndarray,
    ion_eics: np.ndarray,
    precursor_apex_scan: int,
    config: MsdecConfig,
    rt_array: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Deconvolve the MS2 spectrum at one precursor peak (MSDecHandler.cs
    GetMSDecResult, LC-MS overload).

    Parameters
    ----------
    ion_mzs : (N_ions,) product-ion m/z values.
    ion_eics : (N_ions, N_scans) per-ion windowed chromatograms, MS2-scan grid.
    precursor_apex_scan : precursor apex as a column index into ``ion_eics``.
    config : MsdecConfig.
    rt_array : (N_scans,) retention times; defaults to the scan index.

    Returns
    -------
    (out_mz, out_intensity) : the deconvoluted spectrum, m/z ascending. The
    product-ion m/z are kept as-is (NOT re-centroided). An empty spectrum is
    returned when no model peak exists or the precursor apex is further than
    ``apex_model_tolerance`` scans from every model (raw-spectrum fallback is
    the caller's responsibility).
    """
    ion_mzs = np.asarray(ion_mzs, dtype=np.float64)
    eics = np.asarray(ion_eics, dtype=np.float64)
    if eics.ndim != 2 or eics.shape[0] == 0:
        return np.zeros(0), np.zeros(0)
    n_scans = eics.shape[1]
    if rt_array is None:
        rt = np.arange(n_scans, dtype=np.float64)
    else:
        rt = np.asarray(rt_array, dtype=np.float64)

    level = int(config.smoothing_level)
    smoothed_eics = np.array([_lwma_msdial(eics[i], level) for i in range(eics.shape[0])])

    # 1. Per-ion model-peak detection + shape scores.
    spots = _detect_model_peaks(ion_mzs, eics, smoothed_eics, rt, config)
    if not spots:
        return np.zeros(0), np.zeros(0)

    # 2. Bin array: quality + per-bin max sharpness + peak-spot membership.
    total_sharp = np.zeros(n_scans, dtype=np.float64)
    bin_spots: list[list[int]] = [[] for _ in range(n_scans)]
    for idx, ps in enumerate(spots):
        scan = ps.top
        if ps.quality in ("High", "Middle"):
            if total_sharp[scan] < ps.shapeness:
                total_sharp[scan] = ps.shapeness
        bin_spots[scan].append(idx)

    # 3. Matched filter → 4. region markers.
    mf = matched_filter(total_sharp, config.sigma_window, config.matched_filter_half_point)
    regions = region_markers(mf, config.region_margin)

    # 5. One model chromatogram per region (High→Middle→Low selection).
    models: list[_Model] = []
    for (rb, re) in regions:
        areas: list[_PeakSpot] = []
        for quality in ("High", "Middle", "Low"):
            areas = [
                spots[i]
                for s in range(rb, re + 1)
                for i in bin_spots[s]
                if spots[i].quality == quality
            ]
            if areas:
                break
        if not areas:
            continue
        model = _build_model_chromatogram(smoothed_eics, areas, rt, config)
        if model is not None:
            models.append(model)
    if not models:
        return np.zeros(0), np.zeros(0)
    models = _refine_models(models)

    # 6. Link precursor apex to the nearest model (<= apex_model_tolerance).
    diffs = [abs(precursor_apex_scan - m.scan_top) for m in models]
    min_id = int(np.argmin(diffs))
    if diffs[min_id] > config.apex_model_tolerance:
        return np.zeros(0), np.zeros(0)

    # 7. Model vector with overlapping neighbour models.
    mv = _build_model_vector(min_id, models)
    win_left = mv.chrom_scan_list[0]
    win_right = mv.chrom_scan_list[-1]
    target_top = mv.target_scan_top

    # 8. Windowed baseline-corrected trace per product ion (getMs2Chromatograms).
    exp_rows: list[tuple[float, np.ndarray]] = []
    rt_win = rt[win_left : win_right + 1]
    for ion_idx in range(eics.shape[0]):
        trace = smoothed_eics[ion_idx][win_left : win_right + 1]
        bc = baseline_correct(trace, rt_win, target_top)
        if bc[target_top] <= config.amplitude_cutoff:
            continue
        exp_rows.append((float(ion_mzs[ion_idx]), bc))

    # 9. Least squares: Gram inverse computed once; per-ion target coefficient.
    n_win = mv.target_array.shape[0]
    linear = np.arange(n_win, dtype=np.float64)
    const = np.ones(n_win, dtype=np.float64)
    basis = [mv.target_array, *mv.neighbor_arrays, linear, const]
    row0 = _gram_inverse_row0(basis)
    if row0 is None:
        basis = [mv.target_array, linear, const]
        row0 = _gram_inverse_row0(basis)
        if row0 is None:
            return np.zeros(0), np.zeros(0)

    target_apex_model = float(mv.target_array[target_top])
    spectrum: list[tuple[float, float]] = []
    for mz, exp in exp_rows:
        z = np.array([float(np.dot(b, exp)) for b in basis], dtype=np.float64)
        coeff = float(np.dot(row0, z))
        if coeff <= 0:
            adhoc = _adhoc_coefficient(exp, mv.target_array, target_top)
            if adhoc is None:
                continue
            coeff = adhoc
        # Apex cap (getDeconvolutedValuePeaks): deconvoluted <= observed apex.
        if target_apex_model == 0:
            continue
        if exp[target_top] < coeff * target_apex_model:
            coeff = exp[target_top] / target_apex_model
        intensity = coeff * target_apex_model
        if intensity <= 0:
            continue
        spectrum.append((mz, intensity))

    # 10. Refine spectrum (4-decimal dedupe + cutoff).
    return _refine_spectrum(spectrum, config)
