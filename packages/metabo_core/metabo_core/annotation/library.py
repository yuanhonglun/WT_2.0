"""Library annotation core algorithm.

Algorithmic core extracted from ASFAM stage 6.5. The core does not depend on
``ProcessingConfig`` or any ASFAM-specific data structure; it accepts plain
candidate features (with ``precursor_mz`` / ``rt_apex`` / ``ms2_mz`` /
``ms2_intensity``) and returns top-N ``AnnotationMatch`` results. ASFAM stage
glue is responsible for iterating replicates and writing results back.

Platform invariant (see CLAUDE.md "分析不变量"): a spectral library lives
only for the duration of the annotation stage. Whatever this module hands
back — a lean spectrum list or an m/z index — the calling stage must drop
its reference and ``gc.collect()`` before returning, so the library never
survives into align / export / GUI rendering.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from metabo_core.algorithms.similarity import composite_similarity_breakdown
from metabo_core.config import AnnotationConfig, SimilarityConfig
from metabo_core.io.spectral_library import read_msp, read_mgf
from metabo_core.models import AnnotationMatch

logger = logging.getLogger(__name__)

# The only metadata fields the annotation matcher / AnnotationMatch reads.
# The on-disk LC-MS library (lib/lcms/pos.msp ~6.7GB) also carries inchi /
# smiles / comment / splash / synon / instrument / ... per spectrum, which
# nothing downstream uses but which dominate the in-memory footprint
# (~24GB fully loaded). Loading lean — only these fields, plus float64
# numpy peak arrays — cuts that to ~6-7GB with byte-identical matching.
# Extend this set (and AnnotationMatch) if a field needs to reach the UI.
ANNOTATION_METADATA_FIELDS = frozenset(
    {"name", "precursor_mz", "formula", "adduct", "rt"}
)

# The exact matcher tests ``|q - r - shift| <= mz_tolerance`` while the batch
# screens test ``|q - (r + shift)| <= tol``; the two can differ in the last ulp.
# Widening can only *add* candidates, which costs a scoring call, never a hit.
_TOL_SLACK = 1e-9
_TOL_ABS = 1e-12

# The WDP bound sums TQ / TR pairwise while the exact scorer accumulates them
# sequentially, and np.sqrt is only correctly rounded. Both leave relative errors
# around 1e-13; widening the bound by 1e-9 buys four orders of margin at the cost
# of an occasional extra scoring call.
_WDP_UB_SLACK = 1e-9

# ``_three_scores_core`` identifies unmatched peaks by ``round(mz, 4)``, so two
# peaks of one spectrum closer than 1e-4 collapse: if one of them is matched, the
# other drops out of the scalar sum entirely. That shrinks WDP's denominator below
# "sum over every peak", which is exactly what :func:`_wdp_upper_bound` assumes
# when it bounds the denominator from below — a spectrum with such a pair can
# score a *higher* wdp than the bound. Those candidates opt out of the screen.
# (5e-5 apart, one matched: true wdp 0.938 vs a modelled 0.500.)
_ROUND4_GAP = 1.0000001e-4


def load_library_lean(
    path: str,
    keep_metadata: Optional[set[str]] = None,
) -> Optional[list[dict]]:
    """Load an MSP/MGF library in the *lean* representation.

    The single shared loader behind every app's annotation stage. It keeps
    only the metadata keys in ``keep_metadata`` (default
    ``ANNOTATION_METADATA_FIELDS``) and stores peaks as float64 numpy
    arrays, which bounds the in-memory footprint of a multi-GB library
    without changing any matching result (float64 preserves the exact peak
    values, and the dropped fields are ones no matcher reads).

    ``keep_metadata`` is matched against metadata keys *exactly*, so a
    caller that reads a field under several aliases must list every alias.
    GC-MS, for instance, reads RI as either ``ri`` or ``retention_index``
    and must pass both; a missing alias silently yields ``None`` rather
    than an error. The default set stays LC-MS-shaped (it carries
    ``precursor_mz``, which EI libraries lack) — extend it per call site,
    never in place.

    Returns ``None`` on an unsupported extension, an empty library, or any
    read error; callers treat that as "no library, skip annotation".
    """
    try:
        keep = (
            set(keep_metadata)
            if keep_metadata is not None
            else set(ANNOTATION_METADATA_FIELDS)
        )
        if path.lower().endswith(".msp"):
            spectra = read_msp(path, keep_metadata=keep, as_arrays=True)
        elif path.lower().endswith(".mgf"):
            spectra = read_mgf(path, keep_metadata=keep, as_arrays=True)
        else:
            logger.warning("Unsupported library format: %s", path)
            return None

        if not spectra:
            logger.warning("Library is empty: %s", path)
            return None

        return spectra
    except Exception as exc:
        logger.warning("Failed to load library: %s", exc)
        return None


def load_and_index_library(path: str) -> Optional[dict]:
    """Load an MSP/MGF library lean and build an integer m/z index.

    Convenience composition of :func:`load_library_lean` and
    :func:`build_index_from_list` for the LC-MS annotators (DDA stage 3,
    ASFAM stage 6.5). Callers that need the flat spectrum list instead of
    an index — ASFAM stage 2.5 — call ``load_library_lean`` directly.
    """
    spectra = load_library_lean(path)
    if not spectra:
        return None
    return build_index_from_list(spectra)


def _build_bucket_csr(
    spectra: list[dict],
    indices: list[int],
    share_peaks: bool = False,
) -> dict:
    """Flatten one integer-m/z bucket's reference peaks into CSR layout.

    ``match_feature_topn`` screens a whole bucket with a handful of numpy calls
    (``searchsorted`` / boolean AND / ``bincount``) instead of a Python loop
    over candidates. That needs every candidate's peaks in one contiguous
    array plus a peak→candidate map, which is exactly CSR:

    * ``flat_mz``   — all peaks of all candidates, concatenated in candidate order
    * ``offsets``   — candidate ``i`` owns ``flat_mz[offsets[i]:offsets[i+1]]``
    * ``sig_mask``  — per peak, whether it is a *significant* reference peak
      (intensity ≥ 1% of that candidate's base peak) — the Matched% numerator
      only ever counts these
    * ``flat_sqrt_nr`` — per peak, ``sqrt(intensity / base peak)``, the factor
      the WDP covariance bound multiplies by
    * ``n_sig_ref`` — per candidate, the Matched% denominator (floor 1)
    * ``TR`` / ``SR0`` — per candidate, ``Σ nr·mz_r`` and ``Σ nr``: WDP's
      reference-side scalar and the coefficient of its lower bound
    * ``wdp_bound_ok`` — per candidate, whether the WDP bound is applicable
      (see :data:`_ROUND4_GAP`)
    * ``n_peaks`` / ``ref_pmz`` / ``spec_idx`` — per candidate

    Candidate order is the bucket's ``mz_index`` order, which is what keeps the
    equal-score tie order of ``hits.sort`` identical to the pre-CSR loop.

    Three omissions, all to keep a multi-GB library from growing (``pos.msp``
    carries 134 M peaks, so every stored byte per peak costs ~134 MB):

    * no ``flat_int`` — ``flat_sqrt_nr`` already carries the only intensity the
      screens read, with the per-candidate normalisation baked in, and the exact
      scorer re-reads raw intensities from the spectrum;
    * no stored ``penalty`` — five brackets recomputed from ``n_peaks``;
    * no stored ``owner`` — :func:`_bucket_owner` rebuilds the peak→candidate
      map with one ``np.repeat`` per (feature, bucket), which costs ~30 µs and
      saves 4 bytes/peak.

    ``flat_sqrt_nr`` is float32 rounded *up* (``nextafter`` toward +inf), halving
    its cost. Rounding up is what keeps ``ub_cov`` an upper bound; a value stored
    even one ulp low would silently drop real hits.

    ``share_peaks`` re-points each spectrum's ``mz`` at its slice of
    ``flat_mz``. The values are identical (both float64), and it makes the
    flattening memory-neutral instead of duplicating 8 bytes/peak. Only
    :func:`build_index_from_list` sets it — that function already annotates the
    spec dicts, and it owns the library for the rest of the annotation stage.
    """
    n = len(indices)
    counts = np.zeros(n, dtype=np.int64)
    ref_pmz = np.zeros(n, dtype=np.float64)
    n_sig_ref = np.ones(n, dtype=np.int64)
    max_r = np.ones(n, dtype=np.float64)
    mz_parts: list[np.ndarray] = []
    int_parts: list[np.ndarray] = []
    sig_parts: list[np.ndarray] = []

    for k, idx in enumerate(indices):
        spec = spectra[idx]
        raw_mz = spec.get("mz")
        raw_int = spec.get("intensity")
        ref_pmz[k] = spec.get("_precursor_mz") or 0.0
        if raw_mz is None or raw_int is None or len(raw_mz) == 0:
            mz_parts.append(np.zeros(0, dtype=np.float64))
            int_parts.append(np.zeros(0, dtype=np.float64))
            sig_parts.append(np.zeros(0, dtype=bool))
            continue
        mz = np.asarray(raw_mz, dtype=np.float64)
        inten = np.asarray(raw_int, dtype=np.float64)
        # Mirrors composite_similarity_breakdown: thresh = max_r * 0.01, and
        # the Matched% denominator floors at 1.
        peak = float(inten.max())
        sig = inten >= peak * 0.01
        counts[k] = mz.size
        n_sig_ref[k] = max(int(np.count_nonzero(sig)), 1)
        # A flat-zero reference normalises to nr = 0, hence TR = SR0 = 0, hence
        # an invalid (kept) bound — which is right: the exact scorer bails out
        # with wdp = 0 on max_r < 1e-12 anyway.
        max_r[k] = peak if peak > 0.0 else 1.0
        mz_parts.append(mz)
        int_parts.append(inten)
        sig_parts.append(sig)

    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    flat_mz = np.concatenate(mz_parts) if n else np.zeros(0, dtype=np.float64)
    flat_int = np.concatenate(int_parts) if n else np.zeros(0, dtype=np.float64)

    # Per-peak normalised reference intensity, vectorised over the whole bucket:
    # 3.35 M library spectra make per-spectrum numpy calls the dominant cost of
    # index building.
    owner = np.repeat(np.arange(n, dtype=np.int64), counts)
    nr = np.maximum(flat_int / max_r[owner], 0.0)
    flat_sqrt_nr = np.nextafter(
        np.sqrt(nr).astype(np.float32), np.float32(np.inf))
    TR = np.bincount(owner, weights=nr * flat_mz, minlength=n)
    SR0 = np.bincount(owner, weights=nr, minlength=n)

    # A candidate opts out of the WDP screen unless its peaks are strictly
    # ascending with every gap wide enough to survive round(mz, 4).
    wdp_bound_ok = np.ones(n, dtype=bool)
    if flat_mz.size > 1:
        adjacent = owner[1:] == owner[:-1]
        collides = adjacent & ~(np.diff(flat_mz) > _ROUND4_GAP)
        if collides.any():
            wdp_bound_ok[owner[1:][collides]] = False

    if share_peaks:
        # Drop the per-spectrum buffers in favour of views into flat_mz; the
        # originals are freed as the last reference goes away.
        for k, idx in enumerate(indices):
            if counts[k]:
                spectra[idx]["mz"] = flat_mz[offsets[k]:offsets[k + 1]]

    return {
        "spec_idx": np.asarray(indices, dtype=np.int64),
        "n_peaks": counts,
        "ref_pmz": ref_pmz,
        "n_sig_ref": n_sig_ref,
        "offsets": offsets,
        "flat_mz": flat_mz,
        "flat_sqrt_nr": flat_sqrt_nr,
        "TR": TR,
        "SR0": SR0,
        "wdp_bound_ok": wdp_bound_ok,
        "sig_mask": (np.concatenate(sig_parts) if n
                     else np.zeros(0, dtype=bool)),
    }


def _bucket_owner(bucket: dict) -> np.ndarray:
    """Per-peak candidate id, rebuilt on demand (see :func:`_build_bucket_csr`)."""
    return np.repeat(
        np.arange(bucket["spec_idx"].size, dtype=np.int32), bucket["n_peaks"])


def build_index_from_list(spectra: list[dict]) -> Optional[dict]:
    """Build an integer m/z index from a pre-loaded list of spectra.

    As a one-time amortization, cache per-spectrum parsed metadata
    (``precursor_mz`` / ``rt``) onto each spec dict under ``_`` -prefixed
    keys. ``match_feature_topn`` reads them by direct dict lookup instead
    of recomputing per (feature, candidate) pair.

    The returned dict keeps its long-standing ``"spectra"`` / ``"index"``
    keys (DDA's ``ctx.library_index`` and the annotate stages read them);
    ``"csr"`` is additive — a per-bucket flattening used by the batch
    pre-screen. Pass it to ``match_feature_topn(..., csr=...)``; omitting it
    is correct but rebuilds the three needed buckets on every call.
    """
    if not spectra:
        return None
    mz_index: dict[int, list[int]] = {}
    for idx, spec in enumerate(spectra):
        meta = spec.get("metadata", {})
        pmz = meta.get("precursor_mz")
        if pmz is not None:
            key = int(round(pmz))
            mz_index.setdefault(key, []).append(idx)

        # Cache parsed metadata used per-candidate by match_feature_topn.
        spec["_precursor_mz"] = float(pmz) if pmz is not None else None
        rt_raw = meta.get("rt", 0)
        spec["_rt"] = float(rt_raw) if rt_raw else 0.0

    csr = {
        key: _build_bucket_csr(spectra, idxs, share_peaks=True)
        for key, idxs in mz_index.items()
    }
    logger.info(
        "Library indexed: %d spectra, %d unique integer m/z values",
        len(spectra), len(mz_index),
    )
    return {"spectra": spectra, "index": mz_index, "csr": csr}


def _passes_prefilter(
    query_bins_wide: set[int],
    ref_mz,
    shift: float,
    inv_tol: float,
    min_matched: int,
) -> bool:
    """Cheap binned shared-peak screen — an upper bound on ``n_matched``.

    Returns ``False`` only when *neither* the direct nor the neutral-loss
    (shifted) frame can reach ``min_matched`` shared peaks; such a candidate
    cannot pass the ``n_matched >= min_matched`` gate, so skipping it before
    the expensive composite scorer is result-preserving.

    A reference peak counts as a potential match when its integer m/z bin (or
    an adjacent bin) is occupied by a query peak. Because a true within-±tol
    match always lands in one of those three bins, the count never undercounts
    real matches — the screen never drops a candidate the full scorer would
    have kept. It may overcount (keep a candidate that ultimately fails), which
    only costs an unnecessary full score, never a missed hit.

    The neutral-loss frame mirrors :func:`composite_similarity_breakdown`:
    a query peak matches ``ref_mz + shift`` where ``shift = precursor_ref -
    precursor_query``.
    """
    if min_matched <= 0:
        return True

    direct = 0
    for r in ref_mz:
        if int(r * inv_tol) in query_bins_wide:
            direct += 1
            if direct >= min_matched:
                return True

    shifted = 0
    for r in ref_mz:
        if int((r + shift) * inv_tol) in query_bins_wide:
            shifted += 1
            if shifted >= min_matched:
                return True
    return False


def _widened_tol(mz_tol: float) -> float:
    """The m/z tolerance the batch screens use (see :data:`_TOL_SLACK`)."""
    return mz_tol * (1.0 + _TOL_SLACK) + _TOL_ABS


def _nearest_query_within(q_sorted: np.ndarray, x: np.ndarray, tol: float) -> np.ndarray:
    """Per element of ``x``: does some query peak lie within ``±tol``?

    ``q_sorted`` is the query m/z sorted ascending, so the closest query peak to
    ``x`` is one of the two straddling it. Sorting is safe: "a query peak exists
    within tol" does not depend on peak order.
    """
    n = q_sorted.size
    pos = np.searchsorted(q_sorted, x)
    hi = np.minimum(pos, n - 1)
    lo = np.maximum(pos - 1, 0)
    dist = np.minimum(np.abs(q_sorted[hi] - x), np.abs(x - q_sorted[lo]))
    return dist <= tol


def _matched_pct_upper_bound(
    bucket: dict,
    q_sorted: np.ndarray,
    feat_mz: float,
    mz_tol: float,
) -> np.ndarray:
    """A per-candidate upper bound on ``CompositeSimilarityResult.matched_pct``.

    ``matched_pct``'s numerator counts the *significant* reference peaks that the
    greedy matcher paired with a query peak; its denominator is ``n_sig_ref``.
    Greedy matching is mutually exclusive, so it can only pair *fewer* reference
    peaks than the number that merely have a query peak within ``±tol``. Counting
    the latter — with the same denominator — therefore never underestimates:

        true matched_pct  ≤  ub_pct

    The scorer picks whichever of the direct / neutral-loss frame yields more
    matched pairs, so the bound is taken as the element-wise max over both frames.

    Dropping candidates with ``ub_pct < min_matched_pct`` is result-preserving:
    such a candidate could never have cleared the ``matched_pct`` gate.

    The tolerance is widened by a relative epsilon because the exact matcher
    evaluates ``|q - r - shift|`` while this screen evaluates ``|q - (r + shift)|``;
    those can differ in the last ulp. Widening can only *add* candidates, which
    costs a scoring call and never a hit.
    """
    owner = _bucket_owner(bucket)
    flat_mz = bucket["flat_mz"]
    sig = bucket["sig_mask"]
    n_cand = bucket["spec_idx"].size
    tol = _widened_tol(mz_tol)

    hit = _nearest_query_within(q_sorted, flat_mz, tol) & sig
    ub_num = np.bincount(owner[hit], minlength=n_cand)

    if feat_mz > 0:
        # Neutral-loss frame, exactly as composite_similarity_breakdown gates it:
        # only attempted when both precursors are positive.
        ref_pmz = bucket["ref_pmz"]
        shifted = flat_mz + (ref_pmz - feat_mz)[owner]
        hit_nl = (_nearest_query_within(q_sorted, shifted, tol)
                  & sig & (ref_pmz > 0)[owner])
        np.maximum(ub_num, np.bincount(owner[hit_nl], minlength=n_cand),
                   out=ub_num)

    return ub_num / bucket["n_sig_ref"]


def _sparse_range_max_table(a: np.ndarray) -> np.ndarray:
    """Sparse table for O(1) range-max queries: row ``k`` = ``max(a[i:i+2**k])``.

    Built once per feature over the query's normalised intensities (a few hundred
    entries, so O(n log n) is free), then queried once per (reference peak, frame).
    """
    n = a.size
    levels = int(np.frexp(float(n))[1]) - 1 if n else 0  # floor(log2(n))
    rows = np.empty((levels + 1, n), dtype=np.float64)
    rows[0] = a
    for k in range(1, levels + 1):
        half = 1 << (k - 1)
        width = n - (1 << k) + 1
        rows[k, :width] = np.maximum(rows[k - 1, :width],
                                     rows[k - 1, half:half + width])
        rows[k, width:] = rows[k - 1, width:]  # out of range; never queried
    return rows


def _range_max(rows: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Element-wise ``max(a[lo[i]:hi[i]])``. Every window must be non-empty."""
    k = np.frexp((hi - lo).astype(np.float64))[1] - 1
    width = np.left_shift(1, k)
    return np.maximum(rows[k, lo], rows[k, hi - width])


def _wdp_upper_bound(
    bucket: dict,
    sel: np.ndarray,
    q_sorted: np.ndarray,
    q_nq_table: np.ndarray,
    TQ: float,
    SQ0: float,
    feat_mz: float,
    mz_tol: float,
) -> np.ndarray:
    """A per-candidate upper bound on ``CompositeSimilarityResult.wdp``.

    ``matched_pct``'s gate barely bites — the median library spectrum has 5
    significant peaks, so one matched peak clears ``>= 0.25``. ``min_wdp`` is the
    screen that actually rejects three quarters of the candidates that reach the
    exact scorer, and this is its cheap, provable predictor.

    For one frame with shift ``s`` (``0`` direct, ``ref_pmz - feat_mz`` neutral
    loss), the scorer greedily pairs ``|mz_q - mz_r - s| <= tol`` and computes::

        cov_w = Σ_M sqrt(nq·nr)·mz_avg      mz_avg = (mz_q + mz_r) / 2
        SQ    = Σ_M nq·mz_avg + Σ_{q∉M} nq·mz_q
        SR    = Σ_M nr·mz_avg + Σ_{r∉M} nr·mz_r
        wdp   = min(cov_w² / (SQ·SR) · penalty, 1)

    Bound the numerator from above and both denominators from below. With
    ``N(r)`` the query peaks inside ``r``'s window and ``R_hit`` the references
    with a non-empty window::

        ub_cov = Σ_{R_hit} sqrt(nr · max_{N(r)} nq) · max(mz_r + (s+tol)/2, 0)
        SQ_lb  = TQ - max(s + tol, 0)/2 · SQ0        TQ = Σ_Q nq·mz_q, SQ0 = Σ_Q nq
        SR_lb  = TR - max(tol - s, 0)/2 · SR0        TR = Σ_R nr·mz_r, SR0 = Σ_R nr

    ``ub_cov >= cov_w`` because every matched ``r`` is in ``R_hit`` with
    ``nq <= max_{N(r)} nq`` and ``mz_avg <= mz_r + (s+tol)/2``, and dropping the
    greedy matcher's mutual exclusivity only adds non-negative terms. ``SQ_lb <=
    SQ`` because ``mz_avg >= mz_q - (s+tol)/2`` and ``Σ_M nq <= SQ0``; ``SR_lb <=
    SR`` symmetrically. So ``ub_wdp >= wdp``, and dropping candidates with
    ``ub_wdp < min_wdp`` is result-preserving.

    ``max_{N(r)} nq`` need not come from the same query peak that maximises
    ``mz_avg`` — the max of a product is bounded by the product of the maxima,
    which is all the bound needs.

    Two ways the bound stops applying, both of which return ``+inf`` so the
    candidate survives to the exact scorer:

    * a denominator lower bound that is not positive (large ``s`` drives
      ``SQ_lb`` negative). Clamping it to a small positive number instead would
      silently turn the bound into a fiction.
    * ``wdp_bound_ok`` false — the spectrum has a ``round(mz, 4)`` collision, so
      its true ``SR`` is smaller than the model above (see :data:`_ROUND4_GAP`).

    The frames are bounded independently and the max is taken: the scorer scores
    exactly one of them, and that one's bound dominates it.
    """
    tol = _widened_tol(mz_tol)
    n_sel = sel.size
    counts = bucket["n_peaks"][sel]
    total = int(counts.sum())
    if total == 0:
        return np.full(n_sel, np.inf)

    # Gather just the surviving candidates' peaks out of the bucket's CSR arrays;
    # after S1 + S2' that is a few percent of the bucket.
    owner = np.repeat(np.arange(n_sel, dtype=np.int64), counts)
    seg_start = np.cumsum(counts) - counts
    peak_idx = (np.arange(total, dtype=np.int64)
                - seg_start[owner] + bucket["offsets"][sel][owner])
    mz_r = bucket["flat_mz"][peak_idx]
    sqrt_nr = bucket["flat_sqrt_nr"][peak_idx]

    TR = bucket["TR"][sel]
    SR0 = bucket["SR0"][sel]
    penalty = np.where(counts <= 1, 0.75,
                       np.where(counts == 2, 0.88,
                                np.where(counts == 3, 0.94,
                                         np.where(counts <= 5, 0.97, 1.0))))

    def _frame(shift: np.ndarray, active: Optional[np.ndarray]) -> np.ndarray:
        shift_peak = shift[owner]
        x = mz_r + shift_peak
        lo = np.searchsorted(q_sorted, x - tol, side="left")
        hi = np.searchsorted(q_sorted, x + tol, side="right")
        hit = hi > lo
        if active is not None:
            hit &= active[owner]
        # Empty windows are clamped to a valid one-element range and then zeroed.
        nq_star = _range_max(q_nq_table, np.where(hit, lo, 0), np.where(hit, hi, 1))
        contrib = np.where(
            hit,
            sqrt_nr * np.sqrt(nq_star)
            * np.maximum(mz_r + 0.5 * (shift_peak + tol), 0.0),
            0.0)
        ub_cov = np.bincount(owner, weights=contrib, minlength=n_sel)

        SQ_lb = TQ - 0.5 * np.maximum(shift + tol, 0.0) * SQ0
        SR_lb = TR - 0.5 * np.maximum(tol - shift, 0.0) * SR0
        out = np.full(n_sel, np.inf)
        ok = (SQ_lb > 1e-12) & (SR_lb > 1e-12)
        if active is not None:
            out[~active] = 0.0  # frame not attempted: contributes no bound
            ok &= active
        j = np.flatnonzero(ok)
        if j.size:
            ratio = (ub_cov[j] ** 2 / (SQ_lb[j] * SR_lb[j])
                     * penalty[j] * (1.0 + _WDP_UB_SLACK))
            out[j] = np.minimum(ratio, 1.0)
        return out

    ub = _frame(np.zeros(n_sel), None)
    if feat_mz > 0:
        # Neutral-loss frame, gated exactly as composite_similarity_breakdown
        # gates it: both precursors positive.
        ref_pmz = bucket["ref_pmz"][sel]
        nl_active = ref_pmz > 0
        if nl_active.any():
            np.maximum(ub, _frame(ref_pmz - feat_mz, nl_active), out=ub)

    ub[~bucket["wdp_bound_ok"][sel]] = np.inf
    return ub


def match_feature_topn(
    feature,
    spectra: list[dict],
    mz_index: dict[int, list[int]],
    annotation: AnnotationConfig,
    similarity: SimilarityConfig,
    top_n: Optional[int] = None,
    csr: Optional[dict] = None,
) -> list[AnnotationMatch]:
    """Match one feature against an indexed library and return top-N hits.

    ``csr`` is the ``"csr"`` entry of :func:`build_index_from_list`. Passing it
    is what makes the batch screen cheap; when it is omitted the three buckets
    this feature touches are flattened on the fly, through the same code, so the
    result is identical either way.

    Screening order (each stage is result-preserving, so the order is free, but
    they are ordered cheapest-first): S1 precursor window → S2' batch Matched%
    upper bound → S2" batch WDP upper bound → S2 binned shared-peak filter →
    exact composite score. Only survivors reach the scorer, and only the top-N
    survivors pay for a ``ref_peaks`` list.
    """
    feat_mz = feature.precursor_mz
    feat_mz_int = int(round(feat_mz))

    bucket_keys = [k for k in (feat_mz_int - 1, feat_mz_int, feat_mz_int + 1)
                   if k in mz_index]
    if not bucket_keys:
        return []

    limit = top_n if top_n is not None else annotation.top_n

    rt_query = feature.rt_apex
    mz_tol = similarity.mz_tolerance
    ms1_tol = similarity.ms1_tolerance
    use_rt = similarity.use_rt
    sim_thresh = annotation.similarity_threshold
    min_matched = annotation.min_matched_peaks
    min_peaks = annotation.min_peaks_to_match
    min_matched_pct = annotation.min_matched_pct
    min_wdp = annotation.min_wdp

    # Per-feature precompute, amortized across all candidates:
    #  * float64 query arrays reused by the array-based matcher (S3) so the
    #    composite never rebuilds them per candidate;
    #  * sorted query m/z for the batch upper bounds (S2' / S2");
    #  * widened integer m/z bins of the query for the cheap shared-peak
    #    pre-filter (S2). Bin width == mz_tolerance; each query peak occupies
    #    its bin plus the two neighbours so a ±tol match is never missed.
    q_mz = np.asarray(feature.ms2_mz, dtype=np.float64)
    q_int = np.asarray(feature.ms2_intensity, dtype=np.float64)
    if q_mz.size < min_peaks:
        return []
    q_order = np.argsort(q_mz, kind="stable")
    q_sorted = q_mz[q_order]
    inv_tol = 1.0 / mz_tol if mz_tol > 0 else 0.0
    query_bins_wide: set[int] = set()
    if inv_tol > 0:
        for m in q_mz:
            b = int(m * inv_tol)
            query_bins_wide.add(b - 1)
            query_bins_wide.add(b)
            query_bins_wide.add(b + 1)

    # S2" precompute. Skipped whole when min_wdp <= 0 (AnnotationConfig's default,
    # and GC-MS): the screen must then cost nothing at all. Also skipped when the
    # query itself has a round(mz, 4) collision, which would make SQ_lb a fiction
    # the same way _ROUND4_GAP describes for the reference side.
    wdp_screen = min_wdp > 0.0 and mz_tol > 0.0 and q_mz.size > 0
    if wdp_screen:
        max_q = float(q_int.max())
        wdp_screen = max_q >= 1e-12 and bool(
            np.all(np.diff(q_sorted) > _ROUND4_GAP))
    if wdp_screen:
        nq = q_int / max_q
        q_nq_table = _sparse_range_max_table(np.maximum(nq[q_order], 0.0))
        TQ = float((nq * q_mz).sum())
        SQ0 = float(nq.sum())

    # Batch screening runs bucket by bucket in (-1, 0, +1) order, and each
    # bucket keeps its mz_index order, so surviving candidates reach `hits` in
    # exactly the enumeration order the pre-CSR loop used. `hits.sort` is
    # stable, so equal-score ties keep that order and top-N is unchanged.
    hits = []
    for key in bucket_keys:
        bucket = (csr[key] if csr is not None
                  else _build_bucket_csr(spectra, mz_index[key]))

        # S1: precursor candidate window, tightened from ±1.0 to ±0.5 Da.
        keep = bucket["n_peaks"] >= min_peaks
        np.logical_and(keep, np.abs(bucket["ref_pmz"] - feat_mz) <= 0.5, out=keep)

        # S2': batch Matched% upper bound. This is the screen that makes a
        # 3.35M-spectrum library tractable — it kills, in a few vectorised
        # passes, the candidates the exact scorer would have rejected anyway.
        if min_matched_pct > 0 and mz_tol > 0 and q_sorted.size and keep.any():
            ub_pct = _matched_pct_upper_bound(bucket, q_sorted, feat_mz, mz_tol)
            np.logical_and(keep, ub_pct >= min_matched_pct, out=keep)

        # S2": batch WDP upper bound, orthogonal to S2' (spectral shape, not a
        # significant-peak count). This is the screen that bites: min_wdp rejects
        # three quarters of everything S2' lets through.
        if wdp_screen and keep.any():
            sel = np.flatnonzero(keep)
            ub_wdp = _wdp_upper_bound(bucket, sel, q_sorted, q_nq_table,
                                      TQ, SQ0, feat_mz, mz_tol)
            keep[sel] = ub_wdp >= min_wdp

        for k in np.flatnonzero(keep):
            idx = int(bucket["spec_idx"][k])
            spec = spectra[idx]
            ref_mz_arr = spec.get("mz")
            ref_int_arr = spec.get("intensity")
            if ref_mz_arr is None or ref_int_arr is None:
                continue
            ref_pmz = spec.get("_precursor_mz")

            # S2: cheap binned shared-peak pre-filter. Orthogonal to S2' — it
            # bounds `n_matched`, not `matched_pct` — and it is the only screen
            # that bites when min_matched_peaks > 1 (DDA runs it at 3).
            if inv_tol > 0 and min_matched > 0:
                shift = (ref_pmz - feat_mz) if ref_pmz is not None else 0.0
                if not _passes_prefilter(query_bins_wide, ref_mz_arr, shift,
                                         inv_tol, min_matched):
                    continue

            ref_pmz_val = ref_pmz if ref_pmz is not None else 0.0
            breakdown = composite_similarity_breakdown(
                None, None, mz_tol,
                precursor_query=feat_mz,
                precursor_ref=ref_pmz_val,
                rt_query=rt_query,
                rt_ref=spec.get("_rt", 0.0),
                ms1_tolerance=ms1_tol,
                use_rt=use_rt,
                q_arrays=(q_mz, q_int),
                r_arrays=(np.asarray(ref_mz_arr, dtype=np.float64),
                          np.asarray(ref_int_arr, dtype=np.float64)),
            )
            score = breakdown.score
            n_matched = breakdown.n_matched

            # Gate on the MS-DIAL-faithful bounded Matched% from B2's breakdown
            # (same ≥1%-of-base significant-ref denominator, numerator restricted
            # to significant matches). NOTE: B2 widened `score` to the [0,2]
            # TotalScore range, so the score/matched_pct gate sensitivities shift;
            # retuning thresholds for the new scale is deferred to PR-E (spec §5.3).
            matched_pct = breakdown.matched_pct

            # min_wdp guard (default 0.0 = disabled): reject inflated-query false
            # positives whose weighted dot product (true m/z-weighted spectral
            # shape) is near zero even though matched_pct=1.0 / high rdp push the
            # total over the high-confidence line. A real match keeps a
            # substantial wdp, so this never drops a genuine hit.
            if (score >= sim_thresh and
                    n_matched >= min_matched and
                    matched_pct >= min_matched_pct and
                    breakdown.wdp >= min_wdp):
                hits.append({
                    "spec_idx": idx,
                    "score": score,
                    "n_matched": n_matched,
                    "wdp": breakdown.wdp,
                    "sdp": breakdown.sdp,
                    "rdp": breakdown.rdp,
                    "matched_pct": matched_pct,
                    "total_score": breakdown.total_score,
                })

    hits.sort(key=lambda h: h["score"], reverse=True)

    matches = []
    for i, h in enumerate(hits[:limit]):
        spec = spectra[h["spec_idx"]]
        meta = spec.get("metadata", {})
        matches.append(AnnotationMatch(
            rank=i + 1,
            name=meta.get("name", ""),
            formula=meta.get("formula", ""),
            score=h["score"],
            n_matched=h["n_matched"],
            # Built only for the top-N survivors, and coerced to plain Python
            # floats: the lean loader stores peaks as numpy arrays, and numpy
            # scalars are not JSON-serializable (project save). Mirrors the
            # GC-MS annotator's ref_peaks.
            ref_peaks=[(float(m), float(v))
                       for m, v in zip(spec["mz"], spec["intensity"])],
            ref_precursor_mz=meta.get("precursor_mz"),
            adduct=meta.get("adduct", ""),
            wdp=h["wdp"],
            sdp=h["sdp"],
            rdp=h["rdp"],
            matched_pct=h["matched_pct"],
            total_score=h["total_score"],
        ))
    return matches
