"""MS-DIAL-style peak joiner: union master list, bucketed greedy claim.

Ported from ``LcmsPeakJoiner.cs`` (master list ``:69-123``, claim ``:140-166``)
and ``DataObjConverter.cs:183-196`` (per-spot representative), with one
deliberate departure: the claim score keeps METRA's MS2 cosine term.

**Peaks are matched on the ion they were quantified on** — ``align_mz``, and
never ``precursor_mz``. MS-DIAL can align on the precursor because in DDA a
precursor is a real ion its MS1 picker found. ASFAM is all-ion fragmentation:
nothing selected a precursor, so ``ms1_precursor_mz`` is the intensity-weighted
centroid of a whole 1-Da isolation window and it drifts with whatever else
co-isolated in *this* sample. On the two ASFAM benchmarks it sits more than the
whole 0.02 Da match tolerance away from the quantitation ion in **42.7%** of
peaks — so one compound's peaks landed outside each other's boxes, founded
master peaks of their own, and the row count inflated 2.2x (rice) to 3.2x
(cancer) while gap fill quietly re-integrated the samples the split had orphaned
(66% of all cancer quantitation cells were filled, not detected). We ported
MS-DIAL's joiner faithfully and fed it a key that does not mean the same thing
twice. Gap filling had already learned this lesson and integrates ``align_mz``;
the joiner had not.

A peak also matches only inside its own **quantitation route** (:data:`MS1` vs
:data:`PRODUCT`) — see :attr:`~metabo_core.models.CandidateFeature.quant_route`.
An ``ms2_only`` peak's height is read off a product-ion slice, an ``ms1_detected``
peak's off an MS1 window; both routes derive their *precursor* from the same
1-Da window centroid, so they used to land 0.005 Da apart and merge into one
spot — 11% of rows carried heights measured on two different things, which makes
that row's CV and fold change meaningless. The guard is not optional.

By default, matching is an **identity** test, not a proximity test. Route,
window, ``align_mz``
and RT only say two peaks are *near* each other, and the build step used exactly
that to decide a peak may not found a master of its own — while the claim step
lets a sample put only one peak in a master. So where a sample had k peaks in one
box, k-1 were suppressed from founding a master *and* could not claim one: they
vanished from the run, in no row and no cell. At the RT tolerance real
cross-sample drift demands (0.2 min), distinct compounds share a box routinely —
1,242 (rice) / 2,026 (cancer) of the deleted peaks are a different compound from
the peak that beat them, by MS2 cosine. :func:`_matches` therefore ends in an MS2
identity gate: spectra that disagree do not share a box, and the loser founds its
own master instead of disappearing. Split peaks and duplicates, whose spectra
agree, are suppressed exactly as before and gap fill recovers their quantitation.

ASFAM explicitly exempts the :data:`MS1` route from that MS2 veto and score:
its MS1 features carry all-ion spectra for a co-eluting segment, so spectrum
variation across samples is not molecular identity evidence. PRODUCT keeps the
gate and isolation-window rule. Shared-core callers retain the identity path by
default.

Three things this replaces, and why each mattered:

*Union master list.* The old core seeded the master list from the reference
replicate alone, so a feature detected in every sample *but* the reference was
silently dropped — 34% of them on the ASFAM benchmark. Here the reference only
seeds the buckets; every other sample contributes its unmatched peaks.

*Bucketed candidates + greedy claim.* The old core built a dense
``n_ref x n_target`` similarity matrix and ran the Hungarian algorithm over it:
8.9 GiB and ~5 min per replicate. Only pairs within ``+/-mz_tol`` and
``+/-rt_tol`` can ever match, so the matrix was ~99.997% zeros. Enumerating just
those pairs gives ~16k edges (2 MiB, ~0.6 s), of which only ~150 are contested
— the optimality the Hungarian algorithm bought applies to a handful of edges
and is not worth four orders of magnitude in memory.

*Per-spot representative.* The old core copied every annotation field from the
reference replicate's peak. A spot whose reference-sample spectrum happened to
be poor dragged its annotation down even when another sample had a clean one.
MS-DIAL picks a representative per spot instead: prefer peaks that carry MS2,
then argmax ``(best TotalScore, peak height)``.

The joiner reads MS2 spectra lazily through an ``ms2_reader`` callback — one
call per candidate edge (~16k), not per matrix cell (~600M) — so a caller
holding spectra on disk never has to page them all in.
"""
from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from metabo_core.algorithms.similarity import cosine_similarity
from metabo_core.alignment.identity import (
    IdentityEvidence,
    IdentityState,
    ms2_identity_evidence,
)
from metabo_core.alignment.replicates import reference_replicate_quality
from metabo_core.config import JoinerConfig
from metabo_core.models import MS1, PRODUCT, CandidateFeature, Feature

logger = logging.getLogger(__name__)


# ``(sample_id, feature) -> (mz, intensity)``, or ``None`` when the feature has
# no MS2. Lets the caller decide where spectra live: in memory on the feature,
# or behind a seek pointer into a spill file.
Ms2Reader = Callable[[str, Any], Optional[tuple[np.ndarray, np.ndarray]]]


@dataclass
class FoldedPeak:
    """A natural peak intentionally folded into a SAME-compound cell.

    Keeping the source object and evidence on the spot lets ASFAM emit a
    source-peak -> keeper mapping instead of making a losing detection vanish.
    """

    sample_id: str
    peak: CandidateFeature
    reason: str
    target_peak_id: str
    cosine: float
    n_matched_fragments: int
    #: ``ms2_identity`` for the shared-core fold contract, or an app-specific
    #: evidence kind such as ASFAM's direct ``ms1_natural_peak`` witness.
    evidence_kind: str = "ms2_identity"
    evidence_details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlignmentSpot:
    """One master peak and the peak each sample contributed to it."""
    index: int
    master_mz: float
    master_rt: float
    #: Sample that seeded this master peak — not necessarily the representative.
    origin_sample_id: str
    #: ``sample_id -> peak``. Right after the join only the samples that matched
    #: appear; gap filling adds one peak for every remaining sample.
    peaks: dict[str, CandidateFeature] = field(default_factory=dict)
    representative_sample_id: str = ""
    #: The representative's MS2, fetched through the joiner's cache so it costs
    #: no extra read when the claim step already pulled that spectrum.
    representative_ms2: Optional[tuple[np.ndarray, np.ndarray]] = None
    #: Consensus centres are populated by ASFAM's pre-gap-fill reconciliation.
    #: Other callers leave them unset and retain representative-based behaviour.
    alignment_mz: Optional[float] = None
    alignment_rt: Optional[float] = None
    alignment_window: Optional[int] = None
    alignment_segment: str = ""
    #: Natural peaks deliberately omitted from the one-cell-per-sample matrix,
    #: each backed by reliable SAME evidence and a keeper peak id.
    folded_peaks: list[FoldedPeak] = field(default_factory=list)

    @property
    def representative(self) -> CandidateFeature:
        return self.peaks[self.representative_sample_id]

    @property
    def detected_peaks(self) -> list[CandidateFeature]:
        """Peaks the pipeline actually picked, in sample order.

        The single gate for "may this peak speak for the spot" — representative
        choice, the spot-level mean / CV / S/N, and PR-6's detection counts all
        read it. MS-DIAL spells it ``PeakID >= 0`` and checks it in the same
        three places (``DataObjConverter.cs:104,113,148,183``).
        """
        return [self.peaks[sid] for sid in sorted(self.peaks)
                if self.peaks[sid].gap_fill_status == "detected"]

    @property
    def n_detected(self) -> int:
        return len(self.detected_peaks)

    @property
    def quant_route(self) -> str:
        """The one route every peak here shares — what gap fill will integrate.

        Well-defined because the joiner only lets peaks of one route into a spot,
        and a filled peak inherits the route of the template it was copied from.
        """
        return self.peaks[self.representative_sample_id].quant_route


@dataclass
class JoinStats:
    """Everything the acceptance criteria ask the joiner to report."""
    n_spots: int = 0
    n_seeded: int = 0
    #: ``sample_id -> peaks contributed to the master list``
    n_added_by_sample: dict[str, int] = field(default_factory=dict)
    #: ``sample_id -> (matched, unmatched)``
    matched_by_sample: dict[str, tuple[int, int]] = field(default_factory=dict)
    #: ``sample_id -> spots this sample represents``
    representative_by_sample: dict[str, int] = field(default_factory=dict)
    n_candidate_edges: int = 0
    n_ms2_reads: int = 0
    n_ms2_cache_hits: int = 0
    n_empty_spots: int = 0
    #: ``quant_route -> spots``. Replaces the old nominal-vs-exact m/z split,
    #: which counted the ms2_only spots keyed on an isolation-window floor — an
    #: artefact of keying on the precursor that ``align_mz`` removes outright.
    spots_by_route: dict[str, int] = field(default_factory=dict)
    #: Peaks with no quantitation ion at all (``align_mz is None``). They cannot
    #: be aligned onto one or gap-filled on one, so they take no part in the
    #: join. Measured incidence on both ASFAM benchmarks: 0. Not silent.
    #: Counted by the build step, which sees every peak of every sample once —
    #: the claim step walks the same peaks and must not count them again.
    n_unkeyed_peaks: int = 0
    #: Keyed peaks that ended the claim in no spot at all. A sample may put only
    #: one peak in a master, so where it has k peaks in one master's box, k-1
    #: lose. They are gone from the run — not a row, not a cell.
    n_lost_peaks: int = 0
    #: The lost peaks, judged against the peak of their own sample that beat
    #: them, by the one instrument that owes the alignment nothing: MS2 cosine.
    #: ``same`` is a split peak or a redundant duplicate and dropping it is
    #: right — gap fill re-integrates that sample from the spot that won.
    #: ``different`` is a **deleted compound**, and driving it to ~0 is the whole
    #: point of ``JoinerConfig.ms2_identity_threshold``.
    n_lost_same_compound: int = 0
    n_lost_different_compound: int = 0
    #: Lost peaks the cosine cannot rule on: one side carries no MS2.
    n_lost_unjudgeable: int = 0
    #: Conservation accounting (enabled explicitly by ASFAM).
    n_input_keyed_peaks: int = 0
    n_assigned_peaks: int = 0
    n_collapsed_same_compound: int = 0
    n_promoted_different: int = 0
    n_promoted_unjudgeable: int = 0
    n_unexplained_lost: int = 0
    reference_sample_id: str = ""


# ---------------------------------------------------------------------------
# MS2 access
# ---------------------------------------------------------------------------

def default_ms2_reader(_sample_id: str, feature: Any) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Read a feature's MS2 straight off the object — spectra already in memory."""
    mz = getattr(feature, "ms2_mz", None)
    intensity = getattr(feature, "ms2_intensity", None)
    if mz is None or intensity is None or len(mz) == 0:
        return None
    return mz, intensity


class _Ms2Cache:
    """LRU over ``ms2_reader``, keyed by ``(sample_id, feature_id)``.

    Only *master* spectra go through the cache. A master is scored against every
    target peak whose window covers it, so its spectrum is asked for many times;
    a target's is asked for once per target peak and then never again, so
    caching targets would only evict the masters that still need to be there.
    Callers read a target with :meth:`read_once` and hold it in a local.
    """

    def __init__(self, reader: Ms2Reader, maxsize: int) -> None:
        self._reader = reader
        self._maxsize = max(1, maxsize)
        self._entries: OrderedDict[tuple[str, str], list] = OrderedDict()
        self.n_reads = 0
        self.n_hits = 0

    def _lookup(self, key: tuple[str, str]) -> Optional[list]:
        cached = self._entries.get(key)
        if cached is None:
            return None
        self._entries.move_to_end(key)
        self.n_hits += 1
        return cached

    def _read(self, sample_id: str, feature: Any) -> list:
        self.n_reads += 1
        spectrum = self._reader(sample_id, feature)
        if spectrum is None:
            return []
        mz, intensity = spectrum
        return list(zip(np.asarray(mz).tolist(), np.asarray(intensity).tolist()))

    def read_once(self, sample_id: str, feature: Any) -> list:
        """A spectrum wanted exactly once. Uses the cache but never fills it.

        A target peak that happens to *be* a master peak — every peak matching
        the master it seeded — is already cached by the time we get here,
        because the claim loop primes the window's masters first.
        """
        key = (sample_id, feature.feature_id)
        cached = self._lookup(key)
        return cached if cached is not None else self._read(sample_id, feature)

    def peaks(self, sample_id: str, feature: Any) -> list:
        """Peak list for :func:`cosine_similarity`; empty when there is no MS2."""
        key = (sample_id, feature.feature_id)
        cached = self._lookup(key)
        if cached is not None:
            return cached

        peaks = self._read(sample_id, feature)
        self._entries[key] = peaks
        if len(self._entries) > self._maxsize:
            self._entries.popitem(last=False)
        return peaks

    def arrays(self, sample_id: str, feature: Any) -> tuple[np.ndarray, np.ndarray]:
        """Same spectrum as :meth:`peaks`, as the two float64 arrays a ``Feature`` holds."""
        peaks = self.peaks(sample_id, feature)
        if not peaks:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        arr = np.asarray(peaks, dtype=np.float64)
        return arr[:, 0].copy(), arr[:, 1].copy()


# ---------------------------------------------------------------------------
# Step 4.1 — union master list
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Key:
    """What two peaks must agree on to be the same compound in two samples.

    ``mz`` is ``align_mz`` — the ion the peak was quantified on — never
    ``precursor_mz``. ``window`` is the MS2 isolation window and only constrains
    the :data:`PRODUCT` route (see :func:`_matches`).
    """
    mz: float
    rt: float
    route: str
    window: int


def _key_of(peak: CandidateFeature) -> Optional[_Key]:
    """``None`` when the peak has no quantitation ion — see ``JoinStats.n_unkeyed_peaks``."""
    mz = peak.align_mz
    if mz is None:
        return None
    return _Key(
        float(mz), float(peak.rt_apex),
        peak.quant_route, int(peak.precursor_mz_nominal),
    )


def _stable_peak_key(peak: CandidateFeature) -> tuple:
    """Chemical placement key independent of list insertion order."""
    key = _key_of(peak)
    if key is None:
        return ("~", math.inf, math.inf, math.inf, str(peak.feature_id))
    window = key.window if key.route == PRODUCT else -1
    return (key.route, window, key.mz, key.rt, str(peak.feature_id))


def _assignment_peak_key(peak: CandidateFeature) -> tuple:
    """Stable cell-holder priority for the conservation-only matcher."""
    status_rank = {"detected": 2, "filled": 1, "no_signal": 0}
    selected_score = 0.0
    if peak.annotation_matches:
        selected = int(peak.selected_annotation_idx)
        if 0 <= selected < len(peak.annotation_matches):
            selected_score = float(
                getattr(peak.annotation_matches[selected], "total_score", 0.0)
                or 0.0
            )
    return (
        -status_rank.get(peak.gap_fill_status, -1),
        bool(peak.is_duplicate),
        -int(bool(peak.annotation_matches)),
        -selected_score,
        -(peak.ms1_sn or 0.0),
        -(peak.ms1_height or 0.0),
        _stable_peak_key(peak),
    )


@dataclass
class _MasterPeak:
    #: ``align_mz`` of the peak that seeded this master, not its ``precursor_mz``.
    mz: float
    rt: float
    route: str
    window: int
    sample_id: str
    feature: CandidateFeature


class _Spec:
    """One peak's MS2, read at most once and only if some box wants to judge it.

    The gate is asked about a peak once per master in its m/z neighbourhood, but
    the geometric part of the box rejects nearly all of them for free, and
    ``n_fragments`` answers "too sparse to judge" for free as well. So the read
    is deferred to the first master that is both close enough and judgeable, and
    memoized for the rest.
    """
    __slots__ = ("sample_id", "peak", "_cache", "_peaks")

    def __init__(self, cache: _Ms2Cache, sample_id: str, peak: CandidateFeature) -> None:
        self.sample_id = sample_id
        self.peak = peak
        self._cache = cache
        self._peaks: Optional[list] = None

    @property
    def n_fragments(self) -> int:
        return int(self.peak.n_fragments or 0)

    def peaks(self) -> list:
        if self._peaks is None:
            self._peaks = self._cache.read_once(self.sample_id, self.peak)
        return self._peaks


def _uses_ms2_identity(route: str, config: JoinerConfig) -> bool:
    """Return whether this quantitation route has identity-bearing MS2."""
    return route != MS1 or config.use_ms2_identity_for_ms1


def _same_compound(
    master: _MasterPeak, spec: _Spec, config: JoinerConfig, cache: _Ms2Cache,
) -> bool:
    """Do the two spectra agree that this is one compound?

    Abstains — returns ``True``, i.e. "cannot tell them apart, keep them in one
    box" — whenever either side carries too few fragments for a cosine to mean
    anything. Abstaining *has* to suppress rather than split: a gate that let
    every unjudgeable pair found its own master would give a row to every
    fragment-poor peak in the run.

    Both cheap tests come first: the threshold switch, then ``n_fragments`` off
    the feature object. Only a pair that is geometrically in the box *and*
    judgeable on both sides ever costs a spectrum read.
    """
    if not _uses_ms2_identity(master.route, config):
        return True
    if config.ms2_identity_threshold <= 0:
        return True
    min_frags = config.ms2_identity_min_fragments
    if spec.n_fragments < min_frags:
        return True
    if int(master.feature.n_fragments or 0) < min_frags:
        return True

    master_peaks = cache.peaks(master.sample_id, master.feature)
    target_peaks = spec.peaks()
    if not master_peaks or not target_peaks:
        return True

    if config.use_reliable_ms2_identity:
        evidence = ms2_identity_evidence(
            master_peaks,
            target_peaks,
            mz_tolerance=config.ms2_mz_tolerance,
            same_threshold=config.ms2_identity_threshold,
            min_fragments=config.ms2_identity_min_fragments,
            min_matched_fragments=config.ms2_identity_min_matched_fragments,
        )
        # UNJUDGEABLE remains geometrically matchable across samples.  It may
        # align provisionally, but it can never justify folding a same-sample
        # loser during the conservation pass below.
        return evidence.state is not IdentityState.DIFFERENT

    cos, _n = cosine_similarity(master_peaks, target_peaks, config.ms2_mz_tolerance)
    return cos >= config.ms2_identity_threshold


def _in_box(master: _MasterPeak, key: _Key, mz_tol: float, rt_tol: float) -> bool:
    """The geometric half of the box: same route, same window, close in m/z and RT.

    Proximity only — it says two peaks are *near* each other, not that they are
    the same compound. Split out from :func:`_matches` because the lost-peak
    accounting has to measure the box the gate is *changing*, with an instrument
    that does not change with it (see :func:`_classify_lost_peaks`). Nothing in
    the join itself may use this on its own: build and claim both go through
    :func:`_matches`.

    The isolation window is a condition on the :data:`PRODUCT` route only. An
    ``ms2_only`` peak is keyed on a *fragment*, and common fragments (85.03,
    69.03) are shared by many co-eluting precursors — without the window they
    would merge into one spot. Two :data:`MS1` peaks of one compound, on the
    other hand, can legitimately sit in adjacent isolation windows when the
    precursor is near a window edge, and they share a base peak; requiring the
    same window there would re-split exactly what this change exists to join.
    """
    if master.route != key.route:
        return False
    if master.route == PRODUCT and master.window != key.window:
        return False
    return abs(master.mz - key.mz) <= mz_tol and abs(master.rt - key.rt) <= rt_tol


def _matches(
    master: _MasterPeak, key: _Key, spec: _Spec,
    config: JoinerConfig, cache: _Ms2Cache,
) -> bool:
    """The box test, shared by the build step and the claim step.

    Both steps have to ask the same question or a peak can be suppressed from
    founding a master by a box it then cannot claim — it would vanish from the
    run entirely, and the spot it was suppressed by would gap-fill its sample
    instead. That is why the MS2 gate lives *here*, in the one function both
    steps call, and not in the build step where it would be cheaper: putting it
    in only one of them re-creates exactly the bug it exists to fix. (Scoring is
    still the claim step's alone; this is only the box.)

    By default proximity is necessary but not sufficient. ``align_mz`` + RT +
    route + window
    say two peaks are *near* each other; they do not say they are the same
    compound, and at the RT tolerance the real cross-sample drift demands (0.2
    min) two or three distinct compounds routinely share one box. Suppressing a
    peak on proximity alone is what deleted 1,242 (rice) / 2,026 (cancer) real
    features. So :func:`_same_compound` gets a veto: peaks whose MS2 disagree do
    not share a box, and the loser founds a master of its own instead of
    vanishing. Peaks whose MS2 agree — a split peak, a redundant duplicate — are
    still suppressed, still lose the claim, and gap fill still recovers their
    quantitation, exactly as before.

    Callers may explicitly make MS1 geometry-only when its spectrum is all-ion
    rather than precursor-selected. PRODUCT remains identity-gated regardless.
    """
    if not _in_box(master, key, config.mz_tolerance, config.rt_tolerance):
        return False
    return _same_compound(master, spec, config, cache)


def select_reference_sample(
    features_by_sample: dict[str, list[CandidateFeature]],
    requested: Optional[str] = None,
) -> str:
    """The sample that seeds the master buckets.

    ``requested`` wins when it names a sample that exists; otherwise the same
    ``(n_passes_sn3, mean_sn, n_total)`` quality ordering the old core used. A
    stale name — a saved config pointing at a sample no longer in the run — logs
    and falls back rather than raising: after the union master list the choice
    only decides who claims a bucket first.
    """
    sample_ids = sorted(features_by_sample)
    if not sample_ids:
        raise ValueError("no samples to align")
    if requested:
        if requested in features_by_sample:
            return requested
        logger.warning(
            "  Requested reference sample %r is not in this run; falling back "
            "to automatic selection", requested,
        )
    return max(sample_ids, key=lambda s: reference_replicate_quality(features_by_sample[s]))


def _build_master_list(
    features_by_sample: dict[str, list[CandidateFeature]],
    reference_id: str,
    config: JoinerConfig,
    stats: JoinStats,
    cache: _Ms2Cache,
) -> list[_MasterPeak]:
    """Reference peaks seed the buckets; every other sample adds its unmatched.

    The bucket width is ``2 * mz_tolerance`` and we scan ``+/-1`` bucket, so the
    search covers at least ``+/-mz_tolerance`` around any peak regardless of
    where in its bucket it sits. Buckets are cut on ``align_mz``, the same key
    :func:`_matches` compares — bucketing on one m/z and testing another would
    hide pairs from each other.

    "Unmatched" is :func:`_matches` and nothing else — the same box, MS2 gate
    included, that the claim step will use. It used to be a scoreless arithmetic
    test (MS-DIAL's ``LcmsPeakJoiner.cs:51-53``), which is why the ``cache``
    argument is new: a peak suppressed here must be a peak that can be claimed
    there, so this step now has to be able to read MS2 too. Scoring still
    belongs to the claim step alone.
    """
    width = 2.0 * config.mz_tolerance if config.mz_tolerance > 0 else 1.0

    master: list[_MasterPeak] = []
    buckets: dict[int, list[int]] = {}

    def add(peak: CandidateFeature, key: _Key, sample_id: str) -> None:
        buckets.setdefault(int(key.mz // width), []).append(len(master))
        master.append(_MasterPeak(
            key.mz, key.rt, key.route, key.window, sample_id, peak,
        ))

    def already_covered(key: _Key, spec: _Spec) -> bool:
        b = int(key.mz // width)
        return any(
            _matches(master[i], key, spec, config, cache)
            for k in (b - 1, b, b + 1)
            for i in buckets.get(k, ())
        )

    reference_peaks = features_by_sample[reference_id]
    if config.conserve_detected_peaks:
        reference_peaks = sorted(reference_peaks, key=_stable_peak_key)
    for peak in reference_peaks:
        key = _key_of(peak)
        if key is None:
            stats.n_unkeyed_peaks += 1
            continue
        add(peak, key, reference_id)
    stats.n_seeded = len(master)

    # sorted() so a re-run adds peaks in the same order and produces the same
    # master list. Which sample a near-duplicate peak lands in depends on it.
    for sample_id in sorted(features_by_sample):
        if sample_id == reference_id:
            continue
        added = 0
        sample_peaks = features_by_sample[sample_id]
        if config.conserve_detected_peaks:
            sample_peaks = sorted(sample_peaks, key=_stable_peak_key)
        for peak in sample_peaks:
            key = _key_of(peak)
            if key is None:
                stats.n_unkeyed_peaks += 1
                continue
            if already_covered(key, _Spec(cache, sample_id, peak)):
                continue
            add(peak, key, sample_id)
            added += 1
        stats.n_added_by_sample[sample_id] = added

    return master


# ---------------------------------------------------------------------------
# Step 4.2 — scoring + greedy claim
# ---------------------------------------------------------------------------

def _score(
    dmz: float, drt: float,
    master_peaks: list, target_peaks: list,
    config: JoinerConfig,
) -> float:
    """Three weighted terms, or two renormalized when either side lacks MS2.

    Renormalizing rather than dropping the MS2 term keeps a no-MS2 pair on the
    same [0, 1] scale as an MS2-bearing one, so ``match_threshold`` means the
    same thing on both paths.
    """
    if config.mz_tolerance <= 0 or config.rt_tolerance <= 0:
        return 0.0
    mz_gauss = math.exp(-0.5 * (dmz / config.mz_tolerance) ** 2)
    rt_gauss = math.exp(-0.5 * (drt / config.rt_tolerance) ** 2)

    if not master_peaks or not target_peaks or config.ms2_weight <= 0:
        denom = config.mz_weight + config.rt_weight
        if denom <= 0:
            return 0.0
        return (config.mz_weight * mz_gauss + config.rt_weight * rt_gauss) / denom

    if config.use_reliable_ms2_identity:
        evidence = ms2_identity_evidence(
            master_peaks,
            target_peaks,
            mz_tolerance=config.ms2_mz_tolerance,
            same_threshold=0.0,
            min_fragments=config.ms2_identity_min_fragments,
            min_matched_fragments=config.ms2_identity_min_matched_fragments,
        )
        if evidence.state is IdentityState.UNJUDGEABLE:
            denom = config.mz_weight + config.rt_weight
            if denom <= 0:
                return 0.0
            return (config.mz_weight * mz_gauss
                    + config.rt_weight * rt_gauss) / denom
        ms2_cos = evidence.cosine
    else:
        ms2_cos, _ = cosine_similarity(
            master_peaks, target_peaks, config.ms2_mz_tolerance,
        )
    return (config.mz_weight * mz_gauss
            + config.rt_weight * rt_gauss
            + config.ms2_weight * ms2_cos)


def _claim(
    features_by_sample: dict[str, list[CandidateFeature]],
    master: list[_MasterPeak],
    config: JoinerConfig,
    cache: _Ms2Cache,
    stats: JoinStats,
) -> list[dict[str, CandidateFeature]]:
    """Every sample greedily claims master peaks; one peak per master per sample.

    A target peak scores the masters inside its box — same box the build step
    used, :func:`_matches` — and takes the best one it can actually get. "Can
    actually get" is part of the argmax, not a test after it: MS-DIAL skips a
    master already held by a higher bid while it is still choosing
    (``LcmsPeakJoiner.cs:161``, ``if (factor > maxMatchs[i] && factor >
    matchFactor)``), so a peak that cannot outbid its favourite lands on its best
    *available* master rather than nowhere. ``AlignPeaksToMasterOverlapTest``
    pins that behaviour.

    We used to take the plain argmax first and bid afterwards, so a peak that
    lost its single bid fell through with nothing — and it had no master of its
    own either, because the build step had already suppressed it as a duplicate
    of the very master it just lost. It vanished from the run, and the spot that
    suppressed it gap-filled its sample instead: **3.1% of all detected peaks**
    on the rice benchmark. (The old docstring here claimed this was "exactly as
    LcmsPeakJoiner.cs:148-166 does". It misread the C# — the check is inside the
    argmax loop, not after it.)

    A peak that *is* outbid for a master it already holds is still displaced and
    not re-offered; MS-DIAL does the same, and gap fill recovers the
    quantitation.
    """
    # Sorted by (mz, rt); only the m/z axis is searched, the rest of the box is
    # _matches' job. The rt tiebreak keeps the candidate order deterministic.
    order = np.lexsort(([m.rt for m in master], [m.mz for m in master]))
    sorted_mz = np.asarray([master[i].mz for i in order], dtype=np.float64)

    mz_tol, rt_tol = config.mz_tolerance, config.rt_tolerance
    assignments: list[dict[str, CandidateFeature]] = [{} for _ in master]

    for sample_id in sorted(features_by_sample):
        best_score = np.zeros(len(master))     # per-master high bid, this sample
        holder: dict[int, CandidateFeature] = {}
        n_matched = 0

        for peak in features_by_sample[sample_id]:
            key = _key_of(peak)
            if key is None:
                continue                   # already counted by _build_master_list
            # searchsorted bounds the m/z axis; the route, the isolation window,
            # RT and the MS2 gate are gates, not weights, so they are applied
            # here rather than discounted in the score.
            lo = int(np.searchsorted(sorted_mz, key.mz - mz_tol, side="left"))
            hi = int(np.searchsorted(sorted_mz, key.mz + mz_tol, side="right"))
            spec = _Spec(cache, sample_id, peak)
            candidates = [
                i for i in (int(order[k]) for k in range(lo, hi))
                if _matches(master[i], key, spec, config, cache)
            ]
            if not candidates:
                continue

            stats.n_candidate_edges += len(candidates)
            # Cache hits: _matches primed each surviving master's spectrum when
            # it judged it against this peak.
            use_ms2 = _uses_ms2_identity(key.route, config)
            master_peaks = (
                [cache.peaks(master[i].sample_id, master[i].feature)
                 for i in candidates]
                if use_ms2 else [[] for _ in candidates]
            )
            # Memoized on the _Spec the box just used: this peak is scored
            # against every master in its window, then never looked at again.
            target_peaks = spec.peaks() if use_ms2 else []

            top_score, top_idx = -1.0, -1
            for idx, m_peaks in zip(candidates, master_peaks):
                mp = master[idx]
                score = _score(
                    abs(mp.mz - key.mz), abs(mp.rt - key.rt),
                    m_peaks, target_peaks, config,
                )
                # strict >: ties keep the earlier master. `> best_score[idx]`
                # inside the argmax is what makes the peak fall to its best
                # available master instead of falling out of the run.
                if (score > config.match_threshold
                        and score > best_score[idx]
                        and score > top_score):
                    top_score, top_idx = score, idx

            if top_idx < 0:
                continue

            # Written once, here, and not inside the loop above. MS-DIAL writes
            # `maxMatchs[i] = factor` while still choosing, so a master this peak
            # considered and then abandoned for a better one keeps a bid nobody
            # holds and turns away every later peak. That is a bug in MS-DIAL,
            # not the intent we are porting.
            if top_idx in holder:
                n_matched -= 1             # the peak we just displaced
            best_score[top_idx] = top_score
            holder[top_idx] = peak
            n_matched += 1

        for idx, peak in holder.items():
            assignments[idx][sample_id] = peak

        n_total = len(features_by_sample[sample_id])
        stats.matched_by_sample[sample_id] = (n_matched, n_total - n_matched)

    return assignments


def _claim_deterministic(
    features_by_sample: dict[str, list[CandidateFeature]],
    master: list[_MasterPeak],
    config: JoinerConfig,
    cache: _Ms2Cache,
    stats: JoinStats,
    promoted_tokens: Optional[set[tuple[str, int]]] = None,
) -> list[dict[str, CandidateFeature]]:
    """Maximum-cardinality sparse matching via deterministic augmenting paths.

    Candidate lists are score ordered, but a peak displaced from one master is
    re-offered to its next choice.  This removes both the one-way greedy loss and
    any dependence on the order candidates happened to be stored in ``.mfeat``.
    """
    order = np.lexsort(([m.rt for m in master], [m.mz for m in master]))
    sorted_mz = np.asarray([master[i].mz for i in order], dtype=np.float64)
    mz_tol = config.mz_tolerance
    assignments: list[dict[str, CandidateFeature]] = [{} for _ in master]

    master_tie_key = {
        i: (
            master[i].route,
            master[i].window if master[i].route == PRODUCT else -1,
            master[i].mz,
            master[i].rt,
            master[i].sample_id,
            str(master[i].feature.feature_id),
            i,
        )
        for i in range(len(master))
    }

    for sample_id in sorted(features_by_sample):
        # Earlier vertices retain a contested last seat in the Kuhn traversal.
        # Put natural, visible, better-supported cells first so a pre-hidden
        # duplicate can never make a visible SAME alternative be the fold.
        priority = promoted_tokens or set()
        peaks = sorted(
            features_by_sample[sample_id],
            key=lambda peak: (
                (sample_id, id(peak)) not in priority,
                _assignment_peak_key(peak),
            ),
        )
        keyed: list[CandidateFeature] = []
        edges: list[list[tuple[int, float]]] = []

        for peak in peaks:
            key = _key_of(peak)
            if key is None:
                continue
            lo = int(np.searchsorted(sorted_mz, key.mz - mz_tol, side="left"))
            hi = int(np.searchsorted(sorted_mz, key.mz + mz_tol, side="right"))
            spec = _Spec(cache, sample_id, peak)
            candidates = [
                i for i in (int(order[k]) for k in range(lo, hi))
                if _matches(master[i], key, spec, config, cache)
            ]
            stats.n_candidate_edges += len(candidates)

            use_ms2 = _uses_ms2_identity(key.route, config)
            target_peaks = spec.peaks() if candidates and use_ms2 else []
            scored = []
            for idx in candidates:
                mp = master[idx]
                score = _score(
                    abs(mp.mz - key.mz),
                    abs(mp.rt - key.rt),
                    (cache.peaks(mp.sample_id, mp.feature) if use_ms2 else []),
                    target_peaks,
                    config,
                )
                if score > config.match_threshold:
                    scored.append((idx, score))
            scored.sort(key=lambda item: (-item[1], master_tie_key[item[0]]))
            keyed.append(peak)
            edges.append(scored)

        holder: dict[int, int] = {}

        def augment(peak_idx: int, seen_masters: set[int]) -> bool:
            for master_idx, _score_value in edges[peak_idx]:
                if master_idx in seen_masters:
                    continue
                seen_masters.add(master_idx)
                previous = holder.get(master_idx)
                if previous is None or augment(previous, seen_masters):
                    holder[master_idx] = peak_idx
                    return True
            return False

        for peak_idx in range(len(keyed)):
            augment(peak_idx, set())

        for master_idx, peak_idx in holder.items():
            assignments[master_idx][sample_id] = keyed[peak_idx]
        stats.matched_by_sample[sample_id] = (
            len(holder), len(features_by_sample[sample_id]) - len(holder),
        )

    return assignments


def _identity_between_peaks(
    left: CandidateFeature,
    right: CandidateFeature,
    sample_id: str,
    config: JoinerConfig,
    cache: _Ms2Cache,
) -> IdentityEvidence:
    left_key = _key_of(left)
    right_key = _key_of(right)
    if (
        left_key is not None
        and right_key is not None
        and left_key.route == MS1
        and right_key.route == MS1
        and not config.use_ms2_identity_for_ms1
        and not (
            left.detection_source == "ms2_driven"
            and right.detection_source == "ms2_driven"
        )
    ):
        # ASFAM's ms1_driven spectrum describes the whole AIF segment. Even a
        # reliable cosine is not direct evidence that two MS1 peaks are one
        # chromatographic peak; the app reconciles dual-path aliases from MS1
        # boundaries before entering this shared matcher.
        return IdentityEvidence(
            IdentityState.UNJUDGEABLE,
            0.0,
            0,
            int(left.n_fragments or 0),
            int(right.n_fragments or 0),
        )
    return ms2_identity_evidence(
        cache.peaks(sample_id, left),
        cache.peaks(sample_id, right),
        mz_tolerance=config.ms2_mz_tolerance,
        same_threshold=config.ms2_identity_threshold,
        min_fragments=config.ms2_identity_min_fragments,
        min_matched_fragments=config.ms2_identity_min_matched_fragments,
    )


def _conserving_claim(
    features_by_sample: dict[str, list[CandidateFeature]],
    master: list[_MasterPeak],
    config: JoinerConfig,
    cache: _Ms2Cache,
    stats: JoinStats,
) -> tuple[list[dict[str, CandidateFeature]], dict[int, list[FoldedPeak]]]:
    """Assign, promote every non-SAME loser, and prove peak conservation."""
    keyed_input = [
        (sample_id, peak)
        for sample_id in sorted(features_by_sample)
        for peak in sorted(features_by_sample[sample_id], key=_stable_peak_key)
        if _key_of(peak) is not None
    ]
    stats.n_input_keyed_peaks = len(keyed_input)
    promoted_different: set[tuple[str, int]] = set()
    promoted_unjudgeable: set[tuple[str, int]] = set()

    assignments: list[dict[str, CandidateFeature]] = []
    folded_by_master: dict[int, list[FoldedPeak]] = {}
    max_rounds = len(keyed_input) + 1

    for _round in range(max_rounds):
        assignments = _claim_deterministic(
            features_by_sample,
            master,
            config,
            cache,
            stats,
            promoted_different | promoted_unjudgeable,
        )
        claimed = {
            (sample_id, id(peak))
            for peaks in assignments
            for sample_id, peak in peaks.items()
        }
        holder_of = {
            (master_idx, sample_id): peak
            for master_idx, peaks in enumerate(assignments)
            for sample_id, peak in peaks.items()
        }

        width = 2.0 * config.mz_tolerance if config.mz_tolerance > 0 else 1.0
        buckets: dict[int, list[int]] = {}
        for i, mp in enumerate(master):
            buckets.setdefault(int(mp.mz // width), []).append(i)

        promotions: list[tuple[str, CandidateFeature, _Key, IdentityState]] = []
        candidate_folds: dict[int, list[FoldedPeak]] = {}
        stuck = False

        for sample_id, peak in keyed_input:
            peak_token = (sample_id, id(peak))
            if peak_token in claimed:
                continue
            key = _key_of(peak)
            assert key is not None
            bucket = int(key.mz // width)
            nearby: list[tuple[int, CandidateFeature, IdentityEvidence]] = []
            for k in (bucket - 1, bucket, bucket + 1):
                for master_idx in buckets.get(k, ()):
                    winner = holder_of.get((master_idx, sample_id))
                    if winner is None:
                        continue
                    winner_key = _key_of(winner)
                    if winner_key is None:
                        continue
                    winner_as_master = _MasterPeak(
                        winner_key.mz,
                        winner_key.rt,
                        winner_key.route,
                        winner_key.window,
                        sample_id,
                        winner,
                    )
                    if not _in_box(
                        winner_as_master,
                        key,
                        config.mz_tolerance,
                        config.rt_tolerance,
                    ):
                        continue
                    evidence = _identity_between_peaks(
                        peak, winner, sample_id, config, cache,
                    )
                    nearby.append((master_idx, winner, evidence))

            same = [item for item in nearby
                    if item[2].state is IdentityState.SAME]
            if same:
                target_idx, winner, evidence = min(
                    same,
                    key=lambda item: (
                        -item[2].cosine,
                        -item[2].n_matched_fragments,
                        abs(item[1].rt_apex - peak.rt_apex),
                        str(item[1].feature_id),
                        item[0],
                    ),
                )
                candidate_folds.setdefault(target_idx, []).append(FoldedPeak(
                    sample_id=sample_id,
                    peak=peak,
                    reason="same_compound_claim_loser",
                    target_peak_id=str(winner.feature_id),
                    cosine=evidence.cosine,
                    n_matched_fragments=evidence.n_matched_fragments,
                ))
                continue

            state = (
                IdentityState.DIFFERENT
                if any(item[2].state is IdentityState.DIFFERENT for item in nearby)
                else IdentityState.UNJUDGEABLE
            )
            promoted = (peak_token in promoted_different
                        or peak_token in promoted_unjudgeable)
            if promoted:
                # A promoted peak has a perfect self edge, so exact matching must
                # seat it.  Reaching this branch is an algorithmic canary.
                stuck = True
                continue
            promotions.append((sample_id, peak, key, state))

        if promotions:
            for sample_id, peak, key, state in promotions:
                master.append(_MasterPeak(
                    key.mz, key.rt, key.route, key.window, sample_id, peak,
                ))
                token = (sample_id, id(peak))
                if state is IdentityState.DIFFERENT:
                    promoted_different.add(token)
                else:
                    promoted_unjudgeable.add(token)
            continue

        folded_by_master = candidate_folds
        if stuck:
            break
        break
    else:  # pragma: no cover - the monotone promotion bound is a hard canary
        raise RuntimeError("detected-peak conservation did not converge")

    assigned_tokens = {
        (sample_id, id(peak))
        for peaks in assignments
        for sample_id, peak in peaks.items()
    }
    folded_tokens = {
        (fold.sample_id, id(fold.peak))
        for folds in folded_by_master.values()
        for fold in folds
    }
    input_tokens = {(sample_id, id(peak)) for sample_id, peak in keyed_input}

    stats.n_assigned_peaks = len(assigned_tokens)
    stats.n_collapsed_same_compound = len(folded_tokens)
    stats.n_promoted_different = len(promoted_different)
    stats.n_promoted_unjudgeable = len(promoted_unjudgeable)
    unexplained_tokens = input_tokens - assigned_tokens - folded_tokens
    stats.n_unexplained_lost = len(unexplained_tokens)
    # Retain the legacy counters with honest final semantics: an intentional
    # SAME fold reaches no cell, while promoted DIFFERENT/UNJUDGEABLE peaks do.
    stats.n_lost_peaks = stats.n_collapsed_same_compound + stats.n_unexplained_lost
    stats.n_lost_same_compound = stats.n_collapsed_same_compound
    stats.n_lost_different_compound = 0
    stats.n_lost_unjudgeable = stats.n_unexplained_lost

    if stats.n_unexplained_lost:
        feature_by_token = {
            (sample_id, id(peak)): str(peak.feature_id)
            for sample_id, peak in keyed_input
        }
        examples = ", ".join(
            f"{sample_id}:{feature_by_token[(sample_id, peak_id)]}"
            for sample_id, peak_id in sorted(unexplained_tokens)[:10]
        )
        raise RuntimeError(
            f"detected-peak conservation failed for {stats.n_unexplained_lost} "
            f"peaks ({examples})"
        )
    return assignments, folded_by_master


# ---------------------------------------------------------------------------
# Lost-peak accounting — the acceptance metric of the MS2 identity gate
# ---------------------------------------------------------------------------

#: The ruler, fixed on purpose and deliberately *not* ``ms2_identity_threshold``.
#: It is the instrument PR-3 used to establish the 1,242 (rice) / 2,026 (cancer)
#: baseline, and a metric that moved with the knob it is grading would not be a
#: metric. Tune the gate all you like; "different compound" keeps meaning this.
_LOST_PEAK_SAME_COMPOUND_COS = 0.5


def _classify_lost_peaks(
    features_by_sample: dict[str, list[CandidateFeature]],
    master: list[_MasterPeak],
    assignments: list[dict[str, CandidateFeature]],
    config: JoinerConfig,
    cache: _Ms2Cache,
    stats: JoinStats,
) -> None:
    """Count the peaks the claim dropped, and say what each one *was*.

    A peak is lost when every master in its box is already held by a peak of its
    own sample that outbid it. The question that matters is what the winner and
    the loser are to each other: a split peak (drop it — gap fill re-integrates
    that sample from the winning spot) or a different compound (dropping it
    deletes a real feature, which is the bug the MS2 gate exists to fix).

    Uses :func:`_in_box`, the *geometric* box, not the gated one. The gate
    narrows the box, so measuring through the gated box would hide precisely the
    losses the gate is supposed to prevent and the number would read ~0 no matter
    what the gate did. The winner is looked up in the final assignments, not
    mid-claim, so a peak that claimed a master and was later displaced counts as
    lost — because it is.
    """
    if not master:
        return

    claimed: dict[str, set[str]] = {sid: set() for sid in features_by_sample}
    holder_of: dict[tuple[int, str], CandidateFeature] = {}
    for idx, peaks in enumerate(assignments):
        for sample_id, peak in peaks.items():
            claimed[sample_id].add(peak.feature_id)
            holder_of[(idx, sample_id)] = peak

    mz_tol, rt_tol = config.mz_tolerance, config.rt_tolerance
    width = 2.0 * mz_tol if mz_tol > 0 else 1.0
    buckets: dict[int, list[int]] = {}
    for i, mp in enumerate(master):
        buckets.setdefault(int(mp.mz // width), []).append(i)

    for sample_id in sorted(features_by_sample):
        for peak in features_by_sample[sample_id]:
            if peak.feature_id in claimed[sample_id]:
                continue
            key = _key_of(peak)
            if key is None:
                continue                   # counted as unkeyed, not as lost
            stats.n_lost_peaks += 1

            b = int(key.mz // width)
            winners = [
                holder_of[(i, sample_id)]
                for k in (b - 1, b, b + 1)
                for i in buckets.get(k, ())
                if (i, sample_id) in holder_of
                and _in_box(master[i], key, mz_tol, rt_tol)
            ]
            if not winners:
                stats.n_lost_unjudgeable += 1
                continue
            # The nearest in RT is the one it actually lost the seat to.
            winner = min(winners, key=lambda w: abs(w.rt_apex - peak.rt_apex))

            # The loser is looked at once and never again; the winner beats
            # several losers, so it goes through the LRU.
            loser_ms2 = cache.read_once(sample_id, peak)
            winner_ms2 = cache.peaks(sample_id, winner)
            if not loser_ms2 or not winner_ms2:
                stats.n_lost_unjudgeable += 1
                continue

            if config.use_reliable_ms2_identity:
                evidence = ms2_identity_evidence(
                    loser_ms2,
                    winner_ms2,
                    mz_tolerance=config.ms2_mz_tolerance,
                    same_threshold=_LOST_PEAK_SAME_COMPOUND_COS,
                    min_fragments=config.ms2_identity_min_fragments,
                    min_matched_fragments=config.ms2_identity_min_matched_fragments,
                )
                if evidence.state is IdentityState.UNJUDGEABLE:
                    stats.n_lost_unjudgeable += 1
                elif evidence.state is IdentityState.SAME:
                    stats.n_lost_same_compound += 1
                else:
                    stats.n_lost_different_compound += 1
            else:
                cos, _n = cosine_similarity(
                    loser_ms2, winner_ms2, config.ms2_mz_tolerance,
                )
                if cos >= _LOST_PEAK_SAME_COMPOUND_COS:
                    stats.n_lost_same_compound += 1
                else:
                    stats.n_lost_different_compound += 1


# ---------------------------------------------------------------------------
# Step 4.3 — per-spot representative
# ---------------------------------------------------------------------------

def _best_total_score(feature: CandidateFeature) -> float:
    if not feature.annotation_matches:
        return 0.0
    return float(getattr(feature.annotation_matches[0], "total_score", 0.0) or 0.0)


def _pick_representative(peaks: dict[str, CandidateFeature]) -> str:
    """Prefer peaks carrying MS2, then argmax ``(best TotalScore, height)``.

    Iterating sorted sample ids makes the ``max`` tie-break deterministic. This
    is the fix for "the reference sample's poor spectrum drags the whole spot's
    annotation down": a spot's identity now comes from whichever sample actually
    measured it best.

    Only detected peaks are eligible (``DataObjConverter.cs:183``). Right after
    the join every peak is detected; this is called again after gap filling has
    added peaks that carry a height but no identity.
    """
    detected = [peaks[sid] for sid in sorted(peaks)
                if peaks[sid].gap_fill_status == "detected"]
    with_ms2 = [p for p in detected if p.n_fragments > 0]
    best = max(
        with_ms2 or detected,
        key=lambda p: (_best_total_score(p), p.ms1_height or 0.0),
    )
    for sid in sorted(peaks):
        if peaks[sid] is best:
            return sid
    raise AssertionError("representative not found among the spot's peaks")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def join_spots(
    features_by_sample: dict[str, list[CandidateFeature]],
    config: JoinerConfig,
    ms2_reader: Optional[Ms2Reader] = None,
) -> tuple[list[AlignmentSpot], JoinStats]:
    """Align features across samples into spots. See the module docstring."""
    stats = JoinStats()
    if not features_by_sample:
        return [], stats

    cache = _Ms2Cache(ms2_reader or default_ms2_reader, config.ms2_cache_size)

    reference_id = select_reference_sample(features_by_sample, config.reference_sample)
    stats.reference_sample_id = reference_id
    logger.info(
        "  Reference sample: %s (%d features, quality=%s)%s",
        reference_id,
        len(features_by_sample[reference_id]),
        reference_replicate_quality(features_by_sample[reference_id]),
        " [user-selected]" if config.reference_sample else "",
    )

    master = _build_master_list(features_by_sample, reference_id, config, stats, cache)
    logger.info(
        "  Master list: %d peaks (%d seeded by %s, %d added: %s)",
        len(master), stats.n_seeded, reference_id,
        sum(stats.n_added_by_sample.values()),
        ", ".join(f"{s}=+{n}" for s, n in sorted(stats.n_added_by_sample.items())) or "-",
    )
    if not master:
        return [], stats

    folded_by_master: dict[int, list[FoldedPeak]] = {}
    if config.conserve_detected_peaks:
        assignments, folded_by_master = _conserving_claim(
            features_by_sample, master, config, cache, stats,
        )
    else:
        assignments = _claim(features_by_sample, master, config, cache, stats)
        _classify_lost_peaks(
            features_by_sample, master, assignments, config, cache, stats,
        )

    spots: list[AlignmentSpot] = []
    for idx, (mp, peaks) in enumerate(zip(master, assignments)):
        if not peaks:
            # Every master peak should be claimed at least by the peak that
            # created it (self-match scores 1.0), so this is a canary, not a
            # routine drop.
            stats.n_empty_spots += 1
            continue
        rep_sample = _pick_representative(peaks)
        spot = AlignmentSpot(
            index=len(spots),
            master_mz=mp.mz,
            master_rt=mp.rt,
            origin_sample_id=mp.sample_id,
            peaks=peaks,
            representative_sample_id=rep_sample,
            representative_ms2=cache.arrays(rep_sample, peaks[rep_sample]),
            folded_peaks=list(folded_by_master.get(idx, ())),
        )
        spots.append(spot)
        stats.representative_by_sample[spot.representative_sample_id] = (
            stats.representative_by_sample.get(spot.representative_sample_id, 0) + 1
        )
        route = spot.quant_route
        stats.spots_by_route[route] = stats.spots_by_route.get(route, 0) + 1

    stats.n_spots = len(spots)
    stats.n_ms2_reads = cache.n_reads
    stats.n_ms2_cache_hits = cache.n_hits
    _log_stats(stats)
    return spots, stats


def _log_stats(stats: JoinStats) -> None:
    logger.info("  Spots after join: %d", stats.n_spots)
    logger.info(
        "  MS2 spectra: %d reads, %d cache hits over %d candidate edges",
        stats.n_ms2_reads, stats.n_ms2_cache_hits, stats.n_candidate_edges,
    )
    for sample_id, (matched, unmatched) in sorted(stats.matched_by_sample.items()):
        logger.info(
            "    %s: %d matched, %d unmatched", sample_id, matched, unmatched,
        )
    if stats.n_lost_peaks and stats.n_input_keyed_peaks:
        logger.info(
            "  Natural peaks outside cells: %d (%d reliable SAME folds with "
            "source-to-keeper evidence; %d unexplained)",
            stats.n_lost_peaks,
            stats.n_lost_same_compound,
            stats.n_unexplained_lost,
        )
    elif stats.n_lost_peaks:
        logger.info(
            "  Peaks that reached no spot: %d (%d were the same compound as the "
            "peak that beat them — a split, gap fill recovers it; %d were a "
            "DIFFERENT compound and are deleted from the run; %d unjudgeable)",
            stats.n_lost_peaks, stats.n_lost_same_compound,
            stats.n_lost_different_compound, stats.n_lost_unjudgeable,
        )
    if stats.n_input_keyed_peaks:
        logger.info(
            "  Peak conservation: input=%d, assigned=%d, SAME-folded=%d, "
            "promoted DIFFERENT=%d, promoted UNJUDGEABLE=%d, unexplained=%d",
            stats.n_input_keyed_peaks,
            stats.n_assigned_peaks,
            stats.n_collapsed_same_compound,
            stats.n_promoted_different,
            stats.n_promoted_unjudgeable,
            stats.n_unexplained_lost,
        )
    if stats.n_lost_different_compound:
        logger.warning(
            "  %d peaks whose MS2 says they are a different compound from the "
            "peak that displaced them were dropped by the claim step. They are "
            "in no row and no cell. Raising JoinerConfig.ms2_identity_threshold "
            "lets them found master peaks of their own",
            stats.n_lost_different_compound,
        )
    logger.info(
        "  Representative sample distribution: %s",
        ", ".join(
            f"{s}={n}" for s, n in sorted(stats.representative_by_sample.items())
        ) or "-",
    )
    if stats.n_empty_spots:
        logger.warning(
            "  %d master peaks ended up with no peak at all and were dropped",
            stats.n_empty_spots,
        )
    logger.info(
        "  Spots by quantitation route: %s",
        ", ".join(f"{r}={n}" for r, n in sorted(stats.spots_by_route.items())) or "-",
    )
    if stats.n_unkeyed_peaks:
        logger.warning(
            "  %d peaks carried no quantitation ion (align_mz is None) and took "
            "no part in the join — they can be neither aligned onto one nor "
            "gap-filled on one",
            stats.n_unkeyed_peaks,
        )


def build_feature(
    spot: AlignmentSpot, feature_id: str, n_samples: Optional[int] = None,
) -> Feature:
    """Assemble one :class:`Feature` from a spot, with fields from its representative.

    Everything identity-bearing comes from the representative sample. ``heights``
    and ``areas`` span **every** sample, gap-filled ones included, because that
    is what the exported quantitation matrix is for. The spot-level
    ``mean_height`` / ``mean_area`` / ``cv`` span the **detected** samples only
    (``DataObjConverter.cs:113,148``): a filled value is an integral at a
    prescribed m/z and RT, not evidence the compound was seen, and letting it
    into the mean would quietly redefine what those columns report.

    Note that a ``v > 0`` filter would *look* like it does this and does not: it
    drops the ``no_signal`` fills, whose height is 0, and admits every fill that
    landed on real intensity.

    ``n_samples`` is the denominator of ``detection_rate`` — the whole run, not
    the samples this spot happens to carry a peak for. It defaults to the latter
    only for callers that skip gap filling entirely, where the two coincide for
    every spot the caller can see.

    Does **not** set ``gaussian_similarity`` / ``isotope_index`` /
    ``isotope_group_id`` / ``adduct_group_id`` — those need app-specific
    aggregation, so a caller that wants them fills them in afterwards.
    """
    rep = spot.representative

    heights, areas, statuses = {}, {}, {}
    for sample_id in sorted(spot.peaks):
        peak = spot.peaks[sample_id]
        heights[sample_id] = peak.ms1_height or 0.0
        areas[sample_id] = peak.ms1_area or 0.0
        statuses[sample_id] = peak.gap_fill_status

    detected_ids = [sid for sid in sorted(spot.peaks)
                    if spot.peaks[sid].gap_fill_status == "detected"]
    h_vals = [heights[sid] for sid in detected_ids if heights[sid] > 0]
    a_vals = [areas[sid] for sid in detected_ids if areas[sid] > 0]
    mean_h = float(np.mean(h_vals)) if h_vals else 0.0
    mean_a = float(np.mean(a_vals)) if a_vals else 0.0
    cv_h = float(np.std(h_vals) / mean_h) if mean_h > 0 and len(h_vals) > 1 else 0.0

    ms2_mz, ms2_intensity = spot.representative_ms2 or (
        np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64),
    )

    # spot.n_detected, not a second count off `statuses`: the spot already owns
    # the "may this peak speak for me" gate, and the representative choice, the
    # means above and this column have to be answering the same question.
    n_detected = spot.n_detected
    denominator = n_samples if n_samples else len(spot.peaks)

    alignment_rt = (
        float(spot.alignment_rt)
        if spot.alignment_rt is not None else float(rep.rt_apex)
    )
    alignment_mz = (
        float(spot.alignment_mz)
        if spot.alignment_mz is not None else rep.align_mz
    )

    feature = Feature(
        feature_id=feature_id,
        precursor_mz=rep.precursor_mz,
        rt=alignment_rt,
        rt_left=rep.rt_left,
        rt_right=rep.rt_right,
        signal_type=rep.signal_type,
        ms2_mz=ms2_mz,
        ms2_intensity=ms2_intensity,
        n_fragments=rep.n_fragments,
        heights=heights,
        areas=areas,
        gap_fill_status=statuses,
        n_detected=n_detected,
        detection_rate=(n_detected / denominator) if denominator else 0.0,
        mean_height=mean_h,
        mean_area=mean_a,
        cv=cv_h,
        formula=rep.inferred_formula,
        adduct=rep.adduct_type,
        sn_ratio=rep.ms1_sn or 0.0,
        ms1_isotopes=rep.ms1_isotopes,
        name=rep.matchms_name,
        height_ion_mz=rep.ms2_rep_ion_mz,
        # The ion this spot was matched and quantified on, carried through so the
        # refiner can key redundancy on it too. It is the only m/z on the row that
        # means the same thing twice — see Feature.align_mz.
        align_mz=alignment_mz,
        representative_rt=float(rep.rt_apex),
        alignment_window=spot.alignment_window,
        alignment_segment=spot.alignment_segment,
        detection_source=rep.detection_source,
        mz_source=rep.mz_source,
        mz_confidence=rep.mz_confidence,
        is_duplicate=rep.is_duplicate,
        duplicate_group_id=rep.duplicate_group_id,
        duplicate_type=rep.duplicate_type,
    )
    if rep.annotation_matches:
        feature.annotation_matches = rep.annotation_matches
        feature.selected_annotation_idx = rep.selected_annotation_idx
    return feature


def join_features(
    features_by_sample: dict[str, list[CandidateFeature]],
    config: JoinerConfig,
    ms2_reader: Optional[Ms2Reader] = None,
) -> list[Feature]:
    """``join_spots`` + ``build_feature`` — the whole pipeline for a simple caller."""
    spots, _ = join_spots(features_by_sample, config, ms2_reader)
    return [build_feature(s, f"F{i:05d}") for i, s in enumerate(spots)]
