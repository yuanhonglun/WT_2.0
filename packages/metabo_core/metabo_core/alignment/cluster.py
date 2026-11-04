"""One-pass, route-agnostic grouping of observations into compounds.

The joiner (:mod:`metabo_core.alignment.joiner`) groups by *quantitation route*
first and everything else second, because it was built for a world where the ion
a peak was measured on is also the ion two samples' peaks must agree on. In
all-ion fragmentation that is false in a specific, load-bearing way: one compound
gets an exact precursor in the sample where its MS1 peak cleared the amplitude
floor, and only fragments in the sample where it did not. Route-first grouping
puts those two observations in different rows *before* any evidence is
consulted, and no later pass can undo it — the rows have already collected
unrelated samples by then.

So this module asks one question instead:

1. **Candidate generation** — index every observation on
   ``(segment, rt bucket, isolation window)`` and look at the neighbouring
   buckets and windows. **There is no m/z in the key.** That is the whole point:
   an m/z key is exactly what cannot be shared across routes.
2. **Complete-link clustering** — an observation joins a cluster only when
   :func:`~metabo_core.alignment.identity.identity` returns SAME against *every*
   member. Cosine is not transitive, so single-link would chain a compound into
   its neighbours one 0.71 at a time.

A cluster may hold **several observations of one sample**. That is deliberate,
and it is what lets the "one cell per sample" rule move to the quantitation step
where it belongs, instead of being a hard constraint on grouping enforced by a
matching algorithm that then has to conserve its own losers.

Nothing here reads raw data, decides a quantitation channel, or integrates
anything. It groups.
"""
from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from metabo_core.alignment.identity import (
    IdentityObservation, IdentityState, identity, lowres_evidence,
)
from metabo_core.config import ClusterConfig
from metabo_core.models import MS1, CandidateFeature

logger = logging.getLogger(__name__)

#: ``(sample_id, feature) -> (mz, intensity)`` or ``None``. Same contract as the
#: joiner's, so a caller can hand the same reader to either.
Ms2Reader = Callable[[str, Any], Optional[tuple[np.ndarray, np.ndarray]]]


@dataclass
class Cluster:
    """One compound's observations, from any sample and any quantitation route."""

    #: ``(sample_id, peak)``, in the order they were admitted. May contain more
    #: than one entry per sample.
    members: list[tuple[str, CandidateFeature]] = field(default_factory=list)
    #: Every tier that carried a SAME verdict into this cluster. Empty for a
    #: singleton, which never had to decide anything. Tier 3 is the cross-route
    #: one, so ``3 in tiers`` is the direct measure of what this module bought.
    tiers: set[int] = field(default_factory=set)
    #: Highest cosine seen on any admitting verdict; diagnostic only.
    best_cosine: float = 0.0

    @property
    def sample_ids(self) -> set[str]:
        return {sample_id for sample_id, _peak in self.members}

    @property
    def signal_types(self) -> set[str]:
        return {str(peak.signal_type) for _sample_id, peak in self.members}

    @property
    def is_cross_route(self) -> bool:
        """Does this cluster hold both an exact-precursor and a fragment-only view?"""
        return len(self.signal_types) > 1


@dataclass
class ClusterStats:
    n_observations: int = 0
    #: Observations with no quantitation ion at all. They cannot be gap-filled on
    #: one either, so they take no part — counted, never silent.
    n_unkeyed: int = 0
    n_clusters: int = 0
    n_multi_sample_clusters: int = 0
    #: Clusters holding more than one observation of the same sample. These are
    #: what the quantitation step has to resolve down to one cell.
    n_clusters_with_repeat_samples: int = 0
    n_cross_route_clusters: int = 0
    #: ``tier -> clusters admitting at least one member on that tier``.
    clusters_by_tier: dict[int, int] = field(default_factory=dict)
    #: ``state -> pairwise identity() calls``, over every member comparison made.
    verdicts: dict[str, int] = field(default_factory=dict)
    n_identity_calls: int = 0
    n_ms2_reads: int = 0
    n_ms2_cache_hits: int = 0
    #: Percentiles of "cluster members compared against, per observation" — the
    #: quantity the O(n^2) cost of complete-link is actually paid in.
    comparisons_p50: int = 0
    comparisons_p99: int = 0
    comparisons_max: int = 0


class _SpectrumCache:
    """LRU over ``ms2_reader``, keyed by ``(sample_id, feature_id)``.

    Bounded rather than complete: an unbounded memo would hold every spectrum in
    the run, which grows with the sample count and breaks the peak-memory
    invariant that the whole per-sample spill exists to keep.
    """

    __slots__ = ("_reader", "_maxsize", "_entries", "n_reads", "n_hits")

    def __init__(self, reader: Optional[Ms2Reader], maxsize: int) -> None:
        self._reader = reader
        self._maxsize = max(1, int(maxsize))
        self._entries: OrderedDict[tuple[str, str], list] = OrderedDict()
        self.n_reads = 0
        self.n_hits = 0

    def peaks(self, sample_id: str, peak: CandidateFeature) -> list:
        key = (sample_id, str(peak.feature_id))
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            self.n_hits += 1
            return cached

        spectrum: list = []
        mz = getattr(peak, "ms2_mz", None)
        intensity = getattr(peak, "ms2_intensity", None)
        if mz is not None and intensity is not None and len(mz):
            spectrum = list(zip(np.asarray(mz).tolist(),
                                np.asarray(intensity).tolist()))
        elif self._reader is not None:
            self.n_reads += 1
            arrays = self._reader(sample_id, peak)
            if arrays is not None:
                spectrum = list(zip(np.asarray(arrays[0]).tolist(),
                                    np.asarray(arrays[1]).tolist()))
        self._entries[key] = spectrum
        if len(self._entries) > self._maxsize:
            self._entries.popitem(last=False)
        return spectrum


def _best_total_score(peak: CandidateFeature) -> float:
    if not peak.annotation_matches:
        return 0.0
    selected = int(peak.selected_annotation_idx)
    if not 0 <= selected < len(peak.annotation_matches):
        return 0.0
    return float(
        getattr(peak.annotation_matches[selected], "total_score", 0.0) or 0.0
    )


def observation_priority(
    sample_id: str,
    peak: CandidateFeature,
    is_annotated: Optional[Callable[[CandidateFeature], bool]] = None,
) -> tuple:
    """Admission order: the best-evidenced observations found the clusters.

    Complete-link is order-dependent — whoever arrives first defines what the
    cluster will accept afterwards — so the order has to be both meaningful and
    reproducible. Meaningful: a visible, annotated, tall peak is the one whose
    identity is least in doubt. Reproducible: it ends in a total order over
    ``(sample_id, feature_id)`` and never touches list position.
    """
    annotated = bool(is_annotated(peak)) if is_annotated is not None else bool(
        peak.annotation_matches
    )
    return (
        bool(peak.is_duplicate),
        not annotated,
        -_best_total_score(peak),
        -float(peak.ms1_height or 0.0),
        str(sample_id),
        str(peak.feature_id),
    )


def observation_of(
    peak: CandidateFeature,
    read_spectrum: Callable[[], list],
) -> IdentityObservation:
    """Adapt one candidate to the identity rule's evidence record.

    The exact precursor is ``align_mz`` on the :data:`MS1` route, not
    ``ms1_quant_mz`` alone: the two differ through ``align_mz``'s documented
    fallback to ``ms1_precursor_mz``, which is the ordinary DDA shape, and a peak
    that has one but not the other must not be mistaken for fragment-only.

    ``read_spectrum`` stays a thunk so the two free gates — fragment counts —
    can settle a pair before any spectrum is fetched.
    """
    precursor = peak.ms1_quant_mz
    if precursor is None and peak.quant_route == MS1:
        precursor = peak.align_mz
    lowres_mz, lowres_source = lowres_evidence(peak)
    return IdentityObservation(
        rt_apex=float(peak.rt_apex),
        n_fragments=int(peak.n_fragments or 0),
        read_spectrum=read_spectrum,
        ms1_quant_mz=precursor,
        ms2_rep_ion_mz=peak.ms2_rep_ion_mz,
        window=int(peak.precursor_mz_nominal),
        window_mz=getattr(peak, "isolation_target_mz", None),
        segment=str(peak.segment_name or ""),
        lowres_mz=lowres_mz,
        lowres_source=lowres_source,
    )


def _as_observation(
    peak: CandidateFeature, sample_id: str, cache: _SpectrumCache,
) -> IdentityObservation:
    return observation_of(peak, lambda: cache.peaks(sample_id, peak))


def _bucket(rt: float, width: float) -> int:
    return int(math.floor(float(rt) / float(width)))


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(math.ceil(q * len(ordered))) - 1)
    return int(ordered[max(0, idx)])


def cluster_observations(
    features_by_sample: dict[str, list[CandidateFeature]],
    config: ClusterConfig,
    ms2_reader: Optional[Ms2Reader] = None,
    *,
    is_annotated: Optional[Callable[[CandidateFeature], bool]] = None,
) -> tuple[list[Cluster], ClusterStats]:
    """Group every sample's observations into compounds. One pass, no matching.

    Returns the clusters and the statistics the acceptance criteria ask for.
    Observations without a quantitation ion are counted and dropped, exactly as
    the joiner does: nothing downstream could integrate one.
    """
    stats = ClusterStats()
    cache = _SpectrumCache(ms2_reader, config.ms2_cache_size)

    observations: list[tuple[str, CandidateFeature]] = []
    for sample_id in sorted(features_by_sample):
        for peak in features_by_sample[sample_id]:
            stats.n_observations += 1
            if peak.align_mz is None:
                stats.n_unkeyed += 1
                continue
            observations.append((sample_id, peak))
    observations.sort(
        key=lambda item: observation_priority(item[0], item[1], is_annotated),
    )

    clusters: list[Cluster] = []
    # ``(segment, rt bucket, window) -> cluster indices with a member there``.
    # A cluster is reachable from any of its members' cells; since admission
    # demands SAME with *every* member, a cluster that could accept must have all
    # its members in reach, so looking up one cell is complete.
    index: dict[tuple[str, int, int], list[int]] = {}
    verdicts: dict[str, int] = {}
    comparisons: list[int] = []

    for sample_id, peak in observations:
        probe = _as_observation(peak, sample_id, cache)
        segment = probe.segment
        window = int(probe.window)
        home = _bucket(probe.rt_apex, config.rt_bucket)

        seen: set[int] = set()
        candidates: list[int] = []
        for d_bucket in (-1, 0, 1):
            for d_window in (-1, 0, 1):
                for cluster_idx in index.get(
                    (segment, home + d_bucket, window + d_window), (),
                ):
                    if cluster_idx not in seen:
                        seen.add(cluster_idx)
                        candidates.append(cluster_idx)
        candidates.sort()

        role = int(peak.isotope_index or 0)
        n_compared = 0
        accepted: list[tuple[float, float, int, set[int]]] = []
        for cluster_idx in candidates:
            cluster = clusters[cluster_idx]
            if not config.allow_intra_sample_merge and any(
                member_sample == sample_id
                for member_sample, _member_peak in cluster.members
            ):
                continue
            lowres = math.inf
            cosine_sum = 0.0
            tiers: set[int] = set()
            ok = True
            for member_sample, member_peak in cluster.members:
                member = _as_observation(member_peak, member_sample, cache)
                # An isotope envelope is one compound sharing nearly every
                # fragment, so no spectrum can separate M from M+1. The
                # per-sample isotope stage's ruling therefore stands, and
                # merging across it would absorb the satellite and delete a row
                # the pipeline promises only ever to *mark*.
                #
                # With one exception, and it is the difference between a
                # correction and an over-correction: two observations from
                # *different* samples that both carry an exact precursor. There
                # the ruling was made twice, independently, and it can disagree
                # about which peak of an envelope each sample saw — while the
                # mass tolerance separates genuine partners on its own, +1.003 Da
                # being fifty times the 0.02 Da gate. Enforcing the role there
                # splits one compound over a per-sample bookkeeping difference.
                if (
                    config.respect_isotope_role
                    and int(member_peak.isotope_index or 0) != role
                    and (
                        member_sample == sample_id
                        or not (probe.has_precursor and member.has_precursor)
                    )
                ):
                    ok = False
                    break
                verdict = identity(probe, member, config.identity)
                n_compared += 1
                verdicts[verdict.state.value] = verdicts.get(
                    verdict.state.value, 0,
                ) + 1
                if verdict.state is not IdentityState.SAME:
                    ok = False
                    break
                tiers.add(verdict.tier)
                cosine_sum += verdict.cosine
                delta = verdict.support.get("lowres_mz_delta")
                if delta is not None:
                    lowres = min(lowres, float(delta))
            if ok:
                accepted.append((
                    lowres,
                    -cosine_sum / max(1, len(cluster.members)),
                    cluster_idx,
                    tiers,
                ))
        comparisons.append(n_compared)

        if accepted:
            # SPEC 6.5: where several clusters all say SAME, the one whose
            # isolation-window centroid corroborates best wins, then the one that
            # agrees most strongly, then the older one for reproducibility.
            _lowres, mean_cos, cluster_idx, tiers = min(accepted)
            cluster = clusters[cluster_idx]
            cluster.members.append((sample_id, peak))
            cluster.tiers |= tiers
            cluster.best_cosine = max(cluster.best_cosine, -mean_cos)
        else:
            cluster_idx = len(clusters)
            clusters.append(Cluster(members=[(sample_id, peak)]))

        index.setdefault((segment, home, window), []).append(cluster_idx)

    tier_counts: dict[int, int] = {}
    for cluster in clusters:
        if len({sample_id for sample_id, _ in cluster.members}) > 1:
            stats.n_multi_sample_clusters += 1
        if len(cluster.sample_ids) != len(cluster.members):
            stats.n_clusters_with_repeat_samples += 1
        if cluster.is_cross_route:
            stats.n_cross_route_clusters += 1
        for tier in cluster.tiers:
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

    stats.n_clusters = len(clusters)
    stats.clusters_by_tier = dict(sorted(tier_counts.items()))
    stats.verdicts = dict(sorted(verdicts.items()))
    stats.n_identity_calls = sum(comparisons)
    stats.n_ms2_reads = cache.n_reads
    stats.n_ms2_cache_hits = cache.n_hits
    stats.comparisons_p50 = _percentile(comparisons, 0.50)
    stats.comparisons_p99 = _percentile(comparisons, 0.99)
    stats.comparisons_max = max(comparisons) if comparisons else 0

    logger.info(
        "  Clustering: %d observations -> %d clusters "
        "(%d cross-route, %d with repeat samples); "
        "comparisons p50/p99/max = %d/%d/%d",
        stats.n_observations, stats.n_clusters, stats.n_cross_route_clusters,
        stats.n_clusters_with_repeat_samples,
        stats.comparisons_p50, stats.comparisons_p99, stats.comparisons_max,
    )
    return clusters, stats
