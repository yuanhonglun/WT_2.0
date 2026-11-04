"""GC-MS peak joiner — a port of MS-DIAL's ``GcmsPeakJoiner``.

``MsdialGcMsApi/Algorithm/Alignment/GcmsPeakJoiner.cs`` in MSDIAL5. Four steps,
in this order::

    build_master_list   -> the union of every sample's peaks, frozen
    assign_to_master    -> each sample claims at most one peak per row
    reset_quant_masses  -> each row votes on ONE quantitation ion

What makes GC-MS different from the LC-MS joiner next door is what the criterion
is allowed to look at. A GC-MS feature's ``mz`` is whichever fragment was tallest
in *that one sample* — an argmax, and near-ties resolve differently elsewhere, so
the same compound keys on m/z 73 in one sample and 147 in another. MS-DIAL's
answer is to keep m/z out of the criterion entirely:

* the gate is ``|delta index| <= tolerance`` **AND** ``EI spectrum similarity``;
* the score that ranks survivors is ``ei_factor*EI + index_factor*gauss(delta)``;
* the row's single quantitation ion is decided afterwards, by vote.

There is no m/z term and no m/z pre-filter anywhere in this module. Adding one
re-creates the split this port exists to remove.

Everything here is plain data — dicts, floats and :class:`GcmsPeak` — so the
module imports nothing from any app. See ``config/gcms_alignment.py`` for the
parameters and their MS-DIAL provenance.
"""
from __future__ import annotations

import bisect
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

from metabo_core.algorithms.similarity import ei_similarity
from metabo_core.config.gcms_alignment import GcmsAlignmentConfig

# Nothing here logs: every peak this module declines to place is counted in
# :class:`JoinStats`, and the caller reports the whole block in one line. A
# library that logs per-peak would bury that summary in 16k messages.

Spectrum = tuple[tuple[float, float], ...]

# The "strict" half of MS-DIAL's IsSpectrumMatch (MsScanMatching.cs:441-447).
# These are the defaults of the MsRefSearchParameterBase the GC-MS joiner
# constructs for itself (GcmsPeakJoiner.cs:48 builds one with only the m/z
# tolerances overridden), so in the GC-MS path they are constants, not knobs.
#
# Note this branch is provably redundant at the shipped ei_threshold: when all
# five conditions hold, EI >= (3*0.6 + 2*0.8 + 0.25)/6 = 0.608 > 0.5, so the OR's
# left arm has already fired. It only ever matters if a user raises the threshold
# above 0.608. Implemented anyway — it costs one comparison and keeps the gate
# honest against a re-tuned threshold.
_STRICT_SQ_WDP_CUTOFF = 0.6 * 0.6
_STRICT_SQ_SDP_CUTOFF = 0.6 * 0.6
_STRICT_SQ_RDP_CUTOFF = 0.8 * 0.8
_STRICT_MATCHED_PCT_CUTOFF = 0.25
_STRICT_MIN_MATCHED_PEAKS = 3

# Fraction of the representative spectrum's base peak a candidate quantitation
# ion must exceed to count as "present" (GcmsPeakJoiner.cs:135).
_QUANT_MASS_MIN_REL_INTENSITY = 0.1


@dataclass(frozen=True)
class GcmsPeak:
    """One sample's view of one peak — the minimum the joiner needs.

    Assembled by the app from its feature dict plus the deconvolved spectrum.
    ``index`` is RT or RI depending on ``GcmsAlignmentConfig.index_type``; the
    joiner never asks which, it only compares.
    """

    sample: str
    feature_id: str
    #: RT (min) or RI, per the run's index type. ``<= 0`` means "not measured",
    #: which MS-DIAL treats as never matching (``GetGaussianSimilarity`` returns
    #: -1 and leaves ``isInTolerance`` false, ``MsScanMatching.cs:4569-4575``).
    index: float
    #: This sample's own model ion. Reported and voted on — never matched on.
    mz: float
    height: float
    area: float
    spectrum: Spectrum = ()
    #: Did library annotation claim this peak? Only used to let an annotated
    #: peak displace an unannotated master (``GcmsPeakJoiner.cs:261-268``).
    annotated: bool = False
    #: Best library total score, for picking a row's representative cell
    #: (``DataObjConverter.cs:183-196`` ranks on ``(max TotalScore, height)``).
    annotation_score: float = 0.0


@dataclass
class GcmsSpot:
    """One alignment row: a master peak plus one cell per sample."""

    master: GcmsPeak
    #: sample -> the peak that sample contributed, or ``None`` if it has none.
    cells: dict[str, Optional[GcmsPeak]] = field(default_factory=dict)
    #: The row's single quantitation ion, decided by :func:`reset_quant_masses`.
    quant_mz: float = 0.0
    #: The cell the quantitation ion was validated against.
    representative: Optional[GcmsPeak] = None
    #: Set on rows created by :func:`promote_unclaimed` for a peak another peak
    #: from the same sample already represents. Exported, filtered downstream —
    #: never a reason to delete the row.
    is_duplicate: bool = False
    duplicate_type: str = ""

    @property
    def detected(self) -> list[GcmsPeak]:
        return [p for p in self.cells.values() if p is not None]

    @property
    def n_detected(self) -> int:
        return sum(1 for p in self.cells.values() if p is not None)


@dataclass
class JoinStats:
    """Counts every peak the join drops, so nothing disappears silently."""

    n_master: int = 0
    #: Masters replaced by an annotated target (``GcmsPeakJoiner.cs:261-268``).
    n_master_swapped: int = 0
    #: Features the caller could not even offer to the joiner because they carry
    #: no deconvolved spectrum. Counted here so one stats block accounts for
    #: every peak that entered the stage.
    dropped_no_spectrum: int = 0
    #: Targets with no gated master anywhere in their retention window.
    dropped_no_master: int = 0
    #: Targets that had candidates but lost the cell to a better-scoring peak
    #: from the same sample. MS-DIAL discards these rather than giving them a
    #: row of their own (``GcmsPeakJoiner.cs:293-318``); with
    #: ``conserve_unclaimed_peaks`` they are promoted to a marked row instead
    #: and :attr:`n_promoted_redundant` counts them again.
    dropped_lost_claim: int = 0
    #: Rows no sample ended up contributing to.
    dropped_no_detected: int = 0
    #: Unclaimed peaks given a marked row of their own.
    n_promoted_redundant: int = 0
    #: Unclaimed peaks given a *visible* row: nothing in the frozen master list
    #: represented them, so they are not redundant with anything.
    n_promoted_visible: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "n_master": self.n_master,
            "n_master_swapped": self.n_master_swapped,
            "dropped_no_spectrum": self.dropped_no_spectrum,
            "dropped_no_master": self.dropped_no_master,
            "dropped_lost_claim": self.dropped_lost_claim,
            "dropped_no_detected": self.dropped_no_detected,
            "n_promoted_redundant": self.n_promoted_redundant,
            "n_promoted_visible": self.n_promoted_visible,
        }


# ---------------------------------------------------------------------------
# criterion
# ---------------------------------------------------------------------------

def _retention_gauss(master: GcmsPeak, target: GcmsPeak, tol: float) -> float:
    """``GetGaussianSimilarity`` (``MsScanMatching.cs:4569-4595``).

    Returns ``-1.0`` when either index is missing, which is MS-DIAL's sentinel
    for "no retention information"; the caller then scores on EI alone.
    """
    if master.index <= 0 or target.index <= 0:
        return -1.0
    return math.exp(-0.5 * ((master.index - target.index) / tol) ** 2)


def _compare(
    master: GcmsPeak, target: GcmsPeak, cfg: GcmsAlignmentConfig,
) -> tuple[bool, float]:
    """Gate and score in one pass — ``(passes_gate, score)``.

    MS-DIAL calls ``CompareEIMSScanProperties`` twice per candidate pair, once
    from ``IsSimilarTo`` and once from ``GetSimilality`` (``:142-155``). Both
    calls compute the identical result; the joiner's inner loop runs this on
    every (master, target) pair inside a retention window, so computing the EI
    similarity once and deriving both answers from it halves the work.
    """
    tol = cfg.index_tolerance
    gauss = _retention_gauss(master, target, tol)
    retention_ok = (
        master.index > 0 and target.index > 0
        and abs(master.index - target.index) <= tol
    )

    # Direction matters: master is the query (scan1), target the reference
    # (scan2). RDP and Matched% are computed against the reference end.
    ei = ei_similarity(
        master.spectrum, target.spectrum, mz_tolerance=cfg.spectrum_mz_tolerance,
    )
    strict_ok = (
        ei.wdp >= _STRICT_SQ_WDP_CUTOFF
        and ei.sdp >= _STRICT_SQ_SDP_CUTOFF
        and ei.rdp >= _STRICT_SQ_RDP_CUTOFF
        and ei.matched_pct >= _STRICT_MATCHED_PCT_CUTOFF
        and ei.n_matched_sig >= _STRICT_MIN_MATCHED_PEAKS
    )
    spectrum_ok = ei.ei > cfg.ei_threshold or strict_ok

    if gauss < 0:
        score = ei.ei
    else:
        score = cfg.ei_factor * ei.ei + cfg.index_factor * gauss
    return (retention_ok and spectrum_ok), score


def is_similar_to(
    master: GcmsPeak, target: GcmsPeak, cfg: GcmsAlignmentConfig,
) -> bool:
    """``GcmsPeakJoiner.IsSimilarTo`` (``:142-148``) — two hard gates, AND.

    ``retention_ok AND (ei > ei_threshold OR strict_branch)``. A missing or
    empty spectrum on either side scores ``ei = -1`` and therefore never
    matches, matching MS-DIAL's ``IsComparedAvailable``.
    """
    return _compare(master, target, cfg)[0]


def similarity(
    master: GcmsPeak, target: GcmsPeak, cfg: GcmsAlignmentConfig,
) -> float:
    """``GcmsPeakJoiner.GetSimilality`` (``:150-155``) — ranks gated candidates.

    ``ei_factor * EI + index_factor * gauss(delta index)``, falling back to EI
    alone when either peak has no retention value (``GetTotalSimilarity``,
    ``MsScanMatching.cs:4931-4938``). This only orders candidates that already
    passed :func:`is_similar_to`; it can never admit one.
    """
    return _compare(master, target, cfg)[1]


# ---------------------------------------------------------------------------
# pass 1 — the union master list
# ---------------------------------------------------------------------------

def _bucket(index: float, width: float) -> int:
    return math.ceil(index / width) if width > 0 else 0


def build_master_list(
    peaks_by_sample: Mapping[str, Sequence[GcmsPeak]],
    reference: str,
    cfg: GcmsAlignmentConfig,
    stats: Optional[JoinStats] = None,
) -> list[GcmsPeak]:
    """``GetMasterList`` (``GcmsPeakJoiner.cs:219-279``).

    Seed the buckets with the reference sample, then offer every other sample's
    peaks to them: a peak similar to an existing master is absorbed, one similar
    to nothing becomes a master itself. The result is the **union** of all
    samples and is frozen before any claiming happens — which is what lets a
    compound absent from the reference sample still get a row.

    Bucket width is ``2 * index_tolerance`` and the scan covers ``[b-1, b+1]``
    (``_rtWidth = ceil(tol / (2*tol)) = 1``), so a pair within tolerance is
    always compared regardless of which side of a bucket edge it fell on.

    One deliberate difference from MS-DIAL: the non-reference samples are
    visited in sorted key order rather than in the order the user listed the
    files. When two samples both hold a peak the reference lacks, whichever is
    visited first becomes the master — so without this the master list, and with
    it every row's reported index, would depend on file ordering.
    """
    st = stats if stats is not None else JoinStats()
    width = 2.0 * cfg.index_tolerance
    master: dict[int, list[GcmsPeak]] = defaultdict(list)
    for peak in peaks_by_sample.get(reference, ()):
        master[_bucket(peak.index, width)].append(peak)

    for sample in sorted(k for k in peaks_by_sample if k != reference):
        for target in peaks_by_sample[sample]:
            _set_to_master(master, target, width, cfg, st)

    out = [p for bucket in master.values() for p in bucket]
    out.sort(key=lambda p: (p.index, p.mz, p.sample, p.feature_id))
    st.n_master = len(out)
    return out


def _set_to_master(
    master: dict[int, list[GcmsPeak]],
    target: GcmsPeak,
    width: float,
    cfg: GcmsAlignmentConfig,
    stats: JoinStats,
) -> bool:
    """``SetToMaster`` (``:254-279``). True if ``target`` entered the master list."""
    home = _bucket(target.index, width)
    for b in range(home - 1, home + 2):
        for i, candidate in enumerate(master.get(b, ())):
            if not is_similar_to(candidate, target, cfg):
                continue
            if not candidate.annotated and target.annotated:
                # An annotated peak displaces the unannotated master it matched:
                # the row is going to be named after its master, so prefer the
                # one the library could identify (``:261-268``).
                del master[b][i]
                master[home].append(target)
                stats.n_master_swapped += 1
                return True
            return False
    master[home].append(target)
    return True


# ---------------------------------------------------------------------------
# pass 2 — claiming
# ---------------------------------------------------------------------------

def assign_to_master(
    masters: Sequence[GcmsPeak],
    peaks_by_sample: Mapping[str, Sequence[GcmsPeak]],
    cfg: GcmsAlignmentConfig,
    stats: Optional[JoinStats] = None,
) -> tuple[list[GcmsSpot], list[tuple[GcmsPeak, str]], JoinStats]:
    """``JoinAll`` + ``AlignPeaksToMaster`` (``GcmsPeakJoiner.cs:281-318``).

    One spot per master, one cell per sample, and **no new spots**: a peak that
    loses its cell to a better-scoring peak from the same sample is dropped, not
    given a row of its own. That is what stops one compound from spawning a
    trail of single-sample rows.

    ``max_matchs`` is the claim register that makes "best peak wins" work: a
    master already holding a peak scoring 0.8 will not accept one scoring 0.6.

    Returns ``(spots, unclaimed, stats)``. ``unclaimed`` is every peak that got
    no cell, paired with the reason (``"lost_claim"`` or ``"no_master"``); MS-DIAL
    simply forgets these, and :func:`promote_unclaimed` is what ASFAMProcessor does with
    them instead.

    Two deliberate differences from MS-DIAL:

    * the RI path uses ``max_matchs`` too. MS-DIAL's RI joiner declares the
      array and never writes to it (``:412-435``), so it is stuck at zero and
      every later target unconditionally overwrites the earlier one.
    * a master is only registered in ``max_matchs`` when a target actually
      claims it. MS-DIAL writes the register for every new best-so-far as it
      scans (``:311``), including masters the target then walks past, which
      leaves those masters unclaimable by weaker targets that never lost to
      anything.
    """
    st = stats if stats is not None else JoinStats()
    samples = sorted(peaks_by_sample)
    # Sorted defensively: the window scan below binary-searches the master
    # index, so an unsorted list silently returns wrong candidates rather than
    # failing. MS-DIAL's RI joiner has exactly that bug — ``GetMasterList``
    # returns the buckets unordered (``:371``) and ``AlignPeaksToMaster`` then
    # calls ``LowerBound`` on them (``:417``).
    masters = sorted(
        masters, key=lambda p: (p.index, p.mz, p.sample, p.feature_id),
    )
    spots = [
        GcmsSpot(master=m, cells={s: None for s in samples})
        for m in masters
    ]
    unclaimed: list[tuple[GcmsPeak, str]] = []
    if not spots:
        for sample in samples:
            for p in peaks_by_sample[sample]:
                st.dropped_no_master += 1
                unclaimed.append((p, "no_master"))
        return spots, unclaimed, st

    tol = cfg.index_tolerance
    index_axis = [m.index for m in masters]        # already sorted ascending
    n = len(masters)

    for sample in samples:
        max_matchs = [float("-inf")] * n
        # Sorting the targets removes the last hidden dependency on input
        # ordering: with ties broken by strict ">", whoever is examined first
        # keeps the cell.
        targets = sorted(
            peaks_by_sample[sample],
            key=lambda p: (p.index, p.mz, p.feature_id),
        )
        for target in targets:
            best_idx: Optional[int] = None
            best_score = float("-inf")
            saw_candidate = False
            lo = bisect.bisect_left(index_axis, target.index - tol)
            for i in range(lo, n):
                if masters[i].index > target.index + tol:
                    break
                ok, score = _compare(masters[i], target, cfg)
                if not ok:
                    continue
                saw_candidate = True
                if score > max_matchs[i] and score > best_score:
                    best_idx = i
                    best_score = score
            if best_idx is None:
                if saw_candidate:
                    st.dropped_lost_claim += 1
                    unclaimed.append((target, "lost_claim"))
                else:
                    st.dropped_no_master += 1
                    unclaimed.append((target, "no_master"))
                continue
            evicted = spots[best_idx].cells[sample]
            if evicted is not None:
                # The incoming peak scored higher on this master.
                st.dropped_lost_claim += 1
                unclaimed.append((evicted, "lost_claim"))
            max_matchs[best_idx] = best_score
            spots[best_idx].cells[sample] = target

    return spots, unclaimed, st


def promote_unclaimed(
    unclaimed: Sequence[tuple[GcmsPeak, str]],
    samples: Sequence[str],
    cfg: GcmsAlignmentConfig,
    stats: Optional[JoinStats] = None,
) -> list[GcmsSpot]:
    """Give every peak that got no cell a row of its own, rather than dropping it.

    **On by default** — ``conserve_unclaimed_peaks`` is ``True``. The port
    originally shipped MS-DIAL's behaviour (drop them); the user reviewed the
    cost on 2026-08-03 and chose ``CLAUDE.md``'s "mark, never delete" instead.
    Set the flag to ``False`` to get MS-DIAL's row count back.

    The cost that decided it is measured, not assumed: 5.1% of the features on
    the three real fullscan samples, 38 of 39 of them losing to a peak from
    their own sample that sits on its own master at score 1.0 — the join is
    doing a second, looser intra-sample dedup, and dropping would delete the
    loser. Promoting keeps it as a row flagged ``alignment_redundant``, filtered
    by default but still there to be counted.

    ``no_master`` peaks are promoted **visible**: nothing in the frozen master
    list gated against them, so there is no row they are redundant with. (They
    arise when the master that absorbed a peak is later displaced by an
    annotated one, ``GcmsPeakJoiner.cs:261-268``.)
    """
    st = stats if stats is not None else JoinStats()
    if not cfg.conserve_unclaimed_peaks:
        return []
    out: list[GcmsSpot] = []
    for peak, reason in unclaimed:
        redundant = reason != "no_master"
        if redundant:
            st.n_promoted_redundant += 1
        else:
            st.n_promoted_visible += 1
        cells: dict[str, Optional[GcmsPeak]] = {s: None for s in samples}
        cells[peak.sample] = peak
        out.append(GcmsSpot(
            master=peak,
            cells=cells,
            is_duplicate=redundant,
            duplicate_type="alignment_redundant" if redundant else "",
        ))
    return out


# ---------------------------------------------------------------------------
# pass 3 — the consensus quantitation ion
# ---------------------------------------------------------------------------

def _base_peak_mz(spectrum: Spectrum) -> float:
    if not spectrum:
        return 0.0
    return max(spectrum, key=lambda p: (p[1], -p[0]))[0]


def quant_mass_exists(
    quant_mz: float, spectrum: Spectrum, tolerance: float,
) -> bool:
    """``QuantMassExists`` (``GcmsPeakJoiner.cs:124-140``).

    The voted ion has to actually be in the representative spectrum, and be
    more than 10% of its base peak — a row cannot be quantified on an ion the
    representative barely shows.
    """
    if not spectrum:
        return False
    max_intensity = max(i for _, i in spectrum)
    if max_intensity <= 0:
        return False
    floor = max_intensity * _QUANT_MASS_MIN_REL_INTENSITY
    return any(
        abs(mz - quant_mz) <= tolerance and intensity > floor
        for mz, intensity in spectrum
    )


def _representative(detected: Iterable[GcmsPeak]) -> Optional[GcmsPeak]:
    """``DataObjConverter.GetRepresentativePeak`` (``:183-196``).

    Highest library score, then tallest. The MS2 preference in the original is
    dead code for GC-MS — EI acquisitions have no MS2 — so it is not ported.
    ``feature_id`` breaks the remaining ties so the choice cannot depend on
    dict ordering.
    """
    peaks = list(detected)
    if not peaks:
        return None
    return max(
        peaks,
        key=lambda p: (p.annotation_score, p.height, p.sample, p.feature_id),
    )


def reset_quant_masses(
    spots: Sequence[GcmsSpot],
    cfg: GcmsAlignmentConfig,
    stats: Optional[JoinStats] = None,
) -> tuple[list[GcmsSpot], JoinStats]:
    """``ResetQuantMasses`` (``GcmsPeakJoiner.cs:72-122``).

    Gives every row **one** quantitation ion, so the numbers along a row are all
    read off the same ion. Without this a row can hold sample A's height on
    fragment 73 next to sample B's on fragment 147, and its CV and fold change
    mean nothing.

    The ion is the mode of the detected cells' own ions — cells at least
    ``quant_vote_height_ratio`` of the row's tallest get a vote, votes are
    grouped by nominal mass, and the largest group's mean wins provided the
    representative spectrum actually contains it. Rows with no detected cell at
    all are dropped: there is nothing to quantify and nothing was observed.

    Ties between equally-sized vote groups are broken by summed height, then by
    the lower m/z. MS-DIAL's ``Argmax`` keeps whichever group it met first,
    which is file order.
    """
    st = stats if stats is not None else JoinStats()
    out: list[GcmsSpot] = []
    for spot in spots:
        detected = spot.detected
        if not detected:
            st.dropped_no_detected += 1
            continue
        rep = _representative(detected)
        assert rep is not None  # detected is non-empty
        spot.representative = rep

        if cfg.quant_mass_from_base_peak:
            spot.quant_mz = _base_peak_mz(rep.spectrum) or rep.mz
            out.append(spot)
            continue

        spot.quant_mz = _vote_quant_mass(detected, rep, cfg)
        out.append(spot)
    return out, st


def _vote_quant_mass(
    detected: Sequence[GcmsPeak], rep: GcmsPeak, cfg: GcmsAlignmentConfig,
) -> float:
    max_height = max(p.height for p in detected)
    if max_height > 0:
        voters = [
            p for p in detected
            if p.height >= cfg.quant_vote_height_ratio * max_height
        ]
    else:
        voters = list(detected)
    if not voters:                      # every height was 0 and the ratio is 0
        voters = list(detected)

    groups: dict[float, list[GcmsPeak]] = defaultdict(list)
    for p in voters:
        groups[round(p.mz, cfg.quant_mass_bin_digits)].append(p)
    key = max(
        groups,
        key=lambda k: (len(groups[k]), sum(p.height for p in groups[k]), -k),
    )
    candidate = sum(p.mz for p in groups[key]) / len(groups[key])

    tol = cfg.quant_mass_tolerance
    if quant_mass_exists(candidate, rep.spectrum, tol):
        return candidate
    if quant_mass_exists(rep.mz, rep.spectrum, tol):
        return rep.mz
    return _base_peak_mz(rep.spectrum) or rep.mz
