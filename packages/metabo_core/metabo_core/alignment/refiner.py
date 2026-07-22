"""MS-DIAL's alignment refinement, with every deletion turned into a mark.

Ported from ``LcmsAlignmentRefiner.cs:27-106`` (``GetCleanedSpots`` /
``TryMergeToMaster`` / ``SetAlignmentID``) and ``SpotAction.cs:24-116``
(``MatchResultAnnotationDeduplicator``).

**Why any of this is needed.** PR-4's union master list lets every sample
contribute the peaks it alone saw, which is what stopped 34% of them from being
silently dropped. The price is that one compound whose m/z or RT drifts a little
between samples can seed two master peaks and come out as two adjacent spots.
MS-DIAL solves that by returning only the spots that reached ``cSpots`` — the
rest are gone. METRA never deletes a feature (decision D1), so the losers are
marked ``is_duplicate`` with ``duplicate_type = "cross_sample_redundant"``
instead, and every downstream reader filters on that column.

**Two questions, not one.** "Did the union master list emit this compound twice?"
and "has MS1 already seen this compound, so its MS2-only row is not an MS2-only
detection?" read alike and are answered on different evidence. Conflating them is
what this module used to do, and it got both wrong:

*Same route.* Two MS1 rows, or two product rows. Compare the ion they were
**quantified** on — :attr:`~metabo_core.models.Feature.align_mz` — never the m/z
they *report*. On an ASFAM ms2_only row the reported ``precursor_mz`` is the
intensity-weighted centroid of a whole 1-Da isolation window, read off a cycle
where the compound's MS1 was too weak for any peak picker to find: it is noise.
Keying on it marked unrelated rows and missed real duplicates, and neither error
was measurable, because the key itself was the noise. (The joiner and the gap
filler each learned this separately; this was the last place still keying on the
window centroid.)

*Across routes.* An MS1 row's ion is a precursor, a product row's is a fragment.
**There is no comparable m/z between them, so none is compared.** They are judged
on RT plus MS2 cosine — the one identity signal in ASFAM that does not depend on
the route, since nothing selected a precursor and both spectra come from the same
all-ion segment. The product row is the one marked (:data:`MS1_COVERED`): the
surviving row is then quantified on a standard precursor, comparable with every
other MS1 row and with MS-DIAL, and a visible ``ms2_only`` row keeps its meaning
— a compound **no sample's MS1 ever saw**.

Four pieces, in order:

*Cross-sample redundancy.* Spots claim a slot in a master list in priority
order: the identified ones first, ranked by TotalScore, then everything else by
mean height. A spot lands within the m/z and RT gate of a slot already taken?
It is redundant. MS-DIAL runs four such loops (MSP hit, any reference hit,
TextDB hit, unidentified) — METRA has one library, so the first three collapse
into one. One master list **per route**, for the reason above.

*MS1 coverage.* The cross-route pass, over what the first one left visible.

*Name deduplication.* One compound reported twice under one name is worse than
useless. The lower-scoring spot is renamed ``"Putative: <name>"`` — nothing else
about it changes, not its quantitation and not its ``annotation_matches``.
MS-DIAL deduplicates by library id, then InChIKey, then name; METRA's
``AnnotationMatch`` carries neither id nor InChIKey, so only the name pass
survives the port.

This runs **second**, and only over the rows a reader will see — the ones left
``is_duplicate = False`` by the pass above and by the per-sample dedup stages.
MS-DIAL runs it over what ``GetCleanedSpots`` returned, which is exactly the
rows it is about to emit; under D1 nothing is deleted, so "about to emit" means
"not marked".

Get either half of that wrong and the visible table loses a compound. Let a
redundant row into the pass and it keeps the plain name while the row that
survives the filter is renamed ``"Putative: ..."`` — the compound then reads as
merely putative to everyone downstream. The same happens if an M+1 satellite or
an in-source fragment, marked by a per-sample stage, outscores the peak it came
from. MS-DIAL cannot reach either state: it deletes the first kind and has no
duplicate column to hide the second. Marking rather than deleting exposes both.

*Renumbering by m/z.* Separate, and it does **not** run here — see
:func:`order_spots_by_mz`.

Nothing in this module removes a feature from the list it is handed.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from metabo_core.algorithms.similarity import cosine_similarity
from metabo_core.alignment.identity import IdentityState, ms2_identity_evidence
from metabo_core.alignment.joiner import AlignmentSpot
from metabo_core.config import RefinerConfig
from metabo_core.models import MS1, PRODUCT, Feature

logger = logging.getLogger(__name__)

#: ``ChangeAnnotationToLowScore`` (``SpotAction.cs:112-115``).
PUTATIVE_PREFIX = "Putative: "

#: ``duplicate_type`` minted by :func:`mark_cross_sample_redundant`. The
#: per-sample dedup stages own "isotope" / "adduct" / "isf" / "spectral" and
#: this never overwrites them.
CROSS_SAMPLE_REDUNDANT = "cross_sample_redundant"

#: ``duplicate_type`` minted by :func:`mark_ms1_covered`. Deliberately **not**
#: :data:`CROSS_SAMPLE_REDUNDANT`: that one says "the union master list emitted
#: this compound twice", this one says "MS1 saw this compound, so its MS2-only row
#: is not an MS2-only detection". A reader that cannot tell the two apart cannot
#: count either — and the second is exactly the row METRA's MS2-only claim rests
#: on.
MS1_COVERED = "ms1_covered"
MS1_COVERED_PARTIAL = "ms1_covered_partial"

#: Master-list bucket width, in Da. Any value comfortably above twice the widest
#: m/z gate works; 1 Da keeps a handful of masters per bucket over an ASFAM run's
#: ~930 Da span.
_BUCKET_WIDTH = 1.0


@dataclass
class RefineStats:
    """Everything the acceptance criteria ask the refiner to report."""

    n_features: int = 0
    n_renamed: int = 0
    n_masters: int = 0
    n_marked_redundant: int = 0
    #: Redundant spots a per-sample dedup stage had already claimed. They keep
    #: their original ``duplicate_type``; only the count records that they were
    #: also cross-sample redundant.
    n_redundant_already_marked: int = 0
    #: Unidentified isotope satellites skipped by the second placement loop.
    n_isotope_satellites_skipped: int = 0
    #: Visible product-route rows the cross-route pass examined, and the ones it
    #: marked :data:`MS1_COVERED`. The difference is the MS2-only claim: rows no
    #: sample's MS1 ever saw.
    n_cross_route_candidates: int = 0
    n_marked_ms1_covered: int = 0
    n_ms1_covered_partial: int = 0
    #: ``(product feature_id, covering MS1 feature_id, cosine)``, one per mark.
    #: The covering row is where a reader has to go for the compound's
    #: quantitation, so the pass records it rather than only counting.
    ms1_covered_pairs: list = field(default_factory=list)
    #: Rows with no ``align_mz``, which fall back to the reported precursor m/z.
    #: ``build_feature`` always sets it, so on a pipeline run this is 0 — a
    #: nonzero count means the plumbing broke and the noisy key is back.
    n_no_align_mz: int = 0
    #: ``duplicate_type -> count``, after refinement. Empty type omitted.
    duplicate_type_counts: dict = field(default_factory=dict)
    #: ``n_detected -> spots``.
    detection_histogram: dict = field(default_factory=dict)


AnnotationPredicate = Callable[[Feature], bool]


# ---------------------------------------------------------------------------
# Step 6.4 — renumber by m/z
# ---------------------------------------------------------------------------

def order_spots_by_mz(spots: list[AlignmentSpot]) -> list[AlignmentSpot]:
    """Sort spots by precursor m/z and rewrite ``spot.index`` to the new position.

    ``SetAlignmentID`` (``LcmsAlignmentRefiner.cs:57-67``) is the last thing
    MS-DIAL's refiner does. **Ours has to be the first**, and it runs from stage
    7, not from :func:`refine_features`.

    The reason is ``alignment.eic``. Gap filling writes one chromatogram bundle
    per spot and keys it by ``feature_id``; the store is deliberately not indexed
    by position so that this renumbering cannot make it hand back a neighbour's
    chromatogram. But the key *is* the feature id, so minting the ids after the
    store is written trades one silent corruption for another: the GUI would plot
    the wrong feature and nothing would raise. Ordering the spots before any id
    is minted keeps a single, consistent numbering — and since refinement only
    marks and renames, never deletes or reorders, doing it early is equivalent to
    doing it last.

    Ties break on RT then original index, so a re-run of the same data numbers
    the features the same way.
    """
    def stable_source(spot: AlignmentSpot) -> tuple:
        return tuple(sorted(
            (sample_id, str(peak.feature_id))
            for sample_id, peak in spot.peaks.items()
            if peak.gap_fill_status == "detected"
        ))

    ordered = sorted(
        spots,
        key=lambda s: (
            (s.alignment_mz if s.alignment_mz is not None
             else s.representative.precursor_mz),
            (s.alignment_rt if s.alignment_rt is not None
             else s.representative.rt_apex),
            # The stable chemical signature is ASFAM's tie-break only after it
            # explicitly supplies a consensus centre.  Shared-core callers that
            # leave those fields unset retain the historical ``index`` tie-break
            # exactly, including DDA output numbering.
            (stable_source(s)
             if s.alignment_mz is not None or s.alignment_rt is not None else ()),
            s.index,
        ),
    )
    for position, spot in enumerate(ordered):
        spot.index = position
    return ordered


# ---------------------------------------------------------------------------
# Step 6.1 — annotation name deduplication
# ---------------------------------------------------------------------------

def _total_score(feature: Feature) -> float:
    selected = feature.selected_annotation
    if selected is None:
        return 0.0
    return float(getattr(selected, "total_score", 0.0) or 0.0)


def deduplicate_names(
    features: Iterable[Feature],
    is_annotated: AnnotationPredicate,
    eligible: Optional[set[str]] = None,
) -> int:
    """Rename all but the best-scoring spot of each compound name. Returns the count.

    Only identified spots take part (``SpotAction.cs:33`` filters on
    ``IsReferenceMatched``), so an unidentified spot that happens to carry a
    suggested name is left alone. Grouping is case-insensitive, which is what
    ``:92-94`` compares on.

    ``eligible`` is the set of feature ids that may keep a plain name: the rows a
    reader sees, i.e. the ones still ``is_duplicate = False`` once redundancy has
    been marked. A row that is hidden from the exported table must neither claim
    a compound's name nor be renamed — see the module docstring. ``None`` means
    every feature is eligible.

    Renaming touches ``feature.name`` and nothing else — in particular the
    ``annotated`` column keeps reading ``selected_annotation``, so a renamed row
    stays annotated and the identified-feature count cannot fall because of this
    pass. That is by design: it says "two spots claim this compound, and this is
    the weaker claim", not "this is no longer a hit".
    """
    by_name: dict[str, list[Feature]] = {}
    for feature in features:
        name = (feature.name or "").strip()
        if not name or name.startswith(PUTATIVE_PREFIX) or not is_annotated(feature):
            continue
        if eligible is not None and feature.feature_id not in eligible:
            continue
        by_name.setdefault(name.lower(), []).append(feature)

    renamed = 0
    for group in by_name.values():
        if len(group) < 2:
            continue
        # The keeper is the highest TotalScore; feature_id breaks ties so the
        # choice does not depend on list order.
        group.sort(key=lambda f: (-_total_score(f), f.feature_id))
        for loser in group[1:]:
            loser.name = PUTATIVE_PREFIX + (loser.name or "")
            renamed += 1
    return renamed


# ---------------------------------------------------------------------------
# Step 6.2 — cross-sample redundancy
# ---------------------------------------------------------------------------

def key_mz(feature: Feature) -> float:
    """The m/z to compare — the ion the row was **quantified** on.

    Not ``precursor_mz``, which is only what the row *reports*. See the module
    docstring, and :attr:`~metabo_core.models.Feature.align_mz`.

    The fallback exists for a ``Feature`` no joiner built (a hand-made one, or an
    old project file): ``build_feature`` always sets ``align_mz``, and
    :attr:`RefineStats.n_no_align_mz` counts anything that reaches this branch on
    a pipeline run, where it must be 0.
    """
    return feature.align_mz if feature.align_mz is not None else feature.precursor_mz


class _MasterSlots:
    """The spots that have claimed a slot, bucketed on the quantitation ion.

    One of these **per quantitation route**. Buckets and gate are cut on the same
    quantity — :func:`key_mz` — because an index that disagrees with the gate it
    indexes hides masters from the far side of a boundary, silently and only for
    some m/z.
    """

    def __init__(self) -> None:
        self._buckets: dict[int, list[Feature]] = {}
        self.n = 0

    def collides(self, feature: Feature, config: RefinerConfig) -> bool:
        """Is ``feature`` inside the gates of a slot already taken?

        The gates are the spot's own (``TryMergeToMaster`` recomputes ``ms1Tol``
        from the candidate's m/z, not the master's), and both comparisons are
        strict ``<``, as in ``:100-103``.

        The bucket span is derived from the gate rather than fixed at +/-1: at
        the default tolerance one bucket is 25x wider than the gate, but a caller
        who widens ``mz_tolerance`` past the bucket would otherwise start missing
        masters on the far side of a boundary — silently, and only for some m/z.
        """
        mz, rt = key_mz(feature), feature.rt
        mz_gate, rt_gate = config.mz_gate(mz), config.rt_gate
        key = int(mz // _BUCKET_WIDTH)
        span = int(mz_gate // _BUCKET_WIDTH) + 1
        for k in range(key - span, key + span + 1):
            for master in self._buckets.get(k, ()):
                if not (abs(key_mz(master) - mz) < mz_gate
                        and abs(master.rt - rt) < rt_gate):
                    continue
                if (
                    config.require_product_window_match
                    and feature.quant_route == PRODUCT
                    and feature.alignment_window != master.alignment_window
                ):
                    continue
                if config.use_reliable_ms2_identity:
                    evidence = ms2_identity_evidence(
                        _ms2_peaks(feature),
                        _ms2_peaks(master),
                        mz_tolerance=config.ms2_mz_tolerance,
                        same_threshold=config.same_route_ms2_identity_threshold,
                        min_fragments=config.ms2_identity_min_fragments,
                        min_matched_fragments=(
                            config.ms2_identity_min_matched_fragments
                        ),
                    )
                    if evidence.state is not IdentityState.SAME:
                        continue
                return True
        return False

    def claim(self, feature: Feature) -> None:
        key = int(key_mz(feature) // _BUCKET_WIDTH)
        self._buckets.setdefault(key, []).append(feature)
        self.n += 1


def mark_cross_sample_redundant(
    features: Iterable[Feature],
    is_annotated: AnnotationPredicate,
    config: RefinerConfig,
    stats: RefineStats,
) -> None:
    """Mark, but never drop, the spots that duplicate a better one nearby.

    Placement order is MS-DIAL's, collapsed from four loops to two because METRA
    matches against a single library:

    1. identified spots, best TotalScore first (``:32-44``);
    2. everything else, tallest first, **skipping unidentified isotope
       satellites** (``:47-51``).

    That skip is why this function cannot run before stage 7 has copied
    ``isotope_index`` off the representative peak: with every index still 0, an
    M+1 satellite that happens to be taller than some other compound's
    monoisotopic peak a hundredth of a Dalton away would take the slot and mark
    the real peak redundant. The result would be wrong and every test would pass.

    A spot a per-sample dedup stage already marked keeps that mark — "isotope"
    says more than "cross_sample_redundant" does — but it still competes for a
    slot, and can hold one.

    **One master list per quantitation route.** A row quantified on a precursor
    and a row quantified on a fragment do not have an m/z to compare, so they
    never compete for the same slot and the m/z gate never sees the pair. What is
    left of that question — is the compound the same? — is
    :func:`mark_ms1_covered`'s, and it is answered on MS2.
    """
    if not config.same_route_redundancy:
        return

    annotated, others = [], []
    for feature in features:
        (annotated if is_annotated(feature) else others).append(feature)

    annotated.sort(key=lambda f: (-_total_score(f), f.feature_id))
    others.sort(key=lambda f: (-f.mean_height, f.feature_id))

    masters: dict[str, _MasterSlots] = defaultdict(_MasterSlots)

    def place(feature: Feature) -> None:
        if (
            config.visible_keepers_only
            and feature.is_duplicate
        ):
            # It remains hidden for its existing reason, but cannot establish a
            # slot that subsequently hides a visible row.
            return
        slots = masters[feature.quant_route]
        if not slots.collides(feature, config):
            slots.claim(feature)
            return
        if feature.is_duplicate and feature.duplicate_type:
            stats.n_redundant_already_marked += 1
            return
        feature.is_duplicate = True
        feature.duplicate_type = CROSS_SAMPLE_REDUNDANT
        stats.n_marked_redundant += 1

    for feature in annotated:
        place(feature)
    for feature in others:
        if feature.isotope_index > 0:
            stats.n_isotope_satellites_skipped += 1
            continue
        place(feature)

    stats.n_masters = sum(slots.n for slots in masters.values())


# ---------------------------------------------------------------------------
# Step 6.3 — MS1 coverage (cross-route)
# ---------------------------------------------------------------------------

def _ms2_peaks(feature: Feature) -> list:
    return [(float(mz), float(intensity))
            for mz, intensity in zip(feature.ms2_mz, feature.ms2_intensity)]


def mark_ms1_covered(
    features: Iterable[Feature], config: RefinerConfig, stats: RefineStats,
) -> None:
    """Mark the product-route rows whose compound a visible MS1 row already holds.

    The question is **not** "are these the same peak" — they are not, they are two
    ions of one compound, measured on two chromatograms. It is "has MS1 seen this
    compound anywhere in the run", because a visible ``ms2_only`` row is METRA's
    claim that it has *not*, and that claim is what the MS2-only count means.

    So: no m/z is compared. A product row is covered when a visible MS1 row sits
    within the same RT gate and their MS2 spectra agree at
    ``config.ms2_identity_threshold``. Both rows' spectra come off the same
    all-ion segment, so the same compound scores high; MSDEC pulls co-eluting
    compounds in one isolation window apart, so different compounds do not. On a
    tie the highest cosine wins, then the lowest feature id, so a re-run marks
    the same rows.

    **The MS1 row survives** and the product row is marked. It is the one
    quantified on a standard precursor — comparable with every other MS1 row and
    with MS-DIAL — and leaving it visible is what keeps "visible ``ms2_only``" a
    statement about MS1 rather than about which route happened to win a slot. The
    price is real and is measured, not assumed: a sample that had the compound's
    MS2 but not its MS1 was *detected* on the product row and is only *gap-filled*
    on the surviving MS1 row. Nothing here moves that quantitation across — the
    product row keeps every number it had, one column away, per D1.

    Only rows still visible take part, on both sides. A hidden row is one no
    reader counts; letting it cover a product row would hide a detection on the
    authority of a row nobody sees, and a hidden product row is already hidden.
    """
    rt_gate = config.rt_gate
    threshold = config.ms2_identity_threshold
    if rt_gate <= 0:
        return

    coverers: dict[int, list[Feature]] = defaultdict(list)
    candidates: list[Feature] = []
    for feature in features:
        if feature.is_duplicate:
            continue
        if feature.quant_route == MS1:
            coverers[int(feature.rt // rt_gate)].append(feature)
        elif feature.quant_route == PRODUCT:
            candidates.append(feature)

    if not coverers or not candidates:
        return

    peaks: dict[str, list] = {}

    def spectrum(feature: Feature) -> list:
        cached = peaks.get(feature.feature_id)
        if cached is None:
            cached = peaks[feature.feature_id] = _ms2_peaks(feature)
        return cached

    stats.n_cross_route_candidates = len(candidates)
    candidates.sort(key=lambda f: f.feature_id)

    for candidate in candidates:
        mine = spectrum(candidate)
        if not mine:
            continue
        key = int(candidate.rt // rt_gate)
        best: Optional[tuple[float, str, Feature]] = None
        # The gate is one bucket wide, so a coverer within it is at most one
        # bucket away.
        for k in (key - 1, key, key + 1):
            for coverer in coverers.get(k, ()):
                if abs(coverer.rt - candidate.rt) >= rt_gate:
                    continue
                theirs = spectrum(coverer)
                if not theirs:
                    continue
                if config.require_cross_route_window_match:
                    if (
                        candidate.alignment_window is not None
                        and coverer.alignment_window is not None
                        and candidate.alignment_window != coverer.alignment_window
                    ):
                        continue
                    if (
                        candidate.alignment_segment
                        and coverer.alignment_segment
                        and candidate.alignment_segment != coverer.alignment_segment
                    ):
                        continue
                if config.use_reliable_ms2_identity:
                    evidence = ms2_identity_evidence(
                        mine,
                        theirs,
                        mz_tolerance=config.ms2_mz_tolerance,
                        same_threshold=threshold,
                        min_fragments=config.ms2_identity_min_fragments,
                        min_matched_fragments=(
                            config.ms2_identity_min_matched_fragments
                        ),
                    )
                    if evidence.state is not IdentityState.SAME:
                        continue
                    score = evidence.cosine
                else:
                    score, _ = cosine_similarity(
                        mine, theirs, config.ms2_mz_tolerance,
                    )
                    if score < threshold:
                        continue
                bid = (-score, coverer.feature_id, coverer)
                if best is None or bid[:2] < best[:2]:
                    best = bid
        if best is None:
            continue
        coverer = best[2]
        if config.preserve_cross_route_unique_detections:
            candidate.alignment_related_feature_id = coverer.feature_id
            product_detected = {
                sample_id for sample_id, status in candidate.gap_fill_status.items()
                if status == "detected"
            }
            ms1_detected = {
                sample_id for sample_id, status in coverer.gap_fill_status.items()
                if status == "detected"
            }
            if not product_detected.issubset(ms1_detected):
                candidate.alignment_relation = MS1_COVERED_PARTIAL
                stats.n_ms1_covered_partial += 1
                continue
        candidate.is_duplicate = True
        candidate.duplicate_type = MS1_COVERED
        if config.preserve_cross_route_unique_detections:
            candidate.alignment_relation = MS1_COVERED
        stats.n_marked_ms1_covered += 1
        stats.ms1_covered_pairs.append(
            (candidate.feature_id, coverer.feature_id, -best[0]),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def refine_features(
    features: list[Feature],
    config: RefinerConfig,
    is_annotated: AnnotationPredicate,
) -> RefineStats:
    """Deduplicate names, mark cross-sample redundancy, report. Mutates in place.

    ``is_annotated`` is passed in rather than recomputed here: the thresholds
    that define it are app config, and this module may not import app code. Give
    it :func:`metabo_core.annotation.is_high_confidence` bound to the same
    :class:`~metabo_core.config.annotation.ConfidenceConfig` the export uses, or
    the two loops below will partition the spots differently from the
    ``annotated`` column they are meant to agree with.

    ``n_detected`` / ``detection_rate`` are not computed here — the joiner sets
    them off ``AlignmentSpot.n_detected`` when it builds the feature, so there is
    one detection gate rather than three. This function only reports them.

    Both redundancy passes run before names are deduplicated, and the cross-route
    one runs over what the same-route one left visible. The module docstring says
    why the order matters: a name may only be claimed by a row a reader sees.
    """
    stats = RefineStats(n_features=len(features))
    if not features:
        return stats

    stats.n_no_align_mz = sum(1 for f in features if f.align_mz is None)

    mark_cross_sample_redundant(features, is_annotated, config, stats)
    mark_ms1_covered(features, config, stats)
    visible = {f.feature_id for f in features if not f.is_duplicate}
    stats.n_renamed = deduplicate_names(features, is_annotated, visible)

    stats.duplicate_type_counts = dict(
        Counter(f.duplicate_type for f in features if f.is_duplicate and f.duplicate_type)
    )
    stats.detection_histogram = dict(Counter(f.n_detected for f in features))
    _log_stats(stats, config)
    return stats


def _log_stats(stats: RefineStats, config: RefinerConfig) -> None:
    logger.info(
        "  Refine: %d spots, gates |dmz| < %.4f Da (x m/z 500 above), |drt| < %.4f min",
        stats.n_features, config.mz_tolerance, config.rt_gate,
    )
    if stats.n_no_align_mz:
        # build_feature always sets align_mz. Anything here fell back to the
        # reported precursor m/z, which on an ms2_only row is a window centroid
        # measured off noise — the exact key this pass exists to stop using.
        logger.warning(
            "    %d spots carry no align_mz and were keyed on the reported "
            "precursor m/z", stats.n_no_align_mz,
        )
    logger.info(
        "    Name dedup: %d spots renamed to \"%s...\"", stats.n_renamed, PUTATIVE_PREFIX,
    )
    logger.info(
        "    Redundancy: %d spots hold a master slot (per route, keyed on "
        "align_mz), %d newly marked %s, %d already marked by a per-sample stage, "
        "%d isotope satellites skipped",
        stats.n_masters, stats.n_marked_redundant, CROSS_SAMPLE_REDUNDANT,
        stats.n_redundant_already_marked, stats.n_isotope_satellites_skipped,
    )
    logger.info(
        "    MS1 coverage: %d of %d visible product-route spots marked %s "
        "(MS2 cosine >= %.2f, |drt| < %.4f min) -> %d stay MS2-only",
        stats.n_marked_ms1_covered, stats.n_cross_route_candidates, MS1_COVERED,
        config.ms2_identity_threshold, config.rt_gate,
        stats.n_cross_route_candidates - stats.n_marked_ms1_covered,
    )
    total_dup = sum(stats.duplicate_type_counts.values())
    logger.info(
        "    duplicate_type: %s (%d of %d spots, %d unique)",
        ", ".join(f"{k}={v}" for k, v in sorted(stats.duplicate_type_counts.items())) or "-",
        total_dup, stats.n_features, stats.n_features - total_dup,
    )
    logger.info(
        "    detection_rate: %s",
        ", ".join(
            f"{n}/N={c}" for n, c in sorted(stats.detection_histogram.items(), reverse=True)
        ) or "-",
    )
