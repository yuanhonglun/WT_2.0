"""GC-MS peak-alignment configuration — a port of MS-DIAL's GC-MS parameters.

Deliberately *not* :class:`~metabo_core.config.alignment.AlignmentConfig`. That
one scores a pair as a weighted sum of an m/z Gaussian, an RT Gaussian and a
spectral cosine, then compares the sum to one threshold. MS-DIAL's GC-MS joiner
does something structurally different: **two hard gates joined by AND**, and
**no m/z term at all**::

    match  <=>  |delta index| <= index_tolerance
                AND (ei > ei_threshold OR strict_branch)

The m/z of a GC-MS feature is whichever fragment happened to be tallest in that
one sample. It is an argmax, so it moves between samples and cannot be an
alignment key; the whole EI spectrum can. The row gets a single quantitation ion
afterwards, by vote — see :func:`~metabo_core.alignment.gcms_joiner.reset_quant_masses`.

Reference: ``MsdialGcMsApi/Algorithm/Alignment/GcmsPeakJoiner.cs`` and
``MsdialGcmsParameter.cs`` in MSDIAL5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

IndexType = Literal["rt", "ri"]


@dataclass
class GcmsAlignmentConfig:
    """Parameters of the MS-DIAL-equivalent GC-MS joiner / gap filler / refiner.

    Every default below is MS-DIAL's own default, cited per field. The old
    ``mz_tolerance`` / ``mz_weight`` / ``rt_weight`` / ``match_threshold`` are
    gone rather than remapped: they encode the weighted-sum criterion this
    module exists to replace, so silently carrying them forward would hide the
    change instead of making it.
    """

    #: Which chromatographic axis the joiner aligns on. ``AlignmentIndexType``
    #: (``MsdialGcmsParameter.cs:28``, default RT). ``"ri"`` requires every
    #: feature to carry an ``ri``; the caller must fall back to ``"rt"`` and
    #: warn when it does not, never align on RT while reporting RI.
    index_type: IndexType = "rt"

    #: ``RetentionTimeAlignmentTolerance`` (``ParameterBase.cs:1153``). Half-width
    #: of the retention gate **and** the sigma of the retention Gaussian in the
    #: score, in minutes.
    rt_tolerance: float = 0.10

    #: ``RetentionIndexAlignmentTolerance`` (``MsdialGcmsParameter.cs:30``), in
    #: RI units. Used in place of :attr:`rt_tolerance` when ``index_type == "ri"``.
    ri_tolerance: float = 20.0

    #: The spectral gate. MS-DIAL overloads ``Ms1AlignmentTolerance`` to mean
    #: "EI similarity threshold" in GC-MS (``MsdialGcmsParameter.cs:15``; the GUI
    #: field is literally labelled *EI similarity tolerance*,
    #: ``GcmsAlignmentParameterSettingModel.cs:38,80-89``). Nothing about m/z.
    ei_threshold: float = 0.5

    #: ``Ms1AlignmentFactor`` (``ParameterBase.cs:1147``) — weight of the EI term
    #: in the *ranking* score. The score only orders candidates that already
    #: passed both gates; it can never admit one.
    ei_factor: float = 0.5

    #: ``RetentionTimeAlignmentFactor`` (``ParameterBase.cs:1145``) — weight of
    #: the retention Gaussian in the ranking score.
    index_factor: float = 0.5

    #: Fragment-matching tolerance inside the EI similarity, in Da.
    #: ``_msMatchParam.Ms1Tolerance`` (``GcmsPeakJoiner.cs:48``). On unit-mass EI
    #: spectra 0.5 Da means "one integer nominal mass is one bin".
    spectrum_mz_tolerance: float = 0.5

    #: Half-width used when checking that a voted quantitation ion really exists
    #: in the representative spectrum, and when extracting the chromatogram to
    #: integrate it on. ``CentroidMs1Tolerance`` in a nominal-mass context
    #: (``GcmsPeakJoiner.cs:125``).
    quant_mass_tolerance: float = 0.5

    #: Only cells at least this fraction of the row's tallest cell get a vote on
    #: the quantitation ion (``GcmsPeakJoiner.cs:101``). Keeps a near-noise cell
    #: from outvoting the real detections.
    quant_vote_height_ratio: float = 0.1

    #: Rounding applied before grouping the votes. ``AccuracyType.IsNominal ? 0 : 2``
    #: (``GcmsPeakJoiner.cs:52``); 0 = group by integer nominal mass.
    quant_mass_bin_digits: int = 0

    #: ``IsRepresentativeQuantMassBasedOnBasePeakMz``
    #: (``MsdialGcmsParameter.cs:34``). True skips the vote and takes the base
    #: peak of the representative spectrum.
    quant_mass_from_base_peak: bool = False

    #: Which sample seeds the master-list buckets. ``AlignmentReferenceFileID``
    #: (``ParameterBase.cs:1143``) defaults to file 0; ``None`` here means
    #: "choose automatically via ``reference_replicate_quality``" — a deliberate
    #: difference (spec X1). File 0 can be the worst sample in the batch, and on
    #: a union master list the reference only decides who occupies a bucket
    #: first, so choosing it on S/N costs nothing and avoids that failure.
    reference_sample: Optional[str] = None

    #: What happens to a peak that gated against a master but lost the cell to a
    #: better-scoring peak from the same sample.
    #:
    #: ``True`` (default) is the platform rule: the peak gets a row of its own
    #: marked ``is_duplicate=True, duplicate_type="alignment_redundant"``, so it
    #: is filtered by default but still present and still countable. ``False``
    #: is MS-DIAL, which discards it (``GcmsPeakJoiner.cs:293-318`` simply never
    #: writes it anywhere).
    #:
    #: **The default is a user decision (2026-08-03).** The port shipped as
    #: ``False`` to match MS-DIAL; the user then reviewed the cost and chose the
    #: platform rule instead, so GC-MS no longer deletes anything and
    #: ``CLAUDE.md``'s "任何 feature 都不被删除，只被标记" holds across all three
    #: modules with no exception.
    #:
    #: The cost that decided it: on the three real fullscan samples the drop
    #: rule removed 39 of 771 features (5.1%), and 38 of them lost to a peak
    #: from their *own* sample sitting on its own master at score 1.0 — the join
    #: performs a second, looser intra-sample dedup (EI > 0.5 within 0.1 min) on
    #: top of ``stage_dedup_components`` and deleted the loser. The pairs are
    #: mostly consecutive feature ids at adjacent RT, i.e. split apices. Marking
    #: them keeps that judgement visible to the reader instead of spending it
    #: silently.
    #:
    #: Set to ``False`` only to reproduce MS-DIAL's row count exactly. Then
    #: :attr:`JoinStats.dropped_lost_claim` / :attr:`JoinStats.dropped_no_master`
    #: report what went, and ``stage_align`` logs it — the drop is never silent.
    conserve_unclaimed_peaks: bool = True

    #: ``IsForceInsertForGapFilling`` (``ParameterBase.cs:1239``, default true):
    #: with no peak top in range, take the point nearest the centre as apex.
    force_insert: bool = True

    #: LWMA smoothing level for the gap-fill chromatogram. ``None`` = inherit the
    #: run's peak-detection smoothing level rather than hard-code a second one.
    gap_fill_smoothing_level: Optional[int] = None

    #: ``PeakCountFilter`` (``PeakAligner.cs:78-91``), in percent of files.
    #: 0 = no filter, as MS-DIAL ships.
    peak_count_filter: float = 0.0

    #: ``NPercentDetectedInOneGroup``. 0 = no filter, as MS-DIAL ships.
    n_percent_detected_in_one_group: float = 0.0

    #: Retention half-width of the refine pass's redundancy gate, in minutes.
    #: ``GcmsAlignmentRefiner.cs:51``. Much tighter than :attr:`rt_tolerance`:
    #: two rows have to be far closer than the match window to be one compound
    #: that got split.
    refine_rt_tolerance: float = 0.025

    #: RI counterpart of :attr:`refine_rt_tolerance` (``GcmsAlignmentRefiner.cs:38``).
    refine_ri_tolerance: float = 2.5

    def __post_init__(self) -> None:
        if self.index_type not in ("rt", "ri"):
            raise ValueError(
                f"GcmsAlignmentConfig.index_type must be 'rt' or 'ri'; "
                f"got {self.index_type!r}"
            )
        if self.index_tolerance <= 0:
            raise ValueError(
                f"GcmsAlignmentConfig: tolerance must be > 0 for "
                f"index_type={self.index_type!r}; got rt_tolerance="
                f"{self.rt_tolerance}, ri_tolerance={self.ri_tolerance}"
            )

    @property
    def index_tolerance(self) -> float:
        """Half-width of the retention gate on whichever axis is in use."""
        return self.ri_tolerance if self.index_type == "ri" else self.rt_tolerance

    @property
    def refine_index_tolerance(self) -> float:
        """Retention half-width of the refine pass, on whichever axis is in use."""
        return (
            self.refine_ri_tolerance if self.index_type == "ri"
            else self.refine_rt_tolerance
        )
