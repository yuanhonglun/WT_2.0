"""Tests for the MS-DIAL-equivalent GC-MS joiner.

The two headline tests are :func:`test_no_mz_term_in_gate` and
:func:`test_spectrum_can_veto` — they pin the two failure directions the port
exists to fix (spec §5 examples 1 and 2). Loosening either one to make some
other test pass defeats the whole exercise.
"""
from __future__ import annotations

import pytest

from metabo_core.alignment.gcms_joiner import (
    GcmsPeak,
    JoinStats,
    assign_to_master,
    build_master_list,
    is_similar_to,
    promote_unclaimed,
    quant_mass_exists,
    reset_quant_masses,
    similarity,
)
from metabo_core.config.gcms_alignment import GcmsAlignmentConfig

# Six peaks so the MS-DIAL peak-count penalty is 1.0 and a self-match scores
# exactly 1.0; that keeps the assertions about the 0.5 gate unambiguous.
SPEC_X = (
    (43.0, 999.0), (58.0, 420.0), (71.0, 310.0),
    (91.0, 180.0), (105.0, 90.0), (121.0, 55.0),
)
SPEC_X_DRIFTED = (
    (43.0, 970.0), (58.0, 450.0), (71.0, 295.0),
    (91.0, 200.0), (105.0, 80.0), (121.0, 60.0),
)
# Shares no m/z with SPEC_X: a different compound that happens to co-elute.
SPEC_Y = (
    (147.0, 999.0), (163.0, 500.0), (191.0, 260.0),
    (207.0, 120.0), (221.0, 70.0), (233.0, 40.0),
)


def cfg(**kw) -> GcmsAlignmentConfig:
    return GcmsAlignmentConfig(**kw)


def peak(
    sample: str, fid: str, index: float, mz: float, *,
    height: float = 1000.0, spectrum=SPEC_X, annotated: bool = False,
    annotation_score: float = 0.0,
) -> GcmsPeak:
    return GcmsPeak(
        sample=sample, feature_id=fid, index=index, mz=mz,
        height=height, area=height * 0.1, spectrum=spectrum,
        annotated=annotated, annotation_score=annotation_score,
    )


# ---------------------------------------------------------------------------
# the criterion
# ---------------------------------------------------------------------------

def test_no_mz_term_in_gate() -> None:
    """A2 — same compound, but the tallest fragment moved from 73 to 147.

    delta m/z is 74 Da, which the old criterion pre-filtered away with
    ``abs(dmz) > 3 * mz_tolerance: continue``. The EI spectra agree, so this is
    one compound and must land in one row.
    """
    a = peak("s1", "F0", 10.00, 73.0, spectrum=SPEC_X)
    b = peak("s2", "F0", 10.01, 147.0, spectrum=SPEC_X_DRIFTED)
    assert abs(a.mz - b.mz) == 74.0
    assert is_similar_to(a, b, cfg())


def test_spectrum_can_veto() -> None:
    """A1 — same model ion, RT 0.05 min apart, spectra unrelated.

    The old criterion scored this 0.60 (geometry carried 0.6 of the weight and
    the spectrum could not veto) and merged it. Two co-eluting compounds that
    both base-peak on m/z 73 is the single most common GC-MS situation after
    silylation, so this must not match.
    """
    a = peak("s1", "F0", 10.00, 73.0, spectrum=SPEC_X)
    b = peak("s2", "F0", 10.05, 73.0, spectrum=SPEC_Y)
    assert not is_similar_to(a, b, cfg())


def test_retention_gate_is_hard() -> None:
    """Identical spectra still lose if the retention difference exceeds the
    tolerance — the two gates are AND, not a trade-off."""
    a = peak("s1", "F0", 10.00, 73.0)
    near = peak("s2", "F0", 10.09, 73.0)
    far = peak("s2", "F1", 10.11, 73.0)
    assert is_similar_to(a, near, cfg())
    assert not is_similar_to(a, far, cfg())


def test_missing_spectrum_never_matches() -> None:
    """MS-DIAL ``IsComparedAvailable``: no spectrum, no opinion, no match."""
    a = peak("s1", "F0", 10.0, 73.0, spectrum=SPEC_X)
    blank = peak("s2", "F0", 10.0, 73.0, spectrum=())
    assert not is_similar_to(a, blank, cfg())
    assert not is_similar_to(blank, a, cfg())


def test_score_has_no_mz_term() -> None:
    """Two candidates identical but for their model ion must score the same."""
    master = peak("s1", "F0", 10.0, 73.0)
    same_ion = peak("s2", "F0", 10.02, 73.0, spectrum=SPEC_X_DRIFTED)
    other_ion = peak("s2", "F1", 10.02, 147.0, spectrum=SPEC_X_DRIFTED)
    assert similarity(master, same_ion, cfg()) == similarity(master, other_ion, cfg())


def test_score_is_ei_and_retention_only() -> None:
    master = peak("s1", "F0", 10.0, 73.0)
    target = peak("s2", "F0", 10.0, 73.0)
    # Identical spectra and identical RT: EI = 1, gauss = 1.
    assert similarity(master, target, cfg()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# pass 1 — union master list
# ---------------------------------------------------------------------------

def test_master_list_is_union() -> None:
    """A compound missing from the reference sample still gets a row."""
    peaks = {
        "ref": [peak("ref", "R0", 5.0, 73.0)],
        "s2": [peak("s2", "A0", 5.0, 73.0), peak("s2", "A1", 20.0, 91.0, spectrum=SPEC_Y)],
        "s3": [peak("s3", "B0", 5.0, 73.0), peak("s3", "B1", 20.01, 91.0, spectrum=SPEC_Y)],
    }
    c = cfg()
    masters = build_master_list(peaks, "ref", c)
    assert len(masters) == 2

    spots, unclaimed, stats = assign_to_master(masters, peaks, c)
    spots, stats = reset_quant_masses(spots, c, stats)
    assert len(spots) == 2
    by_index = {round(s.master.index): s for s in spots}
    assert by_index[5].n_detected == 3
    # The compound absent from the reference is aligned across s2 and s3.
    assert by_index[20].n_detected == 2
    assert set(by_index[20].cells) == {"ref", "s2", "s3"}
    assert by_index[20].cells["ref"] is None


def test_annotated_target_displaces_unannotated_master() -> None:
    """``GcmsPeakJoiner.cs:261-268`` — the row is named after its master, so an
    identified peak takes the slot from an unidentified one."""
    peaks = {
        "ref": [peak("ref", "R0", 5.0, 73.0, annotated=False)],
        "s2": [peak("s2", "A0", 5.01, 73.0, annotated=True)],
    }
    stats = JoinStats()
    masters = build_master_list(peaks, "ref", cfg(), stats)
    assert len(masters) == 1
    assert masters[0].sample == "s2"
    assert stats.n_master_swapped == 1


def test_order_independent() -> None:
    """A6 — the result must not depend on how the caller ordered the samples."""
    p = {
        "a": [peak("a", "A0", 5.0, 73.0), peak("a", "A1", 12.0, 91.0, spectrum=SPEC_Y)],
        "b": [peak("b", "B0", 5.02, 73.0, spectrum=SPEC_X_DRIFTED)],
        "c": [peak("c", "C0", 12.03, 91.0, spectrum=SPEC_Y)],
    }
    c = cfg()

    def run(order):
        ordered = {k: p[k] for k in order}
        masters = build_master_list(ordered, "a", c)
        spots, unclaimed, _ = assign_to_master(masters, ordered, c)
        spots, _ = reset_quant_masses(spots, c)
        return sorted(
            (round(s.master.index, 6), round(s.quant_mz, 6), s.n_detected)
            for s in spots
        )

    forward = run(["a", "b", "c"])
    assert forward == run(["c", "b", "a"])
    assert forward == run(["b", "c", "a"])


# ---------------------------------------------------------------------------
# pass 2 — claiming
# ---------------------------------------------------------------------------

def test_one_peak_per_sample_per_spot() -> None:
    """MS-DIAL discards the losing peak instead of giving it a row of its own —
    that is what stops one compound from trailing single-sample rows."""
    peaks = {
        "ref": [peak("ref", "R0", 5.0, 73.0)],
        "s2": [
            peak("s2", "A0", 5.05, 73.0, spectrum=SPEC_X_DRIFTED),   # worse RT
            peak("s2", "A1", 5.00, 73.0, spectrum=SPEC_X),           # perfect
        ],
    }
    c = cfg()
    masters = build_master_list(peaks, "ref", c)
    assert len(masters) == 1                       # both s2 peaks were absorbed

    spots, unclaimed, stats = assign_to_master(masters, peaks, c)
    assert len(spots) == 1
    assert spots[0].cells["s2"].feature_id == "A1"
    assert stats.dropped_lost_claim == 1


def test_better_peak_evicts_the_earlier_claim() -> None:
    """The claim register is "best wins", not "first wins"."""
    master = peak("ref", "R0", 5.0, 73.0)
    peaks = {
        "ref": [master],
        # Sorted by index, so the weaker (5.04) peak is examined first.
        "s2": [peak("s2", "A0", 5.04, 73.0), peak("s2", "A1", 5.001, 73.0)],
    }
    c = cfg()
    spots, unclaimed, stats = assign_to_master([master], peaks, c)
    assert spots[0].cells["s2"].feature_id == "A1"
    assert stats.dropped_lost_claim == 1


def test_no_new_spots_in_pass_two() -> None:
    """A peak the frozen master list cannot host is dropped and counted, never
    turned into a row (spec's "no silent caps")."""
    master = peak("ref", "R0", 5.0, 73.0)
    peaks = {
        "ref": [master],
        "s2": [peak("s2", "A0", 40.0, 73.0, spectrum=SPEC_Y)],
    }
    spots, unclaimed, stats = assign_to_master([master], peaks, cfg())
    assert len(spots) == 1
    assert stats.dropped_no_master == 1


def test_ri_joiner_uses_max_matchs() -> None:
    """Spec X2 — MS-DIAL's RI joiner declares ``maxMatchs`` and never writes to
    it, so a later, worse target overwrites an earlier, better one. The RI path
    here behaves like the RT path."""
    c = cfg(index_type="ri", ri_tolerance=20.0)
    master = peak("ref", "R0", 1200.0, 73.0)
    peaks = {
        "ref": [master],
        # Examined in index order: the good one (1200) first, the poor one
        # (1215) second. Without the register the poor one would win by being
        # last.
        "s2": [peak("s2", "A0", 1200.0, 73.0), peak("s2", "A1", 1215.0, 73.0)],
    }
    spots, unclaimed, stats = assign_to_master([master], peaks, c)
    assert spots[0].cells["s2"].feature_id == "A0"
    assert stats.dropped_lost_claim == 1


# ---------------------------------------------------------------------------
# unclaimed peaks: marked, not deleted
# ---------------------------------------------------------------------------

def _contested():
    return {
        "ref": [peak("ref", "R0", 5.0, 73.0)],
        "s2": [
            peak("s2", "A0", 5.05, 73.0, spectrum=SPEC_X_DRIFTED),
            peak("s2", "A1", 5.00, 73.0, spectrum=SPEC_X),
        ],
    }


def test_default_marks_the_unclaimed_peak_and_never_deletes_it() -> None:
    """Shipped behaviour (user decision, 2026-08-03): the loser is kept as a
    flagged row, which is what ``CLAUDE.md``'s "mark, never delete" asks for.

    MS-DIAL discards it (``GcmsPeakJoiner.cs:293-318``); on real data that
    removed ~5% of features, and the user chose the platform rule over parity
    once the cost was measured.
    """
    peaks = _contested()
    c = cfg()
    assert c.conserve_unclaimed_peaks is True
    masters = build_master_list(peaks, "ref", c)
    _, unclaimed, stats = assign_to_master(masters, peaks, c)
    assert [p.feature_id for p, _ in unclaimed] == ["A0"]      # reported...
    promoted = promote_unclaimed(unclaimed, sorted(peaks), c, stats)
    assert len(promoted) == 1                                  # ...and kept
    assert promoted[0].is_duplicate is True


def test_msdial_parity_is_still_reachable_and_still_counted() -> None:
    """``conserve_unclaimed_peaks=False`` restores MS-DIAL's row count. The
    drop is never silent — the stats block reports it either way."""
    peaks = _contested()
    c = cfg(conserve_unclaimed_peaks=False)
    masters = build_master_list(peaks, "ref", c)
    _, unclaimed, stats = assign_to_master(masters, peaks, c)
    assert stats.dropped_lost_claim == 1                       # counted...
    assert promote_unclaimed(unclaimed, sorted(peaks), c, stats) == []   # ...then dropped


def test_unclaimed_peaks_are_kept_and_marked() -> None:
    """The loser becomes a flagged row rather than disappearing."""
    peaks = _contested()
    c = cfg(conserve_unclaimed_peaks=True)
    masters = build_master_list(peaks, "ref", c)
    spots, unclaimed, stats = assign_to_master(masters, peaks, c)

    extra = promote_unclaimed(unclaimed, sorted(peaks), c, stats)
    assert len(extra) == 1
    assert extra[0].is_duplicate is True
    assert extra[0].duplicate_type == "alignment_redundant"
    assert extra[0].cells["s2"].feature_id == "A0"
    assert extra[0].cells["ref"] is None
    assert stats.n_promoted_redundant == 1

    # It survives reset_quant_masses like any other row — it has a detection.
    rows, _ = reset_quant_masses(list(spots) + extra, c, stats)
    assert len(rows) == 2
    assert sum(1 for r in rows if r.is_duplicate) == 1


def test_peak_with_no_gated_master_is_promoted_visible() -> None:
    """When conserving, a peak nothing represented is not redundant with
    anything and must stay visible."""
    master = peak("ref", "R0", 5.0, 73.0)
    peaks = {"ref": [master], "s2": [peak("s2", "A0", 40.0, 73.0, spectrum=SPEC_Y)]}
    c = cfg(conserve_unclaimed_peaks=True)
    _, unclaimed, stats = assign_to_master([master], peaks, c)
    extra = promote_unclaimed(unclaimed, sorted(peaks), c, stats)
    assert len(extra) == 1
    assert extra[0].is_duplicate is False
    assert stats.n_promoted_visible == 1


def test_every_input_peak_is_accounted_for_when_conserving() -> None:
    """With conservation on, every input either occupies a cell or owns a
    promoted row — no third outcome."""
    peaks = {
        "ref": [peak("ref", "R0", 5.0, 73.0), peak("ref", "R1", 30.0, 91.0, spectrum=SPEC_Y)],
        "s2": [
            peak("s2", "A0", 5.03, 73.0, spectrum=SPEC_X_DRIFTED),
            peak("s2", "A1", 5.00, 73.0, spectrum=SPEC_X),
            peak("s2", "A2", 30.02, 91.0, spectrum=SPEC_Y),
            peak("s2", "A3", 80.0, 55.0, spectrum=SPEC_Y),
        ],
    }
    c = cfg(conserve_unclaimed_peaks=True)
    masters = build_master_list(peaks, "ref", c)
    spots, unclaimed, stats = assign_to_master(masters, peaks, c)
    rows, stats = reset_quant_masses(
        list(spots) + promote_unclaimed(unclaimed, sorted(peaks), c, stats), c, stats,
    )
    placed = {
        (p.sample, p.feature_id)
        for r in rows for p in r.cells.values() if p is not None
    }
    expected = {(p.sample, p.feature_id) for ps in peaks.values() for p in ps}
    assert placed == expected


def test_dropped_peaks_are_always_counted() -> None:
    """"No silent caps": whichever mode is on, the number of peaks the join did
    not place must be recoverable from the stats alone."""
    peaks = _contested()
    for conserve in (False, True):
        c = cfg(conserve_unclaimed_peaks=conserve)
        masters = build_master_list(peaks, "ref", c)
        spots, unclaimed, stats = assign_to_master(masters, peaks, c)
        promote_unclaimed(unclaimed, sorted(peaks), c, stats)
        n_in = sum(len(v) for v in peaks.values())
        n_placed = sum(s.n_detected for s in spots)
        accounted = (
            n_placed
            + stats.dropped_lost_claim
            + stats.dropped_no_master
        )
        assert accounted == n_in, (conserve, stats.as_dict())


# ---------------------------------------------------------------------------
# pass 3 — consensus quantitation ion
# ---------------------------------------------------------------------------

def _spot_with(mzs_and_heights, spectrum=SPEC_X):
    from metabo_core.alignment.gcms_joiner import GcmsSpot
    cells = {}
    for i, (mz, h) in enumerate(mzs_and_heights):
        s = f"s{i}"
        cells[s] = peak(s, f"F{i}", 10.0, mz, height=h, spectrum=spectrum)
    master = next(iter(cells.values()))
    return GcmsSpot(master=master, cells=cells)


def test_reset_quant_masses_votes() -> None:
    """Four detected cells on 43 / 43 / 71 / 43 settle on 43 — the mode, not
    whatever the tallest cell happened to use.

    The 71 cell is 7x taller than any 43 cell and still loses: the vote counts
    cells, and every cell here clears the 10% height floor. (Drop the 43 cells
    below that floor and 71 wins on being the only voter — which is the
    behaviour :func:`test_reset_quant_masses_ignores_near_noise_cells` pins.)
    """
    spot = _spot_with([(43.0, 1000.0), (43.0, 1200.0), (71.0, 9000.0), (43.0, 1100.0)])
    out, _ = reset_quant_masses([spot], cfg())
    assert out[0].quant_mz == pytest.approx(43.0)


def test_reset_quant_masses_ignores_near_noise_cells() -> None:
    """Cells below 10% of the row's tallest do not get a vote
    (``GcmsPeakJoiner.cs:101``)."""
    spot = _spot_with([(43.0, 10_000.0), (71.0, 100.0), (71.0, 90.0), (71.0, 80.0)])
    out, _ = reset_quant_masses([spot], cfg())
    assert out[0].quant_mz == pytest.approx(43.0)


def test_reset_quant_masses_rejects_absent_mass() -> None:
    """A voted ion the representative spectrum does not actually carry falls
    back to the representative's own ion (``:106-114``)."""
    # 300 wins the vote but appears nowhere in SPEC_X.
    spot = _spot_with([(300.0, 500.0), (300.0, 400.0), (43.0, 1000.0)])
    out, _ = reset_quant_masses([spot], cfg())
    # Representative is the tallest cell (43), and 43 is SPEC_X's base peak.
    assert out[0].quant_mz == pytest.approx(43.0)


def test_reset_quant_masses_falls_back_to_base_peak() -> None:
    """Neither the vote nor the representative's own ion is in the spectrum:
    take the representative's base peak."""
    spot = _spot_with([(300.0, 500.0), (300.0, 400.0)])
    out, _ = reset_quant_masses([spot], cfg())
    assert out[0].quant_mz == pytest.approx(43.0)      # SPEC_X base peak


def test_rows_without_any_detection_are_dropped() -> None:
    from metabo_core.alignment.gcms_joiner import GcmsSpot
    empty = GcmsSpot(master=peak("ref", "R0", 5.0, 73.0), cells={"ref": None})
    out, stats = reset_quant_masses([empty], cfg())
    assert out == []
    assert stats.dropped_no_detected == 1


def test_quant_mass_exists_needs_ten_percent_of_base_peak() -> None:
    spectrum = ((43.0, 1000.0), (58.0, 400.0), (300.0, 50.0))
    assert quant_mass_exists(43.0, spectrum, 0.5)
    assert quant_mass_exists(58.0, spectrum, 0.5)
    assert not quant_mass_exists(300.0, spectrum, 0.5)     # only 5% of base peak
    assert not quant_mass_exists(999.0, spectrum, 0.5)


def test_quant_mass_from_base_peak_option() -> None:
    spot = _spot_with([(71.0, 100.0), (71.0, 120.0)])
    out, _ = reset_quant_masses([spot], cfg(quant_mass_from_base_peak=True))
    assert out[0].quant_mz == pytest.approx(43.0)         # SPEC_X base peak


def test_every_detected_cell_agrees_with_the_row_after_requantitation_is_needed() -> None:
    """A5's precondition: the row commits to one ion, and the cells that used a
    different one are exactly the ones gap fill has to re-integrate."""
    spot = _spot_with([(43.0, 1000.0), (43.0, 900.0), (71.0, 800.0)])
    out, _ = reset_quant_masses([spot], cfg())
    row = out[0]
    assert row.quant_mz == pytest.approx(43.0)
    mismatched = [p for p in row.detected if abs(p.mz - row.quant_mz) > 0.5]
    assert len(mismatched) == 1 and mismatched[0].mz == 71.0
