"""``spotmap.json``: the candidate id -> spot id map the single-sample view reads.

Without it the single-sample EIC panel looks ``rep1_00042`` up in a store keyed by
``F00042`` and misses — which is what it did, for every feature it was ever shown,
while the whole suite stayed green.

Two things pinned here. The first is the obvious one: every candidate a spot
claimed is in the map, pointing at that spot. The second is the subtle one, and
it is an *ordering*: the map is built before gap filling. Gap filling mints a
peak into every spot for every sample that had none, under a derived id
(``rep1_00042@gap:2``) that no candidate in ``_work/`` carries. Built afterwards,
the map fills with keys nothing can ever look up.
"""
from __future__ import annotations

import numpy as np

from asfam.config import ProcessingConfig
from asfam.io.eic_store import SPOTMAP_NAME, load_spot_map
from asfam.models import CandidateFeature
from asfam.pipeline import stage7_alignment
from asfam.pipeline.stage7_alignment import GapFillContext, run_stage7
from metabo_core.alignment.gap_filler import (
    FILLED, GapFillResult, gap_fill_target, make_filled_peak,
)


def _peak(sample: str, index: int, mz: float, rt: float, fragment_mz: float,
          height: float = 1000.0) -> CandidateFeature:
    """A candidate carrying the id stage 3 mints for it: ``rep<sample>_<index>``."""
    return CandidateFeature(
        feature_id=f"rep{sample}_{index:05d}",
        segment_name="280-292",
        replicate_id=int(sample),
        precursor_mz_nominal=int(round(mz)),
        rt_apex=rt, rt_left=rt - 0.1, rt_right=rt + 0.1,
        # Distinct fragments per compound, identical across samples: the joiner's
        # MS2 cosine term then makes each peak score highest against its own
        # master, so the claim below is not a coin toss.
        ms2_mz=np.array([fragment_mz, fragment_mz + 30.0], dtype=np.float64),
        ms2_intensity=np.array([1000.0, 500.0], dtype=np.float64),
        n_fragments=2,
        ms1_precursor_mz=mz, ms1_quant_mz=mz, ms1_height=height,
    )


#: Sample "1" has both compounds; sample "2" only the one at m/z 300, and its peak
#: is the taller of that pair, so it represents the spot they share. The spot at
#: m/z 500 therefore has a hole, which gap filling would plug.
def _two_samples() -> dict[str, list[CandidateFeature]]:
    return {
        "1": [_peak("1", 0, 300.0, 5.00, 80.0),
              _peak("1", 1, 500.0, 7.00, 140.0)],
        "2": [_peak("2", 0, 300.0, 5.01, 80.0, height=5000.0)],
    }


def _context(tmp_path) -> GapFillContext:
    # The fills below are faked, so these paths are never opened.
    return GapFillContext(
        sample_files={"1": ["s1.mzML"], "2": ["s2.mzML"]},
        output_dir=str(tmp_path),
    )


def _fill(spots, sample_ids, config) -> None:
    """What the real gap fill does to ``spot.peaks``, minus the raw data."""
    gap_config = config.gap_fill_view()
    for spot in spots:
        target = gap_fill_target(spot.detected_peaks, gap_config)
        if target is None:
            continue
        for sample_id in sample_ids:
            if sample_id not in spot.peaks:
                spot.peaks[sample_id] = make_filled_peak(
                    spot.representative, sample_id, target,
                    GapFillResult(status=FILLED, height=10.0, area=20.0),
                )


def test_every_claimed_candidate_maps_to_the_spot_that_claimed_it(tmp_path, monkeypatch):
    monkeypatch.setattr(stage7_alignment, "run_gap_fill",
                        lambda spots, sids, cfg, ctx, cb=None: None)

    out = run_stage7(_two_samples(), ProcessingConfig(),
                     gap_fill=_context(tmp_path))

    spot_of = load_spot_map(tmp_path).spot_of
    assert set(spot_of) == {"rep1_00000", "rep1_00001", "rep2_00000"}
    # Both samples' peaks on the shared compound reach the one spot: that is the
    # whole point — a candidate id has to land on the aligned spot's traces.
    assert spot_of["rep1_00000"] == spot_of["rep2_00000"]
    assert spot_of["rep1_00001"] != spot_of["rep1_00000"]
    # And every id the map hands out is one the export actually minted.
    assert set(spot_of.values()) <= {f.feature_id for f in out}


def test_the_map_records_the_sample_the_stored_fragments_came_from(tmp_path, monkeypatch):
    """Gap fill writes fragment chromatograms for the representative only, so the
    single-sample panel has to be able to say whose they are."""
    monkeypatch.setattr(stage7_alignment, "run_gap_fill",
                        lambda spots, sids, cfg, ctx, cb=None: None)

    run_stage7(_two_samples(), ProcessingConfig(), gap_fill=_context(tmp_path))

    spot_map = load_spot_map(tmp_path)
    shared = spot_map.spot_of["rep1_00000"]
    lone = spot_map.spot_of["rep1_00001"]
    assert set(spot_map.representative_of) == {shared, lone}
    # Neither peak is annotated, so height decides the representative, and sample
    # "2"'s is 5x sample "1"'s.
    assert spot_map.representative_of[shared] == "2"
    assert spot_map.representative_of[lone] == "1"


def test_the_map_is_settled_before_gap_filling_can_pollute_it(tmp_path, monkeypatch):
    """The ordering. Move the build below ``run_gap_fill`` and this fails twice:
    the map is absent when the fill starts, and it ends up carrying ``@gap:`` ids
    that match no candidate in ``_work/``."""
    seen: dict = {}

    def fake_gap_fill(spots, sample_ids, config, context, progress_callback=None):
        seen["when_fill_started"] = load_spot_map(context.output_dir).spot_of
        _fill(spots, sample_ids, config)
        seen["in_memory"] = {p.feature_id for s in spots for p in s.peaks.values()}

    monkeypatch.setattr(stage7_alignment, "run_gap_fill", fake_gap_fill)

    run_stage7(_two_samples(), ProcessingConfig(), gap_fill=_context(tmp_path))

    on_disk = load_spot_map(tmp_path).spot_of
    assert seen["when_fill_started"] == on_disk, \
        "the map must be final before the fill runs, not written from its leftovers"
    assert set(on_disk) == {"rep1_00000", "rep1_00001", "rep2_00000"}
    # Not a vacuous pass: the fill really did leave ids behind that a map built
    # afterwards would have swallowed.
    assert seen["in_memory"] - set(on_disk), "the fake fill polluted nothing"


def test_no_map_without_a_store_to_key_into(tmp_path, monkeypatch):
    """The map and ``alignment.eic`` are a pair. Disabling the fill writes neither,
    so a stale store is never paired with a map minted by a different join."""
    monkeypatch.setattr(stage7_alignment, "run_gap_fill",
                        lambda spots, sids, cfg, ctx, cb=None: None)
    config = ProcessingConfig()
    config.gap_fill_enabled = False

    run_stage7(_two_samples(), config, gap_fill=_context(tmp_path))

    assert not (tmp_path / SPOTMAP_NAME).exists()
