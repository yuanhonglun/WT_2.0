"""``alignment.eic``: the sample-major -> spot-major transpose, and random reads.

Plus ``spotmap.json``, the key map spilled beside it, at the bottom.
"""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from asfam.io.eic_store import (
    EIC_MAGIC,
    SPOTMAP_NAME,
    EicSpillWriter,
    EicStore,
    EicStoreError,
    SpotChromatograms,
    SpotMap,
    Trace,
    load_spot_map,
    open_store,
    resolve_store,
    save_spot_map,
)


def _trace(label: str, n: int = 5, scale: float = 1.0, status: str = "detected") -> Trace:
    rt = np.linspace(4.0, 5.0, n)
    return Trace(label=label, rt=rt, intensity=scale * np.arange(n, dtype=float),
                 status=status, rt_left=4.2, rt_right=4.8, rt_apex=4.5)


def _sample(keys, sample_id, scale, fragments_on=None):
    """One sample's contribution: a quant trace per spot, fragments on one spot."""
    for i, _ in enumerate(keys):
        body = SpotChromatograms(quant=[_trace(sample_id, scale=scale + i)])
        if fragments_on == i:
            body.fragments = [(70.5, _trace("70.500")), (90.25, _trace("90.250"))]
        yield body


def test_transpose_gathers_every_sample_under_each_spot(tmp_path):
    keys = ["F00000", "F00001", "F00002"]
    with EicSpillWriter(keys, temp_dir=tmp_path) as writer:
        writer.add_sample(_sample(keys, "s2", 10.0))
        writer.add_sample(_sample(keys, "s1", 100.0, fragments_on=1))
        size = writer.transpose(tmp_path / "alignment.eic")

    assert size > 0
    store = EicStore(tmp_path / "alignment.eic")
    assert len(store) == 3

    spot = store.get("F00001")
    # Samples come back sorted by id, not in the order they were spilled.
    assert [t.label for t in spot.quant] == ["s1", "s2"]
    assert spot.quant[0].intensity[-1] == pytest.approx(101.0 * 4)
    assert spot.quant[1].intensity[-1] == pytest.approx(11.0 * 4)
    assert [mz for mz, _ in spot.fragments] == [70.5, 90.25]

    # Fragments live on exactly the spot whose representative supplied them.
    assert store.get("F00000").fragments == []
    assert len(store.get("F00002").quant) == 2


def test_reads_are_keyed_by_feature_id_not_position(tmp_path):
    # PR-6 renumbers features by m/z; a positional index would then hand back
    # some other feature's chromatogram without erroring.
    keys = ["F00007", "F00003"]
    with EicSpillWriter(keys, temp_dir=tmp_path) as writer:
        writer.add_sample(_sample(keys, "s1", 1.0))
        writer.transpose(tmp_path / "alignment.eic")

    store = EicStore(tmp_path / "alignment.eic")
    assert "F00007" in store and "F00003" in store
    assert store.get("F00000") is None
    assert store.get("F00007").quant[0].intensity[-1] == pytest.approx(4.0)
    assert store.get("F00003").quant[0].intensity[-1] == pytest.approx(8.0)


def test_status_and_peak_window_survive_the_round_trip(tmp_path):
    keys = ["F00000"]
    with EicSpillWriter(keys, temp_dir=tmp_path) as writer:
        writer.add_sample(iter([SpotChromatograms(
            quant=[_trace("s1", status="no_signal")],
        )]))
        writer.transpose(tmp_path / "alignment.eic")

    trace = EicStore(tmp_path / "alignment.eic").get("F00000").quant[0]
    assert trace.status == "no_signal"
    assert trace.rt_left == pytest.approx(4.2)
    assert trace.rt_right == pytest.approx(4.8)
    assert trace.rt_apex == pytest.approx(4.5)


def test_a_sample_must_cover_every_spot(tmp_path):
    with EicSpillWriter(["F00000", "F00001"], temp_dir=tmp_path) as writer:
        with pytest.raises(ValueError, match="expected 2"):
            writer.add_sample(iter([SpotChromatograms()]))


def test_temporaries_are_removed_even_when_the_fill_dies(tmp_path):
    writer = EicSpillWriter(["F00000"], temp_dir=tmp_path)
    with pytest.raises(RuntimeError):
        with writer:
            writer.add_sample(iter([SpotChromatograms(quant=[_trace("s1")])]))
            raise RuntimeError("gap fill cancelled")

    assert list(tmp_path.glob("asfam_eic_*")) == []
    assert not (tmp_path / "alignment.eic").exists()


def test_an_empty_spot_round_trips(tmp_path):
    with EicSpillWriter(["F00000"], temp_dir=tmp_path) as writer:
        writer.add_sample(iter([SpotChromatograms()]))
        writer.transpose(tmp_path / "alignment.eic")

    spot = EicStore(tmp_path / "alignment.eic").get("F00000")
    assert spot.quant == [] and spot.fragments == []


def test_a_future_version_is_refused_not_guessed_at(tmp_path):
    path = tmp_path / "alignment.eic"
    path.write_bytes(EIC_MAGIC + struct.pack("<ii", 99, 0))
    with pytest.raises(EicStoreError, match="format version 99"):
        EicStore(path)


def test_open_store_is_quiet_about_a_missing_or_broken_file(tmp_path):
    assert open_store(tmp_path) is None
    (tmp_path / "alignment.eic").write_bytes(b"not an eic file")
    assert open_store(tmp_path) is None


# ---------------------------------------------------------------------------
# Finding the store a project belongs to
# ---------------------------------------------------------------------------

def _write_store(directory, n_spots: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    keys = [f"F{i:05d}" for i in range(n_spots)]
    with EicSpillWriter(keys, temp_dir=directory) as writer:
        writer.add_sample(_sample(keys, "s1", 1.0))
        writer.transpose(directory / "alignment.eic")


def test_the_recorded_directory_is_used_when_it_is_still_there(tmp_path):
    out = tmp_path / "result"
    _write_store(out, 2)
    store, found = resolve_store(recorded_dirs=[out])
    assert found == out and len(store) == 2


def test_a_store_beside_the_project_is_found(tmp_path):
    """What a user who copied one file by hand would expect, and what a saved
    project's own directory looks like. The reader used to derive exactly one
    path — ``work_dir``'s parent — so a store sitting right next to the project
    was never even looked at."""
    proj = tmp_path / "ASFAMProject_20260727_090000.asfam"
    proj.write_bytes(b"")
    _write_store(tmp_path, 2)

    store, found = resolve_store(project_path=proj,
                                 recorded_dirs=[r"D:\gone\result"])
    assert found == tmp_path and len(store) == 2


def test_a_result_folder_that_moved_with_the_project_is_followed(tmp_path):
    """The reported failure: the run was archived to another disk, so the
    recorded ``D:\\...`` directory is gone and every reopened project came back
    with no chromatograms at all."""
    out = tmp_path / "result_0722"
    _write_store(out, 3)
    proj = tmp_path / "ASFAMProject_20260722_192205.asfam"
    proj.write_bytes(b"")

    store, found = resolve_store(
        project_path=proj,
        recorded_dirs=[r"D:\YHL\rice\result_0722"],
    )
    assert found == out and len(store) == 3


def test_the_feature_count_settles_which_run_the_store_belongs_to(tmp_path):
    """Two runs over the same data mint the same ``F00042`` ids, so a stale
    store answers every lookup — with another run's chromatograms. Only the spot
    count tells them apart, and it outranks the search order: here the stale one
    is sitting right beside the project."""
    _write_store(tmp_path, 5)                  # beside the project: another run
    out = tmp_path / "result_0722"
    _write_store(out, 3)                       # this project's own
    proj = tmp_path / "ASFAMProject_20260722_192205.asfam"
    proj.write_bytes(b"")

    store, found = resolve_store(
        project_path=proj,
        recorded_dirs=[str(out)],
        expected_spots=3,
    )
    assert found == out and len(store) == 3


def test_a_count_mismatch_still_beats_showing_nothing(tmp_path):
    """No candidate matches: plot something and say so in the log, rather than
    leave the panel empty. Old projects predate the recorded directory and can
    legitimately land here."""
    _write_store(tmp_path, 5)
    proj = tmp_path / "p.asfam"
    proj.write_bytes(b"")

    store, found = resolve_store(project_path=proj, expected_spots=3)
    assert found == tmp_path and len(store) == 5


def test_no_store_anywhere_is_reported_as_absent(tmp_path):
    proj = tmp_path / "p.asfam"
    proj.write_bytes(b"")
    assert resolve_store(project_path=proj,
                         recorded_dirs=[r"D:\gone", None]) == (None, None)


# ---------------------------------------------------------------------------
# spotmap.json
# ---------------------------------------------------------------------------

def test_the_spot_map_round_trips(tmp_path):
    save_spot_map(tmp_path / SPOTMAP_NAME, SpotMap(
        spot_of={"rep1_00042": "F00007", "rep2_00013": "F00007"},
        representative_of={"F00007": "2"},
        fold_reason_of={"rep2_00013": "same_compound_claim_loser"},
        fold_evidence_of={
            "rep2_00013": {
                "target_peak_id": "rep2_00012",
                "cosine": 0.95,
                "n_matched_fragments": 4,
            },
        },
    ))

    loaded = load_spot_map(tmp_path)
    assert loaded.spot_of == {"rep1_00042": "F00007", "rep2_00013": "F00007"}
    assert loaded.representative_of == {"F00007": "2"}
    assert loaded.fold_reason_of == {
        "rep2_00013": "same_compound_claim_loser",
    }
    assert loaded.fold_evidence_of["rep2_00013"]["n_matched_fragments"] == 4


def test_load_spot_map_is_quiet_about_a_missing_or_broken_file(tmp_path):
    # A project written before spotmap.json existed simply has none. The caller
    # falls back to saying it has no chromatogram; it must not die here.
    assert load_spot_map(tmp_path).spot_of == {}

    (tmp_path / SPOTMAP_NAME).write_text("{ not json")
    assert load_spot_map(tmp_path).spot_of == {}


def test_a_future_spot_map_version_is_ignored_not_guessed_at(tmp_path):
    (tmp_path / SPOTMAP_NAME).write_text(json.dumps(
        {"version": 99, "spot_of": {"rep1_00000": "wherever"}}))

    assert load_spot_map(tmp_path).spot_of == {}
