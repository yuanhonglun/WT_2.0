"""Opening a project after the run was archived to another disk.

The reported failure, end to end: analysis ran on ``D:\\runs\\study_a``, the
whole tree was copied to an external drive, and from then on *every* reopened project
showed no chromatograms. Nothing was corrupt — the reader derived exactly one
path, ``work_dir``'s parent, and ``work_dir`` named a directory on a disk that
was no longer there. Copying ``alignment.eic`` next to the project by hand did
not help either, because that location was never looked at.

Both halves are pinned here, plus the trap the manual copy sets: the copied
store was from a *different* run of the same data, and spot ids are minted per
run, so it answers every lookup with the wrong chromatogram.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PyQt5")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from asfam.config import ProcessingConfig
from asfam.io import spill
from asfam.io.eic_store import (
    SPOTMAP_NAME, STORE_NAME, EicSpillWriter, SpotChromatograms, SpotMap, Trace,
    save_spot_map,
)
from asfam.io.project_file import save_project
from asfam.models import CandidateFeature, Feature

#: Where the run happened. Gone by the time the project is opened.
DEAD_ROOT = r"D:\gone\study_a"


@pytest.fixture
def main_window(monkeypatch):
    from PyQt5.QtWidgets import QApplication

    # Bound, not discarded: an unreferenced QApplication is collected the moment
    # it is built, and the next QWidget takes the interpreter down with it.
    app = QApplication.instance() or QApplication([])
    assert app is not None
    from asfam.gui import main_window as mw_mod

    monkeypatch.setattr(mw_mod._UpdateChecker, "start", lambda self: None)
    win = mw_mod.MainWindow()
    yield win
    win.close()


def _feature(i: int) -> Feature:
    return Feature(
        feature_id=f"F{i:05d}", precursor_mz=195.0 + i, rt=5.0,
        rt_left=4.9, rt_right=5.1, signal_type="ms1_detected",
        ms2_mz=np.array([110.07], dtype=np.float64),
        ms2_intensity=np.array([1.0], dtype=np.float64), n_fragments=1,
    )


def _trace(label: str) -> Trace:
    return Trace(label=label, rt=np.linspace(4.0, 5.0, 5),
                 intensity=np.arange(5, dtype=float) * 100.0,
                 status="detected", rt_left=4.2, rt_right=4.8, rt_apex=4.5)


def _write_store(directory, n_spots: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    keys = [f"F{i:05d}" for i in range(n_spots)]
    with EicSpillWriter(keys, temp_dir=directory) as writer:
        writer.add_sample(iter([
            SpotChromatograms(quant=[_trace("1")]) for _ in keys
        ]))
        writer.transpose(directory / STORE_NAME)


def _archived_run(tmp_path, *, n_features: int = 3):
    """An archive as the user's external drive holds it.

    The project sits beside the raw data (where it was auto-saved before this
    fix), the result folder came along under a new root, and the store the user
    copied up next to the project belongs to an earlier run.
    """
    archive = tmp_path / "archive"
    out = archive / "result_0722"
    work = out / "_work"

    _write_store(out, n_features)
    save_spot_map(out / SPOTMAP_NAME,
                  SpotMap(spot_of={"rep1_00000": "F00000"},
                          representative_of={"F00000": "1"}))
    spill.write_sample(
        work / "1",
        [CandidateFeature(
            feature_id="rep1_00000", segment_name="190-200", replicate_id=1,
            precursor_mz_nominal=195, rt_apex=5.0, rt_left=4.9, rt_right=5.1,
            ms2_mz=np.array([110.07], dtype=np.float64),
            ms2_intensity=np.array([1.0], dtype=np.float64), n_fragments=1,
            ms1_precursor_mz=195.09, ms1_height=1e4, ms1_area=2e4,
        )],
        sample_id="1",
    )

    raw = archive / "MIX_190-200_1.mzML"
    raw.write_bytes(b"")

    project = archive / "ASFAMProject_20260722_192205.asfam"
    save_project(
        str(project), ProcessingConfig(), [_feature(i) for i in range(n_features)],
        mzml_paths=[os.path.join(DEAD_ROOT, "MIX_190-200_1.mzML")],
        work_dir=os.path.join(DEAD_ROOT, "result_0722", "_work"),
    )
    return archive, out, project


def _open(mw, monkeypatch, project):
    """Answer the Open dialog with ``project``.

    The stub replaces the name ``main_window`` looks the dialog up under, not an
    attribute on the sip type itself — patching a PyQt class and then undoing it
    takes the interpreter down with it.
    """
    from asfam.gui import main_window as mw_mod

    class _Dialog:
        @staticmethod
        def getOpenFileName(*args, **kwargs):
            return str(project), ""

    monkeypatch.setattr(mw_mod, "QFileDialog", _Dialog)
    mw._on_open_project()


def test_the_chromatograms_are_found_after_the_run_was_archived(main_window,
                                                                monkeypatch,
                                                                tmp_path):
    mw = main_window
    _, out, project = _archived_run(tmp_path)

    _open(mw, monkeypatch, project)

    assert mw._eic_dir == str(out)
    assert mw.eic_plot._store is not None
    assert len(mw.eic_plot._store) == 3
    # The key map has to come from the same directory as the store, or the
    # single-sample view translates candidate ids through another run's table.
    assert mw._spotmap.spot_of == {"rep1_00000": "F00000"}


def test_a_store_copied_next_to_the_project_does_not_win_on_position(main_window,
                                                                     monkeypatch,
                                                                     tmp_path):
    """The manual workaround, kept honest. Spot ids are minted per run, so this
    stale store answers every lookup — the feature count is the only thing that
    tells the two apart."""
    mw = main_window
    archive, out, project = _archived_run(tmp_path)
    _write_store(archive, 5)          # an earlier run, copied up by hand

    _open(mw, monkeypatch, project)

    assert mw._eic_dir == str(out), "took the stale store sitting beside the project"
    assert len(mw.eic_plot._store) == 3


def test_the_intermediates_and_raw_files_are_relocated_too(main_window,
                                                           monkeypatch,
                                                           tmp_path):
    """Same dead root, so re-annotation and the single-sample view died with the
    EIC panel. An empty ``_work`` beside the project must not be mistaken for
    the real one."""
    mw = main_window
    archive, out, project = _archived_run(tmp_path)
    (archive / "_work").mkdir()       # look-alike: no checkpoints in it

    _open(mw, monkeypatch, project)

    assert mw._work_dir == str(out / "_work")
    assert mw.act_reannotate.isEnabled()
    assert list(mw._sample_files) == ["1"]
    assert mw._sample_files["1"] == [str(archive / "MIX_190-200_1.mzML")]


def test_the_auto_save_lands_with_the_files_it_reads(main_window, tmp_path):
    """It used to land next to the raw data — the far side of the run from
    ``alignment.eic``, ``spotmap.json`` and ``_work/``. Archiving the results
    then left the project pointing across a disk boundary at all three."""
    mw = main_window
    out = tmp_path / "result_0722"
    _write_store(out, 2)
    (out / "_work").mkdir()
    mw.features = [_feature(i) for i in range(2)]
    mw._mzml_paths = [str(tmp_path / "raw" / "MIX_190-200_1.mzML")]
    mw._work_dir = str(out / "_work")
    mw._load_eic_store()

    mw._auto_save_project()

    saved = list(out.glob("ASFAMProject_*.asfam"))
    assert len(saved) == 1
    assert mw._project_path == str(saved[0])


def test_a_run_then_a_reopen_finds_its_own_store(main_window, monkeypatch,
                                                 tmp_path):
    """The round trip the fix is for: what the run writes is what the reopened
    project reads, with no file copied by hand in between."""
    mw = main_window
    out = tmp_path / "result_0722"
    _write_store(out, 2)
    (out / "_work").mkdir()
    mw.features = [_feature(i) for i in range(2)]
    mw._mzml_paths = [str(tmp_path / "MIX_190-200_1.mzML")]
    mw._work_dir = str(out / "_work")
    mw._load_eic_store()
    mw._auto_save_project()
    saved = next(out.glob("ASFAMProject_*.asfam"))

    fresh = mw.__class__()
    try:
        _open(fresh, monkeypatch, saved)
        assert fresh._eic_dir == str(out)
        assert len(fresh.eic_plot._store) == 2
    finally:
        fresh.close()


def test_a_project_still_on_its_own_machine_is_unaffected(main_window,
                                                          monkeypatch,
                                                          tmp_path):
    """The recorded paths are tried first and answer immediately; relocation
    only ever runs when they do not."""
    mw = main_window
    out = tmp_path / "result_0722"
    _write_store(out, 2)
    project = tmp_path / "p.asfam"
    save_project(str(project), ProcessingConfig(), [_feature(i) for i in range(2)],
                 mzml_paths=[], work_dir=str(out / "_work"), eic_dir=str(out))

    _open(mw, monkeypatch, project)

    assert mw._eic_dir == str(out)
    assert len(mw.eic_plot._store) == 2
