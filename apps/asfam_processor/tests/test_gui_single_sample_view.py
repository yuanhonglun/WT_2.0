"""The single-sample view reads candidates from ``_work/`` and hides nothing.

Three regressions this pins:

1. The view used to skip ``c.status != "active"``. That was a no-op only because
   ``_restore_duplicate_status()`` had already flipped every dedup-excluded
   feature back to "active" just before annotation. With the restore sweep gone,
   the same filter would silently drop every isotope / adduct / ISF copy from
   the per-sample table and scatter.
2. The pipeline no longer hands the GUI its candidates in memory; they are read
   back from the spill on demand.
3. The view lists *candidates* (``rep1_00000``) while ``alignment.eic`` is keyed
   by *aligned spot* (``F00000``). When the EIC viewer stopped reading raw scans
   and started reading the store, it looked features up by ``feature_id`` — and
   the two namespaces never collide, so every single feature in this view came
   back "No chromatogram stored". Nothing in this file touched ``eic_plot``, so
   the suite stayed green. Hence the end-to-end tests at the bottom: they select
   a feature and look at what the viewer actually drew.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PyQt5")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from asfam.io import spill
from asfam.io.eic_store import (
    SPOTMAP_NAME, STORE_NAME, EicSpillWriter, SpotChromatograms, SpotMap, Trace,
    save_spot_map,
)
from asfam.models import CandidateFeature


@pytest.fixture
def qapp_factory():
    from PyQt5.QtWidgets import QApplication

    def _factory():
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    return _factory


def _candidate(fid: str, mz: float, **overrides) -> CandidateFeature:
    c = CandidateFeature(
        feature_id=fid, segment_name="190-200", replicate_id=1,
        precursor_mz_nominal=int(mz), rt_apex=5.0, rt_left=4.9, rt_right=5.1,
        ms2_mz=np.array([110.07, 138.07], dtype=np.float64),
        ms2_intensity=np.array([0.3, 1.0], dtype=np.float64),
        n_fragments=2, ms1_precursor_mz=mz, ms1_height=1e4, ms1_area=2e4,
    )
    for key, value in overrides.items():
        setattr(c, key, value)
    return c


@pytest.fixture
def main_window(qapp_factory, monkeypatch):
    _ = qapp_factory()
    from asfam.gui import main_window as mw_mod

    monkeypatch.setattr(mw_mod._UpdateChecker, "start", lambda self: None)
    win = mw_mod.MainWindow()
    yield win
    win.close()


def test_single_sample_view_shows_dedup_excluded_features(main_window, tmp_path):
    mw = main_window
    cands = [
        _candidate("F000", 195.09),
        _candidate("F001", 196.09, status="isotope_excluded",
                   is_duplicate=True, duplicate_type="isotope"),
        _candidate("F002", 217.07, status="adduct_excluded",
                   is_duplicate=True, duplicate_type="adduct"),
        _candidate("F003", 110.07, status="isf_excluded",
                   is_duplicate=True, duplicate_type="isf"),
    ]
    work = tmp_path / "_work"
    spill.write_sample(work / "1", cands, sample_id="1")

    mw._work_dir = str(work)
    mw._sample_files = {"1": ["MIX_190-200_1.mzML"]}
    mw._rep_id_map = {1: "1"}

    mw._on_view_changed(1)

    assert mw._view_mode == "1"
    shown = mw.feature_table.model._features
    assert [f.feature_id for f in shown] == ["F000", "F001", "F002", "F003"]
    assert sum(f.is_duplicate for f in shown) == 3


def test_single_sample_view_survives_a_missing_work_dir(main_window, tmp_path):
    """Clearing the intermediates must not raise when a sample view is opened."""
    mw = main_window
    mw._work_dir = str(tmp_path / "_work")
    mw._rep_id_map = {1: "1"}
    mw._on_view_changed(1)   # no spill on disk
    assert mw._view_mode == "1"


# ---------------------------------------------------------------------------
# End to end: the store, the key map, and what the viewer draws
# ---------------------------------------------------------------------------

def _trace(label: str, rt_lo: float) -> Trace:
    """Five points from ``rt_lo``. Samples get disjoint RT ranges here so a plotted
    line can be traced back to the sample it belongs to."""
    return Trace(label=label, rt=np.linspace(rt_lo, rt_lo + 1.0, 5),
                 intensity=np.arange(5, dtype=float) * 100.0,
                 status="detected", rt_left=rt_lo + 0.2, rt_right=rt_lo + 0.8,
                 rt_apex=rt_lo + 0.5)


def _spill_store(output_dir, *, with_map: bool) -> None:
    """What stage 7 leaves beside ``_work/``: two spots, two samples.

    Sample "2" represents both spots, so it is the only one whose fragments were
    extracted — the sample we go on to view ("1") has none of its own.
    """
    keys = ["F00000", "F00001"]
    with EicSpillWriter(keys, temp_dir=output_dir) as writer:
        writer.add_sample(iter([
            SpotChromatograms(quant=[_trace("1", 4.0)]),
            SpotChromatograms(quant=[_trace("1", 4.0)]),
        ]))
        writer.add_sample(iter([
            SpotChromatograms(quant=[_trace("2", 8.0)],
                              fragments=[(110.07, _trace("110.070", 8.0))]),
            SpotChromatograms(quant=[_trace("2", 8.0)]),
        ]))
        writer.transpose(output_dir / STORE_NAME)

    if with_map:
        save_spot_map(output_dir / SPOTMAP_NAME, SpotMap(
            # rep1_00002 is deliberately absent: no spot claimed it.
            spot_of={"rep1_00000": "F00000", "rep1_00001": "F00001"},
            representative_of={"F00000": "2", "F00001": "2"},
        ))


def _open_sample_view(mw, tmp_path, *, with_map: bool = True):
    cands = [_candidate("rep1_00000", 195.09),
             _candidate("rep1_00001", 217.07),
             _candidate("rep1_00002", 301.14)]
    work = tmp_path / "_work"
    spill.write_sample(work / "1", cands, sample_id="1")
    _spill_store(tmp_path, with_map=with_map)

    mw._work_dir = str(work)
    mw._sample_files = {"1": ["MIX_190-200_1.mzML"]}
    mw._rep_id_map = {1: "1"}
    mw._load_eic_store()
    mw._on_view_changed(1)


def _drawn(ax) -> list:
    """The chromatogram lines on ``ax`` — not the apex marker ``axvline`` adds."""
    return [line for line in ax.get_lines()
            if line.get_label().startswith("MS1")]


def test_single_sample_view_plots_this_sample_s_own_chromatogram(main_window, tmp_path):
    mw = main_window
    _open_sample_view(mw, tmp_path)

    mw._on_feature_selected(0)

    # The candidate id was translated; the raw one would have missed.
    assert mw.eic_plot._store_key == "F00000"
    assert [t.label for t in mw.eic_plot._store.get("F00000").quant] == ["1", "2"]

    # Drawn, not messaged: _message() leaves the widget with no ``_axes`` at all.
    assert len(mw.eic_plot._axes) == 2
    lines = _drawn(mw.eic_plot._axes[0])
    assert len(lines) == 1, "single-sample mode plots one sample, not every sample"
    assert max(lines[0].get_xdata()) < 6.0, "and it must be sample 1's own trace"


def test_the_fragment_panel_names_the_sample_its_traces_came_from(main_window, tmp_path):
    """Gap fill extracts fragments from the representative only. Read as this
    sample's, they would be MS2 evidence it does not have."""
    mw = main_window
    _open_sample_view(mw, tmp_path)

    mw._on_feature_selected(0)

    assert "from sample 2" in mw.eic_plot._axes[1].get_title()


def test_a_candidate_no_spot_claimed_says_why(main_window, tmp_path):
    """~3% of detected peaks lose the claim and are in no spot, so no chromatogram
    was ever extracted for them. That has to read as a reason, not as a blank."""
    mw = main_window
    _open_sample_view(mw, tmp_path)

    mw._on_feature_selected(2)          # rep1_00002, absent from the map

    assert mw.eic_plot._store_key is None
    assert mw.eic_plot._axes == []
    assert "not claimed by any alignment spot" in mw.eic_plot.fig.axes[0].get_title()


def test_a_project_older_than_the_spot_map_falls_back(main_window, tmp_path):
    """No spotmap.json to translate through: say there is no chromatogram, which is
    what such a project has always said. Never plot some other feature's."""
    mw = main_window
    _open_sample_view(mw, tmp_path, with_map=False)

    mw._on_feature_selected(0)

    assert mw.eic_plot._store_key == "rep1_00000"
    assert mw.eic_plot._axes == []
    assert "No chromatogram stored" in mw.eic_plot.fig.axes[0].get_title()
