"""Checkpoint resume: a spilled sample is skipped, a changed parameter is not.

Stages 0-6.5 take hours on a full-range run, and the two ASFAM crashes we have
logs for lost all of it. Each spilled sample is now a resume point, keyed on a
fingerprint of every parameter that can change its output.
"""
from __future__ import annotations

import numpy as np
import pytest

from asfam.config import ProcessingConfig
from asfam.io import spill
from asfam.models import CandidateFeature
from asfam.pipeline import orchestrator as orch


FILES = ["fake_270-271_1.mzML", "fake_270-271_2.mzML"]


def _candidate(fid="F1") -> CandidateFeature:
    return CandidateFeature(
        feature_id=fid, segment_name="270-271", replicate_id=1,
        precursor_mz_nominal=270, rt_apex=0.5, rt_left=0.45, rt_right=0.55,
        ms2_mz=np.array([50.0, 100.0], dtype=np.float64),
        ms2_intensity=np.array([1000.0, 800.0], dtype=np.float64),
        n_fragments=2, ms1_precursor_mz=270.5, ms1_height=10000.0,
    )


@pytest.fixture()
def stubbed(monkeypatch):
    """Stub every stage; count how many samples reach stage 1."""
    processed: list[str] = []
    current: dict = {}

    def _stage1(data, *a, **k):
        processed.extend(data)
        current.clear()
        current.update({sid: [_candidate()] for sid in data})
        return current

    monkeypatch.setattr(orch, "run_stage0_one_sample", lambda *a, **k: ["segment"])
    monkeypatch.setattr(orch, "run_stage1", _stage1)
    for name in ("run_stage1b", "run_stage2", "run_stage2b", "run_stage3",
                 "run_stage4", "run_stage5", "run_stage5b", "run_stage6",
                 "run_stage6b_annotation"):
        monkeypatch.setattr(orch, name, lambda *a, **k: current)
    monkeypatch.setattr(orch, "run_stage7", lambda *a, **k: [])
    monkeypatch.setattr(orch, "run_stage8", lambda *a, **k: {})
    return processed


def _run(tmp_path, cfg=None, reuse=True):
    o = orch.PipelineOrchestrator(cfg or ProcessingConfig())
    o.reuse_checkpoints = reuse
    o.run(FILES, str(tmp_path))
    return o


def test_second_run_skips_the_spilled_samples(tmp_path, stubbed):
    o = _run(tmp_path)
    assert o.sample_ids == ["1", "2"]
    assert stubbed == ["1", "2"]

    stubbed.clear()
    _run(tmp_path)
    assert stubbed == [], "already-spilled samples were recomputed"


def test_a_half_written_sample_is_recomputed(tmp_path, stubbed):
    """The manifest is written last, so a crash before it means "not done"."""
    _run(tmp_path)
    (tmp_path / orch.WORK_DIRNAME / "2.json").unlink()

    stubbed.clear()
    _run(tmp_path)
    assert stubbed == ["2"]


def test_changing_a_stage_parameter_invalidates_the_checkpoint(tmp_path, stubbed):
    _run(tmp_path)

    changed = ProcessingConfig()
    changed.ms1_min_height += 1.0
    stubbed.clear()
    _run(tmp_path, changed)
    assert stubbed == ["1", "2"]


def test_changing_an_alignment_or_export_parameter_does_not(tmp_path, stubbed):
    """The spill is the state at the end of stage 6.5; stage 7/8 knobs and the
    worker count cannot change it, and re-tuning them must stay cheap."""
    _run(tmp_path)

    changed = ProcessingConfig()
    changed.alignment_rt_tolerance += 0.05
    # Stage 7 only, like the rest: sweeping it must not cost 10 minutes of
    # stages 0-6.5 per point.
    changed.alignment_ms1_covered_threshold = 0.8
    changed.alignment_ms2_identity_min_matched_fragments += 1
    changed.export_include_duplicates = not changed.export_include_duplicates
    changed.n_workers += 1
    stubbed.clear()
    _run(tmp_path, changed)
    assert stubbed == []


def test_declining_reuse_wipes_the_work_dir(tmp_path, stubbed):
    _run(tmp_path)
    stubbed.clear()
    _run(tmp_path, reuse=False)
    assert stubbed == ["1", "2"]


def test_export_keeps_the_work_dir(tmp_path, stubbed):
    """``_work/`` survives a successful run on purpose — only the GUI's explicit
    "clear intermediates" action deletes it."""
    o = _run(tmp_path)
    assert sorted(sid for sid, _ in spill.scan_checkpoints(o.work_dir)) == ["1", "2"]


# --- PR-7: the library is loaded on first use, not before the loop ----------


def _run_with_library(tmp_path, lib_path, loads: list) -> None:
    o = orch.PipelineOrchestrator(ProcessingConfig())
    o.reuse_checkpoints = True

    def _counting_load(self, path, need_lean):   # noqa: ANN001
        loads.append(path)
        return None, None

    orch.PipelineOrchestrator._load_library, real = _counting_load, \
        orch.PipelineOrchestrator._load_library
    try:
        o.run(FILES, str(tmp_path), str(lib_path))
    finally:
        orch.PipelineOrchestrator._load_library = real


def test_a_fully_resumed_run_never_loads_the_library(tmp_path, stubbed):
    """Reading and indexing ``lib/lcms/pos.msp`` costs minutes and gigabytes.
    A rerun that only wants stage 7+8 hits every checkpoint and annotates
    nothing, so it must never open the library at all."""
    lib = tmp_path / "lib.msp"
    lib.write_text("NAME: x\nNum Peaks: 1\n100.0 1.0\n", encoding="utf-8")

    loads: list = []
    _run_with_library(tmp_path, lib, loads)
    assert stubbed == ["1", "2"]
    assert len(loads) == 1, "the library should load once for the whole run"

    stubbed.clear()
    loads.clear()
    _run_with_library(tmp_path, lib, loads)
    assert stubbed == [], "already-spilled samples were recomputed"
    assert loads == [], "a fully-resumed run loaded the library it never reads"
