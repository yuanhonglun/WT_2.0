"""``enable_ms2_driven_branch=False`` skips stage 1 and feeds stage 1b nothing.

The ablation the 2026-10 revision reports (what the MS2-driven branch adds over
MS1-driven detection alone) must be a real switch in the pipeline, not a
post-hoc filter on ``detection_source``: stage 1b decides which MS1 peaks to
keep by looking at the MS2-driven features already present, so a filter after
the fact would under-count the MS1-driven baseline.
"""
from __future__ import annotations

import numpy as np
import pytest

from asfam.config import ProcessingConfig
from asfam.models import CandidateFeature
from asfam.pipeline import orchestrator as orch

FILES = ["fake_100-129_1.mzML"]      # sample id "1", as in test_checkpoint.py


def _candidate(source: str) -> CandidateFeature:
    return CandidateFeature(
        feature_id="F0", segment_name="100-129", replicate_id=1,
        precursor_mz_nominal=120, rt_apex=1.0, rt_left=0.9, rt_right=1.1,
        ms2_mz=np.array([60.0, 70.0, 80.0]), ms2_intensity=np.array([3.0, 2.0, 1.0]),
        n_fragments=3, detection_source=source,
    )


@pytest.fixture()
def stubbed(monkeypatch):
    calls: dict = {"stage1": 0, "stage1b_existing": None}
    current: dict = {}

    def _stage1(data, *a, **k):
        calls["stage1"] += 1
        current.clear()
        current.update({sid: [_candidate("ms2_driven")] for sid in data})
        return current

    def _stage1b(data, existing, *a, **k):
        calls["stage1b_existing"] = {sid: list(feats) for sid, feats in existing.items()}
        merged = {sid: list(existing.get(sid, [])) + [_candidate("ms1_driven")]
                  for sid in data}
        current.clear()          # ``existing`` may be ``current`` itself
        current.update(merged)
        return current

    monkeypatch.setattr(orch, "run_stage0_one_sample", lambda *a, **k: ["segment"])
    monkeypatch.setattr(orch, "run_stage1", _stage1)
    monkeypatch.setattr(orch, "run_stage1b", _stage1b)
    for name in ("run_stage2", "run_stage2b", "run_stage3", "run_stage4",
                 "run_stage5", "run_stage5b", "run_stage6", "run_stage6b_annotation"):
        monkeypatch.setattr(orch, name, lambda *a, **k: current)
    monkeypatch.setattr(orch, "run_stage7", lambda *a, **k: [])
    monkeypatch.setattr(orch, "run_stage8", lambda *a, **k: {})
    return calls, current


def _run(tmp_path, cfg):
    o = orch.PipelineOrchestrator(cfg)
    o.reuse_checkpoints = False
    o.run(FILES, str(tmp_path))
    return o


def test_default_runs_both_branches(tmp_path, stubbed):
    calls, current = stubbed
    _run(tmp_path, ProcessingConfig())
    assert calls["stage1"] == 1
    assert [f.detection_source for f in calls["stage1b_existing"]["1"]] == ["ms2_driven"]


def test_ablation_skips_stage1_and_leaves_only_ms1_driven_features(tmp_path, stubbed):
    calls, current = stubbed
    cfg = ProcessingConfig()
    cfg.enable_ms2_driven_branch = False
    _run(tmp_path, cfg)
    assert calls["stage1"] == 0
    assert calls["stage1b_existing"] == {"1": []}
    assert {f.detection_source for f in current["1"]} == {"ms1_driven"}


def test_the_switch_is_part_of_the_checkpoint_fingerprint():
    from asfam.io.spill import config_fingerprint
    on, off = ProcessingConfig(), ProcessingConfig()
    off.enable_ms2_driven_branch = False
    assert config_fingerprint(on, None, FILES) != config_fingerprint(off, None, FILES)
