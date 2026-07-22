"""The ASFAM orchestrator must release the spectral library after stage 6.5.

Platform invariant (CLAUDE.md, "谱库生命周期"): a library lives only for the
duration of annotation. ASFAM is the strictest case — it loads the library once
for the whole run, before the per-sample loop, so stage 2.5 (MS2-only m/z
inference) and stage 6.5 (annotation) of *every* sample share one load. Without
an explicit release, that list survived stage 7 alignment and stage 8 export,
which are themselves memory-hungry; for ``lib/lcms/pos.msp`` the lean list plus
its CSR index is several GB, and holding both at once is what pushes peak RSS
over the edge.

The library is a function-local of ``run()`` / ``run_reannotate()``, so
``ctx.library_index is None`` (DDA's guard) has no analogue — and asserting
"freed after the run returns" would be vacuous, since the frame dies anyway.
Instead we hand the orchestrator a weak-referenceable library and check the
referent from *inside* the stage-7 stub: it must already be collected by the
time alignment starts. That both proves the object was really freed (not just a
name rebound) and pins the release to the right point.
"""
from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from asfam.config import ProcessingConfig
from asfam.io import spill
from asfam.models import CandidateFeature
from asfam.pipeline import orchestrator as orch
from asfam.pipeline import stage2b_inference


# Two fragments shared with the candidate below, so both stage 2.5 inference
# and stage 6.5 annotation find a real hit and we know the library was read.
TINY_MSP = (
    "NAME: TestCompoundA\nPRECURSORMZ: 270.5\nFORMULA: C10H10N2O5\n"
    "ADDUCT: [M+H]+\nRETENTIONTIME: 0.5\n"
    "INCHIKEY: AAAAAAAAAAAAAA\nSMILES: CCO\nCOMMENT: bulky, unused\n"
    "Num Peaks: 3\n50.0 1000\n100.0 800\n150.0 500\n\n"
)

# parse_filename must be able to read a segment range and a replicate id out of
# this: the per-sample loop groups on filenames before it loads anything.
FAKE_MZML = "fake_270-271_1.mzML"
SAMPLE_ID = "1"


class _TrackedList(list):
    """A list that supports weak references (plain ``list`` does not)."""


@pytest.fixture()
def tiny_lib(tmp_path):
    path = tmp_path / "tiny_lib.msp"
    path.write_text(TINY_MSP, encoding="utf-8")
    return str(path)


def _candidate(signal_type: str = "ms1_detected") -> CandidateFeature:
    return CandidateFeature(
        feature_id="F1",
        segment_name="270-271",
        replicate_id=1,
        precursor_mz_nominal=270,
        rt_apex=0.5,
        rt_left=0.45,
        rt_right=0.55,
        ms2_mz=np.array([50.0, 100.0, 150.0], dtype=np.float64),
        ms2_intensity=np.array([1000.0, 800.0, 500.0], dtype=np.float64),
        n_fragments=3,
        ms1_precursor_mz=270.5,
        ms1_height=10000.0,
        signal_type=signal_type,
        annotation_matches=[],
    )


def _track_lean_library(monkeypatch) -> "list[weakref.ref]":
    """Wrap ``_load_library`` so we hold only a weak reference to its result.

    ``orchestrator`` imports ``_load_library`` from ``stage2b_inference``
    *inside* the function body, so patching the source module is enough.
    """
    real = stage2b_inference._load_library
    refs: list[weakref.ref] = []

    def _wrapped(path):
        spectra = real(path)
        assert spectra, "fixture library should load"
        tracked = _TrackedList(spectra)
        refs.append(weakref.ref(tracked))
        return tracked

    monkeypatch.setattr(stage2b_inference, "_load_library", _wrapped)
    return refs


def _track_library_index(monkeypatch) -> "list[weakref.ref]":
    """Weakly track the spectrum list inside the index ``load_and_index_library``
    returns. The index itself is a plain dict and cannot be weak-referenced, so
    swap its ``spectra`` list — the multi-GB part — for a trackable copy. The
    CSR arrays were built already and hold the spectra, not the list object."""
    import metabo_core.annotation as ann

    real = ann.load_and_index_library
    refs: list[weakref.ref] = []

    def _wrapped(path):
        index = real(path)
        assert index, "fixture library should load"
        tracked = _TrackedList(index["spectra"])
        index["spectra"] = tracked
        refs.append(weakref.ref(tracked))
        return index

    monkeypatch.setattr(ann, "load_and_index_library", _wrapped)
    return refs


def _alive(refs: "list[weakref.ref]") -> bool:
    """Is the tracked library still reachable right now?

    ``gc.collect()`` first: the orchestrator's own collect already ran, but a
    reference cycle created since then could keep the referent alive until the
    next sweep, which would make this probe flaky rather than wrong.
    """
    gc.collect()
    return any(r() is not None for r in refs)


def _stub_stages(monkeypatch, feats: dict, seen: dict, refs) -> None:
    """Stub every stage except the two that genuinely consume the library.

    Keeps this a unit test of the orchestrator's memory contract rather than an
    end-to-end run. Stages 7/8 run *after* the release point, so they sample the
    library's liveness: asserting only after run() returns would pass even
    without the release, because the orchestrator's local dies with the frame.
    """
    monkeypatch.setattr(orch, "run_stage0_one_sample", lambda *a, **k: ["segment"])
    monkeypatch.setattr(orch, "run_stage1", lambda *a, **k: feats)
    for name in ("run_stage1b", "run_stage2", "run_stage3", "run_stage4",
                 "run_stage5", "run_stage5b", "run_stage6"):
        monkeypatch.setattr(orch, name, lambda *a, **k: feats)

    def _stage7(*a, **k):
        seen["align"] = _alive(refs)
        return []

    def _stage8(*a, **k):
        seen["export"] = _alive(refs)
        return {}

    monkeypatch.setattr(orch, "run_stage7", _stage7)
    monkeypatch.setattr(orch, "run_stage8", _stage8)


def test_run_releases_preloaded_library_after_stage6b(monkeypatch, tmp_path, tiny_lib):
    """The stage 2.5 / 6.5 shared preload is freed before alignment + export.

    Stage 2.5's library path is opt-in (``enable_library_mz_inference``
    defaults to False), so enable it explicitly — that is the only
    configuration in which ``run()`` materializes the flat list at all.
    """
    refs = _track_lean_library(monkeypatch)

    feats = {SAMPLE_ID: [_candidate(signal_type="ms2_only")]}
    seen: dict[str, bool] = {}
    _stub_stages(monkeypatch, feats, seen, refs)

    cfg = ProcessingConfig()
    cfg.enable_library_mz_inference = True
    cfg.library_mz_inference_threshold = 0.0  # accept the self-match
    cfg.matchms_min_matched_peaks = 1

    o = orch.PipelineOrchestrator(cfg)
    o.run([FAKE_MZML], str(tmp_path), spectral_library_path=tiny_lib)

    assert refs, "_load_library should have been called (inference is enabled)"
    # The library was really used: stage 2.5 inferred the precursor m/z from it.
    feat = feats[SAMPLE_ID][0]
    assert feat.mz_source == "library", (
        f"stage 2.5 did not use the library (mz_source={feat.mz_source!r})"
    )
    assert feat.matchms_name == "TestCompoundA"  # stage 6.5 read it too

    assert seen == {"align": False, "export": False}, (
        "spectral library was still alive during align/export; it must be "
        f"released once the last sample's stage 6.5 is done (liveness: {seen})"
    )


def test_library_is_loaded_once_for_every_sample(monkeypatch, tmp_path, tiny_lib):
    """The per-sample loop must not re-read (or re-index) the library per sample."""
    import metabo_core.annotation as ann

    calls = {"n": 0}
    real = ann.load_and_index_library

    def _counting(path):
        calls["n"] += 1
        return real(path)

    monkeypatch.setattr(ann, "load_and_index_library", _counting)

    current: dict = {}

    def _stage1(data, *a, **k):
        # Stage 1 is handed {sample_id: segments}; echo one candidate back under
        # whichever sample the loop is on.
        current.clear()
        current.update({sid: [_candidate()] for sid in data})
        return current

    monkeypatch.setattr(orch, "run_stage0_one_sample", lambda *a, **k: ["segment"])
    monkeypatch.setattr(orch, "run_stage1", _stage1)
    for name in ("run_stage1b", "run_stage2", "run_stage3", "run_stage4",
                 "run_stage5", "run_stage5b", "run_stage6"):
        monkeypatch.setattr(orch, name, lambda *a, **k: current)
    monkeypatch.setattr(orch, "run_stage7", lambda *a, **k: [])
    monkeypatch.setattr(orch, "run_stage8", lambda *a, **k: {})

    cfg = ProcessingConfig()
    cfg.matchms_min_matched_peaks = 1
    o = orch.PipelineOrchestrator(cfg)
    o.run(["fake_270-271_1.mzML", "fake_270-271_2.mzML", "fake_270-271_3.mzML"],
          str(tmp_path), spectral_library_path=tiny_lib)

    assert o.sample_ids == ["1", "2", "3"]
    assert calls["n"] == 1, f"library loaded {calls['n']} times for 3 samples"


def test_reannotate_releases_library_after_stage6b(monkeypatch, tmp_path, tiny_lib):
    """``run_reannotate`` (GUI "Re-annotate") loads its own library and must
    free it on the same terms as ``run()``."""
    refs = _track_library_index(monkeypatch)

    work = tmp_path / orch.WORK_DIRNAME
    spill.write_sample(work / "rep0", [_candidate()], sample_id="rep0")

    seen: dict[str, bool] = {}

    def _stage7(*a, **k):
        seen["align"] = _alive(refs)
        return []

    def _stage8(*a, **k):
        seen["export"] = _alive(refs)
        return {}

    monkeypatch.setattr(orch, "run_stage7", _stage7)
    monkeypatch.setattr(orch, "run_stage8", _stage8)

    o = orch.PipelineOrchestrator(ProcessingConfig())
    o.run_reannotate(str(tmp_path), spectral_library_path=tiny_lib)

    assert refs, "load_and_index_library should have been called"
    # Stage 6.5 really matched against it — read the re-spilled sample back.
    reloaded = spill.read_sample_features(work / "rep0", load_ms2=True)
    assert reloaded[0].matchms_name == "TestCompoundA"

    assert seen == {"align": False, "export": False}, (
        "spectral library survived re-annotation; it must be released at the "
        f"end of stage 6.5 (liveness by stage: {seen})"
    )
