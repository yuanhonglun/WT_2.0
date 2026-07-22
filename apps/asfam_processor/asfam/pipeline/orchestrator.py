"""Pipeline orchestrator: one sample at a time, then align.

Stages 0-6.5 run per sample and the finished features are spilled to
``<output_dir>/_work/<sample_id>.{mspec,mfeat}`` before the next sample is
loaded, so peak RSS is set by *one* sample's raw scans plus the spectral
library and does not grow with the number of samples. Alignment and export
then read the spill back, with raw data and library both already freed.

Each spilled sample is also a checkpoint: a re-run with the same parameters
skips it. ``_work/`` is never deleted automatically — the GUI offers an
explicit "clear intermediate results" action.
"""
from __future__ import annotations

import gc
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Callable

from asfam.config import ProcessingConfig
from asfam.models import Feature
from asfam.io import spill
from metabo_core.pipeline_labels import stage_label
from metabo_core.utils.memlog import log_memory

from asfam.pipeline.group_ids import namespace_group_ids
from asfam.pipeline.stage0_load import group_files_by_sample, run_stage0_one_sample
from asfam.pipeline.stage1_ms2_detection import run_stage1
from asfam.pipeline.stage1b_ms1_detection import run_stage1b
from asfam.pipeline.stage2_ms1_assignment import run_stage2
from asfam.pipeline.stage2b_inference import run_stage2b
from asfam.pipeline.stage3_merge_segments import run_stage3
from asfam.pipeline.stage4_isotope_dedup import run_stage4
from asfam.pipeline.stage5_adduct_dedup import run_stage5
from asfam.pipeline.stage5b_duplicate_detection import run_stage5b
from asfam.pipeline.stage6_isf_detection import run_stage6
from asfam.pipeline.stage6b_annotation import run_stage6b_annotation
from asfam.pipeline.stage7_alignment import GapFillContext, run_stage7
from asfam.pipeline.stage8_export import run_stage8

logger = logging.getLogger(__name__)

WORK_DIRNAME = "_work"


def _numeric_stage7_stats(details: dict) -> dict:
    """Flatten Stage 7 count/timing scalars for ``processing_report.txt``."""
    result = {}
    for section, values in sorted(details.items()):
        if not isinstance(values, dict):
            continue
        for name, value in sorted(values.items()):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[f"{section}_{name}"] = value
    return result


class PipelineOrchestrator:
    """Orchestrates the ASFAM processing pipeline."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._callbacks: list[Callable] = []
        self._cancel = threading.Event()
        self.stage_stats: dict[str, dict] = {}
        # Spill directory + the samples in it, in place of the old in-memory
        # ``raw_data`` / ``candidates_by_rep``. The GUI reads what it needs
        # from here on demand.
        self.work_dir: Optional[Path] = None
        self.sample_ids: list[str] = []
        self.sample_files: dict[str, list[str]] = {}
        # Set to False by the GUI when the user declines to reuse a checkpoint.
        self.reuse_checkpoints: bool = True
        self.run_context = None  # populated at the start of run()

    def add_progress_callback(
        self, callback: Callable[[str, int, int, str], None],
    ) -> None:
        """Register a progress callback: (stage, current, total, message)."""
        self._callbacks.append(callback)

    def cancel(self) -> None:
        """Request pipeline cancellation."""
        self._cancel.set()

    # ------------------------------------------------------------------
    # Library lifetime (CLAUDE.md "谱库生命周期")
    # ------------------------------------------------------------------

    class _LazyLibrary:
        """Defers the library load until a sample actually needs annotating.

        ``run()`` used to load before the per-sample loop, so a resumed run whose
        checkpoints all hit still paid for reading and indexing the library it
        would never open — most of the wall clock of a stage 7+8-only rerun on
        ``lib/lcms/pos.msp``.

        Still one load for the whole run, and still released in ``run()``'s
        ``finally`` before alignment: the library-lifetime invariant is about
        *when it dies*, and that is unchanged.
        """

        def __init__(self, load) -> None:
            self._load = load
            self._value: Optional[tuple] = None

        def get(self) -> tuple:
            if self._value is None:
                self._value = self._load()
            return self._value

        @property
        def loaded(self) -> bool:
            return self._value is not None

        def close(self) -> None:
            self._value = None

    def _load_library(self, spectral_library_path: Optional[str], need_lean: bool):
        """Load the library once for the whole run.

        Returns ``(lean_list, index)``. The lean list is only materialized for
        stage 2.5's opt-in library inference — the one consumer that walks the
        flat list. The CSR index costs ~28 s to build on ``lib/lcms/pos.msp``,
        so it is built once and shared by every sample's stage 6.5 rather than
        rebuilt per sample.
        """
        if not spectral_library_path:
            return None, None

        from metabo_core.annotation import build_index_from_list, load_and_index_library

        if need_lean:
            from asfam.pipeline.stage2b_inference import _load_library
            lean = _load_library(spectral_library_path)
            self._mem("library loaded")
            if lean:
                index = build_index_from_list(lean)
                self._mem("library indexed")
                return lean, index

        index = load_and_index_library(spectral_library_path)
        self._mem("library indexed")
        return None, index

    # ------------------------------------------------------------------
    # Per-sample stages 0 - 6.5
    # ------------------------------------------------------------------

    def _process_one_sample(
        self,
        sample_id: str,
        paths: list[str],
        library: "_LazyLibrary",
        spectral_library_path: Optional[str],
    ) -> int:
        """Run stages 0-6.5 for one sample and return its feature count.

        Every stage below already loops per replicate over a dict that now holds
        exactly one entry, so no stage code changed: they never read raw data
        belonging to another sample.
        """
        # First sample that actually needs processing pays the library load; a
        # run whose checkpoints all hit never opens it. See ``_LazyLibrary``.
        lean_library, library_index = library.get()
        tag = f"sample {sample_id}"

        self._check_cancel()
        self._mem(f"{tag} stage0 pre")
        t0 = time.time()
        segments = run_stage0_one_sample(paths, self.config, self._emit)
        if not segments:
            logger.warning("Sample %s: no segment could be loaded, skipping", sample_id)
            return 0
        self._mem(f"{tag} stage0 post")
        self._accumulate("Stage 0: Load",
                         {"files": len(paths), "replicates": 1}, time.time() - t0)

        data = {sample_id: segments}

        self._check_cancel()
        t0 = time.time()
        feats = run_stage1(data, self.config, self._emit)
        n_ms2_driven = len(feats[sample_id])
        self._accumulate("Stage 1: MS2 Detection",
                         {"total_features": n_ms2_driven}, time.time() - t0)

        self._check_cancel()
        self._emit("stage1b", 0, 1, stage_label("asfam", "stage1b", "start"))
        t0 = time.time()
        feats = run_stage1b(data, feats, self.config, self._emit)
        n_after_1b = len(feats[sample_id])
        self._accumulate("Stage 1b: MS1 Detection",
                         {"ms1_driven_new": n_after_1b - n_ms2_driven,
                          "total_features": n_after_1b}, time.time() - t0)

        self._check_cancel()
        t0 = time.time()
        feats = run_stage2(data, feats, self.config, self._emit)
        self._accumulate(
            "Stage 2: MS1 Assignment",
            {"ms1_detected": sum(1 for f in feats[sample_id]
                                 if f.signal_type == "ms1_detected"),
             "ms2_only": sum(1 for f in feats[sample_id]
                             if f.signal_type == "ms2_only")},
            time.time() - t0,
        )

        self._check_cancel()
        self._emit("stage2b", 0, 1, stage_label("asfam", "stage2b", "start"))
        t0 = time.time()
        # Pass the path only when the preload actually produced a list: without
        # this guard a failed preload would make stage 2.5 re-read the library
        # from disk once per sample.
        feats = run_stage2b(feats, self.config,
                            spectral_library_path if lean_library else None,
                            self._emit, preloaded_library=lean_library)
        self._accumulate("Stage 2.5: Inference",
                         {"features_after": len(feats[sample_id])}, time.time() - t0)

        self._check_cancel()
        t0 = time.time()
        feats = run_stage3(feats, self.config, self._emit)
        self._accumulate("Stage 3: Merge",
                         {"features_after": len(feats[sample_id])}, time.time() - t0)

        self._check_cancel()
        self._emit("stage4", 0, 1, stage_label("asfam", "stage4", "start"))
        t0 = time.time()
        feats = run_stage4(feats, self.config, self._emit, data_by_replicate=data)
        elapsed = time.time() - t0
        n_iso = len(feats[sample_id])
        self._accumulate("Stage 4: Isotope Dedup", {"features_after": n_iso}, elapsed)
        self._emit("stage4", 1, 1,
                   stage_label("asfam", "stage4", "done", elapsed=round(elapsed, 1),
                               detail=f"{n_iso} features remain"))

        self._check_cancel()
        self._emit("stage5", 0, 1, stage_label("asfam", "stage5", "start"))
        t0 = time.time()
        feats = run_stage5(feats, data, self.config, self._emit)
        elapsed = time.time() - t0
        n_add = len(feats[sample_id])
        self._accumulate("Stage 5: Adduct Dedup", {"features_after": n_add}, elapsed)
        self._emit("stage5", 1, 1,
                   stage_label("asfam", "stage5", "done", elapsed=round(elapsed, 1),
                               detail=f"{n_add} features remain"))

        self._check_cancel()
        self._emit("stage5b", 0, 1, stage_label("asfam", "stage5b", "start"))
        t0 = time.time()
        feats = run_stage5b(feats, self.config, self._emit, data_by_replicate=data)
        elapsed = time.time() - t0
        n_dups = sum(1 for f in feats[sample_id] if f.is_duplicate)
        self._accumulate("Stage 5b: Duplicate Detection",
                         {"duplicates_flagged": n_dups}, elapsed)
        self._emit("stage5b", 1, 1,
                   stage_label("asfam", "stage5b", "done", elapsed=round(elapsed, 1),
                               detail=f"{n_dups} duplicates flagged"))

        self._check_cancel()
        self._emit("stage6", 0, 1, stage_label("asfam", "stage6", "start"))
        t0 = time.time()
        feats = run_stage6(feats, data, self.config, self._emit)
        elapsed = time.time() - t0
        n_isf = len(feats[sample_id])
        self._accumulate("Stage 6: ISF Detection", {"features_after": n_isf}, elapsed)
        self._emit("stage6", 1, 1,
                   stage_label("asfam", "stage6", "done", elapsed=round(elapsed, 1),
                               detail=f"{n_isf} features remain"))

        # Raw scans are dead from here on: only stages 1-6 read them. Drop them
        # before annotation, which is the other memory-hungry stage.
        del data
        segments = None
        gc.collect()
        self._mem(f"{tag} raw released")

        self._check_cancel()
        self._emit("stage6b", 0, 1, stage_label("asfam", "stage6b", "start"))
        t0 = time.time()
        # Same guard as stage 2.5: a None index means "no usable library", and
        # stage 6.5 must skip rather than try to load one per sample.
        feats = run_stage6b_annotation(
            feats, self.config,
            spectral_library_path if library_index is not None else None,
            self._emit, library_index=library_index,
        )
        elapsed = time.time() - t0
        self._accumulate("Stage 6.5: Annotation", {}, elapsed)
        self._emit("stage6b", 1, 1,
                   stage_label("asfam", "stage6b", "done", elapsed=round(elapsed, 1)))

        sample_features = feats[sample_id]
        n_features = len(sample_features)
        spill.write_sample(
            self.work_dir / sample_id, sample_features,
            fingerprint=self._fingerprint(sample_id, spectral_library_path),
            sample_id=sample_id,
        )
        del feats, sample_features
        gc.collect()
        self._mem(f"{tag} spilled")
        return n_features

    # ------------------------------------------------------------------

    def run(
        self,
        mzml_paths: list[str],
        output_dir: str,
        spectral_library_path: Optional[str] = None,
        sample_groups: Optional[dict] = None,
    ) -> list[Feature]:
        """Run the full pipeline from Stage 0 to Stage 8."""
        # Build run context for feedback system (purely additive; algorithm
        # code must not depend on this).
        try:
            from dataclasses import asdict
            from metabo_gui.feedback import build_run_context
            from metabo_core import __version__ as _metra_version

            self.run_context = build_run_context(
                app="asfam",
                metra_version=_metra_version,
                input_files=mzml_paths,
                library_path=spectral_library_path,
                project_file=None,
                export_dir=output_dir,
                params=asdict(self.config),
            )
        except Exception:
            logger.warning("Failed to build run context (non-fatal)", exc_info=True)

        logger.info("=" * 60)
        logger.info("METRA — ASFAM Pipeline Starting")
        logger.info("  Files: %d", len(mzml_paths))
        logger.info("  Output: %s", output_dir)
        logger.info("  Mode: %s", self.config.ionization_mode)
        logger.info("=" * 60)

        t_start = time.time()
        self.stage_stats = {}

        self.sample_files = group_files_by_sample(mzml_paths, sample_groups)
        # Sorted so the run is reproducible: stage 7 picks its reference by
        # ``max(quality)``, which resolves ties by iteration order.
        self.sample_ids = sorted(self.sample_files)
        self.work_dir = Path(output_dir) / WORK_DIRNAME
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if not self.reuse_checkpoints:
            n = spill.clear_work_dir(self.work_dir)
            if n:
                logger.info("Discarded %d stale intermediate files in %s",
                            n, self.work_dir)

        logger.info("  Samples: %s", ", ".join(self.sample_ids))

        # One library load for the whole run, on first use; released before
        # alignment.
        self._mem("library pre")
        library = self._LazyLibrary(lambda: self._load_library(
            spectral_library_path,
            need_lean=bool(getattr(self.config, "enable_library_mz_inference", False)),
        ))

        loop_completed = False
        try:
            n_total = 0
            for i, sample_id in enumerate(self.sample_ids, start=1):
                stem = self.work_dir / sample_id
                fingerprint = self._fingerprint(sample_id, spectral_library_path)
                if self.reuse_checkpoints and spill.sample_is_complete(stem, fingerprint):
                    manifest = spill.read_manifest(stem) or {}
                    logger.info("Sample %s (%d/%d): reusing checkpoint from %s "
                                "(%s features)", sample_id, i, len(self.sample_ids),
                                manifest.get("created", "?"),
                                manifest.get("n_features", "?"))
                    self._emit("stage0", i, len(self.sample_ids),
                               f"Sample {sample_id}: reusing intermediate results")
                    n_total += int(manifest.get("n_features", 0) or 0)
                    # Without this the processing report would simply omit
                    # stages 0-6.5 on a resumed run, which reads as "they never
                    # ran" rather than "they were reused".
                    self._accumulate("Stage 0-6.5: Reused Checkpoints",
                                     {"samples": 1,
                                      "features": int(manifest.get("n_features", 0) or 0)},
                                     0.0)
                    continue

                logger.info("=" * 60)
                logger.info("Sample %s (%d/%d): %d segments",
                            sample_id, i, len(self.sample_ids),
                            len(self.sample_files[sample_id]))
                logger.info("=" * 60)
                n_total += self._process_one_sample(
                    sample_id, self.sample_files[sample_id],
                    library, spectral_library_path,
                )
            loop_completed = True
        finally:
            # Stage 6.5 is the library's last reader. Drop it before alignment +
            # export, which are themselves memory-hungry; for lib/lcms/pos.msp
            # the lean list plus the CSR index is several GB. See the
            # library-lifetime invariant in CLAUDE.md.
            was_loaded = library.loaded
            library.close()
            gc.collect()
            if was_loaded:
                self._mem("library released")
            elif loop_completed:
                # Not the same as "the load failed": a failed load leaves the
                # holder empty too, and it reaches this `finally` on its way out.
                logger.info("Library never loaded: every sample reused a checkpoint")

        # A sample whose mzML files all failed to load never spilled; drop it
        # rather than let alignment trip over the missing .mfeat.
        spilled = [sid for sid in self.sample_ids
                   if spill.sample_is_complete(self.work_dir / sid)]
        for sid in self.sample_ids:
            if sid not in spilled:
                logger.warning("Sample %s produced no intermediate result "
                               "and is excluded from alignment", sid)
        self.sample_ids = spilled
        if not self.sample_ids:
            raise RuntimeError("No sample could be processed")

        logger.info("  Spilled %d features across %d samples",
                    n_total, len(self.sample_ids))

        # Stage 7: Alignment — reads the spill back, no raw data, no library.
        self._check_cancel()
        self._emit("stage7", 0, 1, stage_label("asfam", "stage7", "start"))
        self._mem("stage7 pre")
        t0 = time.time()
        stage7_details = {}
        features = run_stage7(
            self._read_spilled_features(), self.config, self._emit,
            ms2_reader=self._ms2_reader,
            gap_fill=self._gap_fill_context(output_dir),
            stats_out=stage7_details,
            mapping_output_dir=output_dir,
        )
        self._mem("stage7 post")
        stage7_elapsed = round(time.time() - t0, 1)
        self.stage_stats["Stage 7: Alignment"] = {
            "aligned_features": len(features),
            "time_sec": stage7_elapsed,
            **_numeric_stage7_stats(stage7_details),
        }
        self._emit(
            "stage7", 1, 1,
            stage_label("asfam", "stage7", "done", elapsed=stage7_elapsed,
                        detail=f"{len(features)} aligned features"),
        )

        features = self._export(features, output_dir)

        total_time = time.time() - t_start
        logger.info("=" * 60)
        logger.info("Pipeline complete in %.1f seconds", total_time)
        logger.info("  Final features: %d", len(features))
        logger.info("  Intermediate results kept in %s", self.work_dir)
        logger.info("=" * 60)

        return features

    def run_reannotate(
        self,
        output_dir: str,
        spectral_library_path=None,
        work_dir: Optional[str] = None,
    ) -> list:
        """Re-run annotation (6.5) + alignment (7) + export (8) only.

        Reads each sample's features back from the spill, re-annotates it, and
        writes it out again — so the checkpoint stays the source of truth and
        only one sample's features are ever resident alongside the library.
        """
        logger.info("=" * 60)
        logger.info("Re-annotation Starting (stages 6.5 -> 7 -> 8)")
        logger.info("=" * 60)

        t_start = time.time()
        self.work_dir = Path(work_dir) if work_dir else Path(output_dir) / WORK_DIRNAME
        if not self.sample_ids:
            self.sample_ids = [sid for sid, _ in spill.scan_checkpoints(self.work_dir)]
        if not self.sample_ids:
            raise RuntimeError(
                f"No intermediate results found in {self.work_dir}; "
                "run the full pipeline first."
            )

        # Stage 2.5 does not run here, so the flat list is never needed.
        self._mem("reannotate start")
        lean_library, library_index = self._load_library(
            spectral_library_path, need_lean=False,
        )

        try:
            self._check_cancel()
            self._emit("stage6b", 0, 1, stage_label("asfam", "stage6b", "start"))
            t0 = time.time()
            for sample_id in self.sample_ids:
                stem = self.work_dir / sample_id
                # Deliberately *not* namespaced with group_ids.namespace_group_ids:
                # this read is round-tripped back to disk below, so an offset
                # applied here would be baked into the spill and then applied a
                # second time when alignment reads it. Namespacing belongs to
                # _read_spilled_features(), which never writes.
                feats = spill.read_sample_features(stem, load_ms2=True)
                for c in feats:
                    c.matchms_name = None
                    c.matchms_score = None
                    c.annotation_matches = []
                    c.selected_annotation_idx = 0
                run_stage6b_annotation(
                    {sample_id: feats}, self.config,
                    spectral_library_path if library_index is not None else None,
                    self._emit, library_index=library_index,
                )
                # Re-annotation can run against a *different* library, so the
                # old fingerprint no longer describes what is on disk. Rewrite
                # it when we still know the sample's inputs, otherwise drop it:
                # a null fingerprint never matches, so the next full run
                # recomputes the sample instead of trusting a stale checkpoint.
                spill.write_sample(
                    stem, feats,
                    fingerprint=(self._fingerprint(sample_id, spectral_library_path)
                                 if self.sample_files.get(sample_id) else None),
                    sample_id=sample_id,
                )
                del feats
                gc.collect()
            self.stage_stats["Stage 6.5: Re-annotation"] = {
                "time_sec": round(time.time() - t0, 1),
            }
        finally:
            lean_library = None
            library_index = None
            gc.collect()
            self._mem("library released")

        self._check_cancel()
        self._emit("stage7", 0, 1, stage_label("asfam", "stage7", "start"))
        self._mem("stage7 pre")
        t0 = time.time()
        stage7_details = {}
        features = run_stage7(
            self._read_spilled_features(), self.config, self._emit,
            ms2_reader=self._ms2_reader,
            gap_fill=self._gap_fill_context(output_dir),
            stats_out=stage7_details,
            mapping_output_dir=output_dir,
        )
        self._mem("stage7 post")
        self.stage_stats["Stage 7: Alignment"] = {
            "aligned_features": len(features),
            "time_sec": round(time.time() - t0, 1),
            **_numeric_stage7_stats(stage7_details),
        }

        features = self._export(features, output_dir)

        logger.info("Re-annotation complete in %.1f sec: %d features",
                     time.time() - t_start, len(features))
        return features

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_spilled_features(self) -> dict[str, list]:
        """Read every sample's features back, MS2 spectra left on disk.

        The joiner scores an MS2 cosine only on the ~16k candidate edges, so it
        pulls spectra one at a time through :meth:`_ms2_reader` instead of
        holding every replicate's spectra resident.

        This is the one place several samples' features meet, so it is where the
        dedup stages' per-sample group ids get a namespace: they are written by
        counters that restart at zero for every sample, and alignment copies them
        onto the aligned ``Feature`` unchanged. See
        :func:`asfam.pipeline.group_ids.namespace_group_ids`.
        """
        features = {
            sid: spill.read_sample_features(self.work_dir / sid, load_ms2=False)
            for sid in self.sample_ids
        }
        namespace_group_ids(features)
        return features

    def _gap_fill_context(self, output_dir: str) -> Optional[GapFillContext]:
        """Where gap filling finds raw data — ``None`` when we no longer know.

        ``run_reannotate`` on a project opened from disk has the spill but not
        the mzML paths. Skipping the fill there keeps the exported quantitation
        honest (holes stay holes) rather than inventing zeros, at the cost of a
        stale ``alignment.eic``.
        """
        missing = [sid for sid in self.sample_ids if not self.sample_files.get(sid)]
        if missing:
            logger.warning(
                "Gap fill skipped: no raw file paths for %s", ", ".join(missing),
            )
            return None
        return GapFillContext(
            sample_files={sid: self.sample_files[sid] for sid in self.sample_ids},
            output_dir=output_dir,
            temp_dir=self.work_dir,
        )

    def _ms2_reader(self, sample_id: str, feature) -> Optional[tuple]:
        """Random-read one feature's MS2 out of that sample's ``.mspec``.

        Also back-fills ``ms2_sn`` / ``ms2_gaussian`` on the feature: the arrays
        live in the ``.mspec`` body, and stage 7 aggregates the representative's
        ``ms2_gaussian`` into ``Feature.gaussian_similarity``. The joiner reads
        every representative's spectrum, so the field is always populated by the
        time stage 7 looks at it.
        """
        ptr = getattr(feature, "ms2_seek_ptr", None)
        if ptr is None:
            return None
        mz, intensity, sn, gaussian = spill.read_ms2_full(self.work_dir / sample_id, ptr)
        feature.ms2_sn = sn
        feature.ms2_gaussian = gaussian
        if len(mz) == 0:
            return None
        return mz, intensity

    def _export(self, features: list[Feature], output_dir: str) -> list[Feature]:
        self._check_cancel()
        self._emit("stage8", 0, 1, stage_label("asfam", "stage8", "start"))
        self._mem("stage8 pre")
        t0 = time.time()
        outputs = run_stage8(
            features, output_dir, self.config, self.stage_stats, self._emit,
        )
        self._mem("stage8 post")
        stage8_elapsed = round(time.time() - t0, 1)
        self.stage_stats["Stage 8: Export"] = {
            "output_files": list(outputs.keys()),
            "time_sec": stage8_elapsed,
        }
        self._emit(
            "stage8", 1, 1,
            stage_label("asfam", "stage8", "done", elapsed=stage8_elapsed),
        )
        return features

    def _fingerprint(self, sample_id: str, spectral_library_path: Optional[str]) -> str:
        return spill.config_fingerprint(
            self.config, spectral_library_path, self.sample_files.get(sample_id, []),
        )

    def _accumulate(self, stage_name: str, counters: dict, elapsed: float) -> None:
        """Sum a per-sample stage's counters and wall clock into stage_stats."""
        entry = self.stage_stats.setdefault(stage_name, {"time_sec": 0.0})
        for key, value in counters.items():
            entry[key] = entry.get(key, 0) + value
        entry["time_sec"] = round(entry.get("time_sec", 0.0) + elapsed, 1)

    def _mem(self, label: str) -> None:
        """Record RSS at a stage boundary (see metabo_core.utils.memlog)."""
        log_memory(logger, label)

    def _emit(self, stage: str, current: int, total: int, msg: str) -> None:
        """Notify progress callbacks."""
        for cb in self._callbacks:
            try:
                cb(stage, current, total, msg)
            except Exception:
                pass

    def _check_cancel(self) -> None:
        """Check if cancellation was requested."""
        if self._cancel.is_set():
            raise RuntimeError("Pipeline cancelled by user")
