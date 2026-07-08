"""Pipeline orchestrator: runs all stages in sequence."""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Callable

from asfam.config import ProcessingConfig
from asfam.models import Feature
from metabo_core.pipeline_labels import stage_label

from asfam.pipeline.stage0_load import run_stage0
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
from asfam.pipeline.stage7_alignment import run_stage7
from asfam.pipeline.stage8_export import run_stage8

logger = logging.getLogger(__name__)


def _restore_duplicate_status(features_by_replicate: dict) -> None:
    """Restore is_duplicate features to status="active" so they participate in
    annotation and alignment. Kept-and-flagged duplicates (stages 4/5/5b/6) use
    status="*_removed"; this must run BEFORE stage6b annotation, which gates on
    status=="active". The is_duplicate flag still controls UI/export visibility.
    """
    for feats in features_by_replicate.values():
        for f in feats:
            if f.is_duplicate and f.status != "active":
                f.status = "active"
            if f.is_duplicate and not f.duplicate_type:
                f.duplicate_type = "spectral"


class PipelineOrchestrator:
    """Orchestrates the ASFAM processing pipeline."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._callbacks: list[Callable] = []
        self._cancel = threading.Event()
        self.stage_stats: dict[str, dict] = {}
        self.raw_data: Optional[dict] = None  # persisted for GUI EIC viewing
        self.candidates_by_rep: Optional[dict] = None  # per-replicate candidates
        self.run_context = None  # populated at the start of run()

    def add_progress_callback(
        self, callback: Callable[[str, int, int, str], None],
    ) -> None:
        """Register a progress callback: (stage, current, total, message)."""
        self._callbacks.append(callback)

    def cancel(self) -> None:
        """Request pipeline cancellation."""
        self._cancel.set()

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

        # Stage 0: Load
        self._check_cancel()
        t0 = time.time()
        data_by_replicate = run_stage0(
            mzml_paths, self.config, self._emit, sample_groups,
        )
        self.stage_stats["Stage 0: Load"] = {
            "files": len(mzml_paths),
            "replicates": len(data_by_replicate),
            "time_sec": round(time.time() - t0, 1),
        }
        self.raw_data = data_by_replicate  # persist for GUI

        # Stage 1: MS2 detection
        self._check_cancel()
        t0 = time.time()
        features_by_rep = run_stage1(
            data_by_replicate, self.config, self._emit,
        )
        total_feat = sum(len(f) for f in features_by_rep.values())
        self.stage_stats["Stage 1: MS2 Detection"] = {
            "total_features": total_feat,
            "time_sec": round(time.time() - t0, 1),
        }

        # Stage 1b: MS1-driven detection (complementary)
        self._check_cancel()
        self._emit("stage1b", 0, 1, stage_label("asfam", "stage1b", "start"))
        t0 = time.time()
        features_by_rep = run_stage1b(
            data_by_replicate, features_by_rep, self.config, self._emit,
        )
        total_after_1b = sum(len(f) for f in features_by_rep.values())
        n_ms1_driven = total_after_1b - total_feat
        self.stage_stats["Stage 1b: MS1 Detection"] = {
            "ms1_driven_new": n_ms1_driven,
            "total_features": total_after_1b,
            "time_sec": round(time.time() - t0, 1),
        }

        # Stage 2: MS1 assignment
        self._check_cancel()
        t0 = time.time()
        features_by_rep = run_stage2(
            data_by_replicate, features_by_rep, self.config, self._emit,
        )
        high = sum(1 for feats in features_by_rep.values()
                    for f in feats if f.signal_type == "ms1_detected")
        low = sum(1 for feats in features_by_rep.values()
                   for f in feats if f.signal_type == "ms2_only")
        self.stage_stats["Stage 2: MS1 Assignment"] = {
            "ms1_detected": high,
            "ms2_only": low,
            "time_sec": round(time.time() - t0, 1),
        }

        # Stage 2.5: Inference
        self._check_cancel()
        # Pre-load library once, share between stage 2.5 and 6.5
        self._emit("stage2b", 0, 1, stage_label("asfam", "stage2b", "start"))
        loaded_library = None
        if spectral_library_path and getattr(self.config, "enable_library_mz_inference", False):
            from asfam.pipeline.stage2b_inference import _load_library
            loaded_library = _load_library(spectral_library_path)
        t0 = time.time()
        features_by_rep = run_stage2b(
            features_by_rep, self.config, spectral_library_path, self._emit,
            preloaded_library=loaded_library,
        )
        total_after = sum(len(f) for f in features_by_rep.values())
        self.stage_stats["Stage 2.5: Inference"] = {
            "features_after": total_after,
            "time_sec": round(time.time() - t0, 1),
        }

        # Stage 3: Merge segments
        self._check_cancel()
        t0 = time.time()
        features_by_rep = run_stage3(
            features_by_rep, self.config, self._emit,
        )
        total_merged = sum(len(f) for f in features_by_rep.values())
        self.stage_stats["Stage 3: Merge"] = {
            "features_after": total_merged,
            "time_sec": round(time.time() - t0, 1),
        }

        # Stage 4: Isotope dedup
        self._check_cancel()
        self._emit("stage4", 0, 1, stage_label("asfam", "stage4", "start"))
        t0 = time.time()
        features_by_rep = run_stage4(
            features_by_rep, self.config, self._emit,
            data_by_replicate=data_by_replicate,
        )
        total_iso = sum(len(f) for f in features_by_rep.values())
        stage4_elapsed = round(time.time() - t0, 1)
        self.stage_stats["Stage 4: Isotope Dedup"] = {
            "features_after": total_iso,
            "time_sec": stage4_elapsed,
        }
        self._emit(
            "stage4", 1, 1,
            stage_label("asfam", "stage4", "done", elapsed=stage4_elapsed,
                        detail=f"{total_iso} features remain"),
        )

        # Stage 5: Adduct dedup
        self._check_cancel()
        self._emit("stage5", 0, 1, stage_label("asfam", "stage5", "start"))
        t0 = time.time()
        features_by_rep = run_stage5(
            features_by_rep, data_by_replicate, self.config, self._emit,
        )
        total_add = sum(len(f) for f in features_by_rep.values())
        stage5_elapsed = round(time.time() - t0, 1)
        self.stage_stats["Stage 5: Adduct Dedup"] = {
            "features_after": total_add,
            "time_sec": stage5_elapsed,
        }
        self._emit(
            "stage5", 1, 1,
            stage_label("asfam", "stage5", "done", elapsed=stage5_elapsed,
                        detail=f"{total_add} features remain"),
        )

        # Stage 5b: Duplicate detection
        self._check_cancel()
        self._emit("stage5b", 0, 1, stage_label("asfam", "stage5b", "start"))
        t0 = time.time()
        features_by_rep = run_stage5b(features_by_rep, self.config, self._emit,
                                       data_by_replicate=data_by_replicate)
        n_dups = sum(
            sum(1 for f in feats if f.is_duplicate)
            for feats in features_by_rep.values()
        )
        stage5b_elapsed = round(time.time() - t0, 1)
        self.stage_stats["Stage 5b: Duplicate Detection"] = {
            "duplicates_flagged": n_dups,
            "time_sec": stage5b_elapsed,
        }
        self._emit(
            "stage5b", 1, 1,
            stage_label("asfam", "stage5b", "done", elapsed=stage5b_elapsed,
                        detail=f"{n_dups} duplicates flagged"),
        )

        # Stage 6: ISF detection
        self._check_cancel()
        self._emit("stage6", 0, 1, stage_label("asfam", "stage6", "start"))
        t0 = time.time()
        features_by_rep = run_stage6(
            features_by_rep, data_by_replicate, self.config, self._emit,
        )
        total_isf = sum(len(f) for f in features_by_rep.values())
        stage6_elapsed = round(time.time() - t0, 1)
        self.stage_stats["Stage 6: ISF Detection"] = {
            "features_after": total_isf,
            "time_sec": stage6_elapsed,
        }
        self._emit(
            "stage6", 1, 1,
            stage_label("asfam", "stage6", "done", elapsed=stage6_elapsed,
                        detail=f"{total_isf} features remain"),
        )

        # Stage 6.5: Library annotation
        self._check_cancel()
        # Restore duplicate status BEFORE annotation so isotope/adduct/isf
        # features are not excluded by stage6b's status=="active" gate. The
        # score-floor guard inside stage6b keeps weak/wrong names from leaking
        # back in; the is_duplicate flag still controls UI/export visibility.
        _restore_duplicate_status(features_by_rep)
        self._emit("stage6b", 0, 1, stage_label("asfam", "stage6b", "start"))
        t0 = time.time()
        features_by_rep = run_stage6b_annotation(
            features_by_rep, self.config, spectral_library_path, self._emit,
            preloaded_library=loaded_library,
        )
        stage6b_elapsed = round(time.time() - t0, 1)
        self.stage_stats["Stage 6.5: Annotation"] = {
            "time_sec": stage6b_elapsed,
        }
        self._emit(
            "stage6b", 1, 1,
            stage_label("asfam", "stage6b", "done", elapsed=stage6b_elapsed),
        )

        # Duplicate status was already restored to "active" BEFORE stage6b
        # annotation (see above), so alignment includes these features and
        # their (score-floor-gated) annotations.

        # Stage 7: Alignment
        self._check_cancel()
        self._emit("stage7", 0, 1, stage_label("asfam", "stage7", "start"))
        self.candidates_by_rep = features_by_rep  # save for project file
        t0 = time.time()
        features = run_stage7(
            features_by_rep, data_by_replicate, self.config, self._emit,
        )
        stage7_elapsed = round(time.time() - t0, 1)
        self.stage_stats["Stage 7: Alignment"] = {
            "aligned_features": len(features),
            "time_sec": stage7_elapsed,
        }
        self._emit(
            "stage7", 1, 1,
            stage_label("asfam", "stage7", "done", elapsed=stage7_elapsed,
                        detail=f"{len(features)} aligned features"),
        )

        # Stage 8: Export
        self._check_cancel()
        self._emit("stage8", 0, 1, stage_label("asfam", "stage8", "start"))
        t0 = time.time()
        outputs = run_stage8(
            features, output_dir, self.config, self.stage_stats, self._emit,
        )
        stage8_elapsed = round(time.time() - t0, 1)
        self.stage_stats["Stage 8: Export"] = {
            "output_files": list(outputs.keys()),
            "time_sec": stage8_elapsed,
        }
        self._emit(
            "stage8", 1, 1,
            stage_label("asfam", "stage8", "done", elapsed=stage8_elapsed),
        )

        total_time = time.time() - t_start
        logger.info("=" * 60)
        logger.info("Pipeline complete in %.1f seconds", total_time)
        logger.info("  Final features: %d", len(features))
        logger.info("=" * 60)

        return features

    def run_reannotate(
        self,
        candidates_by_rep: dict,
        data_by_replicate: dict,
        output_dir: str,
        spectral_library_path=None,
    ) -> list:
        """Re-run annotation (6.5) + alignment (7) + export (8) only."""
        logger.info("=" * 60)
        logger.info("Re-annotation Starting (stages 6.5 -> 7 -> 8)")
        logger.info("=" * 60)

        t_start = time.time()

        # Clear previous annotation data
        for rep_id, cands in candidates_by_rep.items():
            for c in cands:
                c.matchms_name = None
                c.matchms_score = None
                c.annotation_matches = []
                c.selected_annotation_idx = 0

        # Load library
        loaded_library = None
        if spectral_library_path:
            from asfam.pipeline.stage2b_inference import _load_library
            loaded_library = _load_library(spectral_library_path)

        # Stage 6.5: Re-annotate
        self._check_cancel()
        self._emit("stage6b", 0, 1, stage_label("asfam", "stage6b", "start"))
        t0 = time.time()
        candidates_by_rep = run_stage6b_annotation(
            candidates_by_rep, self.config, spectral_library_path, self._emit,
            preloaded_library=loaded_library,
        )
        self.stage_stats["Stage 6.5: Re-annotation"] = {
            "time_sec": round(time.time() - t0, 1),
        }
        self.candidates_by_rep = candidates_by_rep

        # Stage 7: Alignment
        self._check_cancel()
        self._emit("stage7", 0, 1, stage_label("asfam", "stage7", "start"))
        t0 = time.time()
        features = run_stage7(
            candidates_by_rep, data_by_replicate or {}, self.config, self._emit,
        )
        self.stage_stats["Stage 7: Alignment"] = {
            "aligned_features": len(features),
            "time_sec": round(time.time() - t0, 1),
        }

        # Stage 8: Export
        self._check_cancel()
        self._emit("stage8", 0, 1, stage_label("asfam", "stage8", "start"))
        t0 = time.time()
        outputs = run_stage8(
            features, output_dir, self.config, self.stage_stats, self._emit,
        )
        self.stage_stats["Stage 8: Export"] = {
            "output_files": list(outputs.keys()),
            "time_sec": round(time.time() - t0, 1),
        }

        logger.info("Re-annotation complete in %.1f sec: %d features",
                     time.time() - t_start, len(features))
        return features

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
