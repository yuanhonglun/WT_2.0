"""Pipeline orchestrator: runs all stages in sequence."""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Callable

from asfam.config import ProcessingConfig
from asfam.models import Feature

from asfam.pipeline.stage0_load import run_stage0
from asfam.pipeline.stage1_ms2_detection import run_stage1
from asfam.pipeline.stage1b_ms1_detection import run_stage1b
from asfam.pipeline.stage2_ms1_assignment import run_stage2
from asfam.pipeline.stage2b_inference import run_stage2b
from asfam.pipeline.stage3_merge_segments import run_stage3
from asfam.pipeline.stage4_isotope_dedup import run_stage4
from asfam.pipeline.stage5_adduct_dedup import run_stage5
from asfam.pipeline.stage6_isf_detection import run_stage6
from asfam.pipeline.stage6b_annotation import run_stage6b_annotation
from asfam.pipeline.stage7_alignment import run_stage7
from asfam.pipeline.stage8_export import run_stage8

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the ASFAM processing pipeline."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._callbacks: list[Callable] = []
        self._cancel = threading.Event()
        self.stage_stats: dict[str, dict] = {}
        self.raw_data: Optional[dict] = None  # persisted for GUI EIC viewing
        self.candidates_by_rep: Optional[dict] = None  # per-replicate candidates

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
        logger.info("=" * 60)
        logger.info("ASFAMProcessor Pipeline Starting")
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
        self._emit("stage1b", 0, 1, "MS1-driven feature detection...")
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
        self._emit("stage2b", 0, 1, "Loading library...")
        loaded_library = None
        if spectral_library_path:
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
        self._emit("stage4", 0, 1, "Isotope deduplication...")
        t0 = time.time()
        features_by_rep = run_stage4(
            features_by_rep, self.config, self._emit,
        )
        total_iso = sum(len(f) for f in features_by_rep.values())
        self.stage_stats["Stage 4: Isotope Dedup"] = {
            "features_after": total_iso,
            "time_sec": round(time.time() - t0, 1),
        }
        self._emit("stage4", 1, 1, f"Done: {total_iso} features remain")

        # Stage 5: Adduct dedup
        self._check_cancel()
        self._emit("stage5", 0, 1, "Adduct deduplication...")
        t0 = time.time()
        features_by_rep = run_stage5(
            features_by_rep, data_by_replicate, self.config, self._emit,
        )
        total_add = sum(len(f) for f in features_by_rep.values())
        self.stage_stats["Stage 5: Adduct Dedup"] = {
            "features_after": total_add,
            "time_sec": round(time.time() - t0, 1),
        }
        self._emit("stage5", 1, 1, f"Done: {total_add} features remain")

        # Stage 6: ISF detection
        self._check_cancel()
        self._emit("stage6", 0, 1, "ISF detection...")
        t0 = time.time()
        features_by_rep = run_stage6(
            features_by_rep, data_by_replicate, self.config, self._emit,
        )
        total_isf = sum(len(f) for f in features_by_rep.values())
        self.stage_stats["Stage 6: ISF Detection"] = {
            "features_after": total_isf,
            "time_sec": round(time.time() - t0, 1),
        }
        self._emit("stage6", 1, 1, f"Done: {total_isf} features remain")

        # Stage 6.5: Library annotation
        self._check_cancel()
        self._emit("stage6b", 0, 1, "Library annotation...")
        t0 = time.time()
        features_by_rep = run_stage6b_annotation(
            features_by_rep, self.config, spectral_library_path, self._emit,
            preloaded_library=loaded_library,
        )
        self.stage_stats["Stage 6.5: Annotation"] = {
            "time_sec": round(time.time() - t0, 1),
        }
        self._emit("stage6b", 1, 1, "Annotation complete")

        # Stage 7: Alignment
        self._check_cancel()
        self._emit("stage7", 0, 1, "Cross-replicate alignment...")
        self.candidates_by_rep = features_by_rep  # save for project file
        t0 = time.time()
        features = run_stage7(
            features_by_rep, data_by_replicate, self.config, self._emit,
        )
        self.stage_stats["Stage 7: Alignment"] = {
            "aligned_features": len(features),
            "time_sec": round(time.time() - t0, 1),
        }
        self._emit("stage7", 1, 1, f"Done: {len(features)} aligned features")

        # Stage 8: Export
        self._check_cancel()
        self._emit("stage8", 0, 1, "Exporting results...")
        t0 = time.time()
        outputs = run_stage8(
            features, output_dir, self.config, self.stage_stats, self._emit,
        )
        self.stage_stats["Stage 8: Export"] = {
            "output_files": list(outputs.keys()),
            "time_sec": round(time.time() - t0, 1),
        }
        self._emit("stage8", 1, 1, "Export complete")

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
        self._emit("stage6b", 0, 1, "Re-annotating...")
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
        self._emit("stage7", 0, 1, "Cross-replicate alignment...")
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
        self._emit("stage8", 0, 1, "Exporting results...")
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
