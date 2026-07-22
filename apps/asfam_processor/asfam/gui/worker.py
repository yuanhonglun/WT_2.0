"""Pipeline worker thread for background execution."""
from __future__ import annotations

import logging
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal

from asfam.config import ProcessingConfig
from asfam.pipeline.orchestrator import PipelineOrchestrator
from metabo_core.utils.memlog import rss_gib


class PipelineWorker(QThread):
    """Runs the processing pipeline in a background thread."""

    progress_update = pyqtSignal(str, int, int, str)
    pipeline_completed = pyqtSignal(object)            # features
    pipeline_error = pyqtSignal(str)

    def __init__(
        self,
        config: ProcessingConfig,
        mzml_paths: list[str],
        output_dir: str,
        library_path: Optional[str] = None,
        sample_groups: Optional[dict] = None,
        reuse_checkpoints: bool = True,
    ):
        super().__init__()
        self.config = config
        self.mzml_paths = mzml_paths
        self.output_dir = output_dir
        self.library_path = library_path
        self.sample_groups = sample_groups
        self.orchestrator = PipelineOrchestrator(config)
        self.orchestrator.reuse_checkpoints = reuse_checkpoints
        self.orchestrator.add_progress_callback(self._on_progress)

    def run(self):
        try:
            features = self.orchestrator.run(
                self.mzml_paths, self.output_dir,
                self.library_path, self.sample_groups,
            )
            self.pipeline_completed.emit(features)
        except MemoryError:
            # A MemoryError means Python's allocator refused. The two ASFAM
            # crashes we have logs for showed no traceback at all -- the OS
            # killed the process. Recording RSS here is what lets the next
            # crash be told apart from those.
            logging.critical("[mem] MemoryError in pipeline worker, RSS=%.2f GiB",
                             rss_gib())
            raise
        except Exception as e:
            logging.exception("Pipeline worker error")
            self.pipeline_error.emit(str(e))

    def cancel(self):
        self.orchestrator.cancel()

    def _on_progress(self, stage, current, total, msg):
        self.progress_update.emit(stage, current, total, msg)


class ReAnnotateWorker(QThread):
    """Runs only re-annotation (stages 6.5 -> 7 -> 8) in a background thread.

    Reads each sample's candidates back from the ``_work/`` spill, one at a
    time, so peak memory stays at one sample plus the library. The caller is
    expected to have warned the user that re-annotation overwrites the current
    annotations -- ``run_reannotate`` clears them before it starts, so a
    cancelled run leaves the spilled candidates without annotations.
    """

    progress_update = pyqtSignal(str, int, int, str)
    pipeline_completed = pyqtSignal(object)
    pipeline_error = pyqtSignal(str)

    def __init__(
        self,
        config: ProcessingConfig,
        output_dir: str,
        library_path: Optional[str] = None,
        work_dir: Optional[str] = None,
        sample_files: Optional[dict] = None,
    ):
        super().__init__()
        self.config = config
        self.output_dir = output_dir
        self.library_path = library_path
        self.work_dir = work_dir
        self.orchestrator = PipelineOrchestrator(config)
        # Re-annotation re-runs alignment, so it re-runs gap fill too — which
        # needs the raw files. Without them the fill is skipped and the
        # quantitation matrix comes back with holes.
        self.orchestrator.sample_files = dict(sample_files or {})
        self.orchestrator.add_progress_callback(self._on_progress)

    def run(self):
        try:
            features = self.orchestrator.run_reannotate(
                self.output_dir, self.library_path, work_dir=self.work_dir,
            )
            self.pipeline_completed.emit(features)
        except MemoryError:
            logging.critical("[mem] MemoryError in re-annotation worker, RSS=%.2f GiB",
                             rss_gib())
            raise
        except Exception as e:
            logging.exception("Re-annotation worker error")
            self.pipeline_error.emit(str(e))

    def cancel(self):
        self.orchestrator.cancel()

    def _on_progress(self, stage, current, total, msg):
        self.progress_update.emit(stage, current, total, msg)
