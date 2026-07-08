"""Pipeline worker thread for background execution."""
from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal

from asfam.config import ProcessingConfig
from asfam.pipeline.orchestrator import PipelineOrchestrator


class PipelineWorker(QThread):
    """Runs the processing pipeline in a background thread."""

    progress_update = pyqtSignal(str, int, int, str)
    pipeline_completed = pyqtSignal(object, object)    # features, raw_data
    pipeline_error = pyqtSignal(str)

    def __init__(
        self,
        config: ProcessingConfig,
        mzml_paths: list[str],
        output_dir: str,
        library_path: Optional[str] = None,
        sample_groups: Optional[dict] = None,
    ):
        super().__init__()
        self.config = config
        self.mzml_paths = mzml_paths
        self.output_dir = output_dir
        self.library_path = library_path
        self.sample_groups = sample_groups
        self.orchestrator = PipelineOrchestrator(config)
        self.orchestrator.add_progress_callback(self._on_progress)

    def run(self):
        try:
            features = self.orchestrator.run(
                self.mzml_paths, self.output_dir,
                self.library_path, self.sample_groups,
            )
            self.pipeline_completed.emit(features, self.orchestrator.raw_data)
        except Exception as e:
            import traceback
            import logging
            logging.exception("Pipeline worker error")
            self.pipeline_error.emit(str(e))

    def cancel(self):
        self.orchestrator.cancel()

    def _on_progress(self, stage, current, total, msg):
        self.progress_update.emit(stage, current, total, msg)


class ReAnnotateWorker(QThread):
    """Runs only re-annotation (stages 6.5 -> 7 -> 8) in a background thread."""

    progress_update = pyqtSignal(str, int, int, str)
    pipeline_completed = pyqtSignal(object, object)
    pipeline_error = pyqtSignal(str)

    def __init__(
        self,
        config: ProcessingConfig,
        candidates_by_rep: dict,
        raw_data: Optional[dict],
        output_dir: str,
        library_path: Optional[str] = None,
    ):
        super().__init__()
        self.config = config
        self.candidates_by_rep = candidates_by_rep
        self.raw_data = raw_data
        self.output_dir = output_dir
        self.library_path = library_path
        self.orchestrator = PipelineOrchestrator(config)
        self.orchestrator.add_progress_callback(self._on_progress)

    def run(self):
        try:
            import copy
            candidates_copy = copy.deepcopy(self.candidates_by_rep)
            features = self.orchestrator.run_reannotate(
                candidates_copy, self.raw_data or {},
                self.output_dir, self.library_path,
            )
            self.pipeline_completed.emit(features, self.raw_data)
        except Exception as e:
            import logging
            logging.exception("Re-annotation worker error")
            self.pipeline_error.emit(str(e))

    def cancel(self):
        self.orchestrator.cancel()

    def _on_progress(self, stage, current, total, msg):
        self.progress_update.emit(stage, current, total, msg)
