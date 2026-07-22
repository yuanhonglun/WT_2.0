"""Main application window with themed UI, sample/aligned view switch."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QToolBar, QAction, QMessageBox, QFileDialog,
    QComboBox, QLabel,
)
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThread, pyqtSignal as QtSignal
from PyQt5.QtGui import QIcon, QColor, QPalette

from metabo_gui.about import show_about_dialog
from metabo_gui.resources import app_icon_path
from metabo_gui.theme import STYLESHEET

from asfam.config import ProcessingConfig
from asfam.io.eic_store import SpotMap, load_spot_map, open_store
from asfam.models import Feature
from asfam import __version__

from asfam.gui.setup_panel import SetupPanel
from asfam.gui.scatter_plot import ScatterPlotWidget
from asfam.gui.feature_table import FeatureTableWidget
from asfam.gui.eic_plot import EICPlotWidget
from asfam.gui.ms2_plot import MS2PlotWidget
from asfam.gui.progress_panel import ProgressPanel
from asfam.gui.worker import PipelineWorker

logger = logging.getLogger(__name__)


class _UpdateChecker(QThread):
    """Background thread to check for new releases on GitHub."""
    update_available = QtSignal(str, str)  # (version, download_url)

    def run(self):
        try:
            import urllib.request, json
            url = "https://api.github.com/repos/yuanhonglun/WT_2.0/releases/latest"
            req = urllib.request.Request(url, headers={"User-Agent": "ASFAMProcessor"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            tag = data.get("tag_name", "").lstrip("v")
            html_url = data.get("html_url", "")
            from asfam import __version__
            if tag and tag > __version__:
                self.update_available.emit(tag, html_url)
        except Exception:
            pass


class MainWindow(QMainWindow):
    """ASFAMProcessor main window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"ASFAM Processor v{__version__}")
        self.resize(1400, 900)

        # Set icon — shared platform asset (Plan F-followup-3 #6).
        icon = app_icon_path()
        if icon.exists():
            self.setWindowIcon(QIcon(str(icon)))

        # Apply theme
        self.setStyleSheet(STYLESHEET)

        self.config = ProcessingConfig()
        self.features: list[Feature] = []
        self.worker: Optional[PipelineWorker] = None
        # Per-sample candidates live in <output>/_work, not in memory. A project
        # file saved before v1.0.260709.5 still embeds them; ``_candidates_by_rep``
        # holds those so old projects keep working.
        self._candidates_by_rep: Optional[dict] = None
        self._work_dir: Optional[str] = None
        # Candidate id -> aligned spot id, loaded beside alignment.eic. Empty
        # until a run or a project load fills it in.
        self._spotmap = SpotMap()
        self._sample_files: dict[str, list[str]] = {}
        self._stage_stats: dict = {}
        self._mzml_paths: list[str] = []
        self._library_path: Optional[str] = None
        self._view_mode = "aligned"
        self._saving_project = False  # guard for close protection

        # Feedback system state (Task 10)
        self._feedback_controller = None
        self._notes_dock = None

        self._build_toolbar()
        self._build_ui()
        self._connect_signals()

        self.statusBar().showMessage("Ready")

        # Background version check
        self._update_checker = _UpdateChecker()
        self._update_checker.update_available.connect(self._show_update_dialog)
        self._update_checker.start()

    def _show_update_dialog(self, version: str, url: str):
        QMessageBox.information(
            self, "Update Available",
            f"A new version (v{version}) of ASFAM Processor is available.\n\n"
            f"Download: {url}",
        )

    def _build_toolbar(self):
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.act_run = QAction("Run", self)
        self.act_run.triggered.connect(self._on_run)
        toolbar.addAction(self.act_run)

        self.act_stop = QAction("Stop", self)
        self.act_stop.setEnabled(False)
        self.act_stop.triggered.connect(self._on_stop)
        toolbar.addAction(self.act_stop)

        self.act_reannotate = QAction("Re-annotate", self)
        self.act_reannotate.setEnabled(False)
        self.act_reannotate.triggered.connect(self._on_reannotate)
        toolbar.addAction(self.act_reannotate)

        toolbar.addSeparator()

        self.act_save_project = QAction("Save Project", self)
        self.act_save_project.setEnabled(False)
        self.act_save_project.triggered.connect(self._on_save_project)
        toolbar.addAction(self.act_save_project)

        self.act_open_project = QAction("Open Project", self)
        self.act_open_project.triggered.connect(self._on_open_project)
        toolbar.addAction(self.act_open_project)

        # Intermediate results are kept on purpose (they let a crashed or
        # re-parameterized export skip stages 0-6.5), so the only way to delete
        # them is this explicit action.
        self.act_clear_work = QAction("Clear Intermediates", self)
        self.act_clear_work.setEnabled(False)
        self.act_clear_work.triggered.connect(self._on_clear_work_dir)
        toolbar.addAction(self.act_clear_work)

        toolbar.addSeparator()

        self.act_export = QAction("Export", self)
        self.act_export.setEnabled(False)
        self.act_export.triggered.connect(self._on_export)
        toolbar.addAction(self.act_export)

        toolbar.addSeparator()

        # View mode selector
        toolbar.addWidget(QLabel(" View: "))
        self.view_combo = QComboBox()
        self.view_combo.addItem("Aligned (all replicates)")
        self.view_combo.setEnabled(False)
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        toolbar.addWidget(self.view_combo)

        toolbar.addSeparator()

        # Language selector (disabled for now, will add full i18n later)
        # self.lang_combo = QComboBox()
        # self.lang_combo.addItems(["English", "中文"])
        # self.lang_combo.setFixedWidth(80)
        # self.lang_combo.currentIndexChanged.connect(self._on_language_changed)
        # toolbar.addWidget(self.lang_combo)

        # Notes panel toggle
        self.act_toggle_notes = QAction("Notes Panel", self)
        self.act_toggle_notes.triggered.connect(self._toggle_notes_dock)
        toolbar.addAction(self.act_toggle_notes)

        toolbar.addSeparator()

        # About
        self.act_about = QAction("About", self)
        self.act_about.triggered.connect(self._on_about)
        toolbar.addAction(self.act_about)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        top_splitter = QSplitter(Qt.Horizontal)

        self.setup_panel = SetupPanel(self.config)
        top_splitter.addWidget(self.setup_panel)

        right_splitter = QSplitter(Qt.Vertical)

        upper_splitter = QSplitter(Qt.Vertical)
        self.scatter_plot = ScatterPlotWidget()
        self.scatter_plot.setMinimumHeight(200)
        upper_splitter.addWidget(self.scatter_plot)
        self.feature_table = FeatureTableWidget()
        self.feature_table.setMinimumHeight(100)
        upper_splitter.addWidget(self.feature_table)
        upper_splitter.setSizes([350, 250])
        right_splitter.addWidget(upper_splitter)

        lower_widget = QWidget()
        lower_layout = QHBoxLayout(lower_widget)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        self.eic_plot = EICPlotWidget()
        self.ms2_plot = MS2PlotWidget()
        lower_layout.addWidget(self.eic_plot)
        lower_layout.addWidget(self.ms2_plot)
        right_splitter.addWidget(lower_widget)
        right_splitter.setSizes([500, 300])

        top_splitter.addWidget(right_splitter)
        top_splitter.setSizes([280, 1100])

        main_layout.addWidget(top_splitter, stretch=1)

        self.progress_panel = ProgressPanel()
        main_layout.addWidget(self.progress_panel)

    def _connect_signals(self):
        self.feature_table.featureSelected.connect(self._on_feature_selected)
        self.scatter_plot.pointClicked.connect(self._on_scatter_clicked)
        self.scatter_plot.filterChanged.connect(self._on_scatter_filter_changed)
        self.ms2_plot.annotationSelected.connect(self._on_annotation_selected)

    def _on_scatter_filter_changed(self):
        """Sync feature-table filters with the scatter plot checkboxes.
        ``Annotated only`` flows through too, using the GUI's "Library
        Match Thr" value (``matchms_similarity_threshold``) as the
        single, user-tunable score gate."""
        self.feature_table.proxy.set_show_duplicates(self.scatter_plot.show_duplicates)
        thr = float(getattr(
            self.config, "matchms_similarity_threshold", 0.3,
        ))
        self.scatter_plot.set_annotated_threshold(thr)
        self.feature_table.proxy.set_annotated_threshold(thr)
        mm = int(getattr(self.config, "matchms_min_matched_peaks", 3) or 0)
        self.scatter_plot.set_annotated_min_matched(mm)
        self.feature_table.proxy.set_annotated_min_matched(mm)
        self.feature_table.proxy.set_annotated_only(self.scatter_plot.annotated_only)

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def _on_run(self):
        mzml_paths = self.setup_panel.get_mzml_paths()
        if not mzml_paths:
            QMessageBox.warning(self, "No Files", "Please add mzML files first.")
            return
        output_dir = self.setup_panel.get_output_dir()
        if not output_dir:
            QMessageBox.warning(self, "No Output", "Please select an output directory.")
            return

        self.setup_panel.apply_to_config()

        reuse = self._ask_reuse_checkpoints(output_dir)
        if reuse is None:
            return  # user cancelled

        self.progress_panel.reset()
        self.act_run.setEnabled(False)
        self.act_stop.setEnabled(True)
        self.act_export.setEnabled(False)

        self.worker = PipelineWorker(
            self.config, mzml_paths, output_dir,
            self.setup_panel.get_library_path(),
            self.setup_panel.get_sample_groups(),
            reuse_checkpoints=reuse,
        )
        self.worker.progress_update.connect(self.progress_panel.update_progress)
        self.worker.pipeline_completed.connect(self._on_pipeline_done)
        self.worker.pipeline_error.connect(self._on_pipeline_error)
        self.worker.start()
        self.progress_panel.log(f"Pipeline started with {len(mzml_paths)} files...")

    def _ask_reuse_checkpoints(self, output_dir: str) -> Optional[bool]:
        """Offer to skip samples already spilled under ``<output_dir>/_work``.

        Returns True/False, or None if the user cancelled the run. Answering
        "yes" still lets the orchestrator recompute a sample whose config
        fingerprint changed — the prompt only asks about intent.
        """
        from asfam.io import spill
        from asfam.pipeline.orchestrator import WORK_DIRNAME

        work_dir = os.path.join(output_dir, WORK_DIRNAME)
        found = spill.scan_checkpoints(work_dir)
        if not found:
            return True

        newest = max((created for _, created in found if created), default="?")
        answer = QMessageBox.question(
            self, "Intermediate Results Found",
            f"Found intermediate results for {len(found)} sample(s) "
            f"(latest: {newest}).\n\n"
            "Reuse them and skip stages 0-6.5 for those samples?\n"
            "Choosing No recomputes every sample from the raw files.",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Cancel:
            return None
        return answer == QMessageBox.Yes

    def _on_clear_work_dir(self):
        """Delete the spilled intermediate results (never done automatically)."""
        if not self._work_dir:
            return
        from asfam.io import spill

        answer = QMessageBox.question(
            self, "Clear Intermediates",
            f"Delete the intermediate results in\n{self._work_dir}?\n\n"
            "A later re-run will have to redo stages 0-6.5 for every sample.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        n = spill.clear_work_dir(self._work_dir)
        self.progress_panel.log(f"Removed {n} intermediate file(s).")
        self.act_clear_work.setEnabled(False)
        self.act_reannotate.setEnabled(False)

    def _on_stop(self):
        if self.worker:
            self.worker.cancel()
            self.progress_panel.log("Cancellation requested...")

    def _on_pipeline_done(self, features):
        self.features = features

        if self.worker and self.worker.orchestrator:
            orch = self.worker.orchestrator
            self._stage_stats = orch.stage_stats
            # ReAnnotateWorker's orchestrator never grouped the inputs, so keep
            # whatever the full run (or the project load) already established.
            if orch.work_dir:
                self._work_dir = str(orch.work_dir)
                # Candidates now live in _work/, not memory. Only drop the
                # inline copy a legacy project may have loaded once a run has
                # actually spilled a replacement.
                self._candidates_by_rep = None
            if orch.sample_files:
                self._sample_files = dict(orch.sample_files)
            if hasattr(self.worker, 'mzml_paths'):
                self._mzml_paths = self.worker.mzml_paths
            if hasattr(self.worker, 'library_path'):
                self._library_path = self.worker.library_path
        else:
            self._stage_stats = {}

        # Setup view mode combo with sample names
        self.view_combo.blockSignals(True)
        self.view_combo.clear()
        self.view_combo.addItem("Aligned (all samples)")
        self._rep_id_map = {}  # combo index -> rep_id
        for idx, rep_id in enumerate(sorted(self._sample_files)):
            paths = self._sample_files[rep_id]
            sample_name = Path(paths[0]).stem if paths else f"Sample {rep_id}"
            self.view_combo.addItem(sample_name)
            self._rep_id_map[idx + 1] = rep_id
        self.view_combo.setEnabled(True)
        self.view_combo.setCurrentIndex(0)
        self.view_combo.blockSignals(False)
        self._view_mode = "aligned"

        # Plan F-followup-5 mirror: propagate the "Annotated" display
        # threshold from config so scatter and table agree about which
        # rows count as annotated.
        thr = float(getattr(
            self.config, "matchms_similarity_threshold", 0.3,
        ))
        self.scatter_plot.set_annotated_threshold(thr)
        self.feature_table.proxy.set_annotated_threshold(thr)
        mm = int(getattr(self.config, "matchms_min_matched_peaks", 3) or 0)
        self.scatter_plot.set_annotated_min_matched(mm)
        self.feature_table.proxy.set_annotated_min_matched(mm)
        self.scatter_plot.set_features(features)
        self.feature_table.set_features(features)
        self._load_eic_store()

        self.progress_panel.set_complete(len(features))
        self.act_run.setEnabled(True)
        self.act_stop.setEnabled(False)
        self.act_export.setEnabled(True)
        self.act_save_project.setEnabled(True)
        self.act_reannotate.setEnabled(bool(self._work_dir))
        self.act_clear_work.setEnabled(bool(self._work_dir))

        self.statusBar().showMessage(f"Done: {len(features)} features")
        self.progress_panel.log(f"Pipeline complete: {len(features)} features")

        # Auto-save project in data directory
        self._auto_save_project()

    def _auto_save_project(self):
        """Auto-save project file next to raw data."""
        if not self.features or not self._mzml_paths:
            return
        try:
            self._saving_project = True
            from asfam.io.project_file import save_project
            from metabo_gui.project_autosave import auto_save_path

            proj_path = auto_save_path(
                self._mzml_paths[0],
                app_prefix="ASFAM",
                extension="asfam",
            )

            save_project(
                str(proj_path), self.config, self.features,
                mzml_paths=self._mzml_paths,
                library_path=self._library_path,
                stage_stats=self._stage_stats,
                candidates_by_rep=self._candidates_by_rep,
                work_dir=self._work_dir,
            )
            self.progress_panel.log(f"Project auto-saved: {proj_path.name}")
            self._saving_project = False
            # Install feedback system bound to the auto-save path
            if self._feedback_controller is None:
                self._install_feedback(str(proj_path))
            else:
                self._feedback_controller.set_project_path(str(proj_path))
                self._feedback_controller.save_now()
        except Exception as e:
            self._saving_project = False
            logging.exception("Auto-save failed")
            self.progress_panel.log(f"Auto-save failed: {e}")

    def _on_pipeline_error(self, error_msg):
        self.progress_panel.set_error(error_msg)
        self.act_run.setEnabled(True)
        self.act_stop.setEnabled(False)
        QMessageBox.critical(self, "Pipeline Error", error_msg)

    # ------------------------------------------------------------------
    # View mode switching
    # ------------------------------------------------------------------

    def _on_view_changed(self, index):
        """Switch between aligned view and single sample view."""
        if index == 0:
            self._view_mode = "aligned"
            self.scatter_plot.set_features(self.features)
            self.feature_table.set_features(self.features)
        else:
            rep_id = self._rep_id_map.get(index)
            if rep_id is None:
                return
            self._view_mode = rep_id
            cands = self._candidates_for(rep_id)
            if cands:
                # Convert CandidateFeature to Feature-like for display
                from asfam.models import Feature
                rep_features = []
                for c in cands:
                    # No status filter: a dedup stage moving a feature out of
                    # its own candidate set ("*_excluded") never removed it.
                    f = Feature(
                        feature_id=c.feature_id,
                        precursor_mz=c.precursor_mz,
                        rt=c.rt_apex,
                        rt_left=c.rt_left,
                        rt_right=c.rt_right,
                        signal_type=c.signal_type,
                        ms2_mz=c.ms2_mz,
                        ms2_intensity=c.ms2_intensity,
                        n_fragments=c.n_fragments,
                        heights={rep_id: c.ms1_height or 0},
                        areas={rep_id: c.ms1_area or 0},
                        mean_height=c.ms1_height or 0,
                        mean_area=c.ms1_area or 0,
                        cv=0,
                        name=c.matchms_name,
                        formula=c.inferred_formula,
                        adduct=c.adduct_type,
                        sn_ratio=c.ms1_sn or 0,
                        height_ion_mz=c.ms2_rep_ion_mz,
                        mz_source=c.mz_source,
                        mz_confidence=c.mz_confidence,
                        detection_source=c.detection_source,
                        is_duplicate=c.is_duplicate,
                        duplicate_group_id=c.duplicate_group_id,
                        duplicate_type=c.duplicate_type,
                        annotation_matches=c.annotation_matches,
                        selected_annotation_idx=c.selected_annotation_idx,
                    )
                    rep_features.append(f)
                self.scatter_plot.set_features(rep_features)
                self.feature_table.set_features(rep_features)
        self.eic_plot.clear()
        self.ms2_plot.clear()

    def _candidates_for(self, rep_id: str) -> list:
        """One sample's CandidateFeatures, read from ``_work/`` on demand.

        Projects saved before the spill existed still carry their candidates
        inline; prefer those when present.
        """
        if self._candidates_by_rep and rep_id in self._candidates_by_rep:
            return self._candidates_by_rep[rep_id]
        if not self._work_dir:
            return []
        from asfam.io import spill
        try:
            return spill.read_sample_features(Path(self._work_dir) / rep_id,
                                              load_ms2=True)
        except (OSError, spill.SpillFormatError) as e:
            self.progress_panel.log(f"Could not read sample {rep_id}: {e}")
            return []

    def _load_eic_store(self) -> None:
        """Point the EIC viewer at ``alignment.eic`` and its key map, both of
        which sit beside ``_work/``.

        Absent when gap fill was disabled or could not find the raw data; the
        viewer says so rather than reaching for an mzML.
        """
        output_dir = Path(self._work_dir).parent if self._work_dir else None
        store = open_store(output_dir) if output_dir else None
        self.eic_plot.set_eic_store(store)
        self._spotmap = load_spot_map(output_dir) if output_dir else SpotMap()
        if store is None:
            self.progress_panel.log(
                "No alignment.eic found — EIC panel unavailable")
        else:
            self.progress_panel.log(
                f"Chromatogram store: {len(store)} features")

    def _store_key_for(self, feat: Feature) -> Optional[str]:
        """Where ``feat``'s chromatograms live in ``alignment.eic``.

        The store is keyed by aligned spot id. The aligned view's features carry
        exactly that; the single-sample view's are candidates read back from
        ``_work/``, whose ids belong to a namespace the store has never seen — so
        they go through the spot map. ``None`` means no spot claimed this peak.

        A project written before ``spotmap.json`` existed has no map to go
        through: fall back to the raw id, which misses and leaves the viewer's
        generic message in place, rather than crashing or plotting a stranger.
        """
        if self._view_mode == "aligned" or not self._spotmap.spot_of:
            return feat.feature_id
        return self._spotmap.spot_of.get(feat.feature_id)

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------

    def _on_feature_selected(self, source_row: int):
        feat = self.feature_table.model.get_feature(source_row)
        if feat is None:
            return

        # EIC: show only current replicate in single mode, all in aligned mode.
        # The key is translated here and passed in — the viewer must not derive
        # one from feature_id, which in single-sample mode names a candidate the
        # store knows nothing about.
        key = self._store_key_for(feat)
        self.eic_plot.set_view_mode(self._view_mode)
        self.eic_plot.show_feature(
            feat, store_key=key,
            fragment_source=(self._spotmap.representative_of.get(key)
                             if key else None),
        )

        # MS2: show with annotation candidates (combo box populated internally)
        self.ms2_plot.show_feature(feat)
        self.scatter_plot.highlight_feature(feat.feature_id)

        # Feedback: push selected feature into the notes dock
        if self._notes_dock is not None and self._feedback_controller is not None:
            from metabo_gui.feedback import feature_signature_from_components
            sig_fb = feature_signature_from_components(
                mz=feat.precursor_mz, rt=feat.rt, mode="asfam",
            )
            self._notes_dock.set_current_feature(feat.feature_id, sig_fb)

        sig = "MS1" if feat.signal_type == "ms1_detected" else "MS2"
        name_str = f" [{feat.name}]" if feat.name else ""
        formula_str = f" {feat.formula}" if feat.formula else ""

        # Duplicate group info
        dup_str = ""
        if feat.duplicate_group_id is not None:
            group = [f for f in self.features if f.duplicate_group_id == feat.duplicate_group_id]
            if len(group) > 1:
                rep = next((f for f in group if not f.is_duplicate), group[0])
                members = ", ".join(f.feature_id for f in group if f.feature_id != rep.feature_id)
                dup_str = f" | {feat.duplicate_type} group: rep={rep.feature_id}, dups=[{members}]"

        self.statusBar().showMessage(
            f"{feat.feature_id}: m/z {feat.precursor_mz:.4f}, "
            f"RT {feat.rt:.2f}, {sig}{name_str}{formula_str}{dup_str}"
        )

    def _on_scatter_clicked(self, feature_id: str):
        self.feature_table.select_feature_by_id(feature_id)

    def _on_annotation_selected(self, feature_id: str, selected_idx: int):
        """User selected a different annotation candidate for a feature."""
        for feat in self.features:
            if feat.feature_id == feature_id:
                if 0 <= selected_idx < len(feat.annotation_matches):
                    feat.selected_annotation_idx = selected_idx
                    match = feat.annotation_matches[selected_idx]
                    feat.name = match.name
                    feat.formula = match.formula
                    feat.adduct = match.adduct
                    # Refresh table
                    self.feature_table.model.layoutChanged.emit()
                break

    # ------------------------------------------------------------------
    # Feedback system (Task 10)
    # ------------------------------------------------------------------

    def _install_feedback(self, project_path: str) -> None:
        """Create or recreate FeedbackController + NotesDock + StatusDelegate."""
        from pathlib import Path
        from metabo_gui.feedback import (
            FeatureStatusDelegate, FeedbackController, FeedbackStore, NotesDock,
            load_alongside,
        )
        from metabo_core import __version__ as _software_version

        # 1. Load existing sidecar or create a fresh store
        store = load_alongside(project_path)
        if store is None:
            ctx = self._build_current_run_context_for_feedback(project_path)
            store = FeedbackStore(
                schema_version=1,
                app="asfam",
                software_version=_software_version,
                run_context=ctx,
                entries=[],
            )

        # 2. Controller (recreate if existed; old one is deleteLater'd)
        if self._feedback_controller is not None:
            self._feedback_controller.deleteLater()
        self._feedback_controller = FeedbackController(
            project_path, store, parent=self,
        )

        # 3. Dock (recreate to rebind to new controller)
        if self._notes_dock is not None:
            self.removeDockWidget(self._notes_dock)
            self._notes_dock.deleteLater()
        self._notes_dock = NotesDock(self._feedback_controller, parent=self)
        self.addDockWidget(Qt.RightDockWidgetArea, self._notes_dock)
        # Notes panel is collapsed by default; open it via the "Notes Panel"
        # toolbar toggle.
        self._notes_dock.setVisible(False)

        # 4. Status column delegate
        delegate = FeatureStatusDelegate(
            controller=self._feedback_controller,
            feature_id_provider=lambda idx: self.feature_table.model.feature_id_at_row(
                self.feature_table.proxy.mapToSource(idx).row()
            ),
            parent=self,
        )
        delegate.attach_view(self.feature_table.table)
        self.feature_table.table.setItemDelegateForColumn(
            self.feature_table.model.STATUS_COLUMN_INDEX, delegate,
        )

    def _build_current_run_context_for_feedback(self, project_path: str):
        """Build a RunContext from the current window state."""
        from dataclasses import asdict
        from metabo_gui.feedback import build_run_context
        from metabo_core import __version__ as _software_version

        # Prefer the orchestrator's run_context if the pipeline has run
        orch = None
        if self.worker is not None and self.worker.orchestrator is not None:
            orch = self.worker.orchestrator
        if orch is not None and getattr(orch, "run_context", None) is not None:
            ctx = orch.run_context
            ctx.project_file = project_path
            return ctx

        return build_run_context(
            app="asfam",
            software_version=_software_version,
            input_files=self._mzml_paths or [],
            library_path=self._library_path,
            project_file=project_path,
            export_dir=None,
            params=asdict(self.config) if hasattr(self, "config") else {},
        )

    def _toggle_notes_dock(self) -> None:
        """Toggle visibility of the NotesDock panel."""
        if self._notes_dock is None:
            return
        self._notes_dock.setVisible(not self._notes_dock.isVisible())

    # ------------------------------------------------------------------
    # Re-annotate mode
    # ------------------------------------------------------------------

    def _on_reannotate(self):
        """Re-run annotation only (stages 6.5 -> 7 -> 8)."""
        from asfam.io import spill
        if not self._work_dir or not spill.scan_checkpoints(self._work_dir):
            QMessageBox.warning(self, "No Data",
                "No intermediate results available. Run the full pipeline first "
                "or open a project whose _work directory still exists.")
            return
        output_dir = self.setup_panel.get_output_dir()
        if not output_dir:
            QMessageBox.warning(self, "No Output", "Please select an output directory.")
            return

        # run_reannotate rewrites each sample's spill in place, clearing the old
        # annotations before it starts.
        from asfam.gui.i18n import tr
        confirm = QMessageBox.question(
            self,
            tr("Confirm Re-annotation"),
            tr("Re-annotation overwrites the current annotation results, and the "
               "old results cannot be recovered even if you cancel midway.\n\n"
               "Continue?"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        self.setup_panel.apply_to_config()
        self.progress_panel.reset()
        self.act_run.setEnabled(False)
        self.act_reannotate.setEnabled(False)
        self.act_stop.setEnabled(True)

        from asfam.gui.worker import ReAnnotateWorker
        self.worker = ReAnnotateWorker(
            self.config,
            output_dir,
            self.setup_panel.get_library_path(),
            work_dir=self._work_dir,
            sample_files=self._sample_files,
        )
        self.worker.progress_update.connect(self.progress_panel.update_progress)
        self.worker.pipeline_completed.connect(self._on_pipeline_done)
        self.worker.pipeline_error.connect(self._on_pipeline_error)
        self.worker.start()
        self.progress_panel.log("Re-annotation started (stages 6.5 -> 7 -> 8)...")

    # ------------------------------------------------------------------
    # Project save/load
    # ------------------------------------------------------------------

    def _on_save_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "project.asfam", "ASFAM Project (*.asfam)")
        if not path:
            return
        from asfam.io.project_file import save_project
        save_project(
            path, self.config, self.features,
            mzml_paths=self._mzml_paths,
            library_path=self._library_path,
            stage_stats=self._stage_stats,
            candidates_by_rep=self._candidates_by_rep,
            work_dir=self._work_dir,
        )
        self.progress_panel.log(f"Project saved to {path}")
        QMessageBox.information(self, "Saved", f"Project saved to {path}")

        # Feedback sidecar: update path (Save-As case) and persist
        if self._feedback_controller is not None:
            self._feedback_controller.set_project_path(path)
            self._feedback_controller.save_now()
        else:
            # First Save: install the feedback system now that we have a path
            self._install_feedback(path)

    def _on_open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "ASFAM Project (*.asfam)")
        if not path:
            return
        from asfam.io.project_file import load_project
        try:
            proj = load_project(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load project:\n{e}")
            return

        self.config = proj["config"]
        self.features = proj["features"]
        self._mzml_paths = proj.get("mzml_paths", [])
        self._library_path = proj.get("library_path")
        self._stage_stats = proj.get("stage_stats", {})
        self._candidates_by_rep = proj.get("candidates_by_rep")  # pre-spill projects
        self._work_dir = proj.get("work_dir")
        if self._work_dir and not os.path.isdir(self._work_dir):
            self.progress_panel.log(
                f"Intermediate results directory is gone: {self._work_dir}")
            self._work_dir = None

        from asfam.io import spill
        rep_ids = sorted(self._candidates_by_rep or {})
        if not rep_ids and self._work_dir:
            rep_ids = sorted(sid for sid, _ in spill.scan_checkpoints(self._work_dir))

        # Setup view combo
        self.view_combo.blockSignals(True)
        self.view_combo.clear()
        self.view_combo.addItem("Aligned (all replicates)")
        self._rep_id_map = {}
        for idx, rep_id in enumerate(rep_ids):
            self.view_combo.addItem(f"Replicate {rep_id}")
            self._rep_id_map[idx + 1] = rep_id
        self.view_combo.setEnabled(True)
        self.view_combo.setCurrentIndex(0)
        self.view_combo.blockSignals(False)

        thr = float(getattr(
            self.config, "matchms_similarity_threshold", 0.3,
        ))
        self.scatter_plot.set_annotated_threshold(thr)
        self.feature_table.proxy.set_annotated_threshold(thr)
        mm = int(getattr(self.config, "matchms_min_matched_peaks", 3) or 0)
        self.scatter_plot.set_annotated_min_matched(mm)
        self.feature_table.proxy.set_annotated_min_matched(mm)
        self.scatter_plot.set_features(self.features)
        self.feature_table.set_features(self.features)
        self.act_export.setEnabled(True)
        self.act_save_project.setEnabled(True)
        self.act_reannotate.setEnabled(bool(self._work_dir))
        self.act_clear_work.setEnabled(bool(self._work_dir))

        # Resolve the mzML paths. The EIC viewer no longer needs them — it reads
        # alignment.eic — but re-annotation re-runs gap fill, which does.
        self._sample_files = {}
        if self._mzml_paths:
            proj_dir = os.path.dirname(path)
            resolved = []
            for mp in self._mzml_paths:
                if os.path.exists(mp):
                    resolved.append(mp)
                else:
                    # Try relative to project dir
                    alt = os.path.join(proj_dir, os.path.basename(mp))
                    if os.path.exists(alt):
                        resolved.append(alt)
            if resolved:
                from asfam.pipeline.stage0_load import group_files_by_sample
                try:
                    self._sample_files = group_files_by_sample(
                        resolved, self.setup_panel.get_sample_groups(),
                    )
                    self.progress_panel.log(
                        f"Raw data registered: {len(self._sample_files)} replicates")
                except Exception as e:
                    self.progress_panel.log(f"Warning: could not group mzML files: {e}")
            else:
                missing = [os.path.basename(p) for p in self._mzml_paths]
                self.progress_panel.log(f"Warning: mzML files not found: {', '.join(missing)}")
        self._load_eic_store()

        self.statusBar().showMessage(f"Loaded: {len(self.features)} features from {path}")
        self.progress_panel.log(f"Project loaded: {len(self.features)} features")

        # Install feedback system from sidecar (or fresh store if none found)
        self._install_feedback(path)

    def _on_export(self):
        if not self.features:
            QMessageBox.warning(
                self, "Export",
                "No results to export. Run the pipeline or open a project first.")
            return

        output_dir = self.setup_panel.get_output_dir()
        if not output_dir:
            # After opening a project the output-dir field is empty (it is not
            # stored in the .asfam file), so the export used to silently do
            # nothing. Prompt for a destination instead.
            output_dir = QFileDialog.getExistingDirectory(
                self, "Select Output Directory")
            if not output_dir:
                return  # user cancelled the dialog
            # Reflect the chosen directory back into the panel for reuse.
            self.setup_panel.out_path.setText(output_dir)

        from asfam.pipeline.stage8_export import run_stage8
        feedback_store = (
            self._feedback_controller.store
            if self._feedback_controller is not None
            else None
        )
        outputs = run_stage8(
            self.features, output_dir, self.config,
            feedback_store=feedback_store,
        )
        QMessageBox.information(
            self, "Exported",
            f"Results exported to {output_dir}\nFiles: {', '.join(outputs.keys())}")

    # ------------------------------------------------------------------
    # About / Language
    # ------------------------------------------------------------------

    def _on_about(self):
        from asfam import __version__
        show_about_dialog(
            self,
            app_name="ASFAM Processor",
            version=__version__,
            description=(
                "WT 2.0 data-processing engine — processes All-ion Stepwise "
                "Fragmentation Acquisition Mode (ASFAM) LC-QTOF mass "
                "spectrometry data."
            ),
        )

    def _on_language_changed(self, index):
        from asfam.gui.i18n import set_language
        lang = "zh" if index == 1 else "en"
        set_language(lang)
        self._apply_language()

    def _apply_language(self):
        """Apply current language to all toolbar labels."""
        from asfam.gui.i18n import tr
        self.act_run.setText(tr("Run"))
        self.act_stop.setText(tr("Stop"))
        self.act_reannotate.setText(tr("Re-annotate"))
        self.act_save_project.setText(tr("Save Project"))
        self.act_open_project.setText(tr("Open Project"))
        self.act_export.setText(tr("Export"))
        self.act_about.setText(tr("About"))
        self.statusBar().showMessage(tr("Ready"))

    # ------------------------------------------------------------------
    # Close protection
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        """Protect against closing during project save."""
        if self._saving_project:
            from PyQt5.QtWidgets import QMessageBox
            reply = QMessageBox.warning(
                self, "Saving in Progress",
                "Project is being saved. Closing now may corrupt the file.\n\n"
                "Are you sure you want to close?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
        event.accept()
