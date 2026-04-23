"""Setup panel: file list, library path, output dir, parameter configuration."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QListWidget, QListWidgetItem, QPushButton, QLineEdit,
    QLabel, QTabWidget, QFormLayout, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QFileDialog, QMessageBox,
    QDialog, QDialogButtonBox, QTreeWidget, QTreeWidgetItem,
)
from PyQt5.QtCore import Qt

from asfam.config import ProcessingConfig


class SetupPanel(QWidget):
    """Left panel: file list, library, output dir, and parameter tabs."""

    def __init__(self, config: ProcessingConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setMinimumWidth(200)
        self.setMaximumWidth(450)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # --- File List ---
        file_group = QGroupBox("mzML Files")
        file_layout = QVBoxLayout(file_group)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        file_layout.addWidget(self.file_list)
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("Add")
        self.btn_remove = QPushButton("Remove")
        self.btn_group = QPushButton("Edit Samples")
        self.btn_add.clicked.connect(self._add_files)
        self.btn_remove.clicked.connect(self._remove_files)
        self.btn_group.clicked.connect(self._edit_groups)
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addWidget(self.btn_group)
        file_layout.addLayout(btn_layout)

        # Group info label
        self.group_label = QLabel("Samples: auto-detect from filenames")
        self.group_label.setStyleSheet("color: gray; font-size: 10px;")
        file_layout.addWidget(self.group_label)
        layout.addWidget(file_group)

        # Sample grouping: dict mapping sample_name -> [file_paths]
        self._custom_groups: Optional[dict] = None

        # --- Library ---
        lib_group = QGroupBox("Spectral Library (MSP/MGF)")
        lib_layout = QHBoxLayout(lib_group)
        self.lib_path = QLineEdit()
        self.lib_path.setPlaceholderText("Optional: browse MSP/MGF...")
        lib_browse = QPushButton("...")
        lib_browse.setFixedWidth(30)
        lib_browse.clicked.connect(self._browse_library)
        lib_layout.addWidget(self.lib_path)
        lib_layout.addWidget(lib_browse)
        layout.addWidget(lib_group)

        # --- Output Dir ---
        out_group = QGroupBox("Output Directory")
        out_layout = QHBoxLayout(out_group)
        self.out_path = QLineEdit()
        self.out_path.setPlaceholderText("Select output folder...")
        out_browse = QPushButton("...")
        out_browse.setFixedWidth(30)
        out_browse.clicked.connect(self._browse_output)
        out_layout.addWidget(self.out_path)
        out_layout.addWidget(out_browse)
        layout.addWidget(out_group)

        # --- Parameters ---
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout(param_group)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self._build_general_tab()
        self._build_detection_tab()
        self._build_dedup_tab()
        self._build_alignment_tab()
        self._build_export_tab()
        param_layout.addWidget(self.tabs)

        # Save/Load config
        cfg_layout = QHBoxLayout()
        btn_save = QPushButton("Save Config")
        btn_load = QPushButton("Load Config")
        btn_save.clicked.connect(self._save_config)
        btn_load.clicked.connect(self._load_config)
        cfg_layout.addWidget(btn_save)
        cfg_layout.addWidget(btn_load)
        param_layout.addLayout(cfg_layout)

        layout.addWidget(param_group)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_mzml_paths(self) -> list[str]:
        paths = []
        for i in range(self.file_list.count()):
            paths.append(self.file_list.item(i).data(Qt.UserRole))
        return paths

    def get_library_path(self) -> str:
        return self.lib_path.text().strip() or None

    def get_output_dir(self) -> str:
        return self.out_path.text().strip()

    def get_sample_groups(self) -> Optional[dict]:
        """Return user-defined sample groups, or None for auto-detect.

        Returns dict: sample_name -> [file_paths]
        """
        if self._custom_groups:
            return self._custom_groups
        # Auto-detect from filenames: files sharing the same replicate ID
        paths = self.get_mzml_paths()
        if not paths:
            return None
        return _auto_group_files(paths)

    def apply_to_config(self):
        """Write widget values back to self.config."""
        self.config.ionization_mode = self.combo_mode.currentText()
        self.config.n_workers = self.spin_workers.value()
        # Detection
        self.config.peak_height_threshold = self.spin_height.value()
        self.config.peak_sn_threshold = self.spin_sn.value()
        self.config.peak_width_min = self.spin_width.value()
        self.config.eic_mz_tolerance = self.spin_eic_tol.value()
        self.config.rt_cluster_tolerance = self.spin_rt_cluster.value()
        self.config.min_fragments_per_feature = self.spin_min_frags.value()
        self.config.peak_gaussian_threshold = self.spin_gauss_thr.value()
        self.config.ms1_min_height = self.spin_ms1_height.value()
        self.config.min_fragments_for_inference = self.spin_infer_frags.value()
        self.config.enable_library_mz_inference = self.chk_enable_library_infer.isChecked()
        self.config.matchms_similarity_threshold = self.spin_matchms_thr.value()
        self.config.matchms_use_rt = self.chk_use_rt.isChecked()
        # Dedup
        self.config.isotope_modified_cos_threshold = self.spin_iso_cos.value()
        self.config.adduct_eic_pearson_threshold = self.spin_add_corr.value()
        self.config.isf_eic_pearson_threshold = self.spin_isf_corr.value()
        # Alignment
        self.config.alignment_rt_tolerance = self.spin_align_rt.value()
        self.config.alignment_mz_tolerance = self.spin_align_mz.value()
        self.config.gap_fill_enabled = self.chk_gapfill.isChecked()
        # Export
        self.config.export_mgf = self.chk_mgf.isChecked()
        self.config.export_msp = self.chk_msp.isChecked()
        self.config.export_report = self.chk_report.isChecked()

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_general_tab(self):
        tab = QWidget()
        form = QFormLayout(tab)
        form.setLabelAlignment(Qt.AlignRight)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["positive", "negative"])
        form.addRow("Ion Mode:", self.combo_mode)

        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(1, 32)
        self.spin_workers.setValue(self.config.n_workers)
        form.addRow("Workers:", self.spin_workers)

        self.tabs.addTab(tab, "General")

    def _build_detection_tab(self):
        tab = QWidget()
        form = QFormLayout(tab)
        form.setLabelAlignment(Qt.AlignRight)

        self.spin_height = _dbl_spin(self.config.peak_height_threshold, 0, 1e7, 10)
        form.addRow("Peak Height Min:", self.spin_height)

        self.spin_sn = _dbl_spin(self.config.peak_sn_threshold, 0, 100, 0.5)
        form.addRow("S/N Threshold:", self.spin_sn)

        self.spin_width = QSpinBox()
        self.spin_width.setRange(1, 50)
        self.spin_width.setValue(self.config.peak_width_min)
        form.addRow("Peak Width Min:", self.spin_width)

        self.spin_eic_tol = _dbl_spin(self.config.eic_mz_tolerance, 0.001, 1.0, 0.005, 3)
        form.addRow("EIC m/z Tol (Da):", self.spin_eic_tol)

        self.spin_rt_cluster = _dbl_spin(self.config.rt_cluster_tolerance, 0.001, 1.0, 0.005, 3)
        form.addRow("RT Cluster Tol (min):", self.spin_rt_cluster)

        self.spin_min_frags = QSpinBox()
        self.spin_min_frags.setRange(1, 20)
        self.spin_min_frags.setValue(self.config.min_fragments_per_feature)
        form.addRow("Min Fragments:", self.spin_min_frags)

        self.spin_gauss_thr = _dbl_spin(self.config.peak_gaussian_threshold, 0, 1, 0.05, 2)
        form.addRow("Gaussian Sim Thr:", self.spin_gauss_thr)

        self.spin_ms1_height = _dbl_spin(self.config.ms1_min_height, 0, 1e7, 50)
        form.addRow("MS1 Height Min:", self.spin_ms1_height)

        self.spin_infer_frags = QSpinBox()
        self.spin_infer_frags.setRange(1, 20)
        self.spin_infer_frags.setValue(self.config.min_fragments_for_inference)
        form.addRow("Infer Min Frags:", self.spin_infer_frags)

        self.chk_enable_library_infer = QCheckBox(
            "Enable library m/z inference (slow; few features benefit)"
        )
        self.chk_enable_library_infer.setChecked(
            getattr(self.config, "enable_library_mz_inference", False)
        )
        form.addRow("Library m/z Inference:", self.chk_enable_library_infer)

        self.spin_matchms_thr = _dbl_spin(self.config.matchms_similarity_threshold, 0, 1, 0.05, 2)
        form.addRow("Library Match Thr:", self.spin_matchms_thr)

        self.chk_use_rt = QCheckBox("Use RT in library matching score")
        self.chk_use_rt.setChecked(self.config.matchms_use_rt)
        form.addRow("RT Scoring:", self.chk_use_rt)

        self.tabs.addTab(tab, "Detection")

    def _build_dedup_tab(self):
        tab = QWidget()
        form = QFormLayout(tab)
        form.setLabelAlignment(Qt.AlignRight)

        self.spin_iso_cos = _dbl_spin(self.config.isotope_modified_cos_threshold, 0, 1, 0.05, 2)
        form.addRow("Isotope Cos Thr:", self.spin_iso_cos)

        self.spin_add_corr = _dbl_spin(self.config.adduct_eic_pearson_threshold, 0, 1, 0.05, 2)
        form.addRow("Adduct EIC Corr:", self.spin_add_corr)

        self.spin_isf_corr = _dbl_spin(self.config.isf_eic_pearson_threshold, 0, 1, 0.05, 2)
        form.addRow("ISF EIC Corr:", self.spin_isf_corr)

        self.tabs.addTab(tab, "Dedup")

    def _build_alignment_tab(self):
        tab = QWidget()
        form = QFormLayout(tab)
        form.setLabelAlignment(Qt.AlignRight)

        self.spin_align_rt = _dbl_spin(self.config.alignment_rt_tolerance, 0.01, 5.0, 0.05, 2)
        form.addRow("Align RT Tol (min):", self.spin_align_rt)

        self.spin_align_mz = _dbl_spin(self.config.alignment_mz_tolerance, 0.001, 1.0, 0.005, 3)
        form.addRow("Align m/z Tol (Da):", self.spin_align_mz)

        self.chk_gapfill = QCheckBox("Enable Gap Filling")
        self.chk_gapfill.setChecked(self.config.gap_fill_enabled)
        form.addRow(self.chk_gapfill)

        self.tabs.addTab(tab, "Alignment")

    def _build_export_tab(self):
        tab = QWidget()
        form = QFormLayout(tab)

        self.chk_mgf = QCheckBox("Export MGF")
        self.chk_mgf.setChecked(self.config.export_mgf)
        form.addRow(self.chk_mgf)

        self.chk_msp = QCheckBox("Export MSP")
        self.chk_msp.setChecked(self.config.export_msp)
        form.addRow(self.chk_msp)

        self.chk_report = QCheckBox("Export Report")
        self.chk_report.setChecked(self.config.export_report)
        form.addRow(self.chk_report)

        self.tabs.addTab(tab, "Export")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select mzML Files", "",
            "mzML Files (*.mzML *.mzml);;All Files (*)",
        )
        for p in paths:
            name = Path(p).name
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, str(p))
            self.file_list.addItem(item)
        if paths:
            self._update_sample_info()

    def _remove_files(self):
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))
        self._update_sample_info()

    def _update_sample_info(self):
        """Update the sample info label after files change."""
        paths = self.get_mzml_paths()
        if not paths:
            self.group_label.setText("No files loaded")
            return
        if self._custom_groups:
            n_samples = len(self._custom_groups)
            n_files = sum(len(v) for v in self._custom_groups.values())
            self.group_label.setText(f"Samples: {n_samples} (user-defined, {n_files} files)")
        else:
            groups = _auto_group_files(paths)
            names = ", ".join(sorted(groups.keys())[:3])
            more = f"..." if len(groups) > 3 else ""
            self.group_label.setText(
                f"Auto-detected: {len(groups)} sample(s) [{names}{more}], "
                f"{len(paths)} files"
            )

    def _browse_library(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Spectral Library", "",
            "Library Files (*.msp *.mgf);;All Files (*)",
        )
        if path:
            self.lib_path.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.out_path.setText(path)

    def _save_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "asfam_config.json",
            "JSON Files (*.json)",
        )
        if path:
            self.apply_to_config()
            self.config.save(path)
            QMessageBox.information(self, "Saved", f"Configuration saved to {path}")

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "",
            "JSON Files (*.json)",
        )
        if path:
            self.config = ProcessingConfig.load(path)
            self._refresh_widgets()
            QMessageBox.information(self, "Loaded", f"Configuration loaded from {path}")

    def _refresh_widgets(self):
        """Sync widget values from config."""
        self.combo_mode.setCurrentText(self.config.ionization_mode)
        self.spin_workers.setValue(self.config.n_workers)
        self.spin_height.setValue(self.config.peak_height_threshold)
        self.spin_sn.setValue(self.config.peak_sn_threshold)
        self.spin_width.setValue(self.config.peak_width_min)
        self.spin_eic_tol.setValue(self.config.eic_mz_tolerance)
        self.spin_rt_cluster.setValue(self.config.rt_cluster_tolerance)
        self.spin_min_frags.setValue(self.config.min_fragments_per_feature)
        self.spin_ms1_height.setValue(self.config.ms1_min_height)
        self.spin_infer_frags.setValue(self.config.min_fragments_for_inference)
        self.chk_enable_library_infer.setChecked(
            getattr(self.config, "enable_library_mz_inference", False)
        )
        self.spin_matchms_thr.setValue(self.config.matchms_similarity_threshold)
        self.spin_iso_cos.setValue(self.config.isotope_modified_cos_threshold)
        self.spin_add_corr.setValue(self.config.adduct_eic_pearson_threshold)
        self.spin_isf_corr.setValue(self.config.isf_eic_pearson_threshold)
        self.spin_align_rt.setValue(self.config.alignment_rt_tolerance)
        self.spin_align_mz.setValue(self.config.alignment_mz_tolerance)
        self.chk_gapfill.setChecked(self.config.gap_fill_enabled)
        self.chk_mgf.setChecked(self.config.export_mgf)
        self.chk_msp.setChecked(self.config.export_msp)
        self.chk_report.setChecked(self.config.export_report)

    def _edit_groups(self):
        """Open dialog to manually assign files to sample groups."""
        paths = self.get_mzml_paths()
        if not paths:
            QMessageBox.warning(self, "No Files", "Add mzML files first.")
            return

        current = self._custom_groups or _auto_group_files(paths)
        dlg = GroupEditDialog(current, paths, self)
        if dlg.exec_() == QDialog.Accepted:
            self._custom_groups = dlg.get_groups()
            self._update_sample_info()


class GroupEditDialog(QDialog):
    """Dialog for manually assigning files to sample groups."""

    def __init__(self, groups: dict, all_paths: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Samples")
        self.resize(600, 400)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Drag files between samples. Each sample = one biological sample.\n"
            "Right-click a sample header to rename. Use buttons to add/merge samples."
        ))

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Sample / File"])
        self.tree.setDragDropMode(QTreeWidget.InternalMove)
        layout.addWidget(self.tree)

        # Populate tree
        self._groups = dict(groups)
        for sample_name, file_paths in sorted(self._groups.items()):
            parent_item = QTreeWidgetItem(self.tree, [sample_name])
            parent_item.setFlags(parent_item.flags() | Qt.ItemIsEditable)
            for fp in file_paths:
                child = QTreeWidgetItem(parent_item, [Path(fp).name])
                child.setData(0, Qt.UserRole, fp)
            parent_item.setExpanded(True)

        # Check for unassigned files
        assigned = set()
        for fps in self._groups.values():
            assigned.update(fps)
        unassigned = [p for p in all_paths if p not in assigned]
        if unassigned:
            parent_item = QTreeWidgetItem(self.tree, ["Unassigned"])
            for fp in unassigned:
                child = QTreeWidgetItem(parent_item, [Path(fp).name])
                child.setData(0, Qt.UserRole, fp)
            parent_item.setExpanded(True)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_new = QPushButton("New Sample")
        btn_new.clicked.connect(self._new_group)
        btn_layout.addWidget(btn_new)
        btn_del = QPushButton("Delete Sample")
        btn_del.clicked.connect(self._delete_group)
        btn_layout.addWidget(btn_del)
        btn_layout.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        btn_layout.addWidget(buttons)
        layout.addLayout(btn_layout)

    def _new_group(self):
        item = QTreeWidgetItem(self.tree, ["New Sample"])
        item.setFlags(item.flags() | Qt.ItemIsEditable)

    def _delete_group(self):
        """Delete selected group, moving its files to Unassigned."""
        current = self.tree.currentItem()
        if current is None:
            return
        # If a child is selected, operate on its parent group
        if current.parent() is not None:
            current = current.parent()

        # Find or create Unassigned group
        unassigned = None
        for i in range(self.tree.topLevelItemCount()):
            if self.tree.topLevelItem(i).text(0) == "Unassigned":
                unassigned = self.tree.topLevelItem(i)
                break
        if unassigned is None:
            unassigned = QTreeWidgetItem(self.tree, ["Unassigned"])

        # Move children to Unassigned
        while current.childCount() > 0:
            child = current.takeChild(0)
            unassigned.addChild(child)
        unassigned.setExpanded(True)

        # Remove the empty group
        idx = self.tree.indexOfTopLevelItem(current)
        if idx >= 0:
            self.tree.takeTopLevelItem(idx)

    def get_groups(self) -> dict:
        """Read groups from tree widget."""
        groups = {}
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            name = parent.text(0)
            files = []
            for j in range(parent.childCount()):
                child = parent.child(j)
                fp = child.data(0, Qt.UserRole)
                if fp:
                    files.append(fp)
            if files:
                groups[name] = files
        return groups


def _auto_group_files(paths: list[str]) -> dict:
    """Auto-group files by sample+replicate using flexible filename parsing.

    Uses parse_filename to extract sample name and replicate ID.
    Files with the same (sample, rep) belong to the same sample.
    """
    from asfam.io.mzml_reader import parse_filename

    groups: dict[str, list[str]] = {}

    for p in paths:
        try:
            info = parse_filename(p)
            # Group key: sample name + replicate ID
            key = f"{info['sample']}_rep{info['rep']}"
        except ValueError:
            # Fallback: use filename as-is
            key = Path(p).stem

        if key not in groups:
            groups[key] = []
        groups[key].append(p)

    # Use the grouping key as sample name (e.g., "CK1_rep1", "MIX_rep2")
    return groups


def _dbl_spin(value, lo, hi, step, decimals=1):
    """Helper to create a QDoubleSpinBox."""
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    spin.setValue(value)
    return spin
