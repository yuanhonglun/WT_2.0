"""MS2 spectrum viewer with mirror plot support.

Mouse interactions are provided by ``metabo_gui.canvas.PanZoomMixin``.
``after_view_changed`` is overridden so peak labels are recomputed for
the new viewport on every pan / zoom.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtWidgets import (
    QAction, QComboBox, QFileDialog, QHBoxLayout, QLabel,
    QMenu, QMessageBox, QToolButton, QVBoxLayout, QWidget,
)
from PyQt5.QtCore import pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from metabo_gui.canvas import PanZoomMixin
from metabo_gui.plot_toolbar import make_plot_toolbar
from metabo_gui.spectrum_display import (
    normalize_for_display,
    variant_best_matching_query,
)
from metabo_gui.spectrum_export import write_spectrum
from metabo_gui.theme import PLOT_REF_RED, THEME_BLUE

from asfam.models import Feature


class MS2PlotWidget(QWidget, PanZoomMixin):
    """MS2 stick/bar spectrum viewer with mirror plot for library comparison."""

    annotationSelected = pyqtSignal(str, int)  # (feature_id, selected_index)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(5, 3), dpi=100, facecolor="white")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = make_plot_toolbar(self.canvas, self, white_icons=True, icon_size=28)

        # Aliases so PanZoomMixin can reach the matplotlib objects.
        self._fig = self.fig
        self._canvas = self.canvas
        self._toolbar = self.toolbar

        # Annotation candidate selector
        selector_row = QHBoxLayout()
        selector_row.setContentsMargins(0, 0, 0, 0)
        selector_row.setSpacing(4)
        selector_row.addWidget(self.toolbar)
        lbl = QLabel("Match:")
        lbl.setStyleSheet("font-size: 10px; font-weight: bold;")
        selector_row.addWidget(lbl)
        self.annotation_combo = QComboBox()
        self.annotation_combo.setMinimumWidth(200)
        self.annotation_combo.setStyleSheet("font-size: 10px;")
        self.annotation_combo.currentIndexChanged.connect(self._on_annotation_changed)
        selector_row.addWidget(self.annotation_combo)
        # Export-spectrum menu (Plan F item 14).
        self._export_btn = QToolButton()
        self._export_btn.setText("Export…")
        self._export_btn.setToolTip(
            "Export this feature's measured or reference spectrum to .msp / .mgf"
        )
        self._export_btn.setPopupMode(QToolButton.InstantPopup)
        self._export_menu = QMenu(self._export_btn)
        for label in (
            "Measured → MSP", "Measured → MGF",
            "Reference → MSP", "Reference → MGF",
        ):
            act = QAction(label, self._export_menu)
            act.triggered.connect(
                lambda _checked=False, lbl=label: self._on_export_spectrum(lbl)
            )
            self._export_menu.addAction(act)
        self._export_btn.setMenu(self._export_menu)
        self._export_btn.setStyleSheet("font-size: 10px;")
        selector_row.addWidget(self._export_btn)
        selector_row.addStretch()
        layout.addLayout(selector_row)
        layout.addWidget(self.canvas)

        self._current_feature: Optional[Feature] = None
        self._current_matches: list = []
        self._save_prefix: str = "MS2"

        # Dynamic label data
        self._query_mz = None
        self._query_rel_int = None
        self._ref_mz = None
        self._ref_rel_int = None
        self._annotations: list = []

        self.init_pan_zoom()

    def show_feature(self, feature: Feature, ref_peaks=None, ref_score=0.0, ref_name=""):
        """Draw MS2 spectrum with annotation candidates in combo box."""
        self._current_feature = feature
        self._current_matches = getattr(feature, 'annotation_matches', []) or []
        self._save_prefix = f"MS2_{feature.feature_id}_mz{feature.precursor_mz:.4f}"
        self.toolbar.set_save_prefix(self._save_prefix)

        # Populate combo box with annotation candidates
        self.annotation_combo.blockSignals(True)
        self.annotation_combo.clear()
        if self._current_matches:
            for m in self._current_matches:
                label = f"#{m.rank}: {m.name or 'Unknown'} (score={m.score:.3f})"
                self.annotation_combo.addItem(label)
            selected_idx = getattr(feature, 'selected_annotation_idx', 0)
            self.annotation_combo.setCurrentIndex(min(selected_idx, len(self._current_matches) - 1))
        else:
            self.annotation_combo.addItem("No library matches")
        self.annotation_combo.blockSignals(False)

        self._draw_current()

    def _draw_current(self):
        """Redraw MS2 with the currently selected annotation."""
        feat = self._current_feature
        if feat is None:
            return
        self.ax.clear()
        self._annotations = []  # ax.clear() already removed them from canvas
        mz = feat.ms2_mz
        intensity = feat.ms2_intensity

        if len(mz) == 0:
            self.ax.set_title("No MS2 spectrum")
            self._default_xlim = self._default_ylim = None
            self.canvas.draw()
            return

        max_int = np.max(intensity) if len(intensity) > 0 else 1
        rel_int = intensity / max(max_int, 1) * 100

        # Get ref data from selected annotation
        ref_peaks = None
        ref_score = 0.0
        ref_name = ""
        idx = self.annotation_combo.currentIndex()
        # P7: in the default view (combo not manually switched), when the
        # library holds several collision-energy variants of the SAME compound,
        # display the variant whose spectrum best matches the query so the
        # mirror aligns with MS-DIAL. This only changes which reference is
        # DISPLAYED — the selected / counted annotation stays
        # annotation_matches[0] (top total_score); nothing here mutates the
        # combo selection or the feature's results.
        if idx <= 0 and self._current_matches:
            top_name = self._current_matches[0].name
            query_peaks = list(zip(mz.tolist(), intensity.tolist()))
            idx = variant_best_matching_query(
                query_peaks, self._current_matches, top_name)
        if 0 <= idx < len(self._current_matches):
            match = self._current_matches[idx]
            ref_peaks = match.ref_peaks
            ref_score = match.score
            ref_name = match.name

        if ref_peaks and len(ref_peaks) > 0:
            self._draw_mirror(mz, rel_int, ref_peaks, ref_score, ref_name, feat)
        else:
            self._draw_single(mz, rel_int, feat)

        self.fig.tight_layout()
        self.canvas.draw()
        self.set_pan_zoom_axes([self.ax])
        self.capture_default_lims()

    def _on_annotation_changed(self, index):
        """User switched annotation candidate."""
        if self._current_feature is not None and index >= 0:
            self._draw_current()
            self.annotationSelected.emit(self._current_feature.feature_id, index)

    def _draw_single(self, mz, rel_int, feature):
        markerline, stemlines, baseline = self.ax.stem(
            mz, rel_int, linefmt="-", markerfmt=" ", basefmt="-")
        stemlines.set_color(THEME_BLUE); stemlines.set_linewidth(1.5)
        baseline.set_color("gray"); baseline.set_linewidth(0.5)
        self._query_mz = mz
        self._query_rel_int = rel_int
        self._ref_mz = None
        self._ref_rel_int = None
        self._update_labels()
        self.ax.set_xlabel("m/z", fontsize=9)
        self.ax.set_ylabel("Relative Intensity (%)", fontsize=9)
        self.ax.set_ylim(0, 115)
        # Plain plot — render the y label literally; clear any abs-value
        # formatter that a previous mirror draw might have installed.
        self.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _pos: f"{y:.0f}")
        )
        sig = "MS1" if feature.signal_type == "ms1_detected" else "MS2"
        name_str = f" - {feature.name}" if feature.name else ""
        self.ax.set_title(f"MS2: {feature.feature_id} ({len(mz)} frags, {sig}){name_str}",
                          fontsize=9, color=THEME_BLUE)

    def _draw_mirror(self, mz, rel_int, ref_peaks, score, ref_name, feature):
        markerline, stemlines, baseline = self.ax.stem(
            mz, rel_int, linefmt="-", markerfmt=" ", basefmt=" ")
        stemlines.set_color(THEME_BLUE); stemlines.set_linewidth(1.5)
        ref_mz = np.array([p[0] for p in ref_peaks])
        ref_int = np.array([p[1] for p in ref_peaks])
        max_ref = np.max(ref_int) if len(ref_int) > 0 else 1
        ref_rel = -ref_int / max(max_ref, 1) * 100
        markerline2, stemlines2, baseline2 = self.ax.stem(
            ref_mz, ref_rel, linefmt="-", markerfmt=" ", basefmt=" ")
        stemlines2.set_color(PLOT_REF_RED); stemlines2.set_linewidth(1.5)
        self._query_mz = mz
        self._query_rel_int = rel_int
        self._ref_mz = ref_mz
        self._ref_rel_int = ref_rel
        self._update_labels()
        self.ax.axhline(0, color="gray", linewidth=0.8)
        self.ax.set_xlabel("m/z", fontsize=9)
        self.ax.set_ylabel("Relative Intensity (%)", fontsize=9)
        self.ax.set_ylim(-120, 120)
        # Plan F-followup item 4: the reference half lives at negative y
        # for layout reasons, but the values aren't actually negative.
        # Format y-tick labels as absolute values so the user reads
        # positive intensities on both halves.
        self.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _pos: f"{abs(y):.0f}")
        )
        name_part = f" vs {ref_name}" if ref_name else " vs Library"
        self.ax.set_title(f"Mirror: {feature.feature_id}{name_part}  Score: {score:.3f}",
                          fontsize=9, color=THEME_BLUE, fontweight="bold")
        self.ax.text(0.02, 0.95, "Query", transform=self.ax.transAxes,
                     fontsize=8, color=THEME_BLUE, va="top")
        self.ax.text(0.02, 0.05, "Reference", transform=self.ax.transAxes,
                     fontsize=8, color=PLOT_REF_RED, va="bottom")

    def _label_top_peaks(self, mz, rel_int, y_offset=3, color="#333", top_n=8):
        """Legacy static labeling (kept for compatibility)."""
        abs_int = np.abs(rel_int)
        n_labels = min(top_n, len(mz))
        top_indices = np.argsort(abs_int)[-n_labels:]
        for idx in top_indices:
            if abs_int[idx] > 5:
                self.ax.annotate(f"{mz[idx]:.3f}", (mz[idx], rel_int[idx]),
                                 textcoords="offset points", xytext=(0, y_offset),
                                 ha="center", fontsize=6.5, color=color)

    def _update_labels(self):
        """Recalculate and draw peak labels for the current viewport."""
        for ann in self._annotations:
            try:
                ann.remove()
            except (ValueError, AttributeError):
                pass
        self._annotations = []

        if self._query_mz is None:
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Label query peaks
        q_color = "#333" if self._ref_mz is None else THEME_BLUE
        self._label_visible_peaks(
            self._query_mz, self._query_rel_int,
            xlim, ylim, y_offset=3, color=q_color, max_labels=15,
        )

        # Label reference peaks (mirror mode)
        if self._ref_mz is not None and self._ref_rel_int is not None:
            self._label_visible_peaks(
                self._ref_mz, self._ref_rel_int,
                xlim, ylim, y_offset=-8, color=PLOT_REF_RED, max_labels=15,
            )

    def _label_visible_peaks(self, mz, rel_int, xlim, ylim, y_offset, color,
                             max_labels=15):
        """Label peaks visible in the current viewport with overlap avoidance."""
        if mz is None or len(mz) == 0:
            return

        abs_int = np.abs(rel_int)

        # Filter to visible range
        y_lo, y_hi = min(ylim), max(ylim)
        visible_mask = (
            (mz >= xlim[0]) & (mz <= xlim[1]) &
            (rel_int >= y_lo) & (rel_int <= y_hi)
        )
        visible_idx = np.where(visible_mask)[0]
        if len(visible_idx) == 0:
            return

        # Sort by absolute intensity descending
        sorted_idx = visible_idx[np.argsort(-abs_int[visible_idx])]

        # Minimum m/z spacing to avoid overlap (3% of visible range)
        mz_range = xlim[1] - xlim[0]
        min_spacing = mz_range * 0.03 if mz_range > 0 else 0.1

        placed_mz = []
        n_placed = 0

        for idx in sorted_idx:
            if n_placed >= max_labels:
                break
            too_close = any(abs(mz[idx] - pmz) < min_spacing for pmz in placed_mz)
            if too_close:
                continue

            ann = self.ax.annotate(
                f"{mz[idx]:.3f}", (mz[idx], rel_int[idx]),
                textcoords="offset points", xytext=(0, y_offset),
                ha="center", fontsize=6.5, color=color,
            )
            self._annotations.append(ann)
            placed_mz.append(mz[idx])
            n_placed += 1

    def clear(self):
        self.ax.clear()
        self._annotations = []
        self.ax.set_title("Select a feature", color=THEME_BLUE)
        self.canvas.draw()

    # ------------------------------------------------------------------
    # PanZoomMixin hook: relabel visible peaks after each pan/zoom.
    # ------------------------------------------------------------------

    def after_view_changed(self) -> None:
        self._update_labels()

    # ------------------------------------------------------------------
    # Spectrum export (Plan F item 14): write current measured / reference
    # peaks to MSP or MGF for the selected feature.
    # ------------------------------------------------------------------

    def _on_export_spectrum(self, label: str) -> None:
        if self._current_feature is None:
            QMessageBox.information(
                self, "Export spectrum", "Select a feature first.",
            )
            return

        is_reference = label.startswith("Reference")
        fmt = "msp" if "MSP" in label else "mgf"
        feat = self._current_feature

        if is_reference:
            idx = self.annotation_combo.currentIndex()
            if not (0 <= idx < len(self._current_matches)):
                QMessageBox.warning(
                    self, "Export spectrum",
                    "No reference spectrum available for the selected feature.",
                )
                return
            match = self._current_matches[idx]
            ref_peaks = list(match.ref_peaks or [])
            if not ref_peaks:
                QMessageBox.warning(
                    self, "Export spectrum",
                    "Selected library hit has no peak list.",
                )
                return
            # P7: export the reference spectrum base-peak-normalized to 100 so
            # it matches the on-screen mirror and MS-DIAL's Relative display.
            # Operates on a copy only — match.ref_peaks (fed to the scorer)
            # is never touched.
            peaks = normalize_for_display(
                [(float(p[0]), float(p[1])) for p in ref_peaks], target=100.0
            )
            meta = {
                "Name": getattr(match, "name", "") or "Unknown",
                "Formula": getattr(match, "formula", "") or "",
                "PrecursorMZ": float(feat.precursor_mz),
                "Comment": f"Reference for {feat.feature_id}",
            }
            default_stem = f"{feat.feature_id}_ref"
        else:
            mz_arr = list(feat.ms2_mz or [])
            int_arr = list(feat.ms2_intensity or [])
            if not mz_arr:
                QMessageBox.warning(
                    self, "Export spectrum",
                    "Feature has no measured spectrum.",
                )
                return
            peaks = [(float(m), float(i)) for m, i in zip(mz_arr, int_arr)]
            meta = {
                "Name": getattr(feat, "name", "") or feat.feature_id,
                "Formula": getattr(feat, "formula", "") or "",
                "PrecursorMZ": float(feat.precursor_mz),
                "RetentionTime": float(getattr(feat, "rt", 0.0)),
                "Comment": f"Measured for {feat.feature_id}",
            }
            default_stem = f"{feat.feature_id}_measured"

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Spectrum",
            f"{default_stem}.{fmt}",
            f"{fmt.upper()} (*.{fmt})",
        )
        if not path:
            return
        try:
            written = write_spectrum(path, peaks, meta, fmt=fmt)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        QMessageBox.information(
            self, "Export Spectrum", f"Spectrum exported to:\n{written}",
        )
