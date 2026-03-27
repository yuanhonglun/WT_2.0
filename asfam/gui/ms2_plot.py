"""MS2 spectrum viewer with mirror plot support.

Interactive: scroll zoom, direction-locked left-drag, right-drag rubber-band zoom,
double-click reset.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel
from PyQt5.QtCore import pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from asfam.models import Feature
from asfam.gui.plot_toolbar import make_plot_toolbar

THEME_BLUE = "#2D6A9F"


class MS2PlotWidget(QWidget):
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
        selector_row.addStretch()
        layout.addLayout(selector_row)
        layout.addWidget(self.canvas)

        self._current_feature: Optional[Feature] = None
        self._current_matches: list = []
        self._save_prefix: str = "MS2"

        # Interaction state
        self._default_xlim = None
        self._default_ylim = None
        self._drag_active = False
        self._drag_start_px = None
        self._drag_start_data = None
        self._drag_axis = None
        self._zoom_active = False
        self._zoom_start = None
        self._zoom_rect = None

        # Axis-area drag
        self._axis_drag_active = False
        self._axis_drag_which = None
        self._axis_drag_start_px = None

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

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
        self._default_xlim = self.ax.get_xlim()
        self._default_ylim = self.ax.get_ylim()

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
        self._label_top_peaks(mz, rel_int, y_offset=3, color="#333")
        self.ax.set_xlabel("m/z", fontsize=9)
        self.ax.set_ylabel("Relative Intensity (%)", fontsize=9)
        self.ax.set_ylim(0, 115)
        sig = "MS1" if feature.signal_type == "ms1_detected" else "MS2"
        name_str = f" - {feature.name}" if feature.name else ""
        self.ax.set_title(f"MS2: {feature.feature_id} ({len(mz)} frags, {sig}){name_str}",
                          fontsize=9, color=THEME_BLUE)

    def _draw_mirror(self, mz, rel_int, ref_peaks, score, ref_name, feature):
        markerline, stemlines, baseline = self.ax.stem(
            mz, rel_int, linefmt="-", markerfmt=" ", basefmt=" ")
        stemlines.set_color(THEME_BLUE); stemlines.set_linewidth(1.5)
        self._label_top_peaks(mz, rel_int, y_offset=3, color=THEME_BLUE)
        ref_mz = np.array([p[0] for p in ref_peaks])
        ref_int = np.array([p[1] for p in ref_peaks])
        max_ref = np.max(ref_int) if len(ref_int) > 0 else 1
        ref_rel = -ref_int / max(max_ref, 1) * 100
        markerline2, stemlines2, baseline2 = self.ax.stem(
            ref_mz, ref_rel, linefmt="-", markerfmt=" ", basefmt=" ")
        stemlines2.set_color("#E05050"); stemlines2.set_linewidth(1.5)
        self._label_top_peaks(ref_mz, ref_rel, y_offset=-8, color="#E05050", top_n=8)
        self.ax.axhline(0, color="gray", linewidth=0.8)
        self.ax.set_xlabel("m/z", fontsize=9)
        self.ax.set_ylabel("Relative Intensity (%)", fontsize=9)
        self.ax.set_ylim(-120, 120)
        name_part = f" vs {ref_name}" if ref_name else " vs Library"
        self.ax.set_title(f"Mirror: {feature.feature_id}{name_part}  Score: {score:.3f}",
                          fontsize=9, color=THEME_BLUE, fontweight="bold")
        self.ax.text(0.02, 0.95, "Query", transform=self.ax.transAxes,
                     fontsize=8, color=THEME_BLUE, va="top")
        self.ax.text(0.02, 0.05, "Reference", transform=self.ax.transAxes,
                     fontsize=8, color="#E05050", va="bottom")

    def _label_top_peaks(self, mz, rel_int, y_offset=3, color="#333", top_n=8):
        abs_int = np.abs(rel_int)
        n_labels = min(top_n, len(mz))
        top_indices = np.argsort(abs_int)[-n_labels:]
        for idx in top_indices:
            if abs_int[idx] > 5:
                self.ax.annotate(f"{mz[idx]:.3f}", (mz[idx], rel_int[idx]),
                                 textcoords="offset points", xytext=(0, y_offset),
                                 ha="center", fontsize=6.5, color=color)

    def clear(self):
        self.ax.clear()
        self._default_xlim = self._default_ylim = None
        self.ax.set_title("Select a feature", color=THEME_BLUE)
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Interactive
    # ------------------------------------------------------------------

    def _on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        factor = 0.8 if event.button == "up" else 1.25
        xlim = self.ax.get_xlim(); ylim = self.ax.get_ylim()
        x, y = event.xdata, event.ydata
        self.ax.set_xlim(x - (x - xlim[0]) * factor, x + (xlim[1] - x) * factor)
        self.ax.set_ylim(y - (y - ylim[0]) * factor, y + (ylim[1] - y) * factor)
        self.canvas.draw_idle()

    def _on_press(self, event):
        if self.toolbar.mode:
            return
        # Axis-area click detection
        if event.button == 1 and not event.dblclick and event.inaxes is None:
            which = self._detect_axis_area(event)
            if which:
                self._axis_drag_active = True
                self._axis_drag_which = which
                self._axis_drag_start_px = (event.x, event.y)
                return
        if event.inaxes != self.ax or event.xdata is None:
            return
        if event.button == 1:
            if event.dblclick:
                if self._default_xlim and self._default_ylim:
                    self.ax.set_xlim(self._default_xlim)
                    self.ax.set_ylim(self._default_ylim)
                    self.canvas.draw_idle()
                return
            self._drag_active = True
            self._drag_start_px = (event.x, event.y)
            self._drag_start_data = (event.xdata, event.ydata)
            self._drag_axis = None
        elif event.button == 3:
            self._zoom_active = True
            self._zoom_start = (event.xdata, event.ydata)
            self._zoom_rect = Rectangle(
                (event.xdata, event.ydata), 0, 0,
                linewidth=1, edgecolor="red", facecolor="red", alpha=0.15)
            self.ax.add_patch(self._zoom_rect)

    def _on_release(self, event):
        if event.button == 1 and self._axis_drag_active:
            self._axis_drag_active = False
            self._axis_drag_start_px = None
            return
        if event.button == 1 and self._drag_active:
            self._drag_active = False
            self._drag_start_px = None
            self._drag_axis = None
        elif event.button == 3 and self._zoom_active:
            self._zoom_active = False
            if self._zoom_rect:
                self._zoom_rect.remove()
                self._zoom_rect = None
            if self._zoom_start and event.xdata is not None and event.ydata is not None:
                x0, y0 = self._zoom_start
                x1, y1 = event.xdata, event.ydata
                if abs(x1 - x0) > 1e-6:
                    self.ax.set_xlim(min(x0, x1), max(x0, x1))
                if abs(y1 - y0) > 1e-6:
                    self.ax.set_ylim(min(y0, y1), max(y0, y1))
            self._zoom_start = None
            self.canvas.draw_idle()

    def _on_motion(self, event):
        # Axis-area drag: zoom single axis
        if self._axis_drag_active and self._axis_drag_start_px:
            dx_px = event.x - self._axis_drag_start_px[0]
            dy_px = event.y - self._axis_drag_start_px[1]
            self._axis_drag_start_px = (event.x, event.y)
            factor_per_px = 0.005
            if self._axis_drag_which == "x":
                factor = 1 - dx_px * factor_per_px
                if factor <= 0:
                    return
                xl = self.ax.get_xlim()
                cx = (xl[0] + xl[1]) / 2
                hw = (xl[1] - xl[0]) / 2 * factor
                self.ax.set_xlim(cx - hw, cx + hw)
            else:
                factor = 1 + dy_px * factor_per_px
                if factor <= 0:
                    return
                yl = self.ax.get_ylim()
                cy = (yl[0] + yl[1]) / 2
                hh = (yl[1] - yl[0]) / 2 * factor
                self.ax.set_ylim(cy - hh, cy + hh)
            self.canvas.draw_idle()
            return

        if self._zoom_active and self._zoom_rect and self._zoom_start:
            if event.xdata is not None and event.ydata is not None and event.inaxes == self.ax:
                x0, y0 = self._zoom_start
                self._zoom_rect.set_xy((min(x0, event.xdata), min(y0, event.ydata)))
                self._zoom_rect.set_width(abs(event.xdata - x0))
                self._zoom_rect.set_height(abs(event.ydata - y0))
                self.canvas.draw_idle()
            return
        if not self._drag_active or self._drag_start_px is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if self._drag_axis is None:
            dpx = abs(event.x - self._drag_start_px[0])
            dpy = abs(event.y - self._drag_start_px[1])
            if dpx > 5 or dpy > 5:
                self._drag_axis = "x" if dpx >= dpy else "y"
            else:
                return
        dx = self._drag_start_data[0] - event.xdata
        dy = self._drag_start_data[1] - event.ydata
        xlim = self.ax.get_xlim(); ylim = self.ax.get_ylim()
        if self._drag_axis == "x":
            self.ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
        else:
            self.ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
        self.canvas.draw_idle()

    def _detect_axis_area(self, event):
        """Detect if click is on x-axis or y-axis label area. Returns 'x', 'y', or None."""
        bbox = self.ax.get_window_extent(renderer=self.fig.canvas.get_renderer())
        if bbox.x0 <= event.x <= bbox.x1 and event.y < bbox.y0:
            return "x"
        if event.x < bbox.x0 and bbox.y0 <= event.y <= bbox.y1:
            return "y"
        return None
