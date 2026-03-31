"""EIC chromatogram viewer: MS1 EIC (top) + MS2 product ion EICs (bottom).

Interactive: scroll zoom, direction-locked left-drag (pixel-based), right-drag
rubber-band zoom, double-click reset, smooth/raw toggle.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from asfam.models import Feature, RawSegmentData
from asfam.core.eic import extract_ms1_eic
from asfam.core.smoothing import smooth_eic
from asfam.gui.plot_toolbar import make_plot_toolbar

THEME_BLUE = "#2D6A9F"
COLORS = ["#2D6A9F", "#21A67A", "#8B5CF6", "#E05050", "#F59E0B"]


class EICPlotWidget(QWidget):
    """Two-panel EIC viewer with interactive controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(5, 5), dpi=100, facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = make_plot_toolbar(self.canvas, self, white_icons=True, icon_size=28)

        # Toolbar row with smooth toggle
        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        toolbar_row.setSpacing(2)
        toolbar_row.addWidget(self.toolbar)
        self._btn_smooth = QPushButton("Smoothed")
        self._btn_smooth.setCheckable(True)
        self._btn_smooth.setChecked(True)
        self._btn_smooth.setFixedWidth(80)
        self._btn_smooth.setStyleSheet(
            "QPushButton { font-size: 10px; padding: 2px 6px; }"
            "QPushButton:checked { background: #2D6A9F; color: white; }"
        )
        self._btn_smooth.toggled.connect(self._on_smooth_toggled)
        toolbar_row.addWidget(self._btn_smooth)
        layout.addLayout(toolbar_row)
        layout.addWidget(self.canvas)

        self._raw_data: Optional[dict] = None
        self._view_mode: str = "aligned"
        self._show_smooth: bool = True
        self._current_feature: Optional[Feature] = None
        self._axes: list = []
        self._default_lims: list[tuple] = []
        self._save_prefix: str = "EIC"

        # Left-drag state
        self._drag_active = False
        self._drag_start_px = None
        self._drag_start_data = None
        self._drag_ax = None
        self._drag_axis = None  # "x" or "y"

        # Right-drag zoom state
        self._zoom_active = False
        self._zoom_start = None
        self._zoom_rect = None
        self._zoom_ax = None

        # Axis-area drag state (click on axis labels to zoom one axis)
        self._axis_drag_active = False
        self._axis_drag_which = None  # "x" or "y"
        self._axis_drag_ax = None
        self._axis_drag_start_px = None

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_raw_data(self, raw_data: dict):
        self._raw_data = raw_data

    def set_view_mode(self, mode: str):
        self._view_mode = mode

    def show_feature(self, feature: Feature):
        self._current_feature = feature
        self._save_prefix = f"EIC_{feature.feature_id}_mz{feature.precursor_mz:.4f}"
        self.toolbar.set_save_prefix(self._save_prefix)
        self._redraw()

    def clear(self):
        self.fig.clear()
        self._axes = []
        self._default_lims = []
        self._current_feature = None
        ax = self.fig.add_subplot(111)
        ax.set_title("Select a feature", color=THEME_BLUE)
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _redraw(self):
        feature = self._current_feature
        if feature is None:
            return
        self.fig.clear()
        self._axes = []
        self._default_lims = []

        if self._raw_data is None:
            ax = self.fig.add_subplot(111)
            ax.set_title("No raw data loaded", color=THEME_BLUE)
            self.canvas.draw()
            return

        ax_ms1 = self.fig.add_subplot(211)
        ax_ms2 = self.fig.add_subplot(212)
        self._axes = [ax_ms1, ax_ms2]

        mz = feature.precursor_mz
        prec_nominal = int(round(mz))
        use_smooth = self._show_smooth

        reps = (self._raw_data if self._view_mode == "aligned"
                else {self._view_mode: self._raw_data.get(self._view_mode, [])})

        # --- Top: MS1 EIC ---
        has_ms1 = False
        for i, (rep_id, segments) in enumerate(sorted(reps.items())):
            color = COLORS[i % len(COLORS)]
            for seg in segments:
                if not (seg.segment_low <= mz <= seg.segment_high + 1):
                    continue
                rt_arr, int_arr = extract_ms1_eic(seg, mz, 0.5)
                if len(int_arr) > 0 and np.max(int_arr) > 0:
                    y = smooth_eic(int_arr, "savgol", 7, 3) if use_smooth else int_arr
                    label = f"MS1 {rep_id}" if len(reps) > 1 else "MS1"
                    ax_ms1.plot(rt_arr, y, color=color, alpha=0.8, linewidth=1.2, label=label)
                    has_ms1 = True

        if not has_ms1:
            ax_ms1.text(0.5, 0.5, "No MS1 signal", transform=ax_ms1.transAxes,
                        ha="center", va="center", color="gray", fontsize=10)

        ax_ms1.axvline(feature.rt, color="red", linestyle="-", alpha=0.3, linewidth=1)
        ax_ms1.axvline(feature.rt_left, color="gray", linestyle=":", alpha=0.5)
        ax_ms1.axvline(feature.rt_right, color="gray", linestyle=":", alpha=0.5)
        sig = "MS1" if feature.signal_type == "ms1_detected" else "MS2"
        sm_tag = "" if use_smooth else " [Raw]"
        ax_ms1.set_title(f"MS1 EIC: m/z {mz:.4f} ({sig}){sm_tag}", fontsize=9, color=THEME_BLUE)
        ax_ms1.set_ylabel("Intensity", fontsize=8)
        if has_ms1:
            ax_ms1.legend(fontsize=7, loc="upper right")

        # --- Bottom: MS2 product ion EICs ---
        ms2_mzs = feature.ms2_mz
        ms2_ints = feature.ms2_intensity
        n_ions = len(ms2_mzs)
        max_ions = 8

        if n_ions == 0:
            ax_ms2.text(0.5, 0.5, "No MS2 fragments", transform=ax_ms2.transAxes,
                        ha="center", va="center", color="gray", fontsize=10)
        else:
            top_idx = np.argsort(ms2_ints)[-max_ions:] if n_ions > max_ions else np.arange(n_ions)
            rep_key = list(reps.keys())[0] if reps else None
            seg_to_use = None
            if rep_key and reps.get(rep_key):
                for seg in reps[rep_key]:
                    if prec_nominal in seg.precursor_list:
                        seg_to_use = seg
                        break
            if seg_to_use is not None:
                ion_colors = ["#2D6A9F", "#E05050", "#21A67A", "#F59E0B",
                              "#8B5CF6", "#FF6B6B", "#4ECDC4", "#95A5A6"]
                for j, idx in enumerate(top_idx):
                    ion_mz = ms2_mzs[idx]
                    eic_int = np.zeros(seg_to_use.n_cycles)
                    for ci, cycle in enumerate(seg_to_use.cycles):
                        if prec_nominal in cycle.ms2_scans:
                            prod_mz, prod_int = cycle.ms2_scans[prec_nominal]
                            if len(prod_mz) > 0:
                                mask = np.abs(prod_mz - ion_mz) <= 0.02
                                if np.any(mask):
                                    eic_int[ci] = float(np.max(prod_int[mask]))
                    if np.max(eic_int) > 0:
                        y = smooth_eic(eic_int, "savgol", 7, 3) if use_smooth else eic_int
                        ax_ms2.plot(seg_to_use.rt_array, y, color=ion_colors[j % len(ion_colors)],
                                    alpha=0.7, linewidth=0.9, label=f"{ion_mz:.3f}")
                ax_ms2.axvline(feature.rt, color="red", linestyle="-", alpha=0.3)
                ax_ms2.axvline(feature.rt_left, color="gray", linestyle=":", alpha=0.5)
                ax_ms2.axvline(feature.rt_right, color="gray", linestyle=":", alpha=0.5)
                ax_ms2.legend(fontsize=6, loc="upper right", ncol=2,
                              title="Product m/z", title_fontsize=7)
            else:
                ax_ms2.text(0.5, 0.5, "Raw data not available",
                            transform=ax_ms2.transAxes, ha="center", va="center", color="gray")

        ax_ms2.set_title(f"MS2 Product Ion EICs (top {min(n_ions, max_ions)}){sm_tag}",
                         fontsize=9, color=THEME_BLUE)
        ax_ms2.set_xlabel("RT (min)", fontsize=8)
        ax_ms2.set_ylabel("Intensity", fontsize=8)

        margin = max((feature.rt_right - feature.rt_left) * 2, 0.5)
        for ax in self._axes:
            ax.set_xlim(feature.rt - margin, feature.rt + margin)
        self.fig.tight_layout()
        self.canvas.draw()
        self._default_lims = [(ax.get_xlim(), ax.get_ylim()) for ax in self._axes]

    def _on_smooth_toggled(self, checked):
        self._show_smooth = checked
        self._btn_smooth.setText("Smoothed" if checked else "Raw")
        if self._current_feature:
            # Save current zoom state before redraw
            saved = [(ax.get_xlim(), ax.get_ylim()) for ax in self._axes] if self._axes else None
            self._redraw()
            # Restore zoom state after redraw
            if saved and len(saved) == len(self._axes):
                for i, ax in enumerate(self._axes):
                    ax.set_xlim(saved[i][0])
                    ax.set_ylim(saved[i][1])
                self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Scroll zoom
    # ------------------------------------------------------------------

    def _find_ax(self, event):
        for ax in self._axes:
            if event.inaxes == ax:
                return ax
        return None

    def _on_scroll(self, event):
        ax = self._find_ax(event)
        if ax is None or event.xdata is None:
            return
        factor = 0.8 if event.button == "up" else 1.25
        xlim = ax.get_xlim()
        x = event.xdata
        new_xlim = (x - (x - xlim[0]) * factor, x + (xlim[1] - x) * factor)
        for a in self._axes:
            a.set_xlim(new_xlim)
        ylim = ax.get_ylim()
        y = event.ydata
        ax.set_ylim(y - (y - ylim[0]) * factor, y + (ylim[1] - y) * factor)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Left-drag (direction-locked pan) + Right-drag (rubber-band zoom)
    # ------------------------------------------------------------------

    def _on_press(self, event):
        if self.toolbar.mode:
            return

        # Check if click is on axis area (outside plot but near axis)
        if event.button == 1 and not event.dblclick and event.inaxes is None:
            ax_hit, which = self._detect_axis_area(event)
            if ax_hit is not None:
                self._axis_drag_active = True
                self._axis_drag_which = which
                self._axis_drag_ax = ax_hit
                self._axis_drag_start_px = (event.x, event.y)
                return

        ax = self._find_ax(event)
        if ax is None or event.xdata is None or event.ydata is None:
            return

        if event.button == 1:
            if event.dblclick:
                for i, a in enumerate(self._axes):
                    if i < len(self._default_lims):
                        a.set_xlim(self._default_lims[i][0])
                        a.set_ylim(self._default_lims[i][1])
                self.canvas.draw_idle()
                return
            self._drag_active = True
            self._drag_start_px = (event.x, event.y)
            self._drag_start_data = (event.xdata, event.ydata)
            self._drag_ax = ax
            self._drag_axis = None

        elif event.button == 3:
            self._zoom_active = True
            self._zoom_start = (event.xdata, event.ydata)
            self._zoom_ax = ax
            self._zoom_rect = Rectangle(
                (event.xdata, event.ydata), 0, 0,
                linewidth=1, edgecolor="red", facecolor="red", alpha=0.15,
            )
            ax.add_patch(self._zoom_rect)

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
            if (self._zoom_start and event.xdata is not None and event.ydata is not None
                    and self._zoom_ax):
                x0, y0 = self._zoom_start
                x1, y1 = event.xdata, event.ydata
                if abs(x1 - x0) > 1e-6:
                    new_xlim = (min(x0, x1), max(x0, x1))
                    for a in self._axes:
                        a.set_xlim(new_xlim)
                if abs(y1 - y0) > 1e-6:
                    self._zoom_ax.set_ylim(min(y0, y1), max(y0, y1))
            self._zoom_start = None
            self._zoom_ax = None
            self.canvas.draw_idle()

    def _on_motion(self, event):
        # Axis-area drag: zoom single axis
        if self._axis_drag_active and self._axis_drag_start_px and self._axis_drag_ax:
            dx_px = event.x - self._axis_drag_start_px[0]
            dy_px = event.y - self._axis_drag_start_px[1]
            self._axis_drag_start_px = (event.x, event.y)
            factor_per_px = 0.005  # 5px of mouse movement = 0.5% zoom

            if self._axis_drag_which == "x":
                factor = 1 - dx_px * factor_per_px  # drag right = zoom in (shrink range)
                if factor <= 0:
                    return
                for a in self._axes:
                    xl = a.get_xlim()
                    cx = (xl[0] + xl[1]) / 2
                    hw = (xl[1] - xl[0]) / 2 * factor
                    a.set_xlim(cx - hw, cx + hw)
            else:  # "y"
                factor = 1 + dy_px * factor_per_px  # drag up = zoom in
                if factor <= 0:
                    return
                ax = self._axis_drag_ax
                yl = ax.get_ylim()
                cy = (yl[0] + yl[1]) / 2
                hh = (yl[1] - yl[0]) / 2 * factor
                ax.set_ylim(cy - hh, cy + hh)
            self.canvas.draw_idle()
            return

        # Right-drag zoom rectangle
        if self._zoom_active and self._zoom_rect and self._zoom_start:
            if event.xdata is not None and event.ydata is not None and event.inaxes == self._zoom_ax:
                x0, y0 = self._zoom_start
                self._zoom_rect.set_xy((min(x0, event.xdata), min(y0, event.ydata)))
                self._zoom_rect.set_width(abs(event.xdata - x0))
                self._zoom_rect.set_height(abs(event.ydata - y0))
                self.canvas.draw_idle()
            return

        # Left-drag pan
        if not self._drag_active or self._drag_start_px is None or self._drag_ax is None:
            return
        if event.inaxes != self._drag_ax or event.xdata is None or event.ydata is None:
            return

        # Determine direction using pixel coordinates (uniform scale)
        if self._drag_axis is None:
            dpx = abs(event.x - self._drag_start_px[0])
            dpy = abs(event.y - self._drag_start_px[1])
            if dpx > 5 or dpy > 5:
                self._drag_axis = "x" if dpx >= dpy else "y"
            else:
                return

        dx = self._drag_start_data[0] - event.xdata
        dy = self._drag_start_data[1] - event.ydata

        if self._drag_axis == "x":
            for a in self._axes:
                xl = a.get_xlim()
                a.set_xlim(xl[0] + dx, xl[1] + dx)
        else:  # "y"
            yl = self._drag_ax.get_ylim()
            self._drag_ax.set_ylim(yl[0] + dy, yl[1] + dy)
        self.canvas.draw_idle()

    def _detect_axis_area(self, event):
        """Detect if a click landed on the axis label area (outside plot).

        Returns (axes, "x"/"y") or (None, None).
        """
        for ax in self._axes:
            bbox = ax.get_window_extent(renderer=self.fig.canvas.get_renderer())
            # X-axis area: below the plot bbox, within x range
            if bbox.x0 <= event.x <= bbox.x1 and event.y < bbox.y0:
                return ax, "x"
            # Y-axis area: left of the plot bbox, within y range
            if event.x < bbox.x0 and bbox.y0 <= event.y <= bbox.y1:
                return ax, "y"
        return None, None
