"""RT x m/z feature overview: heatmap with drag pan, scroll zoom, click-only selection."""
from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel
from PyQt5.QtCore import pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from asfam.models import Feature
from asfam.gui.plot_toolbar import make_plot_toolbar

THEME_BLUE = "#2D6A9F"


class ScatterPlotWidget(QWidget):
    pointClicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(8, 5), dpi=100, facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = make_plot_toolbar(self.canvas, self, white_icons=True, icon_size=28)

        # Toolbar row with filter combo
        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        toolbar_row.setSpacing(4)
        toolbar_row.addWidget(self.toolbar)
        lbl = QLabel("Show:")
        lbl.setStyleSheet("font-size: 11px; font-weight: bold;")
        toolbar_row.addWidget(lbl)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "MS1 only", "MS2 only"])
        self.filter_combo.setFixedWidth(100)
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        toolbar_row.addWidget(self.filter_combo)
        layout.addLayout(toolbar_row)
        layout.addWidget(self.canvas)

        self.setMinimumHeight(200)
        self._filter_mode = "all"  # "all", "high", "low"

        self._features: list[Feature] = []
        self._highlight_lines = None
        self._default_xlim = None
        self._default_ylim = None

        self._drag_active = False
        self._drag_start = None
        self._drag_moved = False

        # Right-drag zoom
        self._zoom_active = False
        self._zoom_start = None
        self._zoom_rect = None

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def set_features(self, features: list[Feature]):
        self._features = features
        self._draw()

    def _on_filter_changed(self, index):
        modes = ["all", "high", "low"]
        self._filter_mode = modes[index] if index < len(modes) else "all"
        self._draw()

    def highlight_feature(self, feature_id: str):
        if self._highlight_lines is not None:
            for line in self._highlight_lines:
                try:
                    line.remove()
                except Exception:
                    pass
            self._highlight_lines = None

        for f in self._features:
            if f.feature_id == feature_id:
                ax = self.fig.axes[0] if self.fig.axes else None
                if ax is None:
                    return
                self._highlight_lines = ax.plot(
                    f.rt, f.precursor_mz, "o",
                    markersize=16, markerfacecolor="none",
                    markeredgecolor="red", markeredgewidth=2.5, zorder=10,
                )
                self.canvas.draw_idle()
                return

    def _draw(self):
        # Fully clear figure and recreate axes (prevents colorbar width accumulation)
        self.fig.clear()
        ax = self.fig.add_axes([0.08, 0.12, 0.72, 0.82])  # fixed position
        cax = self.fig.add_axes([0.82, 0.12, 0.03, 0.82])  # fixed colorbar axes

        if not self._features:
            ax.set_xlabel("RT (min)")
            ax.set_ylabel("m/z")
            ax.set_title("No features loaded")
            cax.set_visible(False)
            self.canvas.draw()
            return

        # Apply filter mode
        all_high = [f for f in self._features if f.signal_type == "ms1_detected"]
        all_low = [f for f in self._features if f.signal_type == "ms2_only"]
        if self._filter_mode == "high":
            high = all_high
            low = []
        elif self._filter_mode == "low":
            high = []
            low = all_low
        else:
            high = all_high
            low = all_low

        visible = high + low
        if not visible:
            ax.set_xlabel("RT (min)")
            ax.set_ylabel("m/z")
            ax.set_title("No features to display (filter active)")
            cax.set_visible(False)
            self.canvas.draw()
            return

        # For low features, use MS2 total intensity instead of mean_height
        def get_intensity(f):
            if f.mean_height > 0:
                return f.mean_height
            # Use sum of MS2 intensities
            if f.ms2_intensity is not None and len(f.ms2_intensity) > 0:
                return float(np.sum(f.ms2_intensity))
            return 1.0

        all_intensities = [max(get_intensity(f), 1) for f in visible]
        vmin = max(min(all_intensities), 10)
        vmax = max(max(all_intensities), vmin * 2)
        norm = LogNorm(vmin=vmin, vmax=vmax)

        if high:
            rt_h = [f.rt for f in high]
            mz_h = [f.precursor_mz for f in high]
            int_h = [max(get_intensity(f), 1) for f in high]
            sizes_h = np.clip(np.log10(np.array(int_h)) * 10, 8, 120)
            sc = ax.scatter(rt_h, mz_h, s=sizes_h, c=int_h, cmap="YlOrRd",
                            norm=norm, alpha=0.85, edgecolors="gray", linewidths=0.3,
                            label=f"MS1 ({len(high)})", zorder=5)

        if low:
            rt_l = [f.rt for f in low]
            mz_l = [f.precursor_mz for f in low]
            int_l = [max(get_intensity(f), 1) for f in low]
            sizes_l = np.clip(np.log10(np.array(int_l) + 1) * 8, 6, 60)
            sc_low = ax.scatter(rt_l, mz_l, s=sizes_l, c=int_l, marker="D",
                                cmap="YlOrRd", norm=norm, alpha=0.6,
                                edgecolors=THEME_BLUE, linewidths=0.5,
                                label=f"MS2 ({len(low)})", zorder=4)

        # Colorbar in fixed axes
        import matplotlib.cm as cm
        sm = cm.ScalarMappable(cmap="YlOrRd", norm=norm)
        sm.set_array([])
        self.fig.colorbar(sm, cax=cax, label="Intensity")

        ax.set_xlabel("RT (min)", fontsize=10)
        ax.set_ylabel("m/z", fontsize=10)
        ax.set_title(
            f"Feature Overview ({len(high)} MS1 + {len(low)} MS2 = {len(visible)}"
            f" / {len(self._features)} total)",
            fontsize=11, color=THEME_BLUE,
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

        self._default_xlim = ax.get_xlim()
        self._default_ylim = ax.get_ylim()
        self._highlight_lines = None
        self.canvas.draw()

    def _get_ax(self):
        return self.fig.axes[0] if self.fig.axes else None

    # Scroll zoom
    def _on_scroll(self, event):
        ax = self._get_ax()
        if ax is None or event.inaxes != ax:
            return
        factor = 0.8 if event.button == "up" else 1.25
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x, y = event.xdata, event.ydata
        ax.set_xlim(x - (x - xlim[0]) * factor, x + (xlim[1] - x) * factor)
        ax.set_ylim(y - (y - ylim[0]) * factor, y + (ylim[1] - y) * factor)
        self.canvas.draw_idle()

    # Left click: select or drag. Double click: reset view.
    def _on_press(self, event):
        ax = self._get_ax()
        if ax is None or event.inaxes != ax:
            return
        if self.toolbar.mode:
            return

        if event.button == 1:
            if event.dblclick:
                if self._default_xlim and self._default_ylim:
                    ax.set_xlim(self._default_xlim)
                    ax.set_ylim(self._default_ylim)
                    self.canvas.draw_idle()
                return
            self._drag_active = True
            self._drag_start = (event.xdata, event.ydata)
            self._drag_moved = False

        elif event.button == 3:
            self._zoom_active = True
            self._zoom_start = (event.xdata, event.ydata)
            self._zoom_rect = Rectangle(
                (event.xdata, event.ydata), 0, 0,
                linewidth=1, edgecolor="red", facecolor="red", alpha=0.15)
            ax.add_patch(self._zoom_rect)

    def _on_release(self, event):
        ax = self._get_ax()
        if event.button == 1:
            if self._drag_active and not self._drag_moved and ax and event.inaxes == ax:
                self._select_nearest(event.xdata, event.ydata)
            self._drag_active = False
            self._drag_start = None

        elif event.button == 3 and self._zoom_active:
            self._zoom_active = False
            if self._zoom_rect:
                self._zoom_rect.remove()
                self._zoom_rect = None
            if self._zoom_start and ax and event.xdata is not None and event.ydata is not None:
                x0, y0 = self._zoom_start
                x1, y1 = event.xdata, event.ydata
                if abs(x1 - x0) > 1e-6:
                    ax.set_xlim(min(x0, x1), max(x0, x1))
                if abs(y1 - y0) > 1e-6:
                    ax.set_ylim(min(y0, y1), max(y0, y1))
            self._zoom_start = None
            self.canvas.draw_idle()

    def _on_motion(self, event):
        ax = self._get_ax()

        # Right-drag zoom rectangle
        if self._zoom_active and self._zoom_rect and self._zoom_start:
            if ax and event.inaxes == ax and event.xdata is not None:
                x0, y0 = self._zoom_start
                self._zoom_rect.set_xy((min(x0, event.xdata), min(y0, event.ydata)))
                self._zoom_rect.set_width(abs(event.xdata - x0))
                self._zoom_rect.set_height(abs(event.ydata - y0))
                self.canvas.draw_idle()
            return

        # Left-drag pan
        if not self._drag_active or ax is None or event.inaxes != ax or self._drag_start is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        dx = self._drag_start[0] - event.xdata
        dy = self._drag_start[1] - event.ydata
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        if abs(dx) > x_range * 0.005 or abs(dy) > y_range * 0.005:
            self._drag_moved = True
        if self._drag_moved:
            ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
            ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
            self.canvas.draw_idle()

    def _select_nearest(self, x, y):
        ax = self._get_ax()
        if not self._features or x is None or y is None or ax is None:
            return
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = max(xlim[1] - xlim[0], 1e-6)
        y_range = max(ylim[1] - ylim[0], 1e-6)
        best = None
        best_dist = float("inf")
        for f in self._features:
            d = ((f.rt - x) / x_range) ** 2 + ((f.precursor_mz - y) / y_range) ** 2
            if d < best_dist:
                best_dist = d
                best = f
        if best and best_dist < 0.003:
            self.pointClicked.emit(best.feature_id)
