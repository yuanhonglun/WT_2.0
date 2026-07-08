"""EIC chromatogram viewer: MS1 EIC (top) + MS2 product ion EICs (bottom).

Mouse interactions are provided by the shared ``metabo_gui.canvas.PanZoomMixin``
(scroll zoom, direction-locked left-drag pan, right-drag rubber-band zoom,
double-click reset, axis-area single-axis zoom).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from metabo_gui.canvas import PanZoomMixin
from metabo_gui.plot_toolbar import make_plot_toolbar
from metabo_gui.theme import THEME_BLUE

from asfam.models import Feature, RawSegmentData
from asfam.core.eic import extract_ms1_eic
from asfam.core.smoothing import smooth_eic

COLORS = ["#2D6A9F", "#21A67A", "#8B5CF6", "#E05050", "#F59E0B"]


def _autoscale_y_to_window(ax, x_lo: float, x_hi: float) -> None:
    """Set ``ax`` ylim so the visible peak fills the panel.

    Walks ``ax.get_lines()`` and keeps only y values whose x falls in
    [x_lo, x_hi]; sets ``ax.set_ylim(0, max * 1.2)`` (Plan F-followup-5
    #3 raised the multiplier from 1.1 to 1.2 so the peak top isn't
    pinned to the panel edge — same constant as the GC-MS EIC viewer).
    """
    peak = 0.0
    for line in ax.get_lines():
        try:
            x = np.asarray(line.get_xdata(), dtype=float)
            y = np.asarray(line.get_ydata(), dtype=float)
        except (TypeError, ValueError):
            continue
        if x.size == 0:
            continue
        mask = (x >= x_lo) & (x <= x_hi)
        if np.any(mask):
            local = float(y[mask].max())
            if local > peak:
                peak = local
    if peak > 0.0:
        ax.set_ylim(0.0, peak * 1.2)


class EICPlotWidget(QWidget, PanZoomMixin):
    """Two-panel EIC viewer with interactive controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(5, 5), dpi=100, facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = make_plot_toolbar(self.canvas, self, white_icons=True, icon_size=28)

        # Aliases so PanZoomMixin can reach the matplotlib objects.
        self._fig = self.fig
        self._canvas = self.canvas
        self._toolbar = self.toolbar

        # Toolbar row with smooth toggle
        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        toolbar_row.setSpacing(2)
        toolbar_row.addWidget(self.toolbar)
        # Plan F-followup-5 mirror: drop the custom blue background
        # styleSheet (text rendered too dim against the theme blue) so
        # the button takes the global Qt default — matches the GC-MS
        # EIC viewer's Smoothed/Raw button.
        self._btn_smooth = QPushButton("Smoothed")
        self._btn_smooth.setCheckable(True)
        self._btn_smooth.setChecked(True)
        self._btn_smooth.toggled.connect(self._on_smooth_toggled)
        toolbar_row.addWidget(self._btn_smooth)
        layout.addLayout(toolbar_row)
        layout.addWidget(self.canvas)

        self._raw_data: Optional[dict] = None
        self._view_mode: str = "aligned"
        self._show_smooth: bool = True
        self._current_feature: Optional[Feature] = None
        self._axes: list = []
        self._save_prefix: str = "EIC"

        self.init_pan_zoom()

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

        # Plan F-followup-5 mirror: orange band between rt_left/rt_right
        # plus a dotted red apex line — same visual as the GC-MS EIC
        # viewer so users get a consistent peak-boundary highlight.
        ax_ms1.axvspan(feature.rt_left, feature.rt_right, alpha=0.12, color="orange")
        ax_ms1.axvline(feature.rt, color="red", linestyle=":", alpha=0.4)
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
                ax_ms2.axvspan(
                    feature.rt_left, feature.rt_right, alpha=0.12, color="orange",
                )
                ax_ms2.axvline(feature.rt, color="red", linestyle=":", alpha=0.4)
                ax_ms2.legend(fontsize=6, loc="upper right", ncol=2,
                              title="Product m/z", title_fontsize=7)
            else:
                ax_ms2.text(0.5, 0.5, "Raw data not available",
                            transform=ax_ms2.transAxes, ha="center", va="center", color="gray")

        ax_ms2.set_title(f"MS2 Product Ion EICs (top {min(n_ions, max_ions)}){sm_tag}",
                         fontsize=9, color=THEME_BLUE)
        ax_ms2.set_xlabel("RT (min)", fontsize=8)
        ax_ms2.set_ylabel("Intensity", fontsize=8)

        # Plan F-followup-7: peak boundary × 6 centered on apex (same
        # constants as the GC-MS EIC viewer). ×1.5 (followup-5) felt
        # too cramped — the peak needs surrounding baseline context to
        # visually read as a chromatographic peak. Tiny floor of
        # 0.02 min protects against degenerate rt_left == rt_right.
        peak_width = max(feature.rt_right - feature.rt_left, 0.02)
        half_width = peak_width * 0.5 * 6.0
        x_lo, x_hi = feature.rt - half_width, feature.rt + half_width
        # Plan F-followup-8: Y axis sized to the strongest response
        # inside the peak boundary (rt_left..rt_right) × 1.2 — not the
        # full ×6 visible window. Sizing to the peak only lets the user
        # see the feature's shape; neighbors overflowing the panel top
        # is intentional.
        for ax in self._axes:
            ax.set_xlim(x_lo, x_hi)
            _autoscale_y_to_window(ax, feature.rt_left, feature.rt_right)
        self.fig.tight_layout()
        self.canvas.draw()
        self.set_pan_zoom_axes(self._axes)
        self.capture_default_lims()

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

