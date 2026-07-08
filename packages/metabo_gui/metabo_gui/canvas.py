"""PanZoomMixin — reusable matplotlib mouse interactions.

Mouse map (matches the legacy ASFAM widgets that this code replaces):
  - scroll wheel: zoom around cursor (X is shared across all linked axes,
    Y is per-axis)
  - left-drag inside an axis: direction-locked pan (decided after >5px of
    pixel motion so a near-vertical wiggle doesn't jitter X)
  - right-drag: rubber-band zoom (X shared, Y per-axis)
  - double-click left button: reset to the captured default xlim/ylim
  - drag in axis-label area (outside plot): single-axis zoom (drag-right
    to zoom in on X, drag-up to zoom in on Y)

Subclass contract:
  - assign ``self._fig`` (matplotlib Figure) and ``self._canvas``
    (FigureCanvasQTAgg) before calling ``init_pan_zoom``
  - call ``set_pan_zoom_axes(list_of_axes)`` after each redraw
  - call ``capture_default_lims()`` at the end of redraw to record what
    "double-click home" means for that draw
  - optionally override ``after_view_changed()`` to relabel peaks etc.
  - optionally assign ``self._toolbar`` — when its ``mode`` is non-empty
    (zoom/pan tool active in matplotlib's own toolbar) the mixin yields
    so the two systems don't fight
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

from matplotlib.patches import Rectangle


class PanZoomMixin:
    """Mouse-driven pan/zoom for matplotlib canvases inside Qt widgets."""

    # ------------------------------------------------------------------
    # Setup / state mgmt — call these from the subclass
    # ------------------------------------------------------------------

    def init_pan_zoom(self) -> None:
        """Wire mpl events. Must be called once after ``self._canvas`` is set."""
        self._pan_axes: list = []
        self._default_lims: list[tuple[tuple[float, float], tuple[float, float]]] = []

        # Drag (pan) state
        self._drag_active = False
        self._drag_start_px: Optional[tuple[float, float]] = None
        self._drag_start_data: Optional[tuple[float, float]] = None
        self._drag_ax = None
        self._drag_axis: Optional[str] = None

        # Right-drag (rubber-band zoom) state
        self._zoom_active = False
        self._zoom_start: Optional[tuple[float, float]] = None
        self._zoom_rect: Optional[Rectangle] = None
        self._zoom_ax = None

        # Axis-area drag (single-axis zoom) state
        self._axis_drag_active = False
        self._axis_drag_which: Optional[str] = None
        self._axis_drag_ax = None
        self._axis_drag_start_px: Optional[tuple[float, float]] = None

        canvas = self._canvas
        canvas.mpl_connect("scroll_event", self._pz_on_scroll)
        canvas.mpl_connect("button_press_event", self._pz_on_press)
        canvas.mpl_connect("button_release_event", self._pz_on_release)
        canvas.mpl_connect("motion_notify_event", self._pz_on_motion)

    def set_pan_zoom_axes(self, axes: Sequence) -> None:
        """Set which axes share X-pan/zoom. List order is preserved."""
        self._pan_axes = list(axes)

    def capture_default_lims(self) -> None:
        """Snapshot current xlim/ylim per axis as the "home" view."""
        self._default_lims = [
            (ax.get_xlim(), ax.get_ylim()) for ax in self._pan_axes
        ]

    # ------------------------------------------------------------------
    # Subclass hook — override to react after pan/zoom changed the view
    # ------------------------------------------------------------------

    def after_view_changed(self) -> None:
        """No-op by default. Override for things like peak relabeling."""
        return

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _toolbar_mode(self) -> str:
        tb = getattr(self, "_toolbar", None) or getattr(self, "toolbar", None)
        if tb is None:
            return ""
        return getattr(tb, "mode", "") or ""

    def _find_ax(self, event):
        for ax in self._pan_axes:
            if event.inaxes == ax:
                return ax
        return None

    def _detect_axis_area(self, event) -> tuple[Optional[object], Optional[str]]:
        """Detect a press on the axis-label gutter (returns (ax, 'x'|'y'))."""
        try:
            renderer = self._fig.canvas.get_renderer()
        except Exception:
            return None, None
        for ax in self._pan_axes:
            try:
                bbox = ax.get_window_extent(renderer=renderer)
            except Exception:
                continue
            # X-axis gutter: below the plot bbox, inside its X span
            if bbox.x0 <= event.x <= bbox.x1 and event.y < bbox.y0:
                return ax, "x"
            # Y-axis gutter: left of the plot bbox, inside its Y span
            if event.x < bbox.x0 and bbox.y0 <= event.y <= bbox.y1:
                return ax, "y"
        return None, None

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _pz_on_scroll(self, event) -> None:
        ax = self._find_ax(event)
        if ax is None or event.xdata is None or event.ydata is None:
            return
        factor = 0.8 if event.button == "up" else 1.25
        x = event.xdata
        y = event.ydata
        xlim = ax.get_xlim()
        new_xlim = (x - (x - xlim[0]) * factor, x + (xlim[1] - x) * factor)
        for a in self._pan_axes:
            a.set_xlim(new_xlim)
        ylim = ax.get_ylim()
        ax.set_ylim(y - (y - ylim[0]) * factor, y + (ylim[1] - y) * factor)
        self.after_view_changed()
        self._canvas.draw_idle()

    def _pz_on_press(self, event) -> None:
        if self._toolbar_mode():
            return  # matplotlib's pan/zoom tool is active — yield to it

        # Axis-area drag (click outside plot, in the gutter)
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
                # Double-click = reset to home
                for i, a in enumerate(self._pan_axes):
                    if i < len(self._default_lims):
                        xl, yl = self._default_lims[i]
                        a.set_xlim(xl)
                        a.set_ylim(yl)
                self.after_view_changed()
                self._canvas.draw_idle()
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

    def _pz_on_release(self, event) -> None:
        if event.button == 1 and self._axis_drag_active:
            self._axis_drag_active = False
            self._axis_drag_start_px = None
            self.after_view_changed()
            self._canvas.draw_idle()
            return

        if event.button == 1 and self._drag_active:
            self._drag_active = False
            self._drag_start_px = None
            self._drag_axis = None
            self.after_view_changed()
            self._canvas.draw_idle()
            return

        if event.button == 3 and self._zoom_active:
            self._zoom_active = False
            if self._zoom_rect is not None:
                self._zoom_rect.remove()
                self._zoom_rect = None
            if (
                self._zoom_start is not None
                and event.xdata is not None
                and event.ydata is not None
                and self._zoom_ax is not None
            ):
                x0, y0 = self._zoom_start
                x1, y1 = event.xdata, event.ydata
                if abs(x1 - x0) > 1e-6:
                    new_xlim = (min(x0, x1), max(x0, x1))
                    for a in self._pan_axes:
                        a.set_xlim(new_xlim)
                if abs(y1 - y0) > 1e-6:
                    self._zoom_ax.set_ylim(min(y0, y1), max(y0, y1))
            self._zoom_start = None
            self._zoom_ax = None
            self.after_view_changed()
            self._canvas.draw_idle()

    def _pz_on_motion(self, event) -> None:
        # Axis-area drag: zoom one axis only
        if (
            self._axis_drag_active
            and self._axis_drag_start_px is not None
            and self._axis_drag_ax is not None
        ):
            dx_px = event.x - self._axis_drag_start_px[0]
            dy_px = event.y - self._axis_drag_start_px[1]
            self._axis_drag_start_px = (event.x, event.y)
            factor_per_px = 0.005

            if self._axis_drag_which == "x":
                factor = 1 - dx_px * factor_per_px
                if factor <= 0:
                    return
                for a in self._pan_axes:
                    xl = a.get_xlim()
                    cx = (xl[0] + xl[1]) / 2
                    hw = (xl[1] - xl[0]) / 2 * factor
                    a.set_xlim(cx - hw, cx + hw)
            else:  # "y"
                factor = 1 + dy_px * factor_per_px
                if factor <= 0:
                    return
                ax = self._axis_drag_ax
                yl = ax.get_ylim()
                cy = (yl[0] + yl[1]) / 2
                hh = (yl[1] - yl[0]) / 2 * factor
                ax.set_ylim(cy - hh, cy + hh)
            self.after_view_changed()
            self._canvas.draw_idle()
            return

        # Right-drag rubber-band rectangle
        if self._zoom_active and self._zoom_rect is not None and self._zoom_start is not None:
            if (
                event.xdata is not None
                and event.ydata is not None
                and event.inaxes == self._zoom_ax
            ):
                x0, y0 = self._zoom_start
                self._zoom_rect.set_xy(
                    (min(x0, event.xdata), min(y0, event.ydata))
                )
                self._zoom_rect.set_width(abs(event.xdata - x0))
                self._zoom_rect.set_height(abs(event.ydata - y0))
                self._canvas.draw_idle()
            return

        # Left-drag pan
        if (
            not self._drag_active
            or self._drag_start_px is None
            or self._drag_ax is None
            or self._drag_start_data is None
        ):
            return
        if (
            event.inaxes != self._drag_ax
            or event.xdata is None
            or event.ydata is None
        ):
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

        if self._drag_axis == "x":
            for a in self._pan_axes:
                xl = a.get_xlim()
                a.set_xlim(xl[0] + dx, xl[1] + dx)
        else:  # "y"
            yl = self._drag_ax.get_ylim()
            self._drag_ax.set_ylim(yl[0] + dy, yl[1] + dy)
        self._canvas.draw_idle()


def share_x_iter(axes: Iterable) -> None:
    """Helper: chain ``sharex`` so panning one axis pans the rest.

    Most subclasses don't need this — the mixin already pushes new xlim
    to every axis in ``_pan_axes`` — but it's exposed for callers that
    want native matplotlib ``sharex`` behavior on top of the mixin.
    """
    axes_list = list(axes)
    if len(axes_list) < 2:
        return
    base = axes_list[0]
    for ax in axes_list[1:]:
        ax.sharex(base)
