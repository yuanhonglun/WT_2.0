"""PanZoomMixin reacts to scroll/drag/double-click and respects toolbar mode."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("matplotlib")


class _FakeEvent:
    def __init__(self, *, button=None, xdata=None, ydata=None, x=0, y=0,
                 inaxes=None, dblclick=False):
        self.button = button
        self.xdata = xdata
        self.ydata = ydata
        self.x = x
        self.y = y
        self.inaxes = inaxes
        self.dblclick = dblclick


@pytest.fixture
def widget(qapp):
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from PyQt5.QtWidgets import QWidget

    from metabo_gui.canvas import PanZoomMixin

    class _W(QWidget, PanZoomMixin):
        def __init__(self):
            super().__init__()
            self._fig = Figure()
            self._canvas = FigureCanvasQTAgg(self._fig)
            self.ax = self._fig.add_subplot(111)
            self.ax.set_xlim(0, 10)
            self.ax.set_ylim(0, 10)
            self.init_pan_zoom()
            self.set_pan_zoom_axes([self.ax])
            self.capture_default_lims()

    return _W()


def test_scroll_zooms_around_cursor(widget):
    ev = _FakeEvent(button="up", xdata=5, ydata=5, inaxes=widget.ax)
    widget._pz_on_scroll(ev)
    xlim = widget.ax.get_xlim()
    # zoom-in (factor 0.8) centered at 5
    assert xlim[0] > 0
    assert xlim[1] < 10


def test_double_click_resets(widget):
    widget.ax.set_xlim(2, 4)
    widget.ax.set_ylim(2, 4)
    ev = _FakeEvent(
        button=1, xdata=3, ydata=3, inaxes=widget.ax, dblclick=True,
    )
    widget._pz_on_press(ev)
    assert widget.ax.get_xlim() == (0.0, 10.0)
    assert widget.ax.get_ylim() == (0.0, 10.0)


def test_left_drag_pans_x_when_horizontal_motion(widget):
    press = _FakeEvent(button=1, xdata=5, ydata=5, x=100, y=100, inaxes=widget.ax)
    widget._pz_on_press(press)
    # Move 20px right horizontally
    move = _FakeEvent(xdata=4, ydata=5, x=120, y=100, inaxes=widget.ax)
    widget._pz_on_motion(move)
    assert widget._drag_axis == "x"
    xlim = widget.ax.get_xlim()
    # Drag right -> data shifts left -> xlim[0] increases
    assert xlim[0] > 0


def test_right_drag_zoom_to_rectangle(widget):
    press = _FakeEvent(button=3, xdata=2, ydata=2, x=20, y=20, inaxes=widget.ax)
    widget._pz_on_press(press)
    move = _FakeEvent(xdata=8, ydata=8, x=80, y=80, inaxes=widget.ax)
    widget._pz_on_motion(move)
    release = _FakeEvent(button=3, xdata=8, ydata=8, x=80, y=80, inaxes=widget.ax)
    widget._pz_on_release(release)
    assert widget.ax.get_xlim() == (2.0, 8.0)
    assert widget.ax.get_ylim() == (2.0, 8.0)


def test_after_view_changed_hook_fires_on_double_click(qapp):
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from PyQt5.QtWidgets import QWidget

    from metabo_gui.canvas import PanZoomMixin

    class _W(QWidget, PanZoomMixin):
        hook_calls = 0

        def __init__(self):
            super().__init__()
            self._fig = Figure()
            self._canvas = FigureCanvasQTAgg(self._fig)
            self.ax = self._fig.add_subplot(111)
            self.ax.set_xlim(0, 10); self.ax.set_ylim(0, 10)
            self.init_pan_zoom()
            self.set_pan_zoom_axes([self.ax])
            self.capture_default_lims()

        def after_view_changed(self):
            self.hook_calls += 1

    w = _W()
    ev = _FakeEvent(button=1, xdata=5, ydata=5, inaxes=w.ax, dblclick=True)
    w._pz_on_press(ev)
    assert w.hook_calls >= 1


def test_yields_when_toolbar_mode_active(qapp):
    """When matplotlib's NavigationToolbar pan/zoom tool is active, the
    mixin should NOT intercept presses."""
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from PyQt5.QtWidgets import QWidget

    from metabo_gui.canvas import PanZoomMixin

    class _FakeToolbar:
        mode = "pan"

    class _W(QWidget, PanZoomMixin):
        def __init__(self):
            super().__init__()
            self._fig = Figure()
            self._canvas = FigureCanvasQTAgg(self._fig)
            self._toolbar = _FakeToolbar()
            self.ax = self._fig.add_subplot(111)
            self.ax.set_xlim(0, 10); self.ax.set_ylim(0, 10)
            self.init_pan_zoom()
            self.set_pan_zoom_axes([self.ax])
            self.capture_default_lims()

    w = _W()
    ev = _FakeEvent(button=1, xdata=5, ydata=5, x=50, y=50, inaxes=w.ax)
    w._pz_on_press(ev)
    # Press was ignored — drag state should still be inactive.
    assert w._drag_active is False
