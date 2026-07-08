"""Toolbar suppresses XY coordinates and uses a custom save filename."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("matplotlib")


def _make_canvas(qapp):
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from PyQt5.QtWidgets import QWidget

    parent = QWidget()
    fig = Figure()
    fig.add_subplot(111)
    return FigureCanvasQTAgg(fig), parent


def test_toolbar_hides_coord_label(qapp):
    from metabo_gui.plot_toolbar import make_plot_toolbar

    canvas, parent = _make_canvas(qapp)
    tb = make_plot_toolbar(canvas, parent, default_prefix="EIC_test")

    # locLabel exists on stock matplotlib NavigationToolbar2QT and is the
    # widget that displays "x=… y=…" on hover. It must be hidden.
    assert tb.locLabel.isVisible() is False


def test_set_message_is_noop(qapp):
    from metabo_gui.plot_toolbar import make_plot_toolbar

    canvas, parent = _make_canvas(qapp)
    tb = make_plot_toolbar(canvas, parent)
    # Should not raise and should leave locLabel text unchanged.
    tb.locLabel.setText("INITIAL")
    tb.set_message("x=1.234 y=5.678")
    assert tb.locLabel.text() == "INITIAL"


def test_save_prefix_round_trip(qapp):
    from metabo_gui.plot_toolbar import make_plot_toolbar

    canvas, parent = _make_canvas(qapp)
    tb = make_plot_toolbar(canvas, parent, default_prefix="EIC_test")
    assert tb._save_prefix == "EIC_test"
    tb.set_save_prefix("MS2_F123")
    assert tb._save_prefix == "MS2_F123"


def test_only_kept_actions_remain(qapp):
    from metabo_gui.plot_toolbar import make_plot_toolbar

    canvas, parent = _make_canvas(qapp)
    tb = make_plot_toolbar(canvas, parent)
    texts = {a.text() for a in tb.actions() if a.text()}
    # Pan / Zoom / Home / Forward / Back must be removed
    assert "Pan" not in texts
    assert "Zoom" not in texts
    assert "Home" not in texts
    # Subplots / Customize / Save must remain
    assert texts.issubset({"Subplots", "Customize", "Save"})
