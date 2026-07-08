"""Minimal matplotlib NavigationToolbar for metabo-platform GUIs.

Differences vs matplotlib's stock NavigationToolbar2QT:
  - Only Subplots / Customize / Save actions are kept (Pan/Zoom/Forward/Back
    are removed because the canvas-side ``PanZoomMixin`` provides them
    via mouse).
  - The XY coordinate label is hidden and ``set_message`` is a no-op so
    hovering the plot never paints coordinates anywhere on the GUI.
  - ``save_figure`` uses a configurable filename prefix so users get
    ``EIC_F123_mz147.500.png`` instead of ``image.png`` by default.
  - Optional white-on-blue icon recolor for use over the themed toolbar.
"""
from __future__ import annotations

import os

from PyQt5.QtCore import QSize
from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from metabo_gui.theme import THEME_BLUE


_KEEP_ACTIONS = {"Subplots", "Customize", "Save"}


class _CustomToolbar(NavigationToolbar2QT):
    """Toolbar with custom default save filename and no XY coord display."""

    def __init__(self, canvas, parent, default_prefix: str = "image") -> None:
        super().__init__(canvas, parent)
        self._save_prefix = default_prefix
        # Hide the matplotlib-supplied coordinate label so hovering the
        # plot never displays XY coordinates (per platform convention).
        if getattr(self, "locLabel", None) is not None:
            self.locLabel.setVisible(False)

    def set_message(self, s: str) -> None:  # type: ignore[override]
        # No-op: matplotlib calls this on every mouse-motion event with
        # the formatted XY pair. We swallow it to keep the GUI quiet.
        return

    def set_save_prefix(self, prefix: str) -> None:
        self._save_prefix = prefix

    def save_figure(self, *args) -> None:  # type: ignore[override]
        filetypes = self.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = sorted(filetypes.items())
        default_filetype = self.canvas.get_default_filetype()

        filters: list[str] = []
        selected_filter: str | None = None
        for name, exts in sorted_filetypes:
            exts_list = " ".join(f"*.{ext}" for ext in exts)
            filt = f"{name} ({exts_list})"
            if default_filetype in exts:
                selected_filter = filt
            filters.append(filt)

        fname, _ = QFileDialog.getSaveFileName(
            self.canvas.parent(),
            "Save figure",
            os.path.join(
                os.path.expanduser("~"),
                f"{self._save_prefix}.{default_filetype}",
            ),
            ";;".join(filters),
            selected_filter,
        )
        if fname:
            try:
                self.canvas.figure.savefig(fname, dpi=150, bbox_inches="tight")
            except Exception as exc:
                QMessageBox.critical(
                    self.canvas.parent(), "Error saving figure", str(exc)
                )


def make_plot_toolbar(
    canvas,
    parent,
    *,
    white_icons: bool = False,
    icon_size: int = 24,
    default_prefix: str = "image",
) -> _CustomToolbar:
    """Build a stripped-down toolbar exposing only Subplots/Customize/Save."""
    tb = _CustomToolbar(canvas, parent, default_prefix=default_prefix)
    tb.setMaximumHeight(32)

    for action in list(tb.actions()):
        text = action.text()
        if text and text not in _KEEP_ACTIONS:
            tb.removeAction(action)

    if white_icons:
        _recolor_icons(tb, icon_size)
        tb.setStyleSheet(
            f"QToolBar {{ background-color: {THEME_BLUE}; border: none; }}"
            f"QToolTip {{ background-color: {THEME_BLUE}; color: white;"
            "  border: 1px solid white; font-size: 12px; padding: 3px; }}"
        )
    return tb


def _recolor_icons(toolbar: NavigationToolbar2QT, size: int = 28) -> None:
    """Repaint toolbar icons in white at the requested pixel size."""
    for action in toolbar.actions():
        icon = action.icon()
        if icon.isNull():
            continue
        pixmap = icon.pixmap(size, size)
        wp = QPixmap(pixmap.size())
        wp.fill(QColor(0, 0, 0, 0))
        painter = QPainter(wp)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.drawPixmap(0, 0, pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(wp.rect(), QColor(255, 255, 255))
        painter.end()
        action.setIcon(QIcon(wp))
    toolbar.setIconSize(QSize(size, size))
