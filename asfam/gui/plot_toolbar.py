"""Shared plot toolbar and interaction utilities for EIC, MS2, and scatter plots."""
from __future__ import annotations

import os
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor
from PyQt5.QtCore import QSize
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT


# Actions to KEEP in toolbar (matplotlib internal names)
_KEEP_ACTIONS = {"Subplots", "Customize", "Save"}


class _CustomToolbar(NavigationToolbar2QT):
    """Toolbar with custom default save filename."""

    def __init__(self, canvas, parent, default_prefix="image"):
        super().__init__(canvas, parent)
        self._save_prefix = default_prefix

    def set_save_prefix(self, prefix: str):
        self._save_prefix = prefix

    def save_figure(self, *args):
        """Override to use custom default filename."""
        filetypes = self.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = sorted(filetypes.items())
        default_filetype = self.canvas.get_default_filetype()

        filters = []
        selected_filter = None
        for name, exts in sorted_filetypes:
            exts_list = " ".join(f"*.{ext}" for ext in exts)
            filt = f"{name} ({exts_list})"
            if default_filetype in exts:
                selected_filter = filt
            filters.append(filt)

        fname, chosen_filter = QFileDialog.getSaveFileName(
            self.canvas.parent(),
            "Save figure",
            os.path.join(os.path.expanduser("~"), f"{self._save_prefix}.{default_filetype}"),
            ";;".join(filters),
            selected_filter,
        )
        if fname:
            try:
                self.canvas.figure.savefig(fname, dpi=150, bbox_inches="tight")
            except Exception as e:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self.canvas.parent(), "Error saving figure", str(e))


def make_plot_toolbar(
    canvas, parent, white_icons: bool = False, icon_size: int = 24,
    default_prefix: str = "image",
) -> _CustomToolbar:
    """Create a toolbar with only subplot-config, edit-axis, and save."""
    tb = _CustomToolbar(canvas, parent, default_prefix)
    tb.setMaximumHeight(32)

    # Remove unwanted actions
    for action in list(tb.actions()):
        text = action.text()
        if text and text not in _KEEP_ACTIONS:
            tb.removeAction(action)

    if white_icons:
        _recolor_icons(tb, icon_size)
        tb.setStyleSheet(
            "QToolBar { background-color: #2D6A9F; border: none; }"
            "QToolTip { background-color: #2D6A9F; color: white; "
            "border: 1px solid white; font-size: 12px; padding: 3px; }"
        )
    return tb


def _recolor_icons(toolbar: NavigationToolbar2QT, size: int = 28):
    """Recolor toolbar icons to white and make them larger."""
    for action in toolbar.actions():
        icon = action.icon()
        if icon.isNull():
            continue
        pixmap = icon.pixmap(size, size)
        wp = QPixmap(pixmap.size())
        wp.fill(QColor(0, 0, 0, 0))
        p = QPainter(wp)
        p.setCompositionMode(QPainter.CompositionMode_Source)
        p.drawPixmap(0, 0, pixmap)
        p.setCompositionMode(QPainter.CompositionMode_SourceIn)
        p.fillRect(wp.rect(), QColor(255, 255, 255))
        p.end()
        action.setIcon(QIcon(wp))
    toolbar.setIconSize(QSize(size, size))
