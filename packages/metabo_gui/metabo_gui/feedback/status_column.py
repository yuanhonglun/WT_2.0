"""Renders the Status column cell from a FeedbackEntry."""
from __future__ import annotations

from typing import Callable

from PyQt5.QtCore import QModelIndex, Qt
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem

from .controller import FeedbackController
from .models import FeedbackEntry
from .tags import STATUS_COLUMN_LABELS, TAG_LABELS, VERIFIED_GOOD_TAG


def status_text_for_entry(entry: FeedbackEntry | None) -> str:
    """Return the short status text to display in the cell."""
    if entry is None:
        return ""
    if entry.tags:
        return f"⚠ {STATUS_COLUMN_LABELS.get(entry.tags[0], entry.tags[0])}"
    if entry.verified_good:
        return "✓"
    return ""


def tooltip_for_entry(entry: FeedbackEntry | None) -> str:
    """Return the tooltip text (all tags, comment first line, timestamp)."""
    if entry is None:
        return ""
    parts: list[str] = []
    if entry.tags:
        # Include both tag name and label for clarity
        tag_strs = [f"{t} ({TAG_LABELS.get(t, t)})" for t in entry.tags]
        parts.append("Issues: " + ", ".join(tag_strs))
    if entry.verified_good:
        parts.append("Verified good")
    if entry.comment:
        first_line = entry.comment.splitlines()[0]
        parts.append(f"Comment: {first_line}")
    parts.append(f"Updated {entry.updated_at}")
    return "\n".join(parts)


class FeatureStatusDelegate(QStyledItemDelegate):
    """Paints the Status column for a feature-table row.

    Hosts no state of its own; queries the FeedbackController via the
    feature_id_provider callable each repaint, so it always reflects the
    latest store contents.
    """

    def __init__(
        self,
        controller: FeedbackController,
        feature_id_provider: Callable[[QModelIndex], str | None],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._fid_provider = feature_id_provider
        self._controller.storeChanged.connect(self._on_store_changed)
        self._owner_view = None

    def attach_view(self, view) -> None:
        """Optional: wire the view so it can repaint on storeChanged."""
        self._owner_view = view

    def _on_store_changed(self, _feature_id: str) -> None:
        if self._owner_view is not None:
            self._owner_view.viewport().update()

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        """Paint the cell with status text in appropriate color."""
        fid = self._fid_provider(index)
        entry = self._controller.get_entry(fid) if fid else None
        text = status_text_for_entry(entry)
        if not text:
            super().paint(painter, option, index)
            return
        painter.save()
        if text.startswith("⚠"):
            painter.setPen(QColor("#c62828"))
        elif text == "✓":
            painter.setPen(QColor("#2e7d32"))
        painter.drawText(option.rect, Qt.AlignCenter, text)
        painter.restore()

    def helpEvent(self, event, view, option, index):
        """Show tooltip on hover."""
        fid = self._fid_provider(index)
        entry = self._controller.get_entry(fid) if fid else None
        tip = tooltip_for_entry(entry)
        if tip:
            from PyQt5.QtWidgets import QToolTip
            QToolTip.showText(event.globalPos(), tip, view)
            return True
        return super().helpEvent(event, view, option, index)
