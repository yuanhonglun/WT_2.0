"""NotesDock: dock widget for editing FeedbackEntry of the current feature."""
from __future__ import annotations

import datetime as _dt
from typing import Optional

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QCheckBox, QDockWidget, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
    QPushButton, QVBoxLayout, QWidget,
)

from .controller import FeedbackController
from .models import FeatureSignature, FeedbackEntry
from .tags import ISSUE_TAGS, TAG_LABELS, VERIFIED_GOOD_TAG

DEFAULT_COMMENT_DEBOUNCE_MS = 500


class NotesDock(QDockWidget):
    """Dock for editing the FeedbackEntry of the currently selected feature.

    Tag and verified-good toggles persist immediately; comment edits are
    debounced (default 500 ms). Switching feature flushes pending changes.
    Loading an entry into the view never triggers a writeback.
    """

    def __init__(
        self,
        controller: FeedbackController,
        comment_debounce_ms: int = DEFAULT_COMMENT_DEBOUNCE_MS,
        parent=None,
    ) -> None:
        super().__init__("Notes", parent)
        self.setObjectName("FeedbackNotesDock")
        self._controller = controller
        self._current_feature_id: Optional[str] = None
        self._current_signature: Optional[FeatureSignature] = None
        self._suppress_signals = False

        self._comment_timer = QTimer(self)
        self._comment_timer.setSingleShot(True)
        self._comment_timer.setInterval(comment_debounce_ms)
        self._comment_timer.timeout.connect(self._flush_comment)

        self._build_ui()

    @property
    def current_feature_id(self) -> Optional[str]:
        return self._current_feature_id

    def set_current_feature(
        self,
        feature_id: Optional[str],
        signature: Optional[FeatureSignature],
    ) -> None:
        """Switch the dock to display/edit the entry for ``feature_id``.

        Flushes any pending comment edit on the previously displayed feature
        before switching, so debounced changes are never lost.
        """
        self._flush_comment()
        self._current_feature_id = feature_id
        self._current_signature = signature
        self._refresh_from_store()

    def _build_ui(self) -> None:
        container = QWidget()
        self.setWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)

        self.header_label = QLabel("(无选中 feature)")
        self.header_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.header_label)

        layout.addWidget(QLabel("问题标签 (可多选):"))
        tag_row = QHBoxLayout()
        self.tag_checkboxes: dict[str, QCheckBox] = {}
        for tag in ISSUE_TAGS:
            cb = QCheckBox(TAG_LABELS[tag])
            cb.toggled.connect(self._on_tag_or_verified_toggled)
            tag_row.addWidget(cb)
            self.tag_checkboxes[tag] = cb
        tag_row.addStretch(1)
        layout.addLayout(tag_row)

        self.verified_checkbox = QCheckBox(TAG_LABELS[VERIFIED_GOOD_TAG])
        self.verified_checkbox.toggled.connect(self._on_tag_or_verified_toggled)
        layout.addWidget(self.verified_checkbox)

        layout.addWidget(QLabel("备注:"))
        self.comment_edit = QPlainTextEdit()
        self.comment_edit.textChanged.connect(self._on_comment_changed)
        layout.addWidget(self.comment_edit, stretch=1)

        self.timestamp_label = QLabel("")
        self.timestamp_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self.timestamp_label)

        self.clear_button = QPushButton("清除该 feature 的所有反馈")
        self.clear_button.clicked.connect(self._on_clear_clicked)
        layout.addWidget(self.clear_button)

        self._set_inputs_enabled(False)

    def _refresh_from_store(self) -> None:
        self._suppress_signals = True
        try:
            fid = self._current_feature_id
            sig = self._current_signature
            if fid is None or sig is None:
                self.header_label.setText("(无选中 feature)")
                for cb in self.tag_checkboxes.values():
                    cb.setChecked(False)
                self.verified_checkbox.setChecked(False)
                self.comment_edit.setPlainText("")
                self.timestamp_label.setText("")
                self._set_inputs_enabled(False)
                return
            self.header_label.setText(
                f"Feature {fid}    mz={sig.mz:.4f}    RT={sig.rt:.3f} min    [{sig.mode}]"
            )
            self._set_inputs_enabled(True)
            entry = self._controller.get_entry(fid)
            for tag, cb in self.tag_checkboxes.items():
                cb.setChecked(entry is not None and tag in entry.tags)
            self.verified_checkbox.setChecked(bool(entry and entry.verified_good))
            self.comment_edit.setPlainText(entry.comment if entry else "")
            if entry:
                self.timestamp_label.setText(
                    f"created {entry.created_at}    updated {entry.updated_at}"
                )
            else:
                self.timestamp_label.setText("")
        finally:
            self._suppress_signals = False

    def _set_inputs_enabled(self, enabled: bool) -> None:
        for cb in self.tag_checkboxes.values():
            cb.setEnabled(enabled)
        self.verified_checkbox.setEnabled(enabled)
        self.comment_edit.setEnabled(enabled)
        self.clear_button.setEnabled(enabled)

    def _on_tag_or_verified_toggled(self, _checked: bool) -> None:
        if self._suppress_signals:
            return
        self._persist_current()

    def _on_comment_changed(self) -> None:
        if self._suppress_signals:
            return
        self._comment_timer.start()

    def _flush_comment(self) -> None:
        self._comment_timer.stop()
        if self._suppress_signals:
            return
        if self._current_feature_id is None:
            return
        self._persist_current()

    def _persist_current(self) -> None:
        fid = self._current_feature_id
        sig = self._current_signature
        if fid is None or sig is None:
            return
        tags = [t for t, cb in self.tag_checkboxes.items() if cb.isChecked()]
        verified = self.verified_checkbox.isChecked()
        comment = self.comment_edit.toPlainText()
        if not tags and not verified and not comment:
            existing = self._controller.get_entry(fid)
            if existing is not None:
                self._controller.remove_entry(fid)
            self.timestamp_label.setText("")
            return
        now = _dt.datetime.now().isoformat(timespec="seconds")
        existing = self._controller.get_entry(fid)
        created_at = existing.created_at if existing else now
        run_ts_created = (
            existing.run_timestamp_created if existing
            else self._controller.store.run_context.run_timestamp
        )
        entry = FeedbackEntry(
            feature_id_at_run=fid,
            feature_signature=sig,
            tags=tags,
            verified_good=verified,
            comment=comment,
            created_at=created_at,
            updated_at=now,
            run_timestamp_created=run_ts_created,
        )
        self._controller.upsert_entry(entry)
        self.timestamp_label.setText(f"created {created_at}    updated {now}")

    def _on_clear_clicked(self) -> None:
        if self._current_feature_id is None:
            return
        reply = QMessageBox.question(
            self, "清除反馈",
            f"清除 feature {self._current_feature_id} 的所有反馈？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._clear_current_no_confirm()

    def _clear_current_no_confirm(self) -> None:
        """Test-friendly clear (no modal)."""
        fid = self._current_feature_id
        if fid is None:
            return
        self._suppress_signals = True
        try:
            for cb in self.tag_checkboxes.values():
                cb.setChecked(False)
            self.verified_checkbox.setChecked(False)
            self.comment_edit.setPlainText("")
            self.timestamp_label.setText("")
        finally:
            self._suppress_signals = False
        self._controller.remove_entry(fid)
