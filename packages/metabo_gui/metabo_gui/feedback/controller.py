"""Qt-aware wrapper holding a FeedbackStore + autosave timer + signals."""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from .models import FeedbackEntry, FeedbackStore
from .store import save_alongside

logger = logging.getLogger(__name__)

DEFAULT_AUTOSAVE_MS = 1000


class FeedbackController(QObject):
    """Holds the in-memory FeedbackStore for a project and autosaves it.

    The controller is the single source of truth for the GUI side. Widgets
    read via get_entry / list_entries and write via upsert_entry / remove_entry.
    """

    storeChanged = pyqtSignal(str)  # feature_id that changed

    def __init__(
        self,
        project_path: str | Path,
        store: FeedbackStore,
        autosave_ms: int = DEFAULT_AUTOSAVE_MS,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._project_path = Path(project_path)
        self._store = store
        self._dirty = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(autosave_ms)
        self._timer.timeout.connect(self._autosave)

    @property
    def store(self) -> FeedbackStore:
        return self._store

    @property
    def project_path(self) -> Path:
        return self._project_path

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def set_project_path(self, project_path: str | Path) -> None:
        """For Save-As: rebind the controller to a new project path.

        Future autosaves write to the new path. Caller may want to immediately
        ``save_now()`` after this to persist current state to the new path.
        """
        self._project_path = Path(project_path)

    def get_entry(self, feature_id: str) -> FeedbackEntry | None:
        for e in self._store.entries:
            if e.feature_id_at_run == feature_id:
                return e
        return None

    def upsert_entry(self, entry: FeedbackEntry) -> None:
        for i, e in enumerate(self._store.entries):
            if e.feature_id_at_run == entry.feature_id_at_run:
                self._store.entries[i] = entry
                self._mark_dirty(entry.feature_id_at_run)
                return
        self._store.entries.append(entry)
        self._mark_dirty(entry.feature_id_at_run)

    def remove_entry(self, feature_id: str) -> None:
        before = len(self._store.entries)
        self._store.entries = [
            e for e in self._store.entries if e.feature_id_at_run != feature_id
        ]
        if len(self._store.entries) != before:
            self._mark_dirty(feature_id)

    def save_now(self) -> None:
        self._autosave()

    def _mark_dirty(self, feature_id: str) -> None:
        self._dirty = True
        self._timer.start()
        self.storeChanged.emit(feature_id)

    def _autosave(self) -> None:
        try:
            save_alongside(self._project_path, self._store)
            self._dirty = False
        except OSError as exc:
            logger.warning("Feedback autosave failed: %s", exc)
            # Stay dirty so a future change retriggers; do not crash GUI.
