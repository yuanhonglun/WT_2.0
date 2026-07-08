"""Sidecar file IO. Pure functions, no Qt.

Filename convention: <project_path>.feedback.json
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from .models import SCHEMA_VERSION, FeedbackStore

logger = logging.getLogger(__name__)

SIDECAR_SUFFIX = ".feedback.json"


def sidecar_path_for(project_path: str | os.PathLike) -> Path:
    p = Path(project_path)
    return p.with_name(p.name + SIDECAR_SUFFIX)


def load_alongside(project_path: str | os.PathLike) -> FeedbackStore | None:
    """Return the FeedbackStore for this project, or None if missing/invalid.

    Never raises; logs and returns None on any error. Sidecar with a
    schema_version newer than this code understands is also refused.
    """
    side = sidecar_path_for(project_path)
    if not side.exists():
        return None
    try:
        raw = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read feedback sidecar %s: %s", side, exc)
        return None
    if int(raw.get("schema_version", 0)) > SCHEMA_VERSION:
        logger.warning(
            "Feedback sidecar %s has schema_version %s > supported %s; ignoring",
            side, raw.get("schema_version"), SCHEMA_VERSION,
        )
        return None
    try:
        return FeedbackStore.from_dict(raw)
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Feedback sidecar %s malformed: %s", side, exc)
        return None


def save_alongside(project_path: str | os.PathLike, store: FeedbackStore) -> None:
    """Atomic write: write to .tmp, fsync, rename."""
    side = sidecar_path_for(project_path)
    side.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(store.to_dict(), ensure_ascii=False, indent=2)
    fd, tmp_path = tempfile.mkstemp(
        prefix=side.name, suffix=".tmp", dir=str(side.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, side)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
