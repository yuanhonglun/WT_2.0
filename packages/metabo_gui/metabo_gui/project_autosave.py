"""Helper for auto-saving a project file next to the input data.

Each app picks its own project-file extension (``.asfam``, ``.gcmsproj``, ...)
and serializer; this helper just builds the standard target path so all
apps use the same naming convention:

    <data_dir>/<AppNamePrefix>Project_YYYYMMDD_HHMMSS.<ext>
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable


def auto_save_path(
    first_input: str | Path,
    *,
    app_prefix: str,
    extension: str,
    timestamp: datetime | None = None,
) -> Path:
    """Build an auto-save project path next to ``first_input``.

    Parameters
    ----------
    first_input : str | Path
        Any file in the data directory (typically the first mzML).
    app_prefix : str
        Display prefix used in the filename (e.g. ``"ASFAM"``, ``"Gcms"``).
    extension : str
        Project-file extension WITHOUT the leading dot (e.g. ``"asfam"``,
        ``"gcmsproj"``).
    timestamp : datetime, optional
        Override the timestamp (used by tests). Defaults to ``now()``.
    """
    ts = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    data_dir = Path(first_input).parent
    return data_dir / f"{app_prefix}Project_{ts}.{extension.lstrip('.')}"


def first_existing(paths: Iterable[str | Path]) -> Path | None:
    """Return the first ``Path`` in ``paths`` that exists on disk, or ``None``."""
    for raw in paths:
        p = Path(raw)
        if p.exists():
            return p
    return None
