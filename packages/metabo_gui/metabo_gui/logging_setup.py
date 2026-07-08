"""Per-app crash log directory with N-file rotation.

Each app calls ``setup_app_logging('asfam')`` early in its main entry
point. The function:
  - creates ``~/.<app>_logs/`` if needed
  - removes log files beyond ``keep_n`` (oldest first by mtime)
  - opens a new ``crash_<timestamp>.log`` and attaches a DEBUG file
    handler to the root logger
  - attaches an INFO console handler

Returns the ``Path`` to the active log file so the caller can show it
in a crash dialog.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def setup_app_logging(
    app_short_name: str,
    *,
    keep_n: int = 10,
    log_dir: Path | None = None,
) -> Path:
    """Configure root logging with rotating per-app crash logs.

    Parameters
    ----------
    app_short_name : str
        Used both for the log directory ``~/.<name>_logs/`` and (capitalized)
        as the boot log message.
    keep_n : int, default 10
        Number of historical ``crash_*.log`` files to keep. Older files
        are deleted before opening the new one.
    log_dir : Path, optional
        Override the default ``~/.<name>_logs/`` location (used by tests).
    """
    directory = log_dir or (Path.home() / f".{app_short_name}_logs")
    directory.mkdir(parents=True, exist_ok=True)

    try:
        logs = sorted(directory.glob("crash_*.log"), key=lambda p: p.stat().st_mtime)
        if len(logs) > keep_n:
            for old in logs[:-keep_n]:
                try:
                    old.unlink()
                except OSError:
                    pass
    except OSError:
        pass

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = directory / f"crash_{timestamp}.log"

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"
    ))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console)

    logging.info("%s starting, log file: %s", app_short_name, log_file)
    return log_file
