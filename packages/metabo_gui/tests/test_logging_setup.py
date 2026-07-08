"""Log directory rotation keeps newest N files."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from metabo_gui.logging_setup import setup_app_logging


def _detach_handlers():
    """Remove handlers added by setup_app_logging so tests don't leak state."""
    root = logging.getLogger()
    for h in list(root.handlers):
        try:
            h.close()
        except Exception:
            pass
        root.removeHandler(h)


def test_rotation_keeps_newest_n(tmp_path: Path):
    # Pre-populate 12 fake crash logs with ascending mtimes
    for i in range(12):
        p = tmp_path / f"crash_2026010{i:02d}.log"
        p.write_text("dummy")
        os.utime(p, (time.time() - (12 - i) * 60, time.time() - (12 - i) * 60))

    try:
        path = setup_app_logging("testapp", keep_n=10, log_dir=tmp_path)
        # New log file exists
        assert path.exists()
        # Total files = 10 retained + 1 new = 11 max
        all_logs = sorted(tmp_path.glob("crash_*.log"))
        assert len(all_logs) <= 11
        # Oldest pre-existing files should be gone (the first ones we wrote)
        assert not (tmp_path / "crash_20260100.log").exists()
        assert not (tmp_path / "crash_20260101.log").exists()
    finally:
        _detach_handlers()


def test_creates_directory_if_missing(tmp_path: Path):
    target = tmp_path / "nested" / "logs"
    try:
        path = setup_app_logging("testapp", log_dir=target)
        assert target.is_dir()
        assert path.parent == target
    finally:
        _detach_handlers()


def test_filehandler_attached_to_root(tmp_path: Path):
    try:
        setup_app_logging("testapp", log_dir=tmp_path)
        logging.info("hello from test")
        # Force flush via handler shutdown
        for h in logging.getLogger().handlers:
            h.flush()
        logs = list(tmp_path.glob("crash_*.log"))
        assert logs, "no log file produced"
        text = logs[-1].read_text(encoding="utf-8")
        assert "hello from test" in text
    finally:
        _detach_handlers()
