"""ASFAMProcessor GUI application entry point with crash logging."""
import sys
import os
import logging
import traceback
import multiprocessing
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt


def setup_crash_log():
    """Setup crash logging to file. Auto-cleans old logs (keeps last 10)."""
    log_dir = Path.home() / ".asfam_logs"
    log_dir.mkdir(exist_ok=True)

    # Auto-clean: keep only the newest 10 log files, delete the rest
    try:
        logs = sorted(log_dir.glob("crash_*.log"), key=lambda p: p.stat().st_mtime)
        if len(logs) > 10:
            for old_log in logs[:-10]:
                old_log.unlink()
    except Exception:
        pass

    log_file = log_dir / f"crash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Setup file handler
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))

    # Setup root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)

    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    root.addHandler(console)

    logging.info("ASFAMProcessor starting, log file: %s", log_file)
    return log_file


def global_exception_handler(exc_type, exc_value, exc_tb):
    """Handle uncaught exceptions: log to file and show message."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return

    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logging.critical("UNCAUGHT EXCEPTION:\n%s", error_msg)

    # Try to show error dialog
    try:
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(
            None, "ASFAMProcessor Crash",
            f"An unexpected error occurred:\n\n{exc_value}\n\n"
            f"Details have been logged to:\n{_log_file}\n\n"
            f"Please report this issue.",
        )
    except Exception:
        pass


_log_file = None


def main():
    """Launch the ASFAMProcessor GUI."""
    global _log_file
    _log_file = setup_crash_log()

    # Install global exception handler
    sys.excepthook = global_exception_handler

    # High DPI support
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("ASFAMProcessor")
    app.setApplicationVersion("0.2.260326")

    # Set default font
    from PyQt5.QtGui import QFont
    font = QFont("Arial", 10)
    app.setFont(font)

    from asfam.gui.main_window import MainWindow
    window = MainWindow()
    window.show()

    logging.info("GUI window shown")
    sys.exit(app.exec_())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
