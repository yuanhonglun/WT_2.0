"""ASFAMProcessor GUI application entry point with crash logging."""
import sys
import logging
import traceback
import multiprocessing

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from metabo_gui.logging_setup import setup_app_logging


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
            None, "ASFAM Processor Crash",
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
    _log_file = setup_app_logging("asfam")

    # Install global exception handler
    sys.excepthook = global_exception_handler

    # High DPI support
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("ASFAM Processor")
    from asfam import __version__
    app.setApplicationVersion(__version__)

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
