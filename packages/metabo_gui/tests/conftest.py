"""Shared pytest fixtures for metabo_gui tests."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    """Yield a session-wide ``QApplication`` for offscreen Qt tests."""
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
