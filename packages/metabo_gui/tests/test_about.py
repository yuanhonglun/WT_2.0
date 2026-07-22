"""About dialog factory smoke test + author/email constants."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")


def test_author_constants():
    from metabo_gui import about as a

    assert a.AUTHOR_NAME == "Honglun Yuan"
    assert a.AUTHOR_EMAIL == "yuanhonglun@hotmail.com"
    assert a.AUTHOR_INSTITUTION == "Hainan University"


def test_show_about_dialog_does_not_crash(qapp, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox

    captured: dict = {}

    def fake_about(parent, title, body):
        captured["title"] = title
        captured["body"] = body

    monkeypatch.setattr(QMessageBox, "about", staticmethod(fake_about))

    from metabo_gui.about import show_about_dialog

    show_about_dialog(
        None,
        app_name="ASFAM Processor",
        version="9.9.99",
        description="A test description.",
    )
    assert "ASFAM Processor v9.9.99" in captured["body"]
    assert "Honglun Yuan" in captured["body"]
    assert "yuanhonglun@hotmail.com" in captured["body"]
    assert "Hainan University" in captured["body"]
    assert captured["title"] == "About ASFAM Processor"
