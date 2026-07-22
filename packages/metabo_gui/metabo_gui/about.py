"""Standard About dialog factory used across all apps in this project.

The author / institution / license / email block is built into the
default text so each app only has to supply its own short description.
"""
from __future__ import annotations

from PyQt5.QtWidgets import QMessageBox, QWidget


# Author block — single source of truth for every app's About dialog.
AUTHOR_NAME = "Honglun Yuan"
AUTHOR_EMAIL = "yuanhonglun@hotmail.com"
AUTHOR_INSTITUTION = "Hainan University"
LICENSE_TEXT = (
    "BSD 3-Clause with Non-Commercial Clause<br>"
    "Free for academic and non-commercial use. "
    "Commercial use requires written permission from the author."
)
COPYRIGHT_YEARS = "2025-2026"


def show_about_dialog(
    parent: QWidget | None,
    *,
    app_name: str,
    version: str,
    description: str,
    title: str | None = None,
) -> None:
    """Show the standard About dialog with author + license + copyright.

    Parameters
    ----------
    app_name : str
        Display name (e.g. "ASFAM Processor").
    version : str
        Release version string (typically ``metabo_core.__version__``).
    description : str
        One-paragraph description of what the app does.
    title : str, optional
        Override window title (defaults to ``f"About {app_name}"``).
    """
    body = (
        f"<h3>{app_name} v{version}</h3>"
        f"<p>{description}</p>"
        f"<p><b>Developer:</b> {AUTHOR_NAME}<br>"
        f"<b>Email:</b> {AUTHOR_EMAIL}</p>"
        f"<p><b>License:</b> {LICENSE_TEXT}</p>"
        f"<p>&copy; {COPYRIGHT_YEARS} {AUTHOR_NAME}, {AUTHOR_INSTITUTION}</p>"
    )
    QMessageBox.about(parent, title or f"About {app_name}", body)
