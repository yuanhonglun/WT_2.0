"""Static resources for metabo-platform Qt apps (icons, demo CSV, etc.)."""
from __future__ import annotations

from pathlib import Path


RESOURCES_DIR: Path = Path(__file__).resolve().parent

ICON_PATH: Path = RESOURCES_DIR / "icon.png"


def app_icon_path() -> Path:
    """Return the absolute path to the shared platform icon."""
    return ICON_PATH
