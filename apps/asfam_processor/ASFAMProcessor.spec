# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for ASFAMProcessor (metabo-platform multi-app layout).

Adapted from the legacy single-package ASFAMProcessor spec. Differences:
  - Sources now span three packages: ``asfam`` (this app),
    ``metabo_core`` and ``metabo_gui`` (shared at packages/*).
  - Icon resource lives in ``metabo_gui/resources/icon.png`` rather than
    inside the app package.
  - Hidden imports for our own packages are collected automatically via
    ``collect_submodules`` so adding new modules doesn't silently break
    the frozen build.

The pymzml OBO directory rule is preserved: pymzml looks for the OBO
files in ``os.path.dirname(sys.executable)/obo`` when frozen, so the
folder must sit next to ``ASFAMProcessor.exe`` in dist/.
"""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# Layout: this spec lives at apps/asfam_processor/ASFAMProcessor.spec
APP_DIR = Path(SPEC).resolve().parent           # apps/asfam_processor
REPO_ROOT = APP_DIR.parent.parent                # repo root
CORE_SRC = REPO_ROOT / "packages" / "metabo_core"
GUI_SRC = REPO_ROOT / "packages" / "metabo_gui"

import pymzml  # noqa: E402
PYMZML_DIR = Path(pymzml.__file__).parent

a = Analysis(
    [str(APP_DIR / "asfam" / "gui" / "app.py")],
    pathex=[str(APP_DIR), str(CORE_SRC), str(GUI_SRC)],
    binaries=[],
    datas=[
        # Shared platform icon — used by main_window's app_icon_path()
        (str(GUI_SRC / "metabo_gui" / "resources" / "icon.png"),
         os.path.join("metabo_gui", "resources")),
        # pymzml OBO files. Frozen pymzml resolves the OBO dir relative
        # to ``sys.executable``, so they must end up next to the exe.
        (str(PYMZML_DIR / "obo"), "obo"),
    ],
    hiddenimports=(
        # Auto-collect every submodule of our three first-party packages.
        collect_submodules("asfam")
        + collect_submodules("metabo_core")
        + collect_submodules("metabo_gui")
        + [
            # Scientific stack
            "numpy",
            "scipy", "scipy.signal", "scipy.optimize", "scipy.stats",
            "scipy.spatial",
            "pandas",
            # Native extension; metabo_core.utils.memlog degrades to a warning
            # without it, so a missing bundle shows up as absent [mem] lines.
            "psutil",
            "matplotlib",
            "matplotlib.backends.backend_qt5agg",
            "matplotlib.backends.backend_agg",
            # pymzml + lxml
            "pymzml", "pymzml.run", "pymzml.spec",
            "lxml", "lxml.etree",
            # PyQt5
            "PyQt5", "PyQt5.QtCore", "PyQt5.QtGui", "PyQt5.QtWidgets",
            "PyQt5.sip",
            # multiprocessing support for frozen exe
            "multiprocessing",
            "multiprocessing.pool",
            "multiprocessing.process",
            "multiprocessing.spawn",
            # urllib bits some deps lazily import
            "urllib.request", "urllib.parse", "urllib.error",
        ]
    ),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        # Heavy third-party packages not used by ASFAMProcessor
        "torch", "torchvision", "torchaudio",
        "tensorflow", "keras", "transformers",
        "sympy", "IPython", "notebook", "jupyter",
        "numba", "llvmlite",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ASFAMProcessor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=str(GUI_SRC / "metabo_gui" / "resources" / "icon.png"),
    # PyInstaller 6 defaults to placing data/binary files inside an
    # ``_internal`` subdirectory. Restore the legacy layout (everything
    # next to the exe) because frozen pymzml resolves its OBO directory
    # via ``os.path.dirname(sys.executable)/obo`` and would not find it
    # inside ``_internal/obo``.
    contents_directory=".",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ASFAMProcessor",
)
