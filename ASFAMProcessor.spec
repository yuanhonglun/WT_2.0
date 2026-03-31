# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for ASFAMProcessor GUI."""

import sys
import os
from pathlib import Path

block_cipher = None

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(SPEC))

a = Analysis(
    [os.path.join(PROJECT_ROOT, 'asfam', 'gui', 'app.py')],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=[
        # Include icon resource
        (os.path.join(PROJECT_ROOT, 'asfam', 'resources', 'icon.png'),
         os.path.join('asfam', 'resources')),
        # Include pymzml OBO files — pymzml uses os.path.dirname(sys.executable)/obo/
        # when frozen, so we must place them relative to the exe directory
        (os.path.join(os.path.dirname(__import__('pymzml').__file__), 'obo'),
         'obo'),
    ],
    hiddenimports=[
        # Core scientific packages
        'numpy',
        'scipy',
        'scipy.signal',
        'scipy.optimize',
        'scipy.stats',
        'scipy.spatial',
        'pandas',
        'matplotlib',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.backends.backend_agg',
        # pymzml
        'pymzml',
        'pymzml.run',
        'pymzml.spec',
        # matchms
        'matchms',
        'matchms.similarity',
        # msbuddy
        'msbuddy',
        # lxml
        'lxml',
        'lxml.etree',
        # PyQt5
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'PyQt5.sip',
        # Our own package
        'asfam',
        'asfam.cli',
        'asfam.config',
        'asfam.constants',
        'asfam.models',
        'asfam.core',
        'asfam.core.eic',
        'asfam.core.peak_detection',
        'asfam.core.smoothing',
        'asfam.core.similarity',
        'asfam.core.clustering',
        'asfam.core.mass_utils',
        'asfam.io',
        'asfam.io.mzml_reader',
        'asfam.io.spectral_library',
        'asfam.io.result_export',
        'asfam.io.project_file',
        'asfam.pipeline',
        'asfam.pipeline.orchestrator',
        'asfam.pipeline.stage0_load',
        'asfam.pipeline.stage1_ms2_detection',
        'asfam.pipeline.stage1b_ms1_detection',
        'asfam.pipeline.stage2_ms1_assignment',
        'asfam.pipeline.stage2b_inference',
        'asfam.pipeline.stage3_merge_segments',
        'asfam.pipeline.stage4_isotope_dedup',
        'asfam.pipeline.stage5_adduct_dedup',
        'asfam.pipeline.stage5b_duplicate_detection',
        'asfam.pipeline.stage6_isf_detection',
        'asfam.pipeline.stage6b_annotation',
        'asfam.pipeline.stage7_alignment',
        'asfam.pipeline.stage8_export',
        'asfam.gui',
        'asfam.gui.main_window',
        'asfam.gui.setup_panel',
        'asfam.gui.progress_panel',
        'asfam.gui.feature_table',
        'asfam.gui.eic_plot',
        'asfam.gui.ms2_plot',
        'asfam.gui.scatter_plot',
        'asfam.gui.plot_toolbar',
        'asfam.gui.worker',
        'asfam.gui.i18n',
        # multiprocessing support for frozen exe
        'multiprocessing',
        'multiprocessing.pool',
        'multiprocessing.process',
        'multiprocessing.spawn',
        # stdlib modules needed by dependencies
        'urllib.request',
        'urllib.parse',
        'urllib.error',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        # Exclude large third-party packages not needed by ASFAMProcessor
        'torch',
        'torchvision',
        'torchaudio',
        'tensorflow',
        'keras',
        'transformers',
        'sympy',
        'IPython',
        'notebook',
        'jupyter',
        'numba',
        'llvmlite',
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
    name='ASFAMProcessor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=os.path.join(PROJECT_ROOT, 'asfam', 'resources', 'icon.png'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ASFAMProcessor',
)
