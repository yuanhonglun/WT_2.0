"""Shared GUI building blocks for metabo-platform Qt apps.

Submodules:
- ``theme``: color constants + global Qt stylesheet shared by all apps
- ``plot_toolbar``: minimal matplotlib NavigationToolbar (no XY coord display)
- ``canvas``: PanZoomMixin — scroll/drag/box-zoom/double-click for matplotlib
- ``spectrum_export``: write a single feature's spectrum to msp / mgf
- ``about``: factory for the standard About dialog
- ``logging_setup``: per-app crash log directory with N-file rotation
- ``project_autosave``: build the auto-save path next to the input data
"""
from metabo_core import __version__  # re-export for parity with apps

__all__ = ["__version__"]
