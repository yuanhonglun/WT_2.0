"""Pytest fixtures for ASFAMProcessor tests."""
import os

# Pin the Qt binding to the one the app actually ships with (PyQt5) before
# matplotlib's Qt backend is first imported. Otherwise matplotlib may bind to
# PySide6 (if present), and constructing any widget that hosts a matplotlib
# navigation toolbar fails with a cross-binding TypeError. Run GUI tests
# headless as well.
os.environ.setdefault("QT_API", "PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Lock matplotlib's Qt binding to PyQt5 *now*, before any dependency can pull
# in PySide6. matplotlib's qt_compat prefers an already-imported binding over
# QT_API, so if PySide6.QtCore is imported first it wins and constructing any
# matplotlib-toolbar widget blows up with a cross-binding TypeError. Importing
# qt_compat here — while no Qt binding is loaded yet — makes QT_API decide.
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.backends.qt_compat  # noqa: F401  (import for side effect)

from pathlib import Path

import numpy as np
import pytest

from asfam.config import ProcessingConfig


@pytest.fixture
def demo_data_path():
    """Path to demo mzML files."""
    p = Path(r"D:\HNU\课题\WT 2.0\251219大改\现有代码\new_WT2\Demo_data")
    if not p.exists():
        pytest.skip("Demo data not available")
    return p


@pytest.fixture
def default_config():
    """Default processing configuration."""
    return ProcessingConfig()


@pytest.fixture
def synthetic_gaussian_eic():
    """Synthetic EIC with a single Gaussian peak at RT=5.0, width~0.2 min."""
    n_points = 894
    rt = np.linspace(0, 15, n_points)
    intensity = 5000.0 * np.exp(-0.5 * ((rt - 5.0) / 0.08) ** 2)
    noise = np.random.default_rng(42).normal(0, 20, n_points)
    intensity = np.maximum(intensity + noise, 0)
    return rt, intensity


@pytest.fixture
def synthetic_double_peak_eic():
    """Synthetic EIC with two partially overlapping Gaussian peaks."""
    n_points = 894
    rt = np.linspace(0, 15, n_points)
    peak1 = 3000.0 * np.exp(-0.5 * ((rt - 4.0) / 0.07) ** 2)
    peak2 = 2000.0 * np.exp(-0.5 * ((rt - 4.5) / 0.09) ** 2)
    noise = np.random.default_rng(42).normal(0, 15, n_points)
    intensity = np.maximum(peak1 + peak2 + noise, 0)
    return rt, intensity


@pytest.fixture
def synthetic_noise_eic():
    """Synthetic EIC with noise only, no real peaks."""
    n_points = 894
    rt = np.linspace(0, 15, n_points)
    intensity = np.abs(np.random.default_rng(42).normal(0, 30, n_points))
    return rt, intensity
