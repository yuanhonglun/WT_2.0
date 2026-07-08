"""Boundary test: metabo_core must not import any app or GUI code."""
import re
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[1] / "metabo_core"

FORBIDDEN_PATTERNS = (
    r"\bfrom\s+asfam\b",
    r"\bimport\s+asfam\b",
    r"\bfrom\s+dda\b",
    r"\bimport\s+dda\b",
    r"\bfrom\s+gcms\b",
    r"\bimport\s+gcms\b",
    r"\bPyQt5\b",
    r"apps\.asfam_processor",
    r"apps\.dda_processor",
    r"apps\.gcms_processor",
)


def test_metabo_core_does_not_import_app_or_gui_code():
    offenders = []
    for path in CORE_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, text):
                offenders.append((str(path), pattern))
    assert offenders == [], f"metabo_core boundary violations: {offenders}"


def test_gcms_does_not_use_lc_ms1_finder():
    """spec §4.2/§9: LC MS1 mass-slice finder 仅供 LC 用, GC-MS 不得引用。"""
    gcms_root = Path(__file__).resolve().parents[3] / "apps" / "gcms_processor"
    if not gcms_root.exists():
        import pytest
        pytest.skip("gcms_processor not present")
    offenders = []
    for py in gcms_root.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="ignore")
        if "find_lc_ms1_features" in text or "lc_ms1_features" in text:
            offenders.append(str(py))
    assert not offenders, f"GC-MS must not use LC MS1 finder: {offenders}"
