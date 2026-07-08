"""Theme constants exist and the stylesheet uses them."""
from __future__ import annotations

from metabo_gui.theme import (
    ION_PALETTE,
    PLOT_REF_RED,
    STYLESHEET,
    THEME_BG,
    THEME_BLUE,
    THEME_BORDER,
    THEME_GRID,
    THEME_LIGHT,
    THEME_SELECT,
)


def test_theme_blue_is_referenced_in_stylesheet():
    assert THEME_BLUE in STYLESHEET
    assert THEME_LIGHT in STYLESHEET
    assert THEME_BORDER in STYLESHEET


def test_theme_palette_is_hex():
    for color in (
        THEME_BLUE, THEME_LIGHT, THEME_BG, THEME_SELECT,
        THEME_BORDER, THEME_GRID, PLOT_REF_RED,
    ):
        assert color.startswith("#")
        assert len(color) == 7


def test_ion_palette_is_non_empty():
    assert isinstance(ION_PALETTE, tuple)
    assert len(ION_PALETTE) >= 4
    for color in ION_PALETTE:
        assert color.startswith("#")
        assert len(color) == 7
