"""Color palette and Qt stylesheet shared by all metabo-platform apps.

The palette and stylesheet are deliberately kept as plain string constants
so they can be applied with a single ``QApplication.setStyleSheet`` (or
``QMainWindow.setStyleSheet``) call, without per-widget configuration.
"""
from __future__ import annotations

# Color palette
THEME_BLUE = "#2D6A9F"
THEME_LIGHT = "#E8F0F8"
THEME_BG = "#F5F7FA"
THEME_SELECT = "#5B9BD5"
THEME_BORDER = "#CCD6E0"
THEME_GRID = "#E0E0E0"

# Plot accent colors (used by EIC / spectrum widgets)
PLOT_REF_RED = "#E05050"
ION_PALETTE: tuple[str, ...] = (
    "#2D6A9F", "#E05050", "#21A67A", "#F59E0B",
    "#8B5CF6", "#FF6B6B", "#4ECDC4", "#95A5A6",
)


STYLESHEET = f"""
    QMainWindow {{
        background-color: {THEME_BG};
    }}
    QToolBar {{
        background-color: {THEME_BLUE};
        spacing: 6px;
        padding: 4px;
        border: none;
    }}
    QToolBar QToolButton {{
        color: white;
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 5px 12px;
        font-size: 13px;
        font-weight: bold;
    }}
    QToolBar QToolButton:hover {{
        background-color: rgba(255,255,255,0.2);
        border-color: rgba(255,255,255,0.3);
    }}
    QToolBar QToolButton:disabled {{
        color: rgba(255,255,255,0.4);
    }}
    QToolBar QComboBox {{
        background: white;
        border-radius: 3px;
        padding: 2px 6px;
        min-width: 100px;
    }}
    QToolBar QLabel {{
        color: white;
        font-weight: bold;
        font-size: 11px;
    }}
    QGroupBox {{
        font-weight: bold;
        border: 1px solid {THEME_BORDER};
        border-radius: 4px;
        margin-top: 8px;
        padding-top: 16px;
    }}
    QGroupBox::title {{
        color: {THEME_BLUE};
        subcontrol-origin: margin;
        left: 8px;
    }}
    QTableView {{
        gridline-color: {THEME_GRID};
        selection-background-color: {THEME_SELECT};
        selection-color: white;
        font-size: 12px;
    }}
    QTableView::item:selected {{
        background-color: {THEME_SELECT};
        color: white;
    }}
    QHeaderView::section {{
        background-color: {THEME_LIGHT};
        color: {THEME_BLUE};
        font-weight: bold;
        border: 1px solid #D0D8E0;
        padding: 3px;
    }}
    QProgressBar {{
        border: 1px solid {THEME_BORDER};
        border-radius: 4px;
        text-align: center;
        font-size: 11px;
    }}
    QProgressBar::chunk {{
        background-color: {THEME_BLUE};
        border-radius: 3px;
    }}
    QTabWidget::pane {{
        border: 1px solid {THEME_BORDER};
        border-radius: 4px;
    }}
    QTabBar::tab {{
        background: {THEME_LIGHT};
        border: 1px solid {THEME_BORDER};
        padding: 4px 10px;
        font-size: 10px;
    }}
    QTabBar::tab:selected {{
        background: white;
        color: {THEME_BLUE};
        font-weight: bold;
    }}
"""
