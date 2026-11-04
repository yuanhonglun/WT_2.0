"""Shared colour / typography constants for the revision_261019 figure set.

Okabe-Ito subset without any green-red pairing.  Every categorical mark uses a
light fill (0.25 * colour + 0.75 * white) and the full colour as edge / line.

    cohort groups       HC blue, BLD orange, LC purple
    acquisition methods DDA orange, DIA / SWATH purple, ASFAM / WT 2.0 blue
    signal types        MS1-detected blue, MS2-only purple
    auxiliary           sky blue, vermilion (only when a 4th / 5th hue is
                        unavoidable; never on the same figure as any green)

Typography: Arial 8 pt body, 10 pt bold panel letters, all lines 1 pt, ink
#231815.  Arial is resolved through matplotlib's font lookup; DejaVu Sans is
used only when Arial is absent from the machine.
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover - non-console stdout
    pass

# ---- Okabe-Ito subset ------------------------------------------------------
BLUE = "#0072B2"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"          # auxiliary
VERMILION = "#D55E00"    # auxiliary

# ---- neutrals --------------------------------------------------------------
INK = "#231815"          # text, axes, box edges in plots
NEUTRAL = "#8EA1B3"      # schematic box strokes / arrows (unchanged Fig 1 style)
RULE = "#CFD7DF"         # faint rules / grid in schematics
GREY = "#7F7F7F"         # reference lines (never a series)
GREY_FILL = "#E6E6E6"    # "missed" / not-applicable category fill
WHITE = "#FFFFFF"

# ---- semantic maps ---------------------------------------------------------
COHORT = {"HC": BLUE, "BLD": ORANGE, "LC": PURPLE}
METHOD = {
    "DDA": ORANGE,
    "DIA": PURPLE, "SWATH": PURPLE,
    "ASFAM": BLUE, "WT 2.0": BLUE, "WT2": BLUE, "HR-MRM": BLUE,
}
SIGNAL = {"ms1_detected": BLUE, "ms2_only": PURPLE}
# Fig 4 / Supplementary Fig 1 recovery classes (six categories: five hues + grey)
RECOVERY_CLASS = {
    "Primary": BLUE,
    "Linked-isotope": ORANGE,
    "Linked-adduct": PURPLE,
    "Linked-isf": SKY,
    "Linked-spectral": VERMILION,
    "Missed": GREY,          # draw with a hatch as well: grey vs purple is weak under deuteranopia
}
PALETTE_MAIN = {"blue": BLUE, "orange": ORANGE, "purple": PURPLE}
PALETTE_FULL = {**PALETTE_MAIN, "sky": SKY, "vermilion": VERMILION}


def hex_to_rgb01(colour: str) -> tuple[float, float, float]:
    colour = colour.lstrip("#")
    return tuple(int(colour[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def rgb01_to_hex(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(round(max(0.0, min(1.0, c)) * 255)) for c in rgb))


def mix(colour: str, frac: float, base: str = WHITE) -> str:
    """frac * colour + (1 - frac) * base, in sRGB."""
    c = hex_to_rgb01(colour)
    b = hex_to_rgb01(base)
    return rgb01_to_hex(tuple(frac * ci + (1.0 - frac) * bi for ci, bi in zip(c, b)))


def light(colour: str) -> str:
    """Light fill: 0.25 * colour + 0.75 * white."""
    return mix(colour, 0.25)


def medium(colour: str) -> str:
    """Medium tint (0.55 * colour + 0.45 * white) for a second segment of the same hue."""
    return mix(colour, 0.55)


def dark(colour: str) -> str:
    """Darker shade (0.75 * colour + 0.25 * black) for a secondary trace of the same hue."""
    return mix(colour, 0.75, "#000000")


def pair(colour: str) -> tuple[str, str]:
    """(fill, edge) pair for a categorical mark."""
    return light(colour), colour


def font_family() -> str:
    """'Arial' when matplotlib can find it, else 'DejaVu Sans'."""
    from matplotlib import font_manager
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        return "Arial"
    except (ValueError, RuntimeError):
        return "DejaVu Sans"


def rc_style(extra: dict | None = None) -> dict:
    """matplotlib rcParams for the manuscript style (Arial 8 pt, 1 pt lines, ink)."""
    family = font_family()
    params = {
        "font.family": family,
        "font.sans-serif": [family, "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7, "legend.frameon": False,
        "text.color": INK, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 1.0, "lines.linewidth": 1.0, "patch.linewidth": 1.0,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 300,
        # mathtext (e.g. log$_{10}$) in the same face, not DejaVu Sans
        "mathtext.fontset": "custom", "mathtext.rm": family,
        "mathtext.it": f"{family}:italic", "mathtext.bf": f"{family}:bold",
    }
    if extra:
        params.update(extra)
    return params


def apply_rcparams(extra: dict | None = None) -> None:
    import matplotlib
    matplotlib.rcParams.update(rc_style(extra))


if __name__ == "__main__":
    for name, colour in PALETTE_FULL.items():
        print(f"{name:10s} edge {colour}  fill {light(colour)}  medium {medium(colour)}  dark {dark(colour)}")
    print("font family:", font_family())
