"""Shared helpers for the Fig. 4 / S2.1b / S2.1e / S2.3 scripts of the 2026-10 revision.

Palette: the Okabe-Ito subset mandated for the revision (light fill =
0.25 * colour + 0.75 * white, full-colour edge). ``paper_palette.py`` (written
by the figure-style work package) is imported when present so every figure of
the revision shares one definition; otherwise the same names are defined here.

Also: matplotlib style (Arial 8 pt, 1 pt lines, 10 pt bold panel letters),
features.csv reader (``comment="#"``, ``is_duplicate`` filter, high-score flag),
spectrum string parser and a small ``update_numbers`` helper that merges keys
into ``analysis/rep3_full/numbers_fig4.json`` (flat dict key -> {value, source,
note}).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
REVISION_DIR = SCRIPT_DIR.parent
MANUSCRIPT_DIR = REVISION_DIR.parent
REPO_DIR = MANUSCRIPT_DIR.parent

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - depends on a sibling work package
    import paper_palette as _pp  # type: ignore
    WT_BLUE, DDA_ORANGE, MS2ONLY_PURPLE = _pp.BLUE, _pp.ORANGE, _pp.PURPLE
    SKY_BLUE, VERMILION = _pp.SKY, _pp.VERMILION

    def light(hex_colour: str, frac: float = 0.25) -> str:
        """``frac`` * colour + (1 - frac) * white (paper_palette.mix)."""
        return _pp.mix(hex_colour, frac)
except Exception:
    WT_BLUE = "#0072B2"        # ASFAMProcessor / WT 2.0 / MS1-detected
    DDA_ORANGE = "#E69F00"     # MS-DIAL / DDA
    MS2ONLY_PURPLE = "#CC79A7"  # MS2-only
    SKY_BLUE = "#56B4E9"
    VERMILION = "#D55E00"

    def light(hex_colour: str, frac: float = 0.25) -> str:
        """``frac`` * colour + (1 - frac) * white, as a hex string."""
        h = hex_colour.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        mix = [int(round(frac * c + (1 - frac) * 255)) for c in (r, g, b)]
        return "#{:02X}{:02X}{:02X}".format(*mix)

MS1_BLUE = WT_BLUE
GRAY = "#8EA1B3"
GRAY_LIGHT = "#CFD7DF"
TEXT = "#231815"
LW = 1.0


def tint(hex_colour: str, frac: float) -> str:
    """``frac`` of the colour mixed with white (1.0 = full colour)."""
    return light(hex_colour, frac)


#: Four tints of one hue for the n_support strata 1 / 2 / 3 / >=4 (dark = more votes).
SUPPORT_LEVELS = ["1", "2", "3", ">=4"]
SUPPORT_FRACS = [0.35, 0.55, 0.78, 1.0]


def support_colours(base: str) -> dict:
    return {lvl: tint(base, f) for lvl, f in zip(SUPPORT_LEVELS, SUPPORT_FRACS)}


def support_bin(n) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return ""
    return ">=4" if n >= 4 else str(n)


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": GRAY,
        "axes.linewidth": LW,
        "axes.titlesize": 8,
        "axes.titleweight": "normal",
        "axes.labelsize": 8,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.major.width": LW, "ytick.major.width": LW,
        "legend.fontsize": 8, "legend.frameon": False,
        "lines.linewidth": LW, "patch.linewidth": LW,
        "savefig.transparent": False, "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def add_label(fig, ax, lab: str, dx: float = 0.030, dy: float = 0.012) -> None:
    """Bold 10 pt panel letter, pushed left of a title that overflows its axes."""
    bb = ax.get_position()
    x = bb.x0 - dx
    title = ax.title
    if title.get_text():
        renderer = fig.canvas.get_renderer()
        extent = title.get_window_extent(renderer=renderer)
        title_x0 = extent.transformed(fig.transFigure.inverted()).x0
        x = max(0.004, min(x, title_x0 - 0.026))
    fig.text(x, bb.y1 + dy, lab, fontsize=10, fontweight="bold",
             fontfamily="Arial", color=TEXT, va="bottom", ha="left")


def despine(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def save_all(fig, stem: Path, dpi: int = 300) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), format=ext, dpi=dpi)


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
HIGH_SCORE = 1.4
MIN_MATCHED = 3


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def read_features(path: Path, retained_only: bool = True) -> pd.DataFrame:
    """features.csv of a v1.4 run: comment lines skipped, booleans parsed,
    ``high_score`` = total_score >= 1.4 and n_matched >= 3, ``name_norm`` =
    annotation name without the "Putative: " prefix, lower-cased."""
    # The export starts with "# key=value" header lines. They are skipped by
    # line count, NOT with ``comment="#"``: a library name can contain "#"
    # (ascr#17 in this data set) and ``comment`` would truncate that row at the
    # name, turning a retained, high-score MS1 row into a half-empty one.
    # scripts/features_io.py is the revision's shared reader; fall back to the
    # same logic inline when it is absent.
    try:
        from features_io import read_features as _read_raw
        df = _read_raw(path, retained_only=False)
    except Exception:
        n_header = 0
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("#"):
                    n_header += 1
                else:
                    break
        df = pd.read_csv(path, skiprows=n_header, low_memory=False)
    df["is_duplicate"] = truthy(df["is_duplicate"])
    for c in ("precursor_mz", "mz", "rt", "alignment_mz", "mean_height", "height",
              "total_score", "n_matched", "score", "ms1_window_mz", "window_delta_mda",
              "inferred_mz", "nl_n_support"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["n_fragments"] = pd.to_numeric(df["n_fragments"], errors="coerce").fillna(0).astype(int)
    df["duplicate_type"] = df["duplicate_type"].fillna("").astype(str).replace("nan", "")
    for c in ("mz_source", "mz_confidence", "ms2_only_tier", "name", "formula",
              "detection_source", "signal_type", "alignment_segment"):
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).replace("nan", "")
    df["high_score"] = df["total_score"].ge(HIGH_SCORE) & df["n_matched"].ge(MIN_MATCHED)
    df["name_norm"] = df["name"].map(norm_name)
    if retained_only:
        df = df[~df["is_duplicate"]].copy()
    return df


def norm_name(name) -> str:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = str(name).strip()
    for prefix in ("Putative: ", "putative: ", "low score: "):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s.strip().lower()


def parse_spectrum(s):
    """'mz:int mz:int ...' -> (mz, intensity) float64 arrays (positive intensities)."""
    if not isinstance(s, str) or not s.strip():
        return np.empty(0), np.empty(0)
    pairs = [p for p in s.split() if ":" in p]
    if not pairs:
        return np.empty(0), np.empty(0)
    mz = np.empty(len(pairs)); it = np.empty(len(pairs))
    for i, p in enumerate(pairs):
        a, b = p.split(":", 1)
        try:
            mz[i], it[i] = float(a), float(b)
        except ValueError:
            mz[i], it[i] = np.nan, np.nan
    m = np.isfinite(mz) & np.isfinite(it) & (it > 0)
    return mz[m], it[m]


def update_numbers(path: Path, entries: dict, source: str) -> None:
    """Merge ``entries`` (key -> value, or key -> {value, note}) into the flat
    numbers JSON at ``path``; every record carries ``source``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    for key, val in entries.items():
        if isinstance(val, dict) and "value" in val:
            rec = {"value": _plain(val["value"]), "source": val.get("source", source),
                   "note": val.get("note", "")}
        else:
            rec = {"value": _plain(val), "source": source, "note": ""}
        data[key] = rec
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _plain(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def pct(x: float, nd: int = 1) -> str:
    return f"{x * 100:.{nd}f}%"
