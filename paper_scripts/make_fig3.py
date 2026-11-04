"""Generate Fig. 3 (Development and optimization of ASFAM) as SVG / PDF / PNG.

Revision 261019 edition of manuscript/fig 3/make_fig3.py.  Numbers are
unchanged; only the palette, the comparison qualifier and the output location
differ:
  * palette: DDA orange, SWATH purple, HR-MRM blue (paper_palette); the
    channel-count series in panels e/f is one series and is drawn in blue.
  * the figure compares acquisition modes processed with identical MS-DIAL
    settings, so that qualifier appears on the mode axis (a, b) and in the
    mode legends (c, d), and legend_fig3.txt carries the caption title line.

Layout: 2 rows x 3 cols, panels a-f.  Arial 8 pt body, 10 pt bold panel
labels, 1 pt lines, 7.5 x 5.0 in canvas.

Usage:
    python make_fig3.py [--data-dir DIR] [--out-dir DIR]
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import paper_palette as pp  # noqa: E402

DEFAULT_DATA_DIR = HERE.parents[1] / "fig 3"
DEFAULT_OUT_DIR = HERE.parent / "figures" / "fig3"
INK = pp.INK

pp.apply_rcparams({
    "legend.fontsize": 7,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "patch.edgecolor": INK,
})

# (fill, stroke) pairs: DDA orange, SWATH purple, HR-MRM (ASFAM / WT 2.0) blue
PALETTE = {
    "DDA": pp.pair(pp.METHOD["DDA"]),
    "SWATH": pp.pair(pp.METHOD["SWATH"]),
    "HR-MRM": pp.pair(pp.METHOD["HR-MRM"]),
}
ORDER = ["DDA", "SWATH", "HR-MRM"]
MODE_AXIS_LABEL = "Acquisition mode (identical MS-DIAL processing)"
MODE_LEGEND_TITLE = "Acquisition mode\n(identical MS-DIAL processing)"

LEGEND = """Fig. 3 legend notes (revision 261019; paste into the manuscript)

Title: Acquisition modes compared under identical MS-DIAL processing.

All three modes (segmented DDA, SWATH and HR-MRM) used identical precursor segments and were processed with the same MS-DIAL settings, so the differences reflect acquisition rather than software.
Panel c shows single counts (counts, not means): each library candidate scored in all three modes is counted once per mode according to whether its replicate-mean Total score is >= 1.4 or < 1.4.
Panels a, b, d-f: numbers unchanged from the submitted version; panels e and f use one series (HR-MRM channels per cycle) and are drawn in a single colour.
"""


# ---------------- data ----------------
def load_methods_compare(data_dir: Path):
    wb = openpyxl.load_workbook(data_dir / "Methods_Compare_20260430.xlsx", data_only=True)
    return {s: list(wb[s].iter_rows(values_only=True)) for s in wb.sheetnames}


def load_n_summary(data_dir: Path):
    wb = openpyxl.load_workbook(data_dir / "1-n汇总.xlsx", data_only=True)
    rows = list(wb["总结果汇总"].iter_rows(values_only=True))
    sig, qual = {}, {}
    for r in rows[2:7]:
        if r[0]:
            sig[r[0]] = [v for v in r[1:4] if isinstance(v, (int, float))]
    for r in rows[10:15]:
        if r[0]:
            qual[r[0]] = [v for v in r[1:4] if isinstance(v, (int, float))]
    return sig, qual


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def remove_outliers_iqr(arr, k=1.5):
    """Standard Tukey rule: drop values outside [Q1 - k*IQR, Q3 + k*IQR]."""
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 4:
        return a
    q1, q3 = np.percentile(a, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return a[(a >= lo) & (a <= hi)]


# ---------------- helpers ----------------
def style_axes(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(1.0)
        sp.set_color(INK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=1.0, length=3, colors=INK)


def panel_label(ax, letter, x=-0.15, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left", color=INK)


def violin_3mode(ax, data_dict, ylabel):
    arrays = [np.asarray(data_dict[k], dtype=float) for k in ORDER]
    parts = ax.violinplot(arrays, positions=[1, 2, 3], widths=0.78,
                          showmeans=False, showmedians=False, showextrema=False)
    for body, mode in zip(parts["bodies"], ORDER):
        fill, stroke = PALETTE[mode]
        body.set_facecolor(fill)
        body.set_edgecolor(stroke)
        body.set_linewidth(1.0)
        body.set_alpha(1.0)
    ax.boxplot(arrays, positions=[1, 2, 3], widths=0.18,
               patch_artist=True, showfliers=False,
               medianprops=dict(color=INK, linewidth=1.0),
               whiskerprops=dict(color=INK, linewidth=1.0),
               capprops=dict(color=INK, linewidth=1.0),
               boxprops=dict(facecolor="white", edgecolor=INK, linewidth=1.0))
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(ORDER)
    ax.set_xlabel(MODE_AXIS_LABEL)
    ax.set_ylabel(ylabel)
    style_axes(ax)


# ---------------- panels ----------------
def panel_a(ax, sheets):
    rows = sheets["峰高分布"][1:]
    cols = list(zip(*rows))
    dda   = [v for c in cols[0:3] for v in c if is_num(v) and v > 0]
    asfam = [v for c in cols[3:6] for v in c if is_num(v) and v > 0]
    dia   = [v for c in cols[6:9] for v in c if is_num(v) and v > 0]
    data = {"DDA": np.log10(dda), "SWATH": np.log10(dia), "HR-MRM": np.log10(asfam)}
    violin_3mode(ax, data, ylabel=r"Peak height (log$_{10}$)")
    ax.set_ylim(2, 8)
    panel_label(ax, "a")


def panel_b(ax, sheets):
    """CV across replicates, with outliers removed via Tukey IQR rule per mode."""
    rows = sheets["CV分布"][1:]
    raw = {
        "DDA":   [r[0] for r in rows if is_num(r[0]) and r[0] >= 0],
        "SWATH": [r[2] for r in rows if is_num(r[2]) and r[2] >= 0],
        "HR-MRM": [r[1] for r in rows if is_num(r[1]) and r[1] >= 0],
    }
    cleaned = {k: remove_outliers_iqr(v) for k, v in raw.items()}
    violin_3mode(ax, cleaned, ylabel="CV across replicates")
    ymax = max(np.percentile(cleaned[k], 99) for k in ORDER)
    ax.set_ylim(0, ymax * 1.05)
    panel_label(ax, "b")


def panel_c(ax, sheets):
    """High vs below-high library-candidate count per mode (single counts)."""
    rows = sheets["Total score"][1:]
    high = {k: 0 for k in ORDER}
    low  = {k: 0 for k in ORDER}
    thr = 1.4
    for r in rows:
        dda   = [v for v in r[0:3] if is_num(v)]
        asfam = [v for v in r[3:6] if is_num(v)]
        dia   = [v for v in r[6:9] if is_num(v)]
        if not (dda and asfam and dia):
            continue
        for label, vals in (("DDA", dda), ("SWATH", dia), ("HR-MRM", asfam)):
            (high if statistics.mean(vals) >= thr else low)[label] += 1

    bw = 0.25
    cluster_x = [0, 1.4]
    offsets = [-bw, 0, bw]
    for off, mode in zip(offsets, ORDER):
        fill, stroke = PALETTE[mode]
        vals = [high[mode], low[mode]]
        ax.bar([cluster_x[0] + off, cluster_x[1] + off], vals, bw,
               color=fill, edgecolor=stroke, linewidth=1.0, label=mode)
        for xi, v in zip([cluster_x[0] + off, cluster_x[1] + off], vals):
            ax.text(xi, v + max(low.values()) * 0.018, f"{v}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(cluster_x)
    ax.set_xticklabels([u"Score ≥ 1.4", "Score < 1.4"])
    ax.set_ylabel("Library candidates (n)")
    leg = ax.legend(frameon=False, loc="upper left", ncol=1, title=MODE_LEGEND_TITLE,
                    handlelength=1.2, handleheight=0.9, borderpad=0.2,
                    handletextpad=0.4, labelspacing=0.25, title_fontsize=7, alignment="left")
    leg.get_title().set_color(INK)
    style_axes(ax)
    ax.set_ylim(0, max(low.values()) * 1.80)     # head-room for the titled legend
    panel_label(ax, "c")


def panel_d(ax, sheets):
    """Per-internal-standard Total score, mean +/- SD, 8 standards x 3 modes."""
    rows = sheets["IS_Total score"][1:]
    short_names = {
        "L-Leucine-5,5,5-D3": "Leu-d3",
        "D-Phenylalanine-d5": "Phe-d5",
        "Hippuric Acid-d5": "Hippuric-d5",
        "L-Tryptophan-d5": "Trp-d5",
        "Testosterone-16,16,17-D3": "Testos-d3",
        "[Phe (13C9,15N) ]-Phe": "Phe-13C9,15N",
        "Glycochenodeoxycholic acid_D4": "GCDCA-d4",
        "1-stearoyl-d35-sn-glycero-3-phosphocholine": "PC18:0-d35",
    }
    labels = []
    means = {k: [] for k in ORDER}
    sds   = {k: [] for k in ORDER}
    raw   = {k: [] for k in ORDER}
    for r in rows:
        if r[0] is None:
            continue
        name = r[1]
        labels.append(short_names.get(name, str(name)[:12]))
        dda   = [v for v in r[3:6] if is_num(v)]
        asfam = [v for v in r[6:9] if is_num(v)]
        dia   = [v for v in r[9:12] if is_num(v)]
        for label, vals in (("DDA", dda), ("SWATH", dia), ("HR-MRM", asfam)):
            means[label].append(statistics.mean(vals) if vals else 0.0)
            sds[label].append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
            raw[label].append(vals)
    n = len(labels)
    x = np.arange(n)
    bw = 0.27
    for off, mode in zip([-bw, 0, bw], ORDER):
        fill, stroke = PALETTE[mode]
        ax.bar(x + off, means[mode], bw, color=fill, edgecolor=stroke,
               linewidth=1.0, label=mode,
               yerr=sds[mode],
               error_kw=dict(elinewidth=1.0, capsize=2, ecolor=INK))
        for xi, values in enumerate(raw[mode]):
            jitter = np.linspace(-0.045, 0.045, len(values)) if values else np.array([])
            ax.scatter(xi + off + jitter, values, s=8, facecolor="white",
                       edgecolor=stroke, linewidth=0.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Total score")
    ax.set_ylim(0, 2.45)
    leg = ax.legend(frameon=False, loc="upper center", ncol=3,
                    bbox_to_anchor=(0.5, 1.02), title="Acquisition mode (identical MS-DIAL processing)",
                    handlelength=1.2, handleheight=0.9, borderpad=0.0,
                    columnspacing=1.0, handletextpad=0.35, title_fontsize=6.5)
    leg.get_title().set_color(INK)
    style_axes(ax)
    panel_label(ax, "d")


def panel_n_bar(ax, summary, label_y, panel_letter, drop_keys=("1-100",)):
    keys_all = ["1-20", "1-30", "1-40", "1-50", "1-100"]
    keys = [k for k in keys_all if k not in drop_keys]
    means = np.array([statistics.mean(summary[k]) for k in keys])
    sds   = np.array([statistics.pstdev(summary[k]) if len(summary[k]) > 1 else 0.0
                      for k in keys])
    x = np.arange(len(keys))
    fill, stroke = PALETTE["HR-MRM"]          # one series -> one colour
    ax.bar(x, means, 0.6, color=fill, edgecolor=stroke, linewidth=1.0,
           yerr=sds, error_kw=dict(elinewidth=1.0, capsize=2.5, ecolor=INK))
    for xi, key in zip(x, keys):
        values = np.asarray(summary[key], dtype=float)
        jitter = np.linspace(-0.10, 0.10, len(values))
        ax.scatter(xi + jitter, values, s=12, facecolor="white", edgecolor=stroke,
                   linewidth=1.0, zorder=4)
    ymax_axis = (means + sds).max() * 1.30
    label_y_pos = means + sds + ymax_axis * 0.03
    for xi, m, ly in zip(x, means, label_y_pos):
        ax.text(xi, ly, f"{m:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([k.split("-")[1] for k in keys])
    ax.set_xlabel("HR-MRM channels per cycle (n)")
    ax.set_ylabel(label_y)
    style_axes(ax)
    ax.set_ylim(0, ymax_axis)
    panel_label(ax, panel_letter)


# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="folder holding Methods_Compare_20260430.xlsx and 1-n汇总.xlsx")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sheets = load_methods_compare(data_dir)
    sig, qual = load_n_summary(data_dir)

    fig = plt.figure(figsize=(7.5, 5.0))
    gs = fig.add_gridspec(2, 3, left=0.075, right=0.985, top=0.95, bottom=0.12,
                          hspace=0.70, wspace=0.38)
    panel_a(fig.add_subplot(gs[0, 0]), sheets)
    panel_b(fig.add_subplot(gs[0, 1]), sheets)
    panel_c(fig.add_subplot(gs[0, 2]), sheets)
    panel_d(fig.add_subplot(gs[1, 0]), sheets)
    panel_n_bar(fig.add_subplot(gs[1, 1]), sig,  "Detected features (n)", "e")
    panel_n_bar(fig.add_subplot(gs[1, 2]), qual, "High-scoring annotations (n)", "f")

    fig.savefig(out_dir / "Fig3.svg", format="svg")
    fig.savefig(out_dir / "Fig3.pdf", format="pdf")
    fig.savefig(out_dir / "Fig3.png", format="png", dpi=300)
    (out_dir / "legend_fig3.txt").write_text(LEGEND, encoding="utf-8")
    print("[OK] Fig3.svg, Fig3.pdf, Fig3.png, legend_fig3.txt ->", out_dir)


if __name__ == "__main__":
    main()
