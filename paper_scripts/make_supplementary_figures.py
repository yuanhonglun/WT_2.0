"""Supplementary figures for revision 261019, numbered as in the submitted SI.

    sfig1  Sensitivity analysis of the ASFAMProcessor vs MS-DIAL benchmark
           (recovery classes for the lenient reference; height-stratified
           recall).  Redrawn here from the submitted match table
           (fig 4/match_table_lenient.csv) in the new palette ONLY as a
           preview: the numbers change once rep 3 is re-processed, so the
           output is named SFig1_preview_old_numbers.*.
    sfig2  Authentic-standard evidence (feruloyl putrescine, beta-carboline,
           3-formylindole, cholic acid, tauroursodeoxycholic acid, glycocholic
           acid).  There is no plotting source for this figure: it came from a
           third-party PDF.  The PDF-converted SVG (revision_260722) is
           recoloured here - measured traces/spectra black -> blue, authentic
           standard red -> orange - and rendered with headless Edge.
    sfig3  MS2-only validation examples: built by make_ms2only_validation_figure.py
           (needs the raw mzML files and the re-processed feature table); not
           handled here.
    sfig4  Cohort distributions of the nine non-bile-acid metabolites
           (port of revision_260722 "Supplementary Fig. 2"; NOT run now - the
           data change later).
    sfig5  ROC curves of the four additional classifiers
           (port of revision_260722 "Supplementary Fig. 3"; NOT run now).

The script is self-contained (the violin / ROC helpers that used to come from
analyze_cancer.py are included) and takes every input from --data-dir
arguments whose defaults point at the original manuscript folders.

Usage:
    python make_supplementary_figures.py --only sfig1,sfig2 [--out-root DIR]
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import itertools
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lxml import etree
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import paper_palette as pp  # noqa: E402

MANUSCRIPT = HERE.parents[1]
REVISION_260722 = MANUSCRIPT / "revision_260722"
DEFAULT_OUT_ROOT = HERE.parent / "figures"
INK = pp.INK
LW = 1.0
MM = 1 / 25.4
EDGE_CANDIDATES = [
    Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
]


def apply_style() -> None:
    pp.apply_rcparams({"legend.fontsize": 8})


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        fig.savefig(out_dir / f"{name}.{extension}",
                    dpi=600 if extension == "png" else None, facecolor="white")
    plt.close(fig)
    print(f"[{name}] -> {out_dir}")


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=LW)


def panel_letter(ax: plt.Axes, letter: str, dx: float = -34, dy: float = 12) -> None:
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy),
                textcoords="offset points", fontsize=10, fontweight="bold",
                va="bottom", ha="left", color=INK, annotation_clip=False)


# =========================================================================== #
# sfig1 - benchmark sensitivity analysis (preview in the new palette)
# =========================================================================== #
CLASS_ORDER = ["Primary", "Linked-isotope", "Linked-adduct", "Linked-isf",
               "Linked-spectral", "Missed"]


def make_sfig1(data_dir: Path, out_dir: Path) -> None:
    table = data_dir / "match_table_lenient.csv"
    m_len = pd.read_csv(table, low_memory=False)
    apply_style()
    fig = plt.figure(figsize=(7.5, 3.0), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.10, wspace=0.18)
    gs = fig.add_gridspec(1, 2)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    counts = m_len["match_class"].value_counts()
    classes = [c for c in CLASS_ORDER if counts.get(c, 0) > 0]
    vals = [int(counts[c]) for c in classes]
    total = sum(vals)
    x = np.arange(len(classes))
    fills = [pp.light(pp.RECOVERY_CLASS[c]) if c != "Missed" else pp.GREY_FILL for c in classes]
    edges = [pp.RECOVERY_CLASS[c] for c in classes]
    hatches = ["////" if c == "Missed" else "" for c in classes]   # CVD-proof mark for "Missed"
    bars = axA.bar(x, vals, color=fills, edgecolor=edges, linewidth=LW, hatch=hatches)
    for bar, v in zip(bars, vals):
        axA.text(bar.get_x() + bar.get_width() / 2, v + total * 0.015,
                 f"{v}\n{v / total * 100:.1f}%", ha="center", va="bottom",
                 fontsize=8, color=INK, linespacing=0.9)
    axA.set_xticks(x)
    axA.set_xticklabels([c.replace("Linked-", "L-") for c in classes], rotation=30, ha="right")
    axA.set_ylabel("MS-DIAL feature count")
    axA.set_ylim(0, max(vals) * 1.30)
    axA.set_title(f"Recovery - lenient ref (n = {total})", fontweight="normal")
    style_axis(axA)

    bands = [(0, 1e3, "<10\u00b3"), (1e3, 1e4, "10\u00b3\u201310\u2074"),
             (1e4, 1e5, "10\u2074\u201310\u2075"), (1e5, 1e6, "10\u2075\u201310\u2076"),
             (1e6, np.inf, ">10\u2076")]
    h = pd.to_numeric(m_len["ms_height"], errors="coerce").fillna(0).to_numpy()
    rec = (m_len["match_class"] != "Missed").to_numpy()
    labels, rates, ns = [], [], []
    for lo, hi, lab in bands:
        sel = (h >= lo) & (h < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        labels.append(lab); rates.append(float(rec[sel].mean() * 100)); ns.append(n)
    xb = np.arange(len(rates))
    fill, edge = pp.pair(pp.BLUE)
    bars = axB.bar(xb, rates, color=fill, edgecolor=edge, linewidth=LW)
    for bar, r, n in zip(bars, rates, ns):
        axB.text(bar.get_x() + bar.get_width() / 2, r + 1.5, f"{r:.0f}\nn={n}",
                 ha="center", va="bottom", fontsize=8, color=INK, linespacing=0.9)
    axB.set_xticks(xb); axB.set_xticklabels(labels, rotation=30, ha="right")
    axB.set_ylabel("Recall (%)"); axB.set_ylim(0, 118)
    axB.set_xlabel("MS-DIAL peak height")
    axB.set_title("Height-stratified recall", fontweight="normal")
    style_axis(axB)

    fig.canvas.draw()
    for ax, lab in zip([axA, axB], "ab"):
        bb = ax.get_position()
        fig.text(max(0.004, bb.x0 - 0.030), bb.y1 + 0.012, lab, fontsize=10,
                 fontweight="bold", color=INK, va="bottom", ha="left")
    fig.text(0.995, 0.008, "PREVIEW - old numbers (pre rep-3 re-processing)", ha="right",
             va="bottom", fontsize=7, color=pp.GREY, style="italic")
    save(fig, out_dir, "SFig1_preview_old_numbers")
    (out_dir / "legend_sfig1.txt").write_text(
        "Supplementary Fig. 1 legend note (revision 261019)\n\n"
        "Panels a and b show single counts (counts, not means). Colours: Primary blue, "
        "L-isotope orange, L-adduct purple, L-isf sky blue, L-spectral vermilion, Missed grey "
        "(paper_palette.RECOVERY_CLASS). The present file is a PREVIEW drawn from the submitted "
        "match table; numbers will be regenerated by gen_fig4.py after rep 3 is re-processed.\n",
        encoding="utf-8")


# =========================================================================== #
# sfig2 - authentic-standard evidence: recolour the PDF-converted SVG
# =========================================================================== #
SVG_NS = "http://www.w3.org/2000/svg"
NS = "{%s}" % SVG_NS
BLACKS = {"#171717", "#2c292a", "#231f20", "#030505", "#000000"}
RED = "#ed272d"


def find_edge() -> Path | None:
    """msedge from PATH first, then the two standard Program Files locations."""
    found = shutil.which("msedge")
    if found:
        return Path(found)
    for p in EDGE_CANDIDATES:
        if p.exists():
            return p
    return None


def _path_bbox(el):
    """Axis-aligned bbox of an M/L/H/V path after its matrix() transform."""
    m = re.search(r"matrix\(([^)]*)\)", el.get("transform", ""))
    a, b, c, d, e, f = map(float, m.group(1).split(",")) if m else (1, 0, 0, 1, 0, 0)
    pts, nums, cmd = [], [], None
    x = y = 0.0

    def flush():
        nonlocal x, y, nums
        if cmd in ("M", "L"):
            for i in range(0, len(nums) - 1, 2):
                x, y = nums[i], nums[i + 1]; pts.append((x, y))
        elif cmd == "H":
            for v in nums:
                x = v; pts.append((x, y))
        elif cmd == "V":
            for v in nums:
                y = v; pts.append((x, y))
        nums = []

    for t_cmd, t_num in re.findall(r"([MLHVZmlhvz])|(-?[\d.]+(?:e-?\d+)?)", el.get("d", "")):
        if t_cmd:
            flush(); cmd = t_cmd.upper()
        else:
            nums.append(float(t_num))
    flush()
    if not pts:
        return None
    xs = [a * px + c * py + e for px, py in pts]
    ys = [b * px + d * py + f for px, py in pts]
    return min(xs), min(ys), max(xs), max(ys)


def recolour_sfig2_svg(src: Path, dst: Path) -> dict[str, int]:
    """Data marks live inside clip-path groups: black -> blue (measured),
    red -> orange (authentic standard).  Axes, ticks and glyphs stay ink.
    One spectrum (panel e, upper) was exported without a clip group; its sticks
    are caught geometrically: un-clipped vertical black lines 6-53 pt tall
    (axes are 54.16 pt, ticks < 6 pt) that do not sit on an axis x position."""
    tree = etree.parse(str(src))
    root = tree.getroot()
    stats = {"black->blue": 0, "red->orange": 0, "red-outside-clip->orange": 0,
             "black-stick-outside-clip->blue": 0}
    clipped_elems = []      # keep the lxml proxies alive so that id() stays stable
    clipped = set()
    axis_x = set()
    for group in root.iter(NS + "g"):
        if not group.get("clip-path"):
            continue
        for el in group.iter(NS + "path"):
            clipped_elems.append(el)
            clipped.add(id(el))
            stroke = (el.get("stroke") or "").lower()
            if stroke in BLACKS:
                el.set("stroke", pp.BLUE); stats["black->blue"] += 1
            elif stroke == RED:
                el.set("stroke", pp.ORANGE); stats["red->orange"] += 1
    for el in root.iter(NS + "path"):          # a few red traces sit outside clip groups
        if (el.get("stroke") or "").lower() == RED:
            el.set("stroke", pp.ORANGE); stats["red-outside-clip->orange"] += 1
        if (el.get("fill") or "").lower() == RED:
            el.set("fill", pp.ORANGE); stats["red-outside-clip->orange"] += 1
    all_paths = list(root.iter(NS + "path"))    # pinned proxies, same reason
    unclipped_black = [el for el in all_paths
                       if id(el) not in clipped and (el.get("stroke") or "").lower() in BLACKS]
    boxes = {id(el): _path_bbox(el) for el in unclipped_black}
    for el in unclipped_black:                 # axis lines: tall verticals >= 54 pt
        bb = boxes[id(el)]
        if bb and bb[2] - bb[0] < 0.5 and bb[3] - bb[1] >= 53.5:
            axis_x.add(round(bb[0], 1))
    for el in unclipped_black:
        bb = boxes[id(el)]
        if not bb:
            continue
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if w < 0.5 and 6.0 < h < 53.5 and all(abs(bb[0] - ax) > 2.0 for ax in axis_x):
            el.set("stroke", pp.BLUE); stats["black-stick-outside-clip->blue"] += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(dst), xml_declaration=True, encoding="utf-8")
    return stats


def render_svg_with_edge(svg: Path, width_pt: float, height_pt: float, scale: int = 5) -> None:
    edge = find_edge()
    if edge is None:
        print("[warn] msedge.exe not found; PNG/PDF not rendered")
        return
    png = svg.with_suffix(".png")
    cmd = [str(edge), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
           "--no-default-browser-check", f"--screenshot={png}",
           f"--window-size={int(round(width_pt))},{int(round(height_pt))}",
           f"--force-device-scale-factor={scale}", svg.as_uri()]
    subprocess.run(cmd, check=False, capture_output=True, timeout=180)
    if png.exists():
        img = Image.open(png)
        img.save(png, dpi=(72 * scale, 72 * scale))
        print(f"[png] {png} {img.size[0]}x{img.size[1]} @ {72 * scale} dpi")
    tmp = Path(tempfile.mkdtemp(prefix="sfig_"))
    html = (f'<!doctype html><html><head><meta charset="utf-8"><style>'
            f'@page{{size:{width_pt}pt {height_pt}pt;margin:0}} html,body{{margin:0;padding:0}} '
            f'img{{display:block;width:{width_pt}pt;height:{height_pt}pt}}</style></head>'
            f'<body><img src="{svg.as_uri()}"></body></html>')
    wrapper = tmp / "print.html"
    wrapper.write_text(html, encoding="utf-8")
    pdf = svg.with_suffix(".pdf")
    cmd = [str(edge), "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
           "--no-pdf-header-footer", f"--print-to-pdf={pdf}", wrapper.as_uri()]
    subprocess.run(cmd, check=False, capture_output=True, timeout=180)
    print(f"[pdf] {pdf}" if pdf.exists() else "[warn] Edge print-to-pdf failed")
    shutil.rmtree(tmp, ignore_errors=True)


def make_sfig2(data_dir: Path, out_dir: Path) -> None:
    src = data_dir / "Supplementary Fig. 1.svg"      # revision_260722 numbering = current SFig 2
    dst = out_dir / "Supplementary Fig. 2.svg"
    stats = recolour_sfig2_svg(src, dst)
    print("[sfig2] recoloured:", stats)
    text = dst.read_text(encoding="utf-8")
    w = float(re.search(r'width="([\d.]+)"', text).group(1))
    h = float(re.search(r'height="([\d.]+)"', text).group(1))
    render_svg_with_edge(dst, w, h, scale=5)
    (out_dir / "legend_sfig2.txt").write_text(
        "Supplementary Fig. 2 legend note (revision 261019)\n\n"
        "Colours: blue, measured trace / spectrum from the WT 2.0 data; orange, authentic "
        "standard. Layout, numbers and labels unchanged from the submitted figure (the figure "
        "has no plotting source; the PDF-converted vector file was recoloured).\n",
        encoding="utf-8")


# =========================================================================== #
# sfig4 / sfig5 - ports of revision_260722 (cohort violins, ROC) - not run now
# =========================================================================== #
VIOLIN_SHEET = "\u5168\u90e8\u6570\u636e\u79bb\u7fa4\u503c\u68c0\u67e5-IQR\u65b9\u6cd5"
SFIG4_METABOLITES = [
    ("MN07051", "Lactic acid"), ("MN09828", "Azelaic acid"), ("MP01254", "Inosine"),
    ("MP01263", "Guanosine"), ("MP01302", "Aspartic acid"), ("MP01410", "Uric acid"),
    ("P3P0946", "Phenylalanine"), ("P3P1748", "Hypoxanthine"), ("P3P1751", "Xanthine"),
]
SFIG5_MODELS = [("LR", "Logistic regression"), ("NB", "Naive Bayes"),
                ("RF", "Random forest"), ("SVM", "Support vector machine")]
STAGES = ["HC", "BLD", "Stage I LC", "Stage III LC", "Stage IV LC"]
STAGE_SHORT = ["HC", "BLD", "I", "III", "IV"]
# HC blue, BLD orange, LC purple (three LC stages = three tints of the same purple)
STAGE_EDGE = [pp.COHORT["HC"], pp.COHORT["BLD"], pp.mix(pp.COHORT["LC"], 0.75),
              pp.COHORT["LC"], pp.dark(pp.COHORT["LC"])]
STAGE_FILL = [pp.light(pp.COHORT["HC"]), pp.light(pp.COHORT["BLD"]),
              pp.mix(pp.COHORT["LC"], 0.18), pp.light(pp.COHORT["LC"]), pp.mix(pp.COHORT["LC"], 0.40)]


def bh(values) -> np.ndarray:
    p = np.asarray(values, float)
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def dunn(groups: dict[str, np.ndarray]):
    from scipy import stats
    keys = list(groups)
    pooled = np.concatenate([groups[k] for k in keys])
    ranks_all = stats.rankdata(pooled)
    ranks, at = {}, 0
    for key in keys:
        ranks[key] = ranks_all[at:at + len(groups[key])]
        at += len(groups[key])
    _, counts = np.unique(pooled, return_counts=True)
    ties = float(((counts ** 3) - counts).sum())
    sigma2 = len(pooled) * (len(pooled) + 1) / 12 - ties / (12 * (len(pooled) - 1))
    result = []
    for a, b in itertools.combinations(keys, 2):
        z = abs(ranks[a].mean() - ranks[b].mean()) / math.sqrt(
            sigma2 * (1 / len(groups[a]) + 1 / len(groups[b])))
        result.append((a, b, 2 * stats.norm.sf(z)))
    return result


def stars(q: float) -> str:
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def clean_violin_groups(frame: pd.DataFrame, metabolite_id: str) -> dict[str, np.ndarray]:
    sub = frame[["Stage", metabolite_id, f"{metabolite_id}_IQR"]].copy()
    sub[metabolite_id] = pd.to_numeric(sub[metabolite_id], errors="coerce")
    sub = sub[sub[f"{metabolite_id}_IQR"].astype(str).str.lower().ne("true")]
    sub = sub.dropna(subset=[metabolite_id])
    return {stage: sub.loc[sub["Stage"].eq(stage), metabolite_id].to_numpy() for stage in STAGES}


def violin_panel(ax, frame, metabolite_id, title, rng, standard_confirmed=False) -> None:
    groups = clean_violin_groups(frame, metabolite_id)
    data = [groups[s] for s in STAGES]
    ns = [len(v) for v in data]
    ymax = max(v.max() for v in data)
    exponent = int(np.floor(np.log10(ymax)))
    scale = 10.0 ** exponent
    plotted = [v / scale for v in data]
    x = np.arange(5)
    parts = ax.violinplot(plotted, positions=x, widths=0.80, showextrema=False)
    for body, fill, edge in zip(parts["bodies"], STAGE_FILL, STAGE_EDGE):
        body.set_facecolor(fill); body.set_edgecolor(edge); body.set_linewidth(LW); body.set_alpha(1.0)
    for xi, values, edge in zip(x, plotted, STAGE_EDGE):
        ax.scatter(xi + rng.uniform(-0.16, 0.16, len(values)), values, s=1.6,
                   color=edge, alpha=0.45, linewidth=0, zorder=2)
    ax.boxplot(plotted, positions=x, widths=0.16, showfliers=False, patch_artist=True,
               medianprops={"color": INK, "linewidth": LW},
               boxprops={"facecolor": "white", "edgecolor": INK, "linewidth": LW},
               whiskerprops={"color": INK, "linewidth": LW},
               capprops={"color": INK, "linewidth": LW})
    result = dunn(groups)
    q_values = bh([p for _, _, p in result])
    hits = [(STAGES.index(a), STAGES.index(b), q)
            for (a, b, _), q in zip(result, q_values) if q < 0.05]
    hits.sort(key=lambda item: (abs(item[1] - item[0]), item[0]))
    top = max(v.max() for v in plotted)
    step = 0.115 * top
    for index, (i, j, q) in enumerate(hits):
        yy = top * 1.06 + index * step
        ax.plot([i, i, j, j], [yy, yy + step * 0.22, yy + step * 0.22, yy],
                color=INK, linewidth=LW, clip_on=False)
        ax.text((i + j) / 2, yy + step * 0.26, stars(float(q)), ha="center",
                va="bottom", fontsize=8, clip_on=False)
    ax.set_ylim(0, top * 1.06 + max(len(hits), 1) * step + 0.10 * top)
    ax.set_xticks(x, [f"{s}\n({n})" for s, n in zip(STAGE_SHORT, ns)])
    ax.set_ylabel(f"Intensity ($10^{{{exponent}}}$ cps)", labelpad=2)
    marker = " \u2020" if standard_confirmed else ""
    ax.set_title(f"{title}{marker}", fontsize=8, fontweight="bold", pad=4)
    style_axis(ax)


def roc_curve(y: np.ndarray, score: np.ndarray):
    order = np.argsort(-score, kind="mergesort")
    y, score = y[order], score[order]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    last = np.r_[np.flatnonzero(np.diff(score)), len(score) - 1]
    tpr = np.r_[0.0, tp[last] / tp[-1]]
    fpr = np.r_[0.0, fp[last] / fp[-1]]
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return fpr, tpr, float(integrate(tpr, fpr))


def make_sfig4(data_dir: Path, out_dir: Path) -> pd.DataFrame:
    data = pd.read_excel(data_dir / "violin_data.xlsx", sheet_name=VIOLIN_SHEET)
    missing = [i for i, _ in SFIG4_METABOLITES if i not in data.columns]
    if missing:
        raise RuntimeError(f"Cohort metabolites absent from the source table: {missing}")
    apply_style()
    fig = plt.figure(figsize=(210 * MM, 168 * MM))
    gs = fig.add_gridspec(3, 3, left=0.075, right=0.985, top=0.945, bottom=0.065,
                          hspace=0.62, wspace=0.34)
    rng = np.random.default_rng(20260723)
    for index, (metabolite_id, title) in enumerate(SFIG4_METABOLITES):
        ax = fig.add_subplot(gs[index // 3, index % 3])
        violin_panel(ax, data, metabolite_id, title, rng, standard_confirmed=False)
        panel_letter(ax, "abcdefghi"[index], dx=-32, dy=6)
    save(fig, out_dir, "Supplementary Fig. 4")
    records = []
    for metabolite_id, title in SFIG4_METABOLITES:
        frame = data[["Index", "Group", "Stage", metabolite_id, f"{metabolite_id}_IQR"]].copy()
        frame.columns = ["Index", "Group", "Stage", "Intensity_cps", "IQR_outlier_flag"]
        frame.insert(0, "Metabolite", title)
        frame.insert(0, "Transition_ID", metabolite_id)
        records.append(frame)
    source = pd.concat(records, ignore_index=True)
    source.to_csv(out_dir / "sfig4_source.csv", index=False)
    return source


def make_sfig5(data_dir: Path, out_dir: Path) -> pd.DataFrame:
    metrics = pd.read_csv(data_dir / "merge_metrics.details.txt", sep="\t")
    scores = {"Training": pd.read_csv(data_dir / "roc_train.txt", sep="\t"),
              "Validation": pd.read_csv(data_dir / "roc_valid.txt", sep="\t")}
    apply_style()
    fig = plt.figure(figsize=(180 * MM, 165 * MM))
    gs = fig.add_gridspec(2, 2, left=0.095, right=0.985, top=0.945, bottom=0.075,
                          hspace=0.40, wspace=0.28)
    records = []
    for index, (model, title) in enumerate(SFIG5_MODELS):
        ax = fig.add_subplot(gs[index // 2, index % 2])
        reference = metrics[metrics["Method"].eq(model)].set_index("DType")
        for tag, edge in (("Training", pp.BLUE), ("Validation", pp.ORANGE)):
            frame = scores[tag]
            fpr, tpr, auc = roc_curve(frame["Real"].to_numpy(float), frame[model].to_numpy(float))
            expected = float(reference.loc["train" if tag == "Training" else "valid", "AUC"])
            if abs(auc - expected) > 5e-3:
                raise RuntimeError(f"ROC source mismatch for {model} {tag}: {auc} vs {expected}")
            ax.plot(fpr, tpr, color=edge, linewidth=LW, label=f"{tag} (AUC = {expected:.3f})")
            block = frame[["Index", "Real", model]].copy()
            block.columns = ["Sample", "Class_label", "Model_score"]
            block.insert(0, "Subset", tag)
            block.insert(0, "Model", model)
            records.append(block)
        ax.plot([0, 1], [0, 1], color=pp.GREY, linestyle=(0, (4, 2)), linewidth=LW)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
        ax.set_title(title, fontweight="bold", pad=4)
        ax.legend(loc="lower right", handlelength=1.6, handletextpad=0.5)
        style_axis(ax)
        panel_letter(ax, "abcd"[index], dx=-34, dy=6)
    save(fig, out_dir, "Supplementary Fig. 5")
    source = pd.concat(records, ignore_index=True)
    source.to_csv(out_dir / "sfig5_source.csv", index=False)
    return source


# =========================================================================== #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="sfig1,sfig2",
                        help="comma list of sfig1, sfig2, sfig4, sfig5 (default: the two that are valid now)")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--sfig1-data-dir", default=str(MANUSCRIPT / "fig 4"))
    parser.add_argument("--sfig2-data-dir", default=str(REVISION_260722 / "supplementary_figures"))
    parser.add_argument("--cohort-data-dir", default=str(MANUSCRIPT / "fig 7" / "data"),
                        help="violin_data.xlsx, merge_metrics.details.txt, roc_train.txt, roc_valid.txt")
    args = parser.parse_args()
    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    out_root = Path(args.out_root)
    if "sfig1" in wanted:
        make_sfig1(Path(args.sfig1_data_dir), out_root / "sfig1")
    if "sfig2" in wanted:
        make_sfig2(Path(args.sfig2_data_dir), out_root / "sfig2")
    if "sfig4" in wanted:
        make_sfig4(Path(args.cohort_data_dir), out_root / "sfig4")
    if "sfig5" in wanted:
        make_sfig5(Path(args.cohort_data_dir), out_root / "sfig5")
    if "sfig3" in wanted:
        print("[sfig3] built by make_ms2only_validation_figure.py (raw mzML + re-processed table); skipped here")


if __name__ == "__main__":
    main()
