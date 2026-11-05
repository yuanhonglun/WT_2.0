# -*- coding: utf-8 -*-
"""Fig. 5 - complete acquisition-plus-processing workflows compared:
WT 2.0 with ASFAMProcessor vs DDA and DIA with MS-DIAL.

Revision 261019 edition of manuscript/fig 5/make_fig5.py.  Palette and
caption title changed; the ASFAM inputs can be redirected with --asfam-dir so
the re-processed exports (runs/rep{1,2,3}_full/features.csv) can replace the
submitted files later.  The exports are read with features_io.read_features
(header lines skipped by count, never comment="#").  While the default (old) inputs are used the outputs
are named Fig5_preview_old_numbers.* so nobody mistakes them for final numbers.

All three replicates, all three methods gated at S/N >= 3 (MS-DIAL's native
floor).  Bar panels (a/b/c) show mean +/- population SD (ddof=0, one number
per replicate).  Distribution panels (d/e) pool the feature-level values from
all three replicates into a single violin.

Palette (paper_palette): WT 2.0 blue, DDA orange, DIA purple; the MS2-only
segment of the WT 2.0 bar in panel b is a medium tint of the same blue
(purple is already taken by DIA in that panel).  Arial 8 pt body, 10 pt bold
panel letters, 1 pt lines, A4 width x 122 mm.

Usage:
    python make_fig5.py [--asfam-dir DIR] [--msdial-dir DIR] [--out-dir DIR] [--out-name NAME]
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import paper_palette as pp  # noqa: E402
from features_io import read_features  # noqa: E402

REPS = (1, 2, 3)
# Neutral default; pass --asfam-dir explicitly when the exports live elsewhere.
DEFAULT_ASFAM_DIR = HERE.parents[1] / "fig 5" / "asfam"
DEFAULT_MSDIAL_DIR = HERE.parents[1] / "fig 5" / "msdial_dda_dia"
DEFAULT_OUT_DIR = HERE.parent / "figures" / "fig5"

# ---------- palette ----------
INK = pp.INK
WT_FILL,  WT_EDGE  = pp.pair(pp.METHOD["WT 2.0"])
DDA_FILL, DDA_EDGE = pp.pair(pp.METHOD["DDA"])
DIA_FILL, DIA_EDGE = pp.pair(pp.METHOD["DIA"])
MS2ONLY_FILL, MS2ONLY_EDGE = pp.medium(pp.METHOD["WT 2.0"]), WT_EDGE   # same hue, deeper tint
REFLINE = pp.GREY     # neutral grey for reference lines (never a series)
ERRKW = dict(elinewidth=1.0, capsize=2.5, ecolor=INK)

pp.apply_rcparams({
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})

LEGEND = """Fig. 5 legend notes (revision 261019; paste into the manuscript)

Title: Complete acquisition-plus-processing workflows compared (WT 2.0 with ASFAMProcessor vs DDA and DIA with MS-DIAL).

Each workflow is the full acquisition-plus-processing chain: WT 2.0 (ASFAM acquisition, 31 injections per sample) processed with ASFAMProcessor, and DDA or DIA (the same 31 injections) processed with MS-DIAL; differences therefore reflect the combination of acquisition and software, not acquisition alone (contrast Fig. 3). All three methods are gated at S/N >= 3. Bars show mean +/- SD of three replicates (n = 3; open symbols, individual replicates); violins pool feature-level values across the three replicates.
Panel b: the WT 2.0 bar is split into MS1-detected (light blue) and MS2-only (deeper blue) features.
NOTE: the current file is a PREVIEW drawn from the submitted (old) ASFAM exports; the WT 2.0 numbers will change once rep1-3 are re-processed with v1.4 (run with --asfam-dir pointing at runs/).
"""


# ---------- load data (all three replicates) ----------
def asfam_file(asfam_dir: Path, rep: int) -> Path:
    """Old submission file (asfam-<i>.csv) or a re-processed run folder."""
    for cand in (asfam_dir / f"asfam-{rep}.csv",
                 asfam_dir / f"rep{rep}_full" / "features.csv",
                 asfam_dir / f"rep{rep}" / "features.csv",
                 asfam_dir / f"rep{rep}_full.csv"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"no ASFAM export for rep {rep} under {asfam_dir}")


def not_duplicate(frame: pd.DataFrame) -> pd.Series:
    return frame["is_duplicate"].astype(str).str.strip().str.lower() != "true"


def load_asfam(asfam_dir: Path):
    """WT 2.0 / ASFAM non-redundant set. Returns per-rep counts + pooled distributions."""
    feat, ms1, ms2only, high, high_strict, tier_d = [], [], [], [], [], []
    sn, score = [], []
    for i in REPS:
        path = asfam_file(asfam_dir, i)
        a = read_features(path)
        a = a[a["feature_id"].astype(str).str.fullmatch(r"F\d+")]
        a = a[not_duplicate(a)]                          # is_duplicate == False only
        s = pd.to_numeric(a["sn_ratio"], errors="coerce")
        a = a[s >= 3]                                    # shared S/N >= 3 gate (MS-DIAL floor)
        # Headline counts use MS2-only tiers A to C only (>= 3 product ions); the
        # fragment-poor tier D rows (v1.4 exports) are reported once, not counted here.
        n_tier_d = 0
        if "ms2_only_tier" in a.columns:
            is_d = a["ms2_only_tier"].fillna("").astype(str).str.strip() == "D"
            n_tier_d = int(is_d.sum())
            a = a[~is_d]
        tier_d.append(n_tier_d)
        feat.append(len(a))
        ms1.append(int((a["signal_type"] == "ms1_detected").sum()))
        ms2only.append(int((a["signal_type"] == "ms2_only").sum()))
        ts = pd.to_numeric(a["total_score"], errors="coerce")
        high.append(int((ts >= 1.4).sum()))
        nm = pd.to_numeric(a["n_matched"], errors="coerce").fillna(0) if "n_matched" in a.columns else pd.Series(0, index=a.index)
        high_strict.append(int(((ts >= 1.4) & (nm >= 3)).sum()))
        sn.append(pd.to_numeric(a["sn_ratio"], errors="coerce").values)
        score.append(ts.dropna().values)
        print(f"  ASFAM rep{i}: {path.name}  features={len(a)} (tier D excluded: {n_tier_d})")
    ms1, ms2only = np.array(ms1), np.array(ms2only)
    return dict(feat=np.array(feat), ms1=ms1, ms2only=ms2only,
                ms2cont=ms1 + ms2only, high=np.array(high), high_strict=np.array(high_strict),
                tier_d=np.array(tier_d),
                sn=np.concatenate(sn), score=np.concatenate(score))


def load_msdial(msdial_dir: Path, tag: str):
    """DDA or DIA MS-DIAL export. Returns per-rep counts + pooled distributions."""
    feat, ms2cont, high, high_strict = [], [], [], []
    sn, score = [], []
    for i in REPS:
        m = pd.read_csv(msdial_dir / f"{tag}-{i}.csv", low_memory=False)
        s = pd.to_numeric(m["S/N"], errors="coerce")
        m = m[s >= 3]
        feat.append(len(m))
        msms = m["MSMS spectrum"].notna() & (m["MSMS spectrum"].astype(str).str.strip() != "")
        ms2cont.append(int(msms.sum()))
        ts = pd.to_numeric(m["Total score"], errors="coerce")
        high.append(int((ts >= 1.4).sum()))
        nm = pd.to_numeric(m["Matched peaks count"], errors="coerce").fillna(0)
        high_strict.append(int(((ts >= 1.4) & (nm >= 3)).sum()))
        sn.append(pd.to_numeric(m["S/N"], errors="coerce").values)
        score.append(ts.dropna().values)
    return dict(feat=np.array(feat), ms2cont=np.array(ms2cont), high=np.array(high),
                high_strict=np.array(high_strict),
                sn=np.concatenate(sn), score=np.concatenate(score))


def ms(arr):
    """mean, population SD (ddof=0) - matches Fig 3's statistics.pstdev."""
    return float(np.mean(arr)), float(np.std(arr))


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=1.0)


def panel_letter(ax, letter):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                xytext=(-34, 12), textcoords="offset points",
                fontsize=10, fontweight="bold",
                va="bottom", ha="left", color=INK, annotation_clip=False)


def bar_labels(ax, xs, ys, texts, errs=None, dy_frac=0.02):
    ymax = ax.get_ylim()[1]
    errs = [0] * len(xs) if errs is None else errs
    for x, y, e, t in zip(xs, ys, errs, texts):
        ax.text(x, y + e + dy_frac * ymax, t, ha="center", va="bottom",
                fontsize=8, color=INK)


def replicate_points(ax, position, values, edge, marker="o", offsets=(-0.10, 0.0, 0.10)):
    values = np.asarray(values, dtype=float)
    ax.scatter(np.asarray(offsets[:len(values)]) + position, values, s=12,
               marker=marker, facecolor="white", edgecolor=edge,
               linewidth=1.0, zorder=4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asfam-dir", default=str(DEFAULT_ASFAM_DIR),
                        help="folder with asfam-{1,2,3}.csv (submitted) or rep{1,2,3}_full/features.csv (re-processed)")
    parser.add_argument("--msdial-dir", default=str(DEFAULT_MSDIAL_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--out-name", default=None,
                        help="base file name; default Fig5_preview_old_numbers for the submitted inputs, Fig5 otherwise")
    args = parser.parse_args()
    asfam_dir, msdial_dir, out_dir = Path(args.asfam_dir), Path(args.msdial_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The submitted exports are named asfam-<i>.csv; re-processed runs come as
    # rep<i>_full/features.csv.  The former are a preview of superseded numbers.
    preview = all(asfam_file(asfam_dir, i).name == f"asfam-{i}.csv" for i in REPS)
    out_name = args.out_name or ("Fig5_preview_old_numbers" if preview else "Fig5")

    wt = load_asfam(asfam_dir)
    dda = load_msdial(msdial_dir, "DDA")
    dia = load_msdial(msdial_dir, "DIA")

    METHODS = ["WT 2.0", "DDA", "DIA"]
    FILL = [WT_FILL, DDA_FILL, DIA_FILL]
    EDGE = [WT_EDGE, DDA_EDGE, DIA_EDGE]

    def line(tag, d, keys):
        parts = [f"{k}={ms(d[k])[0]:,.0f}+/-{ms(d[k])[1]:.0f}" for k in keys]
        print(f"{tag:8s}", " | ".join(parts))

    line("WT 2.0", wt, ["feat", "ms1", "ms2only", "ms2cont", "high"])
    line("DDA", dda, ["feat", "ms2cont", "high"])
    line("DIA", dia, ["feat", "ms2cont", "high"])
    print("medians  S/N  WT=%.1f DDA=%.1f DIA=%.1f | score WT=%.2f DDA=%.2f DIA=%.2f"
          % (np.median(wt["sn"]), np.median(dda["sn"]), np.median(dia["sn"]),
             np.median(wt["score"]), np.median(dda["score"]), np.median(dia["score"])))

    # ---------- figure geometry: A4 width, partial height ----------
    MM = 1 / 25.4
    fig = plt.figure(figsize=(210 * MM, 122 * MM))
    gs = fig.add_gridspec(2, 6, left=0.075, right=0.985, top=0.915, bottom=0.115,
                          hspace=0.62, wspace=1.05)
    axA = fig.add_subplot(gs[0, 0:2])
    axB = fig.add_subplot(gs[0, 2:4])
    axC = fig.add_subplot(gs[0, 4:6])
    axD = fig.add_subplot(gs[1, 0:3])
    axE = fig.add_subplot(gs[1, 3:6])
    x = np.arange(3)
    BW = 0.62

    # ===================== Panel a: feature count =====================
    meanA = [ms(wt["feat"])[0], ms(dda["feat"])[0], ms(dia["feat"])[0]]
    sdA = [ms(wt["feat"])[1], ms(dda["feat"])[1], ms(dia["feat"])[1]]
    axA.bar(x, meanA, BW, color=FILL, edgecolor=EDGE, linewidth=1.0, yerr=sdA, error_kw=ERRKW)
    for xi, values, edge in zip(x, [wt["feat"], dda["feat"], dia["feat"]], EDGE):
        replicate_points(axA, xi, values, edge)
    axA.set_ylim(0, (np.array(meanA) + np.array(sdA)).max() * 1.20)
    bar_labels(axA, x, meanA, [f"{v:,.0f}" for v in meanA], errs=sdA)
    axA.set_ylabel("Number of features")
    axA.set_title("Feature count", fontsize=8, fontweight="bold", pad=4)
    axA.set_xticks(x); axA.set_xticklabels(METHODS)
    style_axis(axA); panel_letter(axA, "a")

    # ===================== Panel b: MS2-containing (WT split) =====================
    m_ms1, _ = ms(wt["ms1"])
    m_ms2o, _ = ms(wt["ms2only"])
    m_wt, sd_wt = ms(wt["ms2cont"])
    m_dda, sd_dda = ms(dda["ms2cont"])
    m_dia, sd_dia = ms(dia["ms2cont"])
    axB.bar(0, m_ms1, BW, color=WT_FILL, edgecolor=WT_EDGE, linewidth=1.0)
    axB.bar(0, m_ms2o, BW, bottom=m_ms1, color=MS2ONLY_FILL, edgecolor=MS2ONLY_EDGE,
            linewidth=1.0, yerr=sd_wt, error_kw=ERRKW)
    axB.bar(1, m_dda, BW, color=DDA_FILL, edgecolor=DDA_EDGE, linewidth=1.0, yerr=sd_dda, error_kw=ERRKW)
    axB.bar(2, m_dia, BW, color=DIA_FILL, edgecolor=DIA_EDGE, linewidth=1.0, yerr=sd_dia, error_kw=ERRKW)
    replicate_points(axB, -0.07, wt["ms1"], WT_EDGE, marker="o", offsets=(-0.08, 0.0, 0.08))
    replicate_points(axB, 0.07, wt["ms2cont"], MS2ONLY_EDGE, marker="D", offsets=(-0.08, 0.0, 0.08))
    replicate_points(axB, 1, dda["ms2cont"], DDA_EDGE)
    replicate_points(axB, 2, dia["ms2cont"], DIA_EDGE)
    valsB = [m_wt, m_dda, m_dia]
    sdsB = [sd_wt, sd_dda, sd_dia]
    axB.set_ylim(0, (np.array(valsB) + np.array(sdsB)).max() * 1.20)
    bar_labels(axB, x, valsB, [f"{v:,.0f}" for v in valsB], errs=sdsB)
    axB.text(0, m_ms1 + m_ms2o / 2, f"MS2-only\n{m_ms2o:,.0f}", ha="center", va="center",
             fontsize=7, color=INK)
    axB.text(0, m_ms1 / 2, f"MS1-\ndetected\n{m_ms1:,.0f}", ha="center", va="center",
             fontsize=7, linespacing=0.92, color=INK)
    axB.set_ylabel("Features with MS2")
    axB.set_title("MS2-containing features", fontsize=8, fontweight="bold", pad=4)
    axB.set_xticks(x); axB.set_xticklabels(METHODS)
    style_axis(axB); panel_letter(axB, "b")

    # ===================== Panel c: high-scoring annotations =====================
    # Drawn with the criterion used throughout the paper (Total score >= 1.4 and >= 3 matched
    # library peaks; 2026-11-04). The counts at Total score >= 1.4 alone (the criterion of the
    # submitted figure) stay in the numbers file as fig5_*_high_score.
    meanC = [ms(wt["high_strict"])[0], ms(dda["high_strict"])[0], ms(dia["high_strict"])[0]]
    sdC = [ms(wt["high_strict"])[1], ms(dda["high_strict"])[1], ms(dia["high_strict"])[1]]
    axC.bar(x, meanC, BW, color=FILL, edgecolor=EDGE, linewidth=1.0, yerr=sdC, error_kw=ERRKW)
    for xi, values, edge in zip(x, [wt["high_strict"], dda["high_strict"], dia["high_strict"]], EDGE):
        replicate_points(axC, xi, values, edge)
    axC.set_ylim(0, (np.array(meanC) + np.array(sdC)).max() * 1.20)
    bar_labels(axC, x, meanC, [f"{v:,.0f}" for v in meanC], errs=sdC)
    axC.set_ylabel("Annotations (score ≥ 1.4,\n≥ 3 matched peaks)")
    axC.set_title("High-scoring annotations", fontsize=8, fontweight="bold", pad=4)
    axC.set_xticks(x); axC.set_xticklabels(METHODS)
    style_axis(axC); panel_letter(axC, "c")

    # ===================== Panel d: S/N distribution (pooled, log) =====================
    sn_data = [np.log10(wt["sn"]), np.log10(dda["sn"]), np.log10(dia["sn"])]
    sn_med = [np.median(wt["sn"]), np.median(dda["sn"]), np.median(dia["sn"])]
    parts = axD.violinplot(sn_data, positions=x, widths=0.78, showextrema=False)
    for pc, fc, ec in zip(parts["bodies"], FILL, EDGE):
        pc.set_facecolor(fc); pc.set_edgecolor(ec); pc.set_linewidth(1.0); pc.set_alpha(1.0)
    axD.boxplot(sn_data, positions=x, widths=0.16, showfliers=False, patch_artist=True,
                medianprops=dict(color=INK, linewidth=1.0),
                boxprops=dict(facecolor="white", edgecolor=INK, linewidth=1.0),
                whiskerprops=dict(color=INK, linewidth=1.0),
                capprops=dict(color=INK, linewidth=1.0))
    for xi, m in zip(x, sn_med):
        axD.text(xi + 0.30, np.log10(m), f"{m:.0f}", ha="left", va="center", fontsize=8, color=INK)
    axD.set_ylim(0.0, 5.4)
    axD.yaxis.set_major_locator(FixedLocator([0, 1, 2, 3, 4, 5]))
    axD.yaxis.set_major_formatter(FixedFormatter(["1", "10", "100", "1k", "10k", "100k"]))
    axD.set_ylabel("Signal-to-noise ratio")
    axD.set_title("S/N distribution", fontsize=8, fontweight="bold", pad=4)
    axD.set_xticks(x)
    axD.set_xticklabels([f"{mm}\n(n={n:,})" for mm, n in
                         zip(METHODS, [len(wt["sn"]), len(dda["sn"]), len(dia["sn"])])])
    style_axis(axD); panel_letter(axD, "d")

    # ===================== Panel e: annotation score distribution (pooled) ==========
    sc_data = [wt["score"], dda["score"], dia["score"]]
    sc_med = [np.median(v) for v in sc_data]
    parts = axE.violinplot(sc_data, positions=x, widths=0.78, showextrema=False)
    for pc, fc, ec in zip(parts["bodies"], FILL, EDGE):
        pc.set_facecolor(fc); pc.set_edgecolor(ec); pc.set_linewidth(1.0); pc.set_alpha(1.0)
    axE.boxplot(sc_data, positions=x, widths=0.16, showfliers=False, patch_artist=True,
                medianprops=dict(color=INK, linewidth=1.0),
                boxprops=dict(facecolor="white", edgecolor=INK, linewidth=1.0),
                whiskerprops=dict(color=INK, linewidth=1.0),
                capprops=dict(color=INK, linewidth=1.0))
    axE.axhline(1.4, color=REFLINE, linestyle=(0, (4, 2)), linewidth=1.0, zorder=0.5)
    axE.text(2.55, 1.87, "high-score\nthreshold 1.4", ha="right", va="top",
             fontsize=8, color=INK, linespacing=1.0)
    for xi, m in zip(x, sc_med):
        axE.text(xi + 0.30, m, f"{m:.2f}", ha="left", va="center", fontsize=8, color=INK)
    axE.set_ylim(0.3, 2.05)
    axE.set_ylabel("Annotation total score")
    axE.set_title("Annotation score distribution", fontsize=8, fontweight="bold", pad=4)
    axE.set_xticks(x)
    axE.set_xticklabels([f"{mm}\n(n={n:,})" for mm, n in
                         zip(METHODS, [len(wt["score"]), len(dda["score"]), len(dia["score"])])])
    style_axis(axE); panel_letter(axE, "e")

    if preview:
        fig.text(0.985, 0.985, "PREVIEW - old ASFAM numbers (pre re-processing)", ha="right", va="top",
                 fontsize=7, color=pp.GREY, style="italic")

    for ext in ("svg", "pdf", "png"):
        fig.savefig(out_dir / f"{out_name}.{ext}", dpi=300, facecolor="white")
        print("wrote", out_dir / f"{out_name}.{ext}")

    # Numbers for the placeholders of the text (means of three replicates; medians of the
    # pooled feature-level values).  Written next to the figure; merge_numbers.py picks it up.
    import json
    src = f"scripts/make_fig5.py on {asfam_dir} (WT 2.0) and {msdial_dir} (MS-DIAL)"
    def mean_int(arr):
        return int(round(float(np.mean(arr))))
    numbers = {
        "fig5_wt_features": {"value": mean_int(wt["feat"]), "source": src,
                             "note": "retained rows, S/N >= 3, MS2-only tier D excluded; mean of three replicates"},
        "fig5_wt_ms1_detected": {"value": mean_int(wt["ms1"]), "source": src, "note": "mean of three replicates"},
        "fig5_wt_ms2only": {"value": mean_int(wt["ms2only"]), "source": src,
                            "note": "MS2-only tiers A to C (>= 3 product ions); mean of three replicates"},
        "fig5_wt_tier_d_excluded": {"value": mean_int(wt["tier_d"]), "source": src,
                                    "note": "fragment-poor MS2-only rows (tier D) above S/N 3 that are not counted; mean of three replicates"},
        "fig5_wt_high_score": {"value": mean_int(wt["high"]), "source": src, "note": "Total score >= 1.4 alone (criterion of the submitted Fig. 5c); mean of three replicates"},
        "fig5_wt_high_score_strict": {"value": mean_int(wt["high_strict"]), "source": src,
                                      "note": "Total score >= 1.4 and >= 3 matched library peaks (drawn in Fig. 5c); mean of three replicates"},
        "fig5_dda_high_score_strict": {"value": mean_int(dda["high_strict"]), "source": src, "note": "Total score >= 1.4 and >= 3 matched peaks (MS-DIAL 'Matched peaks count'); mean of three replicates"},
        "fig5_dia_high_score_strict": {"value": mean_int(dia["high_strict"]), "source": src, "note": "Total score >= 1.4 and >= 3 matched peaks (MS-DIAL 'Matched peaks count'); mean of three replicates"},
        "fig5_wt_median_sn": {"value": round(float(np.median(wt["sn"])), 1), "source": src, "note": "median over the pooled features of three replicates"},
        "fig5_wt_median_score": {"value": round(float(np.median(wt["score"])), 2), "source": src, "note": "median over the pooled annotated features of three replicates"},
        "fig5_dda_features": {"value": mean_int(dda["feat"]), "source": src, "note": "mean of three replicates"},
        "fig5_dia_features": {"value": mean_int(dia["feat"]), "source": src, "note": "mean of three replicates"},
        "fig5_dda_high_score": {"value": mean_int(dda["high"]), "source": src, "note": "Total score >= 1.4 alone"},
        "fig5_dia_high_score": {"value": mean_int(dia["high"]), "source": src, "note": "Total score >= 1.4 alone"},
        "fig5_wt_features_per_rep": {"value": [int(v) for v in wt["feat"]], "source": src, "note": "rep 1, 2, 3"},
        "fig5_wt_ms2only_per_rep": {"value": [int(v) for v in wt["ms2only"]], "source": src, "note": "rep 1, 2, 3"},
        "fig5_wt_high_score_strict_per_rep": {"value": [int(v) for v in wt["high_strict"]], "source": src, "note": "rep 1, 2, 3"},
        "fig5_preview": {"value": bool(preview), "source": src, "note": "True when drawn from the submitted (old) exports"},
    }
    with open(out_dir / f"numbers_{out_name.lower()}.json", "w", encoding="utf-8") as fh:
        json.dump(numbers, fh, indent=2, ensure_ascii=False)
    print("wrote", out_dir / f"numbers_{out_name.lower()}.json")
    legend = LEGEND if preview else "\n".join(l for l in LEGEND.splitlines() if not l.startswith("NOTE:")) + "\n"
    (out_dir / "legend_fig5.txt").write_text(legend, encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    main()
