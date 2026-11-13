"""S2.3: MS1-driven-only ablation vs the full ASFAMProcessor pipeline (rep-3, v1.4).

Reads the two runs of the same 31 rep-3 mzML files
(``runs/rep3_full`` and ``runs/rep3_ms1only`` = ``enable_ms2_driven_branch=false``),
their per-feature checkpoints (``_work/*.mfeat``), the stage ledgers already
computed by ``redundancy_ledger.py`` and the processing reports, and writes

  analysis/rep3_full/ablation_table.csv        one row per metric, columns ms1_only / full
  analysis/rep3_full/ablation_overlap_pairs.csv every retained MS1-only row with its full-run
                                                partner (RT <= 0.1 min and m/z <= 0.01 Da on the
                                                reported precursor, else on the quantitation ion,
                                                else precursor <= 0.05 Da)
  analysis/rep3_full/ablation_missing_hits.csv  MS1-only high-score annotations that the full run
                                                does not reproduce (input to annotation_outcome.py,
                                                which re-matches those spectra at wider windows)
  analysis/rep3_full/ablation_hit_overlap.csv   high-score annotation overlap by category / name
  figures/sfig_ablation/sfig_ablation_draft.{svg,pdf,png} + legend.txt

Figure layout (2026-11-12): panel a = candidates vs retained rows (left) and high-score
annotations per branch (right); panel b = redundancy-marking ledger; panel c = post hoc
family analysis of the retained MS2-only rows, read from ``<out>/ms2only_families.csv``
(written by ``redundancy_ledger.py``; nothing is recomputed here).

Every number here is a count on one sample (rep 3), not a mean.

Usage:  python ablation_table.py [--full DIR] [--ms1only DIR] [--out DIR] [--figdir DIR]
                                 [--families CSV] [--figure-only]
``--figure-only`` redraws the figure and legend.txt from the same inputs without rewriting
any analysis CSV or numbers file.
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig4_common import (  # noqa: E402
    GRAY, GRAY_LIGHT, LW, MS1_BLUE, MS2ONLY_PURPLE, REVISION_DIR, SKY_BLUE, TEXT, WT_BLUE,
    add_label, despine, light, read_features, save_all, setup_style, tint, update_numbers,
)
import matplotlib.pyplot as plt  # noqa: E402

RT_TOL = 0.1
MZ_TOL = 0.01
MZ_TOL_WIDE = 0.05


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def parse_report(path: Path) -> dict:
    """'--- Stage X: Name ---' blocks -> {'Stage X': {key: value}}."""
    out, cur = {}, None
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^--- (Stage [^:]+):", line.strip())
        if m:
            cur = m.group(1); out[cur] = {}; continue
        m = re.match(r"^\s+([a-z0-9_]+):\s*(.+)$", line)
        if m and cur:
            v = m.group(2).strip()
            try:
                v = float(v) if "." in v else int(v)
            except ValueError:
                pass
            out[cur][m.group(1)] = v
    return out


def candidate_counts(run: Path) -> dict:
    """Counts over the per-sample checkpoint (every candidate, before alignment)."""
    c = Counter(); n = 0; marks = Counter(); groups = {"isotope": set(), "adduct": set()}
    for mf in sorted((run / "_work").glob("*.mfeat")):
        with open(mf, "rb") as fh:
            payload = pickle.load(fh)
        for r in payload["features"]:
            n += 1
            c[(r.get("detection_source"), r.get("signal_type"), r.get("mz_source") or "")] += 1
            if r.get("is_duplicate"):
                marks[r.get("duplicate_type") or "unknown"] += 1
            if r.get("isotope_group_id") is not None:
                groups["isotope"].add(r["isotope_group_id"])
            if r.get("adduct_group_id") is not None:
                groups["adduct"].add(r["adduct_group_id"])
    return {"n": n, "by_source": c, "marks": marks,
            "isotope_groups": len(groups["isotope"]), "adduct_groups": len(groups["adduct"])}


def ledger_lookup(path: Path) -> dict:
    """ledger_stages.csv -> {stage prefix: row dict}."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        key = str(r["stage"]).split(" ")[0]
        out[key] = {k: (None if pd.isna(v) else v) for k, v in r.items()}
    return out


# --------------------------------------------------------------------------- #
# Overlap of the two retained tables
# --------------------------------------------------------------------------- #
def match_rows(ms1: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    full = full.reset_index(drop=True)
    f_rt = full["rt"].to_numpy(float)
    f_prec = full["precursor_mz"].to_numpy(float)
    f_quant = full["alignment_mz"].to_numpy(float)
    f_is_ms1 = (full["signal_type"] == "ms1_detected").to_numpy()
    order = np.argsort(f_rt)
    f_rt_sorted = f_rt[order]
    rows = []
    for _, r in ms1.iterrows():
        rt, prec, quant = float(r["rt"]), float(r["precursor_mz"]), float(r["alignment_mz"])
        lo = np.searchsorted(f_rt_sorted, rt - RT_TOL, side="left")
        hi = np.searchsorted(f_rt_sorted, rt + RT_TOL, side="right")
        idx = order[lo:hi]
        rec = {"ms1only_feature_id": r["feature_id"], "ms1only_mz": r["mz"], "ms1only_rt": rt,
               "ms1only_precursor_mz": prec, "ms1only_quant_mz": quant,
               "ms1only_height": r["height"], "ms1only_name": r["name"],
               "ms1only_name_norm": r["name_norm"], "ms1only_formula": r["formula"],
               "ms1only_total_score": r["total_score"], "ms1only_n_matched": r["n_matched"],
               "ms1only_high_score": bool(r["high_score"]), "match_key": "none",
               "full_feature_id": "", "full_signal_type": "", "full_detection_source": "",
               "full_mz_source": "", "full_precursor_mz": np.nan, "full_quant_mz": np.nan,
               "full_rt": np.nan, "full_height": np.nan, "full_name": "", "full_name_norm": "",
               "full_formula": "", "full_total_score": np.nan, "full_n_matched": np.nan,
               "full_high_score": False, "delta_precursor_da": np.nan, "delta_quant_da": np.nan,
               "delta_rt_min": np.nan}
        best = None
        if idx.size:
            drt = np.abs(f_rt[idx] - rt)
            d_prec = np.abs(f_prec[idx] - prec)
            d_quant = np.abs(f_quant[idx] - quant)
            tiers = [
                ("precursor_0.01", d_prec <= MZ_TOL, d_prec, MZ_TOL),
                ("quant_ion_0.01", (d_quant <= MZ_TOL) & f_is_ms1[idx], d_quant, MZ_TOL),
                ("precursor_0.05", d_prec <= MZ_TOL_WIDE, d_prec, MZ_TOL_WIDE),
            ]
            for key, ok, dmz, tol in tiers:
                if ok.any():
                    cand = np.where(ok)[0]
                    cost = dmz[cand] / tol + drt[cand] / RT_TOL
                    best = (key, idx[cand[int(np.argmin(cost))]])
                    break
        if best is not None:
            key, j = best
            fr = full.iloc[j]
            rec.update({
                "match_key": key, "full_feature_id": fr["feature_id"],
                "full_signal_type": fr["signal_type"], "full_detection_source": fr["detection_source"],
                "full_mz_source": fr["mz_source"], "full_precursor_mz": fr["precursor_mz"],
                "full_quant_mz": fr["alignment_mz"], "full_rt": fr["rt"], "full_height": fr["height"],
                "full_name": fr["name"], "full_name_norm": fr["name_norm"], "full_formula": fr["formula"],
                "full_total_score": fr["total_score"], "full_n_matched": fr["n_matched"],
                "full_high_score": bool(fr["high_score"]),
                "delta_precursor_da": float(fr["precursor_mz"]) - prec,
                "delta_quant_da": float(fr["alignment_mz"]) - quant,
                "delta_rt_min": float(fr["rt"]) - rt,
            })
        rows.append(rec)
    return pd.DataFrame(rows)


def hit_category(p: pd.Series) -> str:
    if p["match_key"] == "none":
        return "no_partner"
    if not p["full_high_score"]:
        return "partner_not_high_score"
    if p["full_name_norm"] == p["ms1only_name_norm"]:
        return "partner_same_name"
    if str(p["full_formula"]) and p["full_formula"] == p["ms1only_formula"]:
        return "partner_same_formula_other_name"
    return "partner_other_formula"


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def stacked_bar(ax, x, parts, width=0.6):
    """parts: list of (value, face, edge, hatch). Returns top."""
    bottom = 0
    for v, face, edge, hatch in parts:
        if v <= 0:
            continue
        ax.bar(x, v, width=width, bottom=bottom, color=face, edgecolor=edge, linewidth=LW, hatch=hatch)
        bottom += v
    return bottom


def panel_a_left(ax, F):
    """Candidates and retained rows per run, stacked by branch / signal class."""
    runs = [("MS1-driven\nonly", "ms1only"), ("Full\npipeline", "full")]
    styles = {
        "ms1_driven": (light(MS1_BLUE), MS1_BLUE, None, "MS1-detected, MS1-driven branch"),
        "ms2_driven_ms1": (light(SKY_BLUE), SKY_BLUE, None, "MS1-detected, MS2-driven branch"),
        "ms2_only_abc": (light(MS2ONLY_PURPLE), MS2ONLY_PURPLE, None, "MS2-only, tier A–C"),
        "ms2_only_d": ("white", MS2ONLY_PURPLE, "////", "MS2-only, tier D (<3 fragments)"),
    }
    xs, labels = [], []
    pos = 0
    for run_lab, key in runs:
        for stage in ("candidates", "retained"):
            parts = [(F[key][stage][k], *styles[k][:3]) for k in styles]
            top = stacked_bar(ax, pos, parts)
            ax.text(pos, top + 250, f"{int(top):,}", ha="center", va="bottom", fontsize=8, color=TEXT)
            xs.append(pos); labels.append("cand." if stage == "candidates" else "retained")
            pos += 1
        pos += 0.6
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    for i, (run_lab, _) in enumerate(runs):
        ax.text(i * 2.6 + 0.5, -0.20, run_lab, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color=TEXT, linespacing=0.95)
    ax.set_ylabel("Feature count")
    ymax = max(F[k]["candidates_total"] for k in ("ms1only", "full"))
    ax.set_ylim(0, ymax * 1.75)
    ax.set_title("Candidates vs retained rows")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=s[0], edgecolor=s[1], hatch=s[2], linewidth=LW)
               for s in styles.values()]
    ax.legend(handles, [s[3] for s in styles.values()], loc="upper left", fontsize=7,
              handlelength=1.0, handletextpad=0.5, labelspacing=0.3, borderaxespad=0.2)
    despine(ax)


def panel_a_right(ax, F):
    """High-score annotations (Total score >= 1.4 and >= 3 matched peaks) per branch."""
    bars = [
        ("MS1-driven", F["ms1only"]["hs_ms1_driven"], light(MS1_BLUE), MS1_BLUE, None),
        ("MS1-driven", F["full"]["hs_ms1_driven"], light(MS1_BLUE), MS1_BLUE, None),
        ("MS2-driven, MS1-det.", F["full"]["hs_ms2_driven_ms1"], light(SKY_BLUE), SKY_BLUE, None),
        ("MS2-only", F["full"]["hs_ms2_only"], light(MS2ONLY_PURPLE), MS2ONLY_PURPLE, None),
    ]
    xs = [0, 1.75, 3.0, 4.25]
    for x, (lab, v, face, edge, hatch) in zip(xs, bars):
        ax.bar(x, v, width=0.8, color=face, edgecolor=edge, linewidth=LW, hatch=hatch)
        ax.text(x, v + 8, f"{int(v)}", ha="center", va="bottom", fontsize=8, color=TEXT)
    ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in bars], fontsize=7, rotation=30, ha="right")
    ax.set_xlim(-0.6, 4.85)
    ax.text(0, -0.40, "MS1-driven\nonly", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=8, linespacing=0.95)
    ax.text(3.0, -0.40, f"Full pipeline\n(total {int(F['full']['hs_total'])})",
            transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, linespacing=0.95)
    ax.set_ylabel("High-score annotations")
    ax.set_ylim(0, max(b[1] for b in bars) * 1.25)
    ax.set_title("Library annotations per branch")
    despine(ax)


def panel_b(ax, F):
    """Redundancy ledger as a waterfall: candidates -> per-sample marks -> alignment marks -> retained."""
    steps = ["Candidates", "Folded at\nalignment", "Isotope", "Adduct", "Spectral", "ISF", "Retained"]
    runs = [("ms1only", SKY_BLUE, "MS1-driven only"), ("full", WT_BLUE, "Full pipeline")]
    w = 0.38
    ymax = max(F[k]["candidates_total"] for k in ("ms1only", "full"))
    for k, (key, col, lab) in enumerate(runs):
        L = F[key]["ledger"]
        total = L["candidates"]
        decs = [L["folded"], L["isotope"], L["adduct"], L["spectral"], L["isf"]]
        x = np.arange(len(steps)) + (k - 0.5) * w
        ax.bar(x[0], total, width=w, color=light(col), edgecolor=col, linewidth=LW, label=lab)
        run_total = total
        for i, d in enumerate(decs, start=1):
            ax.bar(x[i], d, width=w, bottom=run_total - d, color="white", edgecolor=col,
                   linewidth=LW, hatch="////")
            # labels: full run above its bar, MS1-only run below its bar
            if key == "full":
                ax.text(x[i], run_total + ymax * 0.015, f"−{int(d):,}", ha="center", va="bottom",
                        fontsize=6.5, color=col, rotation=90)
            else:
                ax.text(x[i], run_total - d - ymax * 0.015, f"−{int(d):,}", ha="center", va="top",
                        fontsize=6.5, color=col, rotation=90)
            run_total -= d
        ax.bar(x[-1], L["retained"], width=w, color=light(col), edgecolor=col, linewidth=LW)
        for xi, v in ((x[0], total), (x[-1], L["retained"])):
            ax.text(xi, v + ymax * 0.015, f"{int(v):,}", ha="center", va="bottom", fontsize=6.5,
                    color=col, rotation=90)
    ax.set_xticks(np.arange(len(steps))); ax.set_xticklabels(steps, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Rows")
    ax.set_ylim(0, ymax * 1.45)
    ax.set_title("Redundancy marking ledger")
    ax.legend(loc="upper right", handlelength=1.0, handletextpad=0.5, labelspacing=0.3, fontsize=7)
    despine(ax)


FAMILY_BINS = [("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4", 4, 4), ("5–9", 5, 9), ("≥10", 10, None)]


def family_stats(path: Path) -> dict:
    """Family-size distribution of the retained MS2-only rows (ms2only_families.csv, as written)."""
    fam = pd.read_csv(path)
    sizes = fam.groupby("family_id").size()
    stored = fam.groupby("family_id")["family_size"].first()
    if not (sizes == stored).all():
        raise SystemExit(f"{path}: family_size column disagrees with the family_id grouping")
    bins = []
    for lab, lo, hi in FAMILY_BINS:
        sel = sizes >= lo if hi is None else (sizes >= lo) & (sizes <= hi)
        bins.append((lab, int(sel.sum())))
    return {"n_rows": int(len(fam)), "n_families": int(sizes.size),
            "n_singletons": int((sizes == 1).sum()), "n_families_ge2": int((sizes >= 2).sum()),
            "n_rows_ge2": int(sizes[sizes >= 2].sum()), "largest": int(sizes.max()), "bins": bins}


def panel_c(ax, S):
    """Post hoc families of the retained MS2-only rows: number of families per family size."""
    labels = [b[0] for b in S["bins"]]
    counts = [b[1] for b in S["bins"]]
    xs = np.arange(len(labels))
    ax.bar(xs, counts, width=0.7, color=light(MS2ONLY_PURPLE), edgecolor=MS2ONLY_PURPLE, linewidth=LW)
    for x, v in zip(xs, counts):
        ax.text(x, v * 1.12, f"{v:,}", ha="center", va="bottom", fontsize=8, color=TEXT)
    ax.set_yscale("log")
    ax.set_ylim(10, max(counts) * 4.5)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_xlabel("Family size (rows)")
    ax.set_ylabel("Families (log scale)")
    ax.set_title("MS2-only families (post hoc)")
    note = (f"{S['n_rows']:,} rows in {S['n_families']:,} families\n"
            f"{S['n_singletons']:,} singletons\n"
            f"{S['n_rows_ge2']:,} rows in {S['n_families_ge2']:,} families of ≥ 2\n"
            f"largest family: {S['largest']} rows")
    ax.text(0.98, 0.97, note, transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color=TEXT, linespacing=1.25)
    despine(ax)


def make_figure(F, figdir: Path, families: dict):
    setup_style()
    fig = plt.figure(figsize=(7.5, 5.6), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.10, wspace=0.10, hspace=0.08)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0])
    ax_a1 = fig.add_subplot(gs[0, 0]); ax_a2 = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, 0]); ax_c = fig.add_subplot(gs[1, 1])
    panel_a_left(ax_a1, F); panel_a_right(ax_a2, F); panel_b(ax_b, F); panel_c(ax_c, families)
    fig.canvas.draw()
    # panel a spans the two upper sub-plots and carries one letter
    for ax, lab in ((ax_a1, "a"), (ax_b, "b"), (ax_c, "c")):
        add_label(fig, ax, lab)
    save_all(fig, figdir / "sfig_ablation_draft.svg")
    plt.close(fig)


LEGEND = """Supplementary Figure (draft) | Ablation of the MS2-driven branch on the rep-3 validation set (31 segments, one replicate, positive mode; ASFAMProcessor v{ver}).
Both runs process the same 31 mzML files with identical parameters; the "MS1-driven only" run sets enable_ms2_driven_branch = false, so stage 1 (MS2-driven detection) is skipped and the MS1-driven branch keeps every MS1 peak. All values are counts on this one sample (counts, not means).
(a) Left: candidate features (stages 1 + 1b, before redundancy marking and alignment; "cand.") and rows retained after redundancy marking and alignment ("retained"), stacked by how the row was detected: MS1-detected features from the MS1-driven branch (blue), MS1-detected features from the MS2-driven branch (sky blue), MS2-only features with >= 3 fragments (tiers A–C, purple) and MS2-only features with < 3 fragments (tier D, hatched). MS1-driven only: {c_ms1} candidates -> {r_ms1} retained; full pipeline: {c_full} candidates -> {r_full} retained ({r_full_ms1} MS1-detected, {r_full_ms2} MS2-only). Right: high-score library annotations (Total score >= 1.4 and >= 3 matched peaks, lib/lcms/pos.msp) per branch. MS1-driven only: {hs_ms1}; full pipeline: {hs_full} in total = {hs_full_ms1drv} (MS1-driven branch) + {hs_full_ms2drv} (MS2-driven, MS1-detected) + {hs_full_ms2only} (MS2-only). The MS1-driven-only run annotates more features than the full pipeline: in the full run an MS1-detected, MS2-driven feature reported its precursor as the intensity-weighted centroid of the whole 1-Da isolation window, which lies > 0.01 Da from the MS1-only run's ROI centroid for {frac_gt_001} of the {n_pairs_ms2drv} features present in both runs; the two branches also build the MS2 spectrum of such a feature differently. How much of the difference the 0.01 Da library window explains is quantified by re-matching those spectra at wider windows (annotation_outcome.py -> ablation_recovery.csv / ablation_window_scan.csv).
(b) Redundancy-marking ledger as a waterfall: candidates, minus candidates the stage-7 joiner folded into another row of the same sample (the same compound seen in two overlapping segments or by both branches; {fold_ms1} / {fold_full}), minus aligned rows carrying an isotope, adduct, spectral-duplicate or in-source-fragment mark (stages 4, 5, 5b, 6 within the sample, stage 7.2 across rows; {marks_ms1} / {marks_full} rows), equals retained rows ({r_ms1} / {r_full}). Before the fold the per-sample stages had marked {ps_ms1} / {ps_full} candidates (isotope {iso_ms1} / {iso_full}, adduct {add_ms1} / {add_full}, spectral {spec_ms1} / {spec_full}, ISF {isf_ms1} / {isf_full}); marked candidates folded into their representative are not counted twice. Nothing is deleted at any step; marked rows stay in features.csv with is_duplicate = True. Values are given as MS1-driven only / full pipeline.
(c) Post hoc family analysis of the {fam_rows} retained MS2-only rows of the full pipeline (rows within 0.05 min whose spectra have cosine > 0.7 or share at least three fragments are linked): number of families by family size. {fam_n} families, {fam_single} of them singletons; {fam_rows_ge2} rows lie in {fam_ge2} families of two or more and the largest family has {fam_largest} rows.
Data: analysis/rep3_full/ablation_table.csv; ledgers analysis/rep3_*/ledger_stages.csv; ms2only_families.csv (redundancy_ledger.py); processing_report.txt of each run.
"""


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--full", default=str(REVISION_DIR / "runs" / "rep3_full"))
    ap.add_argument("--ms1only", default=str(REVISION_DIR / "runs" / "rep3_ms1only"))
    ap.add_argument("--out", default=str(REVISION_DIR / "analysis" / "rep3_full"))
    ap.add_argument("--figdir", default=str(REVISION_DIR / "figures" / "sfig_ablation"))
    ap.add_argument("--families", default=None,
                    help="ms2only_families.csv of the full run (default: <out>/ms2only_families.csv)")
    ap.add_argument("--figure-only", action="store_true",
                    help="redraw the figure and legend.txt only; do not rewrite analysis CSVs or numbers")
    args = ap.parse_args()
    runs = {"full": Path(args.full), "ms1only": Path(args.ms1only)}
    out, figdir = Path(args.out), Path(args.figdir)
    out.mkdir(parents=True, exist_ok=True); figdir.mkdir(parents=True, exist_ok=True)
    write_tables = not args.figure_only
    families = family_stats(Path(args.families) if args.families else out / "ms2only_families.csv")

    feats, reports, cands, ledgers = {}, {}, {}, {}
    for key, run in runs.items():
        feats[key] = read_features(run / "features.csv")
        reports[key] = parse_report(run / "_debug" / "processing_report.txt")
        cands[key] = candidate_counts(run)
        ledgers[key] = ledger_lookup(REVISION_DIR / "analysis" / run.name / "ledger_stages.csv")
        print(f"[{key}] retained rows {len(feats[key]):,}, candidates {cands[key]['n']:,}")

    version = ""
    for line in (runs["full"] / "features.csv").open(encoding="utf-8", errors="replace"):
        if line.startswith("# version="):
            version = line.split("=", 1)[1].strip(); break

    F = {}
    table = []

    def add(section, metric, note=""):
        table.append({"section": section, "metric": metric,
                      "ms1_only": F["ms1only"].get(metric), "full": F["full"].get(metric), "note": note})

    for key in runs:
        df, rep, cc, L = feats[key], reports[key], cands[key], ledgers[key]
        bs = cc["by_source"]
        d = {}
        d["stage1_ms2_driven_candidates"] = rep.get("Stage 1", {}).get("total_features", 0)
        d["stage1b_ms1_driven_added"] = rep.get("Stage 1b", {}).get("ms1_driven_new", 0)
        d["candidates_total"] = cc["n"]
        d["cand_ms1_detected"] = sum(v for (ds, st, src), v in bs.items() if st == "ms1_detected")
        d["cand_ms1_detected_ms1_driven"] = sum(v for (ds, st, src), v in bs.items() if st == "ms1_detected" and ds == "ms1_driven")
        d["cand_ms1_detected_ms2_driven"] = sum(v for (ds, st, src), v in bs.items() if st == "ms1_detected" and ds == "ms2_driven")
        d["cand_ms2_only"] = sum(v for (ds, st, src), v in bs.items() if st == "ms2_only")
        s25 = rep.get("Stage 2.5", {})
        d["stage25_nl_consensus"] = s25.get("nl_assigned", 0)
        d["stage25_window_fallback"] = s25.get("window_fallback", 0)
        d["stage25_no_precursor"] = s25.get("no_precursor", 0)
        d["stage25_retained_fragment_poor"] = s25.get("retained_fragment_poor", 0)
        marks = cc["marks"]
        d["marked_isotope"] = marks.get("isotope", 0); d["marked_adduct"] = marks.get("adduct", 0)
        d["marked_spectral"] = marks.get("spectral", 0); d["marked_isf"] = marks.get("isf", 0)
        d["marked_total_per_sample"] = sum(marks.values())
        d["isotope_groups"] = cc["isotope_groups"]; d["adduct_groups"] = cc["adduct_groups"]
        # aligned-table marks straight from features.csv (every row); the ledger
        # CSV of redundancy_ledger.py, when present, is only cross-checked
        all_rows = read_features(runs[key] / "features.csv", retained_only=False)
        marked = all_rows[all_rows["is_duplicate"]]
        d["aligned_rows"] = len(all_rows)
        d["marked_in_alignment"] = len(marked)
        for t in ("isf", "isotope", "adduct", "spectral"):
            d[f"aligned_dup_{t}"] = int((marked["duplicate_type"] == t).sum())
        d["aligned_dup_other"] = int((~marked["duplicate_type"].isin(["isf", "isotope", "adduct", "spectral"])).sum())
        L7 = L.get("7", {})
        if L7.get("rows_all") is not None and int(L7["rows_all"]) != d["aligned_rows"]:
            print(f"  [{key}] note: ledger_stages.csv rows_all {int(L7['rows_all'])} != features.csv rows {d['aligned_rows']}")
        d["retained_rows"] = len(df)
        ms1 = df[df["signal_type"] == "ms1_detected"]; ms2 = df[df["signal_type"] == "ms2_only"]
        d["retained_ms1_detected"] = len(ms1)
        d["retained_ms1_detected_ms1_driven"] = int((ms1["detection_source"] == "ms1_driven").sum())
        d["retained_ms1_detected_ms2_driven"] = int((ms1["detection_source"] == "ms2_driven").sum())
        d["retained_ms2_only"] = len(ms2)
        for tier in "ABCD":
            d[f"retained_ms2_only_tier_{tier}"] = int((ms2["ms2_only_tier"] == tier).sum())
        d["retained_ms2_only_tier_ABC"] = int(ms2["ms2_only_tier"].isin(list("ABC")).sum())
        d["hs_total"] = int(df["high_score"].sum())
        d["hs_ms1_detected"] = int(ms1["high_score"].sum())
        d["hs_ms1_driven"] = int(ms1.loc[ms1["detection_source"] == "ms1_driven", "high_score"].sum())
        d["hs_ms2_driven_ms1"] = int(ms1.loc[ms1["detection_source"] == "ms2_driven", "high_score"].sum())
        d["hs_ms2_only"] = int(ms2["high_score"].sum())
        d["hs_ms2_only_tier_ABC"] = int(ms2.loc[ms2["ms2_only_tier"].isin(list("ABC")), "high_score"].sum())
        d["hs_ms2_only_tier_D"] = int(ms2.loc[ms2["ms2_only_tier"] == "D", "high_score"].sum())
        d["hs_unique_names"] = int(df.loc[df["high_score"], "name_norm"].nunique())
        d["candidates"] = {"ms1_driven": d["cand_ms1_detected_ms1_driven"],
                           "ms2_driven_ms1": d["cand_ms1_detected_ms2_driven"],
                           "ms2_only_abc": sum(v for (ds, st, src), v in bs.items() if st == "ms2_only") - 0,
                           "ms2_only_d": 0}
        # tier D among candidates is not stored; split MS2-only candidates by the retained A-C/D ratio proxy:
        # use n_fragments < 3 from the checkpoint instead (exact).
        d["candidates"]["ms2_only_abc"], d["candidates"]["ms2_only_d"] = _split_ms2only_candidates(runs[key])
        d["retained"] = {"ms1_driven": d["retained_ms1_detected_ms1_driven"],
                         "ms2_driven_ms1": d["retained_ms1_detected_ms2_driven"],
                         "ms2_only_abc": d["retained_ms2_only_tier_ABC"],
                         "ms2_only_d": d["retained_ms2_only_tier_D"]}
        # Stage 7 folds candidates the joiner judges to be the same compound of the
        # same sample (overlapping segments, MS2-driven + MS1-driven twins) into one
        # row; the marks that survive on the aligned rows are the aligned_dup_* counts.
        s7 = rep.get("Stage 7", {})
        d["folded_at_alignment"] = int(s7.get("conservation_n_collapsed_same_compound", 0) or 0)
        d["ledger"] = {"candidates": d["candidates_total"], "folded": d["folded_at_alignment"],
                       "isotope": d["aligned_dup_isotope"], "adduct": d["aligned_dup_adduct"],
                       "spectral": d["aligned_dup_spectral"], "isf": d["aligned_dup_isf"] + d["aligned_dup_other"],
                       "retained": d["retained_rows"]}
        # ledger balance: candidates - folded - marks on aligned rows == retained
        d["ledger_residual"] = (d["candidates_total"] - d["folded_at_alignment"]
                                - d["marked_in_alignment"] - d["retained_rows"])
        F[key] = d

    # ---- overlap of the two retained tables -------------------------------
    pairs = match_rows(feats["ms1only"], feats["full"])
    pairs["hit_category"] = pairs.apply(hit_category, axis=1)
    # second pass: do the unmatched MS1-only rows exist in the full run as MARKED rows?
    full_all = read_features(runs["full"] / "features.csv", retained_only=False)
    full_marked = full_all[full_all["is_duplicate"]].copy()
    unmatched = pairs[pairs["match_key"] == "none"]
    second = match_rows(feats["ms1only"][feats["ms1only"]["feature_id"].isin(unmatched["ms1only_feature_id"])],
                        full_marked)
    second = second[second["match_key"] != "none"].set_index("ms1only_feature_id")
    dup_type = full_marked.set_index("feature_id")["duplicate_type"]
    pairs["unmatched_partner_among_marked_rows"] = ""
    pairs["unmatched_partner_duplicate_type"] = ""
    sel = pairs["ms1only_feature_id"].isin(second.index)
    pairs.loc[sel, "unmatched_partner_among_marked_rows"] = pairs.loc[sel, "ms1only_feature_id"].map(second["full_feature_id"])
    pairs.loc[sel, "unmatched_partner_duplicate_type"] = pairs.loc[sel, "unmatched_partner_among_marked_rows"].map(dup_type).fillna("")
    if write_tables:
        pairs.to_csv(out / "ablation_overlap_pairs.csv", index=False)
    ov = {}
    ov["ms1only_rows"] = len(pairs)
    for k in ("precursor_0.01", "quant_ion_0.01", "precursor_0.05", "none"):
        ov[f"ms1only_rows_matched_{k}"] = int((pairs["match_key"] == k).sum())
    ov["ms1only_rows_unmatched_but_marked_row_in_full"] = int(sel.sum())
    for t, n in pairs.loc[sel, "unmatched_partner_duplicate_type"].value_counts().items():
        ov[f"ms1only_rows_unmatched_marked_as_{t or 'unknown'}"] = int(n)
    ov["ms1only_rows_absent_from_full"] = int((pairs["match_key"] == "none").sum() - sel.sum())
    matched = pairs[pairs["match_key"] != "none"]
    ov["ms1only_rows_matched_any"] = len(matched)
    ov["ms1only_rows_matched_unique_full_partners"] = int(matched["full_feature_id"].nunique())
    for st in ("ms1_detected", "ms2_only"):
        ov[f"partner_signal_{st}"] = int((matched["full_signal_type"] == st).sum())
    for ds in ("ms1_driven", "ms2_driven"):
        ov[f"partner_ms1_detected_{ds}"] = int(((matched["full_signal_type"] == "ms1_detected")
                                                 & (matched["full_detection_source"] == ds)).sum())
    ov["full_ms1_detected_rows_with_ms1only_partner"] = int(
        matched.loc[matched["full_signal_type"] == "ms1_detected", "full_feature_id"].nunique())
    ov["full_ms1_detected_rows"] = F["full"]["retained_ms1_detected"]
    # precursor delta for MS1-detected partners, by full detection_source
    for ds in ("ms1_driven", "ms2_driven"):
        sub = matched[(matched["full_signal_type"] == "ms1_detected") & (matched["full_detection_source"] == ds)]
        a = np.abs(sub["delta_precursor_da"].to_numpy(float))
        ov[f"pairs_{ds}_n"] = len(sub)
        if len(sub):
            ov[f"pairs_{ds}_abs_delta_precursor_p50_da"] = float(np.median(a))
            ov[f"pairs_{ds}_abs_delta_precursor_p90_da"] = float(np.percentile(a, 90))
            ov[f"pairs_{ds}_abs_delta_precursor_p95_da"] = float(np.percentile(a, 95))
            ov[f"pairs_{ds}_frac_delta_precursor_gt_0.01"] = float((a > 0.01).mean())
            ov[f"pairs_{ds}_frac_delta_precursor_gt_0.02"] = float((a > 0.02).mean())
            ov[f"pairs_{ds}_frac_delta_precursor_gt_0.05"] = float((a > 0.05).mean())
            q = np.abs(sub["delta_quant_da"].to_numpy(float))
            ov[f"pairs_{ds}_frac_delta_quant_ion_gt_0.01"] = float((q > 0.01).mean())
    # high-score hits of the MS1-only run and what the full run did with them
    hs = pairs[pairs["ms1only_high_score"]]
    ov["ms1only_high_score"] = len(hs)
    cat_counts = hs["hit_category"].value_counts()
    for c in ("partner_same_name", "partner_same_formula_other_name", "partner_other_formula",
              "partner_not_high_score", "no_partner"):
        ov[f"ms1only_hs_{c}"] = int(cat_counts.get(c, 0))
    hs_np = hs[hs["hit_category"] == "no_partner"]
    ov["ms1only_hs_no_partner_but_marked_row_in_full"] = int((hs_np["unmatched_partner_among_marked_rows"] != "").sum())
    for t, n in hs_np.loc[hs_np["unmatched_partner_among_marked_rows"] != "", "unmatched_partner_duplicate_type"].value_counts().items():
        ov[f"ms1only_hs_no_partner_marked_as_{t or 'unknown'}"] = int(n)
    for ds in ("ms1_driven", "ms2_driven"):
        sub = hs[(hs["full_signal_type"] == "ms1_detected") & (hs["full_detection_source"] == ds)]
        ov[f"ms1only_hs_partner_{ds}_n"] = len(sub)
        ov[f"ms1only_hs_partner_{ds}_same_name"] = int((sub["hit_category"] == "partner_same_name").sum())
        ov[f"ms1only_hs_partner_{ds}_not_high_score"] = int((sub["hit_category"] == "partner_not_high_score").sum())
        nh = sub[sub["hit_category"] == "partner_not_high_score"]
        if len(nh):
            a = np.abs(nh["delta_precursor_da"].to_numpy(float))
            ov[f"ms1only_hs_partner_{ds}_not_hs_frac_delta_gt_0.01"] = float((a > 0.01).mean())
    missing = hs[hs["hit_category"] != "partner_same_name"].copy()
    # attach the spectra for the re-matching script
    spec_full = feats["full"].set_index("feature_id")["ms2_spectrum"]
    spec_ms1 = feats["ms1only"].set_index("feature_id")["ms2_spectrum"]
    missing["ms1only_ms2_spectrum"] = missing["ms1only_feature_id"].map(spec_ms1)
    missing["full_ms2_spectrum"] = missing["full_feature_id"].map(spec_full).fillna("")
    if write_tables:
        missing.to_csv(out / "ablation_missing_hits.csv", index=False)
    # names
    names_ms1 = set(feats["ms1only"].loc[feats["ms1only"]["high_score"], "name_norm"]) - {""}
    names_full = set(feats["full"].loc[feats["full"]["high_score"], "name_norm"]) - {""}
    full_hs = feats["full"][feats["full"]["high_score"]]
    names_full_ms1 = set(full_hs.loc[full_hs["signal_type"] == "ms1_detected", "name_norm"]) - {""}
    names_full_ms2 = set(full_hs.loc[full_hs["signal_type"] == "ms2_only", "name_norm"]) - {""}
    ov["hs_names_ms1only"] = len(names_ms1); ov["hs_names_full"] = len(names_full)
    ov["hs_names_common"] = len(names_ms1 & names_full)
    ov["hs_names_only_ms1only"] = len(names_ms1 - names_full)
    ov["hs_names_only_full"] = len(names_full - names_ms1)
    ov["hs_names_only_full_from_ms2_only_rows"] = len((names_full - names_ms1) & names_full_ms2 - names_full_ms1)
    ov["hs_names_ms2_only_rows_total"] = len(names_full_ms2)
    ov["hs_names_ms2_only_not_in_ms1only_run"] = len(names_full_ms2 - names_ms1)
    hit_tbl = pd.DataFrame([{"metric": k, "value": v} for k, v in ov.items()])
    if write_tables:
        hit_tbl.to_csv(out / "ablation_hit_overlap.csv", index=False)

    # ---- the table ---------------------------------------------------------
    sec = [
        ("detection", "stage1_ms2_driven_candidates", "stage 1 (MS2-driven), processing_report"),
        ("detection", "stage1b_ms1_driven_added", "stage 1b MS1 peaks not explained by an MS2-driven feature"),
        ("detection", "candidates_total", "checkpoint _work/*.mfeat"),
        ("stage 2", "cand_ms1_detected", ""), ("stage 2", "cand_ms1_detected_ms1_driven", ""),
        ("stage 2", "cand_ms1_detected_ms2_driven", ""), ("stage 2", "cand_ms2_only", ""),
        ("stage 2.5", "stage25_nl_consensus", "processing_report"), ("stage 2.5", "stage25_window_fallback", ""),
        ("stage 2.5", "stage25_no_precursor", ""), ("stage 2.5", "stage25_retained_fragment_poor", "tier D, counted inside nl/fallback"),
        ("per-sample marks", "marked_isotope", "stage 4"), ("per-sample marks", "marked_adduct", "stage 5"),
        ("per-sample marks", "marked_spectral", "stage 5b"), ("per-sample marks", "marked_isf", "stage 6"),
        ("per-sample marks", "marked_total_per_sample", ""), ("per-sample marks", "isotope_groups", ""),
        ("per-sample marks", "adduct_groups", ""),
        ("alignment", "folded_at_alignment", "candidates folded into another row of the same sample by the stage-7 joiner (same compound)"),
        ("alignment", "aligned_rows", "stage 7 rows = candidates - folded"),
        ("alignment", "marked_in_alignment", "aligned rows carrying is_duplicate = True (per-sample marks that survived the fold + stage 7.2 marks)"),
        ("alignment", "aligned_dup_isotope", ""), ("alignment", "aligned_dup_adduct", ""),
        ("alignment", "aligned_dup_spectral", ""), ("alignment", "aligned_dup_isf", ""),
        ("alignment", "aligned_dup_other", "any other duplicate_type (counted with ISF in the waterfall)"),
        ("alignment", "ledger_residual", "candidates - folded - marked aligned rows - retained (0 = balanced)"),
        ("retained", "retained_rows", "is_duplicate == False"), ("retained", "retained_ms1_detected", ""),
        ("retained", "retained_ms1_detected_ms1_driven", ""), ("retained", "retained_ms1_detected_ms2_driven", ""),
        ("retained", "retained_ms2_only", ""), ("retained", "retained_ms2_only_tier_A", ""),
        ("retained", "retained_ms2_only_tier_B", ""), ("retained", "retained_ms2_only_tier_C", ""),
        ("retained", "retained_ms2_only_tier_D", ""), ("retained", "retained_ms2_only_tier_ABC", ""),
        ("high-score annotations", "hs_total", "Total score >= 1.4 and n_matched >= 3"),
        ("high-score annotations", "hs_ms1_detected", ""), ("high-score annotations", "hs_ms1_driven", ""),
        ("high-score annotations", "hs_ms2_driven_ms1", ""), ("high-score annotations", "hs_ms2_only", ""),
        ("high-score annotations", "hs_ms2_only_tier_ABC", ""), ("high-score annotations", "hs_ms2_only_tier_D", ""),
        ("high-score annotations", "hs_unique_names", "distinct names after stripping 'Putative: '"),
    ]
    for s, m, note in sec:
        add(s, m, note)
    for k, v in ov.items():
        table.append({"section": "overlap (MS1-only rows -> full-run partner)", "metric": k,
                      "ms1_only": "", "full": v, "note": ""})
    tbl = pd.DataFrame(table)
    if write_tables:
        tbl.to_csv(out / "ablation_table.csv", index=False)
    pd.set_option("display.width", 200)
    print(tbl.to_string(index=False))

    # ---- figure + legend ----------------------------------------------------
    make_figure(F, figdir, families)
    n_pairs = ov.get("pairs_ms2_driven_n", 0)
    legend = LEGEND.format(
        ver=version, c_ms1=f"{F['ms1only']['candidates_total']:,}", r_ms1=f"{F['ms1only']['retained_rows']:,}",
        c_full=f"{F['full']['candidates_total']:,}", r_full=f"{F['full']['retained_rows']:,}",
        r_full_ms1=f"{F['full']['retained_ms1_detected']:,}", r_full_ms2=f"{F['full']['retained_ms2_only']:,}",
        hs_ms1=F["ms1only"]["hs_total"], hs_full=F["full"]["hs_total"], hs_full_ms1drv=F["full"]["hs_ms1_driven"],
        hs_full_ms2drv=F["full"]["hs_ms2_driven_ms1"], hs_full_ms2only=F["full"]["hs_ms2_only"],
        frac_gt_001=f"{ov.get('pairs_ms2_driven_frac_delta_precursor_gt_0.01', float('nan')) * 100:.0f}%",
        n_pairs_ms2drv=f"{n_pairs:,}",
        fold_ms1=f"{F['ms1only']['folded_at_alignment']:,}", fold_full=f"{F['full']['folded_at_alignment']:,}",
        marks_ms1=f"{F['ms1only']['marked_in_alignment']:,}", marks_full=f"{F['full']['marked_in_alignment']:,}",
        ps_ms1=f"{F['ms1only']['marked_total_per_sample']:,}", ps_full=f"{F['full']['marked_total_per_sample']:,}",
        iso_ms1=f"{F['ms1only']['marked_isotope']:,}", iso_full=f"{F['full']['marked_isotope']:,}",
        add_ms1=F["ms1only"]["marked_adduct"], add_full=F["full"]["marked_adduct"],
        spec_ms1=F["ms1only"]["marked_spectral"], spec_full=F["full"]["marked_spectral"],
        isf_ms1=F["ms1only"]["marked_isf"], isf_full=F["full"]["marked_isf"],
        fam_rows=f"{families['n_rows']:,}", fam_n=f"{families['n_families']:,}",
        fam_single=f"{families['n_singletons']:,}", fam_rows_ge2=f"{families['n_rows_ge2']:,}",
        fam_ge2=families["n_families_ge2"], fam_largest=families["largest"])
    (figdir / "legend.txt").write_text(legend, encoding="utf-8")

    numbers = {}
    for key in runs:
        for m in ("candidates_total", "retained_rows", "retained_ms1_detected", "retained_ms2_only",
                  "retained_ms2_only_tier_ABC", "retained_ms2_only_tier_D", "hs_total", "hs_ms1_detected",
                  "hs_ms1_driven", "hs_ms2_driven_ms1", "hs_ms2_only", "marked_total_per_sample",
                  "folded_at_alignment", "marked_in_alignment", "ledger_residual", "hs_unique_names"):
            numbers[f"ablation_{key}_{m}"] = F[key][m]
    for k, v in ov.items():
        numbers[f"ablation_overlap_{k}"] = v
    if write_tables:
        update_numbers(out / "numbers_fig4.json", numbers,
                       source="scripts/ablation_table.py -> analysis/rep3_full/ablation_table.csv")
        print("written:", out / "ablation_table.csv", figdir / "sfig_ablation_draft.png")
    else:
        print("figure only (no analysis file rewritten):", figdir / "sfig_ablation_draft.png")
    return 0


def _split_ms2only_candidates(run: Path) -> tuple[int, int]:
    """(MS2-only candidates with >= 3 fragments, with < 3) from the checkpoint."""
    abc = d = 0
    for mf in sorted((run / "_work").glob("*.mfeat")):
        with open(mf, "rb") as fh:
            payload = pickle.load(fh)
        for r in payload["features"]:
            if r.get("signal_type") != "ms2_only":
                continue
            if int(r.get("n_fragments") or 0) >= 3:
                abc += 1
            else:
                d += 1
    return abc, d


if __name__ == "__main__":
    raise SystemExit(main())
