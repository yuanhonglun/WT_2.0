# -*- coding: utf-8 -*-
"""Supplementary Figs. 4 and 5 of the lung-cancer cohort (revision 2026-10, work package S3, cancer part).

Supplementary Fig. 4   participant-level distributions of the nine non-bile-acid cohort metabolites
                       (lactic acid ... xanthine, panels a-i), drawn with the Fig. 7c,d violin routine on
                       ALL technically valid measurements of Supplementary Table 9 (version B).
sfig4_outlier_removed_published_rule
                       the same routine on the published values with the published outlier flags applied
                       (version A, imputed values retained as in the submitted analysis, drawn as open
                       symbols) for the 14 metabolites of Fig. 7c,d and Supplementary Fig. 4 - the
                       sensitivity comparison referred to in the Fig. 7 legend.  Its SI numbering
                       ([[Supplementary Fig. 4b or a new Supplementary Fig.]]) is decided in S4.
Supplementary Fig. 5   model comparison: AUC with 95 % DeLong CI of the five re-derived classifiers
                       (training = 10-fold out-of-fold, held-out validation) for LC vs BLD, LC vs HC and
                       LC vs non-LC, with the previously submitted 20-metabolite models as a greyed set
                       (training = resubstitution, the only training estimate available for them;
                       validation scores restricted to the same participants).

Everything (data loading, palette, violin routine, Source Data writer) is imported from make_fig7.py,
so the two scripts cannot drift apart.  Run make_fig7.py first: this script appends its sheets to the
same Source Data workbook and its numbers to the same ledger.

Outputs:
  figures/sfig4/Supplementary Fig. 4.{svg,pdf,png}, sfig4_outlier_removed_published_rule.{svg,pdf,png}, legend_sfig4.txt
  figures/sfig5/Supplementary Fig. 5.{svg,pdf,png}, legend_sfig5.txt
  tables/source_data_cancer_draft.xlsx   sheets SFig4_*, SFig5_*  (sex column on every per-participant sheet)
  analysis/cancer/numbers_fig7.json      merged
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cancer_common as cc  # noqa: E402
from make_fig7 import (  # noqa: E402
    BILE_C, BILE_D, GREY, HC_BLUE, INK, LW, MAIN_MODEL, MM, MODEL_LONG, MODELS, REV_DIR, RULE, STAGE_ORDER,
    bracket_plan, load_cohort, panel_letter, parse_args as fig7_parse_args, per_person_table, pp, self_check_dunn,
    style_axis, violin_panel, write_sheets,
)

SCRIPT = Path(__file__).name

# The nine cohort metabolites of Supplementary Fig. 4 in legend order (a-i); none is standard-confirmed.
SFIG4_METABOLITES = [
    ("MN07051", "Lactic acid"), ("MN09828", "Azelaic acid"), ("MP01254", "Inosine"),
    ("MP01263", "Guanosine"), ("MP01302", "Aspartic acid"), ("MP01410", "Uric acid"),
    ("P3P0946", "Phenylalanine"), ("P3P1748", "Hypoxanthine"), ("P3P1751", "Xanthine"),
]
COMPARISONS = [("LC_vs_BLD", "LC vs BLD", "BLD"), ("LC_vs_HC", "LC vs HC", "HC"), ("LC_vs_nonLC", "LC vs non-LC", "non-LC")]


def parse_args(argv=None):
    base = fig7_parse_args([])
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("table9", "ml_dir", "fig7_data", "analysis_dir", "workbook", "numbers"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, default=getattr(base, name))
    p.add_argument("--sfig4-dir", type=Path, default=REV_DIR / "figures" / "sfig4")
    p.add_argument("--sfig5-dir", type=Path, default=REV_DIR / "figures" / "sfig5")
    p.add_argument("--yscale", choices=["log", "linear"], default="log")
    p.add_argument("--seed", type=int, default=20261020)
    return p.parse_args(argv)


def _save(fig, fig_dir: Path, stem: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(fig_dir / f"{stem}.{ext}", dpi=600 if ext == "png" else None, facecolor="white")
        print("  wrote", fig_dir / f"{stem}.{ext}")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Supplementary Fig. 4 (version B) and the outlier-removed version (A)
# --------------------------------------------------------------------------------------
def violin_grid(cohort, metabolites: list[tuple[str, str, bool]], version: str, rng, yscale: str, ncol: int = 3,
                row_height_mm: float = 52.0) -> tuple[plt.Figure, list[dict]]:
    nrow = int(np.ceil(len(metabolites) / ncol))
    fig = plt.figure(figsize=(210 * MM, (row_height_mm * nrow + 12) * MM))
    gs = fig.add_gridspec(nrow, ncol, left=0.075, right=0.985, top=1 - 8 / (row_height_mm * nrow + 12),
                          bottom=0.065 * 3 / nrow, hspace=0.62, wspace=0.34)
    out = []
    for k, (mid, title, dagger) in enumerate(metabolites):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        self_check_dunn(cohort, mid, version)
        out.append(violin_panel(ax, cohort, mid, title, version, rng, yscale, dagger))
        panel_letter(ax, "abcdefghijklmnop"[k], dx=-32, dy=4)
    return fig, out


def make_sfig4(args, cohort, rng, sheets, readme, numbers) -> tuple[list[dict], list[dict]]:
    nine = [(m, n, False) for m, n in SFIG4_METABOLITES]
    fig, printed_b = violin_grid(cohort, nine, "B", rng, args.yscale)
    _save(fig, args.sfig4_dir, "Supplementary Fig. 4")

    fourteen = list(BILE_C) + list(BILE_D) + nine
    fig, printed_a = violin_grid(cohort, fourteen, "A", rng, args.yscale)
    _save(fig, args.sfig4_dir, "sfig4_outlier_removed_published_rule")

    # Source Data
    person = pd.concat([per_person_table(cohort, m, n, f"Supplementary Fig. 4{'abcdefghi'[k]}") for k, (m, n) in enumerate(SFIG4_METABOLITES)], ignore_index=True)
    sheets["SFig4_metabolites"] = person
    nine_ids = [m for m, _ in SFIG4_METABOLITES]
    all_ids = [m for m, _, _ in fourteen]
    tk = cohort.tukey[cohort.tukey["transition_id"].isin(all_ids)].copy()
    plan_b = {(m, STAGE_ORDER[i], STAGE_ORDER[j]): st for m in nine_ids for i, j, st, _ in bracket_plan(cohort, m, "B")}
    plan_a = {(m, STAGE_ORDER[i], STAGE_ORDER[j]): st for m in all_ids for i, j, st, _ in bracket_plan(cohort, m, "A")}
    keys = list(zip(tk["transition_id"], tk["group_a"], tk["group_b"]))
    tk["bracket_drawn_sfig4_all_measurements"] = [k in plan_b for k in keys]
    tk["stars_drawn_sfig4_all_measurements"] = [plan_b.get(k, "") for k in keys]
    tk["bracket_drawn_outlier_removed_version"] = [k in plan_a for k in keys]
    tk["stars_drawn_outlier_removed_version"] = [plan_a.get(k, "") for k in keys]
    sheets["SFig4_pairwise_tests"] = tk
    sheets["SFig4_group_summary"] = cohort.exclusion[cohort.exclusion["transition_id"].isin(all_ids)].copy()
    readme["SFig4_metabolites"] = "per participant (nine metabolites of Supplementary Fig. 4): Table 9 intensity (blank = not detected), sex, age, inclusion under version A (published flags) / B (all measured, plotted) / C (1.5 x IQR per group); the five bile acids of the outlier-removed version are in Fig7c_/Fig7d_ sheets"
    readme["SFig4_pairwise_tests"] = "Dunn + BH q-values (versions A / B / C) for the 14 metabolites, which brackets are drawn in Supplementary Fig. 4 (all measurements) and in the outlier-removed version"
    readme["SFig4_group_summary"] = "per metabolite x group: n, not detected, flagged, analysed n, mean, median under versions A / B / C"

    for v in printed_b:
        mid = v["transition_id"]
        for s, n in v["n_per_group"].items():
            numbers[f"sfig4_{mid}_n_{cc.STAGE_SHORT[s]}"] = {"value": n, "source": f"{SCRIPT}:Table 9 version B", "note": f"{mid} group {s}: measured values plotted"}
        numbers[f"sfig4_{mid}_kruskal_p_B"] = {"value": v["kruskal_p"], "source": f"{SCRIPT}:tukey_sensitivity.csv", "note": "Kruskal-Wallis, version B"}
        for b in v["brackets"]:
            key = b["pair"].replace(" ", "_").replace("Stage_", "").replace("_LC", "")
            numbers[f"sfig4_{mid}_{key}_q_B"] = {"value": b["q_bh"], "source": f"{SCRIPT}:tukey_sensitivity.csv", "note": f"Dunn + BH, version B; drawn as '{b['stars']}'"}
    for v in printed_a:
        mid = v["transition_id"]
        for s, n in v["n_per_group"].items():
            numbers[f"sfig4_outlier_removed_{mid}_n_{cc.STAGE_SHORT[s]}"] = {"value": n, "source": f"{SCRIPT}:violin_data.xlsx version A", "note": f"{mid} group {s}: values plotted after the published flags (imputed values included)"}
        for b in v["brackets"]:
            key = b["pair"].replace(" ", "_").replace("Stage_", "").replace("_LC", "")
            numbers[f"sfig4_outlier_removed_{mid}_{key}_q_A"] = {"value": b["q_bh"], "source": f"{SCRIPT}:tukey_sensitivity.csv", "note": f"Dunn + BH, version A; drawn as '{b['stars']}'"}
    n_changed = int(cohort.tukey[cohort.tukey["transition_id"].isin(all_ids)]["significance_changes_A_vs_B"].sum())
    numbers["sfig4_pairs_whose_significance_changed_A_vs_B"] = {"value": n_changed, "source": f"{SCRIPT}:tukey_sensitivity.csv", "note": "of 140 pairs (14 metabolites x 10); cholic acid HC vs stage I only"}
    return printed_b, printed_a


# --------------------------------------------------------------------------------------
# Supplementary Fig. 5: model comparison
# --------------------------------------------------------------------------------------
def make_sfig5(args, cohort, sheets, readme, numbers) -> list[dict]:
    redo = pd.read_csv(args.analysis_dir / "redo_model_metrics.csv")
    orig = pd.read_csv(args.analysis_dir / "original_models_metrics.csv")
    orig = orig[orig["stage"] == "all LC"]
    pp.apply_rcparams()
    fig = plt.figure(figsize=(210 * MM, 92 * MM))
    gs = fig.add_gridspec(1, 3, left=0.065, right=0.985, top=0.80, bottom=0.27, wspace=0.32)
    rows = []
    series = [  # (source, dataset key, colour, marker, filled, y offset, label)
        ("re-derived", "train_oof_10fold", HC_BLUE, "o", False, 0.27, "re-derived model, training (10-fold out-of-fold)"),
        ("re-derived", "valid", HC_BLUE, "o", True, 0.09, "re-derived model, held-out validation"),
        ("previously submitted", "train", GREY, "D", False, -0.09, "previously submitted 20-transition model, training (resubstitution)"),
        ("previously submitted", "valid", GREY, "D", True, -0.27, "previously submitted model, held-out validation (same participants)"),
    ]
    for k, (comp, title, ctrl) in enumerate(COMPARISONS):
        ax = fig.add_subplot(gs[0, k])
        r = redo[redo["comparison"] == comp]
        o = orig[orig["comparison"] == comp]
        panel_size = int(r["panel_size"].iloc[0])
        for i, m in enumerate(MODELS):
            y0 = len(MODELS) - 1 - i
            if i:
                ax.axhline(y0 + 0.5, color=RULE, linewidth=0.6, zorder=0)
            for source, dkey, colour, marker, filled, dy, label in series:
                tbl = r if source == "re-derived" else o
                row = tbl[(tbl["model"] == m) & (tbl["dataset"] == dkey)]
                assert len(row) == 1, f"{comp} {m} {source} {dkey}: {len(row)} rows"
                row = row.iloc[0]
                auc, lo, hi = float(row["auc"]), float(row["auc_delong_lo"]), float(row["auc_delong_hi"])
                ax.errorbar(auc, y0 + dy, xerr=[[auc - lo], [hi - auc]], fmt=marker, ms=3.6, mfc=colour if filled else "white",
                            mec=colour, ecolor=colour, elinewidth=LW, capsize=1.6, capthick=LW, zorder=3)
                dataset_label = {"train_oof_10fold": "training (10-fold out-of-fold)", "train": "training (resubstitution)", "valid": "held-out validation"}[dkey]
                rows.append({"comparison": comp, "source": source, "model": m, "dataset": dataset_label, "n": int(row["n"]),
                             "n_case": int(row["n_case"]), "n_control": int(row["n_control"]), "auc": auc, "auc_delong_lo": lo, "auc_delong_hi": hi,
                             "auc_boot_lo": float(row["auc_boot_lo"]), "auc_boot_hi": float(row["auc_boot_hi"]),
                             "panel_size": panel_size if source == "re-derived" else 20})
                numbers[f"sfig5_{comp}_{source.replace(' ', '_')}_{m}_{dkey}_auc"] = {"value": auc, "source": f"{SCRIPT}:{'redo_model_metrics.csv' if source == 're-derived' else 'original_models_metrics.csv'}",
                                                                                     "note": f"DeLong 95% CI {lo:.4f}-{hi:.4f}; n = {int(row['n'])} ({int(row['n_case'])} LC vs {int(row['n_control'])})"}
        ax.axvline(0.5, color=RULE, linewidth=LW, linestyle=(0, (1.5, 1.5)), zorder=0)
        ax.set_xlim(0.38, 1.02)
        ax.set_xticks([0.4, 0.6, 0.8, 1.0])
        ax.set_ylim(-0.5, len(MODELS) - 0.5)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels(list(reversed(MODELS)))
        ax.set_xlabel("AUC (95% DeLong CI)", labelpad=2)
        tr = r[r["dataset"] == "train_oof_10fold"].iloc[0]
        va = r[r["dataset"] == "valid"].iloc[0]
        ax.set_title(title, fontsize=8, fontweight="bold", pad=24)
        ax.text(0.5, 1.02, f"re-derived panel of {panel_size} transition{'s' if panel_size != 1 else ''}\ntraining {int(tr['n_case'])} vs {int(tr['n_control'])}, validation {int(va['n_case'])} vs {int(va['n_control'])}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color=INK)
        if comp != "LC_vs_BLD":
            ax.text(0.40, len(MODELS) - 0.62, "exploratory, confounded\nby the HC difference", fontsize=7, color=GREY, ha="left", va="top", style="italic")
        style_axis(ax)
        panel_letter(ax, "abc"[k], dx=-30, dy=24)
        numbers[f"sfig5_{comp}_panel_size"] = {"value": panel_size, "source": f"{SCRIPT}:redo_model_metrics.csv", "note": "re-derived panel"}
    handles = [Line2D([], [], color=c, marker=mk, ms=3.6, mfc=c if f else "white", mec=c, linewidth=LW, label=lab) for _, _, c, mk, f, _, lab in series]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.52, 0.01),
               handlelength=2.2, handletextpad=0.6, columnspacing=1.6, labelspacing=0.4)
    _save(fig, args.sfig5_dir, "Supplementary Fig. 5")

    metrics_df = pd.DataFrame(rows)
    sheets["SFig5_model_metrics"] = metrics_df
    readme["SFig5_model_metrics"] = "every dot of Supplementary Fig. 5: AUC with DeLong and bootstrap 95% CI per comparison x source x model x dataset"
    # per-participant scores behind the training (OOF) dots and the published scores
    frames = []
    for comp, _, _ in COMPARISONS:
        t = pd.read_csv(args.analysis_dir / f"redo_predictions_train_{comp}.csv")
        cl = cohort.clinical.reindex(t["sample"])
        t.insert(1, "comparison", comp)
        t.insert(2, "set", "training")
        t.insert(5, "sex", cl["sex"].values)
        t.insert(6, "age", cl["age"].values)
        frames.append(t.rename(columns={"y": "true_label_LC"}))
    sheets["SFig5_train_OOF_predictions"] = pd.concat(frames, ignore_index=True)
    readme["SFig5_train_OOF_predictions"] = "re-derived models: per training participant in-sample (prob_<model>) and 10-fold out-of-fold (prob_oof_<model>) probabilities, with sex and age; validation probabilities are in the Fig7e sheets"
    pub = cc.load_original_scores(args.fig7_data)
    frames = []
    for name, df in pub.items():
        d = df.copy()
        d.insert(0, "sample", d.index)
        d.insert(1, "set", "training" if name == "train" else "held-out validation")
        sp = cohort.split.reindex(d["sample"])
        cl = cohort.clinical.reindex(d["sample"])
        d.insert(2, "group", sp["group3"].values)
        d.insert(3, "stage", sp["stage"].values)
        d.insert(4, "sex", cl["sex"].values)
        d.insert(5, "age", cl["age"].values)
        frames.append(d.rename(columns={"y": "true_label_LC", **{m: f"prob_{m}" for m in MODELS}}).reset_index(drop=True))
    sheets["SFig5_published_model_scores"] = pd.concat(frames, ignore_index=True)
    readme["SFig5_published_model_scores"] = "previously submitted 20-metabolite models: per participant probabilities of the five models (training and held-out validation), with sex and age; the LC vs BLD / LC vs HC rows of Supplementary Fig. 5 use these scores restricted to the respective participants"
    return rows


# --------------------------------------------------------------------------------------
# legends
# --------------------------------------------------------------------------------------
def write_legends(args, cohort, printed_b: list[dict], printed_a: list[dict], sfig5_rows: list[dict]) -> None:
    n = {v["transition_id"]: v["n_per_group"] for v in printed_b}
    la = n["MN07051"]
    ids14 = [v["transition_id"] for v in printed_a]
    ex14 = cohort.exclusion[cohort.exclusion["transition_id"].isin(ids14)]
    n_imputed_kept = int(ex14["n_imputed_values_inside_A"].sum())
    n_flagged = int(ex14["n_flagged_published"].sum())
    axis_note = ("The intensity axis is logarithmic; the pairwise tests are rank-based and unaffected by the scale. " if args.yscale == "log" else "")
    sf4 = f"""Supplementary Fig. 4. Targeted-cohort serum distributions for nine metabolites across healthy controls, benign lung disease, and lung-cancer stages, all technically valid measurements.
a–i: lactic acid, azelaic acid, inosine, guanosine, aspartic acid, uric acid, phenylalanine, hypoxanthine, and xanthine, respectively (HC, n = {la['HC']}; BLD, n = {la['BLD']}; stage I LC, n = {la['Stage I LC']}; stage III LC, n = {la['Stage III LC']}; stage IV LC, n = {la['Stage IV LC']} participants; the number of measured values plotted is given under each group label and is smaller than the group size only where the signal was not detected). All technically valid measurements are shown; no outlier removal was applied (the version applying the published outlier flags is shown in [[Supplementary Fig. 4b / new Supplementary Fig.: outlier-removed version]]). Points show individual participants; violins show the distribution and embedded boxes show the median and interquartile range (whiskers to 1.5 x IQR). {axis_note}Groups were compared by a Kruskal–Wallis test followed by pairwise Dunn tests adjusted across all ten group pairs by the Benjamini–Hochberg method; brackets are drawn for the pairs annotated in the previously submitted figure and for any pair whose adjusted significance changed when all measurements were included (none among these nine metabolites). *q < 0.05, **q < 0.01, ***q < 0.001. BLD, benign lung disease; HC, healthy control; I, III, and IV denote lung-cancer stage.

[[Supplementary Fig. 4b or new Supplementary Fig. - numbering to be decided; file sfig4_outlier_removed_published_rule]]
Sensitivity of Fig. 7c,d and Supplementary Fig. 4 to the outlier rule applied in the previously submitted analysis.
a–e: cholic acid, deoxycholic acid (isomer), glycocholic acid, taurocholic acid (isomer), tauroursodeoxycholic acid (Fig. 7c,d); f–n: lactic acid, azelaic acid, inosine, guanosine, aspartic acid, uric acid, phenylalanine, hypoxanthine, xanthine (Supplementary Fig. 4). Values and outlier flags are those of the previously submitted analysis: the flags ({n_flagged} values across the 14 metabolites) correspond to a 3 x IQR rule applied to the raw values of all 120 participants pooled (not to the 1.5 x IQR per-group rule stated in the earlier Methods text), and {n_imputed_kept} values that had been imputed for non-detected signals (constant of about 1 x 10^2 cps) were retained as data points; they are drawn as open symbols in the censored band at the base of the axis (every imputed value lies below every measured value, so ranks, boxes and tests are unaffected; see Source Data for the per-participant status). Elements, tests and star thresholds as in Supplementary Fig. 4; brackets are the pairs with q < 0.05 under this version. Across the 140 group pairs (14 metabolites x 10 pairs) the Benjamini–Hochberg significance at q < 0.05 differs between this version and the all-measurements version for one pair only (cholic acid, HC vs stage I LC: q = 0.054 here, q = 0.048 with all measurements). A dagger indicates authentic-standard confirmation.
"""
    path = args.sfig4_dir / "legend_sfig4.txt"
    path.write_text(sf4, encoding="utf-8")
    print("  wrote", path)

    def get(comp, source, m, ds):
        for r in sfig5_rows:
            if r["comparison"] == comp and r["source"] == source and r["model"] == m and r["dataset"].startswith(ds):
                return r
        raise KeyError((comp, source, m, ds))

    kb = get("LC_vs_BLD", "re-derived", MAIN_MODEL, "held-out")
    kbt = get("LC_vs_BLD", "re-derived", MAIN_MODEL, "training")
    ob = get("LC_vs_BLD", "previously submitted", MAIN_MODEL, "held-out")
    sizes = {comp: get(comp, "re-derived", MAIN_MODEL, "held-out")["panel_size"] for comp, _, _ in COMPARISONS}
    valid_ranges = {}
    for comp, _, _ in COMPARISONS:
        v = [get(comp, "re-derived", m, "held-out")["auc"] for m in MODELS]
        valid_ranges[comp] = (min(v), max(v))
    sf5 = f"""Supplementary Fig. 5. Classifier comparison for the targeted serum assay: re-derived models versus the previously submitted models.
a–c: LC vs BLD (the pre-specified primary comparison), LC vs HC and LC vs non-LC. Dots give the AUC and horizontal bars the 95% DeLong confidence interval for the five classifiers (KNN, K-nearest neighbours; LR, logistic regression; NB, naive Bayes; RF, random forest; SVM, support vector machine). Blue: models re-derived entirely inside the training set (missing-value imputation and scaling parameters, univariate screening, LASSO selection over 50 seeds with a 70% reproducibility threshold, correlation pruning, multi-model importance consensus, 3 x 10-fold tuning and the cut-off were all fitted on the training participants only), giving panels of {sizes['LC_vs_BLD']} (LC vs BLD), {sizes['LC_vs_HC']} (LC vs HC) and {sizes['LC_vs_nonLC']} (LC vs non-LC) transitions; open circles are the training estimate from 10-fold out-of-fold predictions, filled circles the single scoring of the held-out validation participants. Grey: the previously submitted 20-transition models, whose candidate pool had been derived on all 120 participants; open diamonds are the training-set resubstitution estimate (the only training estimate available for them) and filled diamonds the held-out validation participants restricted to the respective comparison. Training and validation participants are given above each panel (LC vs control). For LC vs BLD the re-derived KNN model reaches AUC = {kb['auc']:.3f} (95% CI {kb['auc_delong_lo']:.3f}–{kb['auc_delong_hi']:.3f}) on the held-out participants (training out-of-fold {kbt['auc']:.3f}), the five re-derived models span {valid_ranges['LC_vs_BLD'][0]:.3f}–{valid_ranges['LC_vs_BLD'][1]:.3f}, and the previously submitted KNN model scores {ob['auc']:.3f} ({ob['auc_delong_lo']:.3f}–{ob['auc_delong_hi']:.3f}) on the same 20 participants. LC vs HC and LC vs non-LC are exploratory and confounded by the difference of the HC sera described in Methods (the LC vs HC panel collapses to a single presence/absence transition), so their AUCs cannot be attributed to disease. Both sets were drawn from the same 120-participant cohort, so all estimates describe internal training and held-out internal validation, not external validation; the classifier demonstrates that ASFAM-derived transitions can be carried into a targeted assay and modelled, and is not diagnostic evidence. Per-participant probabilities and all metrics (including stratified bootstrap intervals) are given in the Source Data. AUC, area under the curve; BLD, benign lung disease; HC, healthy control; LC, lung cancer.
"""
    path = args.sfig5_dir / "legend_sfig5.txt"
    path.write_text(sf5, encoding="utf-8")
    print("  wrote", path)


def main(argv=None) -> None:
    args = parse_args(argv)
    pp.apply_rcparams()
    cohort = load_cohort(args)
    rng = np.random.default_rng(args.seed)
    sheets: dict[str, pd.DataFrame] = {}
    readme: dict[str, str] = {}
    numbers: dict[str, dict] = {}
    printed_b, printed_a = make_sfig4(args, cohort, rng, sheets, readme, numbers)
    sfig5_rows = make_sfig5(args, cohort, sheets, readme, numbers)
    write_sheets(args.workbook, sheets, readme)
    cc.update_numbers(args.numbers, numbers, SCRIPT)
    print(f"  numbers: {len(numbers)} entries merged into {args.numbers}")
    write_legends(args, cohort, printed_b, printed_a, sfig5_rows)


if __name__ == "__main__":
    main()
