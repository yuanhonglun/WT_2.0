"""Cohort-level tables for the lung-cancer serum assay (work package S2.6, deliverables D, E, F;
reviewer points R2-7 (vi), R2-m7, R4-2).

D. panel_members.csv            the published 20 panel members: Table 10 annotation, Table 9 annotation level,
                                provenance from Table 8 (MS1-detected vs MS2-only discovery feature, discovery score),
                                per-group detection / missingness in Table 9, original model importance, internal-standard flag
E. tukey_sensitivity.csv        Dunn + BH pairwise p-values for every Fig. 7c,d / Supplementary Fig. 4 metabolite under
                                (A) the published exclusion flags, (B) all technically valid measurements, (C) the Methods
                                rule as written (1.5 x IQR per group); plus Kruskal-Wallis per version
   exclusion_counts.csv         per metabolite x group: n, n not detected, n flagged / excluded, n analysed, mean / median
   published_flag_rule_check.csv which Tukey variant reproduces the published flags
F. internal_standards_by_group.csv  Leu-d3, Trp-d5, decanoyl-carnitine-d3: median / IQR / mean / SD / CV per group
   missingness_by_group.csv     per-participant missing-transition fraction by group; counts of group-specific transitions
   missingness_per_transition.csv  per-transition missing fraction per group (all 3,541 transitions)

Inputs:
  Supplementary Table 9.xlsx    3,541 transitions x 120 participants + 2 blanks (MetaboLights MTBLS15138)
  Supplementary Table 8.xlsx    sheet Cohort_assay_features (discovery provenance of 2,172 transitions)
  Supplementary Table 10.xlsx   published panel
  fig 7/data/violin_data.xlsx   sheet "全部数据离群值检查-IQR方法": the collaborator's values (missing values imputed)
                                and the per-metabolite outlier flags actually used for Fig. 7c,d / Supplementary Fig. 4
  20250731.ml_results/          original split, per-model variable importance, multi-model ranking
Preview figures (figures/cancer_preview/): bile_acids_published_vs_all_valid.png, internal_standards_by_group.png,
  missingness_per_sample.png
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cancer_common as cc  # noqa: E402

SCRIPT = "cancer_cohort_tables.py"

# metabolites shown in the violin panels (Source Data sheets Fig7c-d_bile_acids and SFig4_metabolites)
FIGURE_METABOLITES = {
    "MN13238": ("Fig. 7c", "Cholic acid"),
    "MN13487": ("Fig. 7c", "Deoxycholic acid (isomer)"),
    "P3P2784": ("Fig. 7d", "Glycocholic acid"),
    "MN10308": ("Fig. 7d", "Taurocholic acid (isomer)"),
    "P3N0393": ("Fig. 7d", "Tauroursodeoxycholic acid"),
    "MN07051": ("Supplementary Fig. 4a", "Lactic acid"),
    "MN09828": ("Supplementary Fig. 4b", "Azelaic acid"),
    "MP01254": ("Supplementary Fig. 4c", "Inosine"),
    "MP01263": ("Supplementary Fig. 4d", "Guanosine"),
    "MP01302": ("Supplementary Fig. 4e", "Aspartic acid"),
    "MP01410": ("Supplementary Fig. 4f", "Uric acid"),
    "P3P0946": ("Supplementary Fig. 4g", "Phenylalanine"),
    "P3P1748": ("Supplementary Fig. 4h", "Hypoxanthine"),
    "P3P1751": ("Supplementary Fig. 4i", "Xanthine"),
    "P3N0310": ("workbook only (not shown)", "Glycochenodeoxycholic acid"),
}
# values printed on the collaborator's source PNGs (make_fig7.py self-check)
PUBLISHED_CHECK = {
    "MN13238": {"means": [7.94e5, 4.54e5, 1.57e6, 1.12e6, 8.90e5], "q": {("BLD", "Stage I LC"): 1.86e-2, ("BLD", "Stage III LC"): 4.84e-2}},
    "MN10308": {"means": [2.75e4, 8.14e4, 6.13e4, 6.80e4, 8.22e4], "q": {}},
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table9", type=Path, default=cc.DEFAULTS["table9"])
    ap.add_argument("--table8", type=Path, default=cc.DEFAULTS["table8"])
    ap.add_argument("--table10", type=Path, default=cc.DEFAULTS["table10"])
    ap.add_argument("--ml-dir", type=Path, default=cc.DEFAULTS["ml_dir"])
    ap.add_argument("--fig7-data", type=Path, default=cc.DEFAULTS["fig7_data"])
    ap.add_argument("--out-dir", type=Path, default=cc.DEFAULTS["out_dir"])
    ap.add_argument("--fig-dir", type=Path, default=cc.DEFAULTS["fig_dir"])
    return ap.parse_args()


def tukey_mask(values: pd.Series, by: pd.Series, k: float, transform: str | None = None) -> pd.Series:
    """True where a value lies outside [Q1 - k IQR, Q3 + k IQR] of its stratum (NaN never flagged)."""
    flag = pd.Series(False, index=values.index)
    for _, idx in by.groupby(by).groups.items():
        x = values[idx].dropna()
        if len(x) < 4:
            continue
        xt = np.log2(x) if transform == "log2" else x
        q1, q3 = np.percentile(xt, [25, 75])
        iqr = q3 - q1
        flag[x.index] = (xt < q1 - k * iqr) | (xt > q3 + k * iqr)
    return flag


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    numbers = {}
    t9 = cc.load_table9(args.table9)
    t8 = cc.load_table8_map(args.table8)
    table10 = cc.load_table10(args.table10)
    split = cc.load_split(args.ml_dir)
    stage = t9.stage
    g3 = t9.group3
    data = t9.data
    assert data.shape == (3541, 120), data.shape

    # ---------------------------------------------------------------------------------
    # F. internal standards and missingness
    # ---------------------------------------------------------------------------------
    print("internal standards and missingness ...")
    rows = []
    for tid, label in cc.IS_TRANSITIONS.items():
        row = data.loc[tid]
        hc = row[g3.index[g3 == "HC"]].dropna()
        for g in cc.GROUP3_ORDER + ["blank"]:
            v = t9.blank.loc[tid].dropna() if g == "blank" else row[g3.index[g3 == g]]
            vv = v.dropna()
            rows.append({"transition_id": tid, "internal_standard": label, "name_table9": t9.meta.loc[tid, "name_table9"], "group": g,
                         "n": int(len(v)), "n_missing": int(v.isna().sum()), "median": vv.median(), "q1": vv.quantile(0.25), "q3": vv.quantile(0.75),
                         "mean": vv.mean(), "sd": vv.std(ddof=1), "cv": vv.std(ddof=1) / vv.mean() if len(vv) > 1 else np.nan,
                         "median_ratio_vs_HC": vv.median() / hc.median(),
                         "mannwhitney_p_vs_HC": (stats.mannwhitneyu(vv, hc).pvalue if g not in ("HC", "blank") and len(vv) > 0 else np.nan)})
        bld = row[g3.index[g3 == "BLD"]].dropna()
        lc = row[g3.index[g3 == "LC"]].dropna()
        rows[-3]["mannwhitney_p_LC_vs_BLD"] = stats.mannwhitneyu(lc, bld).pvalue   # written on the BLD row for reference
    is_df = pd.DataFrame(rows)
    cc.write_csv(is_df, args.out_dir / "internal_standards_by_group.csv")
    for _, r in is_df.iterrows():
        short = {"P3P0911": "leu_d3", "P3P2641": "trp_d5", "P3P2788": "decanoylcarnitine_d3"}[r.transition_id]
        numbers[f"IS_{short}_{r.group}_median"] = {"value": float(f"{r['median']:.3g}"), "source": f"{SCRIPT}:internal_standards_by_group.csv",
                                                  "note": f"n={r.n}, missing {r.n_missing}; CV {r.cv:.3f}; ratio vs HC median {r.median_ratio_vs_HC:.3g}"
                                                          + ("; Mann-Whitney vs HC p=%.2g" % r.mannwhitney_p_vs_HC if np.isfinite(r.mannwhitney_p_vs_HC) else "")}
    lcbld = is_df.dropna(subset=["mannwhitney_p_LC_vs_BLD"]).set_index("transition_id")["mannwhitney_p_LC_vs_BLD"]
    numbers["IS_P3P2788_is_probably_not_a_spiked_standard"] = {
        "value": True, "source": f"{SCRIPT}:internal_standards_by_group.csv",
        "note": ("P3P2788 (319->257, 'Decanoyl-L-carnitine D3' in Table 9, 'Unknown (319-257)' in Table 10) is not among the 16 standards of Table 12; "
                 f"in the two blanks it is at {is_df.loc[(is_df.transition_id == 'P3P2788') & (is_df.group == 'blank'), 'median_ratio_vs_HC'].iloc[0]:.1%} of the HC median whereas Leu-d3 and Trp-d5 are at "
                 f"{is_df.loc[(is_df.transition_id == 'P3P0911') & (is_df.group == 'blank'), 'median_ratio_vs_HC'].iloc[0]:.0%} / {is_df.loc[(is_df.transition_id == 'P3P2641') & (is_df.group == 'blank'), 'median_ratio_vs_HC'].iloc[0]:.0%} (blanks carry the spiked standards); "
                 f"it differs between LC and BLD (Mann-Whitney p={lcbld['P3P2788']:.2g}) whereas Leu-d3 (p={lcbld['P3P0911']:.2g}) and Trp-d5 (p={lcbld['P3P2641']:.2g}) do not. "
                 "Table 8 traces it to MS2-only discovery feature F10951. Treated as an internal standard (excluded from the redo feature space) only because HANDOFF lists it as one; --keep-internal-standards re-includes it.")}

    miss = data.isna().mean(axis=0)   # per participant
    rows = []
    for g in cc.GROUP3_ORDER:
        v = miss[g3.index[g3 == g]]
        rows.append({"item": f"per-participant missing-transition fraction, {g}", "group": g, "n_participants": int(len(v)),
                     "min": v.min(), "median": v.median(), "mean": v.mean(), "max": v.max()})
    hc_ids = g3.index[g3 == "HC"]
    pt_ids = g3.index[g3 != "HC"]
    m_hc = data[hc_ids].isna().mean(axis=1)
    m_pt = data[pt_ids].isna().mean(axis=1)
    m_bld = data[g3.index[g3 == "BLD"]].isna().mean(axis=1)
    m_lc = data[g3.index[g3 == "LC"]].isna().mean(axis=1)
    counts = {
        "transitions missing in all 50 HC": int((m_hc == 1).sum()),
        "transitions missing in all 50 HC and detected in >= 50 % of patients": int(((m_hc == 1) & (m_pt <= 0.5)).sum()),
        "transitions missing in all 50 HC and detected in all 70 patients": int(((m_hc == 1) & (m_pt == 0)).sum()),
        "transitions missing in >= 90 % of HC and <= 10 % of patients": int(((m_hc >= 0.9) & (m_pt <= 0.1)).sum()),
        "transitions detected in all 50 HC and missing in >= 50 % of patients": int(((m_hc == 0) & (m_pt >= 0.5)).sum()),
        "transitions missing in >= 90 % of patients and <= 10 % of HC": int(((m_pt >= 0.9) & (m_hc <= 0.1)).sum()),
        "transitions with any missing value": int((data.isna().any(axis=1)).sum()),
        "transitions complete in all 120 participants": int((~data.isna().any(axis=1)).sum()),
        "transitions missing in >= 50 % of BLD and <= 10 % of LC": int(((m_bld >= 0.5) & (m_lc <= 0.1)).sum()),
        "transitions missing in >= 50 % of LC and <= 10 % of BLD": int(((m_lc >= 0.5) & (m_bld <= 0.1)).sum()),
        "total missing cells": int(data.isna().sum().sum()),
    }
    for k, v in counts.items():
        rows.append({"item": k, "group": "", "n_participants": np.nan, "min": np.nan, "median": np.nan, "mean": np.nan, "max": v})
    mb = pd.DataFrame(rows)
    cc.write_csv(mb, args.out_dir / "missingness_by_group.csv")
    for _, r in mb[mb["group"] != ""].iterrows():
        numbers[f"missing_fraction_{r.group}_median_pct"] = {"value": round(100 * r["median"], 1), "source": f"{SCRIPT}:missingness_by_group.csv", "note": f"min {100 * r['min']:.1f} %, max {100 * r['max']:.1f} %"}
    numbers["transitions_missing_all_HC_present_ge50pct_patients"] = {"value": counts["transitions missing in all 50 HC and detected in >= 50 % of patients"], "source": f"{SCRIPT}:missingness_by_group.csv"}
    numbers["transitions_missing_all_HC"] = {"value": counts["transitions missing in all 50 HC"], "source": f"{SCRIPT}:missingness_by_group.csv"}
    numbers["transitions_present_all_HC_missing_ge50pct_patients"] = {"value": counts["transitions detected in all 50 HC and missing in >= 50 % of patients"], "source": f"{SCRIPT}:missingness_by_group.csv"}

    per_tr = pd.DataFrame({"transition_id": data.index, "name_table9": t9.meta["name_table9"].values,
                           "missing_frac_HC": m_hc.values, "missing_frac_BLD": m_bld.values, "missing_frac_LC": m_lc.values, "missing_frac_patients": m_pt.values,
                           "missing_frac_all": data.isna().mean(axis=1).values,
                           "missing_frac_train": data[split.index[split["set"] == "train"]].isna().mean(axis=1).values,
                           "missing_frac_valid": data[split.index[split["set"] == "valid"]].isna().mean(axis=1).values})
    per_tr["batch_missingness_pattern"] = np.select(
        [(per_tr["missing_frac_HC"] >= 0.9) & (per_tr["missing_frac_patients"] <= 0.1),
         (per_tr["missing_frac_HC"] <= 0.1) & (per_tr["missing_frac_patients"] >= 0.9),
         (per_tr["missing_frac_HC"] >= 0.5) & (per_tr["missing_frac_patients"] <= 0.1),
         (per_tr["missing_frac_HC"] <= 0.1) & (per_tr["missing_frac_patients"] >= 0.5)],
        ["absent in HC, present in patients", "present in HC, absent in patients", "mostly absent in HC, present in patients", "present in HC, mostly absent in patients"], default="")
    cc.write_csv(per_tr, args.out_dir / "missingness_per_transition.csv")

    # ---------------------------------------------------------------------------------
    # D. panel members
    # ---------------------------------------------------------------------------------
    print("panel members ...")
    panel = cc.load_original_panel(args.ml_dir)
    varimp = cc.load_original_var_imp(args.ml_dir)
    ids = panel["transition_id"].tolist()
    anno = cc.annotate_transitions(ids, t9, t8, table10)
    pm = panel[["multiML_rank", "transition_id", "Score30", "Score"]].rename(columns={"Score30": "collaborator_annotation", "Score": "collaborator_multiML_score"})
    pm = pm.merge(anno, on="transition_id", how="left")
    pm = pm.merge(varimp.reset_index(), on="transition_id", how="left")
    pm["varimp_mean_5_models"] = pm[[f"varimp_{m}" for m in cc.MODELS]].mean(axis=1)
    for g, col in (("HC", "missing_frac_HC"), ("BLD", "missing_frac_BLD"), ("LC", "missing_frac_LC")):
        pm[col] = pm["transition_id"].map(per_tr.set_index("transition_id")[col])
        pm[f"median_{g}"] = [data.loc[t, g3.index[g3 == g]].median() for t in pm["transition_id"]]
    pm["missing_frac_train"] = pm["transition_id"].map(per_tr.set_index("transition_id")["missing_frac_train"])
    pm["missing_frac_valid"] = pm["transition_id"].map(per_tr.set_index("transition_id")["missing_frac_valid"])
    pm["batch_missingness_pattern"] = pm["transition_id"].map(per_tr.set_index("transition_id")["batch_missingness_pattern"])
    pm["on_HC_batch_missingness"] = pm["missing_frac_HC"] >= 0.5
    pm["published_median_fc_LC_vs_HC"] = [t9.stats["LC_vs_HC"].loc[t, "MeanFC"] if t in t9.stats["LC_vs_HC"].index else np.nan for t in pm["transition_id"]]
    pm["published_fdr_LC_vs_HC"] = [t9.stats["LC_vs_HC"].loc[t, "FDR"] for t in pm["transition_id"]]
    pm["published_fdr_LC_vs_BLD"] = [t9.stats["LC_vs_BLD"].loc[t, "FDR"] for t in pm["transition_id"]]
    pm["published_log2fc_LC_vs_BLD"] = [t9.stats["LC_vs_BLD"].loc[t, "MeanLog2FC"] for t in pm["transition_id"]]
    cols = ["multiML_rank", "transition_id", "table10_annotation", "collaborator_annotation", "name_table9", "annotation_level_table9", "polarity",
            "in_table8_cohort_assay", "discovery_feature_id", "signal_type", "discovery_annotation", "discovery_total_score", "discovery_high_score",
            "transition_type", "cohort_annotation_table8", "is_internal_standard", "internal_standard_note", "mz_source",
            "collaborator_multiML_score", "varimp_KNN", "varimp_LR", "varimp_NB", "varimp_RF", "varimp_SVM", "varimp_mean_5_models",
            "missing_frac_HC", "missing_frac_BLD", "missing_frac_LC", "missing_frac_train", "missing_frac_valid", "on_HC_batch_missingness", "batch_missingness_pattern",
            "median_HC", "median_BLD", "median_LC", "published_fdr_LC_vs_HC", "published_fdr_LC_vs_BLD", "published_log2fc_LC_vs_BLD"]
    pm = pm[cols]
    cc.write_csv(pm, args.out_dir / "panel_members.csv")
    numbers["panel20_in_table8"] = {"value": int(pm["in_table8_cohort_assay"].sum()), "source": f"{SCRIPT}:panel_members.csv", "note": "panel members traced to an ASFAM discovery feature in Table 8 Cohort_assay_features"}
    numbers["panel20_ms2_only"] = {"value": int((pm["signal_type"] == "ms2_only").sum()), "source": f"{SCRIPT}:panel_members.csv", "note": ";".join(pm.loc[pm["signal_type"] == "ms2_only", "transition_id"])}
    numbers["panel20_ms1_detected"] = {"value": int((pm["signal_type"] == "ms1_detected").sum()), "source": f"{SCRIPT}:panel_members.csv"}
    numbers["panel20_discovery_high_score"] = {"value": int(pm["discovery_high_score"].sum()), "source": f"{SCRIPT}:panel_members.csv", "note": "discovery_is_annotated = True in Table 8"}
    numbers["panel20_internal_standard_members"] = {"value": int(pm["is_internal_standard"].sum()), "source": f"{SCRIPT}:panel_members.csv", "note": ";".join(pm.loc[pm["is_internal_standard"], "transition_id"])}
    numbers["panel20_members_with_any_missing"] = {"value": int(((pm["missing_frac_HC"] > 0) | (pm["missing_frac_BLD"] > 0) | (pm["missing_frac_LC"] > 0)).sum()), "source": f"{SCRIPT}:panel_members.csv"}
    numbers["panel20_members_on_HC_batch_missingness"] = {"value": int(pm["on_HC_batch_missingness"].sum()), "source": f"{SCRIPT}:panel_members.csv", "note": "missing in >= 50 % of HC"}
    numbers["panel20_members_absent_in_most_patients"] = {"value": int((pm["batch_missingness_pattern"].str.contains("absent in patients")).sum()), "source": f"{SCRIPT}:panel_members.csv", "note": ";".join(pm.loc[pm["batch_missingness_pattern"].str.contains("absent in patients"), "transition_id"])}
    numbers["panel20_members_published_fdr_LC_vs_BLD_below_0.05"] = {"value": int((pm["published_fdr_LC_vs_BLD"] < 0.05).sum()), "source": f"{SCRIPT}:panel_members.csv",
                                                                     "note": "Table 9 LC_vs_BLD FDR (all 120 participants): " + ";".join(pm.loc[pm["published_fdr_LC_vs_BLD"] < 0.05, "transition_id"])}
    numbers["panel20_members_published_fdr_LC_vs_HC_below_0.05"] = {"value": int((pm["published_fdr_LC_vs_HC"] < 0.05).sum()), "source": f"{SCRIPT}:panel_members.csv",
                                                                    "note": "Table 9 LC_vs_HC FDR (all 120 participants); members not below 0.05: " + ";".join(pm.loc[~(pm["published_fdr_LC_vs_HC"] < 0.05), "transition_id"])}
    numbers["panel20_members_table9_annotation_levels"] = {"value": pm["annotation_level_table9"].str.split(" \\(").str[0].value_counts().to_dict(), "source": f"{SCRIPT}:panel_members.csv"}

    # ---------------------------------------------------------------------------------
    # E. Tukey sensitivity
    # ---------------------------------------------------------------------------------
    print("Tukey sensitivity ...")
    v = pd.read_excel(args.fig7_data / "violin_data.xlsx", sheet_name="全部数据离群值检查-IQR方法").set_index("Index")
    assert len(v) == 120
    v_stage = v["Stage"]
    assert (v_stage.reindex(stage.index) == stage).all(), "stage labels differ between the violin workbook and Table 9"
    stage_by = stage
    # which Tukey variant reproduces the published flags?
    rule_rows = []
    for by_name, by in (("all 120 participants pooled", pd.Series("all", index=v.index)), ("per stage group (5)", v_stage), ("per group (HC/BLD/LC)", v["Group"])):
        for k in (1.5, 3.0):
            for tr in (None, "log2"):
                ok = 0
                for mid in FIGURE_METABOLITES:
                    flags = v[f"{mid}_IQR"].astype(str).str.lower().eq("true")
                    rec = tukey_mask(v[mid].astype(float), by, k, tr)
                    ok += int((rec == flags).all())
                rule_rows.append({"stratum": by_name, "k": k, "transform": tr or "raw", "metabolites_reproduced": ok, "of": len(FIGURE_METABOLITES)})
    rule_df = pd.DataFrame(rule_rows).sort_values("metabolites_reproduced", ascending=False)
    cc.write_csv(rule_df, args.out_dir / "published_flag_rule_check.csv")
    best_rule = rule_df.iloc[0]
    print(f"  published flags reproduced for {best_rule['metabolites_reproduced']}/15 metabolites by: {best_rule['stratum']}, k={best_rule['k']}, {best_rule['transform']}")
    numbers["published_outlier_flag_rule"] = {"value": f"{best_rule['stratum']}, k = {best_rule['k']} x IQR, {best_rule['transform']} values (imputed values included)", "source": f"{SCRIPT}:published_flag_rule_check.csv",
                                             "note": f"reproduces the workbook flags for {int(best_rule.metabolites_reproduced)}/15 metabolites; the Methods text states 1.5 x IQR within each group, which reproduces {int(rule_df[(rule_df.k == 1.5) & (rule_df.stratum.str.startswith('per stage'))]['metabolites_reproduced'].iloc[0])}/15"}

    ex_rows, test_rows = [], []
    for mid, (figure, name) in FIGURE_METABOLITES.items():
        pub_vals = v[mid].astype(float)                       # collaborator's values (imputed where not detected)
        pub_flag = v[f"{mid}_IQR"].astype(str).str.lower().eq("true")
        raw = data.loc[mid, v.index]                          # Table 9 values, NaN = not detected
        imputed = raw.isna()
        A = pub_vals[~pub_flag]                               # published analysis
        B = raw.dropna()                                      # all technically valid measurements
        flagC = tukey_mask(raw, stage_by, 1.5)                # Methods rule as written, on detected values
        C = raw[~flagC & raw.notna()]
        groups = {}
        for ver, ser in (("A_published_flags", A), ("B_all_valid", B), ("C_valid_tukey1.5_per_group", C)):
            groups[ver] = {s: ser[stage_by.index[stage_by == s].intersection(ser.index)].to_numpy() for s in cc.STAGE_ORDER}
        for s in cc.STAGE_ORDER:
            idx = stage_by.index[stage_by == s]
            ex_rows.append({"transition_id": mid, "metabolite": name, "figure": figure, "group": s, "n_group": len(idx),
                            "n_not_detected": int(imputed[idx].sum()),
                            "n_flagged_published": int(pub_flag[idx].sum()), "n_flagged_published_that_were_imputed": int((pub_flag[idx] & imputed[idx]).sum()),
                            "n_analysed_A_published": len(groups["A_published_flags"][s]), "n_imputed_values_inside_A": int(((~pub_flag) & imputed)[idx].sum()),
                            "n_analysed_B_all_valid": len(groups["B_all_valid"][s]),
                            "n_excluded_C_tukey1.5_per_group": int(flagC[idx].sum()), "n_analysed_C": len(groups["C_valid_tukey1.5_per_group"][s]),
                            "mean_A_published": np.mean(groups["A_published_flags"][s]) if len(groups["A_published_flags"][s]) else np.nan,
                            "median_A_published": np.median(groups["A_published_flags"][s]) if len(groups["A_published_flags"][s]) else np.nan,
                            "mean_B_all_valid": np.mean(groups["B_all_valid"][s]) if len(groups["B_all_valid"][s]) else np.nan,
                            "median_B_all_valid": np.median(groups["B_all_valid"][s]) if len(groups["B_all_valid"][s]) else np.nan,
                            "mean_C": np.mean(groups["C_valid_tukey1.5_per_group"][s]) if len(groups["C_valid_tukey1.5_per_group"][s]) else np.nan,
                            "median_C": np.median(groups["C_valid_tukey1.5_per_group"][s]) if len(groups["C_valid_tukey1.5_per_group"][s]) else np.nan})
        res = {}
        for ver, gdict in groups.items():
            d = cc.dunn_pairwise(gdict)
            kw = stats.kruskal(*[x for x in gdict.values() if len(x)])
            res[ver] = (d.set_index(["group_a", "group_b"]), kw.pvalue)
        pairs = res["A_published_flags"][0].index
        for a, b in pairs:
            row = {"transition_id": mid, "metabolite": name, "figure": figure, "group_a": a, "group_b": b}
            for ver in groups:
                dd, kwp = res[ver]
                row[f"p_{ver}"] = dd.loc[(a, b), "p"]
                row[f"q_bh_{ver}"] = dd.loc[(a, b), "q_bh"]
                row[f"stars_{ver}"] = cc.stars(dd.loc[(a, b), "q_bh"])
                row[f"kruskal_p_{ver}"] = kwp
            row["significance_changes_A_vs_B"] = (row["q_bh_A_published_flags"] < 0.05) != (row["q_bh_B_all_valid"] < 0.05)
            test_rows.append(row)
        # reproduction of the published values
        if mid in PUBLISHED_CHECK:
            chk = PUBLISHED_CHECK[mid]
            for s, ref in zip(cc.STAGE_ORDER, chk["means"]):
                got = np.mean(groups["A_published_flags"][s])
                assert abs(got - ref) / ref < 5e-3, f"{mid} {s}: mean {got:.4g} != published {ref:.3g}"
            for pair, ref in chk["q"].items():
                got = res["A_published_flags"][0].loc[pair, "q_bh"]
                assert abs(got - ref) / ref < 5e-3, f"{mid} {pair}: q {got:.4g} != published {ref:.3g}"
            print(f"  {mid}: published means and Dunn+BH q-values reproduced under version A")
    ex = pd.DataFrame(ex_rows)
    cc.write_csv(ex, args.out_dir / "exclusion_counts.csv")
    ts = pd.DataFrame(test_rows)
    cc.write_csv(ts, args.out_dir / "tukey_sensitivity.csv")
    nchg = int(ts["significance_changes_A_vs_B"].sum())
    print(f"  pairs whose BH significance (q<0.05) changes between published exclusion and all-valid: {nchg}/{len(ts)}")
    numbers["tukey_pairs_total"] = {"value": int(len(ts)), "source": f"{SCRIPT}:tukey_sensitivity.csv", "note": "15 metabolites x 10 stage pairs"}
    numbers["tukey_pairs_significance_changes_published_vs_all_valid"] = {"value": nchg, "source": f"{SCRIPT}:tukey_sensitivity.csv",
                                                                       "note": "; ".join(f"{r.transition_id} {r.group_a} vs {r.group_b}: q {r['q_bh_A_published_flags']:.3g} -> {r['q_bh_B_all_valid']:.3g}" for _, r in ts[ts['significance_changes_A_vs_B']].iterrows())}
    for _, r in ts[(ts["transition_id"] == "MN13238") & (ts["group_a"] == "BLD") & (ts["group_b"].isin(["Stage I LC", "Stage III LC"]))].iterrows():
        tag = r.group_b.replace(" ", "")
        numbers[f"cholic_acid_BLD_vs_{tag}_q_published"] = {"value": float(f"{r['q_bh_A_published_flags']:.3g}"), "source": f"{SCRIPT}:tukey_sensitivity.csv", "note": "version A (published flags)"}
        numbers[f"cholic_acid_BLD_vs_{tag}_q_all_valid"] = {"value": float(f"{r['q_bh_B_all_valid']:.3g}"), "source": f"{SCRIPT}:tukey_sensitivity.csv", "note": "version B (all detected values, no exclusion)"}
    numbers["tukey_total_flagged_published"] = {"value": int(ex["n_flagged_published"].sum()), "source": f"{SCRIPT}:exclusion_counts.csv", "note": f"of which {int(ex['n_flagged_published_that_were_imputed'].sum())} were imputed (not detected) values; imputed values left inside the published analysis: {int(ex['n_imputed_values_inside_A'].sum())}"}
    numbers["tukey_total_not_detected"] = {"value": int(ex["n_not_detected"].sum()), "source": f"{SCRIPT}:exclusion_counts.csv", "note": "Table 9 NaN cells among the 15 metabolites x 120 participants"}

    # ---------------------------------------------------------------------------------
    # preview figures
    # ---------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bile = [m for m, (f, _) in FIGURE_METABOLITES.items() if f.startswith("Fig. 7")]
    fig, axes = plt.subplots(2, len(bile), figsize=(3.0 * len(bile), 6), sharex=False)
    for j, mid in enumerate(bile):
        raw = data.loc[mid, v.index]
        pub_vals = v[mid].astype(float)
        pub_flag = v[f"{mid}_IQR"].astype(str).str.lower().eq("true")
        for i, (title, ser) in enumerate((("published (flags applied, imputed kept)", pub_vals[~pub_flag]), ("all detected values", raw.dropna()))):
            ax = axes[i, j]
            dat = [ser[stage.index[stage == s].intersection(ser.index)].to_numpy() for s in cc.STAGE_ORDER]
            ax.boxplot(dat, widths=0.6, showfliers=True)
            ax.set_yscale("log")
            ax.set_xticks(range(1, 6)); ax.set_xticklabels([f"{cc.STAGE_SHORT[s]}\n({len(d)})" for s, d in zip(cc.STAGE_ORDER, dat)], fontsize=7)
            ax.set_title(f"{FIGURE_METABOLITES[mid][1]}\n{title}", fontsize=7)
    fig.tight_layout()
    fig.savefig(args.fig_dir / "bile_acids_published_vs_all_valid.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, tid in zip(axes, cc.IS_TRANSITIONS):
        row = data.loc[tid]
        dat = [row[g3.index[g3 == g]].dropna().to_numpy() for g in cc.GROUP3_ORDER]
        ax.boxplot(dat, widths=0.6)
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels(cc.GROUP3_ORDER)
        ax.set_title(f"{tid}: {t9.meta.loc[tid, 'name_table9']}", fontsize=8)
        ax.set_ylabel("intensity (cps)")
    fig.tight_layout()
    fig.savefig(args.fig_dir / "internal_standards_by_group.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.2))
    order = list(hc_ids) + list(g3.index[g3 == "BLD"]) + list(g3.index[g3 == "LC"])
    colors = ["#0072B2"] * len(hc_ids) + ["#E69F00"] * int((g3 == "BLD").sum()) + ["#CC79A7"] * int((g3 == "LC").sum())
    ax.bar(range(len(order)), 100 * miss[order].values, color=colors, width=0.9)
    ax.set_ylabel("missing transitions (%)"); ax.set_xlabel("participants (HC blue | BLD orange | LC purple)")
    ax.set_title("per-participant fraction of the 3,541 transitions not detected", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.fig_dir / "missingness_per_sample.png", dpi=150)
    plt.close(fig)

    cc.update_numbers(args.out_dir / "numbers_s2_6.json", numbers, SCRIPT)
    print("done:", SCRIPT)


if __name__ == "__main__":
    main()
