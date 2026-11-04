"""Metrics of the five PUBLISHED lung-cancer models from their original per-participant probabilities
(work package S2.6, deliverables A and B; reviewer points R2-7 (v), R4-2).

Inputs (all verbatim collaborator / Supplementary-Table copies):
  fig 7/data/roc_train.txt, roc_valid.txt     per-participant predicted probabilities, KNN/LR/NB/RF/SVM,
                                             LC (Real = 1) vs non-LC (Real = 0); Source Data sheets
                                             Fig7e_ROC_training / Fig7e_ROC_validation / SFig5_ROC_scores
  fig 7/data/merge_metrics.details.txt       published Accuracy / AUC / Sensitivity / Specificity per model x set
  20250731.ml_results/ml_results/1.group_split/LC.{train,valid}.group.txt   the 85 / 35 split with stage labels
  20250731.ml_results/ml_results/3.model_result/<model>/*.predict_result.txt  the cut-off the collaborator applied
  20250731.ml_results/clinical_info_120Samples.xlsx  sex, age (all), smoking history, TNM, CEA, CA125, Cyfra21-1, NSE
                                             (patients only); published subset = Supplementary Table 6
  Supplementary Table 9.xlsx                 only for the HC / BLD / LC(stage) label of every participant

Outputs (analysis/cancer/):
  original_cutoff_check.csv        does 0.5 reproduce the published sensitivity / specificity? which cut-off does?
  original_models_metrics.csv      model x set x comparison (LC vs non-LC, LC vs BLD subset, LC vs HC subset,
                                   stage-stratified) with AUC, DeLong and bootstrap 95% CI, confusion matrix
  cyfra_auc.csv                    Cyfra21-1 (and CEA, CA125, NSE) alone, LC vs BLD, train / valid / all patients
  panel_vs_cyfra_paired_delong.csv held-out LC vs BLD: each published model vs Cyfra21-1 on the same participants
  covariate_models.csv, covariate_model_coefficients.csv
                                   logistic regressions of the outcome on panel score, age, sex (+ smoking)
  covariate_balance.csv            age / sex / smoking by group with tests; split composition check
  numbers_s2_6.json (updated)      every number that may be cited
Preview figures (figures/cancer_preview/): original_roc_curves.png, lc_vs_bld_heldout_panel_vs_cyfra.png
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cancer_common as cc  # noqa: E402

SCRIPT = "cancer_metrics.py"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fig7-data", type=Path, default=cc.DEFAULTS["fig7_data"])
    ap.add_argument("--ml-dir", type=Path, default=cc.DEFAULTS["ml_dir"])
    ap.add_argument("--table9", type=Path, default=cc.DEFAULTS["table9"])
    ap.add_argument("--out-dir", type=Path, default=cc.DEFAULTS["out_dir"])
    ap.add_argument("--fig-dir", type=Path, default=cc.DEFAULTS["fig_dir"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--cyfra-clinical-cutoff", type=float, default=3.3,
                    help="commonly used Cyfra21-1 decision threshold (ng/mL) reported alongside the Youden cut-off")
    return ap.parse_args()


# --------------------------------------------------------------------------------------
def build_sample_table(t9: cc.Table9, split: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"group3": t9.group3, "stage": t9.stage, "label_table9": t9.sample_label})
    df = df.join(split[["set", "group_r", "reference"]], how="left")
    df = df.join(clinical[["sex", "age", "smoking_ever", "smoking_raw", "CEA", "CA125", "Cyfra21_1", "NSE", "group3"]].rename(columns={"group3": "group3_clinical"}), how="left")
    if (df["group3"] != df["group3_clinical"]).any():
        bad = df.index[df["group3"] != df["group3_clinical"]].tolist()
        raise ValueError(f"group mismatch between Table 9 and the clinical table: {bad}")
    if df["set"].isna().any():
        raise ValueError("participants missing from the split files")
    return df


def subset_rows(samples: pd.DataFrame, scores: pd.DataFrame, comparison: str, case_groups, control_groups, stage_filter=None):
    """Return y, score frame for a comparison restricted to the given participants."""
    idx = scores.index
    g = samples.loc[idx, "group3"]
    st = samples.loc[idx, "stage"]
    keep_case = g.isin(case_groups)
    if stage_filter is not None:
        keep_case &= st.eq(stage_filter)
    keep = keep_case | g.isin(control_groups)
    sub = scores.loc[keep]
    y = (g.loc[keep].isin(case_groups)).astype(int).values
    return y, sub


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    numbers = {}

    print("loading inputs ...")
    t9 = cc.load_table9(args.table9)
    split = cc.load_split(args.ml_dir)
    clinical = cc.load_clinical(args.ml_dir)
    scores = cc.load_original_scores(args.fig7_data)
    published = cc.load_published_metrics(args.fig7_data)
    cutoffs = cc.load_original_cutoffs(args.ml_dir)
    samples = build_sample_table(t9, split, clinical)

    # ---- sanity: split composition (R2-7 ii)
    comp = samples.groupby(["set", "stage"]).size().unstack(fill_value=0).reindex(columns=cc.STAGE_ORDER)
    print("split composition:\n", comp)
    expected = {"train": {"HC": 35, "BLD": 14, "Stage I LC": 7, "Stage III LC": 8, "Stage IV LC": 21},
                "valid": {"HC": 15, "BLD": 6, "Stage I LC": 3, "Stage III LC": 3, "Stage IV LC": 8}}
    for s, d in expected.items():
        for k, v in d.items():
            assert int(comp.loc[s, k]) == v, (s, k, int(comp.loc[s, k]), v)
    # the scores files must contain exactly the split participants with the right labels
    for s in ("train", "valid"):
        ids = set(scores[s].index)
        assert ids == set(split.index[split["set"] == s]), f"{s}: score file participants differ from the split"
        y_file = scores[s]["y"].astype(int)
        y_split = (samples.loc[scores[s].index, "group3"] == "LC").astype(int)
        assert (y_file.values == y_split.values).all(), f"{s}: Real column disagrees with the LC label"

    # ---------------------------------------------------------------------------------
    # A1. cut-off check
    # ---------------------------------------------------------------------------------
    print("cut-off check ...")
    rows = []
    for m in cc.MODELS:
        ytr, str_ = scores["train"]["y"].values, scores["train"][m].values
        ctl = cc.closest_topleft_cutoff(ytr, str_)
        yj = cc.youden_cutoff(ytr, str_)
        for dset in ("train", "valid"):
            y, s = scores[dset]["y"].values, scores[dset][m].values
            pub = published[(published["Method"] == m) & (published["DType"] == dset)].iloc[0]
            se05, sp05 = cc.sens_spec_at(y, s, 0.5)
            serec, sprec = cc.sens_spec_at(y, s, cutoffs[m])
            sectl, spctl = cc.sens_spec_at(y, s, ctl)
            auc = cc.fast_auc(y, s)
            rows.append({
                "model": m, "dataset": dset, "n": len(y),
                "published_auc": pub["AUC"], "recomputed_auc": round(auc, 4), "auc_match": abs(auc - pub["AUC"]) < 5e-4,
                "published_sensitivity": pub["Sensitivity"], "published_specificity": pub["Specificity"], "published_accuracy": pub["Accuracy"],
                "sens_at_0.5": round(se05, 4), "spec_at_0.5": round(sp05, 4),
                "cutoff_0.5_reproduces_published": (abs(se05 - pub["Sensitivity"]) < 5e-4) and (abs(sp05 - pub["Specificity"]) < 5e-4),
                "recorded_cutoff": cutoffs[m], "sens_at_recorded": round(serec, 4), "spec_at_recorded": round(sprec, 4),
                "recorded_cutoff_reproduces_published": (abs(serec - pub["Sensitivity"]) < 5e-4) and (abs(sprec - pub["Specificity"]) < 5e-4),
                "recomputed_closest_topleft_cutoff_train": ctl, "sens_at_closest_topleft": round(sectl, 4), "spec_at_closest_topleft": round(spctl, 4),
                "recomputed_youden_cutoff_train": yj,
                "note": "recorded_cutoff = column Cutoff of the collaborator's predict_result.txt (pROC closest.topleft on the training ROC)",
            })
    cut_df = pd.DataFrame(rows)
    cc.write_csv(cut_df, args.out_dir / "original_cutoff_check.csv")
    n_fail05 = int((~cut_df["cutoff_0.5_reproduces_published"]).sum())
    n_failrec = int((~cut_df["recorded_cutoff_reproduces_published"]).sum())
    print(f"  0.5 fails to reproduce published sens/spec in {n_fail05}/10 model-sets; recorded cut-off fails in {n_failrec}/10")
    for m in cc.MODELS:
        numbers[f"orig_cutoff_{m}"] = {"value": round(cutoffs[m], 6), "source": f"{SCRIPT}:original_cutoff_check.csv",
                                      "note": "cut-off applied by the collaborator (training-set closest-top-left); 0.5 does not reproduce the published sensitivity/specificity for KNN/NB/RF/SVM"}
    numbers["orig_cutoff_0.5_reproduces_n_of_10"] = {"value": 10 - n_fail05, "source": f"{SCRIPT}:original_cutoff_check.csv"}
    numbers["orig_cutoff_recorded_reproduces_n_of_10"] = {"value": 10 - n_failrec, "source": f"{SCRIPT}:original_cutoff_check.csv"}

    # ---------------------------------------------------------------------------------
    # A2. metrics by comparison and stage
    # ---------------------------------------------------------------------------------
    print("metrics by comparison ...")
    comparisons = dict(cc.COMPARISONS)
    rows = []
    for m in cc.MODELS:
        for dset in ("train", "valid"):
            for cname, (case_g, ctrl_g) in comparisons.items():
                y, sub = subset_rows(samples, scores[dset], cname, case_g, ctrl_g)
                blk = cc.metrics_block(y, sub[m].values, cutoffs[m], n_boot=args.n_boot, seed=args.seed)
                pub = published[(published["Method"] == m) & (published["DType"] == dset)].iloc[0]
                rows.append({"model": m, "dataset": dset, "comparison": cname, "stage": "all LC", **blk,
                             "cutoff_rule": "collaborator's training closest-top-left cut-off, applied unchanged",
                             "published_auc": pub["AUC"] if cname == "LC_vs_nonLC" else np.nan,
                             "published_sensitivity": pub["Sensitivity"] if cname == "LC_vs_nonLC" else np.nan,
                             "published_specificity": pub["Specificity"] if cname == "LC_vs_nonLC" else np.nan})
                for stg in ("Stage I LC", "Stage III LC", "Stage IV LC"):
                    y2, sub2 = subset_rows(samples, scores[dset], cname, case_g, ctrl_g, stage_filter=stg)
                    blk2 = cc.metrics_block(y2, sub2[m].values, cutoffs[m], n_boot=args.n_boot, seed=args.seed)
                    rows.append({"model": m, "dataset": dset, "comparison": cname, "stage": stg, **blk2,
                                 "cutoff_rule": "collaborator's training closest-top-left cut-off, applied unchanged",
                                 "published_auc": np.nan, "published_sensitivity": np.nan, "published_specificity": np.nan})
    met = pd.DataFrame(rows)
    # order columns
    first = ["model", "dataset", "comparison", "stage", "n", "n_case", "n_control", "prevalence", "auc", "auc_delong_lo", "auc_delong_hi",
             "auc_boot_lo", "auc_boot_hi", "auc_se_delong", "cutoff", "cutoff_rule", "sensitivity", "specificity", "ppv", "npv", "accuracy",
             "tp", "fp", "tn", "fn", "small_n_flag", "published_auc", "published_sensitivity", "published_specificity"]
    met = met[first]
    for c in ("prevalence", "auc", "auc_delong_lo", "auc_delong_hi", "auc_boot_lo", "auc_boot_hi", "auc_se_delong", "sensitivity", "specificity", "ppv", "npv", "accuracy"):
        met[c] = met[c].astype(float).round(4)
    assert met["auc"].dropna().between(0, 1).all()
    cc.write_csv(met, args.out_dir / "original_models_metrics.csv")

    # published AUC check (LC vs non-LC)
    chk = met[(met["comparison"] == "LC_vs_nonLC") & (met["stage"] == "all LC")]
    mism = chk[(chk["auc"] - chk["published_auc"]).abs() > 5e-4]
    print("  published AUC mismatches:", len(mism))
    for _, r in chk.iterrows():
        numbers[f"orig_{r.model}_{r.dataset}_LCvsnonLC_auc"] = {"value": r.auc, "source": f"{SCRIPT}:original_models_metrics.csv",
                                                              "note": f"DeLong 95% CI {r.auc_delong_lo:.3f}-{r.auc_delong_hi:.3f}; bootstrap {r.auc_boot_lo:.3f}-{r.auc_boot_hi:.3f}; published {r.published_auc}"}
        numbers[f"orig_{r.model}_{r.dataset}_LCvsnonLC_sens"] = {"value": r.sensitivity, "source": f"{SCRIPT}:original_models_metrics.csv", "note": f"at recorded cut-off {r.cutoff:.4f}; TP/FP/TN/FN {r.tp}/{r.fp}/{r.tn}/{r.fn}"}
        numbers[f"orig_{r.model}_{r.dataset}_LCvsnonLC_spec"] = {"value": r.specificity, "source": f"{SCRIPT}:original_models_metrics.csv"}
    for cname in ("LC_vs_BLD", "LC_vs_HC"):
        sub = met[(met["comparison"] == cname) & (met["stage"] == "all LC") & (met["dataset"] == "valid")]
        for _, r in sub.iterrows():
            numbers[f"orig_{r.model}_valid_{cname}_auc"] = {"value": r.auc, "source": f"{SCRIPT}:original_models_metrics.csv",
                                                           "note": f"held-out subset n={r.n} ({r.n_case} LC vs {r.n_control}); DeLong {r.auc_delong_lo:.3f}-{r.auc_delong_hi:.3f}; bootstrap {r.auc_boot_lo:.3f}-{r.auc_boot_hi:.3f}; sens {r.sensitivity:.3f} spec {r.specificity:.3f} at the original cut-off"}
    # stage rows in validation (KNN) for quick citation
    for _, r in met[(met["model"] == "KNN") & (met["dataset"] == "valid") & (met["stage"] != "all LC")].iterrows():
        numbers[f"orig_KNN_valid_{r.comparison}_{r.stage.replace(' ', '')}_auc"] = {"value": r.auc, "source": f"{SCRIPT}:original_models_metrics.csv",
                                                                                   "note": f"n_case={r.n_case}, n_control={r.n_control}; small_n_flag={r.small_n_flag}"}

    # ---------------------------------------------------------------------------------
    # B1. Cyfra21-1 and the other clinical markers alone, LC vs BLD
    # ---------------------------------------------------------------------------------
    print("clinical markers alone ...")
    rows = []
    pat = samples[samples["group3"].isin(["LC", "BLD"])].copy()
    pat["y"] = (pat["group3"] == "LC").astype(int)
    for marker in ("Cyfra21_1", "CEA", "CA125", "NSE"):
        for dset, sel in (("train", pat["set"] == "train"), ("valid", pat["set"] == "valid"), ("all_patients", pat["set"].notna())):
            d = pat[sel]
            avail = d[marker].notna()
            y, s = d.loc[avail, "y"].values, d.loc[avail, marker].values
            yc = cc.youden_cutoff(y, s) if len(np.unique(y)) == 2 else np.nan
            blk = cc.metrics_block(y, s, yc, n_boot=args.n_boot, seed=args.seed)
            row = {"marker": marker, "comparison": "LC_vs_BLD", "dataset": dset, "n_missing_marker": int((~avail).sum()), **blk,
                   "cutoff_rule": "Youden on the same participants (in-sample)"}
            if marker == "Cyfra21_1":
                se, sp = cc.sens_spec_at(y, s, args.cyfra_clinical_cutoff)
                row.update({"clinical_cutoff": args.cyfra_clinical_cutoff, "sens_at_clinical_cutoff": se, "spec_at_clinical_cutoff": sp})
            rows.append(row)
    cy = pd.DataFrame(rows)
    for c in ("prevalence", "auc", "auc_delong_lo", "auc_delong_hi", "auc_boot_lo", "auc_boot_hi", "sensitivity", "specificity", "ppv", "npv", "accuracy"):
        cy[c] = cy[c].astype(float).round(4)
    cc.write_csv(cy, args.out_dir / "cyfra_auc.csv")
    for _, r in cy[cy["marker"] == "Cyfra21_1"].iterrows():
        numbers[f"cyfra_LCvsBLD_{r.dataset}_auc"] = {"value": r.auc, "source": f"{SCRIPT}:cyfra_auc.csv",
                                                   "note": f"n={r.n} ({r.n_case} LC vs {r.n_control} BLD), {r.n_missing_marker} missing; DeLong {r.auc_delong_lo:.3f}-{r.auc_delong_hi:.3f}; bootstrap {r.auc_boot_lo:.3f}-{r.auc_boot_hi:.3f}"}

    # paired comparison panel vs Cyfra on the held-out LC vs BLD participants
    rows = []
    v = samples.loc[scores["valid"].index]
    sel_v = v["group3"].isin(["LC", "BLD"]) & v["Cyfra21_1"].notna()
    heldout_ids = sel_v.index[sel_v]
    yv = (v.loc[heldout_ids, "group3"] == "LC").astype(int).values
    cyv = v.loc[heldout_ids, "Cyfra21_1"].values
    for m in cc.MODELS:
        sm = scores["valid"].loc[heldout_ids, m].values
        a1, a2, z, p = cc.delong_paired_test(yv, sm, cyv)
        rows.append({"dataset": "valid", "comparison": "LC_vs_BLD", "n": int(len(yv)), "n_case": int(yv.sum()), "n_control": int(len(yv) - yv.sum()),
                     "model": m, "auc_model": round(a1, 4), "auc_cyfra21_1": round(a2, 4), "auc_difference": round(a1 - a2, 4), "delong_z": z, "delong_p_two_sided": p})
    cc.write_csv(pd.DataFrame(rows), args.out_dir / "panel_vs_cyfra_paired_delong.csv")
    knn_row = [r for r in rows if r["model"] == "KNN"][0]
    numbers["heldout_LCvsBLD_KNN_vs_cyfra_delong_p"] = {"value": round(knn_row["delong_p_two_sided"], 4), "source": f"{SCRIPT}:panel_vs_cyfra_paired_delong.csv",
                                                       "note": f"paired DeLong, n=20; AUC KNN {knn_row['auc_model']} vs Cyfra21-1 {knn_row['auc_cyfra21_1']}"}

    # ---------------------------------------------------------------------------------
    # B2. covariates: logistic regressions
    # ---------------------------------------------------------------------------------
    print("covariate models ...")
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationError, PerfectSeparationWarning

    def fit_logit(df: pd.DataFrame, terms: list[str], model_id: str, outcome_label: str, dataset_label: str, score_name: str):
        d = df.dropna(subset=terms + ["y"]).copy()
        X = sm.add_constant(d[terms].astype(float), has_constant="add")
        y = d["y"].astype(int).values
        note = ""
        res = None
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerfectSeparationWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                res = sm.Logit(y, X).fit(disp=0, maxiter=200)
                if not res.mle_retvals.get("converged", True) or np.any(np.abs(res.params) > 15):
                    note = "quasi-complete separation / non-convergence: coefficients unstable"
            except (PerfectSeparationError, PerfectSeparationWarning, np.linalg.LinAlgError, ValueError) as exc:
                note = f"MLE failed ({type(exc).__name__}); fitted with ridge penalty alpha=1 (no Wald p-values)"
                res = None
        if res is None:
            res_r = sm.Logit(y, X).fit_regularized(alpha=1.0, L1_wt=0.0, disp=0, maxiter=500)
            params = pd.Series(np.asarray(res_r.params).ravel(), index=X.columns)
            fitted = 1 / (1 + np.exp(-(X.values @ params.values)))
            se = pd.Series(np.nan, index=X.columns)
            pvals = pd.Series(np.nan, index=X.columns)
            aic = np.nan
        else:
            params = res.params
            fitted = res.predict(X)
            se = res.bse
            pvals = res.pvalues
            aic = res.aic
        auc_in, lo, hi, _ = cc.delong_auc_ci(y, fitted)
        # leave-one-out cross-validated predictions
        loo = np.empty(len(y))
        for i in range(len(y)):
            mask = np.ones(len(y), bool)
            mask[i] = False
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    r_i = sm.Logit(y[mask], X.iloc[mask]).fit(disp=0, maxiter=200)
                    loo[i] = r_i.predict(X.iloc[[i]])[0]
                except Exception:
                    r_i = sm.Logit(y[mask], X.iloc[mask]).fit_regularized(alpha=1.0, L1_wt=0.0, disp=0, maxiter=500)
                    loo[i] = 1 / (1 + np.exp(-(X.iloc[[i]].values @ np.asarray(r_i.params).ravel())))[0]
        auc_loo = cc.fast_auc(y, loo)
        summary = {"model_id": model_id, "outcome": outcome_label, "dataset": dataset_label, "panel_score": score_name if any(t == "panel_score" for t in terms) else "",
                   "terms": "+".join(terms), "n": int(len(y)), "n_events": int(y.sum()), "aic": aic,
                   "auc_in_sample": round(auc_in, 4), "auc_in_sample_delong_lo": round(lo, 4), "auc_in_sample_delong_hi": round(hi, 4),
                   "auc_leave_one_out": round(auc_loo, 4), "note": note}
        coefs = []
        for t in X.columns:
            b = float(params[t])
            s_ = float(se[t]) if np.isfinite(se[t]) else np.nan
            coefs.append({"model_id": model_id, "outcome": outcome_label, "dataset": dataset_label, "term": t, "coef": b, "se": s_,
                          "z": (b / s_ if np.isfinite(s_) and s_ > 0 else np.nan), "p": float(pvals[t]) if np.isfinite(pvals[t]) else np.nan,
                          "odds_ratio": float(np.exp(b)), "or_lo": float(np.exp(b - 1.96 * s_)) if np.isfinite(s_) else np.nan,
                          "or_hi": float(np.exp(b + 1.96 * s_)) if np.isfinite(s_) else np.nan,
                          "unit": {"age": "per year", "sex_male": "male vs female", "smoking_ever": "ever vs never", "panel_score": "per unit predicted probability (0-1)"}.get(t, "")})
        return summary, coefs

    summaries, coef_rows = [], []
    allscores = pd.concat([scores["train"], scores["valid"]])
    base = samples.copy()
    base["sex_male"] = (base["sex"] == "male").astype(float)
    base["in_training_set"] = (base["set"] == "train")
    for score_name in ("KNN", "SVM"):
        base["panel_score"] = allscores.loc[base.index, score_name]
        designs = [
            # outcome, participants, dataset label, covariate sets
            ("LC_vs_BLD", base["group3"].isin(["LC", "BLD"]) & (base["set"] == "valid"), "valid (held-out, n=20)", [["age", "sex_male"], ["panel_score"], ["panel_score", "age", "sex_male"], ["age", "sex_male", "smoking_ever"], ["panel_score", "age", "sex_male", "smoking_ever"]]),
            ("LC_vs_BLD", base["group3"].isin(["LC", "BLD"]), "all patients (n=70; panel score in-sample for the 49 training participants)", [["age", "sex_male"], ["age", "sex_male", "smoking_ever"], ["panel_score"], ["panel_score", "age", "sex_male"], ["panel_score", "age", "sex_male", "smoking_ever"]]),
            ("LC_vs_nonLC", base["set"] == "valid", "valid (held-out, n=35)", [["age", "sex_male"], ["panel_score"], ["panel_score", "age", "sex_male"]]),
            ("LC_vs_nonLC", base["set"].notna(), "all participants (n=120; panel score in-sample for the 85 training participants)", [["age", "sex_male"], ["panel_score"], ["panel_score", "age", "sex_male"]]),
        ]
        for outcome, sel, dlabel, covsets in designs:
            d = base[sel].copy()
            case = "LC"
            d["y"] = (d["group3"] == case).astype(int)
            for terms in covsets:
                if "panel_score" not in terms and score_name == "SVM":
                    continue  # covariate-only models do not depend on the panel score
                mid = f"{outcome}|{dlabel.split(' (')[0]}|{'+'.join(terms)}|{score_name if 'panel_score' in terms else '-'}"
                summ, co = fit_logit(d, terms, mid, outcome, dlabel, score_name if "panel_score" in terms else "")
                summaries.append(summ)
                coef_rows.extend(co)
    cov_df = pd.DataFrame(summaries)
    cc.write_csv(cov_df, args.out_dir / "covariate_models.csv")
    coef_df = pd.DataFrame(coef_rows)
    for c in ("coef", "se", "z", "p", "odds_ratio", "or_lo", "or_hi"):
        coef_df[c] = coef_df[c].astype(float).round(4)
    cc.write_csv(coef_df, args.out_dir / "covariate_model_coefficients.csv")
    for _, r in cov_df[(cov_df["panel_score"].isin(["KNN", ""]))].iterrows():
        key = "cov_" + r.model_id.replace("|", "_").replace("+", "_").replace(" ", "")
        numbers[key] = {"value": r.auc_in_sample, "source": f"{SCRIPT}:covariate_models.csv",
                        "note": f"n={r.n}, events={r.n_events}; in-sample AUC (DeLong {r.auc_in_sample_delong_lo}-{r.auc_in_sample_delong_hi}); leave-one-out AUC {r.auc_leave_one_out}; {r.note}"}

    # ---------------------------------------------------------------------------------
    # B3. covariate balance by group
    # ---------------------------------------------------------------------------------
    print("covariate balance ...")
    rows = []
    for dlabel, sel in (("all", samples["set"].notna()), ("train", samples["set"] == "train"), ("valid", samples["set"] == "valid")):
        d = samples[sel]
        for g in cc.GROUP3_ORDER:
            dg = d[d["group3"] == g]
            rows.append({"dataset": dlabel, "group": g, "n": len(dg),
                         "age_mean": dg["age"].mean(), "age_sd": dg["age"].std(ddof=1), "age_median": dg["age"].median(),
                         "age_q1": dg["age"].quantile(0.25), "age_q3": dg["age"].quantile(0.75), "age_min": dg["age"].min(), "age_max": dg["age"].max(),
                         "n_male": int((dg["sex"] == "male").sum()), "n_female": int((dg["sex"] == "female").sum()), "pct_male": 100 * (dg["sex"] == "male").mean(),
                         "n_smoking_ever": int((dg["smoking_ever"] == 1).sum()), "n_smoking_never": int((dg["smoking_ever"] == 0).sum()),
                         "n_smoking_unknown": int(dg["smoking_ever"].isna().sum()),
                         "smoking_note": "not recorded for HC" if g == "HC" else "free-text history dichotomised ever/never; pack-years not consistently recorded",
                         "cyfra_median": dg["Cyfra21_1"].median(), "cyfra_n_missing": int(dg["Cyfra21_1"].isna().sum())})
    bal = pd.DataFrame(rows)
    # tests on all participants
    d = samples
    tests = []
    kw = stats.kruskal(*[d.loc[d["group3"] == g, "age"].dropna() for g in cc.GROUP3_ORDER])
    tests.append({"variable": "age", "test": "Kruskal-Wallis HC/BLD/LC", "statistic": kw.statistic, "p": kw.pvalue})
    for a, b in (("LC", "BLD"), ("LC", "HC"), ("BLD", "HC")):
        mw = stats.mannwhitneyu(d.loc[d["group3"] == a, "age"], d.loc[d["group3"] == b, "age"])
        tests.append({"variable": "age", "test": f"Mann-Whitney {a} vs {b}", "statistic": mw.statistic, "p": mw.pvalue})
    ct = pd.crosstab(d["group3"], d["sex"]).reindex(cc.GROUP3_ORDER)
    chi = stats.chi2_contingency(ct.values)
    tests.append({"variable": "sex", "test": "chi-square HC/BLD/LC", "statistic": chi[0], "p": chi[1]})
    for a, b in (("LC", "BLD"), ("LC", "HC"), ("BLD", "HC")):
        sub = ct.loc[[a, b]]
        fe = stats.fisher_exact(sub.values)
        tests.append({"variable": "sex", "test": f"Fisher exact {a} vs {b}", "statistic": fe[0], "p": fe[1]})
    sm_ct = pd.crosstab(d.loc[d["group3"].isin(["LC", "BLD"]), "group3"], d.loc[d["group3"].isin(["LC", "BLD"]), "smoking_ever"])
    if sm_ct.shape == (2, 2):
        fe = stats.fisher_exact(sm_ct.values)
        tests.append({"variable": "smoking_ever", "test": "Fisher exact LC vs BLD (patients with recorded history)", "statistic": fe[0], "p": fe[1]})
    # split composition check (stratification)
    for g in cc.STAGE_ORDER:
        n_all = int((samples["stage"] == g).sum())
        n_tr = int(((samples["stage"] == g) & (samples["set"] == "train")).sum())
        tests.append({"variable": f"split composition {g}", "test": "training share (expected ~0.70 if stratified)", "statistic": n_tr, "p": np.nan,
                      "note": f"{n_tr}/{n_all} = {n_tr / n_all:.3f} in training"})
    tests_df = pd.DataFrame(tests)
    bal_out = pd.concat([bal, tests_df], axis=0, ignore_index=True)
    cc.write_csv(bal_out, args.out_dir / "covariate_balance.csv")
    for _, r in tests_df.iterrows():
        if np.isfinite(r.get("p", np.nan)):
            numbers["balance_" + (r.variable + "_" + r.test).replace(" ", "_").replace("/", "_")] = {"value": round(float(r.p), 4), "source": f"{SCRIPT}:covariate_balance.csv", "note": "p-value"}
    for _, r in bal[bal["dataset"] == "all"].iterrows():
        numbers[f"balance_{r.group}_age_mean"] = {"value": round(float(r.age_mean), 1), "source": f"{SCRIPT}:covariate_balance.csv", "note": f"SD {r.age_sd:.1f}, median {r.age_median:.0f}, n={r.n}"}
        numbers[f"balance_{r.group}_pct_male"] = {"value": round(float(r.pct_male), 1), "source": f"{SCRIPT}:covariate_balance.csv", "note": f"{r.n_male} male / {r.n_female} female"}
        if r.group != "HC":
            numbers[f"balance_{r.group}_smoking_ever_n"] = {"value": int(r.n_smoking_ever), "source": f"{SCRIPT}:covariate_balance.csv", "note": f"never {r.n_smoking_never}, unknown {r.n_smoking_unknown}"}

    # clinical covariates available
    numbers["clinical_covariates_available"] = {
        "value": "HC sheet: SampleID, Group, TubeID, SampleType, sex, age. Patient sheet: + Disease, smoking history (free text), pathological TNM stage (free text), CEA, CA125, Cyfra21-1, NSE. Not recorded: treatment, collection conditions, fasting, sampling date, batch/plate.",
        "source": f"{SCRIPT}:clinical_info_120Samples.xlsx", "note": "Cyfra21-1 missing for 1 BLD participant (Ca010); smoking missing for 5 patients"}

    # ---------------------------------------------------------------------------------
    # preview figures
    # ---------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (dset, cname) in zip(axes, (("train", "LC_vs_nonLC"), ("valid", "LC_vs_nonLC"), ("valid", "LC_vs_BLD"))):
        case_g, ctrl_g = cc.COMPARISONS[cname]
        y, sub = subset_rows(samples, scores[dset], cname, case_g, ctrl_g)
        for m in cc.MODELS:
            fpr, tpr, _ = roc_curve(y, sub[m].values, drop_intermediate=False)
            ax.plot(fpr, tpr, lw=1.2, label=f"{m} AUC {cc.fast_auc(y, sub[m].values):.3f}")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_title(f"original models, {dset}, {cname} (n={len(y)})", fontsize=9)
        ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity"); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(args.fig_dir / "original_roc_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    knn_heldout = scores["valid"].loc[heldout_ids, "KNN"].values
    fpr, tpr, _ = roc_curve(yv, knn_heldout, drop_intermediate=False)
    ax.plot(fpr, tpr, lw=1.4, label=f"KNN panel AUC {cc.fast_auc(yv, knn_heldout):.3f}")
    fpr, tpr, _ = roc_curve(yv, cyv, drop_intermediate=False)
    ax.plot(fpr, tpr, lw=1.4, label=f"Cyfra21-1 AUC {cc.fast_auc(yv, cyv):.3f}")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_title("held-out LC (14) vs BLD (6)", fontsize=9)
    ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity"); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(args.fig_dir / "lc_vs_bld_heldout_panel_vs_cyfra.png", dpi=150)
    plt.close(fig)

    cc.update_numbers(args.out_dir / "numbers_s2_6.json", numbers, SCRIPT)
    print("done:", SCRIPT)


if __name__ == "__main__":
    main()
