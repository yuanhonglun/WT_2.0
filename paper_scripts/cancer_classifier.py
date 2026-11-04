"""Leak-free re-run of the serum diagnostic-panel chain in Python/scikit-learn
(work package S2.6, deliverable C; decisions D4-1 = A and D4-2 = A; reviewer points R2-7 (i)-(vi), R4-2).

Every learned quantity is derived inside the 85-participant training set only:
  imputation constant (1/5 of the smallest detected training intensity, as in the original pipeline),
  log2 transform, z-scaling parameters, univariate differential screening, LASSO x 50 seeds
  (features reproduced in >= 70 % of the seeds kept), correlation filter (|r| > 0.9 -> drop one),
  multi-algorithm importance consensus (top 20), hyper-parameter tuning of KNN / LR / SVM / RF / NB by
  3 x 10-fold stratified cross-validation, and the decision cut-off (Youden index on the training ROC).
The 35 held-out participants are scored exactly once per model.
The same framework is run separately for LC vs BLD (held-out main conclusion, same acquisition batch),
LC vs HC and LC vs non-LC (batch-confounded, exploratory). A no-screen sensitivity run is added for LC vs BLD
because the strict univariate screen passes very few transitions in that comparison.

Inputs:
  Supplementary Table 9.xlsx     3,541 transitions x 120 participants (NaN = not detected); MetaboLights MTBLS15138
  20250731.ml_results/ml_results/1.group_split/LC.{train,valid}.group.txt   the original 85 / 35 split (reused unchanged)
  Supplementary Table 8.xlsx     sheet Cohort_assay_features: signal_type / discovery annotation of each transition
  Supplementary Table 10.xlsx    the published 20-member panel (for the overlap table)
  fig 7/data/roc_valid.txt, roc_train.txt, merge_metrics.details.txt  original model scores and metrics (for redo_vs_original)

Outputs (analysis/cancer/):
  redo_selected_features_<comparison>.csv   screening statistics, LASSO reproducibility, correlation filter,
                                            importance ranks, final panel flag, Table 8 provenance, mz_source placeholder
  redo_model_metrics.csv                    per comparison x model x dataset: AUC with DeLong and bootstrap 95 % CI,
                                            cut-off, sensitivity, specificity, PPV, NPV, accuracy, confusion matrix
  redo_predictions_{train,valid}_<comparison>.csv  per-participant probabilities (ROC source data)
  redo_vs_original.csv                      original vs redone AUC / sensitivity / specificity, panel overlap
  redo_run_info.json                        seeds, software versions, step counts, screen tier used, runtime
  numbers_s2_6.json (updated)
Preview figures (figures/cancer_preview/): redo_roc_<comparison>.png, redo_lasso_<comparison>.png
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import os
import platform
import time
import warnings
from pathlib import Path

# joblib/loky probes the physical core count with `wmic`, which is absent on recent Windows builds
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cancer_common as cc  # noqa: E402

SCRIPT = "cancer_classifier.py"
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table9", type=Path, default=cc.DEFAULTS["table9"])
    ap.add_argument("--table8", type=Path, default=cc.DEFAULTS["table8"])
    ap.add_argument("--table10", type=Path, default=cc.DEFAULTS["table10"])
    ap.add_argument("--ml-dir", type=Path, default=cc.DEFAULTS["ml_dir"])
    ap.add_argument("--fig7-data", type=Path, default=cc.DEFAULTS["fig7_data"])
    ap.add_argument("--out-dir", type=Path, default=cc.DEFAULTS["out_dir"])
    ap.add_argument("--fig-dir", type=Path, default=cc.DEFAULTS["fig_dir"])
    ap.add_argument("--seed", type=int, default=20261020, help="master random seed (models, CV folds, bootstrap)")
    ap.add_argument("--n-lasso-seeds", type=int, default=50)
    ap.add_argument("--reproducibility", type=float, default=0.70, help="keep features selected in >= this fraction of LASSO seeds")
    ap.add_argument("--corr", type=float, default=0.9, help="|Pearson r| above which one member of a pair is dropped")
    ap.add_argument("--top", type=int, default=20, help="panel size")
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--lfc", type=float, default=1.0)
    ap.add_argument("--min-candidates", type=int, default=20,
                    help="if fewer transitions pass the FDR screen, relax to nominal p < 0.05 (tier 2), then to no screen (tier 3)")
    ap.add_argument("--comparisons", default="LC_vs_BLD,LC_vs_HC,LC_vs_nonLC")
    ap.add_argument("--no-sensitivity-noscreen", action="store_true", help="skip the LC_vs_BLD_noscreen sensitivity run")
    ap.add_argument("--no-sensitivity-detected", action="store_true",
                    help="skip the LC_vs_HC_detected80 sensitivity run (only transitions detected in >= 80 %% of the training participants, i.e. without presence/absence markers)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--keep-internal-standards", action="store_true", help="do not remove the internal-standard transitions from the feature space")
    return ap.parse_args()


# --------------------------------------------------------------------------------------
def scale_0_100(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    if np.nanmax(x) - np.nanmin(x) <= 0:
        return np.zeros_like(x)
    return 100 * (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def univariate_screen(raw_tr: pd.DataFrame, log_tr: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Welch t-test on log2 (imputed) intensities; log2FC of raw (imputed) group means; BH-FDR."""
    case = log_tr.loc[:, y == 1]
    ctrl = log_tr.loc[:, y == 0]
    t, p = stats.ttest_ind(case, ctrl, axis=1, equal_var=False)
    lfc = np.log2(raw_tr.loc[:, y == 1].mean(axis=1) / raw_tr.loc[:, y == 0].mean(axis=1))
    q = multipletests(np.nan_to_num(p, nan=1.0), method="fdr_bh")[1]
    return pd.DataFrame({"screen_t": t, "screen_p": p, "screen_fdr": q, "log2fc_case_vs_control": lfc.values}, index=log_tr.index)


def lasso_reproducibility(X: np.ndarray, y: np.ndarray, n_seeds: int, seed0: int, n_jobs: int):
    """L1-logistic regression with 5-fold CV choice of C, repeated over n_seeds fold assignments."""
    Cs = np.logspace(-2, 1.5, 25)
    n_feat = X.shape[1]
    counts = np.zeros(n_feat)
    abscoef = np.zeros(n_feat)
    chosen_C = []
    seeds = [seed0 + i for i in range(n_seeds)]
    for s in seeds:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        # random_state also seeds liblinear's internal sample shuffling, otherwise near-zero coefficients can flip between runs
        model = LogisticRegressionCV(Cs=Cs, cv=cv, penalty="l1", solver="liblinear", scoring="neg_log_loss",
                                     max_iter=5000, refit=True, n_jobs=n_jobs, random_state=s)
        model.fit(X, y)
        coef = model.coef_.ravel()
        counts += (coef != 0)
        abscoef += np.abs(coef)
        chosen_C.append(float(model.C_[0]))
    return counts, abscoef / n_seeds, seeds, chosen_C


def correlation_filter(Z: pd.DataFrame, order: list[str], threshold: float):
    """Greedy: walk features in priority order, drop any whose |r| with an already-kept feature exceeds threshold."""
    corr = Z[order].corr().abs()
    kept, dropped = [], {}
    for f in order:
        partner = None
        for k in kept:
            if corr.loc[f, k] > threshold:
                partner = (k, float(corr.loc[f, k]))
                break
        if partner is None:
            kept.append(f)
        else:
            dropped[f] = partner
    return kept, dropped


def roc_filter_importance(Z: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """caret::filterVarImp-style importance: per-feature AUC folded to [0.5, 1]."""
    out = []
    for c in Z.columns:
        a = cc.fast_auc(y, Z[c].values)
        out.append(max(a, 1 - a))
    return np.asarray(out)


def model_grid(name: str, n_minority: int, n_features: int, seed: int):
    if name == "KNN":
        ks = [k for k in (3, 5, 7, 9, 11, 13, 15) if k <= max(3, n_minority)]
        return KNeighborsClassifier(), {"n_neighbors": ks, "weights": ["uniform", "distance"]}
    if name == "LR":
        return LogisticRegression(penalty="l2", max_iter=5000, solver="lbfgs"), {"C": np.logspace(-3, 2, 11)}
    if name == "SVM":
        return SVC(kernel="rbf", probability=True, random_state=seed), {"C": [0.25, 0.5, 1, 2, 4, 8, 16], "gamma": ["scale", 0.01, 0.03, 0.1, 0.3]}
    if name == "RF":
        mf = sorted({m for m in (1, 2, 3, max(1, int(round(np.sqrt(n_features)))), max(1, n_features // 3), max(1, n_features // 2)) if m <= n_features})
        return RandomForestClassifier(n_estimators=500, random_state=seed), {"max_features": mf}
    if name == "NB":
        return GaussianNB(), {"var_smoothing": [1e-9, 1e-6, 1e-3, 1e-2, 1e-1, 1.0]}
    raise ValueError(name)


def missingness_pattern(miss_case: pd.Series, miss_ctrl: pd.Series) -> pd.Series:
    """Label a transition by its training-set detection pattern (presence/absence marker or not)."""
    d = miss_case - miss_ctrl
    return pd.Series(np.select(
        [(miss_case >= 0.9) & (miss_ctrl <= 0.1), (miss_ctrl >= 0.9) & (miss_case <= 0.1), d >= 0.5, d <= -0.5],
        ["absent in cases, present in controls", "present in cases, absent in controls",
         "mostly missing in cases only", "mostly missing in controls only"], default=""), index=miss_case.index)


def run_comparison(cname: str, case_g, ctrl_g, screen_mode: str, detect_min: float | None, t9: cc.Table9, split: pd.DataFrame, t8: pd.DataFrame,
                   table10: pd.DataFrame, orig_panel: list[str], args, info: dict):
    t0 = time.time()
    print(f"\n=== {cname} (screen mode: {screen_mode}; detection filter: {detect_min}) ===")
    g3 = t9.group3
    groups = list(case_g) + list(ctrl_g)
    tr_ids = [s for s in split.index[split["set"] == "train"] if g3[s] in groups]
    va_ids = [s for s in split.index[split["set"] == "valid"] if g3[s] in groups]
    y_tr = np.array([1 if g3[s] in case_g else 0 for s in tr_ids])
    y_va = np.array([1 if g3[s] in case_g else 0 for s in va_ids])
    print(f"  training n={len(tr_ids)} (case {y_tr.sum()}, control {len(y_tr) - y_tr.sum()}); validation n={len(va_ids)} (case {y_va.sum()}, control {len(y_va) - y_va.sum()})")

    # ---- feature space (label-free decisions only)
    feats = list(t9.data.index)
    n_all = len(feats)
    if not args.keep_internal_standards:
        feats = [f for f in feats if not t9.meta.loc[f, "is_internal_standard"]]
    n_after_is = len(feats)
    raw_tr = t9.data.loc[feats, tr_ids]
    raw_va = t9.data.loc[feats, va_ids]
    # drop transitions missing in > 50 % of BOTH training groups (original rule, training only)
    miss_case = raw_tr.loc[:, y_tr == 1].isna().mean(axis=1)
    miss_ctrl = raw_tr.loc[:, y_tr == 0].isna().mean(axis=1)
    keep = ~((miss_case > 0.5) & (miss_ctrl > 0.5))
    if detect_min is not None:   # label-free detection filter on the training participants (sensitivity run)
        keep &= raw_tr.notna().mean(axis=1) >= detect_min
    raw_tr, raw_va = raw_tr[keep], raw_va[keep]
    feats = list(raw_tr.index)
    miss_case, miss_ctrl = miss_case[keep], miss_ctrl[keep]
    pattern = missingness_pattern(miss_case, miss_ctrl)
    # imputation constant and transforms from the training set only
    fill = float(np.nanmin(raw_tr.values)) / 5.0
    imp_tr = raw_tr.fillna(fill)
    imp_va = raw_va.fillna(fill)
    log_tr = np.log2(imp_tr)
    log_va = np.log2(imp_va)
    mu = log_tr.mean(axis=1)
    sd = log_tr.std(axis=1, ddof=1).replace(0, np.nan)
    Z_tr = ((log_tr.T - mu) / sd).T.fillna(0.0)
    Z_va = ((log_va.T - mu) / sd).T.fillna(0.0)

    # ---- univariate screen (training only)
    scr = univariate_screen(imp_tr, log_tr, y_tr)
    tier1 = scr.index[(scr["screen_fdr"] < args.fdr) & (scr["log2fc_case_vs_control"].abs() > args.lfc)]
    tier2 = scr.index[(scr["screen_p"] < 0.05) & (scr["log2fc_case_vs_control"].abs() > args.lfc)]
    if screen_mode == "none":
        cand, tier = list(scr.index), "tier 3: no univariate screen (sensitivity run)"
    elif len(tier1) >= args.min_candidates:
        cand, tier = list(tier1), f"tier 1: BH-FDR < {args.fdr} and |log2FC| > {args.lfc}"
    elif len(tier2) >= args.min_candidates:
        cand, tier = list(tier2), f"tier 2: nominal p < 0.05 and |log2FC| > {args.lfc} (tier 1 passed only {len(tier1)} < {args.min_candidates})"
    else:
        cand, tier = list(scr.index), f"tier 3: no screen (tier 1 passed {len(tier1)}, tier 2 passed {len(tier2)} < {args.min_candidates})"
    print(f"  features: {n_all} -> {n_after_is} after removing internal standards -> {len(feats)} after the missingness rule; "
          f"tier-1 screen {len(tier1)}, tier-2 {len(tier2)}; used {len(cand)} candidates [{tier}]")

    # ---- LASSO x seeds
    X = Z_tr.loc[cand].T.values
    counts, mean_abs, seeds, chosen_C = lasso_reproducibility(X, y_tr, args.n_lasso_seeds, args.seed + 1000, args.jobs)
    freq = pd.Series(counts / args.n_lasso_seeds, index=cand)
    mabs = pd.Series(mean_abs, index=cand)
    kept_lasso = list(freq.index[freq >= args.reproducibility])
    lasso_note = ""
    if len(kept_lasso) < 2:
        order = freq.sort_values(ascending=False).index
        kept_lasso = list(order[:5])
        lasso_note = f"fewer than 2 features reached reproducibility >= {args.reproducibility}; fell back to the 5 most frequently selected"
    print(f"  LASSO: {len(kept_lasso)} features reproduced in >= {args.reproducibility:.0%} of {args.n_lasso_seeds} seeds {lasso_note}")

    # ---- correlation filter (priority: reproducibility, then mean |coef|)
    prio = sorted(kept_lasso, key=lambda f: (-freq[f], -mabs[f], f))
    kept_corr, dropped = correlation_filter(Z_tr.T, prio, args.corr)
    print(f"  correlation filter |r| > {args.corr}: {len(kept_corr)} kept, {len(dropped)} dropped")

    # ---- multi-algorithm importance consensus (training only)
    Zk = Z_tr.loc[kept_corr].T
    imp_roc = scale_0_100(roc_filter_importance(Zk, y_tr))
    rf = RandomForestClassifier(n_estimators=500, random_state=args.seed).fit(Zk.values, y_tr)
    imp_rf = scale_0_100(rf.feature_importances_)
    lr = LogisticRegression(penalty="l2", C=1.0, max_iter=5000).fit(Zk.values, y_tr)
    imp_lr = scale_0_100(np.abs(lr.coef_.ravel()))
    imp = pd.DataFrame({"imp_SVM_roc": imp_roc, "imp_RF_gini": imp_rf, "imp_LR_abscoef": imp_lr, "imp_KNN_roc": imp_roc}, index=kept_corr)
    imp["importance_score"] = imp.mean(axis=1)
    imp = imp.sort_values(["importance_score", "imp_RF_gini"], ascending=False)
    imp["importance_rank"] = np.arange(1, len(imp) + 1)
    panel = list(imp.index[:args.top])
    print(f"  panel: {len(panel)} transitions; overlap with the original 20: {len(set(panel) & set(orig_panel))}")

    # ---- feature table
    ft = scr.loc[feats].copy()
    ft.insert(0, "transition_id", ft.index)
    ft["passed_screen_tier1"] = ft.index.isin(tier1)
    ft["passed_screen_tier2"] = ft.index.isin(tier2)
    ft["candidate_for_lasso"] = ft.index.isin(cand)
    ft["lasso_selection_frequency"] = freq.reindex(ft.index)
    ft["lasso_mean_abs_coef"] = mabs.reindex(ft.index)
    ft["passed_reproducibility"] = ft.index.isin(kept_lasso)
    ft["dropped_by_correlation_with"] = [f"{dropped[f][0]} (r={dropped[f][1]:.3f})" if f in dropped else "" for f in ft.index]
    for c in ("imp_SVM_roc", "imp_RF_gini", "imp_LR_abscoef", "imp_KNN_roc", "importance_score", "importance_rank"):
        ft[c] = imp[c].reindex(ft.index)
    ft["in_final_panel"] = ft.index.isin(panel)
    ft["in_original_panel"] = ft.index.isin(orig_panel)
    ft["missing_frac_train_case"] = miss_case.reindex(ft.index)
    ft["missing_frac_train_control"] = miss_ctrl.reindex(ft.index)
    ft["batch_missingness_pattern_train"] = pattern.reindex(ft.index)
    ft["missing_frac_valid"] = raw_va.isna().mean(axis=1).reindex(ft.index)
    ft = ft[ft["candidate_for_lasso"] | ft["in_original_panel"]].copy()   # keep the table readable
    anno = cc.annotate_transitions(list(ft.index), t9, t8, table10).set_index("transition_id")
    ft = ft.join(anno, how="left")
    ft = ft.sort_values(["in_final_panel", "importance_rank", "lasso_selection_frequency", "screen_p"], ascending=[False, True, False, True])
    ft.insert(1, "comparison", cname)
    ft.insert(2, "screen_tier_used", tier)
    cc.write_csv(ft.reset_index(drop=True), args.out_dir / f"redo_selected_features_{cname}.csv")

    # ---- models
    Xtr = Z_tr.loc[panel].T.values
    Xva = Z_va.loc[panel].T.values
    n_min = int(min(y_tr.sum(), len(y_tr) - y_tr.sum()))
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=args.seed)
    oof_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
    pred_tr = pd.DataFrame({"sample": tr_ids, "group3": [g3[s] for s in tr_ids], "stage": [t9.stage[s] for s in tr_ids], "y": y_tr})
    pred_va = pd.DataFrame({"sample": va_ids, "group3": [g3[s] for s in va_ids], "stage": [t9.stage[s] for s in va_ids], "y": y_va})
    metric_rows = []
    best_params = {}
    n_pa = int(pattern.reindex(panel).isin(["absent in cases, present in controls", "present in cases, absent in controls"]).sum())
    n_pa_loose = int(pattern.reindex(panel).ne("").sum())
    collapse_note = ""
    if len(panel) < args.top:
        collapse_note = (f"panel smaller than {args.top}: {len(kept_lasso)} LASSO-reproducible features, {len(kept_corr)} after the |r| > {args.corr} filter"
                         + ("; the LASSO-retained features were mutually redundant presence/absence markers" if len(dropped) >= len(kept_lasso) // 2 and n_pa > 0 else ""))
    for name in cc.MODELS:
        est, grid = model_grid(name, n_min, len(panel), args.seed)
        # ROC AUC is invariant to the ridge strength of a logistic model when the training classes are (nearly)
        # separable, so the LR grid is scored by log-loss; the other models are scored by ROC AUC
        scoring = "neg_log_loss" if name == "LR" else "roc_auc"
        gs = GridSearchCV(est, grid, scoring=scoring, cv=cv, n_jobs=args.jobs, refit=True)
        gs.fit(Xtr, y_tr)
        best = gs.best_estimator_
        best_params[name] = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)) for k, v in gs.best_params_.items()}
        p_tr = best.predict_proba(Xtr)[:, 1]
        p_oof = cross_val_predict(gs.best_estimator_, Xtr, y_tr, cv=oof_cv, method="predict_proba")[:, 1]
        p_va = best.predict_proba(Xva)[:, 1]          # the held-out set is scored exactly once
        cut_in = cc.youden_cutoff(y_tr, p_tr)
        cut_oof = cc.youden_cutoff(y_tr, p_oof)
        pred_tr[f"prob_{name}"] = p_tr
        pred_tr[f"prob_oof_{name}"] = p_oof
        pred_va[f"prob_{name}"] = p_va
        for dset, yy, pp in (("train_insample", y_tr, p_tr), ("train_oof_10fold", y_tr, p_oof), ("valid", y_va, p_va)):
            blk = cc.metrics_block(yy, pp, cut_in, n_boot=args.n_boot, seed=args.seed)
            se_oof, sp_oof = cc.sens_spec_at(yy, pp, cut_oof)
            metric_rows.append({"comparison": cname, "screen_mode": screen_mode, "detection_filter_train": detect_min if detect_min is not None else "",
                                "screen_tier_used": tier, "n_candidates": len(cand),
                                "n_lasso_kept": len(kept_lasso), "n_after_correlation": len(kept_corr), "panel_size": len(panel),
                                "panel_presence_absence_members": n_pa, "panel_members_with_group_specific_missingness": n_pa_loose,
                                "model": name, "best_params": json.dumps(best_params[name]), "tuning_scoring": scoring, "cv_score_best": round(float(gs.best_score_), 4),
                                "dataset": dset, **blk, "cutoff_rule": "Youden index on in-sample training probabilities",
                                "cutoff_oof_youden": cut_oof, "sensitivity_at_oof_cutoff": se_oof, "specificity_at_oof_cutoff": sp_oof, "note": collapse_note})
        v = [r for r in metric_rows if r["model"] == name and r["dataset"] == "valid"][0]
        o = [r for r in metric_rows if r["model"] == name and r["dataset"] == "train_oof_10fold"][0]
        print(f"  {name:3s} best {best_params[name]} | train OOF AUC {o['auc']:.3f} | valid AUC {v['auc']:.3f} "
              f"[DeLong {v['auc_delong_lo']:.3f}-{v['auc_delong_hi']:.3f}] sens {v['sensitivity']:.3f} spec {v['specificity']:.3f}")
    cc.write_csv(pred_tr, args.out_dir / f"redo_predictions_train_{cname}.csv")
    cc.write_csv(pred_va, args.out_dir / f"redo_predictions_valid_{cname}.csv")

    info[cname] = {
        "screen_mode": screen_mode, "detection_filter_train": detect_min, "screen_tier_used": tier, "n_train": len(tr_ids), "n_valid": len(va_ids),
        "panel_presence_absence_members": n_pa, "panel_members_with_group_specific_missingness": n_pa_loose, "note": collapse_note,
        "panel_missingness_pattern": {p: pattern[p] for p in panel},
        "n_case_train": int(y_tr.sum()), "n_control_train": int(len(y_tr) - y_tr.sum()),
        "n_case_valid": int(y_va.sum()), "n_control_valid": int(len(y_va) - y_va.sum()),
        "n_transitions_total": n_all, "n_after_removing_internal_standards": n_after_is, "n_after_missingness_rule": len(feats),
        "imputation_constant_training_min_over_5": fill, "n_screen_tier1": int(len(tier1)), "n_screen_tier2": int(len(tier2)),
        "n_candidates": len(cand), "lasso_seeds": seeds, "lasso_C_grid": "logspace(-2, 1.5, 25)", "lasso_C_chosen_per_seed": chosen_C,
        "n_lasso_kept": len(kept_lasso), "lasso_note": lasso_note, "n_after_correlation": len(kept_corr), "correlation_dropped": {k: list(v) for k, v in dropped.items()},
        "panel": panel, "panel_overlap_with_original": sorted(set(panel) & set(orig_panel)), "best_params": best_params,
        "runtime_s": round(time.time() - t0, 1),
    }
    return pd.DataFrame(metric_rows), pred_tr, pred_va, panel


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    np.random.seed(args.seed)

    t9 = cc.load_table9(args.table9)
    split = cc.load_split(args.ml_dir)
    t8 = cc.load_table8_map(args.table8)
    table10 = cc.load_table10(args.table10)
    orig_panel = cc.load_original_panel(args.ml_dir)["transition_id"].tolist()[:20]
    assert set(orig_panel) == set(table10["transition_id"]), "Table 10 and the collaborator's panel_anno disagree"
    assert set(split.index) == set(t9.data.columns)

    info = {"seed": args.seed, "lasso_seed_base": args.seed + 1000, "python": platform.python_version(), "sklearn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "statement": "All random processes (LASSO fold assignments, RF, SVM Platt scaling, CV folds, bootstrap) use fixed seeds derived from --seed; re-running the script reproduces every number."}
    runs = [(c, cc.COMPARISONS[c][0], cc.COMPARISONS[c][1], "auto", None) for c in args.comparisons.split(",")]
    if not args.no_sensitivity_noscreen:
        runs.append(("LC_vs_BLD_noscreen", cc.COMPARISONS["LC_vs_BLD"][0], cc.COMPARISONS["LC_vs_BLD"][1], "none", None))
    if not args.no_sensitivity_detected:
        runs.append(("LC_vs_HC_detected80", cc.COMPARISONS["LC_vs_HC"][0], cc.COMPARISONS["LC_vs_HC"][1], "auto", 0.8))
    all_metrics, panels, preds_valid = [], {}, {}
    for cname, case_g, ctrl_g, mode, detect_min in runs:
        met, ptr, pva, panel = run_comparison(cname, case_g, ctrl_g, mode, detect_min, t9, split, t8, table10, orig_panel, args, info)
        all_metrics.append(met)
        panels[cname] = panel
        preds_valid[cname] = pva
    met = pd.concat(all_metrics, ignore_index=True)
    for c in ("prevalence", "auc", "auc_delong_lo", "auc_delong_hi", "auc_boot_lo", "auc_boot_hi", "auc_se_delong", "sensitivity", "specificity", "ppv", "npv", "accuracy", "sensitivity_at_oof_cutoff", "specificity_at_oof_cutoff"):
        met[c] = met[c].astype(float).round(4)
    assert met["auc"].dropna().between(0, 1).all()
    cc.write_csv(met, args.out_dir / "redo_model_metrics.csv")

    # ---- redo vs original
    scores = cc.load_original_scores(args.fig7_data)
    published = cc.load_published_metrics(args.fig7_data)
    cutoffs = cc.load_original_cutoffs(args.ml_dir)
    g3 = t9.group3
    rows = []
    for cname in panels:
        base = next(k for k in sorted(cc.COMPARISONS, key=len, reverse=True) if cname.startswith(k))
        case_g, ctrl_g = cc.COMPARISONS[base]
        for m in cc.MODELS:
            for dset in ("train", "valid"):
                sc = scores[dset]
                sel = [s for s in sc.index if g3[s] in list(case_g) + list(ctrl_g)]
                yo = np.array([1 if g3[s] in case_g else 0 for s in sel])
                so = sc.loc[sel, m].values
                o_auc, o_lo, o_hi, _ = cc.delong_auc_ci(yo, so)
                o_se, o_sp = cc.sens_spec_at(yo, so, cutoffs[m])
                r = met[(met["comparison"] == cname) & (met["model"] == m) & (met["dataset"] == ("train_insample" if dset == "train" else "valid"))].iloc[0]
                pub = published[(published["Method"] == m) & (published["DType"] == dset)].iloc[0] if base == "LC_vs_nonLC" else None
                rows.append({"comparison": cname, "model": m, "dataset": dset, "n": int(len(yo)),
                             "original_auc": round(o_auc, 4), "original_auc_delong_lo": round(o_lo, 4), "original_auc_delong_hi": round(o_hi, 4),
                             "original_sensitivity": round(o_se, 4), "original_specificity": round(o_sp, 4),
                             "original_published_auc": (pub["AUC"] if pub is not None else np.nan),
                             "original_note": "original scores (candidate pool derived on all 120 participants: leakage); subset metrics use the original cut-off" if base != "LC_vs_nonLC" else "original scores; candidate pool derived on all 120 participants (leakage)",
                             "redo_auc": r["auc"], "redo_auc_delong_lo": r["auc_delong_lo"], "redo_auc_delong_hi": r["auc_delong_hi"],
                             "redo_auc_boot_lo": r["auc_boot_lo"], "redo_auc_boot_hi": r["auc_boot_hi"],
                             "redo_sensitivity": r["sensitivity"], "redo_specificity": r["specificity"], "redo_panel_size": r["panel_size"],
                             "redo_screen_tier": r["screen_tier_used"],
                             "panel_overlap_with_original_n": len(set(panels[cname]) & set(orig_panel)),
                             "panel_overlap_with_original_ids": ";".join(sorted(set(panels[cname]) & set(orig_panel)))})
    cmp_df = pd.DataFrame(rows)
    cc.write_csv(cmp_df, args.out_dir / "redo_vs_original.csv")

    info["total_runtime_s"] = round(time.time() - t_start, 1)
    with open(args.out_dir / "redo_run_info.json", "w", encoding="utf-8") as fh:
        json.dump(cc._jsonable(info), fh, indent=2, ensure_ascii=False)

    # ---- numbers
    numbers = {"redo_random_seed": {"value": args.seed, "source": f"{SCRIPT}:redo_run_info.json", "note": info["statement"]},
               "redo_software": {"value": f"Python {platform.python_version()}, scikit-learn {sklearn.__version__}, numpy {np.__version__}, pandas {pd.__version__}", "source": f"{SCRIPT}:redo_run_info.json"}}
    for cname in panels:
        i = info[cname]
        numbers[f"redo_{cname}_panel_size"] = {"value": len(i["panel"]), "source": f"{SCRIPT}:redo_selected_features_{cname}.csv", "note": f"screen: {i['screen_tier_used']}; candidates {i['n_candidates']}; LASSO kept {i['n_lasso_kept']}; after correlation {i['n_after_correlation']}"}
        numbers[f"redo_{cname}_panel_overlap_with_original"] = {"value": len(i["panel_overlap_with_original"]), "source": f"{SCRIPT}:redo_vs_original.csv", "note": ";".join(i["panel_overlap_with_original"])}
        numbers[f"redo_{cname}_panel_presence_absence_members"] = {"value": i["panel_presence_absence_members"], "source": f"{SCRIPT}:redo_selected_features_{cname}.csv",
                                                                   "note": f"panel members detected in >= 90 % of one training group and <= 10 % of the other; members with any group-specific missingness (difference >= 0.5): {i['panel_members_with_group_specific_missingness']}; {i['note']}"}
        numbers[f"redo_{cname}_n_screen_tier1"] = {"value": i["n_screen_tier1"], "source": f"{SCRIPT}:redo_run_info.json", "note": f"training-set transitions with BH-FDR < {args.fdr} and |log2FC| > {args.lfc}; tier 2 (p<0.05) {i['n_screen_tier2']}"}
        for m in cc.MODELS:
            for dset in ("valid", "train_oof_10fold"):
                r = met[(met["comparison"] == cname) & (met["model"] == m) & (met["dataset"] == dset)].iloc[0]
                numbers[f"redo_{cname}_{m}_{dset}_auc"] = {"value": r["auc"], "source": f"{SCRIPT}:redo_model_metrics.csv",
                                                           "note": f"n={r['n']} ({r['n_case']} vs {r['n_control']}); DeLong {r['auc_delong_lo']}-{r['auc_delong_hi']}; bootstrap {r['auc_boot_lo']}-{r['auc_boot_hi']}; sens {r['sensitivity']} spec {r['specificity']} at training Youden cut-off {r['cutoff']:.4f}; TP/FP/TN/FN {r['tp']}/{r['fp']}/{r['tn']}/{r['fn']}"}
    cc.update_numbers(args.out_dir / "numbers_s2_6.json", numbers, SCRIPT)

    # ---- preview figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    for cname, pva in preds_valid.items():
        fig, ax = plt.subplots(figsize=(4.6, 4.3))
        for m in cc.MODELS:
            fpr, tpr, _ = roc_curve(pva["y"], pva[f"prob_{m}"], drop_intermediate=False)
            ax.plot(fpr, tpr, lw=1.2, label=f"{m} AUC {cc.fast_auc(pva['y'], pva[f'prob_{m}']):.3f}")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_title(f"redo, held-out validation, {cname}\n(n={len(pva)}: {int(pva['y'].sum())} vs {int(len(pva) - pva['y'].sum())}; panel {len(panels[cname])})", fontsize=9)
        ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity"); ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(args.fig_dir / f"redo_roc_{cname}.png", dpi=150)
        plt.close(fig)
        ft = pd.read_csv(args.out_dir / f"redo_selected_features_{cname}.csv")
        top = ft[ft["candidate_for_lasso"]].sort_values("lasso_selection_frequency", ascending=False).head(40)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(len(top)), top["lasso_selection_frequency"], color=["#0072B2" if f else "#999999" for f in top["in_final_panel"]])
        ax.axhline(args.reproducibility, color="#D55E00", lw=1)
        ax.set_xticks(range(len(top))); ax.set_xticklabels(top["transition_id"], rotation=90, fontsize=6)
        ax.set_ylabel("LASSO selection frequency (50 seeds)")
        ax.set_title(f"{cname}: top candidates (blue = final panel)", fontsize=9)
        fig.tight_layout()
        fig.savefig(args.fig_dir / f"redo_lasso_{cname}.png", dpi=150)
        plt.close(fig)
    print(f"\ndone: {SCRIPT} in {info['total_runtime_s']} s")


if __name__ == "__main__":
    main()
