"""Shared helpers for the lung-cancer cohort re-analysis (revision 2026-10, work package S2.6).

Imported by cancer_metrics.py, cancer_classifier.py and cancer_cohort_tables.py.
Nothing here runs an analysis on its own; it only loads the published inputs and
provides the statistics used by all three scripts (AUC with DeLong and bootstrap
confidence intervals, cut-off rules, confusion matrices, Dunn tests).

Inputs it knows how to read (MetaboLights MTBLS15138 / Supplementary Tables):
  Supplementary Table 9.xlsx   targeted serum profiling matrix, 3,541 scheduled-MRM
                               transitions x 120 participants (+2 blanks) with the
                               published group-wise statistics in the trailing columns
  Supplementary Table 8.xlsx   sheet "Cohort_assay_features": the 2,172 cohort transitions
                               traced back to the ASFAM discovery features (signal_type,
                               discovery annotation and score)
  Supplementary Table 10.xlsx  the published 20-metabolite panel (ID, Annotation)
  20250731.ml_results/         collaborator's R/caret run: 85/35 split, LASSO reproducibility,
                               per-model predictions and cut-offs, clinical table
  fig 7/data/roc_train.txt, roc_valid.txt, merge_metrics.details.txt
                               per-participant predicted probabilities of the five published
                               models and the published metrics (Fig. 7e / Supplementary Fig. 5)

Default paths are resolved relative to this file (scripts/ -> revision folder -> manuscript
folder) so the scripts can be published together as paper_scripts/ without edits.
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import itertools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REV_DIR = SCRIPT_DIR.parent          # .../manuscript/revision_261019
MANUSCRIPT_DIR = REV_DIR.parent      # .../manuscript

DEFAULTS = {
    "table9": REV_DIR / "tables" / "Supplementary Table 9.xlsx",
    "table8": REV_DIR / "tables" / "Supplementary Table 8.xlsx",
    "table10": REV_DIR / "tables" / "Supplementary Table 10.xlsx",
    "ml_dir": MANUSCRIPT_DIR / "lung_cancer" / "old_data" / "20250731.ml_results" / "20250731.ml_results",
    "fig7_data": MANUSCRIPT_DIR / "fig 7" / "data",
    "out_dir": REV_DIR / "analysis" / "cancer",
    "fig_dir": REV_DIR / "figures" / "cancer_preview",
}

# Isotope-labelled internal standards carried in the transition list (HANDOFF section 2).
# P3P2788 (319 -> 257) is consistent with decanoyl-L-carnitine-d3 (loss of trimethylamine-d3),
# is annotated "Decanoyl-L-carnitine D3" in Table 9 and "Unknown (319-257)" in Table 10, and is
# a member of the published 20-metabolite panel. It is not one of the 16 standards in Table 12.
IS_TRANSITIONS = {
    "P3P0911": "L-Leucine-5,5,5-d3 (IS_LEUCINE)",
    "P3P2641": "L-Tryptophan-d5 (IS_TRYPTOPHAN)",
    "P3P2788": "decanoyl-L-carnitine-d3 (319->257; published as 'Unknown (319-257)')",
}
MZ_SOURCE_PLACEHOLDER = "[[pending: cancer re-run]]"

GROUP3_ORDER = ["HC", "BLD", "LC"]
STAGE_ORDER = ["HC", "BLD", "Stage I LC", "Stage III LC", "Stage IV LC"]
STAGE_SHORT = {"HC": "HC", "BLD": "BLD", "Stage I LC": "I", "Stage III LC": "III", "Stage IV LC": "IV"}
MODELS = ["KNN", "LR", "NB", "RF", "SVM"]

# Comparisons used throughout: case groups, control groups (group3 labels)
COMPARISONS = {
    "LC_vs_BLD": (["LC"], ["BLD"]),
    "LC_vs_HC": (["LC"], ["HC"]),
    "LC_vs_nonLC": (["LC"], ["HC", "BLD"]),
}


# --------------------------------------------------------------------------------------
# group labels
# --------------------------------------------------------------------------------------
def group3_of(label: str) -> str:
    """HC / BLD / LC from any of the group spellings used in Table 9, the split files or the clinic."""
    s = str(label).strip()
    if s in ("HC", "CK", "NC"):
        return "HC"
    if s in ("BLD", "GX", "BGD"):
        return "BLD"
    if s.upper().startswith(("I", "V")) or s in ("LC", "Case"):
        return "LC"
    raise ValueError(f"unknown group label {label!r}")


def stage_of(label: str) -> str:
    """Five-level stage label (HC, BLD, Stage I/III/IV LC) from a Table 9 / clinical label."""
    s = str(label).strip()
    g = group3_of(s)
    if g != "LC":
        return g
    u = s.upper()
    if u.startswith("IV"):
        return "Stage IV LC"
    if u.startswith("III"):
        return "Stage III LC"
    if u.startswith("I"):
        return "Stage I LC"
    raise ValueError(f"unknown stage label {label!r}")


def stage_roman(label: str) -> str:
    """I / III / IV / HC / BLD."""
    return STAGE_SHORT[stage_of(label)]


# --------------------------------------------------------------------------------------
# Supplementary Table 9
# --------------------------------------------------------------------------------------
@dataclass
class Table9:
    data: pd.DataFrame            # transitions x 120 participants, NaN = not detected
    blank: pd.DataFrame           # transitions x blanks
    meta: pd.DataFrame            # per-transition annotation columns from Table 9
    sample_label: pd.Series       # participant -> raw group label in Table 9 (HC, BLD, I, IIIB, IV, ...)
    stats: dict = field(default_factory=dict)   # comparison -> DataFrame of the published statistics

    @property
    def group3(self) -> pd.Series:
        return self.sample_label.map(group3_of)

    @property
    def stage(self) -> pd.Series:
        return self.sample_label.map(stage_of)

    def samples_in(self, groups: list[str]) -> list[str]:
        g = self.group3
        return list(g.index[g.isin(groups)])


def load_table9(path: Path | str) -> Table9:
    t9 = pd.read_excel(path, header=None)
    hdr1 = t9.iloc[1, :].astype(str).values
    hdr2 = t9.iloc[2, :].astype(str).values
    ids = t9.iloc[3:, 0].astype(str).str.strip().values
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate transition ids in Table 9")
    body = t9.iloc[3:, :]
    sample_cols = [j for j in range(11, t9.shape[1]) if hdr2[j].startswith("Ca")]
    blank_cols = [j for j in range(11, t9.shape[1]) if hdr2[j].startswith("Blank")]
    data = body.iloc[:, sample_cols].apply(pd.to_numeric, errors="coerce")
    data.index = ids
    data.columns = [hdr2[j] for j in sample_cols]
    blank = body.iloc[:, blank_cols].apply(pd.to_numeric, errors="coerce")
    blank.index = ids
    blank.columns = [hdr2[j] for j in blank_cols]
    sample_label = pd.Series([hdr1[j] for j in sample_cols], index=data.columns, name="group_label")

    msdial_name = body.iloc[:, 2].astype(str).str.strip().replace({"nan": ""}).values
    wt_name = body.iloc[:, 6].astype(str).str.strip().replace({"nan": ""}).values
    id_col1 = body.iloc[:, 1].astype(str).str.strip().values
    meta = pd.DataFrame({
        "transition_id": ids,
        "table9_id_column2": id_col1,
        "msdial_name": msdial_name,
        "msdial_weighted_dot": pd.to_numeric(body.iloc[:, 3], errors="coerce").values,
        "msdial_reverse_dot": pd.to_numeric(body.iloc[:, 4], errors="coerce").values,
        "msdial_total_score": pd.to_numeric(body.iloc[:, 5], errors="coerce").values,
        "wt2_name": wt_name,
        "wt2_reverse_score": pd.to_numeric(body.iloc[:, 7], errors="coerce").values,
        "wt2_positive_score": pd.to_numeric(body.iloc[:, 8], errors="coerce").values,
        "wt2_average_score": pd.to_numeric(body.iloc[:, 9], errors="coerce").values,
    }, index=ids)
    meta["polarity"] = ["negative" if re.match(r"^(MN|P3N)", i) else "positive" for i in ids]
    meta["is_internal_standard"] = [(i in IS_TRANSITIONS) or n.startswith("IS_") for i, n in zip(ids, wt_name)]

    def best_name(row):
        if row["wt2_name"]:
            return row["wt2_name"]
        if row["msdial_name"] and row["msdial_name"] != "Unknown":
            return row["msdial_name"]
        return ""

    def level(row):
        if row["wt2_name"]:
            return "WT 2.0 library match (average score %.2f)" % row["wt2_average_score"]
        if row["msdial_name"] and row["msdial_name"] != "Unknown":
            if row["msdial_name"].lower().startswith("low score"):
                return "MS-DIAL low-score match (total score %.2f)" % row["msdial_total_score"]
            return "MS-DIAL library match (total score %.2f)" % row["msdial_total_score"]
        return "no library annotation in Table 9"

    meta["name_table9"] = meta.apply(best_name, axis=1)
    meta["annotation_level_table9"] = meta.apply(level, axis=1)
    meta["annotated_table9"] = meta["name_table9"].ne("")

    # published statistics blocks (computed by the collaborator on all 120 participants)
    blocks: dict[str, dict[str, int]] = {}
    cur = None
    for j in range(max(sample_cols + blank_cols) + 1, t9.shape[1]):
        if hdr1[j] != "nan":
            cur = hdr1[j]
            blocks[cur] = {}
        if cur is not None:
            blocks[cur][hdr2[j]] = j
    stats_blocks = {}
    for comp, cols in blocks.items():
        df = pd.DataFrame({name: pd.to_numeric(body.iloc[:, j], errors="coerce").values for name, j in cols.items()}, index=ids)
        stats_blocks[comp] = df
    return Table9(data=data, blank=blank, meta=meta, sample_label=sample_label, stats=stats_blocks)


# --------------------------------------------------------------------------------------
# collaborator's ML result tree
# --------------------------------------------------------------------------------------
def load_split(ml_dir: Path | str) -> pd.DataFrame:
    """85/35 split. Columns: set (train/valid), group_r (CK/GX/I/III/IV), reference (Case/Control), group3, stage."""
    ml_dir = Path(ml_dir)
    frames = []
    for name in ("train", "valid"):
        df = pd.read_csv(ml_dir / "ml_results" / "1.group_split" / f"LC.{name}.group.txt", sep="\t")
        df = df.rename(columns={"Index": "sample", "Group": "group_r", "Reference": "reference"})
        df["set"] = name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).set_index("sample")
    out["group3"] = out["group_r"].map(group3_of)
    out["stage"] = out["group_r"].map(stage_of)
    return out


def load_clinical(ml_dir: Path | str) -> pd.DataFrame:
    """Clinical table of the 120 participants (two sheets: 50 HC, 70 patients).

    Columns: group_clinical, group3, stage, disease, sex (male/female), age, smoking_raw,
    smoking_ever (1 ever / 0 never / NaN not recorded), tnm, CEA, CA125, Cyfra21_1, NSE.
    """
    path = Path(ml_dir) / "clinical_info_120Samples.xlsx"
    xl = pd.ExcelFile(path)
    frames = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        frames.append(df)
    cl = pd.concat(frames, ignore_index=True)
    ren = {"性别": "sex_raw", "年龄": "age", "抽烟史": "smoking_raw", "病理TNM分期": "tnm",
           "Cyfra21-1": "Cyfra21_1", "Group": "group_clinical", "Disease": "disease"}
    cl = cl.rename(columns=ren)
    cl["sex"] = cl["sex_raw"].map({"男": "male", "女": "female"})
    cl["group3"] = cl["group_clinical"].map(group3_of)
    cl["stage"] = cl["group_clinical"].map(stage_of)

    def smoke(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip()
        if s in ("无", "否", "不吸烟", "无吸烟史"):
            return 0.0
        return 1.0

    cl["smoking_ever"] = cl["smoking_raw"].map(smoke) if "smoking_raw" in cl else np.nan
    for c in ("CEA", "CA125", "Cyfra21_1", "NSE"):
        if c in cl:
            cl[c] = pd.to_numeric(cl[c], errors="coerce")
        else:
            cl[c] = np.nan
    cl = cl.set_index("SampleID")
    keep = ["group_clinical", "group3", "stage", "disease", "sex", "age", "smoking_raw", "smoking_ever", "tnm",
            "CEA", "CA125", "Cyfra21_1", "NSE", "TubeID", "SampleType"]
    return cl[[c for c in keep if c in cl.columns]]


def load_original_scores(fig7_data: Path | str) -> dict[str, pd.DataFrame]:
    """Per-participant predicted probabilities of the five published models (Fig. 7e, SFig. 5)."""
    fig7_data = Path(fig7_data)
    out = {}
    for name, fn in (("train", "roc_train.txt"), ("valid", "roc_valid.txt")):
        df = pd.read_csv(fig7_data / fn, sep="\t").rename(columns={"Index": "sample", "Real": "y"}).set_index("sample")
        out[name] = df
    return out


def load_published_metrics(fig7_data: Path | str) -> pd.DataFrame:
    return pd.read_csv(Path(fig7_data) / "merge_metrics.details.txt", sep="\t")


def load_original_cutoffs(ml_dir: Path | str) -> dict[str, float]:
    """Cut-off actually applied by the collaborator (column Cutoff of the predict_result files)."""
    ml_dir = Path(ml_dir)
    out = {}
    for m in MODELS:
        p = pd.read_csv(ml_dir / "ml_results" / "3.model_result" / m / f"LC.panel_1.{m}.train.predict_result.txt", sep="\t")
        vals = p["Cutoff"].unique()
        if len(vals) != 1:
            raise ValueError(f"{m}: more than one cut-off recorded")
        out[m] = float(vals[0])
    return out


def load_original_panel(ml_dir: Path | str) -> pd.DataFrame:
    """The 20 published panel members in multi-model rank order, with the collaborator's annotation."""
    ml_dir = Path(ml_dir) / "ml_results" / "2.feature_select"
    rank = pd.read_csv(ml_dir / "3.LC.filter_multiML.result.txt", sep="\t").rename(columns={"Index": "transition_id"})
    anno = pd.read_csv(ml_dir / "4.panel_anno.txt", sep="\t").rename(columns={"Index": "transition_id"})
    rank["multiML_rank"] = np.arange(1, len(rank) + 1)
    out = anno.merge(rank, on="transition_id", how="left").sort_values("multiML_rank").reset_index(drop=True)
    return out


def load_original_feature_selection(ml_dir: Path | str) -> dict[str, pd.DataFrame]:
    d = Path(ml_dir) / "ml_results" / "2.feature_select"
    return {
        "reproducibility": pd.read_csv(d / "1.LC.all_lasso.reproducibility.txt", sep="\t"),
        "var_imp": pd.read_csv(d / "1.LC.all_lasso.var_imp.txt", sep="\t"),
        "corr": pd.read_csv(d / "2.LC.all_cor.result.txt", sep="\t", index_col=0),
        "corr_check": pd.read_csv(d / "2.LC.filter_corr.check.txt", sep="\t"),
        "multiML": pd.read_csv(d / "3.LC.filter_multiML.result.txt", sep="\t"),
    }


def load_original_var_imp(ml_dir: Path | str) -> pd.DataFrame:
    """Per-model variable importance of the published models (caret varImp, 0-100)."""
    d = Path(ml_dir) / "ml_results" / "3.model_result"
    frames = []
    for m in MODELS:
        df = pd.read_csv(d / m / f"LC.panel_1.{m}.train.var_imp.txt", sep="\t").rename(columns={"Index": "transition_id", "VarImp": f"varimp_{m}"})
        frames.append(df.set_index("transition_id"))
    return pd.concat(frames, axis=1)


def load_table8_map(path: Path | str) -> pd.DataFrame:
    """Sheet Cohort_assay_features of Table 8, indexed by transition id."""
    t8 = pd.read_excel(path, sheet_name="Cohort_assay_features")
    t8 = t8.set_index("transition_id")
    if not t8.index.is_unique:
        raise ValueError("transition ids in Table 8 are not unique")
    return t8


def load_table10(path: Path | str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, header=None)
    df = df.iloc[2:, :2]
    df.columns = ["transition_id", "table10_annotation"]
    return df.dropna(subset=["transition_id"]).reset_index(drop=True)


def annotate_transitions(ids: list[str], t9: Table9, t8: pd.DataFrame, table10: pd.DataFrame | None = None) -> pd.DataFrame:
    """Standard annotation block for a list of transitions (panel members etc.).

    Columns: transition_id, polarity, name_table9, annotation_level_table9, table10_annotation,
    in_table8_cohort_assay, discovery_feature_id, signal_type, discovery_annotation,
    discovery_total_score, discovery_high_score, transition_type, cohort_annotation_table8,
    is_internal_standard, mz_source (placeholder).
    """
    rows = []
    t10 = table10.set_index("transition_id")["table10_annotation"] if table10 is not None else None
    for tid in ids:
        m = t9.meta.loc[tid] if tid in t9.meta.index else None
        r = {
            "transition_id": tid,
            "polarity": m["polarity"] if m is not None else "",
            "name_table9": m["name_table9"] if m is not None else "",
            "annotation_level_table9": m["annotation_level_table9"] if m is not None else "not in Table 9",
            "table10_annotation": (t10.get(tid, "") if t10 is not None else ""),
            "in_table8_cohort_assay": tid in t8.index,
        }
        if tid in t8.index:
            x = t8.loc[tid]
            r.update({
                "discovery_feature_id": x["feature_id"],
                "signal_type": x["signal_type"],
                "discovery_annotation": x["discovery_annotation"] if pd.notna(x["discovery_annotation"]) else "",
                "discovery_total_score": x["discovery_total_score"],
                "discovery_high_score": bool(x["discovery_is_annotated"]),
                "transition_type": x["transition_type"],
                "cohort_annotation_table8": x["cohort_annotation"] if pd.notna(x["cohort_annotation"]) else "",
            })
        else:
            r.update({
                "discovery_feature_id": "",
                "signal_type": "not traced to an ASFAM discovery feature (transition absent from Table 8 Cohort_assay_features)",
                "discovery_annotation": "", "discovery_total_score": np.nan, "discovery_high_score": False,
                "transition_type": "", "cohort_annotation_table8": "",
            })
        r["is_internal_standard"] = bool(m["is_internal_standard"]) if m is not None else tid in IS_TRANSITIONS
        r["internal_standard_note"] = IS_TRANSITIONS.get(tid, "")
        # Precursor provenance of the discovery feature: carried from Supplementary Table 8
        # (sheet Cohort_assay_features of the re-processed run) when that table has the columns;
        # the submitted Table 8 has none, hence the placeholder.
        prov_cols = [c for c in ("mz_source", "mz_confidence", "nl_n_support", "ms2_only_tier") if c in t8.columns]
        if prov_cols:
            for col in prov_cols:
                v = t8.loc[tid][col] if tid in t8.index else ""
                r[col] = "" if (v is None or (isinstance(v, float) and np.isnan(v))) else v
        else:
            r["mz_source"] = MZ_SOURCE_PLACEHOLDER
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# ROC statistics
# --------------------------------------------------------------------------------------
def _as_arrays(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    if y.shape != s.shape:
        raise ValueError("y and s differ in length")
    mask = ~np.isnan(s)
    return y[mask], s[mask]


def fast_auc(y, s) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    y, s = _as_arrays(y, s)
    m = int(y.sum())
    n = len(y) - m
    if m == 0 or n == 0:
        return np.nan
    r = stats.rankdata(s, method="average")
    return float((r[y == 1].sum() - m * (m + 1) / 2) / (m * n))


def _placements(y, s):
    """DeLong structural components V10 (cases) and V01 (controls)."""
    y, s = _as_arrays(y, s)
    pos = s[y == 1]
    neg = s[y == 0]
    m, n = len(pos), len(neg)
    allv = np.concatenate([pos, neg])
    tx = stats.rankdata(allv, method="average")
    tpos = stats.rankdata(pos, method="average")
    tneg = stats.rankdata(neg, method="average")
    v10 = (tx[:m] - tpos) / n
    v01 = 1.0 - (tx[m:] - tneg) / m
    return v10, v01


def delong_auc_ci(y, s, alpha: float = 0.05) -> tuple[float, float, float, float]:
    """AUC, lower, upper (clipped to [0,1]) and SE by DeLong et al. (1988); identical to pROC::ci.auc(method='delong')."""
    y, s = _as_arrays(y, s)
    m = int(y.sum())
    n = len(y) - m
    if m < 1 or n < 1:
        return np.nan, np.nan, np.nan, np.nan
    v10, v01 = _placements(y, s)
    auc = float(v10.mean())
    s10 = float(v10.var(ddof=1)) if m > 1 else 0.0
    s01 = float(v01.var(ddof=1)) if n > 1 else 0.0
    se = float(np.sqrt(s10 / m + s01 / n))
    z = stats.norm.ppf(1 - alpha / 2)
    return auc, float(max(0.0, auc - z * se)), float(min(1.0, auc + z * se)), se


def delong_paired_test(y, s1, s2) -> tuple[float, float, float, float]:
    """Paired DeLong test for AUC(s1) - AUC(s2) on the same participants. Returns auc1, auc2, z, two-sided p."""
    y = np.asarray(y).astype(int)
    s1 = np.asarray(s1, float)
    s2 = np.asarray(s2, float)
    mask = ~(np.isnan(s1) | np.isnan(s2))
    y, s1, s2 = y[mask], s1[mask], s2[mask]
    m = int(y.sum())
    n = len(y) - m
    a10, a01 = _placements(y, s1)
    b10, b01 = _placements(y, s2)
    auc1, auc2 = float(a10.mean()), float(b10.mean())
    s10 = np.cov(np.vstack([a10, b10]), ddof=1) if m > 1 else np.zeros((2, 2))
    s01 = np.cov(np.vstack([a01, b01]), ddof=1) if n > 1 else np.zeros((2, 2))
    S = s10 / m + s01 / n
    L = np.array([1.0, -1.0])
    var = float(L @ S @ L)
    if var <= 0:
        return auc1, auc2, np.nan, np.nan
    z = (auc1 - auc2) / np.sqrt(var)
    return auc1, auc2, float(z), float(2 * stats.norm.sf(abs(z)))


def bootstrap_auc_ci(y, s, n_boot: int = 2000, seed: int = 20261020, alpha: float = 0.05) -> tuple[float, float]:
    """Stratified (within-class) percentile bootstrap CI of the AUC."""
    y, s = _as_arrays(y, s)
    pos = s[y == 1]
    neg = s[y == 0]
    m, n = len(pos), len(neg)
    if m < 1 or n < 1:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        p = pos[rng.integers(0, m, m)]
        q = neg[rng.integers(0, n, n)]
        allv = np.concatenate([p, q])
        r = stats.rankdata(allv, method="average")
        aucs[b] = (r[:m].sum() - m * (m + 1) / 2) / (m * n)
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def roc_thresholds(s) -> np.ndarray:
    """pROC-style candidate thresholds: midpoints between consecutive unique scores, plus -inf / +inf."""
    u = np.unique(np.asarray(s, float))
    mids = (u[:-1] + u[1:]) / 2 if len(u) > 1 else np.array([])
    return np.concatenate([[-np.inf], mids, [np.inf]])


def sens_spec_at(y, s, cutoff: float, rule: str = "ge") -> tuple[float, float]:
    y, s = _as_arrays(y, s)
    pred = (s >= cutoff) if rule == "ge" else (s > cutoff)
    sens = float(pred[y == 1].mean()) if (y == 1).any() else np.nan
    spec = float((~pred[y == 0]).mean()) if (y == 0).any() else np.nan
    return sens, spec


def youden_cutoff(y, s) -> float:
    """Threshold maximising sensitivity + specificity - 1 on (y, s); ties -> higher specificity, then lower threshold."""
    y, s = _as_arrays(y, s)
    best = None
    for t in roc_thresholds(s):
        se, sp = sens_spec_at(y, s, t)
        key = (se + sp - 1, sp, -t if np.isfinite(t) else -np.inf)
        if best is None or key > best[0]:
            best = (key, t)
    return float(best[1])


def closest_topleft_cutoff(y, s) -> float:
    """pROC best.method='closest.topleft': minimise (1-sens)^2 + (1-spec)^2."""
    y, s = _as_arrays(y, s)
    best = None
    for t in roc_thresholds(s):
        se, sp = sens_spec_at(y, s, t)
        d = (1 - se) ** 2 + (1 - sp) ** 2
        key = (-d, sp, -t if np.isfinite(t) else -np.inf)
        if best is None or key > best[0]:
            best = (key, t)
    return float(best[1])


def confusion_at(y, s, cutoff: float, rule: str = "ge") -> dict:
    y, s = _as_arrays(y, s)
    pred = (s >= cutoff) if rule == "ge" else (s > cutoff)
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    tn = int((~pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    acc = (tp + tn) / len(y) if len(y) else np.nan
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "sensitivity": sens, "specificity": spec,
            "ppv": ppv, "npv": npv, "accuracy": acc}


def metrics_block(y, s, cutoff: float | None, n_boot: int = 2000, seed: int = 20261020) -> dict:
    """n, prevalence, AUC with DeLong and bootstrap CIs, and confusion-matrix metrics at a cut-off."""
    y, s = _as_arrays(y, s)
    m = int(y.sum())
    n = len(y) - m
    out = {"n": int(len(y)), "n_case": m, "n_control": n, "prevalence": (m / len(y) if len(y) else np.nan)}
    if m >= 1 and n >= 1:
        auc, lo, hi, se = delong_auc_ci(y, s)
        blo, bhi = bootstrap_auc_ci(y, s, n_boot=n_boot, seed=seed)
    else:
        auc = lo = hi = se = blo = bhi = np.nan
    out.update({"auc": auc, "auc_delong_lo": lo, "auc_delong_hi": hi, "auc_se_delong": se,
                "auc_boot_lo": blo, "auc_boot_hi": bhi})
    if cutoff is not None and np.isfinite(cutoff):
        out["cutoff"] = cutoff
        out.update(confusion_at(y, s, cutoff))
    out["small_n_flag"] = bool(min(m, n) < 5)
    return out


# --------------------------------------------------------------------------------------
# nonparametric group tests used for the violin panels
# --------------------------------------------------------------------------------------
def bh_adjust(p) -> np.ndarray:
    p = np.asarray(p, float)
    n = p.size
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def dunn_pairwise(groups: dict[str, np.ndarray]) -> pd.DataFrame:
    """Dunn's post-hoc z-tests with tie correction (ggstatsplot nonparametric pairwise mode), BH across all pairs."""
    keys = [k for k in groups if len(groups[k]) > 0]
    pooled = np.concatenate([np.asarray(groups[k], float) for k in keys])
    n = pooled.size
    ranks_all = stats.rankdata(pooled)
    ranks, at = {}, 0
    for k in keys:
        ranks[k] = ranks_all[at:at + len(groups[k])]
        at += len(groups[k])
    _, counts = np.unique(pooled, return_counts=True)
    ties = float(((counts ** 3) - counts).sum())
    sigma2 = n * (n + 1) / 12 - ties / (12 * (n - 1))
    rows = []
    for a, b in itertools.combinations(keys, 2):
        na, nb = len(groups[a]), len(groups[b])
        z = abs(ranks[a].mean() - ranks[b].mean()) / np.sqrt(sigma2 * (1 / na + 1 / nb))
        rows.append({"group_a": a, "group_b": b, "z": z, "p": 2 * stats.norm.sf(z)})
    df = pd.DataFrame(rows)
    df["q_bh"] = bh_adjust(df["p"].values) if len(df) else []
    return df


def stars(q: float) -> str:
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "ns"


# --------------------------------------------------------------------------------------
# bookkeeping
# --------------------------------------------------------------------------------------
def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


def update_numbers(path: Path | str, entries: dict, script: str) -> None:
    """Merge entries (key -> {value, source, note}) into the S2.6 numbers ledger."""
    path = Path(path)
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            ledger = json.load(fh)
    else:
        ledger = {"_meta": {"created": "2026-10-20", "purpose": "numbers produced by the S2.6 lung-cancer scripts; merged into analysis/numbers.json by the owner",
                            "convention": "key -> {value, source: <script>:<table>, note}"}}
    for k, v in entries.items():
        v = dict(v)
        v.setdefault("source", script)
        ledger[k] = _jsonable(v)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(ledger, fh, indent=2, ensure_ascii=False)


def write_csv(df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  wrote {path.name}: {len(df)} rows x {df.shape[1]} cols")


def fmt(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "NA"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)
