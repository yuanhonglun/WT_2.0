# -*- coding: utf-8 -*-
"""Fig. 7 - lung-cancer serum discovery pools and the targeted cohort assay built from them
(revision 2026-10, work package S3, cancer part).

Adapted from manuscript/fig 7/make_fig7.py and the Fig. 7 routine of
revision_260722/scripts/analyze_cancer.py.  Panels:

  a  Signal-type composition of the processed discovery features (counts, not means).
     Default: PLACEHOLDER drawn from the previously submitted Source Data sheet Fig7a_composition.
     With --discovery-dir <dir> (the output of analyze_cancer_discovery.py): drawn from
     <dir>/Fig7a_composition.csv, no placeholder stamp.
  b  PCA of the ten pooled discovery samples (80 % covariance ellipses, descriptive only).
     Default: PLACEHOLDER drawn from the previously submitted sheet Fig7b_PCA (scores) and the
     explained variance recorded in revision_260722/analysis/cancer_summary.json.
     With --discovery-dir: drawn from <dir>/Fig7b_PCA.csv (scores and explained variance).
     When <dir>/discovery_status.json says the negative-mode run is still pending, panels a
     and b carry a "positive mode only" stamp instead.
  c  Free bile acids (cholic acid, deoxycholic acid isomer) across HC / BLD / LC stage I, III, IV.
  d  Conjugated bile acids (glycocholic, taurocholic isomer, tauroursodeoxycholic acid).
     c and d show ALL technically valid measurements of Supplementary Table 9 ("version B":
     NaN = not detected is the only exclusion, no outlier removal).  The version that applies
     the published outlier flags ("version A") is drawn by make_supplementary_figures_cancer.py
     (Supplementary Fig. 4, file sfig4_outlier_removed_published_rule).
  e  ROC curves of the re-derived classifiers (analysis/cancer/redo_predictions_*.csv):
     e1 LC vs BLD, held-out internal validation (14 vs 6) - the main result;
     e2 LC vs HC and e3 LC vs non-LC, exploratory and batch-confounded (HC sera were
     collected and run as a separate batch).  In every ROC panel the five models are drawn
     thin, KNN thick (the model pre-specified in the published figure; it is NOT the
     best-AUC model on the validation set, which would be a selection on held-out data),
     and the published 20-metabolite KNN scores restricted to the same participants as a
     dashed grey reference (its candidate pool was derived on all 120 participants).

Statistics: Kruskal-Wallis across the five groups, then Dunn's pairwise z-tests with
Benjamini-Hochberg adjustment over the 10 pairs (identical to the published pipeline and to
cancer_common.dunn_pairwise).  The q-values drawn come from analysis/cancer/tukey_sensitivity.csv
(column q_bh_B_all_valid) and are re-derived from the plotted values as a self-check.  Brackets
are drawn for the pairs the published figure annotated (q < 0.05 under version A) plus every
pair whose significance changed between A and B (one: cholic acid HC vs stage I); the stars
are those of version B.

Palette: Okabe-Ito subset of scripts/paper_palette.py (HC blue, BLD orange, LC purple with
increasing depth for stage I / III / IV; MS1-detected blue, MS2-only purple).  Arial 8 pt
body, 10 pt bold panel letters, 1 pt lines, text kept editable in SVG (fonttype none / 42).

Inputs (all resolved relative to this file unless overridden on the command line):
  tables/Supplementary Table 9.xlsx                            per-participant MRM intensities
  tables/Source Data.xlsx                                      sheets Fig7a_composition, Fig7b_PCA
  ../lung_cancer/old_data/20250731.ml_results/20250731.ml_results/
      clinical_info_120Samples.xlsx, ml_results/1.group_split  sex, age, 85/35 split
  ../fig 7/data/violin_data.xlsx, roc_valid.txt                published flags + published scores
  ../revision_260722/analysis/cancer_summary.json              PCA explained variance (placeholder b)
  --discovery-dir: Fig7a_composition.csv, Fig7b_PCA.csv,       re-processed discovery pools
      discovery_status.json                                    (analyze_cancer_discovery.py)
  analysis/cancer/tukey_sensitivity.csv, exclusion_counts.csv  Dunn + BH under versions A / B / C
  analysis/cancer/redo_predictions_valid_<comparison>.csv      re-derived per-participant probabilities
  analysis/cancer/redo_model_metrics.csv, original_models_metrics.csv   AUC + DeLong 95 % CI

Outputs:
  figures/fig7/Fig7.{svg,pdf,png}, figures/fig7/legend_fig7.txt
  tables/source_data_cancer_draft.xlsx   (sheets Fig7*, one per panel, sex column on every
                                          per-participant sheet; the supplementary script
                                          adds its SFig* sheets to the same workbook)
  analysis/cancer/numbers_fig7.json      every number printed on the figure
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from scipy import stats  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cancer_common as cc  # noqa: E402

REV_DIR = SCRIPT_DIR.parent            # .../manuscript/revision_261019
MANUSCRIPT_DIR = REV_DIR.parent        # .../manuscript
SCRIPT = Path(__file__).name

# --------------------------------------------------------------------------------------
# palette (scripts/paper_palette.py when present, otherwise the same constants)
# --------------------------------------------------------------------------------------
try:
    import paper_palette as pp
except ImportError:  # pragma: no cover - fallback keeps the script self-contained
    class _Palette:
        BLUE, ORANGE, PURPLE, SKY, VERMILION = "#0072B2", "#E69F00", "#CC79A7", "#56B4E9", "#D55E00"
        INK, GREY, RULE, GREY_FILL, WHITE = "#231815", "#7F7F7F", "#CFD7DF", "#E6E6E6", "#FFFFFF"

        @staticmethod
        def _rgb(c):
            c = c.lstrip("#")
            return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

        def mix(self, colour, frac, base="#FFFFFF"):
            c, b = self._rgb(colour), self._rgb(base)
            return "#{:02X}{:02X}{:02X}".format(*(int(round((frac * ci + (1 - frac) * bi) * 255)) for ci, bi in zip(c, b)))

        def light(self, colour):
            return self.mix(colour, 0.25)

        def medium(self, colour):
            return self.mix(colour, 0.55)

        def dark(self, colour):
            return self.mix(colour, 0.75, "#000000")

        def apply_rcparams(self, extra=None):
            from matplotlib import font_manager
            try:
                font_manager.findfont("Arial", fallback_to_default=False)
                family = "Arial"
            except (ValueError, RuntimeError):
                family = "DejaVu Sans"
            params = {"font.family": family, "font.sans-serif": [family, "DejaVu Sans"], "font.size": 8,
                      "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                      "legend.fontsize": 7, "legend.frameon": False, "text.color": self.INK,
                      "axes.edgecolor": self.INK, "axes.labelcolor": self.INK, "xtick.color": self.INK,
                      "ytick.color": self.INK, "axes.linewidth": 1.0, "lines.linewidth": 1.0,
                      "patch.linewidth": 1.0, "xtick.major.width": 1.0, "ytick.major.width": 1.0,
                      "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 300}
            if extra:
                params.update(extra)
            matplotlib.rcParams.update(params)

    pp = _Palette()

HC_BLUE, BLD_ORANGE, LC_PURPLE = pp.BLUE, pp.ORANGE, pp.PURPLE
SKY, INK, GREY, RULE = pp.SKY, pp.INK, pp.GREY, pp.RULE


def light(colour: str) -> str:
    """Light fill: 0.25 * colour + 0.75 * white."""
    return pp.light(colour)


STAGE_ORDER = list(cc.STAGE_ORDER)                       # HC, BLD, Stage I LC, Stage III LC, Stage IV LC
STAGE_SHORT = [cc.STAGE_SHORT[s] for s in STAGE_ORDER]   # HC, BLD, I, III, IV
# LC stages: the LC purple with increasing depth (fill 25 / 50 / 75 % purple, edge purple -> darker)
STAGE_FILL = [light(HC_BLUE), light(BLD_ORANGE), pp.mix(LC_PURPLE, 0.25), pp.mix(LC_PURPLE, 0.50), pp.mix(LC_PURPLE, 0.75)]
STAGE_EDGE = [HC_BLUE, BLD_ORANGE, LC_PURPLE, pp.mix(LC_PURPLE, 0.75, "#000000"), pp.mix(LC_PURPLE, 0.50, "#000000")]
GROUP_COLOUR = {"HC": HC_BLUE, "BLD": BLD_ORANGE, "LC": LC_PURPLE}
SIGNAL_COLOUR = {"ms1_detected": HC_BLUE, "ms2_only": LC_PURPLE}

MODELS = list(cc.MODELS)                                 # KNN, LR, NB, RF, SVM
MAIN_MODEL = "KNN"                                       # pre-specified in the published figure
MODEL_LONG = {"KNN": "K-nearest neighbours", "LR": "logistic regression", "NB": "naive Bayes",
              "RF": "random forest", "SVM": "support vector machine"}

# Fig. 7c / 7d metabolites: transition id -> (name, authentic-standard confirmed)
BILE_C = [("MN13238", "Cholic acid", True), ("MN13487", "Deoxycholic acid (isomer)", False)]
BILE_D = [("P3P2784", "Glycocholic acid", True), ("MN10308", "Taurocholic acid (isomer)", False),
          ("P3N0393", "Tauroursodeoxycholic acid", True)]

# The first category carried a different label in the previously submitted sheet; that wording was retired by
# the term audit (features are marked, never removed), so the sheet value is renamed on load.
FIRST_CATEGORY = "Features retained after redundancy marking"
COMPOSITION_CATEGORIES = [FIRST_CATEGORY, "Library-annotated", "High-scoring annotations",
                          "Carried into cohort assay"]
COMPOSITION_LABELS = ["Features retained after\nredundancy marking", "Library-\nannotated", "High-scoring\nannotations",
                      "Carried into\ncohort assay"]
MM = 1 / 25.4
LW = 1.0


# --------------------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ml_default = MANUSCRIPT_DIR / "lung_cancer" / "old_data" / "20250731.ml_results" / "20250731.ml_results"
    p.add_argument("--table9", type=Path, default=REV_DIR / "tables" / "Supplementary Table 9.xlsx")
    p.add_argument("--published-source-data", type=Path, default=REV_DIR / "tables" / "Source Data.xlsx",
                   help="previously submitted Source Data workbook (sheets Fig7a_composition, Fig7b_PCA; placeholders)")
    p.add_argument("--pca-summary", type=Path, default=MANUSCRIPT_DIR / "revision_260722" / "analysis" / "cancer_summary.json",
                   help="JSON with pca_explained_percent of the placeholder PCA")
    p.add_argument("--ml-dir", type=Path, default=ml_default)
    p.add_argument("--fig7-data", type=Path, default=MANUSCRIPT_DIR / "fig 7" / "data")
    p.add_argument("--analysis-dir", type=Path, default=REV_DIR / "analysis" / "cancer")
    p.add_argument("--fig-dir", type=Path, default=REV_DIR / "figures" / "fig7")
    p.add_argument("--workbook", type=Path, default=REV_DIR / "tables" / "source_data_cancer_draft.xlsx")
    p.add_argument("--numbers", type=Path, default=REV_DIR / "analysis" / "cancer" / "numbers_fig7.json")
    p.add_argument("--yscale", choices=["log", "linear"], default="log",
                   help="intensity axis of the violin panels (log10 by default: all measurements span up to 3 decades)")
    p.add_argument("--no-placeholder-stamp", action="store_true",
                   help="do not print the '[[placeholder]]' tag inside panels a and b")
    p.add_argument("--discovery-dir", type=Path, default=None,
                   help="output folder of analyze_cancer_discovery.py (Fig7a_composition.csv, Fig7b_PCA.csv, "
                        "discovery_status.json): draw panels a and b from the re-processed discovery pools "
                        "instead of the submitted placeholders")
    p.add_argument("--seed", type=int, default=20261020, help="jitter seed (no effect on statistics)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
def load_discovery_panels(discovery_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[float], dict]:
    """Panel a and b tables written by analyze_cancer_discovery.py (re-processed discovery pools).

    Returns the composition table (category, signal_type, count), the PCA scores (sample, group,
    PC1, PC2), the explained variance [PC1 %, PC2 %] and the status record (modes present,
    negative_pending, software versions)."""
    comp = pd.read_csv(discovery_dir / "Fig7a_composition.csv")
    pca = pd.read_csv(discovery_dir / "Fig7b_PCA.csv")
    explained = [float(pca["PC1_explained_pct"].iloc[0]), float(pca["PC2_explained_pct"].iloc[0])]
    pca = pca[["sample", "group", "PC1", "PC2"]].copy()
    status_path = discovery_dir / "discovery_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    if status.get("include_tier_d") or status.get("legacy_comment_reader"):
        raise ValueError(f"{discovery_dir}: written in reproduction mode (include_tier_d / legacy reader); not for the figure")
    return comp, pca, explained, status


@dataclass
class Cohort:
    t9: cc.Table9
    clinical: pd.DataFrame        # index sample: sex, age, group3, stage ...
    split: pd.DataFrame           # index sample: set (train / valid), group3, stage
    workbook: pd.DataFrame        # collaborator's violin workbook, index sample (published values + flags)
    tukey: pd.DataFrame           # analysis/cancer/tukey_sensitivity.csv
    exclusion: pd.DataFrame       # analysis/cancer/exclusion_counts.csv

    @property
    def stage(self) -> pd.Series:
        return self.t9.stage

    @property
    def samples(self) -> list[str]:
        return list(self.t9.data.columns)


def load_cohort(args) -> Cohort:
    t9 = cc.load_table9(args.table9)
    assert t9.data.shape == (3541, 120), f"Table 9 shape {t9.data.shape}"
    clinical = cc.load_clinical(args.ml_dir)
    split = cc.load_split(args.ml_dir)
    wb = pd.read_excel(args.fig7_data / "violin_data.xlsx", sheet_name="全部数据离群值检查-IQR方法").set_index("Index")
    assert set(clinical.index) == set(t9.data.columns), "clinical table and Table 9 list different participants"
    assert set(wb.index) == set(t9.data.columns), "violin workbook and Table 9 list different participants"
    assert (wb["Stage"].reindex(t9.data.columns) == t9.stage).all(), "stage labels differ between workbook and Table 9"
    assert (split["stage"].reindex(t9.data.columns) == t9.stage).all(), "stage labels differ between split and Table 9"
    assert clinical["sex"].isin(["male", "female"]).all(), "sex missing for some participants"
    tukey = pd.read_csv(args.analysis_dir / "tukey_sensitivity.csv")
    excl = pd.read_csv(args.analysis_dir / "exclusion_counts.csv")
    return Cohort(t9=t9, clinical=clinical, split=split, workbook=wb, tukey=tukey, exclusion=excl)


def tukey_flag_per_group(values: pd.Series, by: pd.Series, k: float = 1.5) -> pd.Series:
    """True where a detected value lies outside [Q1 - k IQR, Q3 + k IQR] of its group (NaN never flagged).
    This is the Methods rule as written ("version C"); it is reported in Source Data only."""
    flag = pd.Series(False, index=values.index)
    for _, idx in by.groupby(by).groups.items():
        x = values.loc[idx].dropna()
        if len(x) < 4:
            continue
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        flag[x.index] = (x < q1 - k * iqr) | (x > q3 + k * iqr)
    return flag


def per_person_table(cohort: Cohort, mid: str, name: str, panel: str) -> pd.DataFrame:
    """One row per participant: Table 9 intensity, published value / flag, inclusion under A / B / C, sex, age."""
    samples = cohort.samples
    raw = cohort.t9.data.loc[mid, samples].astype(float)
    pub = cohort.workbook[mid].reindex(samples).astype(float)
    pub_flag = cohort.workbook[f"{mid}_IQR"].reindex(samples).astype(str).str.lower().eq("true")
    flag_c = tukey_flag_per_group(raw, cohort.stage.reindex(samples), 1.5)
    cl = cohort.clinical.reindex(samples)
    out = pd.DataFrame({
        "transition_id": mid, "metabolite": name, "panel": panel, "sample": samples,
        "group": cohort.t9.group3.reindex(samples).values, "stage": cohort.stage.reindex(samples).values,
        "sex": cl["sex"].values, "age": cl["age"].values,
        "intensity_cps": raw.values,                                   # NaN = not detected
        "detected": raw.notna().values,
        "included_B_all_valid": raw.notna().values,                    # plotted in Fig. 7c,d / SFig. 4 (main)
        "published_value_cps": pub.values,                             # collaborator workbook (imputed where not detected)
        "published_outlier_flag_A": pub_flag.values,
        "value_imputed_in_published_analysis": raw.isna().values,
        "included_A_published": (~pub_flag).values,                    # plotted in the outlier-removed SI version
        "tukey_flag_C_per_group_1.5IQR": flag_c.values,
        "included_C": (raw.notna() & ~flag_c).values,
    })
    return out


def group_values(cohort: Cohort, mid: str, version: str) -> dict[str, np.ndarray]:
    """Values per stage group. 'B' = all detected Table 9 values; 'A' = workbook values minus published flags
    (imputed values included, as published); 'C' = detected values minus the per-group 1.5 x IQR rule."""
    samples = cohort.samples
    raw = cohort.t9.data.loc[mid, samples].astype(float)
    stage = cohort.stage.reindex(samples)
    if version == "B":
        ser = raw.dropna()
    elif version == "A":
        pub = cohort.workbook[mid].reindex(samples).astype(float)
        flag = cohort.workbook[f"{mid}_IQR"].reindex(samples).astype(str).str.lower().eq("true")
        ser = pub[~flag]
    elif version == "C":
        flag = tukey_flag_per_group(raw, stage, 1.5)
        ser = raw[raw.notna() & ~flag]
    else:
        raise ValueError(version)
    return {s: ser[stage.index[stage == s].intersection(ser.index)].to_numpy() for s in STAGE_ORDER}


def imputed_mask(cohort: Cohort, mid: str) -> dict[str, np.ndarray]:
    """For version A: which plotted values are imputed (not detected in Table 9) - drawn as open symbols."""
    samples = cohort.samples
    raw = cohort.t9.data.loc[mid, samples].astype(float)
    pub = cohort.workbook[mid].reindex(samples).astype(float)
    flag = cohort.workbook[f"{mid}_IQR"].reindex(samples).astype(str).str.lower().eq("true")
    stage = cohort.stage.reindex(samples)
    ser = pub[~flag]
    return {s: raw.reindex(ser[stage.index[stage == s].intersection(ser.index)].index).isna().to_numpy() for s in STAGE_ORDER}


VERSION_COL = {"A": "A_published_flags", "B": "B_all_valid", "C": "C_valid_tukey1.5_per_group"}


def pairwise_rows(cohort: Cohort, mid: str) -> pd.DataFrame:
    rows = cohort.tukey[cohort.tukey["transition_id"] == mid].copy()
    assert len(rows) == 10, f"{mid}: expected 10 pairs in tukey_sensitivity.csv, got {len(rows)}"
    return rows


def bracket_plan(cohort: Cohort, mid: str, version: str) -> list[tuple[int, int, str, float]]:
    """(i, j, stars, q) for the brackets to draw.
    version B (main figure): pairs annotated in the published figure (q_A < 0.05) plus pairs whose
    significance changed between A and B; stars from q_B.
    version A (SI sensitivity): pairs with q_A < 0.05, stars from q_A (reproduces the published figure)."""
    rows = pairwise_rows(cohort, mid)
    qa, qb = rows["q_bh_A_published_flags"].to_numpy(), rows["q_bh_B_all_valid"].to_numpy()
    if version == "B":
        draw = (qa < 0.05) | (qb < 0.05)
        q_used = qb
    else:
        draw = qa < 0.05
        q_used = rows[f"q_bh_{VERSION_COL[version]}"].to_numpy()
    plan = []
    for (_, r), d, q in zip(rows.iterrows(), draw, q_used):
        if d:
            plan.append((STAGE_ORDER.index(r["group_a"]), STAGE_ORDER.index(r["group_b"]), cc.stars(q), float(q)))
    plan.sort(key=lambda h: (abs(h[1] - h[0]), h[0]))
    return plan


def self_check_dunn(cohort: Cohort, mid: str, version: str = "B") -> None:
    """Re-derive Dunn + BH from the plotted values and compare with tukey_sensitivity.csv."""
    g = group_values(cohort, mid, version)
    mine = cc.dunn_pairwise(g).set_index(["group_a", "group_b"])["q_bh"]
    rows = pairwise_rows(cohort, mid).set_index(["group_a", "group_b"])[f"q_bh_{VERSION_COL[version]}"]
    diff = (mine.reindex(rows.index) - rows).abs().max()
    assert diff < 1e-9, f"{mid} version {version}: Dunn+BH q-values differ from tukey_sensitivity.csv by {diff:.2e}"


# --------------------------------------------------------------------------------------
# drawing helpers
# --------------------------------------------------------------------------------------
def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=LW)


def panel_letter(ax, letter: str, dx: float = -34, dy: float = 10) -> None:
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points",
                fontsize=10, fontweight="bold", va="bottom", ha="left", color=INK, annotation_clip=False)


def _log_ticks(lo: float, hi: float) -> tuple[list[float], list[str]]:
    """Tick positions (log10 units) and labels for a log10 intensity axis."""
    k0, k1 = int(math.floor(lo)), int(math.ceil(hi))
    inside = []
    for mant in ((1,), (1, 2, 5), (1, 2, 3, 5), (1, 1.5, 2, 3, 5, 7)):   # coarsest set giving >= 2 ticks
        cands = [(k + math.log10(m), k, m) for k in range(k0, k1 + 1) for m in mant]
        inside = [c for c in cands if lo - 1e-9 <= c[0] <= hi + 1e-9]
        if len(inside) >= 2:
            break
    if len(inside) > 5:
        inside = inside[:: int(math.ceil(len(inside) / 5))]
    pos = [c[0] for c in inside]
    lab = [f"$10^{{{c[1]}}}$" if c[2] == 1 else f"${c[2]:g}\\times10^{{{c[1]}}}$" for c in inside]
    return pos, lab


def violin_panel(ax, cohort: Cohort, mid: str, title: str, version: str, rng: np.random.Generator,
                 yscale: str = "log", dagger: bool = False) -> dict:
    """Violin + box + all points for one metabolite; brackets per bracket_plan. Returns the printed numbers."""
    groups = group_values(cohort, mid, version)
    data = [groups[s] for s in STAGE_ORDER]
    ns = [int(v.size) for v in data]
    assert all(n >= 2 for n in ns), f"{mid}: a group has fewer than 2 values"
    open_mask = imputed_mask(cohort, mid) if version == "A" else {s: np.zeros(groups[s].size, bool) for s in STAGE_ORDER}
    band_line = None
    if yscale == "log":
        assert all((v > 0).all() for v in data), f"{mid}: non-positive intensity on a log axis"
        plotted = [np.log10(v) for v in data]
        exponent = None
        masks = [open_mask[s] for s in STAGE_ORDER]
        if any(m.any() for m in masks):
            # Version A kept the imputed constant (about 1e2 cps) for non-detected signals.  Drawing it at its
            # numeric position would stretch the axis by two decades and squash every distribution, so the
            # imputed values are shown as open symbols in a censored band at the base of the axis.  Every
            # imputed value is below every measured value, so ranks - and therefore the boxes and the Dunn
            # tests - are unchanged; only the violin outline differs.
            det = np.concatenate([p[~m] for p, m in zip(plotted, masks)])
            lo_det, hi_det = det.min(), det.max()
            span_det = max(hi_det - lo_det, 0.6)
            floor = lo_det - 0.16 * span_det
            assert all((p[m] < lo_det).all() for p, m in zip(plotted, masks)), f"{mid}: an imputed value exceeds a measured value"
            plotted = [np.where(m, floor, p) for p, m in zip(plotted, masks)]
            band_line = lo_det - 0.07 * span_det
    else:
        ymax = max(v.max() for v in data)
        exponent = int(np.floor(np.log10(ymax)))
        plotted = [v / 10.0 ** exponent for v in data]
    x = np.arange(len(STAGE_ORDER))

    parts = ax.violinplot(plotted, positions=x, widths=0.80, showextrema=False)
    for body, fill, edge in zip(parts["bodies"], STAGE_FILL, STAGE_EDGE):
        body.set_facecolor(fill)
        body.set_edgecolor(edge)
        body.set_linewidth(LW)
        body.set_alpha(0.9)
    for xi, values, edge, s in zip(x, plotted, STAGE_EDGE, STAGE_ORDER):
        jitter = xi + rng.uniform(-0.17, 0.17, len(values))
        om = open_mask[s]
        ax.scatter(jitter[~om], values[~om], s=2.2, color=edge, alpha=0.55, linewidth=0, zorder=2)
        if om.any():   # imputed values retained in the published analysis
            ax.scatter(jitter[om], values[om], s=7, facecolor="white", edgecolor=edge, linewidth=0.6, zorder=2)
    ax.boxplot(plotted, positions=x, widths=0.16, showfliers=False, patch_artist=True,
               medianprops={"color": INK, "linewidth": LW}, boxprops={"facecolor": "white", "edgecolor": INK, "linewidth": LW},
               whiskerprops={"color": INK, "linewidth": LW}, capprops={"color": INK, "linewidth": LW})

    plan = bracket_plan(cohort, mid, version)
    lo = min(v.min() for v in plotted)
    hi = max(v.max() for v in plotted)
    span = max(hi - lo, 0.6 if yscale == "log" else 0.3 * hi)
    # Each bracket level takes a fixed share of the axis height (so the star glyph never touches the
    # next bracket's tick), and the data occupy the rest: range = (data + margins) / (1 - share * k).
    bottom = (lo - 0.08 * span) if yscale == "log" else 0.0
    k_levels = max(len(plan), 1)
    share = 0.105
    rng_total = (hi + 0.05 * span - bottom + 0.04 * span) / (1.0 - share * k_levels)
    step = share * rng_total
    base = hi + 0.05 * span
    for k, (i, j, star, _) in enumerate(plan):
        yy = base + k * step
        ax.plot([i, i, j, j], [yy, yy + step * 0.2, yy + step * 0.2, yy], color=INK, linewidth=LW, clip_on=False, zorder=5)
        ax.text((i + j) / 2, yy + step * 0.22, star, ha="center", va="bottom", fontsize=8 if star != "ns" else 7,
                color=INK, clip_on=False)
    ax.set_ylim(bottom, bottom + rng_total)
    if band_line is not None:
        ax.axhline(band_line, color=GREY, linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
        ax.text(len(STAGE_ORDER) - 0.45, band_line, "imputed", fontsize=6, color=GREY, ha="right", va="bottom", style="italic")
    ax.set_xlim(-0.6, len(STAGE_ORDER) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n({n})" for s, n in zip(STAGE_SHORT, ns)])
    if yscale == "log":
        pos, lab = _log_ticks(band_line if band_line is not None else bottom, base - 0.02 * span)
        ax.set_yticks(pos)
        ax.set_yticklabels(lab)
        ax.set_ylabel("Intensity (cps)", labelpad=2)
    else:
        ax.set_ylabel(f"Intensity ($10^{{{exponent}}}$ cps)", labelpad=2)
    ax.set_title(f"{title}{' †' if dagger else ''}", fontsize=8, fontweight="bold", pad=4)
    style_axis(ax)
    return {"transition_id": mid, "version": version, "n_per_group": dict(zip(STAGE_ORDER, ns)),
            "brackets": [{"pair": f"{STAGE_ORDER[i]} vs {STAGE_ORDER[j]}", "stars": st, "q_bh": q} for i, j, st, q in plan],
            "kruskal_p": float(pairwise_rows(cohort, mid)[f"kruskal_p_{VERSION_COL[version]}"].iloc[0])}


def roc_points(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """ROC with tie handling: KNN emits discrete vote fractions, a per-sample staircase would invent thresholds."""
    y = np.asarray(y, float)
    score = np.asarray(score, float)
    order = np.argsort(-score, kind="mergesort")
    y, score = y[order], score[order]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    last = np.r_[np.flatnonzero(np.diff(score)), len(score) - 1]
    tpr = np.r_[0.0, tp[last] / tp[-1]]
    fpr = np.r_[0.0, fp[last] / fp[-1]]
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return fpr, tpr, float(integrate(tpr, fpr))


def fmt_ci(auc: float, lo: float, hi: float) -> str:
    return f"{auc:.3f} ({lo:.3f}–{hi:.3f})"


def roc_panel(ax, preds: pd.DataFrame, metrics: pd.DataFrame, published: pd.Series | None, published_auc: tuple | None,
              title: str, subtitle: str, legend_title: str, cyfra: tuple | None = None) -> dict:
    """One ROC panel: five re-derived models (KNN thick), published scores on the same participants dashed,
    optionally the clinical marker Cyfra21-1 alone (dotted grey; cyfra = (series, auc, lo, hi))."""
    y = preds["y"].to_numpy(int)
    printed = {"n_case": int(y.sum()), "n_control": int((1 - y).sum()), "models": {}}
    others = []
    for m in MODELS:
        fpr, tpr, auc_curve = roc_points(y, preds[f"prob_{m}"].to_numpy(float))
        row = metrics[metrics["model"] == m].iloc[0]
        assert abs(auc_curve - row["auc"]) < 5e-4, f"{title} {m}: curve AUC {auc_curve:.4f} != metrics {row['auc']:.4f}"
        printed["models"][m] = {"auc": float(row["auc"]), "delong_lo": float(row["auc_delong_lo"]), "delong_hi": float(row["auc_delong_hi"])}
        if m == MAIN_MODEL:
            ax.plot(fpr, tpr, color=HC_BLUE, linewidth=1.6, zorder=4,
                    label=f"{m}: {fmt_ci(row['auc'], row['auc_delong_lo'], row['auc_delong_hi'])}")
        else:
            ax.plot(fpr, tpr, color=SKY, linewidth=0.7, zorder=3, label="_nolegend_")
            others.append(float(row["auc"]))
    ax.plot([], [], color=SKY, linewidth=0.7, label=f"LR, NB, RF, SVM: {min(others):.3f}–{max(others):.3f}")
    printed["others_range"] = [min(others), max(others)]
    if published is not None:
        fpr, tpr, auc_pub = roc_points(y, published.to_numpy(float))
        assert abs(auc_pub - published_auc[0]) < 5e-4, f"{title}: published curve AUC {auc_pub:.4f} != {published_auc[0]:.4f}"
        ax.plot(fpr, tpr, color=GREY, linewidth=LW, linestyle=(0, (4, 2)), zorder=2,
                label=f"published panel, same {len(y)}: {auc_pub:.3f}")
        printed["published_same_participants"] = {"auc": float(published_auc[0]), "delong_lo": float(published_auc[1]), "delong_hi": float(published_auc[2])}
    if cyfra is not None:
        series, auc_c, lo_c, hi_c = cyfra
        fpr, tpr, auc_curve = roc_points(y, series.to_numpy(float))
        assert abs(auc_curve - auc_c) < 5e-4, f"{title}: Cyfra21-1 curve AUC {auc_curve:.4f} != {auc_c:.4f}"
        ax.plot(fpr, tpr, color=GREY, linewidth=1.1, linestyle=(0, (1, 1.4)), zorder=2,
                label=f"Cyfra21-1 alone: {fmt_ci(auc_c, lo_c, hi_c)}")
        printed["cyfra21_1"] = {"auc": float(auc_c), "delong_lo": float(lo_c), "delong_hi": float(hi_c)}
    ax.plot([0, 1], [0, 1], color=RULE, linewidth=0.8, zorder=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("1 – Specificity", labelpad=2)
    ax.set_ylabel("Sensitivity", labelpad=2)
    ax.set_title(title, fontsize=8, fontweight="bold", pad=11)
    ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color=INK)
    leg = ax.legend(loc="lower right", fontsize=7, title=legend_title, title_fontsize=7, handlelength=1.5,
                    handletextpad=0.5, borderaxespad=0.6, labelspacing=0.3, alignment="left")
    leg.get_title().set_color(INK)
    style_axis(ax)
    return printed


# --------------------------------------------------------------------------------------
# Source Data workbook
# --------------------------------------------------------------------------------------
def write_sheets(path: Path, sheets: dict[str, pd.DataFrame], readme: dict[str, str]) -> None:
    """Add / replace sheets in the workbook (created when absent); README first, header bold, panes frozen."""
    from openpyxl import load_workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    path.parent.mkdir(parents=True, exist_ok=True)
    existing_readme = {}
    if path.exists():
        try:
            old = pd.read_excel(path, sheet_name="README")
            existing_readme = dict(zip(old["sheet"], old["content"]))
        except Exception:
            existing_readme = {}
    existing_readme.update(readme)
    readme_df = pd.DataFrame({"sheet": list(existing_readme), "content": list(existing_readme.values())})
    mode = "a" if path.exists() else "w"
    kw = {"if_sheet_exists": "replace"} if mode == "a" else {}
    with pd.ExcelWriter(path, engine="openpyxl", mode=mode, **kw) as xw:
        readme_df.to_excel(xw, sheet_name="README", index=False)
        for name, df in sheets.items():
            assert len(name) <= 31, f"sheet name too long: {name}"
            df.to_excel(xw, sheet_name=name, index=False)
    wb = load_workbook(path)
    wb.move_sheet("README", offset=-wb.index(wb["README"]))
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for col in range(1, ws.max_column + 1):
            width = max((len(str(ws.cell(row=r, column=col).value or "")) for r in range(1, min(ws.max_row, 200) + 1)), default=8)
            ws.column_dimensions[get_column_letter(col)].width = min(max(10, width + 2), 60)
    wb.save(path)
    print(f"  workbook {path.name}: sheets {list(sheets)} written")


def _save(fig, fig_dir: Path, stem: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(fig_dir / f"{stem}.{ext}", dpi=600 if ext == "png" else None, facecolor="white")
        print("  wrote", fig_dir / f"{stem}.{ext}")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main(argv=None) -> None:
    args = parse_args(argv)
    pp.apply_rcparams()
    cohort = load_cohort(args)
    rng = np.random.default_rng(args.seed)
    numbers: dict[str, dict] = {}
    sheets: dict[str, pd.DataFrame] = {}
    readme: dict[str, str] = {}

    fig = plt.figure(figsize=(210 * MM, 215 * MM))
    gs = fig.add_gridspec(4, 12, left=0.062, right=0.985, top=0.965, bottom=0.048, hspace=0.72, wspace=1.15,
                          height_ratios=[1.05, 1.0, 1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 2:8])   # starts at column 2 so the two-line category labels stay inside the page
    ax_b = fig.add_subplot(gs[0, 8:12])
    axes_c = [fig.add_subplot(gs[1, 0:6]), fig.add_subplot(gs[1, 6:12])]
    axes_d = [fig.add_subplot(gs[2, i * 4:i * 4 + 4]) for i in range(3)]
    axes_e = [fig.add_subplot(gs[3, i * 4:i * 4 + 4]) for i in range(3)]

    # ---------------- a, b: discovery-pool data (re-processed run via --discovery-dir, else the submitted placeholders) ----------------
    if args.discovery_dir is not None:
        comp, pca, explained, discovery_status = load_discovery_panels(args.discovery_dir)
        negative_pending = bool(discovery_status.get("negative_pending", False))
        versions = ", ".join(f"{m} v{v}" for m, v in discovery_status.get("software_version", {}).items())
        ab_source = f"{args.discovery_dir.name} (analyze_cancer_discovery.py; ASFAMProcessor {versions})"
        ab_note = "re-processed discovery pools" + ("; POSITIVE MODE ONLY, negative-mode re-run pending" if negative_pending else "")
        ab_readme = ("re-processed discovery pools (analyze_cancer_discovery.py, ASFAMProcessor " + versions + ")"
                     + ("; POSITIVE MODE ONLY, to be regenerated when the negative-mode re-run is analysed" if negative_pending else ""))
        placeholder = False
    else:
        comp = pd.read_excel(args.published_source_data, sheet_name="Fig7a_composition")
        legacy = [c for c in comp["category"].unique() if c not in COMPOSITION_CATEGORIES]
        assert len(legacy) <= 1, f"unexpected categories in Fig7a_composition: {legacy}"
        if legacy:
            comp["category"] = comp["category"].replace({legacy[0]: FIRST_CATEGORY})
        pca = pd.read_excel(args.published_source_data, sheet_name="Fig7b_PCA")
        with open(args.pca_summary, encoding="utf-8") as fh:
            explained = json.load(fh)["pca_explained_percent"]
        negative_pending = False
        ab_source = "Fig7a_composition"
        ab_note = "PLACEHOLDER (previously submitted value)"
        ab_readme = "PLACEHOLDER - previously submitted values; to be regenerated after the cancer-discovery re-run"
        placeholder = True
    missing = [c for c in COMPOSITION_CATEGORIES if c not in set(comp["category"])]
    assert not missing, f"Fig7a_composition lacks categories {missing}"

    # ---------------- a: signal-type composition ----------------
    totals = {c: int(comp.loc[comp["category"] == c, "count"].sum()) for c in COMPOSITION_CATEGORIES}
    y = np.arange(len(COMPOSITION_CATEGORIES))
    offsets = {"ms1_detected": -0.17, "ms2_only": 0.17}
    labels = {"ms1_detected": "MS1-detected", "ms2_only": "MS2-only"}
    for signal in ("ms1_detected", "ms2_only"):
        values = [int(comp.loc[(comp["category"] == c) & (comp["signal_type"] == signal), "count"].iloc[0]) for c in COMPOSITION_CATEGORIES]
        colour = SIGNAL_COLOUR[signal]
        ax_a.barh(y + offsets[signal], values, height=0.27, facecolor=light(colour), edgecolor=colour, linewidth=LW, label=labels[signal])
        for yy, value, c in zip(y + offsets[signal], values, COMPOSITION_CATEGORIES):
            share = 100.0 * value / max(totals[c], 1)
            ax_a.text(value * 1.09, yy, f"{value:,} ({share:.0f}%)", va="center", ha="left", fontsize=7)
            numbers[f"fig7a_{c.replace(' ', '_').lower()}_{signal}"] = {"value": value, "source": f"{SCRIPT}:{ab_source}",
                                                                         "note": f"{ab_note}; share {share:.0f}% of {totals[c]:,}"}
    ax_a.set_xscale("log")
    ax_a.set_xlim(100, int(comp["count"].max()) * 4.6)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(COMPOSITION_LABELS)
    ax_a.set_ylim(len(COMPOSITION_CATEGORIES) - 0.45, -0.45)
    ax_a.set_xlabel("Feature count (log scale)", labelpad=2)
    ax_a.set_title("Signal-type composition of the processed serum features", fontsize=8, fontweight="bold", pad=4)
    ax_a.legend(loc="lower right", ncol=2, handlelength=1.1, handletextpad=0.35, borderaxespad=0.2, fontsize=7)
    style_axis(ax_a)
    panel_letter(ax_a, "a", dx=-52)

    # ---------------- b: pooled-sample PCA ----------------
    for group in ("HC", "BLD", "LC"):
        sub = pca[pca["group"] == group]
        colour = GROUP_COLOUR[group]
        ax_b.scatter(sub["PC1"], sub["PC2"], s=26, facecolor=light(colour), edgecolor=colour, linewidth=LW, label=f"{group} ({len(sub)})", zorder=3)
        if len(sub) >= 3:
            xy = sub[["PC1", "PC2"]].to_numpy()
            cov = np.cov(xy, rowvar=False)
            vals, vecs = np.linalg.eigh(cov)
            vals = np.maximum(vals, 1e-9)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            radius = math.sqrt(stats.chi2.ppf(0.80, 2))
            ax_b.add_patch(Ellipse(xy.mean(axis=0), *(2 * radius * np.sqrt(vals)), angle=angle, facecolor=light(colour),
                                   edgecolor=colour, linewidth=LW, alpha=0.35, zorder=1))
    ax_b.axhline(0, color=RULE, linewidth=LW, zorder=0)
    ax_b.axvline(0, color=RULE, linewidth=LW, zorder=0)
    ax_b.set_xlabel(f"PC1 ({explained[0]:.1f}%)", labelpad=2)
    ax_b.set_ylabel(f"PC2 ({explained[1]:.1f}%)", labelpad=2)
    ax_b.set_title("Pooled-sample PCA", fontsize=8, fontweight="bold", pad=4)
    ax_b.legend(loc="lower right", handletextpad=0.25, fontsize=7, labelspacing=0.3)
    style_axis(ax_b)
    panel_letter(ax_b, "b")
    b_source = ab_source if not placeholder else args.pca_summary.name
    b_note = ab_note if not placeholder else "PLACEHOLDER (previously submitted PCA)"
    numbers["fig7b_pc1_explained_percent"] = {"value": round(float(explained[0]), 1), "source": f"{SCRIPT}:{b_source}", "note": b_note}
    numbers["fig7b_pc2_explained_percent"] = {"value": round(float(explained[1]), 1), "source": f"{SCRIPT}:{b_source}", "note": b_note}
    stamp = None
    if placeholder and not args.no_placeholder_stamp:
        stamp = "[[placeholder: old discovery data]]"
    elif negative_pending and not args.no_placeholder_stamp:
        stamp = "[[positive mode only: negative-mode re-run pending]]"
    if stamp:
        for ax, pos, ha, va in ((ax_a, (0.985, 0.62), "right", "center"), (ax_b, (0.03, 0.955), "left", "top")):
            ax.text(*pos, stamp, transform=ax.transAxes, ha=ha, va=va, fontsize=6,
                    color=GREY, style="italic", zorder=6)
    sheets["Fig7a_composition"] = comp
    sheets["Fig7b_PCA"] = pca
    readme["Fig7a_composition"] = ab_readme + "; counts of MS1-detected / MS2-only features per category"
    readme["Fig7b_PCA"] = ab_readme + "; PCA scores of the ten discovery pools"

    # ---------------- c, d: bile acids, version B (all technically valid measurements) ----------------
    violin_numbers = []
    person_rows_c, person_rows_d = [], []
    for ax, (mid, name, dagger) in zip(axes_c, BILE_C):
        self_check_dunn(cohort, mid, "B")
        violin_numbers.append(violin_panel(ax, cohort, mid, name, "B", rng, args.yscale, dagger))
        person_rows_c.append(per_person_table(cohort, mid, name, "Fig. 7c"))
    panel_letter(axes_c[0], "c")
    for ax, (mid, name, dagger) in zip(axes_d, BILE_D):
        self_check_dunn(cohort, mid, "B")
        violin_numbers.append(violin_panel(ax, cohort, mid, name, "B", rng, args.yscale, dagger))
        person_rows_d.append(per_person_table(cohort, mid, name, "Fig. 7d"))
    panel_letter(axes_d[0], "d")
    for v in violin_numbers:
        mid = v["transition_id"]
        for s, n in v["n_per_group"].items():
            numbers[f"fig7cd_{mid}_n_{cc.STAGE_SHORT[s]}"] = {"value": n, "source": f"{SCRIPT}:Table 9 version B", "note": f"{mid} group {s}: measured values plotted (not detected excluded)"}
        numbers[f"fig7cd_{mid}_kruskal_p_B"] = {"value": v["kruskal_p"], "source": f"{SCRIPT}:tukey_sensitivity.csv", "note": "Kruskal-Wallis across the five groups, version B"}
        for b in v["brackets"]:
            key = b["pair"].replace(" ", "_").replace("Stage_", "").replace("_LC", "")
            numbers[f"fig7cd_{mid}_{key}_q_B"] = {"value": b["q_bh"], "source": f"{SCRIPT}:tukey_sensitivity.csv", "note": f"Dunn + BH, version B; drawn as '{b['stars']}'"}
    bile_ids = [m for m, _, _ in BILE_C + BILE_D]
    tk = cohort.tukey[cohort.tukey["transition_id"].isin(bile_ids)].copy()
    plan_lookup = {(m, STAGE_ORDER[i], STAGE_ORDER[j]): st for m, _, _ in BILE_C + BILE_D for i, j, st, _ in bracket_plan(cohort, m, "B")}
    tk["bracket_drawn_fig7"] = [((m, a, b) in plan_lookup) for m, a, b in zip(tk["transition_id"], tk["group_a"], tk["group_b"])]
    tk["stars_drawn_fig7"] = [plan_lookup.get((m, a, b), "") for m, a, b in zip(tk["transition_id"], tk["group_a"], tk["group_b"])]
    sheets["Fig7c_free_bile_acids"] = pd.concat(person_rows_c, ignore_index=True)
    sheets["Fig7d_conjugated_bile_acids"] = pd.concat(person_rows_d, ignore_index=True)
    sheets["Fig7cd_pairwise_tests"] = tk
    sheets["Fig7cd_group_summary"] = cohort.exclusion[cohort.exclusion["transition_id"].isin(bile_ids)].copy()
    readme["Fig7c_free_bile_acids"] = "per participant: Table 9 intensity (blank = not detected), sex, age, inclusion under version A (published flags) / B (all measured, plotted) / C (1.5 x IQR per group)"
    readme["Fig7d_conjugated_bile_acids"] = "as Fig7c_free_bile_acids for the three conjugated bile acids"
    readme["Fig7cd_pairwise_tests"] = "Dunn + Benjamini-Hochberg q-values for the 10 group pairs under versions A / B / C, Kruskal-Wallis p, and which brackets / stars are drawn"
    readme["Fig7cd_group_summary"] = "per metabolite x group: n, not detected, flagged, analysed n, mean, median under versions A / B / C"

    # ---------------- e: ROC, re-derived classifiers ----------------
    redo = pd.read_csv(args.analysis_dir / "redo_model_metrics.csv")
    orig = pd.read_csv(args.analysis_dir / "original_models_metrics.csv")
    pub_scores = cc.load_original_scores(args.fig7_data)["valid"]
    roc_specs = [
        ("LC_vs_BLD", "LC vs BLD, held-out validation", "BLD"),
        ("LC_vs_HC", "LC vs HC, exploratory", "HC"),
        ("LC_vs_nonLC", "LC vs non-LC, exploratory", "non-LC"),
    ]
    roc_numbers = {}
    for ax, (comp_name, title, ctrl) in zip(axes_e, roc_specs):
        preds = pd.read_csv(args.analysis_dir / f"redo_predictions_valid_{comp_name}.csv")
        assert set(preds["sample"]) <= set(pub_scores.index), f"{comp_name}: validation participants not in roc_valid.txt"
        assert (cohort.split.reindex(preds["sample"])["set"] == "valid").all(), f"{comp_name}: a participant is not in the held-out set"
        met = redo[(redo["comparison"] == comp_name) & (redo["dataset"] == "valid")]
        panel_size = int(met["panel_size"].iloc[0])
        o = orig[(orig["model"] == MAIN_MODEL) & (orig["dataset"] == "valid") & (orig["comparison"] == comp_name) & (orig["stage"] == "all LC")].iloc[0]
        pub_same = pub_scores.reindex(preds["sample"])[MAIN_MODEL]
        assert (pub_scores.reindex(preds["sample"])["y"].to_numpy(int) == preds["y"].to_numpy(int)).all(), f"{comp_name}: labels differ between redo and published files"
        n_case, n_ctrl = int(preds["y"].sum()), int((1 - preds["y"]).sum())
        subtitle = f"panel of {panel_size} transition{'s' if panel_size != 1 else ''}; trained on {int(redo[(redo['comparison'] == comp_name) & (redo['dataset'] == 'train_insample')]['n'].iloc[0])}"
        legend_title = f"LC n = {n_case} vs {ctrl} n = {n_ctrl}" + ("" if comp_name == "LC_vs_BLD" else "\nbatch-confounded")
        cyfra = None
        if comp_name == "LC_vs_BLD":
            # clinical Cyfra21-1 alone on the same held-out participants (recorded for patients only)
            cy = cohort.clinical.reindex(preds["sample"])["Cyfra21_1"].astype(float)
            assert cy.notna().all(), "Cyfra21-1 missing for a held-out LC/BLD participant"
            cref = pd.read_csv(args.analysis_dir / "cyfra_auc.csv")
            cref = cref[(cref["marker"] == "Cyfra21_1") & (cref["comparison"] == comp_name) & (cref["dataset"] == "valid")].iloc[0]
            auc_c, lo_c, hi_c, _ = cc.delong_auc_ci(preds["y"], cy)
            assert abs(auc_c - cref["auc"]) < 5e-4 and abs(lo_c - cref["auc_delong_lo"]) < 5e-4, "Cyfra21-1 AUC differs from cyfra_auc.csv"
            cyfra = (cy, auc_c, lo_c, hi_c)
        printed = roc_panel(ax, preds, met, pub_same, (o["auc"], o["auc_delong_lo"], o["auc_delong_hi"]), title, subtitle, legend_title, cyfra)
        printed["panel_size"] = panel_size
        if cyfra is not None:
            _, _, _, p_redo = cc.delong_paired_test(preds["y"], preds[f"prob_{MAIN_MODEL}"], cyfra[0])
            _, _, _, p_pub = cc.delong_paired_test(preds["y"], pub_same, cyfra[0])
            pref = pd.read_csv(args.analysis_dir / "panel_vs_cyfra_paired_delong.csv")
            pref = pref[(pref["model"] == MAIN_MODEL) & (pref["comparison"] == comp_name)].iloc[0]
            assert abs(p_pub - pref["delong_p_two_sided"]) < 1e-3, "paired DeLong p differs from panel_vs_cyfra_paired_delong.csv"
            printed["cyfra21_1"].update({"paired_delong_p_vs_rederived_KNN": float(p_redo), "paired_delong_p_vs_published_KNN": float(p_pub)})
            numbers[f"fig7e_{comp_name}_valid_cyfra21_1_auc"] = {"value": round(float(auc_c), 4), "source": f"{SCRIPT}:cyfra_auc.csv", "note": f"Cyfra21-1 alone, same 20 participants; DeLong 95% CI {lo_c:.4f}-{hi_c:.4f}"}
            numbers[f"fig7e_{comp_name}_paired_delong_p_rederived_KNN_vs_cyfra21_1"] = {"value": round(float(p_redo), 4), "source": SCRIPT, "note": "paired DeLong, two-sided, same 20 participants"}
            numbers[f"fig7e_{comp_name}_paired_delong_p_published_KNN_vs_cyfra21_1"] = {"value": round(float(p_pub), 4), "source": f"{SCRIPT}:panel_vs_cyfra_paired_delong.csv", "note": "paired DeLong, two-sided, same 20 participants"}
        roc_numbers[comp_name] = printed
        sheet = preds.copy()
        cl = cohort.clinical.reindex(sheet["sample"])
        sheet.insert(3, "sex", cl["sex"].values)
        sheet.insert(4, "age", cl["age"].values)
        sheet.insert(1, "set", "held-out validation")
        sheet["published_panel_prob_KNN"] = pub_same.values
        if cyfra is not None:
            sheet["Cyfra21_1_clinical"] = cyfra[0].values
        sheet = sheet.rename(columns={"y": "true_label_LC"})
        sname = f"Fig7e{'123'[roc_specs.index((comp_name, title, ctrl))]}_{comp_name}"
        sheets[sname] = sheet
        readme[sname] = (f"{title}: per held-out participant true label, sex, age and predicted probability of the five re-derived models (prob_<model>) plus the published 20-metabolite KNN score on the same participant"
                         + ("; Cyfra21_1_clinical = Cyfra21-1 value as recorded in the clinical table (dotted curve)" if cyfra is not None else ""))
        for m, d in printed["models"].items():
            numbers[f"fig7e_{comp_name}_valid_{m}_auc"] = {"value": d["auc"], "source": f"{SCRIPT}:redo_model_metrics.csv", "note": f"DeLong 95% CI {d['delong_lo']:.4f}-{d['delong_hi']:.4f}; printed {'thick' if m == MAIN_MODEL else 'thin'}"}
        numbers[f"fig7e_{comp_name}_valid_published_same_participants_auc"] = {"value": printed["published_same_participants"]["auc"], "source": f"{SCRIPT}:original_models_metrics.csv",
                                                                                  "note": f"published 20-metabolite KNN scores restricted to the same {n_case + n_ctrl} participants; DeLong CI {o['auc_delong_lo']:.4f}-{o['auc_delong_hi']:.4f}"}
        numbers[f"fig7e_{comp_name}_n_case"] = {"value": n_case, "source": SCRIPT, "note": "held-out validation LC"}
        numbers[f"fig7e_{comp_name}_n_control"] = {"value": n_ctrl, "source": SCRIPT, "note": f"held-out validation {ctrl}"}
        numbers[f"fig7e_{comp_name}_panel_size"] = {"value": panel_size, "source": f"{SCRIPT}:redo_model_metrics.csv", "note": "transitions in the re-derived panel"}
    panel_letter(axes_e[0], "e")
    summary_rows = []
    for comp_name, d in roc_numbers.items():
        for m, v in d["models"].items():
            summary_rows.append({"comparison": comp_name, "model": m, "panel_size": d["panel_size"], "dataset": "held-out validation",
                                 "n_case": d["n_case"], "n_control": d["n_control"], "auc": v["auc"], "auc_delong_lo": v["delong_lo"],
                                 "auc_delong_hi": v["delong_hi"], "line": "thick" if m == MAIN_MODEL else "thin"})
        p = d["published_same_participants"]
        summary_rows.append({"comparison": comp_name, "model": f"published 20-metabolite {MAIN_MODEL}", "panel_size": 20, "dataset": "held-out validation (same participants)",
                             "n_case": d["n_case"], "n_control": d["n_control"], "auc": p["auc"], "auc_delong_lo": p["delong_lo"], "auc_delong_hi": p["delong_hi"], "line": "dashed grey"})
        if "cyfra21_1" in d:
            c = d["cyfra21_1"]
            summary_rows.append({"comparison": comp_name, "model": "Cyfra21-1 alone", "panel_size": 1, "dataset": "held-out validation (same participants)",
                                 "n_case": d["n_case"], "n_control": d["n_control"], "auc": c["auc"], "auc_delong_lo": c["delong_lo"], "auc_delong_hi": c["delong_hi"], "line": "dotted grey"})
    sheets["Fig7e_AUC_summary"] = pd.DataFrame(summary_rows)
    readme["Fig7e_AUC_summary"] = "AUC and DeLong 95% CI printed in the three ROC panels"

    _save(fig, args.fig_dir, "Fig7")

    # ---------------- Source Data, numbers, legend ----------------
    readme["README"] = "Draft Source Data for the lung-cancer figures (Fig. 7, Supplementary Figs. 4-5), revision 2026-10. Every per-participant sheet carries sex and age."
    write_sheets(args.workbook, sheets, readme)
    args.numbers.parent.mkdir(parents=True, exist_ok=True)
    with open(args.numbers, "w", encoding="utf-8") as fh:
        json.dump({"_meta": {"created": "2026-10-20", "purpose": "numbers printed on Fig. 7 and Supplementary Figs. 4-5 (S3 cancer)",
                             "convention": "key -> {value, source: <script>:<table>, note}"}}, fh, indent=2, ensure_ascii=False)
    cc.update_numbers(args.numbers, numbers, SCRIPT)
    print(f"  numbers: {len(numbers)} entries -> {args.numbers}")
    if placeholder:
        ab_banner = "[[Fig 7a,b: to be regenerated after the cancer-discovery re-run]]\n\n"
    elif negative_pending:
        ab_banner = "[[Fig 7a,b: positive mode only; to be regenerated when the negative-mode re-run is analysed]]\n\n"
    else:
        ab_banner = ""
    write_legend(args.fig_dir / "legend_fig7.txt", violin_numbers, roc_numbers, args.yscale, ab_banner)


def write_legend(path: Path, violin_numbers: list[dict], roc_numbers: dict, yscale: str, ab_banner: str = "") -> None:
    n_by = {v["transition_id"]: v["n_per_group"] for v in violin_numbers}
    hc = n_by["MN13238"]
    e1 = roc_numbers["LC_vs_BLD"]
    e2 = roc_numbers["LC_vs_HC"]
    e3 = roc_numbers["LC_vs_nonLC"]
    k1 = e1["models"][MAIN_MODEL]
    k2 = e2["models"][MAIN_MODEL]
    k3 = e3["models"][MAIN_MODEL]
    p1, p2, p3 = e1["published_same_participants"], e2["published_same_participants"], e3["published_same_participants"]
    cy = e1["cyfra21_1"]
    axis_note = ("The intensity axis is logarithmic because the unfiltered measurements span up to three orders of magnitude; the pairwise tests are rank-based and unaffected by the scale."
                 if yscale == "log" else "")
    text = f"""Fig. 7. WT 2.0 processing of lung-cancer serum discovery pools and the targeted cohort assay built from them.

{ab_banner}a: Signal-type composition of the processed discovery features (counts, not means): all features retained after redundancy marking, those carrying a library annotation, those meeting the high-scoring criteria (Total score ≥ 1.4 and at least three matched peaks), and those taken forward into the scheduled-MRM transition list quantified in the cohort. Percentages are the MS1-detected and MS2-only shares within each category.
b: PCA of the ten pooled discovery samples processed with ASFAMProcessor (HC, n = 3 pools; BLD, n = 3 pools; LC, n = 4 pools); ellipses show the 80% covariance region and are descriptive only.
c, d: Participant-level distributions of five bile-acid signals in the targeted cohort (HC, n = {hc['HC']}; BLD, n = {hc['BLD']}; stage I LC, n = {hc['Stage I LC']}; stage III LC, n = {hc['Stage III LC']}; stage IV LC, n = {hc['Stage IV LC']} participants; the number of measured values plotted is given under each group label and is smaller than the group size only where a signal was not detected). All technically valid measurements are shown; no outlier removal was applied. The version that applies the published outlier flags is given in the Supplementary Information ([[Supplementary Fig. 4b or a new Supplementary Fig.; numbering to be decided]]) and the per-participant inclusion status under both rules in the Source Data. Points show individual participants; violins show the distribution and embedded boxes show the median and interquartile range (whiskers to 1.5 x IQR). {axis_note} Groups were compared by a Kruskal–Wallis test followed by pairwise Dunn tests adjusted across all ten group pairs by the Benjamini–Hochberg method; brackets are drawn for the pairs annotated in the previously submitted figure and for the one pair whose adjusted significance changed when all measurements were included (cholic acid, HC vs stage I LC; q = 0.054 with the published flags, q = 0.048 with all measurements). *q < 0.05, **q < 0.01, ***q < 0.001. A dagger indicates authentic-standard confirmation; deoxycholic and taurocholic acid are isomer-level assignments.
e: ROC curves of the classifiers re-derived entirely inside the 85-participant training set (feature screening, LASSO selection, tuning and cut-off) and scored once on the held-out internal validation participants. Left: LC vs BLD, the pre-specified primary comparison (held-out LC n = {e1['n_case']}, BLD n = {e1['n_control']}; panel of {e1['panel_size']} transitions); {MODEL_LONG[MAIN_MODEL]} (KNN, thick line, the model used in the previously submitted figure) AUC = {k1['auc']:.3f} (95% DeLong CI {k1['delong_lo']:.3f}–{k1['delong_hi']:.3f}); the four other models are drawn as thin lines (AUC {e1['others_range'][0]:.3f}–{e1['others_range'][1]:.3f}). The dashed grey curve is the previously submitted 20-metabolite KNN panel, whose candidate pool had been derived on all 120 participants, scored on the same held-out participants (AUC = {p1['auc']:.3f}, 95% CI {p1['delong_lo']:.3f}–{p1['delong_hi']:.3f}). The dotted grey curve is the clinical marker Cyfra21-1 alone on the same 20 participants (AUC = {cy['auc']:.3f}, 95% CI {cy['delong_lo']:.3f}–{cy['delong_hi']:.3f}); paired DeLong tests do not separate it from the re-derived KNN panel (p = {cy['paired_delong_p_vs_rederived_KNN']:.2f}) or from the previously submitted panel (p = {cy['paired_delong_p_vs_published_KNN']:.2f}). Middle and right: the same procedure for LC vs HC (held-out {e2['n_case']} vs {e2['n_control']}; panel of {e2['panel_size']} transition; KNN AUC = {k2['auc']:.3f}, 95% CI {k2['delong_lo']:.3f}–{k2['delong_hi']:.3f}) and LC vs non-LC (held-out {e3['n_case']} vs {e3['n_control']}; panel of {e3['panel_size']} transitions; KNN AUC = {k3['auc']:.3f}, 95% CI {k3['delong_lo']:.3f}–{k3['delong_hi']:.3f}; previously submitted panel AUC = {p3['auc']:.3f}). These two comparisons are exploratory and batch-confounded: the HC sera were collected and acquired as a separate batch, so any HC-involving separation cannot be attributed to disease. Both sets were drawn from the same 120-participant cohort, so all curves describe internal training and held-out internal validation, not external validation. The classifier is shown to demonstrate that ASFAM-derived transitions can be carried into a targeted assay and modelled; it is not diagnostic evidence. Per-model AUCs with confidence intervals for training (10-fold out-of-fold) and validation are given in Supplementary Fig. 5.
AUC, area under the curve; BLD, benign lung disease; HC, healthy control; KNN, K-nearest neighbours; LC, lung cancer; LR, logistic regression; NB, naive Bayes; PCA, principal component analysis; RF, random forest; ROC, receiver operating characteristic; SVM, support vector machine.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print("  wrote", path)


if __name__ == "__main__":
    main()
