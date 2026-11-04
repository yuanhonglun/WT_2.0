"""Rice salt-stress statistics: Supplementary Table 5, Fig. 6 and the Fig. 6 Source Data.

Third-round revision (2026-10) of the ASFAMProcessor rice analysis.  Compared with the
submitted version of this script the following changed:

* Main differential criterion = BH-FDR <= 0.05 and |log2FC| >= 1 (column ``differential``).
  The submitted criterion (p <= 0.05 and |log2FC| >= 1) is kept as ``exploratory_p0.05``.
* Fragment-poor MS2-only features (``ms2_only_tier == "D"``, fewer than three fragments)
  stay in the table but are *not evaluated*: they are excluded from the test family and
  from every headline count, and reported once separately.
* The precursor-source columns written by ASFAMProcessor >= v1.4 (``mz_source``,
  ``mz_confidence``, ``inferred_mz``, ``nl_n_support``, ``ms1_window_mz``,
  ``window_delta_mda``, ``ms2_only_tier``) are carried into Supplementary Table 5 when
  present.  On an older export the tier is derived as a proxy (D when ``signal_type`` is
  ``ms2_only`` and ``n_fragments`` < 3, otherwise unknown) and a note is printed.
* Authentic standards: 3-formylindole is confirmed salt-responsive; feruloyl putrescine
  and beta-carboline are reported as exploratory (q > 0.05).  Statuses are derived from
  the data, not hard-coded, so a re-run updates them.
* Fig. 6 uses an Okabe-Ito subset (CK blue, salt orange; MS1-detected blue, MS2-only
  purple) with light fills and full-colour edges; panel c/d report q (BH) instead of p.

The script also recomputes the statistics over the *submitted* test family (fragment-poor
rows included) as a reproduction check against the submitted Supplementary Table 5.

Usage (defaults are relative to this file; see ``--help``)::

    python analyze_rice.py --input <features.csv> --run-label old
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import math
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from openpyxl.styles import Alignment, Font, PatternFill
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REVISION_DIR = SCRIPT_DIR.parent
MANUSCRIPT_DIR = REVISION_DIR.parent
# The re-processed rice run (``run_dataset.py rice``); the submitted export lives in
# a locally named folder and is passed explicitly with ``--input`` when needed.
DEFAULT_INPUT = REVISION_DIR / "runs" / "rice" / "features.csv"

SAMPLE_NAMES = ["CK1", "CK2", "CK3", "Salt1", "Salt2", "Salt3"]
GROUPS = np.array(["CK", "CK", "CK", "Salt", "Salt", "Salt"])
HEIGHT_COLS = [f"height_rep{i}" for i in range(1, 7)]
STATUS_COLS = [f"gap_fill_status_rep{i}" for i in range(1, 7)]
SIGNAL_TYPES = ["ms1_detected", "ms2_only"]
N_INPUT_FILES = 186  # 6 samples x 31 segments of the rice ASFAM acquisition

# Precursor-source columns exported by ASFAMProcessor >= v1.4; carried into the table
# when present (reviewer request: precursor source and supporting fragments per feature).
PRECURSOR_COLUMNS = [
    "mz_source", "mz_confidence", "inferred_mz", "nl_n_support",
    "ms1_window_mz", "window_delta_mda", "ms2_only_tier",
]
TIER_VALUES = {"A", "B", "C", "D", ""}

HIGH_SCORE_MIN_SCORE = 1.4
HIGH_SCORE_MIN_MATCHED = 3
MIN_DETECTED = 3
FDR_ALPHA = 0.05
P_ALPHA = 0.05
LOG2FC_MIN = 1.0
PCA_TOP_FEATURES = 2000
EXCEL_CELL_LIMIT = 32000

# Authentic standards (MSI level 1 identity; Supplementary Fig. 2a-c).  The salt-response
# status of each is derived from the data (see ``salt_response_status``).  Only
# 3-formylindole may be described as "confirmed"; the other two are "exploratory".
STANDARD_FEATURES = {
    "3-formylindole": "F04771",
    "β-carboline": "F05817",
    "Feruloyl putrescine": "F06224",
}
CONFIRMED_STANDARD = "3-formylindole"

# Four high-scoring MS2-only library matches shown in Fig. 6e.  They are tentative
# candidates (MSI level 2); the selection is carried over from the first revision and no
# literature-novelty claim is attached to them.
SELECTED_MS2_ONLY_CANDIDATES = {
    "Phenacylamine": "F01816",
    "Quinoxaline": "F03762",
    "N-acetyl-tyrosine": "F04317",
    "Putative methoxylated flavanone glycoside": "F10828",
}
CANDIDATE_EVIDENCE_LEVEL = "High-scoring library match; tentative candidate (MSI level 2)"


def mix_with_white(hex_colour: str, white_fraction: float = 0.75) -> str:
    """Light fill: the colour mixed with white (default 75 % white)."""
    rgb = np.asarray(mcolors.to_rgb(hex_colour), dtype=float)
    mixed = (1.0 - white_fraction) * rgb + white_fraction * np.ones(3)
    return mcolors.to_hex(mixed)


# Okabe-Ito subset, no green-red pairing.  Fill = 25 % colour + 75 % white, edge = colour.
BLUE = "#0072B2"      # CK; MS1-detected
ORANGE = "#E69F00"    # Salt
PURPLE = "#CC79A7"    # MS2-only
GROUP_COLOURS = {"CK": BLUE, "Salt": ORANGE}
SIGNAL_COLOURS = {"ms1_detected": BLUE, "ms2_only": PURPLE}
INK = "#231815"
GRAY_EDGE = "#8C8C8C"
GRAY_FILL = mix_with_white(GRAY_EDGE, 0.6)
LW = 1.0


def resolve_font_family() -> str:
    """Arial when matplotlib can find it, otherwise DejaVu Sans."""
    names = {font.name for font in font_manager.fontManager.ttflist}
    return "Arial" if "Arial" in names else "DejaVu Sans"


FONT_FAMILY = resolve_font_family()


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def bh_adjust(values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values; NaN input stays NaN."""
    p = np.asarray(values, float)
    out = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    pv = p[valid]
    if not pv.size:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * pv.size / np.arange(1, pv.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty_like(ranked)
    restored[order] = np.minimum(ranked, 1.0)
    out[valid] = restored
    return out


def normalize_name(value: object) -> str:
    text = str(value) if pd.notna(value) else ""
    text = re.sub(r"^Putative:\s*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_export_version(path: Path) -> str:
    """Software version from the ``# version=`` comment line of features.csv."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        for _ in range(10):
            line = handle.readline()
            if not line.startswith("#"):
                break
            match = re.match(r"#\s*version\s*=\s*(\S+)", line)
            if match:
                return match.group(1)
    return "unknown"


def prepare_data(
    input_path: Path, tier_column: str, min_fragments: int, include_tier_d: bool = False
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    # Not comment="#": a library name containing "#" (e.g. ascr#17) would be
    # truncated and its row lose every later column (see features_io).
    from features_io import read_features
    frame = read_features(input_path)
    n_rows_read = len(frame)
    frame = frame[frame["feature_id"].astype(str).str.fullmatch(r"F\d+")].copy()
    not_duplicate = ~truthy(frame["is_duplicate"])
    classified = frame["signal_type"].isin(SIGNAL_TYPES)
    # Retained after redundancy marking = is_duplicate False; the (rare) retained rows
    # without a signal type are counted but not tabulated, as in the submitted table.
    frame["is_retained"] = not_duplicate & classified
    n_unclassified = int((not_duplicate & ~classified).sum())
    frame["n_fragments"] = pd.to_numeric(frame["n_fragments"], errors="coerce")
    frame["n_matched"] = pd.to_numeric(frame["n_matched"], errors="coerce")
    frame["total_score"] = pd.to_numeric(frame["total_score"], errors="coerce")
    frame["is_annotated"] = truthy(frame["annotated"])

    tier_is_proxy = tier_column not in frame.columns
    if tier_is_proxy:
        frame["ms2_only_tier"] = np.where(
            frame["signal_type"].eq("ms2_only") & frame["n_fragments"].lt(min_fragments), "D", ""
        )
        print(
            f"NOTE: column '{tier_column}' is absent from {input_path.name} (export version "
            f"{read_export_version(input_path)}). Using the PROXY tier: D = signal_type "
            f"'ms2_only' and n_fragments < {min_fragments}; every other MS2-only row is left "
            "blank (unknown), because tiers A-C need mz_source / mz_confidence, which "
            "ASFAMProcessor exports from v1.4 onwards."
        )
    else:
        tier = frame[tier_column].fillna("").astype(str).str.strip().str.upper()
        unknown = sorted(set(tier.unique()) - TIER_VALUES)
        if unknown:
            raise ValueError(f"unexpected values in {tier_column}: {unknown}")
        frame["ms2_only_tier"] = tier
        print(f"Using exported tier column '{tier_column}'.")
    frame["is_fragment_poor"] = frame["is_retained"] & frame["ms2_only_tier"].eq("D")
    # Evaluated = retained and not fragment-poor; every headline statistic uses this mask.
    frame["is_evaluated"] = frame["is_retained"] & ~frame["is_fragment_poor"]
    if include_tier_d:
        # Reproduction mode only: the submitted table evaluated every retained row, so
        # fragment-poor rows stay in the test family and in every count.
        frame["is_evaluated"] = frame["is_retained"]
        print("REPRODUCTION MODE (--include-tier-d): fragment-poor MS2-only rows are kept in the "
              "test family and in every count; the headline numbers equal the submitted family.")
    frame["high_score_annotation"] = (
        frame["is_evaluated"]
        & frame["is_annotated"]
        & frame["total_score"].ge(HIGH_SCORE_MIN_SCORE)
        & frame["n_matched"].ge(HIGH_SCORE_MIN_MATCHED)
    )

    raw = frame[HEIGHT_COLS].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
    raw = np.clip(raw, 0, None)
    # Median-scale each sample before feature-wise missing-value imputation.
    positive_medians = np.array([
        np.median(raw[:, j][raw[:, j] > 0]) if np.any(raw[:, j] > 0) else 1.0
        for j in range(raw.shape[1])
    ])
    target = float(np.median(positive_medians))
    normalized = raw * (target / positive_medians)[None, :]

    imputed = normalized.copy()
    for i in range(imputed.shape[0]):
        positive = imputed[i, imputed[i] > 0]
        floor = float(positive.min() / 5.0) if positive.size else 1.0
        imputed[i, imputed[i] <= 0] = floor
    log2 = np.log2(imputed)

    detected = np.zeros(raw.shape, dtype=bool)
    for j, col in enumerate(STATUS_COLS):
        detected[:, j] = frame[col].astype(str).str.lower().eq("detected").to_numpy()

    meta = {
        "input": str(input_path),
        "software_version": read_export_version(input_path),
        "n_rows_read": int(n_rows_read),
        "n_feature_rows": int(len(frame)),
        "n_retained_unclassified": n_unclassified,
        "tier_column": tier_column,
        "tier_is_proxy": bool(tier_is_proxy),
        "include_tier_d": bool(include_tier_d),
        "min_fragments": int(min_fragments),
        "precursor_columns_present": [c for c in PRECURSOR_COLUMNS if c in frame.columns],
    }
    return frame, raw, normalized, imputed, log2, detected, meta


def welch_statistics(log2: np.ndarray, imputed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """log2FC of normalized means, Welch p-value and the half-width of the 95 % CI."""
    ck, salt = slice(0, 3), slice(3, 6)
    log2fc = np.log2(imputed[:, salt].mean(axis=1) / imputed[:, ck].mean(axis=1))
    with warnings.catch_warnings():
        # Rows whose six values are (nearly) identical trigger a precision warning in
        # scipy; their p-value is NaN or 1 and they never pass any criterion.
        warnings.filterwarnings("ignore", message="Precision loss occurred", category=RuntimeWarning)
        with np.errstate(invalid="ignore", divide="ignore"):
            _, p = stats.ttest_ind(log2[:, salt], log2[:, ck], axis=1, equal_var=False)
    a, b = log2[:, ck], log2[:, salt]
    va, vb = a.var(axis=1, ddof=1), b.var(axis=1, ddof=1)
    se = np.sqrt(va / 3 + vb / 3)
    numerator = (va / 3 + vb / 3) ** 2
    denominator = (va / 3) ** 2 / 2 + (vb / 3) ** 2 / 2
    dof = np.divide(numerator, denominator, out=np.full_like(numerator, 4.0), where=denominator > 0)
    half_width = stats.t.ppf(0.975, dof) * se
    return log2fc, p, half_width


def add_statistics(
    frame: pd.DataFrame, imputed: np.ndarray, log2: np.ndarray, detected: np.ndarray
) -> pd.DataFrame:
    out = frame.copy()
    log2fc, p, half_width = welch_statistics(log2, imputed)
    ck, salt = slice(0, 3), slice(3, 6)
    # Test family: evaluated rows (fragment-poor MS2-only rows are not evaluated) that
    # were directly detected in >= 3 of 6 samples.  BH is applied over this family.
    eligible = (
        out["is_evaluated"].to_numpy()
        & (detected.sum(axis=1) >= MIN_DETECTED)
        & np.isfinite(p)
    )
    q = np.full(len(out), np.nan)
    q[eligible] = bh_adjust(p[eligible])
    differential = eligible & (q <= FDR_ALPHA) & (np.abs(log2fc) >= LOG2FC_MIN)
    exploratory = eligible & (p <= P_ALPHA) & (np.abs(log2fc) >= LOG2FC_MIN)

    out["mean_CK_normalized"] = imputed[:, ck].mean(axis=1)
    out["mean_salt_normalized"] = imputed[:, salt].mean(axis=1)
    out["log2FC_salt_vs_CK"] = log2fc
    out["log2FC_CI_low"] = log2fc - half_width
    out["log2FC_CI_high"] = log2fc + half_width
    out["p_value"] = p
    out["FDR_BH"] = q
    out["n_directly_detected_CK"] = detected[:, :3].sum(axis=1)
    out["n_directly_detected_salt"] = detected[:, 3:].sum(axis=1)
    out["eligible_for_differential_test"] = eligible
    out["differential"] = differential
    out["exploratory_p0.05"] = exploratory
    out["annotation_name_normalized"] = out["name"].map(normalize_name)
    return out


def submitted_family_check(frame: pd.DataFrame, imputed: np.ndarray, log2: np.ndarray, detected: np.ndarray) -> dict:
    """Statistics over the submitted test family (fragment-poor rows included).

    Reproduction check against the submitted Supplementary Table 5: the family is every
    retained row directly detected in >= 3 samples, regardless of ``ms2_only_tier``.
    """
    log2fc, p, _ = welch_statistics(log2, imputed)
    family = frame["is_retained"].to_numpy() & (detected.sum(axis=1) >= MIN_DETECTED) & np.isfinite(p)
    q = np.full(len(frame), np.nan)
    q[family] = bh_adjust(p[family])
    by_fdr = family & (q <= FDR_ALPHA) & (np.abs(log2fc) >= LOG2FC_MIN)
    by_p = family & (p <= P_ALPHA) & (np.abs(log2fc) >= LOG2FC_MIN)
    ms2 = frame["signal_type"].eq("ms2_only").to_numpy()
    high = frame["high_score_annotation"].to_numpy()
    tier_d = frame["is_fragment_poor"].to_numpy()
    ids = frame["feature_id"].astype(str).to_numpy()
    result = {
        "eligible": int(family.sum()),
        "differential_p": int(by_p.sum()),
        "differential_p_tier_D": int((by_p & tier_d).sum()),
        "differential_fdr": int(by_fdr.sum()),
        "differential_fdr_up": int((by_fdr & (log2fc > 0)).sum()),
        "differential_fdr_down": int((by_fdr & (log2fc < 0)).sum()),
        "differential_fdr_ms2_only": int((by_fdr & ms2).sum()),
        "differential_fdr_high_score": int((by_fdr & high).sum()),
        "differential_fdr_high_score_ms2_only": int((by_fdr & high & ms2).sum()),
        "differential_fdr_tier_D": int((by_fdr & tier_d).sum()),
    }
    for label, feature_id in STANDARD_FEATURES.items():
        idx = np.flatnonzero(ids == feature_id)
        if idx.size:
            result[f"q_{feature_id}"] = float(q[idx[0]])
            result[f"log2FC_{feature_id}"] = float(log2fc[idx[0]])
    return result


def pca_scores(stats_frame: pd.DataFrame, log2: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    usable = (
        stats_frame["is_evaluated"].to_numpy()
        & (pd.to_numeric(stats_frame["n_detected"], errors="coerce").fillna(0).to_numpy(float) >= MIN_DETECTED)
    )
    indices = np.flatnonzero(usable)
    variance = np.nanvar(log2[indices], axis=1, ddof=1)
    top = indices[np.argsort(variance)[-min(PCA_TOP_FEATURES, len(indices)):]]
    matrix = StandardScaler(with_mean=True, with_std=True).fit_transform(log2[top].T)
    model = PCA(n_components=2, random_state=20260722)
    scores = model.fit_transform(matrix)
    result = pd.DataFrame({
        "sample": SAMPLE_NAMES,
        "group": GROUPS,
        "PC1": scores[:, 0],
        "PC2": scores[:, 1],
    })
    return result, model.explained_variance_ratio_ * 100


def ranked_ms2_only_candidates(stats_frame: pd.DataFrame) -> pd.DataFrame:
    keep = (
        stats_frame["is_evaluated"]
        & stats_frame["high_score_annotation"]
        & stats_frame["differential"]
        & stats_frame["signal_type"].eq("ms2_only")
        & stats_frame["annotation_name_normalized"].ne("")
    )
    cols = [
        "feature_id", "mz", "rt", "annotation_name_normalized", "formula", "adduct",
        "signal_type", "detection_source",
        *[c for c in PRECURSOR_COLUMNS if c in stats_frame.columns],
        "total_score", "n_matched", "matched_pct",
        "n_fragments", "n_directly_detected_CK", "n_directly_detected_salt",
        "log2FC_salt_vs_CK", "log2FC_CI_low", "log2FC_CI_high", "p_value", "FDR_BH",
    ]
    ranked = stats_frame.loc[keep, cols].copy()
    return ranked.sort_values(["total_score", "n_matched", "FDR_BH"], ascending=[False, False, True])


def salt_response_status(row: pd.Series, confirmed: bool = False) -> str:
    """Data-derived salt-response status under the main criterion."""
    q = float(row["FDR_BH"])
    fc = float(row["log2FC_salt_vs_CK"])
    if bool(row["differential"]):
        head = "confirmed salt-responsive" if confirmed else "salt-responsive"
        return f"{head} (q = {q:.2g}, log2FC = {fc:.2f}; BH-FDR <= {FDR_ALPHA} and |log2FC| >= {LOG2FC_MIN:g})"
    reasons = []
    if not np.isfinite(q):
        reasons.append("not eligible for the differential test")
    else:
        if q > FDR_ALPHA:
            reasons.append(f"q = {q:.2g} > {FDR_ALPHA}")
        if abs(fc) < LOG2FC_MIN:
            reasons.append(f"|log2FC| = {abs(fc):.2f} < {LOG2FC_MIN:g}")
    return "exploratory (" + "; ".join(reasons) + ")"


def short_status(row: pd.Series, confirmed: bool = False) -> str:
    if bool(row["differential"]):
        return "confirmed salt-responsive" if confirmed else "salt-responsive"
    return "exploratory"


def selected_evidence(stats_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    by_id = stats_frame.set_index("feature_id", drop=False)
    for label, feature_id in STANDARD_FEATURES.items():
        row = by_id.loc[feature_id]
        confirmed = label == CONFIRMED_STANDARD
        rows.append({
            "display_name": label,
            "feature_id": feature_id,
            "evidence_level": (
                "Authentic standard (MSI level 1); identity confirmed by RT and MS2 match"
                if confirmed else
                "Authentic standard (MSI level 1); RT and MS2 match"
            ),
            "salt_response_status": salt_response_status(row, confirmed=confirmed),
            **row.to_dict(),
        })
    for label, feature_id in SELECTED_MS2_ONLY_CANDIDATES.items():
        row = by_id.loc[feature_id]
        rows.append({
            "display_name": label,
            "feature_id": feature_id,
            "evidence_level": CANDIDATE_EVIDENCE_LEVEL,
            "salt_response_status": salt_response_status(row),
            **row.to_dict(),
        })
    return pd.DataFrame(rows)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=LW, length=2.5)


def panel_letter(ax: plt.Axes, letter: str, dx: float = -28, dy: float = 5) -> None:
    ax.annotate(
        letter, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy),
        textcoords="offset points", fontsize=10, fontweight="bold",
        fontfamily=FONT_FAMILY, ha="left", va="bottom", color=INK, annotation_clip=False,
    )


def ellipse(ax: plt.Axes, x: np.ndarray, y: np.ndarray, colour: str) -> None:
    if len(x) < 3:
        return
    cov = np.cov(np.column_stack([x, y]), rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0)
    theta = np.linspace(0, 2 * np.pi, 160)
    radius = math.sqrt(5.991)  # 95 % normal ellipse, 2 df (descriptive only)
    xy = vecs @ (np.sqrt(vals)[:, None] * np.vstack([np.cos(theta), np.sin(theta)])) * radius
    ax.plot(xy[0] + x.mean(), xy[1] + y.mean(), color=colour, linewidth=LW)


def make_figure(
    stats_frame: pd.DataFrame,
    normalized: np.ndarray,
    pca: pd.DataFrame,
    explained: np.ndarray,
    evidence: pd.DataFrame,
    figure_dir: Path,
    stem: str,
) -> dict:
    """Draw Fig. 6 and return the plotted values of each panel (Source Data)."""
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": 8,
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT_FAMILY,
        "mathtext.it": f"{FONT_FAMILY}:italic",
        "mathtext.bf": f"{FONT_FAMILY}:bold",
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "axes.linewidth": LW,
        "axes.titlesize": 8,
        "axes.titleweight": "normal",
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": LW,
        "patch.linewidth": LW,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.dpi": 300,
    })
    source: dict[str, pd.DataFrame] = {}
    mm = 1 / 25.4
    fig = plt.figure(figsize=(210 * mm, 145 * mm))
    gs = fig.add_gridspec(
        2, 12, left=0.065, right=0.985, top=0.935, bottom=0.095,
        hspace=0.66, wspace=1.30, height_ratios=[1.0, 1.05]
    )
    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:7])
    ax_c = fig.add_subplot(gs[0, 7:12])
    d_gs = gs[1, 0:7].subgridspec(1, 3, wspace=0.52)
    axes_d = [fig.add_subplot(d_gs[0, i]) for i in range(3)]
    ax_e = fig.add_subplot(gs[1, 7:12])

    group_style = {g: (mix_with_white(c), c) for g, c in GROUP_COLOURS.items()}
    signal_style = {s: (mix_with_white(c), c) for s, c in SIGNAL_COLOURS.items()}

    # a: counts (not means) of evaluated features and high-scoring annotations by signal type.
    evaluated = stats_frame[stats_frame["is_evaluated"]]
    high = stats_frame[stats_frame["high_score_annotation"]]
    count_matrix = np.array([
        [evaluated["signal_type"].eq("ms1_detected").sum(), evaluated["signal_type"].eq("ms2_only").sum()],
        [high["signal_type"].eq("ms1_detected").sum(), high["signal_type"].eq("ms2_only").sum()],
    ], dtype=float)
    x = np.arange(2)
    scaled = count_matrix / 1000.0  # thousands keep the tick labels narrow
    ax_a.bar(x, scaled[:, 0], width=0.62, color=signal_style["ms1_detected"][0],
             edgecolor=signal_style["ms1_detected"][1], label="MS1-detected")
    ax_a.bar(x, scaled[:, 1], width=0.62, bottom=scaled[:, 0], color=signal_style["ms2_only"][0],
             edgecolor=signal_style["ms2_only"][1], label="MS2-only")
    ax_a.set_ylim(0, float(scaled.sum(axis=1).max()) * 1.30)
    for j in range(2):
        ax_a.annotate(f"{int(count_matrix[j].sum()):,}", xy=(j, scaled[j].sum()),
                      xytext=(0, 2.5), textcoords="offset points",
                      ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax_a.set_xticks(x, ["Retained\nfeatures", "High-scoring\nannotations"])
    ax_a.set_ylabel(r"Count ($\times$10$^3$)", labelpad=2)
    ax_a.set_title("Counts by signal type", fontweight="bold", pad=4)
    ax_a.legend(loc="upper right", handlelength=1.1, handletextpad=0.35, borderaxespad=0.15)
    style_axis(ax_a)
    panel_letter(ax_a, "a")
    source["Fig6a"] = pd.DataFrame({
        "bar": ["Retained features", "Retained features", "High-scoring annotations", "High-scoring annotations"],
        "segment": ["MS1-detected", "MS2-only", "MS1-detected", "MS2-only"],
        "count": [int(count_matrix[0, 0]), int(count_matrix[0, 1]), int(count_matrix[1, 0]), int(count_matrix[1, 1])],
        "bar_total": [int(count_matrix[0].sum())] * 2 + [int(count_matrix[1].sum())] * 2,
    })

    # b: unsupervised PCA; ellipses are descriptive only.
    for group, (fill, edge) in group_style.items():
        sub = pca[pca["group"] == group]
        ax_b.scatter(sub["PC1"], sub["PC2"], s=28, facecolor=fill, edgecolor=edge,
                     linewidth=LW, label=group, zorder=3)
        ellipse(ax_b, sub["PC1"].to_numpy(), sub["PC2"].to_numpy(), edge)
    ax_b.axhline(0, color=GRAY_FILL, linewidth=LW, zorder=0)
    ax_b.axvline(0, color=GRAY_FILL, linewidth=LW, zorder=0)
    ax_b.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax_b.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    ax_b.set_title("PCA of salt-stress profiles", fontweight="bold", pad=4)
    ax_b.legend(loc="best", handletextpad=0.25)
    style_axis(ax_b)
    panel_letter(ax_b, "b")
    pca_source = pca.copy()
    pca_source["PC1_explained_pct"] = float(explained[0])
    pca_source["PC2_explained_pct"] = float(explained[1])
    source["Fig6b"] = pca_source

    # c: differential landscape on the q scale (BH-adjusted p).
    qval = stats_frame["FDR_BH"].to_numpy(float)
    eligible = np.isfinite(qval)
    diff = stats_frame["differential"].to_numpy()
    ms2 = stats_frame["signal_type"].eq("ms2_only").to_numpy()
    xfc = stats_frame["log2FC_salt_vs_CK"].to_numpy(float)
    yval = -np.log10(np.clip(qval, 1e-300, 1))
    ax_c.scatter(xfc[eligible & ~diff], yval[eligible & ~diff], s=5,
                 facecolor=GRAY_FILL, edgecolor="none", alpha=0.45, rasterized=True)
    ax_c.scatter(xfc[diff & ~ms2], yval[diff & ~ms2], s=7,
                 facecolor=signal_style["ms1_detected"][0], edgecolor=signal_style["ms1_detected"][1],
                 linewidth=0.35, alpha=0.8)
    ax_c.scatter(xfc[diff & ms2], yval[diff & ms2], s=7,
                 facecolor=signal_style["ms2_only"][0], edgecolor=signal_style["ms2_only"][1],
                 linewidth=0.35, alpha=0.85)
    for xline in (-LOG2FC_MIN, LOG2FC_MIN):
        ax_c.axvline(xline, color=GRAY_EDGE, linestyle=(0, (3, 2)), linewidth=LW)
    ax_c.axhline(-math.log10(FDR_ALPHA), color=GRAY_EDGE, linestyle=(0, (3, 2)), linewidth=LW)
    y_top = float(np.nanmax(yval[eligible])) if np.any(eligible) else 1.0
    ax_c.set_ylim(-0.3, y_top * 1.32)
    label_offsets = {
        "3-formylindole": (9, 9),
        "β-carboline": (-9, -13),
        "Feruloyl putrescine": (10, -13),
    }
    label_alignment = {"3-formylindole": "left", "β-carboline": "right", "Feruloyl putrescine": "left"}
    standard_of = {fid: name for name, fid in STANDARD_FEATURES.items()}
    for label, feature_id in STANDARD_FEATURES.items():
        row = stats_frame.loc[stats_frame["feature_id"] == feature_id].iloc[0]
        point = (float(row["log2FC_salt_vs_CK"]), -math.log10(max(float(row["FDR_BH"]), 1e-300)))
        ax_c.scatter([point[0]], [point[1]], s=34, facecolor="white", edgecolor=INK, linewidth=1.1, zorder=6)
        ax_c.annotate(
            label, point, xytext=label_offsets[label], textcoords="offset points",
            fontsize=7, ha=label_alignment[label], va="center", zorder=7,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.7, "shrinkA": 1.0, "shrinkB": 3.0},
        )
    ax_c.set_xlabel(r"log$_2$ fold change (salt / CK)")
    ax_c.set_ylabel(r"$-$log$_{10}$ ($q$, BH-adjusted $p$)")
    ax_c.set_title("Salt-responsive feature landscape", fontweight="bold", pad=4)
    key = [
        Line2D([], [], linestyle="none", marker="o", markersize=3.4, markerfacecolor=GRAY_FILL,
               markeredgecolor="none", label="Not salt-responsive"),
        Line2D([], [], linestyle="none", marker="o", markersize=3.4,
               markerfacecolor=signal_style["ms1_detected"][0], markeredgecolor=signal_style["ms1_detected"][1],
               markeredgewidth=0.6, label="MS1-detected"),
        Line2D([], [], linestyle="none", marker="o", markersize=3.4,
               markerfacecolor=signal_style["ms2_only"][0], markeredgecolor=signal_style["ms2_only"][1],
               markeredgewidth=0.6, label="MS2-only"),
        Line2D([], [], linestyle="none", marker="o", markersize=4.2, markerfacecolor="white",
               markeredgecolor=INK, markeredgewidth=1.1, label="Authentic standard"),
    ]
    ax_c.legend(handles=key, loc="upper left", ncol=2, fontsize=7, handletextpad=0.2,
                columnspacing=0.9, labelspacing=0.35, borderaxespad=0.2, borderpad=0.2)
    style_axis(ax_c)
    panel_letter(ax_c, "c")
    plotted = stats_frame.loc[eligible, ["feature_id", "signal_type", "ms2_only_tier", "log2FC_salt_vs_CK",
                                         "p_value", "FDR_BH", "differential"]].copy()
    plotted["neg_log10_q"] = -np.log10(np.clip(plotted["FDR_BH"].to_numpy(float), 1e-300, 1))
    plotted["plotted_category"] = np.where(
        plotted["differential"], np.where(plotted["signal_type"].eq("ms2_only"), "MS2-only", "MS1-detected"),
        "Not salt-responsive")
    plotted["authentic_standard"] = plotted["feature_id"].map(standard_of).fillna("")
    source["Fig6c"] = plotted

    # d: all biological replicates for the three authentic standards (log2 median-normalized
    # intensity; points = replicates, bar = mean +/- s.d.); q and status in each panel.
    rng = np.random.default_rng(20260722)
    by_id = stats_frame.set_index("feature_id")
    d_rows = []
    for k, (ax, (label, feature_id)) in enumerate(zip(axes_d, STANDARD_FEATURES.items())):
        row_index = stats_frame.index[stats_frame["feature_id"].eq(feature_id)][0]
        vals = np.log2(np.clip(normalized[row_index], 0, None) + 1)
        for j, group in enumerate(["CK", "Salt"]):
            v = vals[:3] if group == "CK" else vals[3:]
            fill, edge = group_style[group]
            jitter = rng.normal(0, 0.035, size=3)
            ax.scatter(np.full(3, j) + jitter, v, s=18, facecolor=fill, edgecolor=edge, linewidth=LW, zorder=3)
            ax.errorbar(j, v.mean(), yerr=v.std(ddof=1), fmt="_", markersize=12, color=edge,
                        capsize=2.5, linewidth=LW, zorder=2)
            for sample, value, raw_value in zip(
                SAMPLE_NAMES[:3] if group == "CK" else SAMPLE_NAMES[3:], v,
                normalized[row_index][:3] if group == "CK" else normalized[row_index][3:]
            ):
                d_rows.append({
                    "display_name": label, "feature_id": feature_id, "sample": sample, "group": group,
                    "normalized_height": float(raw_value), "log2_normalized_plus1": float(value),
                    "group_mean": float(v.mean()), "group_sd": float(v.std(ddof=1)),
                })
        ax.set_xticks([0, 1], ["CK", "Salt"])
        if k == 0:
            ax.set_ylabel(r"log$_2$ (normalized intensity + 1)")
            panel_letter(ax, "d")
        ax.set_title(label, fontweight="bold", fontsize=8, pad=4)
        row = by_id.loc[feature_id]
        low, high_lim = ax.get_ylim()
        ax.set_ylim(low, high_lim + (high_lim - low) * 0.34)
        status = short_status(row, confirmed=(label == CONFIRMED_STANDARD))
        ax.text(0.5, 0.985, f"FC={2 ** row['log2FC_salt_vs_CK']:.2g}, $q$={row['FDR_BH']:.2g}",
                transform=ax.transAxes, ha="center", va="top", fontsize=7)
        ax.text(0.5, 0.905, status, transform=ax.transAxes, ha="center", va="top", fontsize=6.5)
        style_axis(ax)
    d_source = pd.DataFrame(d_rows)
    d_source["FC_salt_vs_CK"] = d_source["feature_id"].map(lambda f: float(2 ** by_id.loc[f, "log2FC_salt_vs_CK"]))
    d_source["FDR_BH"] = d_source["feature_id"].map(lambda f: float(by_id.loc[f, "FDR_BH"]))
    d_source["status"] = d_source["feature_id"].map(
        lambda f: short_status(by_id.loc[f], confirmed=(standard_of[f] == CONFIRMED_STANDARD)))
    source["Fig6d"] = d_source

    # e: effect sizes of the selected high-scoring MS2-only tentative candidates.
    cands = evidence[evidence["evidence_level"].eq(CANDIDATE_EVIDENCE_LEVEL)].copy()
    fill, edge = signal_style["ms2_only"]
    if cands.empty:
        ax_e.text(0.5, 0.5, "No MS2-only candidates selected", ha="center", va="center",
                  color=GRAY_EDGE, transform=ax_e.transAxes)
        ax_e.set_axis_off()
    else:
        cands = cands.sort_values("log2FC_salt_vs_CK")
        y = np.arange(len(cands))
        x0 = cands["log2FC_salt_vs_CK"].to_numpy(float)
        lo = cands["log2FC_CI_low"].to_numpy(float)
        hi = cands["log2FC_CI_high"].to_numpy(float)
        is_diff = cands["differential"].to_numpy(bool)
        ax_e.hlines(y, lo, hi, color=edge, linewidth=LW)
        ax_e.scatter(x0[is_diff], y[is_diff], s=35, facecolor=fill, edgecolor=edge, linewidth=LW, zorder=3)
        if np.any(~is_diff):
            ax_e.scatter(x0[~is_diff], y[~is_diff], s=35, facecolor="white", edgecolor=edge, linewidth=LW, zorder=3)
        ax_e.axvline(0, color=GRAY_EDGE, linestyle=(0, (3, 2)), linewidth=LW)
        plot_labels = cands["display_name"].replace({
            "Putative methoxylated flavanone glycoside": "Methoxylated flavanone\nglycoside (putative)"
        })
        ax_e.set_yticks(y)
        ax_e.set_yticklabels([])
        ax_e.set_ylim(-0.55, len(cands) - 0.35 + 0.85)
        right_limit = max(4.25, float(np.nanmax(hi)) + 1.3)
        ax_e.set_xlim(-1.95, right_limit)
        ax_e.set_xticks(np.arange(0, math.floor(right_limit - 1.0) + 1, 1))
        for yy, label, h, q, d in zip(y, plot_labels, hi, cands["FDR_BH"].to_numpy(float), is_diff):
            ax_e.text(-1.82, yy, str(label), ha="left", va="center", fontsize=7, linespacing=0.9)
            ax_e.text(h + 0.12, yy, f"$q$={q:.2g}" + ("" if d else " (exploratory)"),
                      ha="left", va="center", fontsize=6.5, color=INK)
        ax_e.set_xlabel(r"log$_2$ fold change (95% CI)")
        ax_e.set_title("Selected MS2-only tentative candidates", fontweight="bold", pad=4)
        handles = [Line2D([], [], linestyle="none", marker="o", markersize=4.5, markerfacecolor=fill,
                          markeredgecolor=edge, markeredgewidth=0.8, label=r"$q \leq 0.05$ and |log$_2$FC| $\geq$ 1")]
        if np.any(~is_diff):
            handles.append(Line2D([], [], linestyle="none", marker="o", markersize=4.5, markerfacecolor="white",
                                  markeredgecolor=edge, markeredgewidth=0.8, label="exploratory"))
        ax_e.legend(handles=handles, loc="upper left", fontsize=6.5, handletextpad=0.2,
                    borderaxespad=0.2, borderpad=0.2)
        style_axis(ax_e)
        panel_letter(ax_e, "e", dx=-16)
        e_source = cands[["display_name", "feature_id", "annotation_name_normalized", "signal_type", "ms2_only_tier",
                          "total_score", "n_matched", "n_fragments", "log2FC_salt_vs_CK", "log2FC_CI_low",
                          "log2FC_CI_high", "p_value", "FDR_BH", "differential", "salt_response_status"]].copy()
        e_source["marker"] = np.where(e_source["differential"], "filled", "open")
        source["Fig6e"] = e_source

    figure_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(figure_dir / f"{stem}.{ext}", dpi=600 if ext == "png" else None, facecolor="white")
    plt.close(fig)
    return source


def no_formulas(frame: pd.DataFrame) -> pd.DataFrame:
    """Prevent text that starts with '=' from being written as an Excel formula."""
    out = frame.copy()
    for col in out.columns:
        if out[col].dtype == object:
            mask = out[col].astype(str).str.startswith("=")
            if mask.any():
                out.loc[mask, col] = " " + out.loc[mask, col].astype(str)
    return out


def format_workbook(path: Path) -> None:
    from openpyxl import load_workbook

    book = load_workbook(path)
    try:
        for font in book._fonts:  # workbook-wide default font
            font.name = FONT_FAMILY
            font.sz = 11
    except Exception:
        pass
    header_fill = PatternFill("solid", fgColor=mix_with_white(BLUE, 0.8).lstrip("#").upper())
    for sheet in book.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        sheet.sheet_view.showGridLines = False
        for cell in sheet[1]:
            cell.font = Font(name=FONT_FAMILY, size=11, bold=True)
            cell.fill = header_fill
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        for column in sheet.iter_cols(min_row=1, max_row=min(sheet.max_row, 100)):
            letter = column[0].column_letter
            values = [str(c.value) if c.value is not None else "" for c in column]
            sheet.column_dimensions[letter].width = min(max(max((len(v) for v in values), default=0) + 2, 10), 36)
    book.save(path)


def table_columns(stats_frame: pd.DataFrame) -> list[str]:
    """Column order of Supplementary Table 5; names match the submitted table."""
    essential = [
        "feature_id", "mz", "rt", "signal_type", "detection_source",
        *[c for c in PRECURSOR_COLUMNS if c in stats_frame.columns],
        "name", "formula", "adduct", "total_score", "annotated", "high_score_annotation",
        "n_matched", "matched_pct", "n_fragments", "sn_ratio", "gaussian_similarity",
        "n_detected", "detection_rate", "cv", "mean_height", "mean_area",
        "log2FC_salt_vs_CK", "log2FC_CI_low", "log2FC_CI_high", "p_value", "FDR_BH",
        "n_directly_detected_CK", "n_directly_detected_salt",
        "eligible_for_differential_test", "differential", "exploratory_p0.05",
        *HEIGHT_COLS, *STATUS_COLS, "ms2_spectrum",
    ]
    missing = [c for c in essential if c not in stats_frame.columns]
    if missing:
        print("WARNING: columns missing from the export and skipped:", missing)
    return [c for c in essential if c in stats_frame.columns]


COLUMN_DEFINITIONS = {
    "feature_id": "Aligned feature identifier assigned by ASFAMProcessor.",
    "mz": "Reported precursor m/z of the feature.",
    "rt": "Retention time (min).",
    "signal_type": "ms1_detected = quantified on an MS1 peak; ms2_only = recovered from product-ion evidence only.",
    "detection_source": "Detection route (ms2_driven / ms1_driven).",
    "mz_source": "Origin of the reported precursor m/z of an MS2-only feature: ms1_peak, ms1_relaxed, nl_consensus (neutral-loss consensus), ms1_lowres (isolation-window centroid) or library (ASFAMProcessor >= v1.4).",
    "mz_confidence": "Confidence label attached to mz_source (high / medium / low; ASFAMProcessor >= v1.4).",
    "inferred_mz": "Precursor m/z inferred by the neutral-loss consensus, when applicable (ASFAMProcessor >= v1.4).",
    "nl_n_support": "Number of fragments supporting the neutral-loss consensus precursor (ASFAMProcessor >= v1.4).",
    "ms1_window_mz": "Intensity-weighted centroid of the 1 Da isolation window at the apex, stored as an independent cross-check (ASFAMProcessor >= v1.4).",
    "window_delta_mda": "Reported precursor minus ms1_window_mz, in mDa; blank where the two are the same estimate (ASFAMProcessor >= v1.4).",
    "ms2_only_tier": "Evidence tier of an MS2-only precursor: A = >= 3 fragments and a detected MS1 peak; B = >= 3 fragments and a neutral-loss consensus supported by >= 2 fragments; C = >= 3 fragments and weaker precursor evidence; D = fewer than 3 fragments (fragment-poor: retained in this table, not evaluated). Blank for MS1-detected features.",
    "name": "Library annotation (tentative candidate unless matched to an authentic standard).",
    "formula": "Library formula of the annotation.",
    "adduct": "Library adduct of the annotation.",
    "total_score": "Composite library-match score.",
    "annotated": "True when a library match was accepted by ASFAMProcessor.",
    "high_score_annotation": f"True when annotated, total_score >= {HIGH_SCORE_MIN_SCORE} and n_matched >= {HIGH_SCORE_MIN_MATCHED} (evaluated features only).",
    "n_matched": "Number of matched library peaks.",
    "matched_pct": "Fraction of library peaks matched.",
    "n_fragments": "Number of fragments in the representative MS2 spectrum.",
    "sn_ratio": "Signal-to-noise ratio of the representative peak.",
    "gaussian_similarity": "Peak-shape similarity of the representative peak.",
    "n_detected": "Number of samples in which the feature was directly detected (gap-filled values excluded).",
    "detection_rate": "n_detected / 6.",
    "cv": "Coefficient of variation over directly detected samples.",
    "mean_height": "Mean peak height over directly detected samples.",
    "mean_area": "Mean peak area over directly detected samples.",
    "log2FC_salt_vs_CK": "log2 of the ratio of mean median-normalized, imputed intensities (salt / CK).",
    "log2FC_CI_low": "Lower bound of the Welch 95 % confidence interval of the log2 difference.",
    "log2FC_CI_high": "Upper bound of the Welch 95 % confidence interval of the log2 difference.",
    "p_value": "Two-sided Welch t-test p-value on log2 normalized intensities (descriptive outside the test family).",
    "FDR_BH": "Benjamini-Hochberg adjusted p-value (q) over the test family; blank for features that were not tested.",
    "n_directly_detected_CK": "Directly detected CK replicates (0-3).",
    "n_directly_detected_salt": "Directly detected salt replicates (0-3).",
    "eligible_for_differential_test": f"True when evaluated (ms2_only_tier != D), directly detected in >= {MIN_DETECTED} of 6 samples and the test is defined; this is the BH family.",
    "differential": f"MAIN CRITERION (revised): eligible and FDR_BH <= {FDR_ALPHA} and |log2FC_salt_vs_CK| >= {LOG2FC_MIN:g}.",
    "exploratory_p0.05": f"Submitted criterion, kept for reference only: eligible and p_value <= {P_ALPHA} and |log2FC_salt_vs_CK| >= {LOG2FC_MIN:g}.",
    "height_rep1..6": "Peak height per sample (rep1-3 = CK1-3, rep4-6 = Salt1-3), gap-filled where not directly detected.",
    "gap_fill_status_rep1..6": "detected = directly detected; filled = gap-filled from the raw data; no_signal = no signal found.",
    "ms2_spectrum": f"Representative MS2 spectrum (m/z:intensity pairs), truncated at {EXCEL_CELL_LIMIT:,} characters for Excel; full spectra are in the MGF export.",
}


def readme_frame(meta: dict, counts: dict) -> pd.DataFrame:
    proxy = meta["tier_is_proxy"]
    tier_note = (
        f"THIS DRAFT: the input export (ASFAMProcessor v{meta['software_version']}) predates the tier column, so "
        f"ms2_only_tier is a PROXY: D when signal_type = ms2_only and n_fragments < {meta['min_fragments']}, otherwise "
        "blank (unknown; tiers A-C cannot be derived without mz_source). The other precursor-source columns are absent."
        if proxy else
        f"ms2_only_tier and the precursor-source columns are taken unchanged from the ASFAMProcessor v{meta['software_version']} export."
    )
    items = [
        ("Title", "Supplementary Table 5. ASFAMProcessor analysis of rice leaves under salt stress (revised: FDR-based main criterion, MS2-only evidence tiers)."),
        ("Input", f"ASFAMProcessor v{meta['software_version']} features.csv generated from {N_INPUT_FILES} mzML files (6 samples x 31 segments)."),
        ("Sample mapping", "rep1-3 = CK1-3; rep4-6 = salt-stress biological replicates 1-3."),
        ("Retained after redundancy marking", "Rows with is_duplicate = False after isotope, adduct, spectral-duplicate and in-source-fragment marking, and with a signal type (ms1_detected or ms2_only). ASFAMProcessor marks redundant rows instead of deleting them; the marked rows are not part of this table."),
        ("MS2-only evidence tiers", COLUMN_DEFINITIONS["ms2_only_tier"] + " " + tier_note),
        ("Fragment-poor MS2-only features (tier D)", (
            f"REPRODUCTION MODE (--include-tier-d): the {counts['ms2_only_tier_D']:,} fragment-poor rows were kept in the test family and in every count, as in the submitted table. Not for publication."
            if meta.get("include_tier_d") else
            f"Retained in sheet All_features_retained but NOT evaluated: excluded from the test family, from high-scoring annotations and from every count reported in the manuscript ({counts['ms2_only_tier_D']:,} rows in this table). They are reported once as 'fragment-poor MS2-only features: retained in the table, not evaluated'.")),
        ("Annotation evidence", "Library matches are tentative candidates (MSI level 2) unless matched to an authentic standard; MS2-only annotations in particular are tentative candidates, not identified metabolites. Precursor m/z of MS2-only features is inferred (see mz_source / ms2_only_tier), not measured on an MS1 peak."),
        ("High-scoring annotation rule", f"Evaluated, annotated = True, total_score >= {HIGH_SCORE_MIN_SCORE} and >= {HIGH_SCORE_MIN_MATCHED} matched MS2 peaks."),
        ("Differential test", f"Two-sided Welch t-test on log2 median-normalized intensities of evaluated features directly detected in >= {MIN_DETECTED} of 6 samples (test family, n = {counts['eligible']:,}); Benjamini-Hochberg adjustment over this family. MAIN CRITERION: differential = FDR_BH <= {FDR_ALPHA} and |log2FC| >= {LOG2FC_MIN:g} (n = {counts['differential']:,}). The submitted criterion (p <= {P_ALPHA} and |log2FC| >= {LOG2FC_MIN:g}) is kept as the reference column exploratory_p0.05 (n = {counts['exploratory_p0.05']:,}) and is not used for any claim."),
        ("Missing values", "Zero values were imputed per feature as one-fifth of its minimum positive normalized intensity."),
        ("PCA", f"Unsupervised PCA on the {PCA_TOP_FEATURES:,} most variable evaluated features (directly detected in >= {MIN_DETECTED} samples) after log2 transformation and autoscaling."),
        ("MS2 spectra", f"The spectrum column is capped at {EXCEL_CELL_LIMIT:,} characters for Excel; full spectra are in the MGF export."),
        ("Sheets", "All_features_retained = every feature retained after redundancy marking (the first data sheet of the submitted version, renamed; column names unchanged except the added evidence-tier / precursor-source columns and exploratory_p0.05); High_score_annotations; Differential_features (main criterion); MS2only_candidate_ranking (differential, high-scoring MS2-only tentative candidates ranked by score); Selected_evidence (authentic standards and the Fig. 6e candidates with their data-derived salt-response status); Fig6_PCA_source; Fig6_selected_source."),
    ]
    rows = [{"Item": k, "Description": v} for k, v in items]
    rows.append({"Item": "--- Column definitions ---", "Description": ""})
    for column, definition in COLUMN_DEFINITIONS.items():
        if column in ("mz_source", "mz_confidence", "inferred_mz", "nl_n_support", "ms1_window_mz", "window_delta_mda") and proxy:
            continue
        rows.append({"Item": column, "Description": definition})
    return pd.DataFrame(rows)


def export_tables(
    stats_frame: pd.DataFrame,
    normalized: np.ndarray,
    pca: pd.DataFrame,
    evidence: pd.DataFrame,
    ranked: pd.DataFrame,
    meta: dict,
    counts: dict,
    analysis_dir: Path,
) -> tuple[Path, pd.DataFrame]:
    path = analysis_dir / "rice_table5_draft.xlsx"
    columns = table_columns(stats_frame)
    all_retained = stats_frame.loc[stats_frame["is_retained"], columns].copy()
    all_retained["ms2_spectrum"] = all_retained["ms2_spectrum"].fillna("").astype(str).str.slice(0, EXCEL_CELL_LIMIT)
    high = all_retained[all_retained["high_score_annotation"]].copy()
    diff = all_retained[all_retained["differential"]].copy()
    source_rows = []
    for label, feature_id in {**STANDARD_FEATURES, **SELECTED_MS2_ONLY_CANDIDATES}.items():
        idx = stats_frame.index[stats_frame["feature_id"].eq(feature_id)][0]
        for sample, group, value in zip(SAMPLE_NAMES, GROUPS, normalized[idx]):
            source_rows.append({
                "panel": "d" if feature_id in STANDARD_FEATURES.values() else "e",
                "display_name": label,
                "feature_id": feature_id,
                "sample": sample,
                "group": group,
                "normalized_height": value,
            })
    source = pd.DataFrame(source_rows)
    readme = readme_frame(meta, counts)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        no_formulas(all_retained).to_excel(writer, sheet_name="All_features_retained", index=False)
        no_formulas(high).to_excel(writer, sheet_name="High_score_annotations", index=False)
        no_formulas(diff).to_excel(writer, sheet_name="Differential_features", index=False)
        no_formulas(ranked).to_excel(writer, sheet_name="MS2only_candidate_ranking", index=False)
        no_formulas(evidence).to_excel(writer, sheet_name="Selected_evidence", index=False)
        pca.to_excel(writer, sheet_name="Fig6_PCA_source", index=False)
        source.to_excel(writer, sheet_name="Fig6_selected_source", index=False)
    format_workbook(path)
    return path, all_retained


def export_source_data(panel_sources: dict, summary: pd.DataFrame, analysis_dir: Path) -> Path:
    path = analysis_dir / "rice_source_data_fig6.xlsx"
    readme = summary[["panel", "item", "value", "caption_note"]].copy()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        for sheet, frame in panel_sources.items():
            no_formulas(frame).to_excel(writer, sheet_name=sheet, index=False)
    format_workbook(path)
    return path


def headline_counts(stats_frame: pd.DataFrame) -> dict:
    retained = stats_frame["is_retained"]
    evaluated = stats_frame["is_evaluated"]
    ms2 = stats_frame["signal_type"].eq("ms2_only")
    ms1 = stats_frame["signal_type"].eq("ms1_detected")
    high = stats_frame["high_score_annotation"]
    diff = stats_frame["differential"]
    expl = stats_frame["exploratory_p0.05"]
    fc = stats_frame["log2FC_salt_vs_CK"]
    counts = {
        "retained_rows": int(retained.sum()),
        "retained_ms1_detected": int((retained & ms1).sum()),
        "retained_ms2_only": int((retained & ms2).sum()),
        "ms2_only_tier_D": int(stats_frame["is_fragment_poor"].sum()),
        "evaluated_features": int(evaluated.sum()),
        "evaluated_ms1_detected": int((evaluated & ms1).sum()),
        "evaluated_ms2_only": int((evaluated & ms2).sum()),
        "annotated_evaluated": int((evaluated & stats_frame["is_annotated"]).sum()),
        "annotated_tier_D": int((stats_frame["is_fragment_poor"] & stats_frame["is_annotated"]).sum()),
        "high_score_annotations": int(high.sum()),
        "high_score_ms2_only": int((high & ms2).sum()),
        "eligible": int(stats_frame["eligible_for_differential_test"].sum()),
        "differential": int(diff.sum()),
        "differential_up": int((diff & fc.gt(0)).sum()),
        "differential_down": int((diff & fc.lt(0)).sum()),
        "differential_ms2_only": int((diff & ms2).sum()),
        "differential_high_score": int((diff & high).sum()),
        "differential_high_score_ms2_only": int((diff & high & ms2).sum()),
        "exploratory_p0.05": int(expl.sum()),
        "exploratory_p0.05_ms2_only": int((expl & ms2).sum()),
    }
    for tier, n in stats_frame.loc[retained & ms2, "ms2_only_tier"].value_counts().items():
        counts[f"ms2_only_tier_{tier or 'unknown'}"] = int(n)
    for tier, n in stats_frame.loc[diff & ms2, "ms2_only_tier"].value_counts().items():
        counts[f"differential_ms2_only_tier_{tier or 'unknown'}"] = int(n)
    return counts


def build_summary(counts: dict, explained: np.ndarray, evidence: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Per-panel values and the caption notes to be transferred into the Fig. 6 legend."""
    proxy = " (proxy tier on this export: n_fragments < 3)" if meta["tier_is_proxy"] else ""
    standards = evidence[evidence["feature_id"].isin(STANDARD_FEATURES.values())]
    standard_text = "; ".join(
        f"{r.display_name}: {r.salt_response_status}" for r in standards.itertuples()
    )
    cands = evidence[evidence["evidence_level"].eq(CANDIDATE_EVIDENCE_LEVEL)]
    n_cand_diff = int(cands["differential"].sum())
    rows = [
        {"panel": "a", "item": "Retained features (MS1-detected / MS2-only)",
         "value": f"{counts['evaluated_ms1_detected']:,} / {counts['evaluated_ms2_only']:,} (total {counts['evaluated_features']:,})",
         "caption_note": ("counts, not means. Bars show features retained after redundancy marking and evaluated "
                          f"(fragment-poor MS2-only features of tier D, n = {counts['ms2_only_tier_D']:,}{proxy}, are retained "
                          "in Supplementary Table 5 but not evaluated and not shown) and high-scoring annotations "
                          f"(total score >= {HIGH_SCORE_MIN_SCORE}, >= {HIGH_SCORE_MIN_MATCHED} matched peaks); MS2-only "
                          "annotations are tentative candidates.")},
        {"panel": "a", "item": "High-scoring annotations (MS1-detected / MS2-only)",
         "value": f"{counts['high_score_annotations'] - counts['high_score_ms2_only']:,} / {counts['high_score_ms2_only']:,} (total {counts['high_score_annotations']:,})",
         "caption_note": "counts, not means."},
        {"panel": "b", "item": "PCA explained variance PC1 / PC2 (%)",
         "value": f"{explained[0]:.1f} / {explained[1]:.1f}",
         "caption_note": (f"PCA of the {PCA_TOP_FEATURES:,} most variable evaluated features (median-normalized, log2, "
                          "autoscaled); points are biological replicates (n = 3 per group); ellipses are descriptive "
                          "95 % covariance regions.")},
        {"panel": "c", "item": "Tested / salt-responsive features",
         "value": f"{counts['eligible']:,} / {counts['differential']:,} ({counts['differential_up']:,} up, {counts['differential_down']:,} down; MS2-only {counts['differential_ms2_only']:,})",
         "caption_note": (f"y axis is -log10 of the BH-adjusted p-value (q); horizontal line q = {FDR_ALPHA}, vertical lines "
                          f"|log2 fold change| = {LOG2FC_MIN:g}; coloured points meet both (main criterion) and are "
                          "coloured by signal type; the three authentic standards are ringed and labelled.")},
        {"panel": "d", "item": "Authentic standards",
         "value": standard_text,
         "caption_note": ("points are individual biological replicates (n = 3 per group) of log2(median-normalized "
                          "intensity + 1); bars show mean +/- s.d. (NOTE: the submitted legend said 95 % CI and "
                          "'normalized', while the submitted panel drew s.d. of raw heights; this draft plots normalized "
                          "intensities and states s.d.); fold change and q (BH) are shown; only 3-formylindole is "
                          "described as confirmed salt-responsive, feruloyl putrescine and beta-carboline are exploratory.")},
        {"panel": "e", "item": "Selected MS2-only tentative candidates (salt-responsive / shown)",
         "value": f"{n_cand_diff} / {len(cands)}",
         "caption_note": ("log2 fold change with Welch 95 % CI for four high-scoring MS2-only library matches "
                          "(tentative candidates, MSI level 2); filled markers meet the main criterion, open markers "
                          "are exploratory; q values are printed. No claim about prior reports in rice is made.")},
        {"panel": "all", "item": "Fragment-poor MS2-only features (tier D)",
         "value": f"{counts['ms2_only_tier_D']:,}",
         "caption_note": "retained in Supplementary Table 5, not evaluated; excluded from every count in this figure" + proxy + "."},
    ]
    return pd.DataFrame(rows)


def build_numbers(counts: dict, check: dict, evidence: pd.DataFrame, explained: np.ndarray, meta: dict, run_label: str) -> dict:
    src = f"scripts/analyze_rice.py (run label '{run_label}', input {Path(meta['input']).name}, ASFAMProcessor v{meta['software_version']})"
    proxy_note = (
        f"tier is a PROXY (D = ms2_only and n_fragments < {meta['min_fragments']}; A-C unknown)"
        if meta["tier_is_proxy"] else "tier from the export"
    )
    numbers: dict[str, dict] = {}

    def put(key: str, value, note: str = "") -> None:
        if isinstance(value, (np.integer,)):
            value = int(value)
        elif isinstance(value, (np.floating,)):
            value = float(value)
        entry = {"value": value, "source": src}
        if note:
            entry["note"] = note
        numbers[f"rice_{key}"] = entry

    put("software_version", meta["software_version"], "features.csv header")
    put("tier_proxy_used", bool(meta["tier_is_proxy"]), proxy_note)
    put("include_tier_d_mode", bool(meta["include_tier_d"]),
        "True only in the --include-tier-d reproduction run (tier D kept in the test family); headline runs are False")
    put("input_rows_all", meta["n_feature_rows"], "all aligned rows incl. is_duplicate = True; never a feature count")
    put("retained_rows", counts["retained_rows"], "is_duplicate = False and classified (ms1_detected / ms2_only) = Table 5 All_features_retained")
    put("retained_unclassified", meta["n_retained_unclassified"], "is_duplicate = False but no signal type; counted, not tabulated")
    put("retained_ms1_detected", counts["retained_ms1_detected"])
    put("retained_ms2_only", counts["retained_ms2_only"], "all tiers incl. D")
    put("ms2_only_tier_D", counts["ms2_only_tier_D"], "fragment-poor MS2-only: retained in the table, not evaluated; " + proxy_note)
    for key in sorted(k for k in counts if k.startswith("ms2_only_tier_") and k != "ms2_only_tier_D"):
        put(key, counts[key], proxy_note)
    put("evaluated_features", counts["evaluated_features"], "retained minus tier D = headline feature count")
    put("evaluated_ms1_detected", counts["evaluated_ms1_detected"], "headline")
    put("evaluated_ms2_only", counts["evaluated_ms2_only"], "headline (tier D excluded)")
    put("annotated_evaluated", counts["annotated_evaluated"], "annotated = True among evaluated features")
    put("annotated_tier_D", counts["annotated_tier_D"], "annotated rows among tier D (excluded from headline)")
    put("high_score_annotations", counts["high_score_annotations"], f"total_score >= {HIGH_SCORE_MIN_SCORE} and n_matched >= {HIGH_SCORE_MIN_MATCHED}, evaluated")
    put("high_score_ms2_only", counts["high_score_ms2_only"], "tentative candidates")
    put("eligible", counts["eligible"], f"test family: evaluated and directly detected in >= {MIN_DETECTED}/6 samples")
    put("differential", counts["differential"], f"MAIN: BH-FDR <= {FDR_ALPHA} and |log2FC| >= {LOG2FC_MIN:g} over the test family")
    put("differential_up", counts["differential_up"])
    put("differential_down", counts["differential_down"])
    put("differential_ms2_only", counts["differential_ms2_only"])
    put("differential_high_score", counts["differential_high_score"])
    put("differential_high_score_ms2_only", counts["differential_high_score_ms2_only"])
    for key in sorted(k for k in counts if k.startswith("differential_ms2_only_tier_")):
        put(key, counts[key], proxy_note)
    put("exploratory_p0.05", counts["exploratory_p0.05"], f"submitted criterion p <= {P_ALPHA} and |log2FC| >= {LOG2FC_MIN:g} over the new test family (tier D excluded); reference only")
    put("exploratory_p0.05_ms2_only", counts["exploratory_p0.05_ms2_only"], "reference only")
    put("pca_pc1_pct", round(float(explained[0]), 2))
    put("pca_pc2_pct", round(float(explained[1]), 2))

    for row in evidence.itertuples():
        slug = re.sub(r"[^a-z0-9]+", "_", str(row.display_name).lower().replace("β", "beta")).strip("_")
        note = f"{row.feature_id} {row.signal_type}; {row.salt_response_status}"
        put(f"{slug}_log2FC", round(float(row.log2FC_salt_vs_CK), 4), note)
        put(f"{slug}_FC", round(float(2 ** row.log2FC_salt_vs_CK), 3), note)
        put(f"{slug}_p", float(f"{row.p_value:.4g}"), note)
        put(f"{slug}_q", float(f"{row.FDR_BH:.4g}"), note)
        put(f"{slug}_differential", bool(row.differential), note)

    rep = "reproduction check: submitted test family (tier D rows included; q identical to the submitted Table 5)"
    put("submitted_family_eligible", check["eligible"], rep)
    put("submitted_family_differential_p", check["differential_p"], rep + f"; p <= {P_ALPHA} and |log2FC| >= {LOG2FC_MIN:g} (submitted criterion)")
    put("submitted_family_differential_p_tier_D", check["differential_p_tier_D"], rep)
    put("submitted_family_differential_fdr", check["differential_fdr"], rep + "; new criterion applied to the submitted family")
    put("submitted_family_differential_fdr_up", check["differential_fdr_up"], rep)
    put("submitted_family_differential_fdr_down", check["differential_fdr_down"], rep)
    put("submitted_family_differential_fdr_ms2_only", check["differential_fdr_ms2_only"], rep)
    put("submitted_family_differential_fdr_high_score", check["differential_fdr_high_score"], rep)
    put("submitted_family_differential_fdr_high_score_ms2_only", check["differential_fdr_high_score_ms2_only"], rep)
    put("submitted_family_differential_fdr_tier_D", check["differential_fdr_tier_D"], rep + "; rows that drop out when tier D is not evaluated")
    for label, feature_id in STANDARD_FEATURES.items():
        slug = re.sub(r"[^a-z0-9]+", "_", label.lower().replace("β", "beta")).strip("_")
        if f"q_{feature_id}" in check:
            put(f"submitted_family_{slug}_q", float(f"{check[f'q_{feature_id}']:.4g}"), rep)
    return numbers


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="ASFAMProcessor features.csv")
    parser.add_argument("--run-label", default="old", help="suffix of the output folder / files (old, rerun, ...)")
    parser.add_argument("--analysis-dir", type=Path, default=None, help="default: <revision>/analysis/rice_<label>")
    parser.add_argument("--figure-dir", type=Path, default=None, help="default: <revision>/figures/rice_preview")
    parser.add_argument("--figure-stem", default=None, help="default: fig6_<label>")
    parser.add_argument("--tier-column", default="ms2_only_tier",
                        help="MS2-only tier column of the export; when absent a proxy is derived (D = n_fragments < --min-fragments)")
    parser.add_argument("--min-fragments", type=int, default=3, help="fragment threshold of the proxy tier D")
    parser.add_argument("--include-tier-d", action="store_true",
                        help="reproduction mode: keep fragment-poor MS2-only rows in the test family and in every "
                             "count (reproduces the submitted family; not for publication)")
    args = parser.parse_args(argv)
    if args.include_tier_d and args.analysis_dir is None and not args.run_label.endswith("_include_tier_d"):
        args.run_label = f"{args.run_label}_include_tier_d"
    args.analysis_dir = args.analysis_dir or REVISION_DIR / "analysis" / f"rice_{args.run_label}"
    args.figure_dir = args.figure_dir or REVISION_DIR / "figures" / "rice_preview"
    args.figure_stem = args.figure_stem or f"fig6_{args.run_label}"
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    print(f"font family: {FONT_FAMILY}")
    frame, raw, normalized, imputed, log2, detected, meta = prepare_data(
        args.input, args.tier_column, args.min_fragments, include_tier_d=args.include_tier_d)
    stats_frame = add_statistics(frame, imputed, log2, detected)
    check = submitted_family_check(stats_frame, imputed, log2, detected)
    counts = headline_counts(stats_frame)
    pca, explained = pca_scores(stats_frame, log2)
    ranked = ranked_ms2_only_candidates(stats_frame)
    evidence = selected_evidence(stats_frame)
    summary = build_summary(counts, explained, evidence, meta)
    numbers = build_numbers(counts, check, evidence, explained, meta, args.run_label)

    panel_sources = make_figure(stats_frame, normalized, pca, explained, evidence, args.figure_dir, args.figure_stem)
    table_path, all_retained = export_tables(stats_frame, normalized, pca, evidence, ranked, meta, counts, args.analysis_dir)
    source_path = export_source_data(panel_sources, summary, args.analysis_dir)

    all_retained.to_csv(args.analysis_dir / "rice_feature_table.csv", index=False)
    ranked.to_csv(args.analysis_dir / "rice_ms2only_candidate_ranking.csv", index=False)
    evidence.to_csv(args.analysis_dir / "rice_selected_evidence.csv", index=False)
    summary.to_csv(args.analysis_dir / "rice_summary.csv", index=False)
    numbers_path = args.analysis_dir / f"numbers_rice_{args.run_label}.json"
    numbers_path.write_text(json.dumps(numbers, ensure_ascii=False, indent=2), encoding="utf-8")

    mode = ("REPRODUCTION MODE: fragment-poor MS2-only features included" if args.include_tier_d
            else "headline (fragment-poor MS2-only features excluded)")
    print(f"\n=== {mode} ===")
    for key, value in counts.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
    print(f"  PCA explained: PC1 {explained[0]:.1f} %, PC2 {explained[1]:.1f} %")
    print("\n=== authentic standards and selected candidates ===")
    for row in evidence.itertuples():
        print(f"  {row.display_name} ({row.feature_id}, {row.signal_type}): log2FC {row.log2FC_salt_vs_CK:.3f}, "
              f"p {row.p_value:.3g}, q {row.FDR_BH:.4g} -> {row.salt_response_status}")
    print("\n=== reproduction check over the submitted test family (tier D included) ===")
    for key, value in check.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value:.4g}")
    print("\noutputs:")
    for path in (table_path, source_path, args.analysis_dir / "rice_feature_table.csv",
                 args.analysis_dir / "rice_summary.csv", numbers_path,
                 args.figure_dir / f"{args.figure_stem}.png"):
        print("  ", path)


if __name__ == "__main__":
    main()
