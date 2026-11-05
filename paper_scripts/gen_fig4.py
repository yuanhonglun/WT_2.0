"""gen_fig4.py — Fig. 4 (validation of ASFAMProcessor against MS-DIAL on the
rep-3 full-range data set) regenerated on the v1.4 re-run of the 2026-10 revision.

Adapted copy of manuscript/fig 4/gen_fig4.py (the submitted figure). Panel logic
a–d is unchanged; what changed:

  * ASFAMProcessor side = runs/rep3_full/features.csv (v1.4.261020; same 31 rep-3
    mzML files). Its schema is a superset of the old one: ``segment_name`` is
    taken from ``alignment_segment``, ``rt_apex`` from ``rt``, ``mz_source`` is a
    real column. Rows are read by skipping the "# key=value" header lines (NOT
    ``comment="#"``: the library name ascr#17 would be truncated).
  * MS-DIAL references are the CACHED msdial_ref_strict.csv / msdial_ref_lenient.csv
    of manuscript/fig 4 (built once with the integration tool; never rebuilt here).
  * Palette: Okabe–Ito subset (fig4_common): ASFAMProcessor blue #0072B2, MS-DIAL
    orange #E69F00, MS2-only purple #CC79A7; light fill + full-colour edge.
  * Panels e/f = the NL-consensus self-validation re-run on the v1.4 checkpoint
    (scripts/nl_self_validation.py -> analysis/rep3_full/inference_self_validation.csv),
    stratified by n_support (1 / 2 / 3 / >=4) instead of the old two-tier split.
  * A supplementary figure fig4_supp_nl: NL-vs-window-centroid agreement on real
    MS2-only features by n_support (analysis/rep3_full/consistency_by_support.csv,
    S2.1c) and the annotation outcome under the precursor estimates / NL windows
    (analysis/rep3_full/annotation_outcome_summary.csv, S2.1e).
  * fig4_supp_recovery = the old fig4_supp (lenient reference + height-stratified
    recall).

Outputs: figures/fig4/fig4.{svg,pdf,png}, fig4_supp_nl.{svg,pdf,png},
fig4_supp_recovery.{svg,pdf,png}, legend_fig4.txt; analysis/rep3_full/
match_table_{strict,lenient}.csv, recovery_summary_strict.csv,
similarity_pairs_strict.csv, mirror_pair_strict.csv, height_stratified_recall.csv,
numbers_fig4.json (keys fig4_*).

Usage:  python gen_fig4.py [--run DIR] [--ref-dir DIR] [--analysis DIR] [--figdir DIR]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig4_common import (  # noqa: E402
    DDA_ORANGE, GRAY, GRAY_LIGHT, LW, MANUSCRIPT_DIR, MS2ONLY_PURPLE, REVISION_DIR, SKY_BLUE,
    SUPPORT_LEVELS, TEXT, VERMILION, WT_BLUE, add_label, despine, light, read_features, save_all,
    setup_style, support_colours, tint, update_numbers, parse_spectrum,
)
import matplotlib.pyplot as plt  # noqa: E402
from metabo_core.algorithms.similarity import composite_similarity  # noqa: E402

DEFAULT_RUN = REVISION_DIR / "runs" / "rep3_full"
DEFAULT_REF = MANUSCRIPT_DIR / "fig 4"          # cached MS-DIAL references (read-only)
DEFAULT_ANALYSIS = REVISION_DIR / "analysis" / "rep3_full"
DEFAULT_FIGDIR = REVISION_DIR / "figures" / "fig4"

CLASS_ORDER = ["Primary", "Linked-isotope", "Linked-adduct", "Linked-isf", "Linked-spectral", "Missed"]
_CLASS_TINT = {"Primary": 1.0, "Linked-isotope": 0.7, "Linked-adduct": 0.5, "Linked-isf": 0.35,
               "Linked-spectral": 0.2}
CLASS_S = {c: tint(WT_BLUE, f) for c, f in _CLASS_TINT.items()}
CLASS_S["Missed"] = GRAY
CLASS_F = {c: light(s) for c, s in CLASS_S.items()}
CLASS_F["Missed"] = GRAY_LIGHT
SUPPORT_S = support_colours(WT_BLUE)
SUPPORT_F = {k: light(v) for k, v in SUPPORT_S.items()}
SUPPORT_LABEL = {"1": "1 vote", "2": "2 votes", "3": "3 votes", ">=4": "≥4 votes"}

SEG_RE = re.compile(r"_(\d+-\d+)_")


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def seg_from_mz(mz):
    if mz is None or not np.isfinite(mz):
        return None
    ch = int(round(mz)); k = max(0, min(30, (ch - 75) // 30)); lo = 75 + 30 * k
    return f"{lo}-{lo + 29}"


def load_asfam(path: Path) -> pd.DataFrame:
    """features.csv of the v1.4 run, every row (duplicates are needed for the
    Linked-* classes); schema adapted to the legacy matcher."""
    df = read_features(path, retained_only=False)
    df["rt_apex"] = df["rt"]
    df["segment_name"] = df["alignment_segment"].where(df["alignment_segment"] != "",
                                                       df["precursor_mz"].apply(seg_from_mz))
    df["is_duplicate"] = df["is_duplicate"].astype(int)
    return df


def load_candidates(run: Path) -> pd.DataFrame:
    """Every candidate feature of the per-sample checkpoint (before alignment),
    in the legacy matcher's schema. Used for the candidate-level recovery that
    tells how many reference features were lost by the stage-7 fold rather than
    never detected."""
    import pickle
    rows = []
    for mf in sorted((run / "_work").glob("*.mfeat")):
        with open(mf, "rb") as fh:
            payload = pickle.load(fh)
        for r in payload["features"]:
            prec = r.get("ms1_precursor_mz")
            if prec is None:
                prec = r.get("inferred_mz")
            if prec is None:
                prec = r.get("precursor_mz_nominal")
            rows.append({"feature_id": r["feature_id"], "precursor_mz": float(prec), "rt_apex": float(r["rt_apex"]),
                         "segment_name": r.get("segment_name"), "is_duplicate": int(bool(r.get("is_duplicate"))),
                         "duplicate_type": r.get("duplicate_type") or "", "detection_source": r.get("detection_source"),
                         "signal_type": r.get("signal_type"), "mz_source": r.get("mz_source") or "", "ms2_spectrum": ""})
    return pd.DataFrame(rows)


def load_fold_map(run: Path) -> dict:
    p = run / "alignment_merge_map.json"
    if not p.exists():
        return {}
    import json
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("fold_reason_of", {})
    except Exception:
        return {}


def consistency_by_support(retained: pd.DataFrame) -> pd.DataFrame:
    """NL consensus vs window centroid (window_delta_mda) by n_support, the
    same summary nl_window_consistency.py (S2.1c) writes."""
    df = retained[(retained["mz_source"] == "nl_consensus") & retained["window_delta_mda"].notna()].copy()
    df["support_bin"] = df["nl_n_support"].map(lambda v: ">=4" if v >= 4 else str(int(v)))
    rows = []
    for lvl in SUPPORT_LEVELS:
        a = np.abs(df.loc[df["support_bin"] == lvl, "window_delta_mda"].to_numpy(float))
        if not a.size:
            continue
        rows.append({"support_bin": lvl, "n": int(a.size), "median_abs_mda": float(np.median(a)),
                     "p90_abs_mda": float(np.percentile(a, 90)),
                     "frac_within_10_mda": float((a <= 10).mean()), "frac_within_20_mda": float((a <= 20).mean()),
                     "frac_within_50_mda": float((a <= 50).mean())})
    return pd.DataFrame(rows)


def load_reference(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for c in ("RT (min)", "Precursor m/z", "Height", "S/N", "Isotope"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def seg_of(row):
        s = row.get("ID")
        if isinstance(s, str):
            m = SEG_RE.search(s)
            if m:
                return m.group(1)
        return seg_from_mz(row.get("Precursor m/z"))

    df["segment_name"] = df.apply(seg_of, axis=1)
    return df


def spec_similarity(a, b, prec_a=0.0, prec_b=0.0, mz_tol=0.02):
    pa = list(zip(a[0].tolist(), a[1].tolist())); pb = list(zip(b[0].tolist(), b[1].tolist()))
    if not pa or not pb:
        return 0.0
    score, _ = composite_similarity(pa, pb, mz_tol, precursor_query=prec_a,
                                    precursor_ref=prec_b, ms1_tolerance=0.01)
    return float(score)


# --------------------------------------------------------------------------- #
# Matching (unchanged from the submitted script)
# --------------------------------------------------------------------------- #
def match(asfam: pd.DataFrame, ref: pd.DataFrame, mz_tol=0.01, rt_tol=0.1) -> pd.DataFrame:
    rows = []
    by_seg = {seg: g.reset_index(drop=True) for seg, g in asfam.groupby("segment_name")}
    for _, ms in ref.iterrows():
        seg = ms["segment_name"]
        mmz, mrt = ms.get("Precursor m/z"), ms.get("RT (min)")
        rec = {"ms_segment": seg, "ms_peak_id": ms.get("Peak ID"), "ms_mz": mmz, "ms_rt": mrt,
               "ms_height": ms.get("Height"), "ms_sn": ms.get("S/N"), "ms_msms": ms.get("MSMS spectrum")}
        if pd.isna(mmz) or pd.isna(mrt) or seg not in by_seg:
            rec["match_class"] = "Missed"; rows.append(rec); continue
        g = by_seg[seg]
        dmz = np.abs(g["precursor_mz"].to_numpy() - mmz)
        drt = np.abs(g["rt_apex"].to_numpy() - mrt)
        ok = (dmz <= mz_tol) & (drt <= rt_tol)
        if not ok.any():
            rec["match_class"] = "Missed"; rows.append(rec); continue
        idx = np.where(ok)[0]
        isdup = g["is_duplicate"].to_numpy()[idx]
        cost = dmz[idx] / mz_tol + drt[idx] / rt_tol + isdup * 5.0
        cand = g.iloc[idx[int(np.argmin(cost))]]
        cls = "Primary" if int(cand["is_duplicate"]) == 0 else f"Linked-{cand.get('duplicate_type') or 'unknown'}"
        rec.update({"match_class": cls, "asfam_feature_id": cand["feature_id"], "asfam_mz": cand["precursor_mz"],
                    "asfam_rt": cand["rt_apex"], "asfam_branch": cand["detection_source"],
                    "asfam_signal": cand["signal_type"], "asfam_mz_source": cand["mz_source"],
                    "asfam_ms2": cand["ms2_spectrum"]})
        rows.append(rec)
    return pd.DataFrame(rows)


def pair_similarities(m: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, r in m[m["match_class"] != "Missed"].iterrows():
        a, b = parse_spectrum(r.get("ms_msms")), parse_spectrum(r.get("asfam_ms2"))
        if len(a[0]) == 0 or len(b[0]) == 0:
            continue
        out.append({"ms_peak_id": r.get("ms_peak_id"), "ms_mz": r.get("ms_mz"), "ms_rt": r.get("ms_rt"),
                    "asfam_feature_id": r.get("asfam_feature_id"), "asfam_mz": r.get("asfam_mz"),
                    "asfam_rt": r.get("asfam_rt"), "match_class": r.get("match_class"),
                    "composite_similarity": spec_similarity(a, b, r.get("ms_mz") or 0, r.get("asfam_mz") or 0)})
    return pd.DataFrame(out)


def mirror_pair(m: pd.DataFrame):
    rows = []
    for _, r in m[m["match_class"] != "Missed"].iterrows():
        a, b = parse_spectrum(r.get("ms_msms")), parse_spectrum(r.get("asfam_ms2"))
        if len(a[0]) < 8 or len(b[0]) < 8:
            continue
        rows.append((spec_similarity(a, b, r["ms_mz"], r["asfam_mz"]), r, a, b))
    if not rows:
        return None
    target = float(np.median([x[0] for x in rows]))
    s, r, a, b = min(rows, key=lambda x: abs(x[0] - target))
    return a, b, r, s


def save_mirror_source(mirror, path: Path) -> None:
    rows = []
    if mirror is not None:
        (mz_a, in_a), (mz_b, in_b), record, similarity = mirror
        for source, mz, intensity in (("MS-DIAL", mz_a, in_a), ("ASFAMProcessor", mz_b, in_b)):
            relative = intensity / intensity.max() * 100 if len(intensity) else intensity
            for product_mz, rel in zip(mz, relative):
                rows.append({"source": source, "precursor_mz": record["ms_mz"], "product_mz": product_mz,
                             "relative_intensity": rel, "composite_similarity": similarity})
    pd.DataFrame(rows).to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# Panels a–d
# --------------------------------------------------------------------------- #
def panel_a(ax, m, title=None):
    counts = m["match_class"].value_counts()
    classes = [c for c in CLASS_ORDER if counts.get(c, 0) > 0]
    vals = [int(counts[c]) for c in classes]; total = sum(vals)
    x = np.arange(len(classes))
    bars = ax.bar(x, vals, color=[CLASS_F[c] for c in classes], edgecolor=[CLASS_S[c] for c in classes], linewidth=LW)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + total * 0.015, f"{v}\n{v / total * 100:.1f}%",
                ha="center", va="bottom", fontsize=7, color=TEXT, linespacing=0.9)
    ax.set_xticks(x); ax.set_xticklabels([c.replace("Linked-", "L-") for c in classes], rotation=30, ha="right")
    ax.set_ylabel("MS-DIAL feature count"); ax.set_ylim(0, max(vals) * 1.30)
    ax.set_title(title or f"Recovery (n = {total} MS-DIAL features)")
    despine(ax)


def panel_b(ax, m):
    mm = m[m["match_class"] != "Missed"].copy()
    mm["branch"] = mm["asfam_branch"].map({"ms1_driven": "MS1-driven", "ms2_driven": "MS2-driven"}).fillna("other")
    classes = [c for c in CLASS_ORDER if c != "Missed" and (mm["match_class"] == c).sum() > 0]
    x = np.arange(len(classes)); w = 0.4
    for i, (br, col) in enumerate((("MS1-driven", WT_BLUE), ("MS2-driven", SKY_BLUE))):
        vals = [int(((mm["branch"] == br) & (mm["match_class"] == c)).sum()) for c in classes]
        ax.bar(x + (i - 0.5) * w, vals, width=w, color=light(col), edgecolor=col, linewidth=LW,
               label=f"{br} branch")
    ax.set_xticks(x); ax.set_xticklabels([c.replace("Linked-", "L-") for c in classes], rotation=30, ha="right")
    ax.set_ylabel("Matched count"); ax.set_title("Recovery by detection branch")
    ax.legend(loc="upper right", handlelength=1.0, handletextpad=0.5, labelspacing=0.3)
    despine(ax)


def panel_c(ax, sims):
    counts, _, _ = ax.hist(sims, bins=np.linspace(0, 2, 21), color=light(WT_BLUE), edgecolor=WT_BLUE, linewidth=LW)
    med = float(np.median(sims))
    ax.axvline(med, color=VERMILION, linestyle="--", linewidth=LW)
    ax.text(0.03, 0.97, f"median {med:.3f}", color=VERMILION, ha="left", va="top", transform=ax.transAxes)
    ax.set_xlim(0, 2); ax.set_xlabel("Composite similarity (0–2)"); ax.set_ylabel("Matched-pair count")
    ax.set_title(f"MS2 consistency (n = {len(sims)})")
    despine(ax)


def panel_d(ax, mirror):
    if mirror is None:
        ax.set_title("No mirror pair"); return
    (mz_a, in_a), (mz_b, in_b), row, sim = mirror
    in_a = in_a / in_a.max() * 100 if len(in_a) else in_a
    in_b = in_b / in_b.max() * 100 if len(in_b) else in_b
    ax.vlines(mz_a, 0, in_a, colors=DDA_ORANGE, linewidth=LW)
    ax.vlines(mz_b, 0, -in_b, colors=WT_BLUE, linewidth=LW)
    ax.axhline(0, color=GRAY, linewidth=LW)
    ax.set_ylim(-135, 135); ax.set_xlabel("m/z"); ax.set_ylabel("Relative intensity (%)")
    ax.set_yticks([-100, -50, 0, 50, 100]); ax.set_yticklabels(["100", "50", "0", "50", "100"])
    ax.set_title(f"Mirror plot — m/z {row['ms_mz']:.4f}, sim {sim:.3f}")
    despine(ax)
    ax.text(0.02, 1.0, "MS-DIAL", color=DDA_ORANGE, transform=ax.transAxes, va="top", ha="left")
    ax.text(0.02, 0.0, "ASFAMProcessor", color=WT_BLUE, transform=ax.transAxes, va="bottom", ha="left")


# --------------------------------------------------------------------------- #
# Panels e/f: NL-consensus self-validation by n_support
# --------------------------------------------------------------------------- #
def panel_e(ax, inf):
    for lvl in SUPPORT_LEVELS:
        sub = inf[inf["support_bin"] == lvl]
        if not len(sub):
            continue
        x = np.sort(sub["abs_error_ppm"].to_numpy()); y = np.arange(1, len(x) + 1) / len(x) * 100
        ax.step(x, y, where="post", color=SUPPORT_S[lvl], linewidth=LW,
                label=f"{SUPPORT_LABEL[lvl]} (n = {len(sub)})")
    ax.set_xscale("log"); ax.set_xlim(0.1, 1e4); ax.set_ylim(0, 100)
    ax.axvline(20, color=GRAY, linestyle="--", linewidth=LW)
    ax.text(24, 4, "20 ppm", color=GRAY, ha="left")
    ax.set_xlabel("|inferred − true m/z| (ppm)"); ax.set_ylabel("Cumulative %")
    ax.set_title(f"NL-consensus self-validation (n = {len(inf)})")
    ax.legend(loc="upper left", handlelength=1.0, handletextpad=0.4, labelspacing=0.25, fontsize=6.5,
              borderaxespad=0.2)
    despine(ax)


def panel_f(ax, inf):
    thr = [5, 10, 20]; w = 0.2; x = np.arange(len(thr))
    for i, lvl in enumerate(SUPPORT_LEVELS):
        sub = inf[inf["support_bin"] == lvl]
        vals = [0] * len(thr) if not len(sub) else [(sub["abs_error_ppm"] <= t).mean() * 100 for t in thr]
        bars = ax.bar(x + (i - 1.5) * w, vals, width=w, color=SUPPORT_F[lvl], edgecolor=SUPPORT_S[lvl],
                      linewidth=LW, label=f"{SUPPORT_LABEL[lvl]} (n = {len(sub)})")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.0, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=6.5, color=TEXT)
    ax.set_xticks(x); ax.set_xticklabels([f"≤{t} ppm" for t in thr])
    # headroom above the tallest bar label so the legend never covers a value
    ax.set_ylabel("Within tolerance (%)"); ax.set_ylim(0, 140)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("Precision by number of votes")
    ax.legend(loc="upper left", ncol=2, handlelength=0.9, handletextpad=0.4, columnspacing=0.8,
              labelspacing=0.25, fontsize=6.5, borderaxespad=0.2)
    despine(ax)


# --------------------------------------------------------------------------- #
# Supplementary NL figure
# --------------------------------------------------------------------------- #
def panel_consistency(ax, cons: pd.DataFrame):
    """NL consensus vs isolation-window centroid on real MS2-only features."""
    thr = [("frac_within_10_mda", "≤10 mDa", 0.4), ("frac_within_20_mda", "≤20 mDa", 0.7),
           ("frac_within_50_mda", "≤50 mDa", 1.0)]
    cons = cons.copy(); cons["support_bin"] = cons["support_bin"].astype(str)
    levels = [l for l in SUPPORT_LEVELS if l in set(cons["support_bin"])]
    x = np.arange(len(levels)); w = 0.26
    for i, (col, lab, f) in enumerate(thr):
        vals = [float(cons.loc[cons["support_bin"] == l, col].iloc[0]) * 100 for l in levels]
        c = tint(MS2ONLY_PURPLE, f)
        bars = ax.bar(x + (i - 1) * w, vals, width=w, color=light(c), edgecolor=c, linewidth=LW, label=lab)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.0, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=6.5, color=TEXT)
    ns = [int(cons.loc[cons["support_bin"] == l, "n"].iloc[0]) for l in levels]
    ax.set_xticks(x); ax.set_xticklabels([f"{l.replace('>=', '≥')}\nn={n}" for l, n in zip(levels, ns)],
                                         fontsize=7, linespacing=0.95)
    ax.set_xlabel("Votes (n_support)")
    ax.set_ylabel("NL within tolerance of centroid (%)"); ax.set_ylim(0, 100)
    ax.set_title("NL vs centroid (MS2-only)")
    ax.legend(loc="upper left", handlelength=0.9, handletextpad=0.4, labelspacing=0.25, fontsize=7)
    despine(ax)


OUTCOME_STYLE = [("same_compound", "same compound", WT_BLUE), ("same_formula_other_compound", "same formula, other compound", SKY_BLUE),
                 ("other_formula", "other formula", VERMILION), ("no_high_score_hit", "no high-score hit", GRAY)]
ARM_LABEL = {"truth_0.5": "true\n±0.5", "nl_0.05": "NL\n±0.05", "nl_0.1": "NL\n±0.1", "nl_0.5": "NL\n±0.5",
             "centroid_0.05": "centr.\n±0.05", "basepeak_0.01": "base\n±0.01"}


def panel_outcome(ax, summ: pd.DataFrame):
    arms = [a for a in ("truth_0.5", "nl_0.05", "nl_0.1", "nl_0.5", "centroid_0.05", "basepeak_0.01")
            if a in set(summ["arm"])]
    s = summ[(summ["stratum_type"] == "all")].set_index("arm")
    x = np.arange(len(arms))
    bottom = np.zeros(len(arms))
    for key, lab, col in OUTCOME_STYLE:
        vals = np.array([float(s.loc[a, f"frac_{key}"]) * 100 for a in arms])
        ax.bar(x, vals, bottom=bottom, width=0.65, color=light(col) if key != "no_high_score_hit" else GRAY_LIGHT,
               edgecolor=col, linewidth=LW, label=lab)
        for xi, v, b in zip(x, vals, bottom):
            if v >= 6:
                ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color=TEXT)
        bottom += vals
    n = int(s.loc[arms[0], "n_truth_hit"])
    ax.set_xticks(x); ax.set_xticklabels([ARM_LABEL.get(a, a) for a in arms], fontsize=7, linespacing=0.95)
    ax.set_xlabel("Precursor estimate, window (Da)")
    ax.set_ylabel("Features (%)"); ax.set_ylim(0, 100)
    ax.set_title(f"Top-1 vs truth (n = {n})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=6.5, handlelength=0.9,
              handletextpad=0.4, columnspacing=0.8, labelspacing=0.25)
    despine(ax)


def panel_outcome_support(ax, summ: pd.DataFrame):
    arms = [("nl_0.05", "NL ±0.05 Da", 0.4), ("nl_0.1", "NL ±0.1 Da", 0.7), ("nl_0.5", "NL ±0.5 Da", 1.0)]
    s = summ[summ["stratum_type"] == "n_support"]
    levels = [l for l in SUPPORT_LEVELS if l in set(s["stratum"].astype(str))]
    x = np.arange(len(levels)); w = 0.26
    for i, (arm, lab, f) in enumerate(arms):
        vals = [float(s[(s["arm"] == arm) & (s["stratum"].astype(str) == l)]["frac_same_compound"].iloc[0]) * 100
                for l in levels]
        c = tint(MS2ONLY_PURPLE, f)
        bars = ax.bar(x + (i - 1) * w, vals, width=w, color=light(c), edgecolor=c, linewidth=LW, label=lab)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.0, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=6.5, color=TEXT)
    ns = [int(s[(s["arm"] == arms[0][0]) & (s["stratum"].astype(str) == l)]["n_truth_hit"].iloc[0]) for l in levels]
    ax.set_xticks(x); ax.set_xticklabels([f"{l.replace('>=', '≥')}\nn={n}" for l, n in zip(levels, ns)],
                                         fontsize=7, linespacing=0.95)
    ax.set_xlabel("Votes (n_support)")
    ax.set_ylabel("Same top-1 compound as truth (%)"); ax.set_ylim(0, 130)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("Correct top-1 by NL window")
    ax.legend(loc="upper left", ncol=3, handlelength=0.9, handletextpad=0.4, columnspacing=0.8,
              labelspacing=0.25, fontsize=7)
    despine(ax)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_main(m, inf, sims, mirror, figdir: Path):
    setup_style()
    fig = plt.figure(figsize=(7.5, 5.0), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.10, wspace=0.12, hspace=0.24)
    gs = fig.add_gridspec(2, 3)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    panel_a(axes[0], m); panel_b(axes[1], m); panel_c(axes[2], sims)
    panel_d(axes[3], mirror); panel_e(axes[4], inf); panel_f(axes[5], inf)
    fig.canvas.draw()
    for ax, lab in zip(axes, "abcdef"):
        add_label(fig, ax, lab)
    save_all(fig, figdir / "fig4.svg")
    plt.close(fig)


def make_supp_nl(cons, summ, figdir: Path) -> bool:
    if cons is None and summ is None:
        return False
    setup_style()
    n_panels = (cons is not None) + (2 if summ is not None else 0)
    fig = plt.figure(figsize=(7.5 if n_panels == 3 else 2.6 * n_panels, 3.0), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.10, wspace=0.14)
    gs = fig.add_gridspec(1, n_panels)
    axes = []
    i = 0
    if cons is not None:
        ax = fig.add_subplot(gs[0, i]); panel_consistency(ax, cons); axes.append(ax); i += 1
    if summ is not None:
        ax = fig.add_subplot(gs[0, i]); panel_outcome(ax, summ); axes.append(ax); i += 1
        ax = fig.add_subplot(gs[0, i]); panel_outcome_support(ax, summ); axes.append(ax); i += 1
    fig.canvas.draw()
    for ax, lab in zip(axes, "abc"):
        add_label(fig, ax, lab)
    save_all(fig, figdir / "fig4_supp_nl.svg")
    plt.close(fig)
    return True


def make_supp_recovery(m_len, figdir: Path, analysis: Path):
    setup_style()
    fig = plt.figure(figsize=(7.5, 3.0), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.10, wspace=0.18)
    gs = fig.add_gridspec(1, 2)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
    panel_a(axA, m_len, title=f"Recovery — lenient ref (n = {len(m_len)})")
    bands = [(0, 1e3, "<10³", "<1e3"), (1e3, 1e4, "10³–10⁴", "1e3-1e4"), (1e4, 1e5, "10⁴–10⁵", "1e4-1e5"),
             (1e5, 1e6, "10⁵–10⁶", "1e5-1e6"), (1e6, np.inf, ">10⁶", ">1e6")]
    h = pd.to_numeric(m_len["ms_height"], errors="coerce").fillna(0).to_numpy()
    rec = (m_len["match_class"] != "Missed").to_numpy()
    labels, rates, ns, table = [], [], [], []
    for lo, hi, lab, alab in bands:
        sel = (h >= lo) & (h < hi); n = int(sel.sum())
        if n == 0:
            continue
        r = float(rec[sel].mean() * 100)
        labels.append(lab); rates.append(r); ns.append(n)
        table.append({"height_band": alab, "n_ref": n, "n_recovered": int(rec[sel].sum()), "recall_pct": round(r, 1)})
    x = np.arange(len(rates))
    bars = axB.bar(x, rates, color=light(WT_BLUE), edgecolor=WT_BLUE, linewidth=LW)
    for bar, r, n in zip(bars, rates, ns):
        axB.text(bar.get_x() + bar.get_width() / 2, r + 1.5, f"{r:.0f}\nn={n}", ha="center", va="bottom",
                 fontsize=7, color=TEXT, linespacing=0.9)
    axB.set_xticks(x); axB.set_xticklabels(labels, rotation=30, ha="right")
    axB.set_ylabel("Recall (%)"); axB.set_ylim(0, 118); axB.set_xlabel("MS-DIAL peak height")
    axB.set_title("Height-stratified recall"); despine(axB)
    fig.canvas.draw()
    for ax, lab in zip([axA, axB], "ab"):
        add_label(fig, ax, lab)
    save_all(fig, figdir / "fig4_supp_recovery.svg")
    plt.close(fig)
    pd.DataFrame(table).to_csv(analysis / "height_stratified_recall.csv", index=False)
    return table


# --------------------------------------------------------------------------- #
LEGEND = """Figure 4 | Validation of ASFAMProcessor against MS-DIAL on the full-range rep-3 data set (m/z 75–1004, 31 acquisition segments, one replicate, positive mode; ASFAMProcessor v{ver}, lib/lcms/pos.msp). All panels are counts or fractions on this one sample (counts, not means); panels a, b and f report single counts.
(a) Recovery of the MS-DIAL reference features (strict: height >= 10,000, S/N >= 5, >= 4 fragments; n = {n_strict}). A reference feature is recovered when an ASFAMProcessor row lies within 0.01 Da and 0.1 min; "Primary" = the matching row is retained (is_duplicate = False), "L-isotope / L-adduct / L-isf / L-spectral" = the matching row is a retained row's marked isotope, adduct, in-source fragment or spectral duplicate. Recovered {n_recovered}/{n_strict} ({pct_recovered}); Primary {n_primary}. Before alignment, {n_rec_cand}/{n_strict} ({pct_rec_cand}) reference features have a matching candidate feature; {n_lost_fold} of the reference features missed in the aligned table were matched by a candidate that the stage-7 joiner folded into another row of the same compound (repeated or split peaks of one compound judged identical by spectrum; analysis/rep3_full/fold_audit_summary.csv), so they are represented by a row at another retention time rather than lost.
(b) The recovered reference features by the ASFAMProcessor branch that detected the matching row: MS2-driven (stage 1, product-ion chromatograms) or MS1-driven (stage 1b, complementary MS1 peaks).
(c) Composite spectral similarity (production metric, 0–2) between each matched pair's MS2 spectra (n = {n_sims} pairs with spectra on both sides); median {median_sim}.
(d) Mirror plot of the pair whose similarity is closest to the median (MS-DIAL above, ASFAMProcessor below; precursor m/z {mirror_mz}, similarity {mirror_sim}).
(e, f) Self-validation of the neutral-loss (NL) consensus precursor inference: for {n_truth} MS2-driven features whose precursor was assigned from a detected MS1 peak (truth = ms1_precursor_mz), precursor >= 200 Da and >= 5 fragments, the precursor was withheld, fragments within 0.5 Da of it removed, and the production predictor (asfam.pipeline.stage2b_inference._nl_consensus_predict) re-run on the remaining fragments; it returned a value for {n_pred} ({pct_pred}), stratified by the number of distinct fragments voting for the consensus (n_support; mz_confidence low = 1, medium = 2, high >= 3). (e) Cumulative distribution of |inferred − true| in ppm; dashed line 20 ppm. (f) Fraction within 5 / 10 / 20 ppm. Median |error| {med_1} ppm (1 vote, n = {n_1}), {med_2} ppm (2 votes, n = {n_2}), {med_3} ppm (3 votes, n = {n_3}), {med_4} ppm (>= 4 votes, n = {n_4}); within 20 ppm {w20_1} / {w20_2} / {w20_3} / {w20_4}. Features with exactly 3–4 fragments (the production floor) are in analysis/rep3_full/inference_self_validation_summary.csv.
Source data: analysis/rep3_full/match_table_strict.csv, similarity_pairs_strict.csv, mirror_pair_strict.csv, inference_self_validation.csv.

Supplementary Figure (fig4_supp_nl) | The NL-consensus precursor on real MS2-only features and its consequence for annotation (rep-3, v{ver}; counts, not means).
(a) For the {n_cons} retained MS2-only rows whose precursor is the NL consensus, agreement between the consensus and the isolation-window centroid stored for the same feature (ms1_window_mz; a mixture centroid of everything co-isolated in the 1-Da window), as the fraction within 10 / 20 / 50 mDa, by n_support. Data: analysis/rep3_full/consistency_by_support.csv (S2.1c).
(b) Annotation outcome for the self-validation features of Fig. 4e whose true-precursor match (the measured MS1 precursor ms1_precursor_mz: the isolation-window centroid at the MS1 apex up to v1.4.261020, the base-peak ion from v1.4.261020.2; ±0.01 Da) is a high-score hit (Total score >= 1.4 and >= 3 matched peaks; n = {n_outcome}): top-1 hit obtained with the true precursor at ±0.5 Da, with the NL-inferred precursor at ±0.05 / 0.1 / 0.5 Da, with the window centroid at ±0.05 Da and with the quantitation base-peak ion at ±0.01 Da, classed as the same compound (names equal after stripping "Putative: " and case; the lean library carries no InChIKey), a different compound of the same formula, a different formula, or no high-score hit.
(c) The same-compound fraction by n_support for the three NL windows. {rec_line}
Data: analysis/rep3_full/annotation_outcome_summary.csv, annotation_outcome_windows.csv (S2.1e).

Supplementary Figure (fig4_supp_recovery) | (a) Recovery against the lenient MS-DIAL reference (height >= 1,000, S/N >= 5, >= 2 fragments; n = {n_lenient}; {pct_recovered_len} recovered) and (b) recall by MS-DIAL peak height (counts, not means). Data: analysis/rep3_full/match_table_lenient.csv, height_stratified_recall.csv.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default=str(DEFAULT_RUN))
    ap.add_argument("--ref-dir", default=str(DEFAULT_REF))
    ap.add_argument("--analysis", default=str(DEFAULT_ANALYSIS))
    ap.add_argument("--figdir", default=str(DEFAULT_FIGDIR))
    args = ap.parse_args()
    run, ref_dir, analysis, figdir = Path(args.run), Path(args.ref_dir), Path(args.analysis), Path(args.figdir)
    analysis.mkdir(parents=True, exist_ok=True); figdir.mkdir(parents=True, exist_ok=True)

    print("[1/6] cached MS-DIAL references")
    ref_strict = load_reference(ref_dir / "msdial_ref_strict.csv")
    ref_lenient = load_reference(ref_dir / "msdial_ref_lenient.csv")
    print(f"  strict {len(ref_strict)}  lenient {len(ref_lenient)}")

    print("[2/6] ASFAMProcessor features.csv")
    asfam = load_asfam(run / "features.csv")
    version = ""
    for line in (run / "features.csv").open(encoding="utf-8", errors="replace"):
        if line.startswith("# version="):
            version = line.split("=", 1)[1].strip(); break
    n_active = int((asfam["is_duplicate"] == 0).sum())
    print(f"  rows {len(asfam)}  retained {n_active}  version {version}")

    print("[3/6] matching")
    m_strict = match(asfam, ref_strict); m_lenient = match(asfam, ref_lenient)
    m_strict.to_csv(analysis / "match_table_strict.csv", index=False)
    m_lenient.to_csv(analysis / "match_table_lenient.csv", index=False)
    summ = m_strict["match_class"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    summ.to_csv(analysis / "recovery_summary_strict.csv", header=["count"])
    total = int(summ.sum()); recovered = total - int(summ["Missed"])
    print(summ.to_string()); print(f"  recovered {recovered}/{total} = {recovered / total * 100:.1f}%")
    # candidate-level recovery (before alignment): reference features missed in the
    # aligned table but matched by a candidate were lost by the stage-7 fold
    cands = load_candidates(run)
    m_cand = match(cands, ref_strict)
    m_cand.to_csv(analysis / "match_table_strict_candidates.csv", index=False)
    summ_cand = m_cand["match_class"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    summ_cand.to_csv(analysis / "recovery_summary_strict_candidates.csv", header=["count"])
    rec_cand = total - int(summ_cand["Missed"])
    fold_reason = load_fold_map(run)
    missed_aligned = m_strict["match_class"] == "Missed"
    cand_hit = m_cand["match_class"] != "Missed"
    lost_by_fold = int((missed_aligned & cand_hit & m_cand["asfam_feature_id"].isin(fold_reason)).sum())
    lost_other = int((missed_aligned & cand_hit).sum()) - lost_by_fold
    print(f"  candidate-level recovery {rec_cand}/{total} = {rec_cand / total * 100:.1f}%; "
          f"missed in aligned table but matched by a candidate: {lost_by_fold} folded at alignment, {lost_other} other")

    print("[4/6] NL-consensus self-validation (S2.1b table)")
    sv_path = analysis / "inference_self_validation.csv"
    if not sv_path.exists():
        import nl_self_validation  # noqa: WPS433 (sibling script)
        nl_self_validation.main()
    inf = pd.read_csv(sv_path, low_memory=False)
    inf["support_bin"] = inf["support_bin"].astype(str)
    scen = pd.read_csv(analysis / "inference_self_validation_summary.csv")
    row_all = scen[(scen["scenario"] == "masked_ge5") & (scen["n_support"] == "all")].iloc[0]

    print("[5/6] similarities, mirror, main figure")
    sim_tbl = pair_similarities(m_strict); sim_tbl.to_csv(analysis / "similarity_pairs_strict.csv", index=False)
    sims = sim_tbl["composite_similarity"].to_numpy()
    print(f"  pairs {len(sims)} median {np.median(sims):.3f}")
    mirror = mirror_pair(m_strict); save_mirror_source(mirror, analysis / "mirror_pair_strict.csv")
    make_main(m_strict, inf, sims, mirror, figdir)

    print("[6/6] supplementary figures")
    table = make_supp_recovery(m_lenient, figdir, analysis)
    cons_path = analysis / "consistency_by_support.csv"
    if cons_path.exists():
        cons = pd.read_csv(cons_path)
    else:  # S2.1c table not produced for this run yet: derive it from features.csv
        cons = consistency_by_support(asfam[asfam["is_duplicate"] == 0])
        cons.to_csv(analysis / "fig4_consistency_by_support.csv", index=False)
        print("  consistency_by_support.csv absent -> computed from features.csv (fig4_consistency_by_support.csv)")
    oc_path = analysis / "annotation_outcome_summary.csv"
    oc = pd.read_csv(oc_path) if oc_path.exists() else None
    made = make_supp_nl(cons, oc, figdir)
    if not made:
        print("  fig4_supp_nl skipped (no consistency / outcome tables yet)")
    rec_line = "(annotation outcome not available yet)"
    n_outcome = 0
    win_path = analysis / "annotation_outcome_windows.csv"
    if oc is not None:
        n_outcome = int(oc[oc["stratum_type"] == "all"]["n_truth_hit"].iloc[0])
    if win_path.exists():
        # "recommended" marks the window adopted in the build (annotation_outcome.py), not the
        # window with the largest net count; the legend states both facts.
        win = pd.read_csv(win_path).sort_values("window_da").reset_index(drop=True)
        best = win[win["recommended"]].iloc[0]
        others = win[~win["recommended"]]
        nets = win["net_same_minus_other_formula_minus_unconfirmed"]
        unconf = int(best["n_unconfirmed_hits_without_truth"])
        fewest = unconf <= int(others["n_unconfirmed_hits_without_truth"].min()) if len(others) else True
        rec_line = (f"The {best['window_da']:g} Da window adopted in the build retains the same top-1 compound for "
                    f"{int(best['n_same_compound'])}/{int(best['n_truth_hit'])} "
                    f"({best['n_same_compound'] / best['n_truth_hit'] * 100:.0f}%) and adds "
                    + ("the fewest unconfirmable hits" if fewest else "unconfirmable hits")
                    + f" ({unconf}, against " + " and ".join(f"{int(r['n_unconfirmed_hits_without_truth'])}" for _, r in others.iterrows())
                    + " for " + " and ".join(f"{r['window_da']:g} Da" for _, r in others.iterrows())
                    + f"); the net counts (same compound minus other formula minus unconfirmable) of the three windows "
                    f"differ by at most {int(nets.max() - nets.min())} ("
                    + " / ".join(f"{r['window_da']:g} Da" for _, r in win.iterrows()) + ": "
                    + ", ".join(f"{int(r['n_same_compound'])} same compound / {int(r['n_other_formula'])} other formula / "
                                f"{int(r['n_unconfirmed_hits_without_truth'])} unconfirmable" for _, r in win.iterrows()) + ").")

    def stat(lvl, col):
        r = scen[(scen["scenario"] == "masked_ge5") & (scen["n_support"] == lvl)]
        return r.iloc[0][col] if len(r) else np.nan

    n_lenient = len(m_lenient); rec_len = int((m_lenient["match_class"] != "Missed").sum())
    legend = LEGEND.format(
        ver=version, n_strict=total, n_recovered=recovered, pct_recovered=f"{recovered / total * 100:.1f}%",
        n_primary=int(summ["Primary"]), n_sims=len(sims), median_sim=f"{np.median(sims):.3f}",
        n_rec_cand=rec_cand, pct_rec_cand=f"{rec_cand / total * 100:.1f}%", n_lost_fold=lost_by_fold,
        mirror_mz=f"{mirror[2]['ms_mz']:.4f}" if mirror else "n/a", mirror_sim=f"{mirror[3]:.3f}" if mirror else "n/a",
        n_truth=int(row_all["n_eligible"]), n_pred=int(row_all["n_predicted"]),
        pct_pred=f"{row_all['n_predicted'] / row_all['n_eligible'] * 100:.1f}%",
        med_1=f"{stat('1', 'median_abs_ppm'):.1f}", med_2=f"{stat('2', 'median_abs_ppm'):.1f}",
        med_3=f"{stat('3', 'median_abs_ppm'):.1f}", med_4=f"{stat('>=4', 'median_abs_ppm'):.1f}",
        n_1=int(stat('1', 'n_predicted')), n_2=int(stat('2', 'n_predicted')), n_3=int(stat('3', 'n_predicted')),
        n_4=int(stat('>=4', 'n_predicted')),
        w20_1=f"{stat('1', 'frac_within_20_ppm') * 100:.1f}%", w20_2=f"{stat('2', 'frac_within_20_ppm') * 100:.1f}%",
        w20_3=f"{stat('3', 'frac_within_20_ppm') * 100:.1f}%", w20_4=f"{stat('>=4', 'frac_within_20_ppm') * 100:.1f}%",
        n_cons=int(cons["n"].sum()) if cons is not None else 0, n_outcome=n_outcome, rec_line=rec_line,
        n_lenient=n_lenient, pct_recovered_len=f"{rec_len / n_lenient * 100:.1f}%")
    (figdir / "legend_fig4.txt").write_text(legend, encoding="utf-8")

    numbers = {"fig4_version": version, "fig4_ref_strict_n": total, "fig4_recovered_strict": recovered,
               "fig4_recovered_strict_pct": recovered / total * 100, "fig4_primary_strict": int(summ["Primary"]),
               "fig4_similarity_pairs": len(sims), "fig4_similarity_median": float(np.median(sims)),
               "fig4_mirror_mz": float(mirror[2]["ms_mz"]) if mirror else None,
               "fig4_mirror_similarity": float(mirror[3]) if mirror else None,
               "fig4_ref_lenient_n": n_lenient, "fig4_recovered_lenient": rec_len,
               "fig4_recovered_lenient_pct": rec_len / n_lenient * 100,
               "fig4_rows_total": len(asfam), "fig4_rows_retained": n_active,
               "fig4_recovered_strict_candidates": rec_cand,
               "fig4_recovered_strict_candidates_pct": rec_cand / total * 100,
               "fig4_missed_aligned_lost_by_fold": lost_by_fold,
               "fig4_missed_aligned_candidate_matched_other": lost_other,
               "fig4_submitted_recovered_strict": {"value": 657, "note": "submitted Fig. 4a (v0.9.260705.1, compare_0706): 657/722 = 91.0%"}}
    for c in CLASS_ORDER:
        numbers[f"fig4_strict_{c.lower().replace('-', '_')}"] = int(summ[c])
    for row in table:
        numbers[f"fig4_recall_height_{row['height_band']}"] = row["recall_pct"]
        numbers[f"fig4_recall_height_{row['height_band']}_n"] = row["n_ref"]
    update_numbers(analysis / "numbers_fig4.json", numbers, source="scripts/gen_fig4.py -> figures/fig4/")
    print("DONE:", figdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
