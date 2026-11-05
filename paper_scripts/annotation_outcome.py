"""S2.1e + S2.3 library part: what a precursor estimate does to the annotation.

Everything here needs the full positive-mode library (lib/lcms/pos.msp, ~6 GB
in memory), so it is loaded ONCE, used for three analyses, and released.

A. S2.1e - annotation outcome under three precursor estimates and three NL windows.
   For every feature of the NL self-validation (nl_self_validation.py:
   MS2-driven features whose precursor is a detected MS1 peak, >= 200 Da,
   >= 5 fragments, precursor masked, prediction returned) the production
   matcher ``metabo_core.annotation.match_feature_topn`` is run with
     truth_0.01      the strict MS1 peak m/z (ms1_precursor_mz) at +/-0.01 Da  (truth arm)
     truth_0.5       the same precursor at +/-0.5 Da (what a wide window admits even
                     when the precursor is right)
     nl_0.05/0.1/0.5 the neutral-loss consensus prediction at +/-0.05 / 0.1 / 0.5 Da
     centroid_0.05   the isolation-window centroid (centroid_precision_rows.csv,
                     column window_centroid_mz) at +/-0.05 Da
     basepeak_0.01   the base-peak ion the feature was quantified on (ms1_quant_mz) at 0.01
   and the top-1 hit of every arm is compared with the truth arm's top-1, on
   features whose truth arm has a high-score hit (Total score >= 1.4 and >= 3
   matched peaks): same compound (names equal after stripping "Putative: " and
   case; the lean library keeps name / formula / precursor_mz / adduct only, no
   InChIKey), different compound of the same formula, different formula, or no
   high-score hit. Stratified by n_support, truth Total score band, m/z band and
   MS1 height band. Features whose truth arm has NO high-score hit are counted
   separately: a high-score hit there is a hit the truth could not confirm.

B. S2.3 - why the MS1-driven-only run annotates more features than the full run.
   B1  every MS1-only high-score annotation the full run does not reproduce
       (ablation_table.py -> ablation_missing_hits.csv) is re-matched: the full-run
       partner's spectrum at its own precursor (0.01 / 0.02 / 0.05 / 0.5 Da) and
       at the MS1-only precursor (0.01), and the MS1-only spectrum at the full-run
       precursor (0.01) - which separates "the window excluded the reference" from
       "the two branches built different spectra".
   B2  every retained MS1-detected row of the full run is re-matched at
       0.01 / 0.02 / 0.05 Da on its reported precursor and at 0.01 Da on its
       quantitation ion (alignment_mz = base peak): how many high-score hits each
       window would give, and how many 0.01-Da hits change name.

Memory: waits until psutil reports > 10 GB available (poll 60 s, up to
--max-wait-min, default 90); after that falls back to lib/lcms/pos_test.msp
with a loud note (and ``library_used`` in numbers_fig4.json says so).

Usage:  python annotation_outcome.py [--run DIR] [--out DIR] [--library PATH] [--skip-ablation]
Outputs (analysis/rep3_full/): annotation_outcome.csv, annotation_outcome_summary.csv,
   annotation_outcome_windows.csv, ablation_recovery.csv, ablation_recovery_summary.csv,
   ablation_window_scan.csv, ablation_window_scan_summary.csv, numbers_fig4.json
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import psutil

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig4_common import (  # noqa: E402
    HIGH_SCORE, MIN_MATCHED, REPO_DIR, REVISION_DIR, norm_name, parse_spectrum, read_features,
    support_bin, update_numbers,
)
from asfam.config import ProcessingConfig  # noqa: E402
from asfam.io.spill import read_sample_features  # noqa: E402
from metabo_core.annotation import load_and_index_library, match_feature_topn  # noqa: E402

DEFAULT_RUN = REVISION_DIR / "runs" / "rep3_full"
DEFAULT_OUT = REVISION_DIR / "analysis" / "rep3_full"
DEFAULT_LIB = REPO_DIR / "lib" / "lcms" / "pos.msp"
FALLBACK_LIB = REPO_DIR / "lib" / "lcms" / "pos_test.msp"
MEM_NEEDED = 10e9
TOP_N = 5

ARMS = [  # (name, precursor column, window Da)
    ("truth_0.01", "true_mz", 0.01),
    ("truth_0.5", "true_mz", 0.5),
    ("nl_0.05", "predicted_mz", 0.05),
    ("nl_0.1", "predicted_mz", 0.1),
    ("nl_0.5", "predicted_mz", 0.5),
    ("centroid_0.05", "window_centroid_mz", 0.05),
    ("basepeak_0.01", "ms1_quant_mz", 0.01),
]
NL_ARMS = ["nl_0.05", "nl_0.1", "nl_0.5"]
OUTCOMES = ["same_compound", "same_formula_other_compound", "other_formula", "no_high_score_hit"]


# --------------------------------------------------------------------------- #
# Library
# --------------------------------------------------------------------------- #
def wait_for_memory(max_wait_min: float) -> bool:
    t0 = time.time()
    while True:
        avail = psutil.virtual_memory().available
        if avail > MEM_NEEDED:
            print(f"memory ok: {avail / 1e9:.1f} GB available", flush=True)
            return True
        if time.time() - t0 > max_wait_min * 60:
            return False
        print(f"waiting for memory: {avail / 1e9:.1f} GB available "
              f"({(time.time() - t0) / 60:.0f} min elapsed)", flush=True)
        time.sleep(60)


def load_library(path: Path, max_wait_min: float) -> tuple[dict, str]:
    used = str(path)
    if not wait_for_memory(max_wait_min):
        print("!!! MEMORY NEVER BECAME AVAILABLE - FALLING BACK TO pos_test.msp (1000 spectra). "
              "EVERY NUMBER BELOW IS A SMOKE TEST, NOT A RESULT !!!", flush=True)
        used = str(FALLBACK_LIB)
    t0 = time.time()
    lib = load_and_index_library(used)
    if lib is None:
        raise RuntimeError(f"library could not be loaded: {used}")
    print(f"library loaded: {len(lib['spectra']):,} spectra in {time.time() - t0:.0f}s "
          f"({psutil.Process().memory_info().rss / 1e9:.1f} GB RSS)", flush=True)
    return lib, used


class Matcher:
    def __init__(self, lib: dict):
        cfg = ProcessingConfig()
        self.spectra, self.index, self.csr = lib["spectra"], lib["index"], lib.get("csr")
        self.annotation = cfg.annotation_view()
        self.similarity = cfg.similarity_view()

    def top(self, precursor_mz: float, rt: float, mz: np.ndarray, it: np.ndarray, window: float) -> list:
        if precursor_mz is None or not np.isfinite(precursor_mz) or mz.size == 0:
            return []
        q = SimpleNamespace(precursor_mz=float(precursor_mz), rt_apex=float(rt),
                            ms2_mz=np.asarray(mz, dtype=np.float64),
                            ms2_intensity=np.asarray(it, dtype=np.float64))
        return match_feature_topn(q, self.spectra, self.index, self.annotation, self.similarity,
                                  top_n=TOP_N, csr=self.csr, precursor_window_da=float(window))


def hit_fields(prefix: str, hits: list) -> dict:
    d = {f"{prefix}_n_hits": len(hits), f"{prefix}_name": "", f"{prefix}_name_norm": "",
         f"{prefix}_formula": "", f"{prefix}_total_score": np.nan, f"{prefix}_n_matched": np.nan,
         f"{prefix}_ref_precursor_mz": np.nan, f"{prefix}_high_score": False,
         f"{prefix}_top5_names": ""}
    if hits:
        h = hits[0]
        d.update({f"{prefix}_name": h.name, f"{prefix}_name_norm": norm_name(h.name),
                  f"{prefix}_formula": h.formula or "", f"{prefix}_total_score": float(h.total_score),
                  f"{prefix}_n_matched": int(h.n_matched),
                  f"{prefix}_ref_precursor_mz": (float(h.ref_precursor_mz) if h.ref_precursor_mz is not None else np.nan),
                  f"{prefix}_high_score": bool(h.total_score >= HIGH_SCORE and h.n_matched >= MIN_MATCHED),
                  f"{prefix}_top5_names": " | ".join(norm_name(x.name) for x in hits)})
    return d


def is_high(h) -> bool:
    return bool(h.total_score >= HIGH_SCORE and h.n_matched >= MIN_MATCHED)


# --------------------------------------------------------------------------- #
# A. annotation outcome on the self-validation features
# --------------------------------------------------------------------------- #
def outcome(row: pd.Series, arm: str) -> str:
    if not row[f"{arm}_high_score"]:
        return "no_high_score_hit"
    if row[f"{arm}_name_norm"] == row["truth_0.01_name_norm"]:
        return "same_compound"
    if row[f"{arm}_formula"] and row[f"{arm}_formula"] == row["truth_0.01_formula"]:
        return "same_formula_other_compound"
    return "other_formula"


def score_band(s: float) -> str:
    if not np.isfinite(s):
        return ""
    return "1.4-1.6" if s < 1.6 else ("1.6-1.8" if s < 1.8 else ">=1.8")


def mz_band(m: float) -> str:
    return "200-300" if m < 300 else ("300-500" if m < 500 else ">=500")


def height_band(h) -> str:
    try:
        h = float(h)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(h):
        return ""
    return "<1e4" if h < 1e4 else ("1e4-1e5" if h < 1e5 else ">=1e5")


def run_outcome(matcher: Matcher, feats: dict, selfval: pd.DataFrame, centroid: pd.Series, out: Path) -> dict:
    rows = []
    t0 = time.time()
    n = len(selfval)
    for i, (_, r) in enumerate(selfval.iterrows(), 1):
        f = feats.get(r["feature_id"])
        if f is None:
            continue
        mz = np.asarray(f.ms2_mz, dtype=float); it = np.asarray(f.ms2_intensity, dtype=float)
        ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)
        mz, it = mz[ok], it[ok]
        rec = {"feature_id": r["feature_id"], "segment": r["segment"], "rt_apex": r["rt_apex"],
               "true_mz": r["true_mz"], "predicted_mz": r["predicted_mz"], "n_support": r["n_support"],
               "support_bin": support_bin(r["n_support"]), "abs_error_ppm": r["abs_error_ppm"],
               "abs_error_mda": r["abs_error_mda"], "n_fragments": r["n_fragments"],
               "ms1_height": r["ms1_height"], "ms1_quant_mz": r["ms1_quant_mz"],
               "window_centroid_mz": centroid.get(r["feature_id"], np.nan),
               "mz_band": mz_band(float(r["true_mz"])), "height_band": height_band(r["ms1_height"])}
        for arm, col, win in ARMS:
            prec = rec.get(col)
            hits = matcher.top(prec, r["rt_apex"], mz, it, win) if prec is not None and np.isfinite(prec) else []
            rec.update(hit_fields(arm, hits))
        rows.append(rec)
        if i % 500 == 0:
            print(f"  outcome {i}/{n} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df["truth_high_score"] = df["truth_0.01_high_score"]
    df["truth_score_band"] = df["truth_0.01_total_score"].map(score_band)
    for arm, _, _ in ARMS:
        if arm == "truth_0.01":
            continue
        df[f"{arm}_outcome"] = df.apply(lambda r: outcome(r, arm), axis=1)
        # rank of the truth compound inside this arm's top-5 (0 = absent)
        df[f"{arm}_truth_rank"] = [
            (names.split(" | ").index(t) + 1 if (t and names and t in names.split(" | ")) else 0)
            for names, t in zip(df[f"{arm}_top5_names"], df["truth_0.01_name_norm"])]
    df.to_csv(out / "annotation_outcome.csv", index=False)

    # ---- summaries -----------------------------------------------------------
    truth = df[df["truth_high_score"]]
    no_truth = df[~df["truth_high_score"]]
    strata = [("all", "all", truth)]
    strata += [("n_support", lvl, truth[truth["support_bin"] == lvl]) for lvl in ("1", "2", "3", ">=4")]
    strata += [("truth_score", b, truth[truth["truth_score_band"] == b]) for b in ("1.4-1.6", "1.6-1.8", ">=1.8")]
    strata += [("mz", b, truth[truth["mz_band"] == b]) for b in ("200-300", "300-500", ">=500")]
    strata += [("ms1_height", b, truth[truth["height_band"] == b]) for b in ("<1e4", "1e4-1e5", ">=1e5")]
    summ = []
    for arm, _, win in ARMS:
        if arm == "truth_0.01":
            continue
        for stype, slab, s in strata:
            if not len(s):
                continue
            oc = s[f"{arm}_outcome"].value_counts()
            row = {"arm": arm, "window_da": win, "stratum_type": stype, "stratum": slab,
                   "n_truth_hit": len(s)}
            for o in OUTCOMES:
                row[f"n_{o}"] = int(oc.get(o, 0)); row[f"frac_{o}"] = float(oc.get(o, 0) / len(s))
            row["frac_truth_in_top5"] = float((s[f"{arm}_truth_rank"] > 0).mean())
            summ.append(row)
        # features without a truth high-score hit: hits the truth cannot confirm
        if len(no_truth):
            summ.append({"arm": arm, "window_da": win, "stratum_type": "truth_no_high_score", "stratum": "all",
                         "n_truth_hit": 0, "n_unconfirmed_high_score_hits": int(no_truth[f"{arm}_high_score"].sum()),
                         "n_features_without_truth_hit": len(no_truth),
                         "frac_unconfirmed_high_score_hits": float(no_truth[f"{arm}_high_score"].mean())})
    summ_df = pd.DataFrame(summ)
    summ_df.to_csv(out / "annotation_outcome_summary.csv", index=False)

    numbers = {"outcome_n_features": len(df), "outcome_n_truth_high_score": len(truth),
               "outcome_n_truth_no_high_score": len(no_truth)}
    numbers.update(window_recommendation(summ_df, out))
    for _, r in summ_df.iterrows():
        if r["stratum_type"] in ("all", "n_support", "truth_no_high_score"):
            key = f"outcome_{r['arm']}_{r['stratum_type']}_{str(r['stratum']).replace('>=', 'ge')}"
            for c in summ_df.columns:
                if c.startswith(("n_", "frac_")) and pd.notna(r[c]):
                    numbers[f"{key}_{c}"] = r[c]
    return numbers


def adopted_nl_window_da() -> float:
    """The nl_consensus library window of the build (asfam.config; 0.05 Da since 1.4.261020.2)."""
    try:
        from asfam.config import ProcessingConfig
        return float(ProcessingConfig().annotation_precursor_window_by_source["nl_consensus"])
    except Exception:  # noqa: BLE001 - the script must still run without the app installed
        return 0.05


def window_recommendation(summ_df: pd.DataFrame, out: Path) -> dict:
    """Compare the NL windows; "recommended" marks the window adopted in the build.

    Two counts per window, both on the self-validation features: how many of the
    truth-confirmed top-1 compounds the NL precursor keeps ("correct kept") and
    how many high-score hits it produces on features whose true precursor gives
    NO high-score hit at 0.01 Da ("unconfirmed": a hit the truth cannot back; a
    reference with another precursor, i.e. the wide-window risk reviewer 4
    raised). Different-formula top-1 hits among truth-confirmed features are also
    counted but are rare: when the right reference is inside the window it wins.
    The pick maximises correct kept - different-formula - unconfirmed; the
    marginal gain of each widening step is printed so the trade-off is explicit.
    """
    win_rows = []
    for arm in NL_ARMS:
        s = summ_df[(summ_df["arm"] == arm) & (summ_df["stratum_type"] == "all")].iloc[0]
        u = summ_df[(summ_df["arm"] == arm) & (summ_df["stratum_type"] == "truth_no_high_score")]
        unconfirmed = int(u["n_unconfirmed_high_score_hits"].iloc[0]) if len(u) else 0
        win_rows.append({"arm": arm, "window_da": float(s["window_da"]), "n_truth_hit": int(s["n_truth_hit"]),
                         "n_same_compound": int(s["n_same_compound"]),
                         "n_same_formula_other_compound": int(s["n_same_formula_other_compound"]),
                         "n_other_formula": int(s["n_other_formula"]),
                         "n_no_high_score_hit": int(s["n_no_high_score_hit"]),
                         "n_unconfirmed_hits_without_truth": unconfirmed,
                         "net_same_minus_other_formula": int(s["n_same_compound"]) - int(s["n_other_formula"]),
                         "net_same_minus_other_formula_minus_unconfirmed":
                             int(s["n_same_compound"]) - int(s["n_other_formula"]) - unconfirmed})
    # reference points: the true precursor at 0.5 Da and the base-peak ion at 0.01 Da
    ref = {}
    for arm in ("truth_0.5", "basepeak_0.01", "centroid_0.05"):
        s = summ_df[(summ_df["arm"] == arm) & (summ_df["stratum_type"] == "all")]
        u = summ_df[(summ_df["arm"] == arm) & (summ_df["stratum_type"] == "truth_no_high_score")]
        if len(s):
            ref[arm] = (int(s["n_same_compound"].iloc[0]),
                        int(u["n_unconfirmed_high_score_hits"].iloc[0]) if len(u) else 0)
    win_df = pd.DataFrame(win_rows).sort_values("window_da").reset_index(drop=True)
    win_df["marginal_correct_vs_narrower"] = win_df["n_same_compound"].diff().fillna(0).astype(int)
    win_df["marginal_unconfirmed_vs_narrower"] = win_df["n_unconfirmed_hits_without_truth"].diff().fillna(0).astype(int)
    net_best = win_df.sort_values(["net_same_minus_other_formula_minus_unconfirmed", "window_da"],
                                  ascending=[False, True]).iloc[0]
    widest = win_df.sort_values("window_da").iloc[-1]
    # "recommended" = the nl_consensus window adopted in the build (asfam.config), not the window
    # with the largest net count: the build's setting is the fact the tables report, and the net
    # counts of all three windows are written out so the trade-off stays explicit (coordinator,
    # 2026-11-04; on the v4 run the net counts differ by 3 and the largest one is 0.5 Da).
    adopted_da = adopted_nl_window_da()
    win_df["adopted"] = (win_df["window_da"] - adopted_da).abs() < 1e-9
    if not win_df["adopted"].any():
        raise RuntimeError(f"adopted nl_consensus window {adopted_da} Da is not among the NL arms {list(win_df['window_da'])}")
    win_df["recommended"] = win_df["adopted"]
    win_df["largest_net"] = win_df["arm"] == net_best["arm"]
    best = win_df[win_df["adopted"]].iloc[0]
    others = win_df[~win_df["adopted"]]
    nets = win_df["net_same_minus_other_formula_minus_unconfirmed"]
    unconf = int(best["n_unconfirmed_hits_without_truth"])
    fewest = unconf <= int(others["n_unconfirmed_hits_without_truth"].min()) if len(others) else True
    rec_line = (f"ADOPTED nl_consensus window (build setting): {best['window_da']:g} Da. On the {best['n_truth_hit']} "
                f"self-validation features with a truth-confirmed high-score annotation it keeps "
                f"{best['n_same_compound']} ({best['n_same_compound'] / best['n_truth_hit'] * 100:.0f}%) correct top-1 "
                f"compounds (widest window {widest['window_da']:g} Da: {widest['n_same_compound']}), gives "
                f"{best['n_other_formula']} different-formula top-1 hits, and adds "
                + ("the fewest " if fewest else "") + f"unconfirmed high-score hits on features whose true precursor "
                f"gives none at 0.01 Da ({unconf}, against "
                + " and ".join(f"{int(r['n_unconfirmed_hits_without_truth'])}" for _, r in others.iterrows())
                + " for " + " and ".join(f"{r['window_da']:g} Da" for _, r in others.iterrows())
                + f"). The net counts (correct - other formula - unconfirmed) of the three windows differ by at most "
                f"{int(nets.max() - nets.min())} (largest net: {net_best['window_da']:g} Da): " +
                "; ".join(f"{r['window_da']:g} Da: {r['n_same_compound']} correct / {r['n_other_formula']} other formula / "
                          f"{r['n_unconfirmed_hits_without_truth']} unconfirmed, net "
                          f"{r['net_same_minus_other_formula_minus_unconfirmed']}"
                          + (f" (+{r['marginal_correct_vs_narrower']} correct, +{r['marginal_unconfirmed_vs_narrower']} "
                             f"unconfirmed vs the narrower window)" if i else "")
                          for i, (_, r) in enumerate(win_df.iterrows())) +
                ". Reference arms: true precursor at 0.5 Da keeps " +
                (f"{ref['truth_0.5'][0]} and adds {ref['truth_0.5'][1]} unconfirmed hits" if "truth_0.5" in ref else "n/a") +
                "; base-peak ion at 0.01 Da keeps " +
                (f"{ref['basepeak_0.01'][0]} and adds {ref['basepeak_0.01'][1]} (the truth centroid itself sits > 0.01 Da "
                 f"from the real ion for about half of the features, so part of the unconfirmed hits are genuine)"
                 if "basepeak_0.01" in ref else "n/a") + ".")
    win_df.to_csv(out / "annotation_outcome_windows.csv", index=False)
    pd.set_option("display.width", 250)
    print(win_df.to_string(index=False)); print(rec_line, flush=True)
    return {"outcome_recommended_nl_window_da": float(best["window_da"]),
            "outcome_adopted_nl_window_da": float(best["window_da"]),
            "outcome_largest_net_nl_window_da": float(net_best["window_da"]),
            "outcome_recommendation": rec_line}


# --------------------------------------------------------------------------- #
# B. ablation: missing hits and window scan
# --------------------------------------------------------------------------- #
def run_recovery(matcher: Matcher, out: Path) -> dict:
    path = out / "ablation_missing_hits.csv"
    if not path.exists():
        print("ablation_missing_hits.csv not found - run ablation_table.py first; skipping B1")
        return {}
    miss = pd.read_csv(path, low_memory=False)
    rows = []
    for _, r in miss.iterrows():
        rec = {k: r[k] for k in ("ms1only_feature_id", "ms1only_precursor_mz", "ms1only_rt", "ms1only_name",
                                 "ms1only_name_norm", "ms1only_formula", "ms1only_total_score", "ms1only_n_matched",
                                 "ms1only_height", "match_key", "full_feature_id", "full_detection_source",
                                 "full_signal_type", "full_precursor_mz", "full_quant_mz", "full_name",
                                 "full_total_score", "full_n_matched", "full_high_score", "delta_precursor_da",
                                 "hit_category")}
        m_mz, m_it = parse_spectrum(r.get("ms1only_ms2_spectrum"))
        f_mz, f_it = parse_spectrum(r.get("full_ms2_spectrum"))
        rec["ms1only_n_peaks"] = int(m_mz.size); rec["full_n_peaks"] = int(f_mz.size)
        rt = float(r["ms1only_rt"])
        arms = [("ms1spec_ms1prec_0.01", m_mz, m_it, r["ms1only_precursor_mz"], 0.01)]
        if f_mz.size and pd.notna(r.get("full_precursor_mz")):
            fp = float(r["full_precursor_mz"])
            arms += [("fullspec_fullprec_0.01", f_mz, f_it, fp, 0.01),
                     ("fullspec_fullprec_0.02", f_mz, f_it, fp, 0.02),
                     ("fullspec_fullprec_0.05", f_mz, f_it, fp, 0.05),
                     ("fullspec_fullprec_0.5", f_mz, f_it, fp, 0.5),
                     ("fullspec_ms1prec_0.01", f_mz, f_it, r["ms1only_precursor_mz"], 0.01),
                     ("fullspec_quantion_0.01", f_mz, f_it, r["full_quant_mz"], 0.01),
                     ("ms1spec_fullprec_0.01", m_mz, m_it, fp, 0.01)]
        for name, mz, it, prec, win in arms:
            hits = matcher.top(prec, rt, mz, it, win)
            rec.update(hit_fields(name, hits))
            rec[f"{name}_recovers_ms1only_name"] = bool(
                hits and is_high(hits[0]) and norm_name(hits[0].name) == r["ms1only_name_norm"])
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out / "ablation_recovery.csv", index=False)

    arm_names = ["ms1spec_ms1prec_0.01", "fullspec_fullprec_0.01", "fullspec_fullprec_0.02",
                 "fullspec_fullprec_0.05", "fullspec_fullprec_0.5", "fullspec_ms1prec_0.01",
                 "fullspec_quantion_0.01", "ms1spec_fullprec_0.01"]
    groups = [("all missing", df)]
    groups += [(f"partner {ds}", df[df["full_detection_source"] == ds]) for ds in ("ms1_driven", "ms2_driven")]
    groups += [("partner ms2_driven, |delta precursor| > 0.01",
                df[(df["full_detection_source"] == "ms2_driven") & (df["delta_precursor_da"].abs() > 0.01)]),
               ("partner ms2_driven, |delta precursor| <= 0.01",
                df[(df["full_detection_source"] == "ms2_driven") & (df["delta_precursor_da"].abs() <= 0.01)]),
               ("no partner", df[df["match_key"] == "none"])]
    summ = []
    for lab, g in groups:
        row = {"group": lab, "n": len(g)}
        for a in arm_names:
            col = f"{a}_recovers_ms1only_name"
            hcol = f"{a}_high_score"
            if col in g:
                row[f"{a}_same_name_high_score"] = int(g[col].eq(True).sum())
                row[f"{a}_any_high_score"] = int(g[hcol].eq(True).sum())
        summ.append(row)
    summ_df = pd.DataFrame(summ)
    summ_df.to_csv(out / "ablation_recovery_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ_df.T.to_string())
    numbers = {}
    for _, r in summ_df.iterrows():
        g = r["group"].replace(" ", "_").replace("|", "").replace(">", "gt").replace("<=", "le").replace(",", "")
        for c in summ_df.columns:
            if c != "group" and pd.notna(r[c]):
                numbers[f"recovery_{g}_{c}"] = r[c]
    return numbers


def run_window_scan(matcher: Matcher, full_csv: Path, out: Path) -> dict:
    full = read_features(full_csv)
    ms1 = full[full["signal_type"] == "ms1_detected"].copy()
    rows = []
    t0 = time.time()
    for i, (_, r) in enumerate(ms1.iterrows(), 1):
        mz, it = parse_spectrum(r.get("ms2_spectrum"))
        rec = {"feature_id": r["feature_id"], "detection_source": r["detection_source"],
               "precursor_mz": r["precursor_mz"], "quant_mz": r["alignment_mz"], "rt": r["rt"],
               "height": r["height"], "n_peaks": int(mz.size), "export_name_norm": r["name_norm"],
               "export_total_score": r["total_score"], "export_n_matched": r["n_matched"],
               "export_high_score": bool(r["high_score"]),
               "delta_precursor_quant_da": float(r["precursor_mz"]) - float(r["alignment_mz"])}
        arms = [("prec_0.01", r["precursor_mz"], 0.01), ("prec_0.02", r["precursor_mz"], 0.02),
                ("prec_0.05", r["precursor_mz"], 0.05), ("quant_0.01", r["alignment_mz"], 0.01)]
        for name, prec, win in arms:
            rec.update(hit_fields(name, matcher.top(prec, r["rt"], mz, it, win) if mz.size else []))
        rows.append(rec)
        if i % 1000 == 0:
            print(f"  window scan {i}/{len(ms1)} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df["reproduces_export"] = (df["prec_0.01_high_score"] == df["export_high_score"]) & (
        (~df["export_high_score"]) | (df["prec_0.01_name_norm"] == df["export_name_norm"]))
    for a in ("prec_0.02", "prec_0.05", "quant_0.01"):
        base_hs = df["prec_0.01_high_score"]
        df[f"{a}_gain"] = df[f"{a}_high_score"] & ~base_hs
        df[f"{a}_loss"] = ~df[f"{a}_high_score"] & base_hs
        df[f"{a}_name_change"] = base_hs & df[f"{a}_high_score"] & (df[f"{a}_name_norm"] != df["prec_0.01_name_norm"])
        df[f"{a}_formula_change"] = df[f"{a}_name_change"] & (df[f"{a}_formula"] != df["prec_0.01_formula"])
    df.to_csv(out / "ablation_window_scan.csv", index=False)
    summ = []
    for lab, g in [("all MS1-detected", df)] + [(ds, df[df["detection_source"] == ds]) for ds in ("ms1_driven", "ms2_driven")]:
        row = {"group": lab, "n_rows": len(g), "n_with_spectrum": int((g["n_peaks"] > 0).sum()),
               "export_high_score": int(g["export_high_score"].sum()),
               "reproduced_export_at_0.01": int(g["reproduces_export"].sum()),
               "prec_0.01_high_score": int(g["prec_0.01_high_score"].sum())}
        for a in ("prec_0.02", "prec_0.05", "quant_0.01"):
            row[f"{a}_high_score"] = int(g[f"{a}_high_score"].sum())
            row[f"{a}_gain"] = int(g[f"{a}_gain"].sum()); row[f"{a}_loss"] = int(g[f"{a}_loss"].sum())
            row[f"{a}_name_change"] = int(g[f"{a}_name_change"].sum())
            row[f"{a}_formula_change"] = int(g[f"{a}_formula_change"].sum())
        a = g["delta_precursor_quant_da"].abs()
        row["frac_precursor_gt_0.01_from_quant_ion"] = float((a > 0.01).mean()) if len(g) else np.nan
        summ.append(row)
    summ_df = pd.DataFrame(summ)
    summ_df.to_csv(out / "ablation_window_scan_summary.csv", index=False)
    print(summ_df.T.to_string())
    numbers = {}
    for _, r in summ_df.iterrows():
        g = r["group"].replace(" ", "_").replace("-", "_")
        for c in summ_df.columns:
            if c != "group":
                numbers[f"window_scan_{g}_{c}"] = r[c]
    return numbers


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default=str(DEFAULT_RUN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--library", default=str(DEFAULT_LIB))
    ap.add_argument("--max-wait-min", type=float, default=90.0)
    ap.add_argument("--centroid-wait-min", type=float, default=60.0)
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N self-validation features")
    ap.add_argument("--recompute-windows", action="store_true",
                    help="no library: rebuild annotation_outcome_windows.csv + the recommendation from the saved summary")
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.recompute_windows:
        summ_df = pd.read_csv(out / "annotation_outcome_summary.csv")
        numbers = window_recommendation(summ_df, out)
        update_numbers(out / "numbers_fig4.json", numbers,
                       source="scripts/annotation_outcome.py --recompute-windows -> analysis/rep3_full/annotation_outcome_windows.csv")
        return 0

    selfval = pd.read_csv(out / "inference_self_validation.csv", low_memory=False)
    if args.limit:
        selfval = selfval.head(args.limit)
    print(f"self-validation features with a prediction: {len(selfval)}")

    feats = {}
    wanted = set(selfval["feature_id"])
    for stem in sorted(p.with_suffix("") for p in (run / "_work").glob("*.mfeat")):
        for f in read_sample_features(stem, load_ms2=True):
            if f.feature_id in wanted:
                feats[f.feature_id] = f
    print(f"spectra loaded for {len(feats)} features")

    # Window centroid: from window_centroid_precision.py's table when it exists
    # (v1 runs), else from the checkpoint's own ms1_window_mz (runs >= 1.4.261020.2
    # store the whole-window centroid beside the base-peak precursor).
    centroid_path = out / "centroid_precision_rows.csv"
    from_checkpoint = pd.Series({fid: f.ms1_window_mz for fid, f in feats.items()
                                 if f.ms1_window_mz is not None}, dtype=float)
    t0 = time.time()
    while (not centroid_path.exists() and from_checkpoint.empty
           and time.time() - t0 < args.centroid_wait_min * 60):
        print("waiting for centroid_precision_rows.csv ...", flush=True); time.sleep(120)
    if centroid_path.exists():
        cp = pd.read_csv(centroid_path, low_memory=False)
        centroid = pd.to_numeric(cp.set_index("feature_id")["window_centroid_mz"], errors="coerce")
        print(f"window centroids from centroid_precision_rows.csv: {centroid.notna().sum()} features")
    elif not from_checkpoint.empty:
        centroid = from_checkpoint
        print(f"window centroids from the checkpoint's ms1_window_mz: {centroid.notna().sum()} features")
    else:
        print("!!! no window centroid available; centroid arm will be empty")
        centroid = pd.Series(dtype=float)

    lib, used = load_library(Path(args.library), args.max_wait_min)
    matcher = Matcher(lib)
    numbers = {"library_used": used, "library_n_spectra": len(lib["spectra"])}
    try:
        numbers.update(run_outcome(matcher, feats, selfval, centroid, out))
        if not args.skip_ablation:
            numbers.update(run_recovery(matcher, out))
            numbers.update(run_window_scan(matcher, run / "features.csv", out))
    finally:
        del matcher, lib
        gc.collect()
        print("library released", flush=True)
    update_numbers(out / "numbers_fig4.json", numbers,
                   source="scripts/annotation_outcome.py -> analysis/rep3_full/annotation_outcome*.csv, ablation_recovery*.csv, ablation_window_scan*.csv")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
