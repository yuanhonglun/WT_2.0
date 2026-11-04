"""S2.1b: neutral-loss consensus self-validation on the v1.4 rep-3 re-run.

Reproduces Fig. 4e/f of the submitted manuscript on the per-feature checkpoint
of a run (``<run>/_work/<sample>.mfeat`` + ``.mspec``, every candidate feature
BEFORE alignment) instead of the old diagnostic dump:

  subset   mz_source == "ms1_peak" and detection_source == "ms2_driven"
           (MS2-driven features whose precursor was assigned from a detected
           MS1 peak = the truth), precursor >= 200, n_fragments >= 5;
  masking  the precursor is withheld and every fragment within 0.5 Da of it is
           removed; >= 3 fragments must remain;
  predictor the production ``asfam.pipeline.stage2b_inference._nl_consensus_predict``
           on a duck-typed shim (it reads ms2_mz / ms2_intensity /
           precursor_mz_nominal only);
  error    predicted - true, in ppm and mDa, stratified by n_support
           (1 / 2 / 3 / >=4): median, 90th, 95th percentile, fraction within
           5 / 10 / 20 ppm, fraction of features for which the predictor
           returned nothing.

Scenarios (column ``scenario``):
  masked_ge5        the manuscript definition above (main table / figure);
  unmasked_ge5      same features, no masking (what production would see if the
                    precursor ion survived in the spectrum);
  masked_frag3_4    features with exactly 3 or 4 fragments, masked (>= 3 left);
  unmasked_frag3_4  the same features with all fragments, i.e. the realistic
                    production case at the fragment floor (min_fragments_for_inference = 3).

Naive baseline (reviewer 4's control): for the same features the precursor is
guessed WITHOUT the spectrum as integer channel + the median mass defect
(precursor - channel) of the OTHER MS1-detected features of the run in the same
channel (leave-one-out; the 30-Da segment when fewer than 3 others share the
channel), in the same segment, or in the whole run. Its error is tabulated next
to the NL error by n_support (``naive_baseline.csv``), with the fraction of
features where NL beats the baseline. Reviewer 4 reported on Supplementary
Table 3: baseline 102-139 ppm vs inference 25-28 ppm (1 vote: 82 vs 100 ppm).

Usage:  python nl_self_validation.py [--run DIR] [--out DIR]
Outputs: inference_self_validation.csv (masked_ge5 rows with a prediction),
         inference_self_validation_scenarios.csv (all scenarios, all statuses),
         inference_self_validation_summary.csv, naive_baseline_rows.csv,
         naive_baseline.csv, numbers_fig4.json (keys selfval_*, naive_baseline_*)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig4_common import REVISION_DIR, support_bin, update_numbers  # noqa: E402

from asfam.io.spill import read_sample_features  # noqa: E402
from asfam.pipeline.stage2b_inference import _nl_consensus_predict  # noqa: E402

DEFAULT_RUN = REVISION_DIR / "runs" / "rep3_full"
DEFAULT_OUT = REVISION_DIR / "analysis" / "rep3_full"
PPM_THRESHOLDS = (5, 10, 20)
MASK_DA = 0.5
MIN_LEFT = 3


def checkpoint_stems(run: Path) -> list[Path]:
    return sorted(p.with_suffix("") for p in (run / "_work").glob("*.mfeat"))


def load_spotmap(run: Path) -> dict:
    p = run / "spotmap.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("spot_of", {})
    except Exception:
        return {}


def load_truth_features(run: Path, min_mz: float = 200.0):
    """All ms1_peak & ms2_driven candidate features with precursor >= min_mz,
    MS2 arrays loaded."""
    feats = []
    for stem in checkpoint_stems(run):
        for f in read_sample_features(stem, load_ms2=True):
            if f.mz_source != "ms1_peak" or f.detection_source != "ms2_driven":
                continue
            if f.ms1_precursor_mz is None or f.ms1_precursor_mz < min_mz:
                continue
            feats.append(f)
    return feats


def predict_one(f, masked: bool) -> dict:
    """Run the production predictor on one feature; returns a result row."""
    true_mz = float(f.ms1_precursor_mz)
    mz = np.asarray(f.ms2_mz, dtype=float)
    it = np.asarray(f.ms2_intensity, dtype=float)
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)
    mz, it = mz[ok], it[ok]
    n_in_spectrum = int(mz.size)
    if masked:
        keep = np.abs(mz - true_mz) > MASK_DA
        mz, it = mz[keep], it[keep]
    n_removed = n_in_spectrum - int(mz.size)
    row = {
        "feature_id": f.feature_id, "segment": f.segment_name,
        "channel": int(f.precursor_mz_nominal), "rt_apex": float(f.rt_apex),
        "ms1_height": f.ms1_height, "ms1_quant_mz": f.ms1_quant_mz,
        "n_fragments": int(f.n_fragments), "n_in_spectrum": n_in_spectrum,
        "n_removed_by_mask": n_removed, "n_used": int(mz.size),
        "true_mz": true_mz, "predicted_mz": np.nan, "n_support": np.nan,
        "confidence": "", "error_ppm": np.nan, "abs_error_ppm": np.nan,
        "error_mda": np.nan, "abs_error_mda": np.nan, "status": "",
    }
    if mz.size < MIN_LEFT:
        row["status"] = "too_few_fragments_after_mask" if masked else "too_few_fragments"
        return row
    shim = SimpleNamespace(ms2_mz=mz.tolist(), ms2_intensity=it.tolist(),
                           precursor_mz_nominal=float(f.precursor_mz_nominal))
    res = _nl_consensus_predict(shim)
    if res is None:
        row["status"] = "no_prediction"
        return row
    err_da = float(res["mz"]) - true_mz
    row.update({
        "predicted_mz": float(res["mz"]), "n_support": int(res["n_support"]),
        "confidence": res["confidence"], "error_ppm": err_da / true_mz * 1e6,
        "abs_error_ppm": abs(err_da) / true_mz * 1e6, "error_mda": err_da * 1e3,
        "abs_error_mda": abs(err_da) * 1e3, "status": "predicted",
    })
    return row


def run_scenarios(feats) -> pd.DataFrame:
    rows = []
    for f in feats:
        nf = int(f.n_fragments)
        if nf >= 5:
            for masked, name in ((True, "masked_ge5"), (False, "unmasked_ge5")):
                r = predict_one(f, masked); r["scenario"] = name; rows.append(r)
        elif nf in (3, 4):
            for masked, name in ((True, "masked_frag3_4"), (False, "unmasked_frag3_4")):
                r = predict_one(f, masked); r["scenario"] = name; rows.append(r)
    df = pd.DataFrame(rows)
    df["support_bin"] = df["n_support"].map(support_bin)
    return df


MIN_CHANNEL_OTHERS = 3


def load_ms1_population(run: Path) -> pd.DataFrame:
    """Every MS1-detected candidate (mz_source == ms1_peak, both branches) with
    its channel, segment and mass defect relative to the integer channel."""
    rows = []
    for stem in checkpoint_stems(run):
        for f in read_sample_features(stem, load_ms2=False):
            if f.mz_source != "ms1_peak" or f.ms1_precursor_mz is None:
                continue
            rows.append({"feature_id": f.feature_id, "channel": int(f.precursor_mz_nominal),
                         "segment": f.segment_name, "precursor_mz": float(f.ms1_precursor_mz),
                         "detection_source": f.detection_source})
    df = pd.DataFrame(rows)
    df["defect"] = df["precursor_mz"] - df["channel"]
    return df


def _circular_median(d: np.ndarray) -> float:
    """Median of mass defects that may straddle the +/-0.5 boundary: take the
    linear median on [-0.5, 0.5) and on [0, 1) and keep the parameterisation
    with the smaller median absolute deviation, mapped back to [-0.5, 0.5)."""
    if d.size == 0:
        return float("nan")
    m1 = float(np.median(d)); mad1 = float(np.median(np.abs(d - m1)))
    d2 = np.where(d < 0, d + 1.0, d)
    m2 = float(np.median(d2)); mad2 = float(np.median(np.abs(d2 - m2)))
    if mad2 < mad1:
        return m2 - 1.0 if m2 >= 0.5 else m2
    return m1


def naive_baseline(pop: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Per-feature baseline precursors (leave-one-out) for every row of ``df``."""
    by_channel = {ch: g[["feature_id", "defect"]].to_numpy() for ch, g in pop.groupby("channel")}
    by_segment = {sg: g[["feature_id", "defect"]].to_numpy() for sg, g in pop.groupby("segment")}
    all_arr = pop[["feature_id", "defect"]].to_numpy()

    def loo_median(arr, fid):
        if arr is None or len(arr) == 0:
            return float("nan"), 0
        keep = arr[:, 0] != fid
        d = arr[keep, 1].astype(float)
        return _circular_median(d), int(d.size)

    out = []
    for _, r in df.iterrows():
        fid, ch, sg, true_mz = r["feature_id"], int(r["channel"]), r["segment"], float(r["true_mz"])
        m_ch, n_ch = loo_median(by_channel.get(ch), fid)
        m_sg, n_sg = loo_median(by_segment.get(sg), fid)
        m_all, n_all = loo_median(all_arr, fid)
        level = "channel"
        m_used = m_ch
        if n_ch < MIN_CHANNEL_OTHERS or not np.isfinite(m_ch):
            level, m_used = "segment", m_sg
        rec = {"feature_id": fid, "scenario": r["scenario"], "channel": ch, "segment": sg, "true_mz": true_mz,
               "true_defect": true_mz - ch, "nl_predicted_mz": r["predicted_mz"], "n_support": r["n_support"],
               "support_bin": r["support_bin"], "nl_status": r["status"], "nl_abs_error_ppm": r["abs_error_ppm"],
               "n_channel_others": n_ch, "n_segment_others": n_sg, "baseline_level_used": level}
        for name, m in (("channel_or_segment", m_used), ("channel", m_ch), ("segment", m_sg), ("global", m_all)):
            b = ch + m if np.isfinite(m) else np.nan
            rec[f"baseline_{name}_mz"] = b
            rec[f"baseline_{name}_error_ppm"] = (b - true_mz) / true_mz * 1e6 if np.isfinite(b) else np.nan
            rec[f"baseline_{name}_abs_error_ppm"] = abs(rec[f"baseline_{name}_error_ppm"])
        out.append(rec)
    return pd.DataFrame(out)


BASELINES = ("channel_or_segment", "channel", "segment", "global")


def summarize_baseline(bl: pd.DataFrame) -> pd.DataFrame:
    """NL error vs baseline error, per scenario x stratum."""
    rows = []
    for scen, g in bl.groupby("scenario", sort=False):
        pred = g[g["nl_status"] == "predicted"]
        strata = [("predicted (NL available)", pred)]
        strata += [(f"n_support {lvl}", pred[pred["support_bin"] == lvl]) for lvl in ("1", "2", "3", ">=4")]
        strata += [("no NL prediction (baseline only)", g[g["nl_status"] == "no_prediction"]),
                   ("all eligible", g[~g["nl_status"].str.startswith("too_few")])]
        for lab, s in strata:
            if not len(s):
                continue
            row = {"scenario": scen, "stratum": lab, "n": len(s)}
            a = s["nl_abs_error_ppm"].to_numpy(float); a = a[np.isfinite(a)]
            row["nl_n"] = int(a.size)
            row["nl_median_abs_ppm"] = float(np.median(a)) if a.size else np.nan
            row["nl_p90_abs_ppm"] = float(np.percentile(a, 90)) if a.size else np.nan
            row["nl_frac_within_20_ppm"] = float((a <= 20).mean()) if a.size else np.nan
            for name in BASELINES:
                b = s[f"baseline_{name}_abs_error_ppm"].to_numpy(float); bb = b[np.isfinite(b)]
                row[f"baseline_{name}_median_abs_ppm"] = float(np.median(bb)) if bb.size else np.nan
                row[f"baseline_{name}_p90_abs_ppm"] = float(np.percentile(bb, 90)) if bb.size else np.nan
                row[f"baseline_{name}_frac_within_20_ppm"] = float((bb <= 20).mean()) if bb.size else np.nan
                both = np.isfinite(s["nl_abs_error_ppm"].to_numpy(float)) & np.isfinite(b)
                if both.any():
                    row[f"frac_nl_better_than_{name}"] = float(
                        (s["nl_abs_error_ppm"].to_numpy(float)[both] < b[both]).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per scenario x n_support stratum (+ 'all'): counts and error quantiles."""
    out = []
    for scen, g in df.groupby("scenario", sort=False):
        n_subset = len(g)
        n_too_few = int(g["status"].str.startswith("too_few").sum())
        eligible = g[~g["status"].str.startswith("too_few")]
        n_elig = len(eligible)
        n_none = int((eligible["status"] == "no_prediction").sum())
        pred = eligible[eligible["status"] == "predicted"]
        strata = [("all", pred)] + [(lvl, pred[pred["support_bin"] == lvl]) for lvl in ("1", "2", "3", ">=4")]
        for lvl, s in strata:
            row = {"scenario": scen, "n_support": lvl, "n_subset": n_subset,
                   "n_too_few_fragments": n_too_few, "n_eligible": n_elig,
                   "n_no_prediction": n_none,
                   "frac_no_prediction": (n_none / n_elig) if n_elig else np.nan,
                   "n_predicted": len(s),
                   "frac_of_predicted": (len(s) / len(pred)) if len(pred) else np.nan}
            if len(s):
                a_ppm = s["abs_error_ppm"].to_numpy(float)
                a_mda = s["abs_error_mda"].to_numpy(float)
                row.update({
                    "median_error_ppm_signed": float(np.median(s["error_ppm"])),
                    "median_abs_ppm": float(np.median(a_ppm)),
                    "p90_abs_ppm": float(np.percentile(a_ppm, 90)),
                    "p95_abs_ppm": float(np.percentile(a_ppm, 95)),
                    "median_abs_mda": float(np.median(a_mda)),
                    "p90_abs_mda": float(np.percentile(a_mda, 90)),
                    "p95_abs_mda": float(np.percentile(a_mda, 95)),
                })
                for t in PPM_THRESHOLDS:
                    row[f"frac_within_{t}_ppm"] = float((a_ppm <= t).mean())
                row["frac_within_10_mda"] = float((a_mda <= 10).mean())
                row["frac_within_20_mda"] = float((a_mda <= 20).mean())
            out.append(row)
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default=str(DEFAULT_RUN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--min-mz", type=float, default=200.0)
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    feats = load_truth_features(run, args.min_mz)
    print(f"ms1_peak & ms2_driven & precursor >= {args.min_mz:g}: {len(feats)} candidate features")
    spotmap = load_spotmap(run)
    df = run_scenarios(feats)
    df["spot_id"] = df["feature_id"].map(spotmap).fillna("")
    df.to_csv(out / "inference_self_validation_scenarios.csv", index=False)
    main_rows = df[(df["scenario"] == "masked_ge5") & (df["status"] == "predicted")].copy()
    main_rows.to_csv(out / "inference_self_validation.csv", index=False)
    summ = summarize(df)
    summ.to_csv(out / "inference_self_validation_summary.csv", index=False)

    pd.set_option("display.width", 220)
    cols = ["scenario", "n_support", "n_subset", "n_too_few_fragments", "n_eligible", "n_no_prediction",
            "n_predicted", "median_abs_ppm", "p90_abs_ppm", "p95_abs_ppm", "median_abs_mda",
            "frac_within_5_ppm", "frac_within_10_ppm", "frac_within_20_ppm"]
    print(summ[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    numbers = {}
    for _, r in summ.iterrows():
        key = f"selfval_{r['scenario']}_support_{r['n_support'].replace('>=', 'ge')}"
        for c in ("n_subset", "n_eligible", "n_no_prediction", "frac_no_prediction", "n_predicted",
                  "median_abs_ppm", "p90_abs_ppm", "p95_abs_ppm", "median_abs_mda", "p90_abs_mda",
                  "p95_abs_mda", "frac_within_5_ppm", "frac_within_10_ppm", "frac_within_20_ppm"):
            if c in r and pd.notna(r[c]):
                numbers[f"{key}_{c}"] = r[c]
    numbers["selfval_n_truth_features_ge200"] = len(feats)

    # ---- naive baseline (reviewer 4's control) ------------------------------
    pop = load_ms1_population(run)
    print(f"MS1-detected population for the mass-defect baseline: {len(pop)} features, "
          f"{pop['channel'].nunique()} channels")
    bl = naive_baseline(pop, df[df["scenario"].isin(["masked_ge5", "unmasked_frag3_4"])])
    bl.to_csv(out / "naive_baseline_rows.csv", index=False)
    bl_summ = summarize_baseline(bl)
    bl_summ.to_csv(out / "naive_baseline.csv", index=False)
    show = ["scenario", "stratum", "n", "nl_median_abs_ppm", "nl_p90_abs_ppm",
            "baseline_channel_or_segment_median_abs_ppm", "baseline_channel_or_segment_p90_abs_ppm",
            "baseline_segment_median_abs_ppm", "baseline_global_median_abs_ppm",
            "frac_nl_better_than_channel_or_segment"]
    print(bl_summ[show].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    for _, r in bl_summ.iterrows():
        key = (f"naive_baseline_{r['scenario']}_"
               + r["stratum"].replace(">=", "ge").replace(" ", "_").replace("(", "").replace(")", ""))
        for c in bl_summ.columns:
            if c not in ("scenario", "stratum") and pd.notna(r[c]):
                numbers[f"{key}_{c}"] = r[c]
    numbers["naive_baseline_population_n"] = len(pop)
    numbers["naive_baseline_r4_reference"] = {
        "value": "baseline 102-139 ppm vs inference 25-28 ppm; 1 vote 82 vs 100 ppm",
        "note": "reviewer 4's own computation on Supplementary Table 3 of the submitted version"}
    update_numbers(out / "numbers_fig4.json", numbers,
                   source=f"scripts/nl_self_validation.py -> {out.name}/inference_self_validation_summary.csv, naive_baseline.csv")
    print("written:", out / "inference_self_validation.csv", out / "inference_self_validation_summary.csv",
          out / "naive_baseline.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
