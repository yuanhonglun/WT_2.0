"""S2.1c: do the neutral-loss consensus precursor and the isolation-window
centroid agree on real MS2-only features?

For every retained ``nl_consensus`` row of a run's features.csv the export
carries ``window_delta_mda`` = (consensus m/z - window centroid) in mDa. This
script gives its distribution stratified by the number of supporting fragments
(``nl_n_support``), by the row's quantitation height (``mean_height``, product
ion) and by m/z, with the fraction of rows inside 10 / 20 / 50 mDa, plus the
same summary for the fraction of features whose consensus is confirmed by a
*detected* MS1 peak in another sample of the same row (identity tier 3 rows).

The centroid is a mixture centroid of everything co-isolated in the 1-Da
window, so disagreement is expected when the compound is not the dominant
window ion; the stratification by height is what makes that visible.

Usage:  python nl_window_consistency.py <run_dir> [--out DIR] [--label NAME]
Outputs: consistency_by_support.csv, consistency_by_height.csv,
         consistency_by_mz.csv, consistency_rows.csv, numbers_<label>_consistency.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
REVISION_DIR = SCRIPT_DIR.parent
THRESHOLDS_MDA = (10.0, 20.0, 50.0)


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def summarize(group: pd.DataFrame) -> pd.Series:
    d = group["window_delta_mda"].to_numpy(float)
    a = np.abs(d)
    out = {"n": len(d)}
    if len(d):
        out.update({
            "median_delta_mda": float(np.median(d)),
            "median_abs_mda": float(np.median(a)),
            "p90_abs_mda": float(np.percentile(a, 90)),
            "p95_abs_mda": float(np.percentile(a, 95)),
        })
        for t in THRESHOLDS_MDA:
            out[f"frac_within_{int(t)}_mda"] = float((a <= t).mean())
    return pd.Series(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    run = Path(args.run_dir)
    label = args.label or run.name
    out = Path(args.out) if args.out else REVISION_DIR / "analysis" / label
    out.mkdir(parents=True, exist_ok=True)

    from features_io import read_features
    df = read_features(run / "features.csv", retained_only=True)
    nl = df[(df["signal_type"] == "ms2_only") & (df["mz_source"] == "nl_consensus")].copy()
    nl["window_delta_mda"] = pd.to_numeric(nl["window_delta_mda"], errors="coerce")
    nl["nl_n_support"] = pd.to_numeric(nl["nl_n_support"], errors="coerce")
    nl["mean_height"] = pd.to_numeric(nl["mean_height"], errors="coerce")
    nl["mz"] = pd.to_numeric(nl["mz"], errors="coerce")
    n_total = len(nl)
    has = nl.dropna(subset=["window_delta_mda"]).copy()

    has["support_bin"] = has["nl_n_support"].clip(upper=4).astype(int).map(
        {1: "1", 2: "2", 3: "3", 4: ">=4"})
    has["height_bin"] = pd.cut(np.log10(has["mean_height"].clip(lower=1)),
                               bins=[0, 3, 4, 5, 99], labels=["<1e3", "1e3-1e4", "1e4-1e5", ">1e5"])
    has["mz_bin"] = pd.cut(has["mz"], bins=[0, 200, 400, 600, 2000],
                           labels=["<200", "200-400", "400-600", ">600"])

    def grouped(col: str) -> pd.DataFrame:
        parts = []
        for key, sub in has.groupby(col, observed=True):
            row = summarize(sub)
            row[col] = key
            parts.append(row)
        table = pd.DataFrame(parts)
        return table[[col, *[c for c in table.columns if c != col]]] if len(table) else table

    by_support = grouped("support_bin")
    by_height = grouped("height_bin")
    by_mz = grouped("mz_bin")
    overall = summarize(has)
    by_support.to_csv(out / "consistency_by_support.csv", index=False)
    by_height.to_csv(out / "consistency_by_height.csv", index=False)
    by_mz.to_csv(out / "consistency_by_mz.csv", index=False)
    cols = ["feature_id", "mz", "rt", "mean_height", "n_fragments", "nl_n_support",
            "mz_confidence", "ms1_window_mz", "window_delta_mda", "ms2_only_tier",
            "identity_tier", "n_detected", "annotated", "total_score", "name"]
    has[[c for c in cols if c in has.columns]].to_csv(out / "consistency_rows.csv", index=False)

    numbers = {
        "nl_rows": int(n_total),
        "nl_rows_with_window_centroid": int(len(has)),
        "nl_rows_without_window_centroid": int(n_total - len(has)),
    }
    for k, v in overall.items():
        numbers[f"nl_window_{k}"] = (float(v) if isinstance(v, (float, np.floating)) else int(v))
    for _, row in by_support.iterrows():
        tag = str(row["support_bin"]).replace(">=", "ge")
        numbers[f"nl_window_support_{tag}_n"] = int(row["n"])
        for t in THRESHOLDS_MDA:
            numbers[f"nl_window_support_{tag}_within_{int(t)}"] = float(row[f"frac_within_{int(t)}_mda"])
    with open(out / f"numbers_{label}_consistency.json", "w", encoding="utf-8") as fh:
        json.dump(numbers, fh, indent=2, ensure_ascii=False)
    print(by_support.to_string(index=False))
    print(by_height.to_string(index=False))
    print(json.dumps(numbers, indent=1)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
