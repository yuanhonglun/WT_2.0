"""S2.7: coverage per injection and per hour of instrument time (Reviewer 2-8).

WT 2.0 acquires one sample as 31 injections of ~15 min (31 x 15 min = 7.75 h),
and the DDA / DIA comparison data sets of Fig 5 were acquired with the same 31
injections per sample. Reviewer 2 asks that the method's cost be stated where
its coverage gains are claimed. This table divides the per-method counts of
Fig 5 (three technical replicates, mean +/- population SD, same filters as
``make_fig5.py``: retained rows only, S/N >= 3; high score = Total score >= 1.4)
by 31 injections and by 7.75 h.

Inputs (defaults = the Fig 5 folder, read-only):
  --asfam-dir   folder with asfam-{1,2,3}.csv  (ASFAMProcessor exports; replace
                by runs/rep{1,2,3}_full/features.csv after the re-run via
                --asfam-files)
  --msdial-dir  folder with DDA-{1,2,3}.csv and DIA-{1,2,3}.csv (MS-DIAL exports)
Outputs: analysis/throughput_table.csv, analysis/numbers_throughput.json

Usage:  python throughput_table.py [--asfam-dir DIR | --asfam-files f1 f2 f3]
        [--msdial-dir DIR] [--injections 31] [--minutes-per-injection 15]
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
MANUSCRIPT_DIR = REVISION_DIR.parent
HIGH_SCORE = 1.4
SN_FLOOR = 3.0


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def asfam_counts(path: Path) -> dict:
    from features_io import read_features
    a = read_features(path, retained_only=True)
    a = a[pd.to_numeric(a["sn_ratio"], errors="coerce") >= SN_FLOOR]
    ts = pd.to_numeric(a["total_score"], errors="coerce")
    high = ts >= HIGH_SCORE
    if "n_matched" in a.columns:
        high &= pd.to_numeric(a["n_matched"], errors="coerce") >= 3
    out = {
        "features": len(a),
        "features_with_ms2": int((a["n_fragments"].astype(float) > 0).sum()),
        "ms1_detected": int((a["signal_type"] == "ms1_detected").sum()),
        "ms2_only": int((a["signal_type"] == "ms2_only").sum()),
        "high_score": int(high.sum()),
    }
    if "ms2_only_tier" in a.columns:
        tier = a["ms2_only_tier"].fillna("").astype(str)
        out["ms2_only_tier_abc"] = int(((a["signal_type"] == "ms2_only") & (tier != "D")).sum())
        out["ms2_only_tier_d"] = int((tier == "D").sum())
    return out


def msdial_counts(path: Path) -> dict:
    m = pd.read_csv(path, low_memory=False)
    m = m[pd.to_numeric(m["S/N"], errors="coerce") >= SN_FLOOR]
    msms = m["MSMS spectrum"].notna() & (m["MSMS spectrum"].astype(str).str.strip() != "")
    ts = pd.to_numeric(m["Total score"], errors="coerce")
    high = ts >= HIGH_SCORE
    if "Matched peaks count" in m.columns:
        high &= pd.to_numeric(m["Matched peaks count"], errors="coerce") >= 3
    return {"features": len(m), "features_with_ms2": int(msms.sum()), "high_score": int(high.sum())}


def summarise(rows: list[dict]) -> dict:
    keys = sorted({k for r in rows for k in r})
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if k in r], dtype=float)
        out[k] = (float(vals.mean()), float(vals.std()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asfam-dir", default=str(MANUSCRIPT_DIR / "fig 5" / "asfam"),
                    help="folder with asfam-{1,2,3}.csv (ASFAMProcessor exports of the three replicates)")
    ap.add_argument("--asfam-files", nargs="*", default=None)
    ap.add_argument("--msdial-dir", default=str(MANUSCRIPT_DIR / "fig 5" / "msdial_dda_dia"))
    ap.add_argument("--injections", type=int, default=31)
    ap.add_argument("--minutes-per-injection", type=float, default=15.0)
    ap.add_argument("--out", default=str(REVISION_DIR / "analysis"))
    ap.add_argument("--asfam-label", default="WT 2.0 (ASFAMProcessor)")
    args = ap.parse_args()

    hours = args.injections * args.minutes_per_injection / 60.0
    asfam_files = ([Path(p) for p in args.asfam_files] if args.asfam_files
                   else [Path(args.asfam_dir) / f"asfam-{i}.csv" for i in (1, 2, 3)])
    versions = []
    for p in asfam_files:
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("# version="):
                    versions.append(line.strip().split("=", 1)[1])
                if not line.startswith("#"):
                    break
    methods = {
        args.asfam_label: summarise([asfam_counts(p) for p in asfam_files]),
        "DDA (MS-DIAL)": summarise([msdial_counts(Path(args.msdial_dir) / f"DDA-{i}.csv") for i in (1, 2, 3)]),
        "DIA (MS-DIAL)": summarise([msdial_counts(Path(args.msdial_dir) / f"DIA-{i}.csv") for i in (1, 2, 3)]),
    }
    rows = []
    numbers = {"injections_per_sample": args.injections, "minutes_per_injection": args.minutes_per_injection,
               "hours_per_sample": hours, "asfam_versions": sorted(set(versions))}
    for method, stats in methods.items():
        for key in ("features", "features_with_ms2", "high_score", "ms1_detected", "ms2_only",
                    "ms2_only_tier_abc", "ms2_only_tier_d"):
            if key not in stats:
                continue
            mean, sd = stats[key]
            rows.append({"method": method, "quantity": key, "mean": round(mean, 1), "sd": round(sd, 1),
                         "per_injection": round(mean / args.injections, 1),
                         "per_hour": round(mean / hours, 1), "n_replicates": 3})
            tag = method.split(" ")[0].lower().replace(".", "")
            numbers[f"{tag}_{key}_mean"] = round(mean, 1)
            numbers[f"{tag}_{key}_per_injection"] = round(mean / args.injections, 1)
            numbers[f"{tag}_{key}_per_hour"] = round(mean / hours, 1)
    table = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "throughput_table.csv", index=False)
    with open(out / "numbers_throughput.json", "w", encoding="utf-8") as fh:
        json.dump(numbers, fh, indent=2, ensure_ascii=False)
    print(table.to_string(index=False))
    print(f"instrument time per sample: {args.injections} x {args.minutes_per_injection:g} min = {hours:.2f} h; "
          f"ASFAM export versions: {sorted(set(versions))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
