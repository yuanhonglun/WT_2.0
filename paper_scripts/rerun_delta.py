"""S2.0: old-versus-new comparison of one data set's ASFAMProcessor output.

Produces ``analysis/rerun_delta_<label>.csv`` with one row per quantity (old,
new, delta) so the response letter can state what the re-processing changed:
retained feature rows, MS1-detected / MS2-only counts, precursor-source counts
(new run only; the old export had no source column, so its sources come from
the old ``_work`` checkpoints when they exist), MS2-only tiers, high-score
annotations (annotated & total_score >= 1.4 & n_matched >= 3), and the aligned
row count including duplicates.

Usage:  python rerun_delta.py <old_run_dir> <new_run_dir> --label rice
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
REVISION_DIR = SCRIPT_DIR.parent


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def version_of(run: Path) -> str:
    with open(run / "features.csv", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("# version="):
                return line.strip().split("=", 1)[1]
            if not line.startswith("#"):
                break
    return ""


def checkpoint_sources(run: Path) -> Counter:
    c: Counter = Counter()
    for mf in sorted((run / "_work").glob("*.mfeat")):
        with open(mf, "rb") as fh:
            payload = pickle.load(fh)
        for d in payload["features"]:
            if d.get("signal_type") == "ms2_only":
                c[d.get("mz_source") or "none"] += 1
    return c


def summarize(run: Path) -> dict:
    from features_io import read_features
    df = read_features(run / "features.csv")
    out = {"software_version": version_of(run), "aligned_rows_all": len(df)}
    kept = df[~truthy(df["is_duplicate"])].copy()
    out["aligned_rows_retained"] = len(kept)
    out["ms1_detected"] = int((kept["signal_type"] == "ms1_detected").sum())
    ms2 = kept[kept["signal_type"] == "ms2_only"]
    out["ms2_only"] = int(len(ms2))
    kept["total_score"] = pd.to_numeric(kept.get("total_score"), errors="coerce")
    kept["n_matched"] = pd.to_numeric(kept.get("n_matched"), errors="coerce")
    high = truthy(kept["annotated"]) & kept["total_score"].ge(1.4) & kept["n_matched"].ge(3) \
        if "annotated" in kept.columns else pd.Series(False, index=kept.index)
    out["high_score"] = int(high.sum())
    out["high_score_ms2_only"] = int((high & (kept["signal_type"] == "ms2_only")).sum())
    if "ms2_only_tier" in kept.columns:
        tiers = Counter(ms2["ms2_only_tier"].fillna("").astype(str))
        for t in "ABCD":
            out[f"ms2_only_tier_{t}"] = int(tiers.get(t, 0))
        out["ms2_only_tier_abc"] = int(sum(tiers.get(t, 0) for t in "ABC"))
        if "mz_source" in kept.columns:
            for src, n in Counter(ms2["mz_source"].fillna("").astype(str)).items():
                out[f"aligned_ms2_only_src_{src or 'none'}"] = int(n)
    if (run / "_work").exists():
        for src, n in checkpoint_sources(run).items():
            out[f"checkpoint_ms2_only_src_{src}"] = int(n)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old_run")
    ap.add_argument("new_run")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    old, new = summarize(Path(args.old_run)), summarize(Path(args.new_run))
    keys = list(dict.fromkeys([*old, *new]))
    rows = []
    for k in keys:
        o, n = old.get(k, ""), new.get(k, "")
        delta = (n - o) if isinstance(o, (int, float)) and isinstance(n, (int, float)) else ""
        rows.append({"quantity": k, "old": o, "new": n, "delta": delta})
    table = pd.DataFrame(rows)
    out = Path(args.out) if args.out else REVISION_DIR / "analysis" / f"rerun_delta_{args.label}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
