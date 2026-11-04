"""S2.1a / S2.1g: precursor-source summary of one ASFAMProcessor run.

Reads a run directory produced by ``run_dataset.py`` (or any ASFAMProcessor
output) and tallies, for the aligned table and for the per-sample checkpoints,
where every MS2-only feature's precursor m/z came from.

Inputs
  <run>/features.csv            aligned table (filtered to is_duplicate == False)
  <run>/_work/*.mfeat           per-sample checkpoints (all features, pre-alignment)
  <run>/_debug/processing_report.txt   stage 2.5 counters (if present)

Outputs (``--out`` directory, default analysis/<run name>/)
  sources_aligned.csv       mz_source x mz_confidence x ms2_only_tier, aligned rows
  sources_checkpoint.csv    mz_source x mz_confidence per sample, checkpoint rows
  nl_support_aligned.csv    nl_n_support distribution of the nl_consensus rows
  tier_summary.csv          MS2-only rows per tier, with high-score annotation counts
  numbers_<run name>.json   the headline numbers for analysis/numbers.json

"High-score annotation" keeps the manuscript definition: annotated == True,
total_score >= 1.4 and n_matched >= 3. The table is never counted with
``len(df)``: duplicates are marked, not deleted (CLAUDE.md).

Usage:  python precursor_sources.py <run_dir> [--out DIR] [--label NAME]
"""
from __future__ import annotations

import argparse
import json
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
HIGH_SCORE = 1.4
MIN_MATCHED = 3


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_features(run: Path) -> pd.DataFrame:
    from features_io import read_features
    df = read_features(run / "features.csv", retained_only=True)
    for col in ("mz_source", "mz_confidence", "ms2_only_tier"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    df["total_score"] = pd.to_numeric(df.get("total_score"), errors="coerce")
    df["n_matched"] = pd.to_numeric(df.get("n_matched"), errors="coerce")
    df["high_score"] = (
        truthy(df["annotated"]) if "annotated" in df.columns else False
    ) & df["total_score"].ge(HIGH_SCORE) & df["n_matched"].ge(MIN_MATCHED)
    return df


def aligned_tables(df: pd.DataFrame, out: Path) -> dict:
    ms2 = df[df["signal_type"] == "ms2_only"]
    tab = (ms2.groupby(["mz_source", "mz_confidence", "ms2_only_tier"], dropna=False)
              .agg(n=("feature_id", "size"), high_score=("high_score", "sum"))
              .reset_index())
    tab.to_csv(out / "sources_aligned.csv", index=False)

    tiers = (ms2.groupby("ms2_only_tier")
                .agg(n=("feature_id", "size"), high_score=("high_score", "sum"))
                .reset_index())
    tiers.to_csv(out / "tier_summary.csv", index=False)

    numbers = {
        "aligned_rows_retained": int(len(df)),
        "aligned_ms1_detected": int((df["signal_type"] == "ms1_detected").sum()),
        "aligned_ms2_only": int(len(ms2)),
        "aligned_ms2_only_tier_abc": int((ms2["ms2_only_tier"] != "D").sum()),
        "aligned_ms2_only_tier_d": int((ms2["ms2_only_tier"] == "D").sum()),
        "aligned_high_score": int(df["high_score"].sum()),
        "aligned_high_score_ms2_only": int(ms2["high_score"].sum()),
        "aligned_high_score_ms2_only_tier_abc": int(ms2.loc[ms2["ms2_only_tier"] != "D", "high_score"].sum()),
    }
    for src, n in Counter(ms2["mz_source"]).items():
        numbers[f"aligned_ms2_only_src_{src or 'none'}"] = int(n)
    for tier, n in Counter(ms2["ms2_only_tier"]).items():
        numbers[f"aligned_ms2_only_tier_{tier or 'blank'}"] = int(n)

    if "nl_n_support" in df.columns:
        nl = ms2[ms2["mz_source"] == "nl_consensus"].copy()
        nl["nl_n_support"] = pd.to_numeric(nl["nl_n_support"], errors="coerce")
        sup = (nl.groupby("nl_n_support")
                 .agg(n=("feature_id", "size"), high_score=("high_score", "sum"))
                 .reset_index())
        sup.to_csv(out / "nl_support_aligned.csv", index=False)
        for k, n in zip(sup["nl_n_support"], sup["n"]):
            numbers[f"aligned_nl_support_{int(k)}"] = int(n)
    return numbers


def checkpoint_tables(run: Path, out: Path) -> dict:
    work = run / "_work"
    rows = []
    total = Counter()
    for mf in sorted(work.glob("*.mfeat"), key=lambda p: (len(p.stem), p.stem)):
        with open(mf, "rb") as fh:
            payload = pickle.load(fh)
        feats = payload["features"]
        ms2 = [d for d in feats if d.get("signal_type") == "ms2_only"]
        c = Counter((d.get("mz_source") or "none", d.get("mz_confidence") or "") for d in ms2)
        few = Counter((d.get("mz_source") or "none") for d in ms2 if (d.get("n_fragments") or 0) < 3)
        for (src, conf), n in sorted(c.items()):
            rows.append({"sample": payload.get("sample_id", mf.stem), "n_features": len(feats),
                         "ms2_only": len(ms2), "mz_source": src, "mz_confidence": conf,
                         "n": n, "n_fragments_lt3": few.get(src, 0)})
            total[src] += n
        total["ms2_only"] += len(ms2)
        total["n_features"] += len(feats)
    pd.DataFrame(rows).to_csv(out / "sources_checkpoint.csv", index=False)
    return {f"checkpoint_{k}": int(v) for k, v in total.items()}


def report_counters(run: Path) -> dict:
    path = run / "_debug" / "processing_report.txt"
    if not path.exists():
        return {}
    numbers: dict = {}
    section = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("--- "):
            section = line.strip("- ").strip()
        elif section.startswith("Stage 2.5") and ":" in line:
            key, _, val = line.strip().partition(":")
            try:
                numbers[f"stage2b_{key.strip()}"] = float(val) if "." in val else int(val)
            except ValueError:
                pass
    return numbers


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

    version = ""
    with open(run / "features.csv", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("# version="):
                version = line.strip().split("=", 1)[1]
            if not line.startswith("#"):
                break

    numbers = {"software_version": version, "run_dir": str(run)}
    numbers.update(aligned_tables(load_features(run), out))
    if (run / "_work").exists():
        numbers.update(checkpoint_tables(run, out))
    numbers.update(report_counters(run))
    with open(out / f"numbers_{label}.json", "w", encoding="utf-8") as fh:
        json.dump(numbers, fh, indent=2, ensure_ascii=False)
    for k, v in numbers.items():
        print(f"{k}: {v}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
