"""S2.4: redundancy ledger — what every stage added, marked or merged.

Reviewer 2 (R2-5) asks for a stage-by-stage account of how the feature count
moves from raw detections to the retained table, and Reviewer 2 (R2-1f) for an
estimate of how many MS2-only rows are relatives of one another. Both come out
of one run directory:

  <run>/_debug/processing_report.txt   stage counters (stage 1 / 1b / 2 / 2.5 /
                                       5b / 7) — the pipeline's own bookkeeping
  <run>/_work/*.mfeat                  per-sample status / duplicate_type /
                                       isotope & adduct group ids (pre-alignment)
  <run>/features.csv                   aligned rows: duplicate_type,
                                       alignment_relation, n_detected, group ids

Outputs (``--out`` directory):
  ledger_stages.csv       one row per stage: features in / added / marked / out,
                          summed over samples, plus the aligned-table view
  ledger_duplicate_types.csv   duplicate_type counts, per-sample and aligned
  ms2only_families.csv    MS2-only rows grouped into post-hoc families
                          (same retained-row set, RT within 0.05 min, cosine of
                          the exported spectra > 0.7 OR >= 3 shared fragments at
                          0.02 Da): family sizes and how many rows are singletons
  numbers_<label>_ledger.json

The ledger counts rows; nothing is deleted anywhere in the pipeline except the
opt-in ``discarded_few_fragments`` path, so in = out at every marking stage and
the interesting columns are the marks.

Usage:  python redundancy_ledger.py <run_dir> [--out DIR] [--label NAME]
        [--rt-window 0.05] [--cosine 0.7] [--shared 3] [--mz-tol 0.02]
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
REVISION_DIR = SCRIPT_DIR.parent


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def parse_report(path: Path) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    section = ""
    if not path.exists():
        return stats
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("--- "):
            section = line.strip("- ").strip()
            stats[section] = {}
        elif section and ":" in line:
            key, _, val = line.strip().partition(":")
            try:
                stats[section][key.strip()] = float(val) if "." in val else int(val)
            except ValueError:
                pass
    return stats


def checkpoint_marks(work: Path) -> tuple[pd.DataFrame, Counter]:
    rows = []
    totals: Counter = Counter()
    for mf in sorted(work.glob("*.mfeat"), key=lambda p: (len(p.stem), p.stem)):
        with open(mf, "rb") as fh:
            payload = pickle.load(fh)
        feats = payload["features"]
        sid = payload.get("sample_id", mf.stem)
        c_status = Counter(d.get("status") or "" for d in feats)
        c_dup = Counter((d.get("duplicate_type") or "") for d in feats if d.get("is_duplicate"))
        c_sig = Counter(d.get("signal_type") or "" for d in feats)
        c_det = Counter(d.get("detection_source") or "" for d in feats)
        iso_groups = {d.get("isotope_group_id") for d in feats if d.get("isotope_group_id") is not None}
        add_groups = {d.get("adduct_group_id") for d in feats if d.get("adduct_group_id") is not None}
        row = {"sample": sid, "n_features": len(feats),
               "ms1_detected": c_sig.get("ms1_detected", 0), "ms2_only": c_sig.get("ms2_only", 0),
               "ms2_driven": c_det.get("ms2_driven", 0), "ms1_driven": c_det.get("ms1_driven", 0),
               "is_duplicate": sum(c_dup.values()), "isotope_groups": len(iso_groups),
               "adduct_groups": len(add_groups)}
        for k, v in c_dup.items():
            row[f"dup_{k or 'untyped'}"] = v
        for k, v in c_status.items():
            row[f"status_{k}"] = v
        rows.append(row)
        totals.update({k: v for k, v in row.items() if isinstance(v, (int, float)) and k != "sample"})
    return pd.DataFrame(rows).fillna(0), totals


def parse_spectrum(text: str) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(text, str) or not text.strip():
        return np.empty(0), np.empty(0)
    mz, it = [], []
    for tok in text.split():
        if ":" in tok:
            a, b = tok.split(":", 1)
            try:
                mz.append(float(a)); it.append(float(b))
            except ValueError:
                pass
    return np.asarray(mz), np.asarray(it)


def cosine_and_shared(a, b, tol: float) -> tuple[float, int]:
    mz_a, it_a = a
    mz_b, it_b = b
    if mz_a.size == 0 or mz_b.size == 0:
        return 0.0, 0
    order = np.argsort(mz_b)
    mz_b, it_b = mz_b[order], it_b[order]
    idx = np.searchsorted(mz_b, mz_a)
    shared = 0
    dot = 0.0
    used = set()
    for i, m in enumerate(mz_a):
        best, best_d = -1, tol + 1
        for j in (idx[i] - 1, idx[i]):
            if 0 <= j < mz_b.size and j not in used:
                d = abs(mz_b[j] - m)
                if d <= tol and d < best_d:
                    best, best_d = j, d
        if best >= 0:
            used.add(best)
            shared += 1
            dot += np.sqrt(it_a[i]) * np.sqrt(it_b[best])
    na = np.sqrt(np.sum(it_a)); nb = np.sqrt(np.sum(it_b))
    return (dot / (na * nb)) if na > 0 and nb > 0 else 0.0, shared


def ms2only_families(df: pd.DataFrame, rt_window: float, cos_thr: float,
                     shared_thr: int, mz_tol: float) -> tuple[pd.DataFrame, dict]:
    ms2 = df[df["signal_type"] == "ms2_only"].copy()
    ms2 = ms2.sort_values("rt").reset_index(drop=True)
    spectra = [parse_spectrum(s) for s in ms2.get("ms2_spectrum", pd.Series([""] * len(ms2)))]
    rts = ms2["rt"].to_numpy(float)
    heights = pd.to_numeric(ms2.get("mean_height", pd.Series(np.zeros(len(ms2)))), errors="coerce").fillna(0).to_numpy(float)

    # Leader clustering, not connected components: a chain of pairwise-similar
    # neighbours across a long RT range is not one family, so every member must
    # match the family's leader (its tallest row) directly, within rt_window.
    family = np.full(len(ms2), -1)
    n_pairs = 0
    for leader in np.argsort(-heights):
        if family[leader] >= 0:
            continue
        family[leader] = leader
        lo = np.searchsorted(rts, rts[leader] - rt_window, side="left")
        hi = np.searchsorted(rts, rts[leader] + rt_window, side="right")
        for j in range(lo, hi):
            if family[j] >= 0:
                continue
            cos, shared = cosine_and_shared(spectra[leader], spectra[j], mz_tol)
            if cos > cos_thr or shared >= shared_thr:
                family[j] = leader
                n_pairs += 1
    ms2["family_id"] = family
    sizes = ms2.groupby("family_id").size()
    ms2["family_size"] = ms2["family_id"].map(sizes)
    fam = ms2[["feature_id", "rt", "mz", "n_fragments", "mz_source", "ms2_only_tier",
               "family_id", "family_size"]] if "mz_source" in ms2.columns else \
        ms2[["feature_id", "rt", "mz", "n_fragments", "family_id", "family_size"]]
    numbers = {
        "ms2only_rows": int(len(ms2)),
        "ms2only_families": int(sizes.size),
        "ms2only_singletons": int((sizes == 1).sum()),
        "ms2only_rows_in_families_ge2": int(ms2["family_size"].ge(2).sum()),
        "ms2only_largest_family": int(sizes.max()) if sizes.size else 0,
        "ms2only_linked_pairs": int(n_pairs),
    }
    return fam, numbers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--rt-window", type=float, default=0.05)
    ap.add_argument("--cosine", type=float, default=0.7)
    ap.add_argument("--shared", type=int, default=3)
    ap.add_argument("--mz-tol", type=float, default=0.02)
    args = ap.parse_args()
    run = Path(args.run_dir)
    label = args.label or run.name
    out = Path(args.out) if args.out else REVISION_DIR / "analysis" / label
    out.mkdir(parents=True, exist_ok=True)

    report = parse_report(run / "_debug" / "processing_report.txt")
    from features_io import read_features
    df = read_features(run / "features.csv")
    kept = df[~truthy(df["is_duplicate"])]

    rows = []

    def stage(name, **cols):
        rows.append({"stage": name, **cols})

    s1 = report.get("Stage 1: MS2 Detection", {})
    s1b = report.get("Stage 1b: MS1 Detection", {})
    s2 = report.get("Stage 2: MS1 Assignment", {})
    s25 = report.get("Stage 2.5: Inference", {})
    s5b = report.get("Stage 5b: Duplicate Detection", {})
    s7 = report.get("Stage 7: Alignment", {})
    stage("1 MS2-driven detection", features_out=s1.get("total_features"))
    stage("1b MS1-driven detection", added=s1b.get("ms1_driven_new"), features_out=s1b.get("total_features"))
    stage("2 MS1 assignment", ms1_detected=s2.get("ms1_detected"), ms2_only=s2.get("ms2_only"))
    stage("2.5 precursor inference", features_out=s25.get("features_after"),
          nl_consensus=s25.get("nl_assigned"), window_fallback=s25.get("window_fallback"),
          no_precursor=s25.get("no_precursor"), retained_fragment_poor=s25.get("retained_fragment_poor"),
          discarded_fragment_poor=s25.get("discarded_fragment_poor"))

    if (run / "_work").exists():
        marks, totals = checkpoint_marks(run / "_work")
        marks.to_csv(out / "ledger_duplicate_types.csv", index=False)
        for key, name in (("dup_isotope", "4 isotope marking"), ("dup_adduct", "5 adduct marking"),
                          ("dup_spectral", "5b duplicate marking"), ("dup_isf", "6 ISF marking")):
            stage(name, marked=int(totals.get(key, 0)))
        stage("6.5 end of per-sample stages", features_out=int(totals.get("n_features", 0)),
              marked_total=int(totals.get("is_duplicate", 0)),
              isotope_groups=int(totals.get("isotope_groups", 0)),
              adduct_groups=int(totals.get("adduct_groups", 0)))
    stage("7 alignment", features_out=s7.get("aligned_features"), rows_all=len(df),
          cross_sample_marked=int(truthy(df["is_duplicate"]).sum()),
          ms1_covered=int((df["alignment_relation"] == "ms1_covered").sum()) if "alignment_relation" in df else None,
          ms1_covered_partial=int((df["alignment_relation"] == "ms1_covered_partial").sum()) if "alignment_relation" in df else None)
    dup_aligned = Counter(df.loc[truthy(df["is_duplicate"]), "duplicate_type"].fillna("").astype(str))
    stage("8 retained table", features_out=len(kept),
          ms1_detected=int((kept["signal_type"] == "ms1_detected").sum()),
          ms2_only=int((kept["signal_type"] == "ms2_only").sum()),
          **{f"aligned_dup_{k or 'untyped'}": v for k, v in dup_aligned.items()})
    ledger = pd.DataFrame(rows)
    ledger.to_csv(out / "ledger_stages.csv", index=False)

    fam, fam_numbers = ms2only_families(kept, args.rt_window, args.cosine, args.shared, args.mz_tol)
    fam.to_csv(out / "ms2only_families.csv", index=False)

    numbers = {"rows_all": int(len(df)), "rows_retained": int(len(kept)),
               "rows_marked_in_alignment": int(truthy(df["is_duplicate"]).sum())}
    for k, v in dup_aligned.items():
        numbers[f"aligned_dup_{k or 'untyped'}"] = int(v)
    numbers.update({k: (None if v is None else v) for k, v in rows[-1].items() if k != "stage"})
    numbers.update(fam_numbers)
    with open(out / f"numbers_{label}_ledger.json", "w", encoding="utf-8") as fh:
        json.dump(numbers, fh, indent=2, ensure_ascii=False, default=float)
    print(ledger.to_string(index=False))
    print(json.dumps(fam_numbers, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
