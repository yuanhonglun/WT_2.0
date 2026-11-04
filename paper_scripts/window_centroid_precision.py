"""S2.1d: how precise is the isolation-window centroid as a precursor estimate?

The window centroid is the pipeline's LAST fallback for an MS2-only precursor
(``mz_source="ms1_lowres"``). Its error cannot be measured on MS2-only features
(no truth), so this script measures it where a truth exists: on MS1-detected,
MS2-driven features of a run, whose precursor was assigned from a detected MS1
peak. For each such feature the production algorithm
(``asfam.pipeline.stage2_ms1_assignment.window_centroid_mz``) is applied at the
feature's own apex cycle, exactly as stage 2 does for MS2-only features, and
compared with two references:

  base_peak   ``ms1_quant_mz``: the tallest MS1 ion in the window at the MS1 peak
              apex, i.e. the real ion the feature was quantified on (primary
              truth, ion-level);
  strict_mz   ``ms1_precursor_mz``: the window centroid at the MS1 *peak* apex
              (the reported precursor of an MS1-detected feature).

Both differences are reported in Da and ppm, stratified by MS1 peak height,
by the number of MS1 ions inside the window at the apex cycle, by the base
peak's share of the window intensity ("purity") and by m/z, with 50 / 90 / 95 /
99 % quantiles. The 95 % quantile of |centroid - base_peak| is the value the
2026-10 revision uses for the ``ms1_lowres`` library-match window.

The estimate is optimistic for the real fallback population: these features
had an MS1 peak strong enough to be detected, MS2-only features by definition
did not, and a weaker compound is a smaller share of the window mixture.

Also written: the distribution of |ms1_precursor_mz - ms1_quant_mz| over all
MS1-detected features (how far the reported precursor of an MS1-detected
feature sits from the ion it was quantified on), which decides whether 0.01 Da
is a safe library-match window for ``ms1_peak`` rows.

Inputs: a run directory with ``_work/*.mfeat`` whose ``source_file`` paths still
exist (the raw mzML is re-read one segment at a time).
Usage:  python window_centroid_precision.py <run_dir> [--out DIR] [--label NAME]
        [--max-segments N]
Outputs: centroid_precision_rows.csv, centroid_precision_by_<stratum>.csv,
         precursor_vs_quant_ion.csv, numbers_<label>_centroid.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

from asfam.config import ProcessingConfig                       # noqa: E402
from asfam.pipeline.stage0_load import run_stage0_one_sample    # noqa: E402
from asfam.pipeline.stage2_ms1_assignment import window_centroid_mz  # noqa: E402
from asfam.models import CandidateFeature                       # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REVISION_DIR = SCRIPT_DIR.parent
QUANTILES = (50, 90, 95, 99)


def _feature_stub(rec: dict) -> CandidateFeature:
    return CandidateFeature(
        feature_id=str(rec["feature_id"]), segment_name=rec["segment_name"],
        replicate_id=rec["replicate_id"], precursor_mz_nominal=int(rec["precursor_mz_nominal"]),
        rt_apex=float(rec["rt_apex"]), rt_left=float(rec["rt_left"]), rt_right=float(rec["rt_right"]),
        ms2_mz=np.empty(0), ms2_intensity=np.empty(0), n_fragments=int(rec.get("n_fragments") or 0),
    )


def _window_stats(raw, feat) -> tuple[int, float]:
    """(n ions in window, base-peak share of window intensity) at the apex cycle."""
    rt = raw.rt_array
    cycle = raw.cycles[int(np.argmin(np.abs(rt - feat.rt_apex)))]
    ch = float(feat.precursor_mz_nominal)
    mask = (cycle.ms1_mz >= ch - 0.5) & (cycle.ms1_mz <= ch + 0.5)
    n = int(mask.sum())
    if n == 0:
        return 0, float("nan")
    ints = cycle.ms1_intensity[mask]
    return n, float(ints.max() / ints.sum())


def quantile_row(values: np.ndarray, label: str) -> dict:
    a = np.abs(values[np.isfinite(values)])
    row = {"stratum": label, "n": int(a.size)}
    for q in QUANTILES:
        row[f"p{q}"] = float(np.percentile(a, q)) if a.size else float("nan")
    row["frac_within_0.01"] = float((a <= 0.01).mean()) if a.size else float("nan")
    row["frac_within_0.05"] = float((a <= 0.05).mean()) if a.size else float("nan")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--max-segments", type=int, default=None)
    args = ap.parse_args()
    run = Path(args.run_dir)
    label = args.label or run.name
    out = Path(args.out) if args.out else REVISION_DIR / "analysis" / label
    out.mkdir(parents=True, exist_ok=True)

    config = ProcessingConfig()
    config.n_workers = 1

    by_file: dict[str, list[dict]] = defaultdict(list)
    all_ms1 = []
    for mf in sorted((run / "_work").glob("*.mfeat")):
        with open(mf, "rb") as fh:
            payload = pickle.load(fh)
        for rec in payload["features"]:
            if rec.get("signal_type") != "ms1_detected":
                continue
            all_ms1.append({
                "sample": payload.get("sample_id", mf.stem), "feature_id": rec["feature_id"],
                "detection_source": rec.get("detection_source"), "mz_source": rec.get("mz_source"),
                "ms1_precursor_mz": rec.get("ms1_precursor_mz"), "ms1_quant_mz": rec.get("ms1_quant_mz"),
                "ms1_height": rec.get("ms1_height"),
            })
            if rec.get("detection_source") == "ms2_driven" and rec.get("source_file") \
                    and rec.get("ms1_quant_mz") is not None:
                by_file[rec["source_file"]].append(rec)

    pq = pd.DataFrame(all_ms1)
    pq["delta_precursor_vs_quant_da"] = (
        pd.to_numeric(pq["ms1_precursor_mz"]) - pd.to_numeric(pq["ms1_quant_mz"])
    )
    pq.to_csv(out / "precursor_vs_quant_ion.csv", index=False)
    pq_rows = [quantile_row(pq["delta_precursor_vs_quant_da"].to_numpy(float), "all ms1_detected")]
    for src, grp in pq.groupby("detection_source"):
        pq_rows.append(quantile_row(grp["delta_precursor_vs_quant_da"].to_numpy(float), f"detection_source={src}"))
    pd.DataFrame(pq_rows).to_csv(out / "precursor_vs_quant_ion_quantiles.csv", index=False)

    rows = []
    files = sorted(by_file)
    if args.max_segments:
        files = files[: args.max_segments]
    t0 = time.time()
    for i, path in enumerate(files, 1):
        if not Path(path).exists():
            print(f"  missing raw file, skipped: {path}")
            continue
        segments = run_stage0_one_sample([path], config)
        if not segments:
            continue
        raw = segments[0]
        for rec in by_file[path]:
            feat = _feature_stub(rec)
            centroid = window_centroid_mz(feat, raw)
            n_ions, purity = _window_stats(raw, feat)
            base = float(rec["ms1_quant_mz"])
            strict = float(rec["ms1_precursor_mz"]) if rec.get("ms1_precursor_mz") is not None else np.nan
            rows.append({
                "feature_id": rec["feature_id"], "segment": rec["segment_name"],
                "channel": rec["precursor_mz_nominal"], "rt_apex": rec["rt_apex"],
                "ms1_height": rec.get("ms1_height"), "n_fragments": rec.get("n_fragments"),
                "base_peak_mz": base, "strict_mz": strict,
                "window_centroid_mz": centroid,
                "n_ions_in_window": n_ions, "base_peak_share": purity,
                "err_vs_base_da": (centroid - base) if centroid is not None else np.nan,
                "err_vs_base_ppm": ((centroid - base) / base * 1e6) if centroid is not None else np.nan,
                "err_vs_strict_da": (centroid - strict) if centroid is not None else np.nan,
            })
        del raw, segments
        print(f"  [{i}/{len(files)}] {Path(path).name}: {len(by_file[path])} features, "
              f"{time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "centroid_precision_rows.csv", index=False)
    numbers = {"n_ms1_detected_ms2_driven_evaluated": int(len(df)),
               "n_without_window_signal": int(df["window_centroid_mz"].isna().sum()) if len(df) else 0}
    strata = {
        "all": [("all", df)],
        "height": [(f"height {lo:g}-{hi:g}", df[(df["ms1_height"] >= lo) & (df["ms1_height"] < hi)])
                   for lo, hi in ((0, 1e3), (1e3, 1e4), (1e4, 1e5), (1e5, np.inf))],
        "n_ions": [(f"ions {lab}", sub) for lab, sub in (
            ("1", df[df["n_ions_in_window"] == 1]), ("2-3", df[df["n_ions_in_window"].between(2, 3)]),
            ("4-6", df[df["n_ions_in_window"].between(4, 6)]), (">6", df[df["n_ions_in_window"] > 6]))],
        "purity": [(f"purity {lo:g}-{hi:g}", df[(df["base_peak_share"] >= lo) & (df["base_peak_share"] < hi)])
                   for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01))],
        "mz": [(f"mz {lo}-{hi}", df[(df["channel"] >= lo) & (df["channel"] < hi)])
               for lo, hi in ((0, 200), (200, 400), (400, 600), (600, 2000))],
    }
    for name, groups in strata.items():
        table = []
        for lab, sub in groups:
            if not len(sub):
                continue
            r = quantile_row(sub["err_vs_base_da"].to_numpy(float), lab)
            r_ppm = quantile_row(sub["err_vs_base_ppm"].to_numpy(float), lab)
            r.update({f"ppm_p{q}": r_ppm[f"p{q}"] for q in QUANTILES})
            r_strict = quantile_row(sub["err_vs_strict_da"].to_numpy(float), lab)
            r.update({f"vs_strict_p{q}": r_strict[f"p{q}"] for q in QUANTILES})
            table.append(r)
        pd.DataFrame(table).to_csv(out / f"centroid_precision_by_{name}.csv", index=False)
        if name == "all" and table:
            for q in QUANTILES:
                numbers[f"centroid_err_vs_base_p{q}_da"] = table[0][f"p{q}"]
                numbers[f"centroid_err_vs_base_p{q}_ppm"] = table[0][f"ppm_p{q}"]
                numbers[f"centroid_err_vs_strict_p{q}_da"] = table[0][f"vs_strict_p{q}"]
            numbers["centroid_frac_within_0.01_da"] = table[0]["frac_within_0.01"]
            numbers["centroid_frac_within_0.05_da"] = table[0]["frac_within_0.05"]
    for q in QUANTILES:
        numbers[f"precursor_vs_quant_ion_p{q}_da"] = pq_rows[0][f"p{q}"]
    numbers["precursor_vs_quant_ion_frac_within_0.01_da"] = pq_rows[0]["frac_within_0.01"]
    with open(out / f"numbers_{label}_centroid.json", "w", encoding="utf-8") as fh:
        json.dump(numbers, fh, indent=2, ensure_ascii=False)
    print(json.dumps(numbers, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
