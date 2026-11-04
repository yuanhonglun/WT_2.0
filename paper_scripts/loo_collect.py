"""Collect the per-library leave-one-out outputs of ``library_loo.py`` into the single
files named in the S2.1f brief (work package S2.1f, Task A).

Reads  analysis/decoy/loo_summary_<tag>.csv, numbers_loo_<tag>.json,
       loo_false_hits_examples_<tag>.csv
Writes analysis/decoy/loo_summary.csv            all libraries x windows x rates
       analysis/decoy/numbers_loo.json           {tag: numbers} plus a compact headline table
       analysis/decoy/loo_false_hits_examples.csv the 20 worst MassBank false hits
                                                  (copy of the massbank file, with a library column)
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
from pathlib import Path

import pandas as pd

HEADLINE = [
    "rate_top1_false_neutral_formula", "rate_top1_false_ion_formula", "rate_top1_high_isomer_ion",
    "rate_top1_high", "rate_any_false_neutral_formula", "rate_loo1_top1_false_neutral_formula",
    "rate_loo1_top5_false_neutral_formula", "rate_loo1_top1_same_compound",
]


def main() -> None:
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(here.parents[1] / "analysis" / "decoy"))
    ap.add_argument("--primary", default="massbank", help="tag whose examples become loo_false_hits_examples.csv")
    args = ap.parse_args()
    d = Path(args.dir)

    frames = []
    numbers = {}
    for p in sorted(d.glob("loo_summary_*.csv")):
        tag = p.stem[len("loo_summary_"):]
        s = pd.read_csv(p)
        s.insert(0, "library", tag)
        frames.append(s)
        jp = d / f"numbers_loo_{tag}.json"
        if jp.is_file():
            with open(jp, "r", encoding="utf-8") as fh:
                numbers[tag] = json.load(fh)
    if not frames:
        raise SystemExit(f"no loo_summary_*.csv in {d}")
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(d / "loo_summary.csv", index=False)

    headline = {}
    for tag, j in numbers.items():
        headline[tag] = {
            "n_candidate_spectra": j["n_candidate_spectra"], "n_compounds": j["n_compounds"],
            "n_queries": j["n_queries"], "release_tag": j.get("release_tag", ""),
            "min_precursor_decimals": j.get("min_precursor_decimals", 0),
            "windows": {w: {k: round(r[k], 5) for k in HEADLINE if k in r} for w, r in j["results"].items()},
            "post_filter_verification": j["post_filter_verification"],
            "runtime_seconds": j["runtime_seconds"],
        }
    out = {"reviewer4_reference": {"massbank_0.5_da": 0.038, "massbank_0.01_da": 0.005,
                                   "massbank_plus_gnps_0.5_da": 0.076, "massbank_plus_gnps_0.01_da": 0.027,
                                   "note": "rates quoted in the third-round review (public tag v1.2.260721)"},
           "headline": headline, "libraries": numbers}
    with open(d / "numbers_loo.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    ex = d / f"loo_false_hits_examples_{args.primary}.csv"
    if ex.is_file():
        e = pd.read_csv(ex)
        e.insert(0, "library", args.primary)
        e.to_csv(d / "loo_false_hits_examples.csv", index=False)
    pd.set_option("display.width", 250)
    show = ["library", "window_da", "n_queries"] + HEADLINE[:4] + ["rate_loo1_top1_false_neutral_formula",
                                                                   "rate_loo1_top5_false_neutral_formula"]
    print(summary[show].to_string(index=False))


if __name__ == "__main__":
    main()
