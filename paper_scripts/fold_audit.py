"""Audit of the stage-7 "same compound" fold on a single-sample run.

The v1.4 joiner folds candidate features it judges to be the same compound of
the same sample into one row (``alignment_merge_map.json``: ``fold_reason_of``
/ ``fold_evidence_of``). On the rep-3 re-run that removes 3,910 of 24,635
candidates from the table before any redundancy mark is applied, and it is the
reason the recovery of MS-DIAL reference features in Fig. 4a is lower than in
the submitted figure: repeated / split peaks of one compound that MS-DIAL lists
separately are folded into one ASFAMProcessor row at another retention time.

For every folded candidate this script reports the fold reason, the identity
tier, the spectral cosine and matched-fragment count the fold was based on, the
RT and m/z distance to the row it ended up in (``spot_of``), the signal type on
both sides (an MS1-detected candidate folded into an MS2-only row loses its
measured precursor from the table), and whether the candidate matched a strict
MS-DIAL reference feature (gen_fig4.py -> match_table_strict_candidates.csv).

Usage:  python fold_audit.py [--run DIR] [--out DIR]
Outputs: fold_audit_rows.csv, fold_audit_summary.csv, numbers_fig4.json (keys fold_*)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig4_common import REVISION_DIR, read_features, update_numbers  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default=str(REVISION_DIR / "runs" / "rep3_full"))
    ap.add_argument("--out", default=str(REVISION_DIR / "analysis" / "rep3_full"))
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    mm = json.loads((run / "alignment_merge_map.json").read_text(encoding="utf-8"))
    spot_of, reason_of, evidence_of = mm["spot_of"], mm["fold_reason_of"], mm["fold_evidence_of"]
    cand = {}
    for mf in sorted((run / "_work").glob("*.mfeat")):
        with open(mf, "rb") as fh:
            payload = pickle.load(fh)
        for r in payload["features"]:
            cand[r["feature_id"]] = r
    full = read_features(run / "features.csv", retained_only=False).set_index("feature_id")
    strict_path = out / "match_table_strict_candidates.csv"
    strict_hit = set()
    if strict_path.exists():
        ms = pd.read_csv(strict_path, low_memory=False)
        strict_hit = set(ms.loc[ms["match_class"] != "Missed", "asfam_feature_id"].dropna())

    rows = []
    for fid, reason in reason_of.items():
        c = cand.get(fid)
        if c is None:
            continue
        ev = evidence_of.get(fid, {}) or {}
        spot = spot_of.get(fid)
        tgt = cand.get(ev.get("target_peak_id"), {})
        prec = c.get("ms1_precursor_mz")
        if prec is None:
            prec = c.get("inferred_mz") if c.get("inferred_mz") is not None else c.get("precursor_mz_nominal")
        rec = {"feature_id": fid, "reason": reason, "tier": ev.get("tier"), "cosine": ev.get("cosine"),
               "n_matched_fragments": ev.get("n_matched_fragments"), "target_peak_id": ev.get("target_peak_id"),
               "cand_signal_type": c.get("signal_type"), "cand_detection_source": c.get("detection_source"),
               "cand_mz_source": c.get("mz_source"), "cand_precursor_mz": prec, "cand_rt": c.get("rt_apex"),
               "cand_segment": c.get("segment_name"), "cand_is_duplicate": bool(c.get("is_duplicate")),
               "cand_duplicate_type": c.get("duplicate_type") or "", "cand_ms1_height": c.get("ms1_height"),
               "cand_n_fragments": c.get("n_fragments"),
               "target_signal_type": tgt.get("signal_type"), "target_rt": tgt.get("rt_apex"),
               "spot_id": spot, "matched_strict_reference": fid in strict_hit}
        if spot in full.index:
            s = full.loc[spot]
            rec.update({"spot_precursor_mz": s["precursor_mz"], "spot_rt": s["rt"], "spot_signal_type": s["signal_type"],
                        "spot_detection_source": s["detection_source"], "spot_mz_source": s["mz_source"],
                        "spot_is_duplicate": bool(s["is_duplicate"]), "spot_segment": s["alignment_segment"],
                        "delta_rt_min": float(c.get("rt_apex")) - float(s["rt"]),
                        "delta_mz_da": float(prec) - float(s["precursor_mz"])})
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["route_pair"] = df["cand_signal_type"].fillna("") + " -> " + df["spot_signal_type"].fillna("")
    df.to_csv(out / "fold_audit_rows.csv", index=False)

    def q(a, p):
        a = np.abs(a[np.isfinite(a)])
        return float(np.percentile(a, p)) if a.size else np.nan

    summ = []
    groups = [("all folds", df)]
    groups += [(f"reason {r}", df[df["reason"] == r]) for r in sorted(df["reason"].dropna().unique())]
    groups += [(f"tier {int(t)}", df[df["tier"] == t]) for t in sorted(df["tier"].dropna().unique())]
    groups += [(f"route {r}", df[df["route_pair"] == r]) for r in sorted(df["route_pair"].unique())]
    groups += [("candidate was a strict MS-DIAL match", df[df["matched_strict_reference"]])]
    groups += [("MS1-detected candidate folded into a retained MS2-only row",
                df[(df["cand_signal_type"] == "ms1_detected") & (df["spot_signal_type"] == "ms2_only") & (~df["spot_is_duplicate"].fillna(True))])]
    for lab, g in groups:
        drt = g["delta_rt_min"].to_numpy(float); dmz = g["delta_mz_da"].to_numpy(float)
        summ.append({"group": lab, "n": len(g),
                     "n_cand_ms1_detected": int((g["cand_signal_type"] == "ms1_detected").sum()),
                     "n_cand_marked_duplicate": int(g["cand_is_duplicate"].sum()),
                     "n_spot_ms2_only": int((g["spot_signal_type"] == "ms2_only").sum()),
                     "n_spot_retained": int((~g["spot_is_duplicate"].fillna(True)).sum()),
                     "n_strict_reference_matches": int(g["matched_strict_reference"].sum()),
                     "abs_delta_rt_p50": q(drt, 50), "abs_delta_rt_p90": q(drt, 90), "abs_delta_rt_max": float(np.nanmax(np.abs(drt))) if len(g) else np.nan,
                     "frac_abs_delta_rt_gt_0.1": float((np.abs(drt) > 0.1).mean()) if len(g) else np.nan,
                     "frac_abs_delta_rt_gt_0.2": float((np.abs(drt) > 0.2).mean()) if len(g) else np.nan,
                     "abs_delta_mz_p50": q(dmz, 50), "abs_delta_mz_p90": q(dmz, 90),
                     "frac_abs_delta_mz_gt_0.01": float((np.abs(dmz) > 0.01).mean()) if len(g) else np.nan,
                     "cosine_p50": float(np.nanmedian(g["cosine"].astype(float))) if g["cosine"].notna().any() else np.nan,
                     "n_matched_fragments_p50": float(np.nanmedian(g["n_matched_fragments"].astype(float))) if g["n_matched_fragments"].notna().any() else np.nan})
    summ_df = pd.DataFrame(summ)
    summ_df.to_csv(out / "fold_audit_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ_df.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    numbers = {}
    for _, r in summ_df.iterrows():
        g = r["group"].replace(" ", "_").replace("->", "to").replace("-", "_")
        for c in summ_df.columns:
            if c != "group" and pd.notna(r[c]):
                numbers[f"fold_{g}_{c}"] = r[c]
    update_numbers(out / "numbers_fig4.json", numbers, source="scripts/fold_audit.py -> analysis/rep3_full/fold_audit_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
