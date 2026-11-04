"""Spectrum-based decoy false-discovery rate of ASFAMProcessor library annotations
(work package S2.1f, Task B; reviewer points R2-2 iv/v, R2-6).

Every retained feature's representative MS2 spectrum (``ms2_spectra.msp`` of a run,
rows of ``features.csv`` with ``is_duplicate == False``) is matched with the production
matcher and the production settings (``ProcessingConfig().annotation_view()`` /
``similarity_view()``) against
  * the TARGET library: a seeded random 10 % sample of ``lib/lcms/pos.msp``
    (``sample_library.py``; the full 6.7 GB / 3.35 M-spectrum library does not fit next to
    the other jobs on this machine), and
  * DECOY libraries derived from the same sampled spectra:
      shift    every fragment m/z moved by a random offset with |offset| ~ U(5, 25) Da and
               random sign, re-drawn while the shifted peak would fall below 50 or above
               the precursor m/z (fallback: uniform in [50, precursor]); precursor m/z,
               peak count and intensities are kept ("shifted-peaks" decoy)
      shuffle  every spectrum's fragment m/z values replaced by values drawn without
               replacement from the fragments of the *other* spectra in the same 1-Da
               precursor bucket (a decoy with realistic fragment m/z values; singleton
               buckets fall back to the shift decoy); precursor, peak count and
               intensities kept.
Target and decoy are searched separately. For a query the best hit inside a precursor
window is its top-1; FDR(t) = N_decoy(top-1 score >= t and n_matched >= 3) /
N_target(top-1 score >= t and n_matched >= 3) over t = 0.6 ... 2.0, reported for the
windows 0.5 / 0.05 / 0.01 Da (one matcher call at 0.5 Da, narrower windows by exact
post-filter on |reference precursor - query precursor|; the window never changes a
score) and, when the run carries ``mz_source``, for the production per-source window
(``ProcessingConfig.annotation_precursor_window``). Strata: all queries, MS1-detected,
MS2-only (``signal_type``). ``fdr_env`` is the monotone envelope max_{t' >= t} FDR(t');
the reported thresholds are the smallest grid t with fdr_env <= 5 % / 1 %.

Outputs (``--out-dir``, default analysis/decoy/)
  decoy_hits_<dataset>.csv     per query x window: target and decoy top-1 scores
  fdr_curve_<dataset>.csv      dataset x decoy x window x stratum x t: counts, FDR, envelope
  fdr_summary.csv              FDR at 1.4, thresholds at 5 % / 1 %, counts
  numbers_decoy.json           numbers, provenance (library sample SHA-256, seeds), runtime
  fdr_curve.png                FDR vs score threshold (matplotlib, Agg)

Example
  python spectral_decoy_fdr.py --library-sample <scratch>/pos_sample10.msp \
      --dataset rep3=<...>/runs/rep3_full --dataset rice=<...>/runs/rice
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import platform
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features_io import header_meta, read_features as _read_features_file, truthy  # noqa: E402
from library_loo import production_settings, sha256_of  # noqa: E402

from metabo_core import __version__ as CORE_VERSION  # noqa: E402
from metabo_core.annotation import ANNOTATION_METADATA_FIELDS, build_index_from_list, match_feature_topn  # noqa: E402
from metabo_core.io.spectral_library import read_msp  # noqa: E402
from metabo_core.models import CandidateFeature  # noqa: E402

HIGH_SCORE = 1.4
HIGH_MIN_MATCHED = 3
WINDOWS_DEFAULT = (0.5, 0.05, 0.01)
DEFAULT_SOURCE_WINDOWS = {"ms1_peak": 0.01, "ms1_relaxed": 0.01, "nl_consensus": 0.5, "ms1_lowres": 0.05, "": 0.5}
STRATA = ("all", "ms1_detected", "ms2_only")


# ---- queries ------------------------------------------------------------------------------

def read_features(path: Path) -> pd.DataFrame:
    """``features.csv`` via ``features_io.read_features`` (never ``comment="#"``: a library
    name containing ``#`` would truncate the row)."""
    df = _read_features_file(path)
    df["is_duplicate"] = truthy(df["is_duplicate"])
    if "annotated" in df.columns:
        df["annotated"] = truthy(df["annotated"])
    return df


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _to_int(v) -> int:
    f = _to_float(v)
    return int(f) if np.isfinite(f) else 0


def mgf_feature_ids(path: Path) -> list[str]:
    ids = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("FEATURE_ID="):
                ids.append(line.strip().split("=", 1)[1])
    return ids


def load_queries(run_dir: Path, label: str, max_queries: int, rng: random.Random) -> tuple[list[dict], dict]:
    """Representative spectra of the retained features of one ASFAMProcessor run.

    The MSP export carries no feature id: records are written in ``features.csv`` row
    order for every row with ``n_fragments > 0``. The mapping is verified against the
    precursor m/z of every row and, when the MGF export exists, against its FEATURE_ID
    sequence.
    """
    feats = read_features(run_dir / "features.csv")
    version = header_meta(run_dir / "features.csv").get("version", "")
    with_spec = feats[feats["n_fragments"].fillna(0).astype(int) > 0].reset_index(drop=True)
    msp = read_msp(str(run_dir / "ms2_spectra.msp"),
                   keep_metadata={"name", "precursor_mz", "rt", "formula"}, as_arrays=True)
    if len(msp) != len(with_spec):
        raise SystemExit(f"{label}: {len(msp)} MSP records vs {len(with_spec)} rows with fragments")
    pmz_msp = np.array([s["metadata"].get("precursor_mz", np.nan) for s in msp], dtype=float)
    pmz_csv = with_spec["precursor_mz"].to_numpy(dtype=float)
    both = np.isfinite(pmz_msp) & np.isfinite(pmz_csv)
    delta = np.abs(pmz_msp[both] - pmz_csv[both])
    if not np.all(delta <= 1e-4):
        raise SystemExit(f"{label}: MSP/features.csv precursor mismatch on {int((delta > 1e-4).sum())} rows")
    n_csv_no_precursor = int((~np.isfinite(pmz_csv)).sum())
    mgf_path = run_dir / "ms2_spectra.mgf"
    mgf_check = None
    if mgf_path.is_file():
        ids = mgf_feature_ids(mgf_path)
        mgf_check = ids == with_spec["feature_id"].astype(str).tolist()
        if not mgf_check:
            raise SystemExit(f"{label}: MGF FEATURE_ID sequence does not match features.csv")
    keep = ~with_spec["is_duplicate"].to_numpy()
    idx = np.flatnonzero(keep)
    if max_queries and max_queries < idx.size:
        idx = np.array(sorted(rng.sample(idx.tolist(), max_queries)))
    has_source = "mz_source" in with_spec.columns
    has_tier = "ms2_only_tier" in with_spec.columns
    queries = []
    n_skipped_no_precursor = 0
    for i in idx:
        row = with_spec.iloc[i]
        spec = msp[i]
        pmz = float(row["precursor_mz"]) if np.isfinite(float(row["precursor_mz"])) else float(pmz_msp[i])
        if not np.isfinite(pmz) or pmz <= 0:
            n_skipped_no_precursor += 1
            continue
        queries.append({
            "feature_id": str(row["feature_id"]), "signal_type": str(row["signal_type"]),
            "detection_source": str(row.get("detection_source", "")),
            "mz_source": ("" if not has_source or pd.isna(row["mz_source"]) else str(row["mz_source"])),
            "ms2_only_tier": ("" if not has_tier or pd.isna(row["ms2_only_tier"]) else str(row["ms2_only_tier"])),
            "n_fragments": int(row["n_fragments"]), "precursor_mz": pmz,
            "rt": float(row["rt"]), "height": float(row.get("height", np.nan)),
            "prod_total_score": _to_float(row.get("total_score", np.nan)),
            "prod_n_matched": _to_int(row.get("n_matched", 0)),
            "prod_annotated": bool(row.get("annotated", False)),
            "prod_name": "" if pd.isna(row.get("name", "")) else str(row.get("name", "")),
            "mz": np.asarray(spec["mz"], dtype=np.float64),
            "intensity": np.asarray(spec["intensity"], dtype=np.float64),
        })
    prod_high = int(((feats["total_score"] >= HIGH_SCORE) & (feats["n_matched"] >= HIGH_MIN_MATCHED)
                     & ~feats["is_duplicate"]).sum()) if "total_score" in feats.columns else None
    info = {
        "run_dir": str(run_dir), "export_version": version, "features_rows": int(len(feats)),
        "retained_rows": int((~feats["is_duplicate"]).sum()), "rows_with_spectrum": int(len(with_spec)),
        "queries": len(queries), "msp_records": len(msp), "mgf_feature_id_check": mgf_check,
        "rows_without_precursor_in_csv": n_csv_no_precursor, "queries_skipped_no_precursor": n_skipped_no_precursor,
        "retained_by_signal_type": {k: int(v) for k, v in
                                    feats.loc[~feats["is_duplicate"], "signal_type"].value_counts().items()},
        "production_high_score_retained": prod_high,
        "production_windows": "per mz_source (v1.4)" if has_source else "0.5 Da fixed (pre-v1.4)",
    }
    return queries, info


def make_feature(q: dict, k: int) -> CandidateFeature:
    pmz = q["precursor_mz"]
    feat = CandidateFeature(
        feature_id=q["feature_id"], segment_name="", replicate_id=1,
        precursor_mz_nominal=int(round(pmz)), rt_apex=0.0, rt_left=0.0, rt_right=0.0,
        ms2_mz=q["mz"], ms2_intensity=q["intensity"], n_fragments=int(q["mz"].size),
    )
    feat.ms1_precursor_mz = pmz
    return feat


# ---- decoys -------------------------------------------------------------------------------

def _flatten(spectra: list[dict]):
    counts = np.array([s["mz"].size for s in spectra], dtype=np.int64)
    offsets = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    flat_mz = np.concatenate([np.asarray(s["mz"], dtype=np.float64) for s in spectra])
    flat_int = np.concatenate([np.asarray(s["intensity"], dtype=np.float64) for s in spectra])
    owner = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    pmz = np.array([float(s["metadata"]["precursor_mz"]) for s in spectra], dtype=np.float64)
    return counts, offsets, flat_mz, flat_int, owner, pmz


def _shift_values(flat_mz: np.ndarray, pmz_owner: np.ndarray, rng: np.random.Generator, stats: dict) -> np.ndarray:
    new = np.empty_like(flat_mz)
    todo = np.ones(flat_mz.size, dtype=bool)
    for _ in range(60):
        idx = np.flatnonzero(todo)
        if idx.size == 0:
            break
        mag = rng.uniform(5.0, 25.0, idx.size)
        sign = np.where(rng.random(idx.size) < 0.5, -1.0, 1.0)
        cand = flat_mz[idx] + sign * mag
        ok = (cand >= 50.0) & (cand <= pmz_owner[idx])
        new[idx[ok]] = cand[ok]
        todo[idx[ok]] = False
    rem = np.flatnonzero(todo)
    stats["peaks_total"] = stats.get("peaks_total", 0) + int(flat_mz.size)
    stats["peaks_fallback_uniform"] = stats.get("peaks_fallback_uniform", 0)
    stats["peaks_unshifted"] = stats.get("peaks_unshifted", 0)
    if rem.size:
        hi = pmz_owner[rem]
        can = hi > 50.0
        new[rem[can]] = rng.uniform(50.0, hi[can])
        new[rem[~can]] = flat_mz[rem[~can]]
        stats["peaks_fallback_uniform"] += int(can.sum())
        stats["peaks_unshifted"] += int((~can).sum())
    return new


def _assemble(spectra, counts, offsets, new_mz, flat_int, owner):
    order = np.lexsort((new_mz, owner))
    mz_sorted = new_mz[order]
    int_sorted = flat_int[order]
    decoys = []
    for i, s in enumerate(spectra):
        a, b = offsets[i], offsets[i + 1]
        decoys.append({"mz": mz_sorted[a:b].copy(), "intensity": int_sorted[a:b].copy(),
                       "metadata": dict(s["metadata"])})
    return decoys


def shift_decoys(spectra: list[dict], seed: int) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    counts, offsets, flat_mz, flat_int, owner, pmz = _flatten(spectra)
    stats: dict = {"method": "shift",
                   "peaks_above_precursor": int(np.sum(flat_mz > pmz[owner])),
                   "peaks_below_55": int(np.sum(flat_mz < 55.0)),
                   "spectra_precursor_below_80": int(np.sum(pmz < 80.0))}
    new_mz = _shift_values(flat_mz, pmz[owner], rng, stats)
    return _assemble(spectra, counts, offsets, new_mz, flat_int, owner), stats


def shuffle_decoys(spectra: list[dict], seed: int) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    counts, offsets, flat_mz, flat_int, owner, pmz = _flatten(spectra)
    new_mz = np.empty_like(flat_mz)
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(pmz):
        buckets[int(round(p))].append(i)
    stats = {"method": "shuffle", "singleton_bucket_spectra": 0, "with_replacement_spectra": 0}
    single = []
    for key, idxs in buckets.items():
        if len(idxs) == 1:
            single.extend(idxs)
            continue
        pool_mz = np.concatenate([flat_mz[offsets[i]:offsets[i + 1]] for i in idxs])
        pool_owner = np.repeat(np.array(idxs, dtype=np.int64), counts[idxs])
        for i in idxs:
            n = int(counts[i])
            if n == 0:
                continue
            cand = pool_mz[pool_owner != i]
            replace = cand.size < n
            if replace:
                stats["with_replacement_spectra"] += 1
            new_mz[offsets[i]:offsets[i + 1]] = rng.choice(cand, n, replace=replace)
    if single:
        stats["singleton_bucket_spectra"] = len(single)
        sel = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in single])
        sub_stats: dict = {}
        new_mz[sel] = _shift_values(flat_mz[sel], pmz[owner[sel]], rng, sub_stats)
        stats["singleton_fallback_shift_stats"] = sub_stats
    return _assemble(spectra, counts, offsets, new_mz, flat_int, owner), stats


# ---- FDR ----------------------------------------------------------------------------------

def top1_in_window(matches, pmz: float, window: float):
    for m in matches:
        if abs(float(m.ref_precursor_mz) - pmz) <= window:
            return m
    return None


def fdr_table(d: pd.DataFrame, decoy_col: str, grid: np.ndarray) -> pd.DataFrame:
    t_score = d["target_score"].to_numpy(dtype=float)
    t_ok = d["target_n_matched"].to_numpy() >= HIGH_MIN_MATCHED
    d_score = d[f"{decoy_col}_score"].to_numpy(dtype=float)
    d_ok = d[f"{decoy_col}_n_matched"].to_numpy() >= HIGH_MIN_MATCHED
    n_t = np.array([int(np.sum(t_ok & (t_score >= t))) for t in grid])
    n_d = np.array([int(np.sum(d_ok & (d_score >= t))) for t in grid])
    with np.errstate(divide="ignore", invalid="ignore"):
        fdr = np.where(n_t > 0, n_d / np.maximum(n_t, 1), np.nan)
    env = np.full_like(fdr, np.nan)
    running = 0.0
    seen = False
    for i in range(len(grid) - 1, -1, -1):
        if not np.isnan(fdr[i]):
            running = max(running, fdr[i]) if seen else fdr[i]
            seen = True
            env[i] = running
    return pd.DataFrame({"t": grid, "n_target": n_t, "n_decoy": n_d, "fdr": fdr, "fdr_env": env})


def threshold_at(tab: pd.DataFrame, level: float):
    ok = tab[(tab["fdr_env"] <= level) & tab["fdr_env"].notna()]
    if ok.empty:
        return np.nan, 0
    row = ok.iloc[0]
    return float(row["t"]), int(row["n_target"])


def main() -> None:
    here = Path(__file__).resolve()
    rev_dir = here.parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library-sample", default="", help="sampled MSP written by sample_library.py")
    ap.add_argument("--dataset", action="append", default=None,
                    help="label=run_dir (repeat); default rep3 and rice")
    ap.add_argument("--windows", type=float, nargs="+", default=list(WINDOWS_DEFAULT))
    ap.add_argument("--decoys", nargs="+", default=["shift", "shuffle"], choices=["shift", "shuffle"])
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--grid", type=float, nargs=3, default=[0.6, 2.0, 0.01], metavar=("START", "STOP", "STEP"))
    ap.add_argument("--max-queries", type=int, default=0, help="per dataset (0 = all retained features)")
    ap.add_argument("--verify-n", type=int, default=200)
    ap.add_argument("--out-dir", default=str(rev_dir / "analysis" / "decoy"))
    ap.add_argument("--progress-every", type=int, default=5000)
    ap.add_argument("--plot-only", action="store_true", help="only rebuild fdr_curve.png from saved curves")
    ap.add_argument("--no-merge", action="store_true",
                    help="overwrite fdr_summary.csv / numbers_decoy.json instead of merging the datasets of "
                         "this run into the existing files (default: merge, rows of other datasets are kept)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        replot(out_dir)
        return
    t_start = time.time()
    windows = sorted(float(w) for w in args.windows)
    if max(windows) > 0.5 + 1e-12:
        raise SystemExit("windows above 0.5 Da are not supported by this one-pass design")
    grid = np.round(np.arange(args.grid[0], args.grid[1] + 1e-9, args.grid[2]), 6)
    # Defaults name the re-processed runs; the submitted rice export lives in a
    # locally named folder and is passed explicitly with --dataset when needed.
    datasets = args.dataset or [f"rep3={rev_dir / 'runs' / 'rep3_full'}",
                                f"rice={rev_dir / 'runs' / 'rice'}"]
    ann_cfg, sim_cfg, settings = production_settings()
    print("matcher settings:", settings, flush=True)
    try:
        from asfam.config import ProcessingConfig
        _pc = ProcessingConfig()
        source_window = _pc.annotation_precursor_window
        source_windows_used = dict(_pc.annotation_precursor_window_by_source)
    except Exception:
        source_windows_used = dict(DEFAULT_SOURCE_WINDOWS)

        def source_window(src: str) -> float:
            return float(DEFAULT_SOURCE_WINDOWS.get(src or "", DEFAULT_SOURCE_WINDOWS[""]))

    # ---- target library -------------------------------------------------------------
    if not args.library_sample:
        raise SystemExit("--library-sample is required unless --plot-only")
    lib_path = Path(args.library_sample)
    lib_info = {"path": str(lib_path), "bytes": lib_path.stat().st_size, "sha256": sha256_of(lib_path)}
    side = Path(str(lib_path) + ".json")
    if side.is_file():
        with open(side, "r", encoding="utf-8") as fh:
            lib_info["sampling"] = json.load(fh)
    t0 = time.time()
    target = read_msp(str(lib_path), keep_metadata=set(ANNOTATION_METADATA_FIELDS), as_arrays=True)
    target = [s for s in target if s["metadata"].get("precursor_mz") is not None and s["mz"].size > 0]
    lib_info["spectra_loaded"] = len(target)
    lib_info["peaks_total"] = int(sum(s["mz"].size for s in target))
    print(f"target sample: {len(target)} spectra, {lib_info['peaks_total']} peaks ({time.time() - t0:.0f} s)", flush=True)

    # Decoys are built from the raw arrays before the target index re-points them.
    decoy_sets: dict[str, list[dict]] = {}
    decoy_stats: dict[str, dict] = {}
    for k, method in enumerate(args.decoys):
        t0 = time.time()
        fn = shift_decoys if method == "shift" else shuffle_decoys
        decoy_sets[method], decoy_stats[method] = fn(target, args.seed + 1 + k)
        decoy_stats[method]["seconds"] = round(time.time() - t0, 1)
        print(f"decoy '{method}': {decoy_stats[method]}", flush=True)

    t0 = time.time()
    libs = {"target": build_index_from_list(target)}
    for method, specs in decoy_sets.items():
        libs[f"decoy_{method}"] = build_index_from_list(specs)
    print(f"indexes built in {time.time() - t0:.0f} s", flush=True)
    lib_names = list(libs)

    # ---- matching ---------------------------------------------------------------------
    rng = random.Random(args.seed)
    all_hits: dict[str, pd.DataFrame] = {}
    dataset_info: dict[str, dict] = {}
    verify_all: dict[str, dict] = {}
    for spec in datasets:
        label, run_dir = spec.split("=", 1)
        run_dir = Path(run_dir)
        if not (run_dir / "features.csv").is_file():
            print(f"[{label}] features.csv missing in {run_dir}; skipped", flush=True)
            continue
        queries, info = load_queries(run_dir, label, args.max_queries, rng)
        info["production_source_windows"] = dict(source_windows_used)
        info["export_mtime"] = time.strftime("%Y-%m-%d %H:%M",
                                             time.localtime((run_dir / "features.csv").stat().st_mtime))
        dataset_info[label] = info
        print(f"[{label}] {info}", flush=True)
        rows = []
        t_loop = time.time()
        for k, q in enumerate(queries):
            feat = make_feature(q, k)
            per_lib = {name: match_feature_topn(feat, lib["spectra"], lib["index"], ann_cfg, sim_cfg,
                                                top_n=10 ** 9, csr=lib["csr"], precursor_window_da=0.5)
                       for name, lib in libs.items()}
            win_list = [(f"{w:g}", w) for w in windows]
            if q["mz_source"] or info["production_windows"].startswith("per"):
                win_list.append(("production", float(source_window(q["mz_source"]))))
            for wname, w in win_list:
                row = {"feature_id": q["feature_id"], "signal_type": q["signal_type"],
                       "detection_source": q["detection_source"], "mz_source": q["mz_source"],
                       "ms2_only_tier": q["ms2_only_tier"], "n_fragments": q["n_fragments"],
                       "precursor_mz": q["precursor_mz"], "rt": q["rt"], "height": q["height"],
                       "window": wname, "window_da": w,
                       "prod_total_score": q["prod_total_score"], "prod_n_matched": q["prod_n_matched"],
                       "prod_annotated": q["prod_annotated"], "prod_name": q["prod_name"]}
                for name in lib_names:
                    m = top1_in_window(per_lib[name], q["precursor_mz"], w)
                    col = "target" if name == "target" else name
                    if m is None:
                        row.update({f"{col}_score": np.nan, f"{col}_n_matched": 0, f"{col}_wdp": np.nan,
                                    f"{col}_name": "", f"{col}_formula": "", f"{col}_delta_mz": np.nan})
                    else:
                        row.update({f"{col}_score": float(m.score), f"{col}_n_matched": int(m.n_matched),
                                    f"{col}_wdp": float(m.wdp), f"{col}_name": m.name, f"{col}_formula": m.formula,
                                    f"{col}_delta_mz": float(m.ref_precursor_mz) - q["precursor_mz"]})
                rows.append(row)
            if args.progress_every and (k + 1) % args.progress_every == 0:
                el = time.time() - t_loop
                print(f"  [{label}] {k + 1}/{len(queries)} queries, {el:.0f} s, {el / (k + 1) * 1000:.1f} ms/query",
                      flush=True)
        info["matching_seconds"] = round(time.time() - t_loop, 1)
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"decoy_hits_{label}.csv", index=False)
        all_hits[label] = df
        print(f"[{label}] matched {len(queries)} queries in {info['matching_seconds']} s", flush=True)

        # verification of the post-filter against explicit matcher windows
        if args.verify_n > 0 and queries:
            vq = rng.sample(range(len(queries)), min(args.verify_n, len(queries)))
            mism = 0
            checked = 0
            ref = df.set_index(["feature_id", "window"])
            for k in vq:
                q = queries[k]
                feat = make_feature(q, k)
                for w in windows:
                    for name, lib in libs.items():
                        direct = match_feature_topn(feat, lib["spectra"], lib["index"], ann_cfg, sim_cfg,
                                                    top_n=10 ** 9, csr=lib["csr"], precursor_window_da=w)
                        m = direct[0] if direct else None
                        a = float(m.score) if m is not None else np.nan
                        b = float(ref.loc[(q["feature_id"], f"{w:g}"), f"{name}_score"])
                        checked += 1
                        if not ((np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-12):
                            mism += 1
            verify_all[label] = {"n_checked": checked, "n_mismatch": mism}
            print(f"[{label}] post-filter verification: {verify_all[label]}", flush=True)

    # ---- FDR curves and summary -------------------------------------------------------
    summary_rows = []
    curves: dict[str, pd.DataFrame] = {}
    for label, df in all_hits.items():
        parts = []
        for method in args.decoys:
            dcol = f"decoy_{method}"
            for wname in sorted(df["window"].unique(), key=lambda s: (s == "production", s)):
                dw = df[df["window"] == wname]
                for stratum in STRATA:
                    d = dw if stratum == "all" else dw[dw["signal_type"] == stratum]
                    if d.empty:
                        continue
                    tab = fdr_table(d, dcol, grid)
                    tab.insert(0, "stratum", stratum)
                    tab.insert(0, "window", wname)
                    tab.insert(0, "decoy", method)
                    tab.insert(0, "dataset", label)
                    parts.append(tab)
                    at14 = tab.iloc[int(np.argmin(np.abs(grid - HIGH_SCORE)))]
                    t5, n5 = threshold_at(tab, 0.05)
                    t1, n1 = threshold_at(tab, 0.01)
                    high_t = (d["target_score"] >= HIGH_SCORE) & (d["target_n_matched"] >= HIGH_MIN_MATCHED)
                    high_d = (d[f"{dcol}_score"] >= HIGH_SCORE) & (d[f"{dcol}_n_matched"] >= HIGH_MIN_MATCHED)
                    prod_high = ((d["prod_total_score"] >= HIGH_SCORE) & (d["prod_n_matched"] >= HIGH_MIN_MATCHED)).sum()
                    summary_rows.append({
                        "dataset": label, "decoy": method, "window": wname, "stratum": stratum,
                        "n_queries": int(len(d)),
                        "n_target_high_1.4": int(high_t.sum()), "n_decoy_high_1.4": int(high_d.sum()),
                        "rate_target_high_1.4": float(high_t.mean()), "rate_decoy_high_1.4": float(high_d.mean()),
                        "fdr_at_1.4": float(at14["fdr"]), "fdr_env_at_1.4": float(at14["fdr_env"]),
                        "t_fdr_5pct": t5, "n_target_at_t5": n5, "t_fdr_1pct": t1, "n_target_at_t1": n1,
                        "n_production_high_1.4_full_library": int(prod_high),
                        "n_target_any_hit": int(d["target_score"].notna().sum()),
                        "n_decoy_any_hit": int(d[f"{dcol}_score"].notna().sum()),
                    })
        curve = pd.concat(parts, ignore_index=True)
        curve.to_csv(out_dir / f"fdr_curve_{label}.csv", index=False)
        curves[label] = curve
    summary = pd.DataFrame(summary_rows)
    labels = set(all_hits)
    summary_path = out_dir / "fdr_summary.csv"
    if summary_path.is_file() and not args.no_merge:
        old = pd.read_csv(summary_path, dtype={"window": str})
        summary = pd.concat([old[~old["dataset"].isin(labels)], summary], ignore_index=True)
    summary.to_csv(summary_path, index=False)

    # sanity: the envelope is non-increasing by construction; report how often the raw
    # curve increases with t between 1.0 and 2.0 (small-count noise at the top end)
    sanity = {}
    for label, curve in curves.items():
        viol = 0
        total = 0
        for _, g in curve.groupby(["decoy", "window", "stratum"]):
            f = g[(g["t"] >= 1.0) & (g["t"] <= 1.8)]["fdr"].to_numpy()
            f = f[~np.isnan(f)]
            total += max(len(f) - 1, 0)
            viol += int(np.sum(np.diff(f) > 1e-12))
        sanity[label] = {"raw_fdr_increases_between_1.0_and_1.8": viol, "steps_checked": total}

    # ---- figure -----------------------------------------------------------------------
    fig_path = out_dir / "fdr_curve.png"
    try:
        plot_curves(curves, fig_path, windows_to_plot=["0.5", "0.01"])
    except Exception as exc:  # the numbers are the deliverable; never fail on the picture
        print(f"figure skipped: {exc!r}")

    numbers = {
        "task": "S2.1f Task B spectrum-based decoy FDR",
        "software": "ASFAMProcessor", "metabo_core_version": CORE_VERSION,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "date": time.strftime("%Y-%m-%d"),
        "matcher_settings": settings,
        "high_score_rule": {"total_score_min": HIGH_SCORE, "n_matched_min": HIGH_MIN_MATCHED},
        "library_sample": lib_info,
        "decoys": decoy_stats,
        "seed": args.seed,
        "windows_da": windows, "production_source_windows": source_windows_used,
        "grid": {"start": args.grid[0], "stop": args.grid[1], "step": args.grid[2]},
        "fdr_definition": "N_decoy(top-1 score >= t & n_matched >= 3) / N_target(top-1 score >= t & n_matched >= 3); "
                          "target and decoy searched separately; fdr_env = max over t' >= t",
        "datasets": dataset_info,
        "post_filter_verification": verify_all,
        "sanity": sanity,
        "summary": summary.to_dict(orient="records"),
        "runtime_seconds": round(time.time() - t_start, 1),
        "runs": [{"date": time.strftime("%Y-%m-%d %H:%M"), "datasets": sorted(labels),
                  "metabo_core_version": CORE_VERSION, "production_source_windows": source_windows_used,
                  "runtime_seconds": round(time.time() - t_start, 1)}],
    }
    numbers_path = out_dir / "numbers_decoy.json"
    if numbers_path.is_file() and not args.no_merge:
        # Keep every dataset of earlier runs; each dataset carries its own production
        # window map, so later window changes never re-label earlier rows.
        with open(numbers_path, "r", encoding="utf-8") as fh:
            old = json.load(fh)
        for key in ("datasets", "post_filter_verification", "sanity"):
            merged = dict(old.get(key, {}))
            if key == "datasets":
                for lab, info in merged.items():
                    info.setdefault("production_source_windows", old.get("production_source_windows"))
            merged.update(numbers[key])
            numbers[key] = merged
        numbers["runs"] = old.get("runs", []) + numbers["runs"]
    with open(numbers_path, "w", encoding="utf-8") as fh:
        json.dump(numbers, fh, indent=2, ensure_ascii=False, default=lambda o: None if o != o else str(o))
    show = ["dataset", "decoy", "window", "stratum", "n_queries", "n_target_high_1.4", "n_decoy_high_1.4",
            "fdr_at_1.4", "t_fdr_5pct", "t_fdr_1pct", "n_production_high_1.4_full_library"]
    pd.set_option("display.width", 250)
    print(summary[show].to_string(index=False))
    print(f"total runtime {time.time() - t_start:.0f} s; outputs in {out_dir}")


def plot_curves(curves: dict[str, pd.DataFrame], path: Path, windows_to_plot: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    try:
        import paper_palette as pp
        colors = {"ms1_detected": pp.BLUE, "ms2_only": pp.PURPLE, "all": pp.ORANGE}
        ink, grey = pp.INK, pp.GREY
    except Exception:
        colors = {"ms1_detected": "#0072B2", "ms2_only": "#CC79A7", "all": "#E69F00"}
        ink, grey = "#231815", "#7F7F7F"
    styles = {"0.5": "-", "0.01": "--", "0.05": ":", "production": "-."}
    datasets = list(curves)
    methods = sorted({m for c in curves.values() for m in c["decoy"].unique()})
    fig, axes = plt.subplots(len(datasets), len(methods), figsize=(4.2 * len(methods), 3.3 * len(datasets)),
                             squeeze=False, sharex=True, sharey=True)
    for r, label in enumerate(datasets):
        for c, method in enumerate(methods):
            ax = axes[r][c]
            cur = curves[label]
            for wname in windows_to_plot:
                for stratum in ("ms1_detected", "ms2_only", "all"):
                    g = cur[(cur["decoy"] == method) & (cur["window"] == wname) & (cur["stratum"] == stratum)]
                    if g.empty:
                        continue
                    ax.plot(g["t"], g["fdr"].clip(lower=1e-4), color=colors[stratum], ls=styles.get(wname, "-"),
                            lw=1.6 if stratum != "all" else 1.0, alpha=1.0 if stratum != "all" else 0.7)
            ax.axhline(0.05, color=grey, lw=0.8, ls=":")
            ax.axhline(0.01, color=grey, lw=0.8, ls=":")
            ax.axvline(HIGH_SCORE, color=grey, lw=0.8)
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 1.0)
            ax.set_xlim(0.6, 2.0)
            ax.set_title(f"{label}  |  decoy: {method}", fontsize=9, color=ink, loc="left")
            ax.grid(False)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(labelsize=8, colors=ink)
            if r == len(datasets) - 1:
                ax.set_xlabel("Total score threshold t (n_matched >= 3)", fontsize=8, color=ink)
            if c == 0:
                ax.set_ylabel("decoy / target  (FDR estimate)", fontsize=8, color=ink)
    handles = [Line2D([], [], color=colors["ms1_detected"], lw=1.6, label="MS1-detected"),
               Line2D([], [], color=colors["ms2_only"], lw=1.6, label="MS2-only"),
               Line2D([], [], color=colors["all"], lw=1.0, alpha=0.7, label="all retained features")]
    handles += [Line2D([], [], color=ink, ls=styles[w], lw=1.2, label=f"window {w} Da") for w in windows_to_plot]
    handles += [Line2D([], [], color=grey, ls=":", lw=0.8, label="FDR 5 % / 1 %"),
                Line2D([], [], color=grey, lw=0.8, label="t = 1.4 (manuscript)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Spectrum-level decoy FDR of ASFAMProcessor library annotations (10 % library sample)",
                 fontsize=10, color=ink)
    fig.tight_layout(rect=(0, 0.11, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def replot(out_dir: Path) -> None:
    """Rebuild fdr_curve.png from the saved fdr_curve_<dataset>.csv files."""
    curves = {}
    for p in sorted(out_dir.glob("fdr_curve_*.csv")):
        label = p.stem[len("fdr_curve_"):]
        curves[label] = pd.read_csv(p, dtype={"window": str})
    if not curves:
        raise SystemExit(f"no fdr_curve_*.csv in {out_dir}")
    plot_curves(curves, out_dir / "fdr_curve.png", windows_to_plot=["0.5", "0.01"])
    print(f"re-plotted {len(curves)} dataset(s) -> {out_dir / 'fdr_curve.png'}")


if __name__ == "__main__":
    main()
