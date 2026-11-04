"""Re-run one manuscript data set with the current ASFAMProcessor build (S2.0).

Every re-run of the 2026-10 revision goes through this script so the command,
configuration and output location are identical across data sets and are
recorded next to the output (``<out>/_debug/run_command.json``).

Usage (from anywhere; paths are absolute):

    python run_dataset.py rep3_full
    python run_dataset.py rep3_ms1only
    python run_dataset.py rice        --raw-dir <dir holding CK_*_P_*.mzML and S_*_P_*.mzML>
    python run_dataset.py cancer_pos  --raw-dir <dir holding *_MRMHR_P_*.mzML>
    python run_dataset.py cancer_neg  --raw-dir <dir holding *_MRMHR_N_*.mzML>
    python run_dataset.py rep1        --raw-dir <dir holding RL_ASFAM_*_P_1.mzML>
    python run_dataset.py rep2        --raw-dir <dir holding RL_ASFAM_*_P_2.mzML>

(The rice, cancer and rep1/rep2 raw files are deposited in MetaboLights MTBLS15138;
pass the folder they were downloaded to.)

Options: ``--library`` (default lib/lcms/pos.msp), ``--out`` (default
runs/<dataset>/), ``--dry-run`` (print the command only), ``--set FIELD=JSON``
(repeatable, forwarded to the asfam CLI after the preset's own overrides),
``--expect-version`` (abort unless the installed build reports this version;
use it after the freeze so every re-run provably uses the tagged code).

Presets (file globs are what the submitted runs used, recovered from the
``source_file`` field of their per-sample checkpoints):

    rep3_full     test_data/asfam/full_range/RL_ASFAM_*_P_3.mzML    31 files, 1 sample
    rep3_ms1only  same files, ``enable_ms2_driven_branch=false``     (ablation, S2.3)
    rep1 / rep2   RL_ASFAM_*_P_{1,2}.mzML                            31 files each (Fig 5)
    rice          CK_*_P_{1,2,3}.mzML + S_*_P_{1,2,3}.mzML           186 files, 6 samples
    cancer_pos    *_MRMHR_P_*.mzML, positive mode                    100 files, 10 samples
    cancer_neg    *_MRMHR_N_*.mzML, negative mode                    100 files, 10 samples

Nothing here changes a config value away from the build's defaults except the
documented ablation switch and the ionisation mode: the numbers in the paper
are the defaults of the tagged release.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
REVISION_DIR = SCRIPT_DIR.parent
REPO = REVISION_DIR.parents[1]          # .../metabo-platform
DEFAULT_LIBRARY = REPO / "lib" / "lcms" / "pos.msp"

PRESETS: dict[str, dict] = {
    "rep3_full": {
        "raw_dir": REPO / "test_data" / "asfam" / "full_range",
        "globs": ["RL_ASFAM_*_P_3.mzML"],
        "n_files": 31, "mode": "positive", "set": [],
    },
    "rep3_ms1only": {
        "raw_dir": REPO / "test_data" / "asfam" / "full_range",
        "globs": ["RL_ASFAM_*_P_3.mzML"],
        "n_files": 31, "mode": "positive",
        "set": ["enable_ms2_driven_branch=false"],
    },
    "rep1": {"raw_dir": None, "globs": ["RL_ASFAM_*_P_1.mzML"], "n_files": 31,
             "mode": "positive", "set": []},
    "rep2": {"raw_dir": None, "globs": ["RL_ASFAM_*_P_2.mzML"], "n_files": 31,
             "mode": "positive", "set": []},
    "rice": {"raw_dir": None,
             "globs": ["CK_*_P_*.mzML", "S_*_P_*.mzML"], "n_files": 186,
             "mode": "positive", "set": []},
    "cancer_pos": {"raw_dir": None,
                   "globs": ["*_MRMHR_P_*.mzML"], "n_files": 100,
                   "mode": "positive", "set": []},
    "cancer_neg": {"raw_dir": None,
                   "globs": ["*_MRMHR_N_*.mzML"], "n_files": 100,
                   "mode": "negative", "set": []},
}


def installed_version() -> str:
    out = subprocess.run([sys.executable, "-m", "asfam.cli", "--version"],
                         capture_output=True, text=True)
    return (out.stdout or out.stderr).strip().split()[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", choices=sorted(PRESETS))
    ap.add_argument("--raw-dir", default=None, help="override the preset's raw directory")
    ap.add_argument("--library", default=str(DEFAULT_LIBRARY))
    ap.add_argument("--out", default=None, help="output directory (default runs/<dataset>)")
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=JSON")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--expect-version", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    preset = PRESETS[args.dataset]
    raw_dir = Path(args.raw_dir) if args.raw_dir else preset["raw_dir"]
    if raw_dir is None:
        print(f"[{args.dataset}] no raw directory known; pass --raw-dir", file=sys.stderr)
        return 2
    if not raw_dir.exists():
        print(f"[{args.dataset}] raw directory not found: {raw_dir}", file=sys.stderr)
        return 2
    files: list[str] = []
    for pattern in preset["globs"]:
        files.extend(sorted(glob.glob(str(raw_dir / pattern))))
    files = sorted(set(files))
    if len(files) != preset["n_files"]:
        print(f"[{args.dataset}] expected {preset['n_files']} mzML, found {len(files)} "
              f"in {raw_dir}", file=sys.stderr)
        if not args.dry_run:
            return 2
    library = Path(args.library)
    if not library.exists():
        print(f"library not found: {library}", file=sys.stderr)
        return 2

    version = installed_version()
    if args.expect_version and version != args.expect_version:
        print(f"installed asfam build is {version}, expected {args.expect_version}; "
              f"check out the tag and reinstall first", file=sys.stderr)
        return 3

    out_dir = Path(args.out) if args.out else REVISION_DIR / "runs" / args.dataset
    cmd = [sys.executable, "-m", "asfam.cli", *files, "-o", str(out_dir),
           "--library", str(library), "--mode", preset["mode"],
           "--workers", str(args.workers)]
    for item in [*preset["set"], *args.set]:
        cmd += ["--set", item]

    record = {
        "dataset": args.dataset, "version": version, "raw_dir": str(raw_dir),
        "n_files": len(files), "library": str(library), "mode": preset["mode"],
        "set": [*preset["set"], *args.set], "out_dir": str(out_dir),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": cmd,
    }
    print(json.dumps({k: v for k, v in record.items() if k != "command"},
                     ensure_ascii=False, indent=2))
    if args.dry_run:
        print("DRY RUN:", " ".join(cmd[:6]), "...", " ".join(cmd[-6:]))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_debug").mkdir(exist_ok=True)
    with open(out_dir / "_debug" / "run_command.json", "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with open(out_dir / "_debug" / "run_stdout.log", "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    record["elapsed_s"] = round(time.time() - t0)
    record["returncode"] = proc.returncode
    with open(out_dir / "_debug" / "run_command.json", "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    print(f"[{args.dataset}] exit {proc.returncode} after {record['elapsed_s']} s -> {out_dir}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
