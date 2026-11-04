"""Seeded random sample of an MSP spectral library, written in one streaming pass
(work package S2.1f, Task B helper).

The full positive-mode library ``lib/lcms/pos.msp`` is 6.7 GB / 3.35 M spectra and needs
~6 GB RAM when indexed; the decoy FDR analysis therefore works on a seeded random
subset. Every record (blank-line separated block) is kept with probability
``--fraction`` using ``random.Random(seed)``; kept records are copied verbatim, so the
sample parses exactly like the full library.

Outputs
  <out>              the sampled MSP
  <out>.json         counts (records seen / kept), fraction, seed, input size and
                     SHA-256 of the sample

Example
  python sample_library.py --out <scratch>/pos_sample10.msp --fraction 0.1 --seed 20261020
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import hashlib
import json
import random
import time
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve()
    repo = here.parents[3]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", default=str(repo / "lib" / "lcms" / "pos.msp"))
    ap.add_argument("--out", required=True, help="output MSP path (scratch)")
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=20261020)
    args = ap.parse_args()

    src = Path(args.library)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    t0 = time.time()
    n_seen = 0
    n_kept = 0
    keep = False
    in_record = False
    with open(src, "r", encoding="utf-8", errors="replace") as fin, \
            open(out, "w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            if line.strip():
                if not in_record:
                    in_record = True
                    n_seen += 1
                    keep = rng.random() < args.fraction
                    if keep:
                        n_kept += 1
                if keep:
                    fout.write(line.rstrip("\r\n") + "\n")
            else:
                if in_record and keep:
                    fout.write("\n")
                in_record = False
                keep = False
        if in_record and keep:
            fout.write("\n")
    elapsed = time.time() - t0

    h = hashlib.sha256()
    with open(out, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    info = {
        "source_library": str(src), "source_bytes": src.stat().st_size,
        "fraction": args.fraction, "seed": args.seed,
        "records_seen": n_seen, "records_kept": n_kept,
        "kept_fraction": n_kept / n_seen if n_seen else None,
        "sample_path": str(out), "sample_bytes": out.stat().st_size,
        "sample_sha256": h.hexdigest(), "seconds": round(elapsed, 1),
        "date": time.strftime("%Y-%m-%d"),
    }
    with open(str(out) + ".json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
