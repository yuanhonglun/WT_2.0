"""MS-DIAL <-> METRA ASFAM feature-comparison tool (PR-E Task E1).

Parses an MS-DIAL aligned-result ``.txt`` export and a METRA ``features.csv``
for the same segment, greedily one-to-one matches features by m/z + RT, and
reports coverage / net-add / annotation-similarity metrics per segment and in
total.

Design notes
------------
* ``parse_msdial_txt`` / ``parse_metra_csv`` / ``match_features`` / ``report``
  are pure, module-level, and unit-testable. The CLI lives in ``main()`` under
  ``if __name__ == "__main__"``.
* MS-DIAL column map (1-indexed, verified against the real
  ``RL_ASFAM_<seg>_P_3.txt`` files; Python uses index-1):
  col2=Name, col5=RT(min), col7=Precursor m/z, col8=Height, col11=S/N,
  col18=Isotope (0=mono, >=1=M+n copy), col24=InChIKey,
  col32=Simple dot product, col33=Weighted dot product, col34=Reverse dot
  product, col36=Matched peaks percentage, col37=Total score.
  Self-check (parser MUST reproduce): data-row counts 285-314 -> 323,
  495-524 -> 634, 795-824 -> 479; mono (Isotope==0) for 285-314 -> 248.
* Metric interpretation caveat: MS-DIAL "Total score" folds in precursor-mass
  / RT terms, whereas METRA's ``total_score`` is spectral-only. The annotation
  similarity MAE therefore carries an expected systematic component; this is a
  PR-E interpretation matter, not a tool bug.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Segments compared by the CLI (each maps to one METRA dir + one MS-DIAL file).
SEGMENTS = ["285-314", "495-524", "795-824"]


# --------------------------------------------------------------------------- #
# Safe coercion helpers
# --------------------------------------------------------------------------- #
def _to_float(value, default: Optional[float] = None) -> Optional[float]:
    """Coerce ``value`` to float, returning ``default`` on failure / NaN."""
    if value is None:
        return default
    try:
        f = float(str(value).strip())
    except (ValueError, TypeError):
        return default
    if math.isnan(f):
        return default
    return f


def _to_int(value, default: int = 0) -> int:
    """Coerce ``value`` to int (via float), returning ``default`` on failure."""
    f = _to_float(value, None)
    return int(f) if f is not None else default


def _to_str(value) -> str:
    """Coerce ``value`` to a stripped str; NaN / None / ``'nan'`` -> ``''``."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    s = str(value).strip()
    if s.lower() == "nan":
        return ""
    return s


def _is_confident_name(name: str) -> bool:
    """Whether a Name field is a confident annotation.

    Empty / ``null`` / ``Unknown*`` / ``low score: ...`` count as NOT confident.
    """
    n = _to_str(name)
    if not n:
        return False
    low = n.lower()
    if low == "null":
        return False
    if low.startswith("low score:"):
        return False
    if low.startswith("unknown"):
        return False
    return True


def _is_ms2_driven(detection_source: str) -> bool:
    """Whether a METRA feature's detection_source carries an MS2-driven origin.

    Counts ``ms2_driven`` and ``both`` (the "METRA 特色净增" set). Robust to
    compound forms that embed ``ms2``.
    """
    d = _to_str(detection_source).lower()
    return ("ms2" in d) or (d == "both")


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #
def parse_msdial_txt(path) -> list[dict]:
    """Parse an MS-DIAL aligned-result ``.txt`` into a list of feature dicts.

    Tab-delimited; line 1 is the header, every later non-empty line is one
    feature. Rows too short to hold the Total-score column (index 36) are
    skipped; numeric cells coerce safely (bad -> ``None``), so the returned
    length reproduces the file's data-row count.
    """
    path = Path(path)
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    for line in lines[1:]:                       # skip header
        if not line.strip():
            continue
        c = line.split("\t")
        if len(c) < 37:                          # need up to col37 (index 36)
            continue
        name = _to_str(c[1])
        rec = {
            "name": name,
            "rt": _to_float(c[4]),
            "mz": _to_float(c[6]),
            "height": _to_float(c[7], 0.0),
            "sn": _to_float(c[10]),
            "isotope": _to_int(c[17], 0),
            "inchikey": _to_str(c[23]),
            "sdp": _to_float(c[31]),
            "wdp": _to_float(c[32]),
            "rdp": _to_float(c[33]),
            "matched_pct": _to_float(c[35]),
            "total_score": _to_float(c[36]),
            "annotated": _is_confident_name(name),
        }
        rows.append(rec)
    return rows


def parse_msdial_csv(path) -> list[dict]:
    """Parse an *integrated* MS-DIAL result CSV (named columns) into feature dicts.

    Unlike ``parse_msdial_txt`` (raw per-segment ``.txt``, parsed by column
    index), this reads the comma-separated, **named-column** CSV produced by the
    MS-DIAL results-integration tool (``merged_df.to_csv``, e.g.
    ``ASFAM-3.csv``). Columns are matched by NAME, so column order is
    irrelevant. The returned dict schema is identical to ``parse_msdial_txt`` so
    every downstream recall helper (``recall_match`` / ``dynamic_mz_tol`` /
    segment + m/z-bin reports / sentinel / missed CSV) is reused verbatim.

    Recognised columns (others ignored): ``Precursor m/z``->mz, ``RT (min)``->rt,
    ``Height``->height, ``S/N``->sn, ``Isotope``->isotope, ``Name``->name,
    ``InChIKey``->inchikey, ``MW``->mw, ``Adduct``->adduct. The integration tool
    emits a second ``S/N`` column which pandas auto-renames ``S/N.1``; the first
    (``S/N``) is the one read here.
    """
    path = Path(path)
    df = pd.read_csv(path)
    rows: list[dict] = []
    for _, r in df.iterrows():
        name = _to_str(r.get("Name"))
        rows.append({
            "name": name,
            "rt": _to_float(r.get("RT (min)")),
            "mz": _to_float(r.get("Precursor m/z")),
            "height": _to_float(r.get("Height"), 0.0),
            "sn": _to_float(r.get("S/N")),
            "isotope": _to_int(r.get("Isotope"), 0),
            "inchikey": _to_str(r.get("InChIKey")),
            "mw": _to_float(r.get("MW")),
            "adduct": _to_str(r.get("Adduct")),
            "annotated": _is_confident_name(name),
        })
    return rows


def parse_metra_csv(path) -> list[dict]:
    """Parse a METRA ``features.csv`` into a list of feature dicts.

    The first lines starting with ``#`` are comments (mode / version /
    chromatographic_mode); ``pd.read_csv(comment='#')`` drops them. Annotation
    cells are empty when a feature has no library hit; ``detection_source`` may
    be absent in older exports (callers degrade the net-add metric in that
    case).
    """
    path = Path(path)
    df = pd.read_csv(path, comment="#")
    records: list[dict] = []
    for _, r in df.iterrows():
        name = _to_str(r.get("name"))
        rec = {
            "mz": _to_float(r.get("mz")),
            "rt": _to_float(r.get("rt")),
            "height": _to_float(r.get("height"), None),
            "mean_height": _to_float(r.get("mean_height"), None),
            "isotope": _to_int(r.get("isotope_index"), 0),
            "name": name,
            "total_score": _to_float(r.get("total_score")),
            "wdp": _to_float(r.get("wdp_score")),
            "sdp": _to_float(r.get("sdp_score")),
            "rdp": _to_float(r.get("rdp_score")),
            "matched_pct": _to_float(r.get("matched_pct")),
            "sn": _to_float(r.get("sn_ratio")),
            "inchikey": _to_str(r.get("inchikey")),
            "detection_source": _to_str(r.get("detection_source")),
            "is_duplicate": r.get("is_duplicate"),
            "duplicate_type": _to_str(r.get("duplicate_type")),
            "signal_type": _to_str(r.get("signal_type")),
            "annotated": _is_confident_name(name),
        }
        records.append(rec)
    return records


# --------------------------------------------------------------------------- #
# Matching (pure)
# --------------------------------------------------------------------------- #
def _height_of(m: dict) -> float:
    h = m.get("height")
    if h is None:
        h = m.get("mean_height")
    return h if h is not None else 0.0


def match_features(metra: list[dict], msdial: list[dict],
                   mz_tol: float = 0.01, rt_tol: float = 0.1):
    """Greedy one-to-one match of METRA features against MS-DIAL features.

    METRA is sorted by height (``mean_height`` fallback) descending; each METRA
    item claims its nearest still-unmatched MS-DIAL item within
    ``abs(dmz) <= mz_tol and abs(drt) <= rt_tol`` (nearest by combined distance
    ``abs(dmz)/mz_tol + abs(drt)/rt_tol``). Pure, no I/O.

    Returns ``(matched, msdial_only, metra_only)`` where ``matched`` is a list
    of ``(metra, msdial)`` pairs and the ``*_only`` lists are the unmatched
    remainders.
    """
    metra_sorted = sorted(metra, key=_height_of, reverse=True)
    used = [False] * len(msdial)
    matched: list[tuple] = []
    metra_only: list[dict] = []

    for mt in metra_sorted:
        mz, rt = mt.get("mz"), mt.get("rt")
        if mz is None or rt is None:
            metra_only.append(mt)
            continue
        best_j, best_d = -1, None
        for j, md in enumerate(msdial):
            if used[j]:
                continue
            mmz, mrt = md.get("mz"), md.get("rt")
            if mmz is None or mrt is None:
                continue
            dmz, drt = abs(mz - mmz), abs(rt - mrt)
            if dmz <= mz_tol and drt <= rt_tol:
                d = dmz / mz_tol + drt / rt_tol
                if best_d is None or d < best_d:
                    best_d, best_j = d, j
        if best_j >= 0:
            used[best_j] = True
            matched.append((mt, msdial[best_j]))
        else:
            metra_only.append(mt)

    msdial_only = [msdial[j] for j in range(len(msdial)) if not used[j]]
    return matched, msdial_only, metra_only


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _same_compound(a: dict, b: dict) -> bool:
    """Same-compound test: InChIKey when both present, else case-insensitive
    name."""
    ka, kb = _to_str(a.get("inchikey")), _to_str(b.get("inchikey"))
    if ka and kb:
        return ka.upper() == kb.upper()
    na, nb = _to_str(a.get("name")).lower(), _to_str(b.get("name")).lower()
    return bool(na) and na == nb


def report(seg: dict) -> dict:
    """Build the per-segment metric dict from a matched segment bundle.

    ``seg`` must carry ``name``, ``metra``, ``msdial``, ``matched``,
    ``msdial_only``, ``metra_only`` (as produced by ``match_features``).

    Metrics:
      * totals + mono counts (both sides)
      * coverage recall = matched MS-DIAL mono (isotope==0) / MS-DIAL mono
      * net-add = metra_only whose detection_source is MS2-driven (ms2_driven /
        both); degrades to "all metra_only" when the column is absent
      * similarity MAE = mean ``abs(metra.total_score - msdial.total_score)``
        over m/z+RT-matched pairs that both confidently annotate the SAME
        compound; ``None`` when no such pair exists.
    """
    metra = seg["metra"]
    msdial = seg["msdial"]
    matched = seg["matched"]
    metra_only = seg["metra_only"]

    metra_total = len(metra)
    msdial_total = len(msdial)
    metra_mono = sum(1 for m in metra if m.get("isotope", 0) == 0)
    msdial_mono = sum(1 for m in msdial if m.get("isotope", 0) == 0)

    # Coverage recall: count matched pairs whose MS-DIAL side is mono
    # (isotope==0) — isotope copies are never credited to mono recall.
    matched_msdial_mono = sum(
        1 for (_mt, md) in matched if md.get("isotope", 0) == 0
    )
    coverage_recall = (matched_msdial_mono / msdial_mono) if msdial_mono else None

    # METRA feature net-add.
    has_ds = any(_to_str(m.get("detection_source")) for m in metra)
    if has_ds:
        net_add = sum(1 for m in metra_only if _is_ms2_driven(m.get("detection_source")))
        net_add_degraded = False
    else:
        net_add = len(metra_only)        # degraded fallback (no detection_source)
        net_add_degraded = True

    # Annotation-similarity MAE over same-compound matched pairs.
    abs_diffs = []
    for (mt, md) in matched:
        if not (mt.get("annotated") and md.get("annotated")):
            continue
        if not _same_compound(mt, md):
            continue
        ms, ds = mt.get("total_score"), md.get("total_score")
        if ms is None or ds is None:
            continue
        abs_diffs.append(abs(ms - ds))
    mae = (sum(abs_diffs) / len(abs_diffs)) if abs_diffs else None

    return {
        "seg": seg.get("name", ""),
        "metra_total": metra_total,
        "metra_mono": metra_mono,
        "msdial_total": msdial_total,
        "msdial_mono": msdial_mono,
        "matched": len(matched),
        "matched_msdial_mono": matched_msdial_mono,
        "coverage_recall": coverage_recall,
        "net_add": net_add,
        "net_add_degraded": net_add_degraded,
        "mae": mae,
        "mae_n": len(abs_diffs),
        "mae_abs_sum": sum(abs_diffs),
    }


def build_segment(name: str, metra: list[dict], msdial: list[dict],
                  mz_tol: float, rt_tol: float) -> dict:
    """Match ``metra`` vs ``msdial`` and pack a segment bundle for ``report``."""
    matched, msdial_only, metra_only = match_features(metra, msdial, mz_tol, rt_tol)
    return {
        "name": name,
        "metra": metra,
        "msdial": msdial,
        "matched": matched,
        "msdial_only": msdial_only,
        "metra_only": metra_only,
    }


def aggregate(reports: list[dict]) -> dict:
    """Aggregate per-segment reports into a TOTAL row (ratios recomputed)."""
    metra_total = sum(r["metra_total"] for r in reports)
    metra_mono = sum(r["metra_mono"] for r in reports)
    msdial_total = sum(r["msdial_total"] for r in reports)
    msdial_mono = sum(r["msdial_mono"] for r in reports)
    matched_msdial_mono = sum(r["matched_msdial_mono"] for r in reports)
    net_add = sum(r["net_add"] for r in reports)
    mae_n = sum(r["mae_n"] for r in reports)
    mae_abs_sum = sum(r["mae_abs_sum"] for r in reports)
    return {
        "seg": "TOTAL",
        "metra_total": metra_total,
        "metra_mono": metra_mono,
        "msdial_total": msdial_total,
        "msdial_mono": msdial_mono,
        "matched": sum(r["matched"] for r in reports),
        "matched_msdial_mono": matched_msdial_mono,
        "coverage_recall": (matched_msdial_mono / msdial_mono) if msdial_mono else None,
        "net_add": net_add,
        "net_add_degraded": any(r["net_add_degraded"] for r in reports),
        "mae": (mae_abs_sum / mae_n) if mae_n else None,
        "mae_n": mae_n,
        "mae_abs_sum": mae_abs_sum,
    }


# --------------------------------------------------------------------------- #
# Formatting / CLI
# --------------------------------------------------------------------------- #
def _fmt_pct(x: Optional[float]) -> str:
    return f"{x * 100:.1f}%" if x is not None else "N/A"


def _fmt_mae(r: dict) -> str:
    return f"{r['mae']:.4f} (n={r['mae_n']})" if r["mae"] is not None else f"N/A (n={r['mae_n']})"


def _table_rows(reports: list[dict]) -> list[list[str]]:
    rows = []
    for r in reports:
        net_add = f"{r['net_add']}{'*' if r['net_add_degraded'] else ''}"
        rows.append([
            str(r["seg"]),
            str(r["metra_total"]),
            str(r["metra_mono"]),
            str(r["msdial_total"]),
            str(r["msdial_mono"]),
            _fmt_pct(r["coverage_recall"]),
            net_add,
            _fmt_mae(r),
        ])
    return rows


HEADER = ["Seg", "METRA total", "METRA mono", "MS-DIAL total",
          "MS-DIAL mono", "Coverage recall", "Net-add", "MAE"]


def _print_table(reports: list[dict]) -> None:
    all_rows = [HEADER] + _table_rows(reports)
    widths = [max(len(row[i]) for row in all_rows) for i in range(len(HEADER))]
    for ri, row in enumerate(all_rows):
        line = "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        print(line)
        if ri == 0:
            print("  ".join("-" * widths[i] for i in range(len(HEADER))))


def _write_markdown(reports: list[dict], out_path, mz_tol: float,
                    rt_tol: float) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    degraded = any(r["net_add_degraded"] for r in reports)
    lines = [
        "# MS-DIAL vs METRA - ASFAM feature comparison",
        "",
        f"- Tolerances: m/z +/-{mz_tol}, RT +/-{rt_tol} min",
        "- Coverage recall = matched MS-DIAL **mono** (Isotope==0) / MS-DIAL mono.",
        "- Net-add = METRA-only features with an MS2-driven origin "
        "(`detection_source` in {ms2_driven, both}).",
        "- MAE = mean |METRA.total_score - MS-DIAL.total_score| over "
        "m/z+RT-matched pairs annotating the **same compound** "
        "(InChIKey when both present, else name); `n` is the pair count.",
        "",
        "| " + " | ".join(HEADER) + " |",
        "|" + "|".join(["---"] * len(HEADER)) + "|",
    ]
    for row in _table_rows(reports):
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    if degraded:
        lines.append(
            "> `*` Net-add degraded: this segment's `features.csv` lacks the "
            "`detection_source` column (old export), so Net-add counts **all** "
            "METRA-only features, not just MS2-driven ones."
        )
        lines.append("")
    lines.append(
        "> Note: MS-DIAL \"Total score\" folds in precursor-mass / RT terms "
        "whereas METRA's `total_score` is spectral-only, so the MAE carries an "
        "expected systematic offset (PR-E interpretation matter, not a tool "
        "bug)."
    )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Recall mode helpers  (--ours)
# --------------------------------------------------------------------------- #
# Operating point: dynamic m/z tolerance + RT tolerance (mirrors the session
# analysis in test_results/asfam/compare_msdial_vs_ours.py).
_MZ_FLOOR: float = 0.006   # Da – minimum m/z tolerance regardless of ppm
_PPM: float = 25.0          # ppm component: tol = max(_MZ_FLOOR, mz * ppm * 1e-6)
_RT_TOL: float = 0.2        # min

# Sentinel: the single worst missed peak from this session (m/z 801.20964,
# RT 3.919 min, Height ~1.7M; annotated as a flavone glycoside).  Not recovered
# by METRA at the time the measurer was written; the flag turns True once fixed.
_SENTINEL_MZ: float = 801.20964
_SENTINEL_RT: float = 3.919
_SENTINEL_MZ_TOL: float = 0.01
_SENTINEL_RT_TOL: float = 0.30


def _is_recall_annotated(name: str) -> bool:
    """Annotated predicate for the recall-mode MS-DIAL denominator.

    Matches the reference script (test_results/asfam/compare_msdial_vs_ours.py)
    exactly: exact-match exclusion of a small sentinel set + ``low score:``
    prefix.  Intentionally NOT case-folded — ``"Unknown compound"`` etc. DO
    count as annotated (the 10-peak difference vs ``_is_confident_name``).
    """
    n = _to_str(name)
    if n in ("Unknown", "", "null", "no result"):
        return False
    if n.startswith("low score:"):
        return False
    return True


def dynamic_mz_tol(mz: float, ppm: float = _PPM,
                   floor: float = _MZ_FLOOR) -> float:
    """Return max(floor, mz * ppm * 1e-6).  Tolerance grows linearly with m/z."""
    return max(floor, mz * ppm * 1e-6)


def recall_match(
    msdial: list[dict],
    ours: list[dict],
    ppm: float = _PPM,
    mz_floor: float = _MZ_FLOOR,
    rt_tol: float = _RT_TOL,
) -> list[bool]:
    """Non-greedy recall match: for each MS-DIAL peak, is >=1 METRA feature
    within tolerance?

    Unlike ``match_features`` (greedy 1:1), multiple METRA features can claim
    the same MS-DIAL peak.  Returns a bool list parallel to *msdial*.
    """
    ours_mz = np.array(
        [f["mz"] if f["mz"] is not None else float("nan") for f in ours],
        dtype=float,
    )
    ours_rt = np.array(
        [f["rt"] if f["rt"] is not None else float("nan") for f in ours],
        dtype=float,
    )
    order = np.argsort(ours_mz)
    ours_mz_s = ours_mz[order]
    ours_rt_s = ours_rt[order]

    matched: list[bool] = []
    for p in msdial:
        mz = p.get("mz")
        rt = p.get("rt")
        if mz is None or rt is None or not math.isfinite(mz):
            matched.append(False)
            continue
        tol = dynamic_mz_tol(mz, ppm, mz_floor)
        lo = int(np.searchsorted(ours_mz_s, mz - tol, "left"))
        hi = int(np.searchsorted(ours_mz_s, mz + tol, "right"))
        if hi <= lo:
            matched.append(False)
            continue
        cand_rt = ours_rt_s[lo:hi]
        matched.append(bool(np.any(np.abs(cand_rt - rt) <= rt_tol)))
    return matched


def reverse_recall_match(
    ours: list[dict],
    msdial: list[dict],
    ppm: float = _PPM,
    mz_floor: float = _MZ_FLOOR,
    rt_tol: float = _RT_TOL,
) -> list[bool]:
    """Non-greedy reverse match: for each METRA feature, is >=1 MS-DIAL peak
    within tolerance?  Returns a bool list parallel to *ours*."""
    md_mz = np.array(
        [p["mz"] if p["mz"] is not None else float("nan") for p in msdial],
        dtype=float,
    )
    md_rt = np.array(
        [p["rt"] if p["rt"] is not None else float("nan") for p in msdial],
        dtype=float,
    )
    order = np.argsort(md_mz)
    md_mz_s = md_mz[order]
    md_rt_s = md_rt[order]

    matched: list[bool] = []
    for f in ours:
        mz = f.get("mz")
        rt = f.get("rt")
        if mz is None or rt is None or not math.isfinite(mz):
            matched.append(False)
            continue
        tol = dynamic_mz_tol(mz, ppm, mz_floor)
        lo = int(np.searchsorted(md_mz_s, mz - tol, "left"))
        hi = int(np.searchsorted(md_mz_s, mz + tol, "right"))
        if hi <= lo:
            matched.append(False)
            continue
        matched.append(bool(np.any(np.abs(md_rt_s[lo:hi] - rt) <= rt_tol)))
    return matched


def _is_dup(rec: dict) -> bool:
    """Whether a METRA feature is flagged ``is_duplicate`` (bool / np.bool_ / str)."""
    v = rec.get("is_duplicate")
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


def _subset_recall(label: str, peaks: list[dict], flags: list[bool]) -> dict:
    """Recall stats (all / mono / annotated) for one subset of MS-DIAL peaks.

    ``flags`` is the recall bool list parallel to ``peaks`` (from
    ``recall_match``). Used uniformly for per-segment, per-m/z-bin, and TOTAL
    rows so the three never drift. ``label`` lands in the ``seg`` field the
    table printers read.
    """
    n_total = len(peaks)
    n_matched_all = sum(flags)
    mono = [p.get("isotope", 0) == 0 for p in peaks]
    n_mono = sum(mono)
    n_mono_matched = sum(ok for ok, mo in zip(flags, mono) if mo)
    ann = [_is_recall_annotated(p.get("name", "")) for p in peaks]
    n_ann = sum(ann)
    n_ann_matched = sum(ok for ok, an in zip(flags, ann) if an)
    return {
        "seg": label,
        "n_total": n_total,
        "n_matched_all": n_matched_all,
        "recall_all": n_matched_all / n_total if n_total else None,
        "n_mono": n_mono,
        "n_matched_mono": n_mono_matched,
        "recall_mono": n_mono_matched / n_mono if n_mono else None,
        "n_ann": n_ann,
        "n_matched_ann": n_ann_matched,
        "recall_annotated": n_ann_matched / n_ann if n_ann else None,
    }


def mz_bin_recall(msdial: list[dict], matched: list[bool],
                  bin_width: float = 100.0) -> list[dict]:
    """Per-m/z-bin recall reports, ascending by bin lower edge.

    Each MS-DIAL peak is assigned to the bin ``[k*bin_width, (k+1)*bin_width)``
    by its m/z; peaks with no / non-finite m/z are skipped. Reuses
    ``_subset_recall`` so each bin row carries the same all/mono/annotated
    triple as segment / TOTAL rows. The ``seg`` label is ``"<lo>-<hi>"``
    (e.g. ``"700-800"``).
    """
    bins: dict[int, list[int]] = {}
    for i, p in enumerate(msdial):
        mz = p.get("mz")
        if mz is None or not math.isfinite(mz):
            continue
        k = int(mz // bin_width)
        bins.setdefault(k, []).append(i)
    reports: list[dict] = []
    for k in sorted(bins):
        idx = bins[k]
        lo = int(k * bin_width)
        hi = int((k + 1) * bin_width)
        peaks = [msdial[i] for i in idx]
        flags = [matched[i] for i in idx]
        reports.append(_subset_recall(f"{lo}-{hi}", peaks, flags))
    return reports


def _check_sentinel(
    msdial_all: list[dict], matched: list[bool]
) -> Optional[bool]:
    """Return True/False if sentinel is/is-not matched; None if not in MS-DIAL."""
    for p, ok in zip(msdial_all, matched):
        mz = p.get("mz")
        rt = p.get("rt")
        if mz is None or rt is None:
            continue
        if (
            abs(mz - _SENTINEL_MZ) <= _SENTINEL_MZ_TOL
            and abs(rt - _SENTINEL_RT) <= _SENTINEL_RT_TOL
        ):
            return ok
    return None


def _write_missed_csv(missed: list[dict], out_path) -> None:
    """Write missed MS-DIAL peaks (height-descending) to *out_path*."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    missed_sorted = sorted(missed, key=lambda p: -(p.get("height") or 0.0))
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["seg", "name", "rt_min", "mz", "height", "sn", "isotope", "annotated"]
        )
        for p in missed_sorted:
            w.writerow([
                p.get("seg", ""),
                p.get("name", ""),
                f"{p['rt']:.3f}" if p.get("rt") is not None else "",
                f"{p['mz']:.5f}" if p.get("mz") is not None else "",
                f"{p['height']:.0f}" if p.get("height") is not None else "",
                f"{p['sn']:.1f}" if p.get("sn") is not None else "",
                str(p.get("isotope", "")),
                str(_is_recall_annotated(p.get("name", ""))),
            ])


def _write_bin_csv(reports: list[dict], out_path) -> None:
    """Write the per-m/z-bin recall table (incl. TOTAL row) to *out_path*."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["mz_bin", "msdial_peaks", "matched", "recall_pct",
                    "n_annotated", "matched_annotated", "recall_annotated_pct"])
        for r in reports:
            ra = r["recall_all"]
            raa = r["recall_annotated"]
            w.writerow([
                r["seg"], r["n_total"], r["n_matched_all"],
                f"{ra * 100:.1f}" if ra is not None else "",
                r["n_ann"], r["n_matched_ann"],
                f"{raa * 100:.1f}" if raa is not None else "",
            ])


_RECALL_HEADER = [
    "Seg", "MS-DIAL peaks",
    "recall_all", "recall_mono", "recall_annotated",
]


def _recall_table_rows(seg_reports: list[dict]) -> list[list[str]]:
    rows = []
    for r in seg_reports:
        rows.append([
            str(r["seg"]),
            str(r["n_total"]),
            f"{r['n_matched_all']}/{r['n_total']} ({_fmt_pct(r['recall_all'])})",
            f"{r['n_matched_mono']}/{r['n_mono']} ({_fmt_pct(r['recall_mono'])})",
            f"{r['n_matched_ann']}/{r['n_ann']} ({_fmt_pct(r['recall_annotated'])})",
        ])
    return rows


def _print_recall_table(seg_reports: list[dict], first_label: str = "Seg") -> None:
    hdr = [first_label] + _RECALL_HEADER[1:]
    all_rows = [hdr] + _recall_table_rows(seg_reports)
    widths = [max(len(row[i]) for row in all_rows) for i in range(len(hdr))]
    for ri, row in enumerate(all_rows):
        line = "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        print(line)
        if ri == 0:
            print("  ".join("-" * widths[i] for i in range(len(hdr))))


def _main_recall(args) -> int:
    """Recall mode (--ours): measure METRA recall vs MS-DIAL with dynamic m/z tol.

    Two MS-DIAL sources, same recall engine:
      * default: per-segment raw ``.txt`` exports under ``--msdial`` (parsed by
        column index; reported per SEGMENT).
      * ``--msdial-csv``: a single integrated, named-column CSV (e.g.
        ``ASFAM-3.csv``, deduped single-isotope distinct set); reported per
        **m/z bin** -- the high-m/z recall question.
    """
    ours_dir = Path(args.ours)

    # Load METRA features (single flat features.csv)
    ours_csv = ours_dir / "features.csv"
    if not ours_csv.exists():
        print(f"[error] METRA features.csv not found: {ours_csv}")
        return 1
    ours = parse_metra_csv(ours_csv)

    # Load MS-DIAL peaks -- integrated named-column CSV, or per-segment txt
    use_bins = bool(args.msdial_csv)
    msdial_all: list[dict] = []
    if use_bins:
        csv_path = Path(args.msdial_csv)
        if not csv_path.exists():
            print(f"[error] MS-DIAL integrated CSV not found: {csv_path}")
            return 1
        msdial_all = parse_msdial_csv(csv_path)
    else:
        msdial_dir = Path(args.msdial)
        for seg in SEGMENTS:
            txt = msdial_dir / f"RL_ASFAM_{seg}_P_3.txt"
            if not txt.exists():
                print(f"[warn] missing MS-DIAL file: {txt} -- skipping seg {seg}")
                continue
            for p in parse_msdial_txt(txt):
                p["seg"] = seg
                msdial_all.append(p)

    if not msdial_all:
        print("[error] no MS-DIAL peaks loaded; check --msdial / --msdial-csv path")
        return 1

    # Recall match: for each MS-DIAL peak, is >=1 METRA feature within tol?
    # Forward direction uses ALL of ours (isotope copies / dups are valid
    # candidates -- more candidates only help recall, per the handoff).
    matched = recall_match(msdial_all, ours)

    # Reverse match: METRA features with no MS-DIAL partner. Against the
    # integrated CSV (deduped, single-isotope) the fair reverse denominator is
    # ours filtered to mono & non-duplicate, else isotope/dup copies inflate the
    # "extra" count asymmetrically.
    if use_bins:
        ours_rev = [f for f in ours
                    if f.get("isotope", 0) == 0 and not _is_dup(f)]
    else:
        ours_rev = ours
    rev_matched = reverse_recall_match(ours_rev, msdial_all)
    n_extra = sum(1 for ok in rev_matched if not ok)

    # Sentinel check (m/z 801.21 @ 3.919; in-range for the full sweep too)
    sentinel_ok = _check_sentinel(msdial_all, matched)
    sentinel_str = (
        "YES" if sentinel_ok is True
        else "no" if sentinel_ok is False
        else "NOT_FOUND_IN_MSDIAL"
    )

    # Subset reports: per m/z bin (CSV mode) or per segment (txt mode).
    if use_bins:
        seg_reports = mz_bin_recall(msdial_all, matched, args.mz_bin_width)
        first_label = "m/z bin"
    else:
        seg_reports = []
        for seg in SEGMENTS:
            idx = [i for i, p in enumerate(msdial_all) if p.get("seg") == seg]
            if not idx:
                continue
            seg_reports.append(_subset_recall(
                seg, [msdial_all[i] for i in idx], [matched[i] for i in idx]))
        first_label = "Seg"

    total_row = _subset_recall("TOTAL", msdial_all, matched)

    # One-line summary
    print(
        f"recall={_fmt_pct(total_row['recall_all'])}"
        f" ({total_row['n_matched_all']}/{total_row['n_total']})"
        f"  recall_mono={_fmt_pct(total_row['recall_mono'])}"
        f"  recall_annotated={_fmt_pct(total_row['recall_annotated'])}"
        f"  reverse_extra={n_extra}/{len(ours_rev)}"
        f"  m/z_801.21@3.919_recovered={sentinel_str}"
    )
    print()
    _print_recall_table(seg_reports + [total_row], first_label=first_label)

    # Write missed peaks CSV (height-descending)
    missed_peaks = [p for p, ok in zip(msdial_all, matched) if not ok]
    missed_csv = ours_dir / "missed_by_msdial.csv"
    _write_missed_csv(missed_peaks, missed_csv)
    print(f"\nwrote missed_by_msdial.csv -> {missed_csv}  ({len(missed_peaks)} peaks)")

    # In bin mode, also dump the per-bin recall table as a CSV artifact
    if use_bins:
        bin_csv = ours_dir / "recall_by_mz_bin.csv"
        _write_bin_csv(seg_reports + [total_row], bin_csv)
        print(f"wrote recall_by_mz_bin.csv -> {bin_csv}  ({len(seg_reports)} bins)")

    return 0


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Compare METRA ASFAM features.csv against MS-DIAL .txt exports.")
    # ---- recall mode (new: --ours) ----
    p.add_argument("--ours",
                   help="METRA results dir containing features.csv  [recall mode]")
    p.add_argument("--msdial-csv",
                   help="integrated MS-DIAL CSV with named columns "
                        "(e.g. ASFAM-3.csv); recall reported per m/z bin "
                        "instead of per segment  [recall mode]")
    p.add_argument("--mz-bin-width", type=float, default=100.0,
                   help="m/z bin width for the --msdial-csv recall table "
                        "(default 100)")
    # ---- shared ----
    p.add_argument("--msdial", default="MS_DIAL_results/asfam/val_260624",
                   help="MS-DIAL results dir (default: MS_DIAL_results/asfam/val_260624)")
    # ---- legacy mode (--metra) ----
    p.add_argument("--metra",
                   help="METRA results dir containing <seg>/features.csv  [legacy mode]")
    p.add_argument("--out", help="output Markdown report path  [legacy mode]")
    p.add_argument("--mz-tol", type=float, default=0.01,
                   help="m/z tolerance [legacy mode] (default 0.01)")
    p.add_argument("--rt-tol", type=float, default=0.1,
                   help="RT tolerance min [legacy mode] (default 0.1)")
    args = p.parse_args(argv)

    if args.msdial_csv and not args.ours:
        p.error("--msdial-csv requires --ours (recall mode)")
    if args.ours:
        return _main_recall(args)

    if not args.metra:
        p.error("one of --ours (recall mode) or --metra (legacy mode) is required")
    if not args.out:
        p.error("--out is required in legacy --metra mode")

    # ---- legacy mode body ----
    metra_dir, msdial_dir = Path(args.metra), Path(args.msdial)
    reports: list[dict] = []
    for seg in SEGMENTS:
        metra_csv = metra_dir / seg / "features.csv"
        msdial_txt = msdial_dir / f"RL_ASFAM_{seg}_P_3.txt"
        if not metra_csv.exists():
            print(f"[warn] missing METRA csv: {metra_csv} -- skipping seg {seg}")
            continue
        if not msdial_txt.exists():
            print(f"[warn] missing MS-DIAL txt: {msdial_txt} -- skipping seg {seg}")
            continue
        metra = parse_metra_csv(metra_csv)
        msdial = parse_msdial_txt(msdial_txt)
        seg_bundle = build_segment(seg, metra, msdial, args.mz_tol, args.rt_tol)
        reports.append(report(seg_bundle))

    if not reports:
        print("[error] no segments processed (no METRA csv found). Nothing written.")
        return 1

    rows_for_print = reports + [aggregate(reports)]
    _print_table(rows_for_print)
    _write_markdown(rows_for_print, args.out, args.mz_tol, args.rt_tol)
    print(f"\nReport written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
