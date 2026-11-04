"""Leave-one-out false-hit analysis of the ASFAMProcessor library matcher on public
spectral libraries (work package S2.1f, Task A; reviewer points R4-1 and R2-2 iv/v).

Reviewer 4 ran the public code on MassBank in leave-one-out mode and reported that,
with the hard-coded +/-0.5 Da precursor candidate window, 3.8 % of the queries
obtained a high-scoring hit of a *different molecular formula* (0.5 % at 0.01 Da).
This script repeats that experiment with the production matcher
(``metabo_core.annotation.match_feature_topn``) and the production ASFAM settings
(``ProcessingConfig().annotation_view()`` / ``similarity_view()``: emit threshold 0,
min matched peaks 1, min peaks to match 1, Matched% >= 0.25, WDP >= 0.10,
fragment tolerance 0.02 Da, RT unused) and the manuscript's high-confidence rule
(Total score >= 1.4 and >= 3 matched peaks) at precursor windows 0.5 / 0.05 / 0.01 Da.
The public tag v1.2.260721 carries the same settings with the window fixed at 0.5 Da.

Procedure
  1. Load one or more libraries (NIST-format MSP or MGF) with the platform parser
     ``metabo_core.io.spectral_library.read_msp`` / ``read_mgf``.
  2. Keep positive-mode MS2 spectra with a precursor m/z, a parsable molecular
     formula and >= 3 peaks (optionally only records whose precursor m/z carries
     >= ``--min-precursor-decimals`` decimals, i.e. high-resolution precursors).
     The compound key is the InChIKey first block (14 characters) when available,
     otherwise ``formula + name``. GNPS MGF records carry no formula / InChIKey: the
     formula comes from the InChI formula layer and the InChIKey is computed with
     RDKit from the InChI (fallback SMILES); the adduct is read from the record name.
  3. Build the m/z index once over the retained set (= candidate set = query set).
  4. For every query, run the matcher once with the 0.5 Da window and keep every
     surviving hit (top_n unbounded). Compound removal is done afterwards by masking
     on the compound key, and the narrower windows by post-filtering on
     |reference precursor - query precursor| <= window. Both steps are exact: the
     window is a pure candidate filter that never changes a score (see
     ``test_precursor_window.py``) and the hit list is score-sorted; the equivalence
     is re-checked on a random subset by calling the matcher with explicit windows.
  5. Classification of a high-scoring hit (score >= 1.4, n_matched >= 3) that is not
     the query compound:
       * different *neutral* formula  (Reviewer 4's reading of "different formula")
       * different *ion* formula      (neutral formula + adduct, e.g. [M+H]+ of a
         C21H20O10 flavone glycoside and [M]+ of a [C21H21O10]+ anthocyanin are the
         same ion C21H21O10+; without a parsable adduct the neutral formula is used)
     Two removal protocols are reported: "compound LOO" removes every spectrum of the
     query compound (top-1 of the rest; also "any hit in window"), and "literal LOO"
     removes only the query spectrum itself (top-1; also any of the top 5, the
     production top_n).

Outputs (``--out-dir``, default analysis/decoy/)
  loo_<tag>.csv                        one row per query x window
  loo_summary_<tag>.csv                window x rates (all variants)
  loo_false_hits_examples_<tag>.csv    the 20 highest-scoring compound-LOO false hits
                                       (different neutral formula)
  numbers_loo_<tag>.json               numbers, provenance (SHA-256, release tag), runtime

Run from anywhere; pass paths as arguments (downloaded libraries are data, never code):
  python library_loo.py --library <MassBank_NISTformat.msp> --tag massbank --release-tag 2026.03
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import hashlib
import json
import platform
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from metabo_core import __version__ as CORE_VERSION
from metabo_core.annotation import build_index_from_list, match_feature_topn
from metabo_core.config import AnnotationConfig, SimilarityConfig
from metabo_core.io.spectral_library import read_mgf, read_msp
from metabo_core.models import CandidateFeature

HIGH_SCORE = 1.4
HIGH_MIN_MATCHED = 3
PRODUCTION_TOP_N = 5
WINDOWS_DEFAULT = (0.5, 0.05, 0.01)

# Metadata keys kept by the lean parser (exact-key match; the parser lower-cases the
# keys it does not map explicitly).
KEEP_META = {
    "name", "precursor_mz", "formula", "adduct", "rt", "inchikey", "inchi", "smiles",
    "ion_mode", "ionmode", "spectrum_type", "mslevel", "db#", "spectrumid", "charge",
    "libraryquality", "instrument_type", "comment",
}

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
_FORMULA_FULL = re.compile(r"(?:[A-Z][a-z]?\d*)+")
_ADDUCT = re.compile(r"\[(.+?)\](\d*)([+-]*)\*?$")
_ADDUCT_TERM = re.compile(r"([+-]?)(\d*)([A-Za-z][A-Za-z0-9]*)")
_NAME_ADDUCT = re.compile(r"(?:^|[\s,;])\[?(M(?:[+-][A-Za-z0-9]+)*)\]?\s*(\d*[+-]?)\s*$")
ADDUCT_ALIASES = {
    "ACN": "C2H3N", "MeOH": "CH4O", "CH3OH": "CH4O", "FA": "CH2O2", "HCOOH": "CH2O2",
    "HAc": "C2H4O2", "CH3COOH": "C2H4O2", "DMSO": "C2H6OS", "IsoProp": "C3H8O",
    "iPrOH": "C3H8O", "HCOO": "CHO2", "CH3COO": "C2H3O2", "TFA": "C2HF3O2",
}


def production_settings() -> tuple[AnnotationConfig, SimilarityConfig, dict]:
    """The ASFAM production matcher settings, taken from the app config when importable."""
    source = "asfam.config.ProcessingConfig"
    try:
        from asfam.config import ProcessingConfig  # app code, read-only use
        cfg = ProcessingConfig()
        ann = cfg.annotation_view()
        sim = cfg.similarity_view()
    except Exception as exc:  # pragma: no cover - fallback for a bare metabo_core install
        source = f"fallback constants ({exc.__class__.__name__})"
        ann = AnnotationConfig(similarity_threshold=0.0, min_matched_peaks=1, min_peaks_to_match=1,
                               min_matched_pct=0.25, min_wdp=0.10, use_rt=False,
                               precursor_window_da=0.5)
        sim = SimilarityConfig(mz_tolerance=0.02, ms1_tolerance=0.01, use_rt=False)
    expected = dict(similarity_threshold=0.0, min_matched_peaks=1, min_peaks_to_match=1,
                    min_matched_pct=0.25, min_wdp=0.10, use_rt=False)
    actual = {k: getattr(ann, k) for k in expected}
    if actual != expected or sim.mz_tolerance != 0.02 or sim.ms1_tolerance != 0.01:
        raise SystemExit(f"production settings differ from the documented ones: {actual}, "
                         f"mz_tol={sim.mz_tolerance}, ms1_tol={sim.ms1_tolerance}")
    record = {"source": source, **actual, "mz_tolerance": sim.mz_tolerance,
              "ms1_tolerance": sim.ms1_tolerance, "use_rt_similarity": sim.use_rt,
              "high_score_threshold": HIGH_SCORE, "high_min_matched_peaks": HIGH_MIN_MATCHED,
              "production_top_n": PRODUCTION_TOP_N}
    return ann, sim, record


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- formulas and adducts ------------------------------------------------------------

def formula_counts(formula: str | None) -> dict[str, int] | None:
    """Element counts of a molecular formula (brackets / trailing charge stripped)."""
    if formula is None:
        return None
    f = str(formula).strip().replace(" ", "")
    f = f.replace("[", "").replace("]", "")
    f = re.sub(r"\^?\d*[+-]+\d*$", "", f)  # trailing charge such as '+', '2+', '+1'
    if not f or not _FORMULA_FULL.fullmatch(f):
        return None
    counts: dict[str, int] = {}
    for el, n in _FORMULA_TOKEN.findall(f):
        counts[el] = counts.get(el, 0) + (int(n) if n else 1)
    return counts


def counts_key(counts: dict[str, int] | None) -> str | None:
    if not counts:
        return None
    return "".join(f"{el}{counts[el]}" for el in sorted(counts) if counts[el] > 0) or None


def formula_key(formula: str | None) -> str | None:
    return counts_key(formula_counts(formula))


def ion_formula_key(neutral_counts: dict[str, int] | None, adduct: str) -> str | None:
    """Formula of the precursor ion, e.g. C9H17NO3 + '[M+H]+' -> C9H18NO3; None if unparsable.

    Electrons are ignored, so '[M]+' and '[M]+*' return the neutral formula itself."""
    if not neutral_counts or not adduct:
        return None
    m = _ADDUCT.match(adduct.strip())
    if not m:
        return None
    counts: dict[str, int] = {}
    mult_m = 0
    for sign, num, term in _ADDUCT_TERM.findall(m.group(1)):
        n = int(num) if num else 1
        sgn = -1 if sign == "-" else 1
        if term == "M":
            mult_m += sgn * n
            continue
        c = formula_counts(ADDUCT_ALIASES.get(term, term))
        if c is None:
            return None
        for el, k in c.items():
            counts[el] = counts.get(el, 0) + sgn * n * k
    if mult_m <= 0:
        return None
    for el, k in neutral_counts.items():
        counts[el] = counts.get(el, 0) + mult_m * k
    if any(v < 0 for v in counts.values()):
        return None
    return counts_key(counts)


def adduct_from_name(name: str, positive: bool) -> str:
    """GNPS-style names end with the adduct ('... M+H', '... M+2H', '... [M+Na]+')."""
    m = _NAME_ADDUCT.search(name or "")
    if not m:
        return ""
    body, charge = m.group(1), m.group(2)
    if not charge or charge[-1] not in "+-":
        charge = (charge or "") + ("+" if positive else "-")
    return f"[{body}]{charge}"


def inchi_formula(inchi: str | None) -> str | None:
    if not inchi or not str(inchi).startswith("InChI="):
        return None
    layers = str(inchi).split("/")
    return layers[1] if len(layers) >= 2 else None


def rdkit_inchikey_block(inchi: str | None, smiles: str | None):
    """InChIKey first block (and neutral formula) from InChI or SMILES via RDKit."""
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
    except Exception:
        return None, None
    mol = None
    if inchi and str(inchi).startswith("InChI="):
        try:
            mol = Chem.MolFromInchi(str(inchi))
        except Exception:
            mol = None
    if mol is None and smiles and str(smiles) not in ("N/A", "", "nan"):
        try:
            mol = Chem.MolFromSmiles(str(smiles))
        except Exception:
            mol = None
    if mol is None:
        return None, None
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None, None
    formula = None
    try:
        from rdkit.Chem import rdMolDescriptors
        formula = rdMolDescriptors.CalcMolFormula(mol)
        m = re.search(r"([+-])(\d*)$", formula)
        if m:  # the SMILES was the ion: remove the protons that carry the charge
            charge = int(m.group(2) or 1) * (1 if m.group(1) == "+" else -1)
            counts = formula_counts(formula)
            if counts is not None and "H" in counts:
                counts["H"] -= charge
                formula = counts_key(counts)
    except Exception:
        formula = None
    return (key[:14] if key and len(key) >= 14 else None), formula


# ---- record filters ---------------------------------------------------------------------

def is_positive(meta: dict) -> bool:
    mode = str(meta.get("ion_mode") or meta.get("ionmode") or "").strip().lower()
    if mode:
        return mode.startswith("p")
    adduct = str(meta.get("adduct") or "")
    charge = str(meta.get("charge") or "")
    return adduct.endswith("+") or charge.endswith("+") or charge in ("1", "2")


def is_ms2(meta: dict) -> bool:
    st = str(meta.get("spectrum_type") or "").strip().upper()
    if st:
        return st == "MS2"
    lvl = str(meta.get("mslevel") or "").strip()
    if lvl:
        return lvl == "2"
    return True  # MGF libraries without MSLEVEL are MS/MS libraries by construction


def precursor_decimals(pmz: float) -> int:
    """Number of decimals the precursor carries (<= 2 flags a low-resolution record)."""
    for d in range(0, 6):
        if abs(round(pmz, d) - pmz) < 1e-7:
            return d
    return 6


def load_library(path: Path, source_tag: str, use_rdkit: bool, min_decimals: int,
                 max_peaks: int = 0) -> tuple[list[dict], dict]:
    """Parse one library file and return the retained lean spectra plus filter counts."""
    t0 = time.time()
    if path.suffix.lower() == ".msp":
        spectra = read_msp(str(path), keep_metadata=set(KEEP_META), as_arrays=True)
    elif path.suffix.lower() == ".mgf":
        spectra = read_mgf(str(path), keep_metadata=set(KEEP_META), as_arrays=True)
    else:
        raise SystemExit(f"unsupported library format: {path}")
    counts = Counter(total=len(spectra))
    kept: list[dict] = []
    for spec in spectra:
        meta = spec["metadata"]
        positive = is_positive(meta)
        if not positive:
            counts["drop_not_positive"] += 1
            continue
        if not is_ms2(meta):
            counts["drop_not_ms2"] += 1
            continue
        pmz = meta.get("precursor_mz")
        if pmz is None or not (float(pmz) > 0):
            counts["drop_no_precursor"] += 1
            continue
        pmz = float(pmz)
        if spec["mz"].size < 3:
            counts["drop_lt3_peaks"] += 1
            continue
        if max_peaks and spec["mz"].size > max_peaks:
            counts["drop_gt_max_peaks"] += 1
            continue
        if float(spec["intensity"].max()) <= 0:
            counts["drop_zero_intensity"] += 1
            continue
        decimals = precursor_decimals(pmz)
        if decimals < min_decimals:
            counts["drop_lowres_precursor"] += 1
            continue
        formula = meta.get("formula")
        inchikey = str(meta.get("inchikey") or "").strip()
        key_block = inchikey[:14] if re.fullmatch(r"[A-Z]{14}-[A-Z]{8,10}-[A-Z]", inchikey) else None
        if formula_key(formula) is None:
            formula = inchi_formula(meta.get("inchi"))
        if (key_block is None or formula_key(formula) is None) and use_rdkit:
            rd_key, rd_formula = rdkit_inchikey_block(meta.get("inchi"), meta.get("smiles"))
            if rd_key is not None:
                counts["rdkit_structures_used"] += 1
            key_block = key_block or rd_key
            if formula_key(formula) is None:
                formula = rd_formula
        fcounts = formula_counts(formula)
        fkey = counts_key(fcounts)
        if fkey is None:
            counts["drop_no_formula"] += 1
            continue
        name = str(meta.get("name") or "")
        adduct = str(meta.get("adduct") or "").strip()
        if not adduct:
            adduct = adduct_from_name(name, positive)
            if adduct:
                counts["adduct_from_name"] += 1
        ion_key = ion_formula_key(fcounts, adduct)
        if ion_key is None:
            counts["no_ion_formula"] += 1
        if key_block is None:
            key = f"F:{fkey}|N:{name.strip().lower()}"
            counts["key_from_formula_name"] += 1
        else:
            key = key_block
            counts["key_from_inchikey"] += 1
        rec_id = meta.get("db#") or meta.get("spectrumid") or ""
        spec["_loo"] = {
            "source": source_tag, "id": str(rec_id), "name": name, "formula": str(formula),
            "formula_key": fkey, "ion_key": ion_key, "key": key, "pmz": pmz, "adduct": adduct,
            "decimals": decimals, "instrument_type": str(meta.get("instrument_type") or ""),
            "quality": str(meta.get("libraryquality") or ""),
        }
        kept.append(spec)
    counts["kept"] = len(kept)
    counts["parse_seconds"] = round(time.time() - t0, 1)
    print(f"[{source_tag}] {dict(counts)}", flush=True)
    return kept, dict(counts)


def make_feature(i: int, spec: dict) -> CandidateFeature:
    pmz = spec["_loo"]["pmz"]
    feat = CandidateFeature(
        feature_id=f"Q{i}", segment_name="", replicate_id=1,
        precursor_mz_nominal=int(round(pmz)), rt_apex=0.0, rt_left=0.0, rt_right=0.0,
        ms2_mz=np.asarray(spec["mz"], dtype=np.float64),
        ms2_intensity=np.asarray(spec["intensity"], dtype=np.float64),
        n_fragments=int(spec["mz"].size),
    )
    feat.ms1_precursor_mz = pmz
    return feat


# ---- classification -----------------------------------------------------------------------

INFO: list[dict] = []  # per spectrum index: the ``_loo`` record


def is_high(m) -> bool:
    return m.score >= HIGH_SCORE and m.n_matched >= HIGH_MIN_MATCHED


def diff_neutral(q: dict, h: dict) -> bool:
    return h["formula_key"] != q["formula_key"]


def diff_ion(q: dict, h: dict) -> bool:
    if q["ion_key"] is not None and h["ion_key"] is not None:
        return h["ion_key"] != q["ion_key"]
    return diff_neutral(q, h)


def classify(matches, qi: int, q: dict, window: float) -> dict:
    """Both LOO protocols for one query at one window, from the 0.5 Da hit list.

    ``matches`` is the matcher's score-sorted hit list; the narrower window is applied
    with the matcher's own test ``|ref_pmz - feat_mz| <= window`` (float64 both ways).
    """
    out = {
        "n_other_hits_in_window": 0, "n_high_diff_neutral_in_window": 0, "n_high_diff_ion_in_window": 0,
        "top1_idx": -1, "top1_score": np.nan, "top1_n_matched": 0, "top1_wdp": np.nan,
        "top1_matched_pct": np.nan, "top1_pmz": np.nan, "top1_delta_mz": np.nan,
        "top1_high": False, "top1_diff_neutral": False, "top1_diff_ion": False,
        "top1_false_neutral": False, "top1_false_ion": False, "top1_isomer_ion": False,
        "loo1_top1_idx": -1, "loo1_top1_score": np.nan, "loo1_top1_same_compound": False,
        "loo1_top1_false_neutral": False, "loo1_top1_false_ion": False,
        "loo1_top5_false_neutral": False, "loo1_top5_false_ion": False,
    }
    best = None
    loo1_rank = 0
    for m in matches:
        j = int(m.name)
        if j == qi:
            continue
        if abs(float(m.ref_precursor_mz) - q["pmz"]) > window:
            continue
        h = INFO[j]
        same_compound = h["key"] == q["key"]
        # literal LOO: everything but the query spectrum itself
        loo1_rank += 1
        if loo1_rank == 1:
            out["loo1_top1_idx"] = j
            out["loo1_top1_score"] = float(m.score)
            out["loo1_top1_same_compound"] = same_compound
            if is_high(m) and not same_compound:
                out["loo1_top1_false_neutral"] = diff_neutral(q, h)
                out["loo1_top1_false_ion"] = diff_ion(q, h)
        if loo1_rank <= PRODUCTION_TOP_N and is_high(m) and not same_compound:
            out["loo1_top5_false_neutral"] |= diff_neutral(q, h)
            out["loo1_top5_false_ion"] |= diff_ion(q, h)
        if same_compound:
            continue
        # compound LOO: every spectrum of the query compound removed
        out["n_other_hits_in_window"] += 1
        if is_high(m):
            if diff_neutral(q, h):
                out["n_high_diff_neutral_in_window"] += 1
            if diff_ion(q, h):
                out["n_high_diff_ion_in_window"] += 1
        if best is None:
            best = m
            high = is_high(m)
            dn, di = diff_neutral(q, h), diff_ion(q, h)
            out.update({
                "top1_idx": j, "top1_score": float(m.score), "top1_n_matched": int(m.n_matched),
                "top1_wdp": float(m.wdp), "top1_matched_pct": float(m.matched_pct),
                "top1_pmz": float(m.ref_precursor_mz), "top1_delta_mz": float(m.ref_precursor_mz) - q["pmz"],
                "top1_high": high, "top1_diff_neutral": dn, "top1_diff_ion": di,
                "top1_false_neutral": high and dn, "top1_false_ion": high and di,
                "top1_isomer_ion": high and not di,
            })
    return out


def summarise(d: pd.DataFrame) -> dict:
    n = len(d)
    false_n = d[d.top1_false_neutral]
    false_i = d[d.top1_false_ion]

    def q(s, p):
        return float(s.quantile(p)) if len(s) else np.nan

    delta = false_n.top1_delta_mz.abs()
    return {
        "n_queries": n,
        "n_with_other_hit": int((d.top1_idx >= 0).sum()),
        "rate_with_other_hit": float((d.top1_idx >= 0).mean()) if n else np.nan,
        # compound LOO, top-1
        "n_top1_high": int(d.top1_high.sum()),
        "rate_top1_high": float(d.top1_high.mean()) if n else np.nan,
        "n_top1_false_neutral_formula": int(len(false_n)),
        "rate_top1_false_neutral_formula": float(d.top1_false_neutral.mean()) if n else np.nan,
        "n_top1_false_ion_formula": int(len(false_i)),
        "rate_top1_false_ion_formula": float(d.top1_false_ion.mean()) if n else np.nan,
        "n_top1_high_isomer_ion": int(d.top1_isomer_ion.sum()),
        "rate_top1_high_isomer_ion": float(d.top1_isomer_ion.mean()) if n else np.nan,
        # compound LOO, any hit in the window
        "rate_any_false_neutral_formula": float((d.n_high_diff_neutral_in_window > 0).mean()) if n else np.nan,
        "rate_any_false_ion_formula": float((d.n_high_diff_ion_in_window > 0).mean()) if n else np.nan,
        # literal LOO (only the query spectrum removed)
        "rate_loo1_top1_same_compound": float(d.loo1_top1_same_compound.mean()) if n else np.nan,
        "rate_loo1_top1_false_neutral_formula": float(d.loo1_top1_false_neutral.mean()) if n else np.nan,
        "rate_loo1_top1_false_ion_formula": float(d.loo1_top1_false_ion.mean()) if n else np.nan,
        "rate_loo1_top5_false_neutral_formula": float(d.loo1_top5_false_neutral.mean()) if n else np.nan,
        "rate_loo1_top5_false_ion_formula": float(d.loo1_top5_false_ion.mean()) if n else np.nan,
        # score distribution of the compound-LOO false hits (neutral formula)
        "false_hit_score_min": float(false_n.top1_score.min()) if len(false_n) else np.nan,
        "false_hit_score_q25": q(false_n.top1_score, 0.25),
        "false_hit_score_median": q(false_n.top1_score, 0.5),
        "false_hit_score_q75": q(false_n.top1_score, 0.75),
        "false_hit_score_max": float(false_n.top1_score.max()) if len(false_n) else np.nan,
        "false_hit_n_matched_median": q(false_n.top1_n_matched, 0.5),
        "false_hit_abs_delta_le_0.01": int((delta <= 0.01).sum()),
        "false_hit_abs_delta_0.01_0.05": int(((delta > 0.01) & (delta <= 0.05)).sum()),
        "false_hit_abs_delta_0.05_0.5": int((delta > 0.05).sum()),
        "false_hit_score_histogram_edges": [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.01],
        "false_hit_score_histogram": (np.histogram(false_n.top1_score,
                                                   bins=[1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.01])[0].tolist()
                                      if len(false_n) else []),
        "false_hit_n_matched_histogram": {str(k): int(v) for k, v in
                                          sorted(Counter(false_n.top1_n_matched).items())},
    }


def main() -> None:
    here = Path(__file__).resolve()
    rev_dir = here.parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", action="append", required=True,
                    help="library file (MSP or MGF); repeat to pool several libraries")
    ap.add_argument("--library-tag", action="append", default=None,
                    help="short source tag per --library (default: file stem)")
    ap.add_argument("--tag", default="massbank", help="output file suffix")
    ap.add_argument("--release-tag", default="", help="recorded provenance, e.g. MassBank-data release 2026.03")
    ap.add_argument("--source-url", default="", help="recorded provenance")
    ap.add_argument("--windows", type=float, nargs="+", default=list(WINDOWS_DEFAULT))
    ap.add_argument("--min-precursor-decimals", type=int, default=0,
                    help="keep only records whose precursor m/z has at least this many decimals (3 = high-res)")
    ap.add_argument("--max-peaks", type=int, default=0,
                    help="drop records with more peaks than this (0 = no cap); profile-like GNPS exports with "
                         "thousands of peaks are unlike any centroided query and dominate the runtime")
    ap.add_argument("--sample", type=int, default=0, help="query sample size (0 = every spectrum)")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--verify-n", type=int, default=300,
                    help="queries re-matched with explicit windows to verify the post-filter")
    ap.add_argument("--no-rdkit", action="store_true", help="never compute InChIKeys with RDKit")
    ap.add_argument("--out-dir", default=str(rev_dir / "analysis" / "decoy"))
    ap.add_argument("--progress-every", type=int, default=5000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = sorted(float(w) for w in args.windows)
    if max(windows) > 0.5 + 1e-12:
        raise SystemExit("windows above 0.5 Da are not supported by this one-pass design")
    t_start = time.time()

    ann_cfg, sim_cfg, settings = production_settings()
    print("matcher settings:", settings)

    tags = args.library_tag or [Path(p).stem for p in args.library]
    if len(tags) != len(args.library):
        raise SystemExit("--library-tag must be given once per --library")
    libraries = []
    spectra: list[dict] = []
    for path_str, tag in zip(args.library, tags):
        path = Path(path_str)
        if not path.is_file():
            raise SystemExit(f"library not found: {path}")
        digest = sha256_of(path)
        kept, counts = load_library(path, tag, use_rdkit=not args.no_rdkit,
                                    min_decimals=args.min_precursor_decimals, max_peaks=args.max_peaks)
        libraries.append({"tag": tag, "path": str(path), "bytes": path.stat().st_size,
                          "sha256": digest, "filters": counts})
        spectra.extend(kept)
    n_spec = len(spectra)
    if n_spec == 0:
        raise SystemExit("no spectrum survived the filters")

    # Index once. The hit's ``name`` carries the spectrum index so every match can be
    # resolved to its record without touching AnnotationMatch.
    INFO[:] = [s["_loo"] for s in spectra]
    for i, s in enumerate(spectra):
        s["metadata"]["name"] = str(i)
    key_counts = Counter(r["key"] for r in INFO)
    lib = build_index_from_list(spectra)
    csr = lib["csr"]
    index = lib["index"]
    decimals_hist = Counter(r["decimals"] for r in INFO)
    instr_hist = Counter(r["instrument_type"] for r in INFO).most_common(12)
    print(f"indexed {n_spec} spectra, {len(key_counts)} compounds, {len(index)} integer m/z buckets "
          f"({time.time() - t_start:.0f} s); precursor decimals {dict(sorted(decimals_hist.items()))}", flush=True)

    rng = random.Random(args.seed)
    query_idx = list(range(n_spec))
    if args.sample and args.sample < n_spec:
        query_idx = sorted(rng.sample(query_idx, args.sample))
    n_q = len(query_idx)
    print(f"queries: {n_q} ({'sample' if n_q < n_spec else 'all'})", flush=True)

    rows = []
    t_loop = time.time()
    n_matches_total = 0
    for k, i in enumerate(query_idx):
        spec = spectra[i]
        q = INFO[i]
        feat = make_feature(i, spec)
        matches = match_feature_topn(feat, spectra, index, ann_cfg, sim_cfg,
                                     top_n=10 ** 9, csr=csr, precursor_window_da=0.5)
        n_matches_total += len(matches)
        base = {
            "query_idx": i, "source": q["source"], "query_id": q["id"], "query_name": q["name"],
            "query_key": q["key"], "query_formula": q["formula"], "query_ion_formula": q["ion_key"] or "",
            "query_pmz": q["pmz"], "query_precursor_decimals": q["decimals"], "query_adduct": q["adduct"],
            "query_instrument_type": q["instrument_type"], "query_n_peaks": int(spec["mz"].size),
            "n_same_compound_spectra": key_counts[q["key"]],
        }
        for w in windows:
            res = classify(matches, i, q, w)
            row = dict(base)
            row["window_da"] = w
            row.update(res)
            j = res["top1_idx"]
            if j >= 0:
                h = INFO[j]
                row.update({"top1_source": h["source"], "top1_id": h["id"], "top1_name": h["name"],
                            "top1_formula": h["formula"], "top1_ion_formula": h["ion_key"] or "",
                            "top1_adduct": h["adduct"], "top1_precursor_decimals": h["decimals"]})
            else:
                row.update({"top1_source": "", "top1_id": "", "top1_name": "", "top1_formula": "",
                            "top1_ion_formula": "", "top1_adduct": "", "top1_precursor_decimals": -1})
            rows.append(row)
        if args.progress_every and (k + 1) % args.progress_every == 0:
            el = time.time() - t_loop
            print(f"  {k + 1}/{n_q} queries, {el:.0f} s, {el / (k + 1) * 1000:.1f} ms/query", flush=True)
    loop_seconds = time.time() - t_loop
    print(f"matching done: {n_q} queries in {loop_seconds:.0f} s "
          f"({n_matches_total / max(n_q, 1):.1f} surviving hits per query at 0.5 Da)", flush=True)

    df = pd.DataFrame(rows)
    per_query_path = out_dir / f"loo_{args.tag}.csv"
    df.to_csv(per_query_path, index=False)

    # ---- verification of the post-filter against explicit matcher windows ---------
    verify = {"n_checked": 0, "n_mismatch": 0}
    if args.verify_n > 0:
        vq = rng.sample(query_idx, min(args.verify_n, n_q))
        mism = 0
        ref_index = df.set_index(["query_idx", "window_da"])
        for i in vq:
            spec = spectra[i]
            q = INFO[i]
            feat = make_feature(i, spec)
            for w in windows:
                direct = match_feature_topn(feat, spectra, index, ann_cfg, sim_cfg,
                                            top_n=10 ** 9, csr=csr, precursor_window_da=w)
                res = classify(direct, i, q, w)
                ref = ref_index.loc[(i, w)]
                same_idx = int(res["top1_idx"]) == int(ref.top1_idx) and int(res["loo1_top1_idx"]) == int(ref.loo1_top1_idx)
                a, b = res["top1_score"], float(ref.top1_score)
                same_score = (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-12
                if not (same_idx and same_score):
                    mism += 1
        verify = {"n_checked": len(vq) * len(windows), "n_mismatch": mism}
        print("post-filter verification:", verify)

    # ---- summary ------------------------------------------------------------------
    results = {}
    summary_rows = []
    for w in windows:
        d = df[df.window_da == w]
        rec = summarise(d)
        results[f"window_{w:g}"] = rec
        summary_rows.append({"window_da": w, **{k: v for k, v in rec.items() if not isinstance(v, (list, dict))}})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / f"loo_summary_{args.tag}.csv", index=False)

    # the same table restricted to high-resolution queries (>= 3 decimals), for reference
    results_hires_queries = {}
    for w in windows:
        d = df[(df.window_da == w) & (df.query_precursor_decimals >= 3)]
        if len(d):
            results_hires_queries[f"window_{w:g}"] = {k: v for k, v in summarise(d).items()
                                                      if not isinstance(v, (list, dict))}

    worst = (df[df.top1_false_neutral]
             .sort_values(["top1_score", "window_da"], ascending=[False, True])
             .drop_duplicates(subset=["query_idx", "top1_idx"])
             .head(20))
    cols = ["window_da", "source", "query_id", "query_name", "query_formula", "query_ion_formula", "query_adduct",
            "query_pmz", "query_n_peaks", "top1_source", "top1_id", "top1_name", "top1_formula", "top1_ion_formula",
            "top1_adduct", "top1_pmz", "top1_delta_mz", "top1_score", "top1_n_matched", "top1_wdp",
            "top1_matched_pct", "top1_false_ion"]
    worst[cols].to_csv(out_dir / f"loo_false_hits_examples_{args.tag}.csv", index=False)

    # sanity gates: rates in [0, 1]; the false-formula rate should not grow when the window
    # narrows. The top-1 variant is not monotone by construction (a narrower window can
    # promote a different hit to top-1), so a violation is recorded, not fatal; the
    # any-hit variant is a subset count and is monotone by construction.
    rates = {w: results[f"window_{w:g}"]["rate_top1_false_neutral_formula"] for w in windows}
    assert all(0.0 <= r <= 1.0 for r in rates.values())
    monotone = all(rates[a] <= rates[b] + 1e-12 for a, b in zip(windows, windows[1:]))
    if not monotone:
        print(f"WARNING: top-1 false-formula rate not monotone in the window: {rates}")

    numbers_out = {
        "task": "S2.1f Task A leave-one-out",
        "software": "ASFAMProcessor", "metabo_core_version": CORE_VERSION,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "date": time.strftime("%Y-%m-%d"),
        "matcher_settings": settings,
        "libraries": libraries,
        "release_tag": args.release_tag, "source_url": args.source_url,
        "min_precursor_decimals": args.min_precursor_decimals, "max_peaks": args.max_peaks,
        "n_candidate_spectra": n_spec, "n_compounds": len(key_counts),
        "precursor_decimals_histogram": {str(k): v for k, v in sorted(decimals_hist.items())},
        "instrument_types_top": instr_hist,
        "n_queries": n_q, "query_mode": "all" if n_q == n_spec else f"seeded random sample (seed {args.seed})",
        "compound_key": "InChIKey first block (14 chars) when available, else formula + name",
        "same_compound_removal": "masking by compound key after one matcher call at 0.5 Da; "
                                 "narrower windows by post-filter on |ref precursor - query precursor|",
        "windows_da": windows,
        "post_filter_verification": verify,
        "sanity_top1_rate_monotone_in_window": monotone,
        "results": results,
        "results_hires_queries_only": results_hires_queries,
        "runtime_seconds": round(time.time() - t_start, 1),
        "matching_seconds": round(loop_seconds, 1),
        "outputs": {"per_query": per_query_path.name, "summary": f"loo_summary_{args.tag}.csv",
                    "examples": f"loo_false_hits_examples_{args.tag}.csv"},
    }
    with open(out_dir / f"numbers_loo_{args.tag}.json", "w", encoding="utf-8") as fh:
        json.dump(numbers_out, fh, indent=2, ensure_ascii=False)
    show = ["window_da", "n_queries", "rate_top1_high", "rate_top1_false_neutral_formula",
            "rate_top1_false_ion_formula", "rate_top1_high_isomer_ion", "rate_any_false_neutral_formula",
            "rate_any_false_ion_formula", "rate_loo1_top1_same_compound", "rate_loo1_top1_false_neutral_formula",
            "rate_loo1_top5_false_neutral_formula", "false_hit_score_median"]
    print(summary[show].to_string(index=False))
    print(f"total runtime {time.time() - t_start:.0f} s; outputs in {out_dir}")


if __name__ == "__main__":
    main()
