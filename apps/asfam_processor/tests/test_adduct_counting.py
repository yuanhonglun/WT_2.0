"""Task D3: adduct copies are counted as independent features end-to-end.

PR-D makes isotope / adduct copies each count as an independent feature
(MS-DIAL convention). Stage 5 already *marks-and-keeps* the non-representative
adduct copy (``status="adduct_excluded"``, ``is_duplicate=True``, shared
``adduct_group_id``) instead of deleting it, and Stage 7 iterates **all**
per-replicate features (no status filter), so each adduct copy already becomes
its own counted ``Feature``. Task D3 plumbs ``adduct_group_id`` onto ``Feature``
+ the CSV so the adduct cluster id survives to export (the symmetric column to
D2's ``isotope_group_id``).

Fixture: a co-eluting ``[M+H]+`` / ``[M+Na]+`` pair of neutral mass M = 285.0
in positive mode. Stage 5's EIC-coelution gate needs the raw MS1 survey to
carry BOTH m/z as co-eluting Gaussians at one RT, so we reuse the
``_make_raw_multi`` + ``_gaussian`` + ``_make_feature`` helper pattern from
``test_isotope_counting.py`` (mz_list / height_list with multiple co-eluting
ions). The higher-intensity ``[M+H]+`` becomes Stage 5's representative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from asfam.config import ProcessingConfig
from asfam.models import CandidateFeature, RawSegmentData, ScanCycle
from asfam.core.mass_utils import check_adduct_pair
from asfam.pipeline.stage5_adduct_dedup import run_stage5
from asfam.pipeline.stage7_alignment import run_stage7
from asfam.pipeline.stage8_export import _export_csv


# Neutral mass M = 285.0, positive mode.
#   [M+H]+  = 285.0 + 1.00727646677 = 286.00727...
#   [M+Na]+ = 285.0 + 22.989218     = 307.989218
_MZ_H = 285.0 + 1.00727646677
_MZ_NA = 285.0 + 22.989218
_CO_APEX_RT = 4.0
_SEG = "280-310"


def _gaussian(rt: np.ndarray, center: float, sigma: float, height: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((rt - center) / sigma) ** 2)


def _make_raw_multi(seg_name: str, n_cycles: int, mz_list: list[float],
                    height_list: list[float], co_apex_rt: float = _CO_APEX_RT,
                    sigma: float = 0.05, replicate_id: int = 1) -> RawSegmentData:
    """RawSegmentData whose MS1 survey carries every m/z in ``mz_list`` as a
    co-eluting Gaussian (so all extract as perfectly-correlated MS1 EICs).
    """
    rt = np.linspace(0.0, 10.0, n_cycles)
    profiles = [_gaussian(rt, co_apex_rt, sigma, h) for h in height_list]
    mz_arr = np.asarray(mz_list, dtype=np.float64)
    cycles = []
    for i in range(n_cycles):
        ms1_int = np.asarray([p[i] for p in profiles], dtype=np.float64)
        cycles.append(
            ScanCycle(
                cycle_index=i,
                rt=float(rt[i]),
                ms1_mz=mz_arr.copy(),
                ms1_intensity=ms1_int,
                ms2_scans={},
            )
        )
    seg_lo = int(np.floor(min(mz_list) - 5))
    seg_hi = int(np.ceil(max(mz_list) + 5))
    return RawSegmentData(
        file_path=f"/fake/{seg_name}.mzML",
        segment_name=seg_name,
        segment_low=seg_lo,
        segment_high=seg_hi,
        replicate_id=replicate_id,
        n_cycles=n_cycles,
        rt_array=rt.astype(np.float64),
        precursor_list=[],
        cycles=cycles,
        collision_energy=20.0,
    )


def _make_feature(feature_id: str, seg_name: str, mz: float,
                  rt_apex: float, height: float) -> CandidateFeature:
    return CandidateFeature(
        feature_id=feature_id,
        segment_name=seg_name,
        replicate_id=1,
        precursor_mz_nominal=int(round(mz)),
        rt_apex=rt_apex,
        rt_left=rt_apex - 0.1,
        rt_right=rt_apex + 0.1,
        ms2_mz=np.array([50.0, 80.0, 120.0], dtype=np.float64),
        ms2_intensity=np.array([1000.0, 500.0, 200.0], dtype=np.float64),
        n_fragments=3,
        ms1_precursor_mz=mz,
        ms1_height=height,
    )


def test_check_adduct_pair_resolves_h_na():
    """Sanity: the fixture m/z really do resolve to one neutral mass (285.0)."""
    pair = check_adduct_pair(_MZ_H, _MZ_NA, "positive", 0.02)
    assert pair == ("[M+H]+", "[M+Na]+")


def test_adduct_copies_counted_end_to_end(tmp_path):
    """[M+H]+ and [M+Na]+ both reach the CSV as independent rows, with the
    shared ``adduct_group_id`` plumbed through and the [M+Na]+ copy flagged."""
    cfg = ProcessingConfig()
    raw = _make_raw_multi(_SEG, n_cycles=800,
                          mz_list=[_MZ_H, _MZ_NA],
                          height_list=[10000.0, 3000.0])
    f_h = _make_feature("rep1_00000", _SEG, mz=_MZ_H, rt_apex=_CO_APEX_RT,
                        height=10000.0)
    f_na = _make_feature("rep1_00001", _SEG, mz=_MZ_NA, rt_apex=_CO_APEX_RT,
                         height=3000.0)

    # Stage 5: mark-and-keep the adduct copy; both share an adduct_group_id.
    run_stage5({"1": [f_h, f_na]}, {"1": [raw]}, cfg)

    # Guard (non-vacuous): the adduct cluster actually formed in Stage 5.
    assert f_h.adduct_group_id is not None
    assert f_na.adduct_group_id is not None
    assert f_h.adduct_group_id == f_na.adduct_group_id
    assert f_h.duplicate_type == "adduct"
    assert f_na.duplicate_type == "adduct"
    # Representative = max-intensity [M+H]+; the [M+Na]+ copy is the flagged one.
    assert f_h.is_duplicate is False
    assert f_na.is_duplicate is True

    # Stage 7: every per-replicate feature becomes its own Feature (no filter).
    feats = run_stage7({"1": [f_h, f_na]}, cfg)
    assert len(feats) == 2

    # Stage 8: export to CSV and read back (skip the leading ``#`` header lines).
    out = tmp_path / "features.csv"
    _export_csv(feats, out, cfg)
    df = pd.read_csv(out, comment="#")

    # Both adduct copies counted — nothing filtered.
    assert len(df) == 2

    for col in ("duplicate_type", "adduct_group_id", "adduct"):
        assert col in df.columns, f"missing column {col}"

    # Both rows carry the adduct duplicate_type.
    assert set(df["duplicate_type"].dropna()) == {"adduct"}

    # Exactly one row is flagged duplicate (the [M+Na]+ copy). Compare via a
    # lowercased-string mask so the assertion is robust to pandas inferring
    # bool vs object dtype for the column.
    dup_mask = df["is_duplicate"].astype(str).str.lower() == "true"
    assert int(dup_mask.sum()) == 1

    # Both adduct labels survive to the ``adduct`` column.
    adducts = set(df["adduct"].dropna())
    assert "[M+H]+" in adducts
    assert "[M+Na]+" in adducts

    # The adduct cluster id reaches the CSV (the D3 plumbing under test): both
    # rows carry a non-empty, shared group id.
    assert df["adduct_group_id"].notna().all()
    assert df["adduct_group_id"].nunique() == 1
