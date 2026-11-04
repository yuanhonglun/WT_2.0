"""Read an ASFAMProcessor ``features.csv`` correctly.

The export starts with three ``# key=value`` header lines. Reading it with
``pandas.read_csv(..., comment="#")`` is WRONG: pandas then treats ``#``
anywhere in a line as the start of a comment, so a row whose library name
contains ``#`` (e.g. the ascaroside ``ascr#17``) is cut off at the name and
every column after it reads as missing — including ``annotated``,
``total_score``, ``n_matched`` and ``signal_type``. On the validation set this
silently dropped one high-scoring annotation and mis-typed one feature.

``read_features`` skips the leading ``#`` lines by count instead, and returns
the header metadata (mode, version, chromatographic_mode) alongside.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def header_lines(path: Path) -> list[str]:
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                out.append(line.rstrip("\n"))
            else:
                break
    return out


def header_meta(path: Path) -> dict[str, str]:
    meta = {}
    for line in header_lines(path):
        key, _, val = line.lstrip("# ").partition("=")
        if key:
            meta[key.strip()] = val.strip()
    return meta


def read_features(path, retained_only: bool = False, **kwargs) -> pd.DataFrame:
    """``features.csv`` as a DataFrame, never truncated at a ``#`` inside a cell.

    ``retained_only=True`` keeps ``is_duplicate == False`` rows — the only rows a
    downstream count may use, because the pipeline marks redundancy and never
    deletes a feature.
    """
    path = Path(path)
    n_skip = len(header_lines(path))
    kwargs.setdefault("low_memory", False)
    df = pd.read_csv(path, skiprows=n_skip, **kwargs)
    if retained_only:
        dup = df["is_duplicate"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        df = df[~dup].copy()
    return df


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
