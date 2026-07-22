"""``alignment.eic``: every spot's chromatograms, addressable one spot at a time.

The GUI plots a feature's MS1 EIC for every replicate plus its product-ion EICs.
Reading those out of the mzML costs ~7.6 s per 157 MiB segment, so with the raw
scans gone from memory (PR-3) a single click cost tens of seconds. Gap filling
already extracts exactly these chromatograms, so it writes them down.

Layout mirrors ``PeakAligner.cs:47,162,209-226``. Gap filling walks *samples*
outermost — one sample's raw data is resident at a time — but the reader wants
one *spot* at a time, so each sample streams into its own temporary file and a
final pass transposes them::

    [magic b"AE"][version int32][n_spots int32]
    [ (key_len int32, key utf8, offset int64) x n_spots ]     <- directory
    [ body x n_spots ]

    body := [n_traces int32] [ trace x n_traces ]
            [n_fragments int32] [ (product_mz float64, trace) x n_fragments ]
    trace := [label_len int32, label utf8]
             [status uint8] [rt_left f32] [rt_right f32] [rt_apex f32]
             [n_points int32] [rt f32 x n] [intensity f32 x n]

Keyed by ``feature_id``, not by position: PR-6 renumbers features by m/z, and a
positional index would silently start returning the wrong chromatogram.

That key is the *aligned spot's* id, which is the one id the single-sample view
does not hold — hence :class:`SpotMap` and ``spotmap.json`` at the bottom of this
module, written beside the store by the same stage.

float32 throughout. These traces are drawn, never integrated — gap filling
measured its numbers off the float64 arrays before they got here.
"""
from __future__ import annotations

import json
import logging
import os
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

EIC_MAGIC = b"AE"
EIC_VERSION = 1

STORE_NAME = "alignment.eic"

_STATUS_CODES = {"detected": 0, "filled": 1, "no_signal": 2}
_STATUS_NAMES = {v: k for k, v in _STATUS_CODES.items()}


class EicStoreError(RuntimeError):
    """The store's magic or version does not match this build."""


@dataclass
class Trace:
    """One chromatogram plus the peak window to shade under it."""

    label: str
    rt: np.ndarray
    intensity: np.ndarray
    status: str = "detected"
    rt_left: float = 0.0
    rt_right: float = 0.0
    rt_apex: float = 0.0


@dataclass
class SpotChromatograms:
    """What the EIC panel needs to draw one feature."""

    #: Quantitation-ion chromatogram per sample, ``label`` = sample id.
    quant: list[Trace] = field(default_factory=list)
    #: Representative sample's strongest fragments, ``label`` = "%.3f" m/z.
    fragments: list[tuple[float, Trace]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# codec
# ---------------------------------------------------------------------------

def _f32(arr) -> bytes:
    return np.ascontiguousarray(arr, dtype="<f4").tobytes()


def _write_str(fh, text: str) -> None:
    raw = text.encode("utf-8")
    fh.write(struct.pack("<i", len(raw)))
    fh.write(raw)


def _read_str(fh) -> str:
    (n,) = struct.unpack("<i", fh.read(4))
    return fh.read(n).decode("utf-8")


def _write_trace(fh, trace: Trace) -> None:
    _write_str(fh, trace.label)
    fh.write(struct.pack(
        "<Bfff",
        _STATUS_CODES.get(trace.status, 0),
        trace.rt_left, trace.rt_right, trace.rt_apex,
    ))
    n = int(np.size(trace.rt))
    fh.write(struct.pack("<i", n))
    fh.write(_f32(trace.rt))
    fh.write(_f32(trace.intensity))


def _read_trace(fh) -> Trace:
    label = _read_str(fh)
    status_code, rt_left, rt_right, rt_apex = struct.unpack("<Bfff", fh.read(13))
    (n,) = struct.unpack("<i", fh.read(4))
    rt = np.frombuffer(fh.read(4 * n), dtype="<f4").astype(np.float64)
    intensity = np.frombuffer(fh.read(4 * n), dtype="<f4").astype(np.float64)
    return Trace(
        label=label, rt=rt, intensity=intensity,
        status=_STATUS_NAMES.get(status_code, "detected"),
        rt_left=rt_left, rt_right=rt_right, rt_apex=rt_apex,
    )


def _write_body(fh, spot: SpotChromatograms) -> None:
    fh.write(struct.pack("<i", len(spot.quant)))
    for trace in spot.quant:
        _write_trace(fh, trace)
    fh.write(struct.pack("<i", len(spot.fragments)))
    for product_mz, trace in spot.fragments:
        fh.write(struct.pack("<d", float(product_mz)))
        _write_trace(fh, trace)


def _read_body(fh) -> SpotChromatograms:
    (n_traces,) = struct.unpack("<i", fh.read(4))
    quant = [_read_trace(fh) for _ in range(n_traces)]
    (n_frags,) = struct.unpack("<i", fh.read(4))
    fragments = []
    for _ in range(n_frags):
        (product_mz,) = struct.unpack("<d", fh.read(8))
        fragments.append((product_mz, _read_trace(fh)))
    return SpotChromatograms(quant=quant, fragments=fragments)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

class EicSpillWriter:
    """Collects one sample at a time, then transposes to spot-major on close.

    Use as a context manager: the temporaries are removed on the way out
    whether or not :meth:`transpose` ran, so a cancelled or crashed gap fill
    never leaves hundreds of MiB behind.
    """

    def __init__(self, keys: list[str], temp_dir: Optional[Path] = None):
        self._keys = list(keys)
        self._dir = Path(tempfile.mkdtemp(prefix="metra_eic_", dir=temp_dir))
        self._parts: list[Path] = []

    def __enter__(self) -> "EicSpillWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()

    def add_sample(self, spots: Iterator[SpotChromatograms]) -> None:
        """Append this sample's contribution to every spot, in ``keys`` order."""
        part = self._dir / f"part{len(self._parts):03d}.tmp"
        seeks: list[int] = []
        with open(part, "wb") as fh:
            fh.write(struct.pack("<i", len(self._keys)))
            fh.write(b"\x00" * (8 * len(self._keys)))
            for spot in spots:
                seeks.append(fh.tell())
                _write_body(fh, spot)
            if len(seeks) != len(self._keys):
                raise ValueError(
                    f"sample produced {len(seeks)} spots, expected {len(self._keys)}"
                )
            fh.seek(4)
            fh.write(np.asarray(seeks, dtype="<i8").tobytes())
        self._parts.append(part)

    def transpose(self, target: Path) -> int:
        """Merge the per-sample parts into ``target``; return its size in bytes."""
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.parent / (target.name + ".tmp")

        handles = [open(p, "rb") for p in self._parts]
        try:
            tables = []
            for fh in handles:
                (n,) = struct.unpack("<i", fh.read(4))
                if n != len(self._keys):
                    raise EicStoreError("temporary part has a different spot count")
                tables.append(np.frombuffer(fh.read(8 * n), dtype="<i8"))

            with open(tmp, "wb") as out:
                out.write(EIC_MAGIC)
                out.write(struct.pack("<ii", EIC_VERSION, len(self._keys)))
                directory_at = out.tell()
                for key in self._keys:
                    _write_str(out, key)
                    out.write(struct.pack("<q", 0))       # back-filled below

                offsets = []
                for i in range(len(self._keys)):
                    offsets.append(out.tell())
                    merged = SpotChromatograms()
                    for fh, table in zip(handles, tables):
                        fh.seek(int(table[i]))
                        part = _read_body(fh)
                        merged.quant.extend(part.quant)
                        merged.fragments.extend(part.fragments)
                    merged.quant.sort(key=lambda t: t.label)
                    _write_body(out, merged)

                out.seek(directory_at)
                for key, offset in zip(self._keys, offsets):
                    _write_str(out, key)
                    out.write(struct.pack("<q", offset))
        finally:
            for fh in handles:
                fh.close()

        os.replace(tmp, target)
        return target.stat().st_size

    def cleanup(self) -> None:
        for part in self._parts:
            try:
                part.unlink()
            except OSError:
                pass
        self._parts = []
        try:
            self._dir.rmdir()
        except OSError:
            logger.warning("Could not remove EIC temp dir %s", self._dir)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

class EicStore:
    """Random access into ``alignment.eic`` by ``feature_id``.

    Holds only the directory (a few hundred KiB); a body is read on demand.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._offsets: dict[str, int] = {}
        with open(self.path, "rb") as fh:
            if fh.read(2) != EIC_MAGIC:
                raise EicStoreError(f"{self.path.name}: bad magic")
            version, n_spots = struct.unpack("<ii", fh.read(8))
            if version != EIC_VERSION:
                raise EicStoreError(
                    f"{self.path.name}: format version {version}, this build "
                    f"writes {EIC_VERSION}"
                )
            for _ in range(n_spots):
                key = _read_str(fh)
                (offset,) = struct.unpack("<q", fh.read(8))
                self._offsets[key] = offset

    def __len__(self) -> int:
        return len(self._offsets)

    def __contains__(self, key: str) -> bool:
        return key in self._offsets

    def get(self, key: str) -> Optional[SpotChromatograms]:
        offset = self._offsets.get(key)
        if offset is None:
            return None
        with open(self.path, "rb") as fh:
            fh.seek(offset)
            return _read_body(fh)


def open_store(output_dir) -> Optional[EicStore]:
    """``alignment.eic`` under ``output_dir``, or ``None`` when absent/unreadable."""
    path = Path(output_dir) / STORE_NAME
    if not path.is_file():
        return None
    try:
        return EicStore(path)
    except (OSError, EicStoreError, struct.error):
        logger.warning("Could not open %s", path, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# spotmap.json — how a per-sample candidate reaches its spot's chromatograms
# ---------------------------------------------------------------------------

SPOTMAP_NAME = "spotmap.json"
SPOTMAP_VERSION = 1


@dataclass
class SpotMap:
    """The store's key translation table, spilled beside it by stage 7.

    The store is keyed by the aligned spot's id (``F00042``). The single-sample
    view lists candidates read back from ``_work/``, which carry the ids stage 3
    minted (``rep1_00042``). The two namespaces never collide, so a candidate id
    looked up straight in the store misses *every* time — which is exactly how
    the single-sample EIC panel came to report "no chromatogram" for every
    feature it was ever shown.

    This is a file rather than a derivation because nothing else survives stage
    7: ``Feature`` does not record which candidates it was built from.
    """

    #: candidate ``feature_id`` -> spot ``feature_id``, i.e. the store's key.
    spot_of: dict[str, str] = field(default_factory=dict)
    #: spot ``feature_id`` -> the sample its stored fragments came from. Gap fill
    #: extracts fragment chromatograms from the representative sample only, so in
    #: a single-sample view they may not belong to the sample on screen.
    representative_of: dict[str, str] = field(default_factory=dict)
    #: Intentionally folded natural candidates retain the scientific reason and
    #: identity evidence that allowed them to share a keeper cell.
    fold_reason_of: dict[str, str] = field(default_factory=dict)
    fold_evidence_of: dict[str, dict] = field(default_factory=dict)


def save_spot_map(path, spot_map: SpotMap) -> None:
    """Write ``spot_map`` to ``path``, published atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({
            "version": SPOTMAP_VERSION,
            "spot_of": spot_map.spot_of,
            "representative_of": spot_map.representative_of,
            "fold_reason_of": spot_map.fold_reason_of,
            "fold_evidence_of": spot_map.fold_evidence_of,
        }, fh)
    os.replace(tmp, path)


def load_spot_map(output_dir) -> SpotMap:
    """``spotmap.json`` under ``output_dir``, or an empty map.

    Empty is an ordinary answer, not an error: a project written before this file
    existed has none. The caller falls back to reporting that it has no
    chromatogram, which is what such a project has always done — never to
    plotting some other feature's.
    """
    path = Path(output_dir) / SPOTMAP_NAME
    if not path.is_file():
        return SpotMap()
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        if raw.get("version") != SPOTMAP_VERSION:
            logger.warning(
                "%s: format version %s, this build writes %d — ignoring it",
                path.name, raw.get("version"), SPOTMAP_VERSION,
            )
            return SpotMap()
        return SpotMap(
            spot_of=dict(raw["spot_of"]),
            representative_of=dict(raw.get("representative_of") or {}),
            fold_reason_of=dict(raw.get("fold_reason_of") or {}),
            fold_evidence_of=dict(raw.get("fold_evidence_of") or {}),
        )
    except (OSError, AttributeError, KeyError, TypeError, ValueError):
        logger.warning("Could not read %s", path, exc_info=True)
        return SpotMap()
