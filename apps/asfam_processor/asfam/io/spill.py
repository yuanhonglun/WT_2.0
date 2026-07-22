"""Per-sample spill files: keep peak RSS independent of the sample count.

The pipeline processes one sample at a time and writes its finished features
to disk before loading the next one, mirroring MS-DIAL's ``.pai`` / ``.dcl``
pair:

``<sample_id>.mspec`` (≈ ``.dcl``)
    Every feature's MS2 arrays, one *result body* per feature, preceded by a
    seek-pointer table so a single body can be read at random without touching
    the rest of the file::

        [magic b"MS"][version int32][N int32][seek int64 x N][body x N]

    Written in two passes, as ``MsdecResultsWriter.cs`` does: the pointer table
    is emitted as zeros, each body records its own byte offset, and the table is
    back-filled at the end.

``<sample_id>.mfeat`` (≈ ``.pai``)
    Every scalar field, dedup flag and annotation match of the features, plus
    each one's ``ms2_seek_ptr`` into the ``.mspec``. Pickled; the MS2 arrays
    themselves are *not* in here.

``<sample_id>.json``
    Manifest. Written last and atomically, so its presence is the signal that
    the sample completed. Carries the config fingerprint used by the
    checkpoint-resume logic.

All three are written to a temporary name and ``os.replace``-d into place, so a
process killed mid-write never leaves a file a later run would trust.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import pickle
import struct
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from asfam.models import CandidateFeature
from asfam.io.project_file import _candidates_to_dicts, _dicts_to_candidates

logger = logging.getLogger(__name__)

MSPEC_MAGIC = b"MS"
MSPEC_VERSION = 1
# v2 adds ``ms1_quant_mz``. A v1 spill would read it back as None and gap
# filling would then have no ion to integrate for its MS1 features, so the
# reader rejects v1 outright rather than silently filling zeros.
MFEAT_VERSION = 2
MANIFEST_VERSION = 1

# Byte offset of the seek-pointer table inside a .mspec: magic(2) + version(4)
# + n_features(4).
_MSPEC_HEADER_SIZE = 2 + 4 + 4

# MS2 arrays live in the .mspec, never in the .mfeat.
_MS2_ARRAY_KEYS = ("ms2_mz", "ms2_intensity", "ms2_sn", "ms2_gaussian")

_EMPTY = np.empty(0, dtype=np.float64)

# One lock per .mspec path, serializing ``read_ms2``. The GUI may fire it from
# several threads while a re-annotation worker is rewriting the same stem;
# ``write_sample`` publishes with ``os.replace``, and on Windows that fails
# outright if a reader holds the file open, so readers must not overlap freely.
_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[str, threading.Lock] = {}


class SpillFormatError(RuntimeError):
    """A spill file's magic or version does not match this build.

    Raised rather than silently falling back: a stale ``_work/`` directory must
    force a recompute, never a half-understood read.
    """


def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock


def _sibling(stem: Path, suffix: str) -> Path:
    """``stem`` + ``suffix``, appended — not ``with_suffix``.

    Sample ids come from filenames ("ungrouped_<stem>") and can contain dots,
    which ``with_suffix`` would treat as an extension and overwrite.
    """
    return stem.parent / (stem.name + suffix)


def _mspec_path(stem: Path) -> Path:
    return _sibling(stem, ".mspec")


def _mfeat_path(stem: Path) -> Path:
    return _sibling(stem, ".mfeat")


def _manifest_path(stem: Path) -> Path:
    return _sibling(stem, ".json")


def _replace_atomic(tmp: Path, final: Path) -> None:
    os.replace(tmp, final)


# ---------------------------------------------------------------------------
# .mspec body codec
# ---------------------------------------------------------------------------

def _as_f64(arr) -> np.ndarray:
    if arr is None:
        return _EMPTY
    return np.ascontiguousarray(arr, dtype="<f8")


def _write_body(fh: io.BufferedWriter, feat: CandidateFeature) -> None:
    """One result body: the two mandatory arrays, then two optional ones.

    Every array is float64. Downgrading to float32 would perturb the m/z values
    the annotation scorer rounds to 4 decimals; the peak identity it derives
    from ``round(mz, 4)`` must survive the round-trip bit for bit.
    """
    mz = _as_f64(feat.ms2_mz)
    intensity = _as_f64(feat.ms2_intensity)
    n = int(mz.size)
    if intensity.size != n:
        raise ValueError(
            f"{feat.feature_id}: ms2_intensity has {intensity.size} peaks, "
            f"ms2_mz has {n}"
        )
    fh.write(struct.pack("<i", n))
    fh.write(mz.tobytes())
    fh.write(intensity.tobytes())
    for arr in (feat.ms2_sn, feat.ms2_gaussian):
        if arr is None:
            fh.write(struct.pack("<b", 0))
            continue
        vec = _as_f64(arr)
        fh.write(struct.pack("<bi", 1, int(vec.size)))
        fh.write(vec.tobytes())


def _read_f64(fh, count: int) -> np.ndarray:
    raw = fh.read(count * 8)
    if len(raw) != count * 8:
        raise SpillFormatError("truncated .mspec body")
    return np.frombuffer(raw, dtype="<f8").astype(np.float64, copy=True)


def _read_body(fh) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    (n,) = struct.unpack("<i", fh.read(4))
    mz = _read_f64(fh, n)
    intensity = _read_f64(fh, n)
    optional: list[Optional[np.ndarray]] = []
    for _ in range(2):
        (has,) = struct.unpack("<b", fh.read(1))
        if not has:
            optional.append(None)
            continue
        (size,) = struct.unpack("<i", fh.read(4))
        optional.append(_read_f64(fh, size))
    return mz, intensity, optional[0], optional[1]


def _open_mspec(stem: Path):
    path = _mspec_path(stem)
    fh = open(path, "rb")
    try:
        magic = fh.read(2)
        if magic != MSPEC_MAGIC:
            raise SpillFormatError(f"{path.name}: bad magic {magic!r}")
        (version, n) = struct.unpack("<ii", fh.read(8))
        if version != MSPEC_VERSION:
            raise SpillFormatError(
                f"{path.name}: format version {version}, this build writes "
                f"{MSPEC_VERSION}. Delete the _work directory and re-run."
            )
    except Exception:
        fh.close()
        raise
    return fh, n


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_sample(
    stem: Path,
    features: list[CandidateFeature],
    fingerprint: Optional[str] = None,
    sample_id: Optional[str] = None,
) -> None:
    """Spill one sample's finished features to ``<stem>.mspec`` + ``.mfeat``.

    Writes the manifest last: a reader treats a sample without one as unfinished.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    mspec_tmp = _sibling(stem, ".mspec.tmp")
    seeks: list[int] = []
    with open(mspec_tmp, "wb") as fh:
        fh.write(MSPEC_MAGIC)
        fh.write(struct.pack("<ii", MSPEC_VERSION, len(features)))
        fh.write(b"\x00" * (8 * len(features)))          # pointer table placeholder
        for feat in features:
            seeks.append(fh.tell())
            _write_body(fh, feat)
        fh.seek(_MSPEC_HEADER_SIZE)                       # pass 2: back-fill
        fh.write(np.asarray(seeks, dtype="<i8").tobytes())
    _replace_atomic(mspec_tmp, _mspec_path(stem))

    records = _candidates_to_dicts(features)
    for rec, ptr, feat in zip(records, seeks, features):
        for key in _MS2_ARRAY_KEYS:
            rec.pop(key, None)
        rec["ms2_seek_ptr"] = ptr
        rec["n_ms2_peaks"] = int(len(feat.ms2_mz)) if feat.ms2_mz is not None else 0

    mfeat_tmp = _sibling(stem, ".mfeat.tmp")
    with open(mfeat_tmp, "wb") as fh:
        pickle.dump(
            {"version": MFEAT_VERSION, "sample_id": sample_id or stem.name,
             "features": records},
            fh, protocol=pickle.HIGHEST_PROTOCOL,
        )
    _replace_atomic(mfeat_tmp, _mfeat_path(stem))

    manifest = {
        "version": MANIFEST_VERSION,
        # Lets ``sample_is_complete`` reject a spill this build cannot read
        # without paying to unpickle it, so a stale _work/ is recomputed rather
        # than blowing up in ``read_sample_features`` after every sample has
        # already been reported as "reused".
        "mfeat_version": MFEAT_VERSION,
        "sample_id": sample_id or stem.name,
        "fingerprint": fingerprint,
        "n_features": len(features),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_tmp = _sibling(stem, ".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _replace_atomic(manifest_tmp, _manifest_path(stem))

    logger.info("  Spilled %d features -> %s.{mspec,mfeat}", len(features), stem.name)


def read_sample_features(
    stem: Path,
    load_ms2: bool = False,
) -> list[CandidateFeature]:
    """Read one sample's features back.

    With ``load_ms2=False`` (the default) each feature carries only its
    ``ms2_seek_ptr``; ``ms2_mz`` / ``ms2_intensity`` are ``None`` and a caller
    that needs a spectrum fetches it with :func:`read_ms2`. With ``load_ms2=True``
    the whole ``.mspec`` is streamed back in one sequential pass.
    """
    stem = Path(stem)
    with open(_mfeat_path(stem), "rb") as fh:
        payload = pickle.load(fh)
    version = payload.get("version")
    if version != MFEAT_VERSION:
        raise SpillFormatError(
            f"{_mfeat_path(stem).name}: format version {version}, this build "
            f"writes {MFEAT_VERSION}. Delete the _work directory and re-run."
        )
    records = payload["features"]

    # _dicts_to_candidates insists on the two array keys; feed it empties and
    # overwrite below rather than materializing throw-away lists.
    for rec in records:
        rec["ms2_mz"] = ()
        rec["ms2_intensity"] = ()
    features = _dicts_to_candidates(records)

    if not load_ms2:
        for feat, rec in zip(features, records):
            feat.ms2_mz = None
            feat.ms2_intensity = None
            feat.ms2_seek_ptr = rec["ms2_seek_ptr"]
        return features

    fh, n = _open_mspec(stem)
    try:
        if n != len(features):
            raise SpillFormatError(
                f"{stem.name}: .mspec holds {n} bodies, .mfeat holds "
                f"{len(features)} features"
            )
        seeks = np.frombuffer(fh.read(8 * n), dtype="<i8")
        for feat, rec, ptr in zip(features, records, seeks):
            fh.seek(int(ptr))
            mz, intensity, sn, gaussian = _read_body(fh)
            feat.ms2_mz = mz
            feat.ms2_intensity = intensity
            feat.ms2_sn = sn
            feat.ms2_gaussian = gaussian
            feat.ms2_seek_ptr = rec["ms2_seek_ptr"]
    finally:
        fh.close()
    return features


def read_ms2(stem: Path, seek_ptr: int) -> tuple[np.ndarray, np.ndarray]:
    """Random-read one feature's MS2 spectrum from ``<stem>.mspec``."""
    mz, intensity, _sn, _gauss = read_ms2_full(stem, seek_ptr)
    return mz, intensity


def read_ms2_full(
    stem: Path, seek_ptr: int,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Random-read one feature's whole MS2 body: ``(mz, intensity, sn, gaussian)``.

    Stage 7 needs ``ms2_gaussian`` to aggregate the representative's peak-shape
    score, which ``read_ms2`` drops.
    """
    stem = Path(stem)
    path = _mspec_path(stem)
    with _lock_for(path):
        fh, _ = _open_mspec(stem)
        try:
            fh.seek(int(seek_ptr))
            return _read_body(fh)
        finally:
            fh.close()


def read_manifest(stem: Path) -> Optional[dict]:
    """Return the sample's manifest, or ``None`` if it never finished."""
    path = _manifest_path(Path(stem))
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def sample_is_complete(stem: Path, fingerprint: Optional[str] = None) -> bool:
    """Is this sample's spill usable as a checkpoint?

    All three files must exist, and — when ``fingerprint`` is given — the
    manifest's fingerprint must match, so a parameter change forces a recompute.
    """
    stem = Path(stem)
    if not (_mspec_path(stem).is_file() and _mfeat_path(stem).is_file()):
        return False
    manifest = read_manifest(stem)
    if manifest is None:
        return False
    if manifest.get("mfeat_version") != MFEAT_VERSION:
        return False
    if fingerprint is not None and manifest.get("fingerprint") != fingerprint:
        return False
    return True


def scan_checkpoints(work_dir: Path) -> list[tuple[str, str]]:
    """List ``(sample_id, created)`` for every completed sample under ``work_dir``.

    Existence-only: the fingerprint check belongs to the orchestrator, which is
    the one that knows the config. Used by the GUI to phrase its "reuse?" prompt.
    """
    work_dir = Path(work_dir)
    if not work_dir.is_dir():
        return []
    found = []
    for manifest_path in sorted(work_dir.glob("*.json")):
        stem = work_dir / manifest_path.name[: -len(".json")]
        manifest = read_manifest(stem)
        if manifest is None:
            continue
        found.append((manifest.get("sample_id", stem.name),
                      manifest.get("created", "")))
    return found


def clear_work_dir(work_dir: Path) -> int:
    """Delete every spill file under ``work_dir``; return how many were removed.

    Never called automatically — ``_work/`` survives a successful export so a
    re-export or a crash-resume can skip stages 0-6.5. Only the GUI's explicit
    "clear intermediate results" action reaches this.
    """
    work_dir = Path(work_dir)
    if not work_dir.is_dir():
        return 0
    removed = 0
    for suffix in (".mspec", ".mfeat", ".json", ".mspec.tmp", ".mfeat.tmp", ".json.tmp"):
        for path in work_dir.glob(f"*{suffix}"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                logger.warning("Could not delete %s", path)
    return removed


# ---------------------------------------------------------------------------
# Checkpoint fingerprint
# ---------------------------------------------------------------------------

# Parameters read only by stage 7 (alignment) or stage 8 (export), plus the
# worker-count knob. Changing one of these must NOT invalidate a spilled
# sample: the spill is the state of the pipeline at the end of stage 6.5.
# Anything not listed here is folded into the fingerprint — better to recompute
# a sample needlessly than to reuse one computed under different parameters.
_FINGERPRINT_EXCLUDED = frozenset({
    "n_workers",
    "alignment_rt_tolerance",
    "alignment_mz_tolerance",
    "alignment_mz_weight",
    "alignment_rt_weight",
    "alignment_ms2_weight",
    "alignment_ms2_identity_threshold",
    "alignment_ms2_identity_min_fragments",
    "alignment_ms2_identity_min_matched_fragments",
    "alignment_reference_sample",
    "alignment_ms1_covered_threshold",
    "gap_fill_enabled",
    "gap_fill_rt_expansion",
    "eic_store_top_fragments",
    "export_mgf",
    "export_msp",
    "export_report",
    "export_include_duplicates",
})


def _file_stamp(path: str) -> list:
    try:
        st = os.stat(path)
        return [os.path.basename(path), st.st_size, st.st_mtime_ns]
    except OSError:
        return [os.path.basename(path), None, None]


def config_fingerprint(
    config,
    library_path: Optional[str],
    mzml_paths: Iterable[str],
) -> str:
    """Hash of everything that can change a sample's stage 0-6.5 output.

    Covers the config (minus the alignment/export/worker knobs above), the
    library file's identity, and the sample's own mzML files (name + size +
    mtime). ``eic_mz_tolerance`` is *not* excluded even though stage 7 reads it
    too — stage 1 reads it as well.
    """
    payload = {
        "config": {
            k: v for k, v in sorted(asdict(config).items())
            if k not in _FINGERPRINT_EXCLUDED
        },
        "library": _file_stamp(library_path) if library_path else None,
        "inputs": [_file_stamp(p) for p in sorted(mzml_paths)],
    }
    blob = json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
