"""Stage 0: Load and organize mzML files with flexible sample grouping."""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Optional, Callable

from asfam.config import ProcessingConfig
from asfam.models import RawSegmentData
from asfam.io.mzml_reader import load_mzml, parse_filename
from metabo_core.utils.memlog import log_memory

logger = logging.getLogger(__name__)

# In frozen (PyInstaller) mode, multiprocessing.Pool on Windows causes
# issues (unpicklable exceptions, re-spawned GUI windows). Use sequential.
_FROZEN = getattr(sys, 'frozen', False)

# Raw segment data is the largest single allocation in the pipeline (15.7 GiB
# for a 6-sample full-range run), so sample RSS as it accumulates rather than
# only at the stage boundary.
_MEM_LOG_EVERY = 20


def group_files_by_sample(
    mzml_paths: list[str],
    sample_groups: Optional[dict] = None,
) -> dict[str, list[str]]:
    """Map sample_id -> mzML paths, *without reading a single file*.

    The per-sample loop has to know the grouping before it loads anything, and
    it can: ``RawSegmentData.replicate_id`` is nothing but
    ``parse_filename(path)["rep"]``, so the filename alone decides the sample.
    Sole definition of the grouping — :func:`run_stage0` defers to it too.

    ``sample_groups`` maps a user-facing sample name to its file paths; groups
    are numbered "1", "2", ... in iteration order, and any input file left over
    becomes its own ``ungrouped_<stem>`` sample.
    """
    if sample_groups:
        return _group_paths_by_custom_groups(mzml_paths, sample_groups)

    by_sample: dict[str, list[str]] = {}
    for path in mzml_paths:
        try:
            rep = parse_filename(path)["rep"]
        except ValueError as e:
            raise RuntimeError(f"Cannot group {Path(path).name}: {e}") from e
        by_sample.setdefault(str(rep), []).append(path)
    return by_sample


def _group_paths_by_custom_groups(
    mzml_paths: list[str],
    sample_groups: dict,
) -> dict[str, list[str]]:
    """Resolve user-defined sample groups against the input list, by path or name."""
    by_name = {Path(p).name: p for p in mzml_paths}
    known = set(mzml_paths)

    by_sample: dict[str, list[str]] = {}
    for file_paths in sample_groups.values():
        resolved = []
        for fp in file_paths:
            if fp in known:
                resolved.append(fp)
            elif Path(fp).name in by_name:
                resolved.append(by_name[Path(fp).name])
            else:
                logger.warning("File not found in input list: %s (skipping)", fp)
        if resolved:
            by_sample[str(len(by_sample) + 1)] = resolved

    assigned = set()
    for file_paths in sample_groups.values():
        assigned.update(file_paths)
        assigned.update(Path(fp).name for fp in file_paths)

    for path in mzml_paths:
        if path not in assigned and Path(path).name not in assigned:
            by_sample[f"ungrouped_{Path(path).stem}"] = [path]
            logger.warning("File %s not in any group, treated as separate sample",
                           Path(path).name)

    return by_sample


def run_stage0_one_sample(
    mzml_paths: list[str],
    config: ProcessingConfig,
    progress_callback: Optional[Callable] = None,
) -> list[RawSegmentData]:
    """Load the segments of a single sample, sorted by ``segment_low``.

    The multiprocessing pool stays *inside* one sample: a full-range ASFAM
    sample is 31 segments, which is enough to keep the workers busy, and the
    caller frees the whole list before the next sample is loaded.
    """
    loaded = _load_paths(mzml_paths, config, progress_callback)
    segments = [loaded[p] for p in mzml_paths if p in loaded]
    segments.sort(key=lambda d: d.segment_low)
    return segments


def _load_paths(
    mzml_paths: list[str],
    config: ProcessingConfig,
    progress_callback: Optional[Callable] = None,
) -> dict[str, RawSegmentData]:
    """Read every path into a ``{path: RawSegmentData}`` dict."""
    loaded: dict[str, RawSegmentData] = {}
    use_parallel = config.n_workers > 1 and len(mzml_paths) > 1 and not _FROZEN
    if use_parallel:
        from multiprocessing import Pool
        with Pool(processes=min(config.n_workers, len(mzml_paths))) as pool:
            for i, result in enumerate(pool.imap(_load_wrapper, mzml_paths)):
                loaded[mzml_paths[i]] = result
                if progress_callback:
                    progress_callback("stage0", i + 1, len(mzml_paths),
                                      f"Loaded {Path(mzml_paths[i]).name}")
                if (i + 1) % _MEM_LOG_EVERY == 0:
                    log_memory(logger, f"stage0 loaded {i + 1}/{len(mzml_paths)}")
    else:
        for i, path in enumerate(mzml_paths):
            try:
                loaded[path] = load_mzml(path, config)
            except Exception as e:
                logger.warning("Failed to load %s: %s (skipping)", Path(path).name, e)
                continue
            if progress_callback:
                progress_callback("stage0", i + 1, len(mzml_paths),
                                  f"Loaded {Path(path).name}")
            if (i + 1) % _MEM_LOG_EVERY == 0:
                log_memory(logger, f"stage0 loaded {i + 1}/{len(mzml_paths)}")
    return loaded


def run_stage0(
    mzml_paths: list[str],
    config: ProcessingConfig,
    progress_callback: Optional[Callable] = None,
    sample_groups: Optional[dict] = None,
) -> dict[str, list[RawSegmentData]]:
    """Load *all* mzML files at once, organized by sample (replicate).

    The pipeline loads one sample at a time (see :func:`run_stage0_one_sample`);
    this whole-dataset variant remains for callers that genuinely want every
    segment resident. Grouping goes through :func:`group_files_by_sample` so
    there is a single definition of "which file belongs to which sample".

    Returns
    -------
    dict mapping sample_id (str) -> list[RawSegmentData] (sorted by segment_low)
    """
    logger.info("Stage 0: Loading %d mzML files...", len(mzml_paths))

    loaded = _load_paths(mzml_paths, config, progress_callback)

    if not loaded:
        raise RuntimeError("No mzML files could be loaded")

    by_sample: dict[str, list[RawSegmentData]] = {}
    for sample_id, paths in group_files_by_sample(list(loaded), sample_groups).items():
        segments = [loaded[p] for p in paths]
        segments.sort(key=lambda d: d.segment_low)
        by_sample[sample_id] = segments

    for sample_id, segments in sorted(by_sample.items()):
        seg_names = [s.segment_name for s in segments]
        total_cycles = sum(s.n_cycles for s in segments)
        logger.info("  Sample %s: segments %s, total %d cycles",
                     sample_id, seg_names, total_cycles)

    return by_sample


def _load_wrapper(path: str) -> RawSegmentData:
    """Wrapper for multiprocessing (top-level function)."""
    return load_mzml(path)
