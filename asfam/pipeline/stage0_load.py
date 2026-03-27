"""Stage 0: Load and organize mzML files with flexible sample grouping."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from multiprocessing import Pool
from typing import Optional, Callable

from asfam.config import ProcessingConfig
from asfam.models import RawSegmentData
from asfam.io.mzml_reader import load_mzml

logger = logging.getLogger(__name__)


def run_stage0(
    mzml_paths: list[str],
    config: ProcessingConfig,
    progress_callback: Optional[Callable] = None,
    sample_groups: Optional[dict] = None,
) -> dict[str, list[RawSegmentData]]:
    """Load all mzML files, organized by sample (replicate).

    Parameters
    ----------
    mzml_paths : list of file paths
    config : ProcessingConfig
    progress_callback : optional callback
    sample_groups : optional dict mapping sample_name -> [file_paths].
        If None, files are auto-grouped by replicate ID in filename.

    Returns
    -------
    dict mapping sample_id (str) -> list[RawSegmentData] (sorted by segment_low)
    """
    logger.info("Stage 0: Loading %d mzML files...", len(mzml_paths))

    # Load all files
    loaded: dict[str, RawSegmentData] = {}  # filepath -> RawSegmentData
    if config.n_workers > 1 and len(mzml_paths) > 1:
        with Pool(processes=min(config.n_workers, len(mzml_paths))) as pool:
            for i, result in enumerate(pool.imap(_load_wrapper, mzml_paths)):
                loaded[mzml_paths[i]] = result
                if progress_callback:
                    progress_callback("stage0", i + 1, len(mzml_paths),
                                      f"Loaded {Path(mzml_paths[i]).name}")
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

    if not loaded:
        raise RuntimeError("No mzML files could be loaded")

    # Organize by sample
    if sample_groups:
        by_sample = _organize_by_custom_groups(loaded, sample_groups)
    else:
        by_sample = _organize_by_replicate(loaded)

    # Sort segments within each sample by segment_low
    for sample_id in by_sample:
        by_sample[sample_id].sort(key=lambda d: d.segment_low)

    # Log summary
    for sample_id, segments in sorted(by_sample.items()):
        seg_names = [s.segment_name for s in segments]
        total_cycles = sum(s.n_cycles for s in segments)
        logger.info("  Sample %s: segments %s, total %d cycles",
                     sample_id, seg_names, total_cycles)

    return by_sample


def _organize_by_replicate(loaded: dict[str, RawSegmentData]) -> dict[str, list[RawSegmentData]]:
    """Auto-group by replicate ID from RawSegmentData."""
    by_rep: dict[str, list[RawSegmentData]] = {}
    for path, data in loaded.items():
        key = str(data.replicate_id)
        if key not in by_rep:
            by_rep[key] = []
        by_rep[key].append(data)
    return by_rep


def _organize_by_custom_groups(
    loaded: dict[str, RawSegmentData],
    sample_groups: dict,
) -> dict[str, list[RawSegmentData]]:
    """Organize using user-defined sample groups."""
    by_sample: dict[str, list[RawSegmentData]] = {}

    for sample_name, file_paths in sample_groups.items():
        segments = []
        for fp in file_paths:
            # Match by full path or filename
            data = loaded.get(fp)
            if data is None:
                # Try matching by filename only
                for loaded_path, loaded_data in loaded.items():
                    if Path(loaded_path).name == Path(fp).name:
                        data = loaded_data
                        break
            if data is not None:
                segments.append(data)
            else:
                logger.warning("File not found in loaded data: %s (skipping)", fp)

        if segments:
            # Use a simple ID for internal tracking
            sample_id = str(len(by_sample) + 1)
            by_sample[sample_id] = segments

    # Handle any loaded files not in any group
    assigned = set()
    for file_paths in sample_groups.values():
        assigned.update(file_paths)
        assigned.update(Path(fp).name for fp in file_paths)

    for path, data in loaded.items():
        if path not in assigned and Path(path).name not in assigned:
            key = f"ungrouped_{Path(path).stem}"
            by_sample[key] = [data]
            logger.warning("File %s not in any group, treated as separate sample",
                           Path(path).name)

    return by_sample


def _load_wrapper(path: str) -> RawSegmentData:
    """Wrapper for multiprocessing (top-level function)."""
    return load_mzml(path)
