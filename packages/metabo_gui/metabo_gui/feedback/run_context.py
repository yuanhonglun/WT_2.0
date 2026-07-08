"""Helpers for building RunContext and signatures from app-side inputs.

Pure functions. No Qt. App orchestrators call build_run_context() at
run start; GUI layer maps app features to signatures.
"""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import Any

import numpy as np

from .models import FeatureSignature, RunContext


def common_input_root(paths: list[str]) -> str:
    if not paths:
        return ""
    # Get the directory of each file, then find the common path
    dirs = [os.path.dirname(str(p)) for p in paths]
    if not dirs or not any(dirs):
        return ""
    return os.path.commonpath(dirs)


def _coerce_for_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_coerce_for_json(v) for v in value.tolist()]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _coerce_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_for_json(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def params_to_jsonable(params: dict) -> dict:
    return {str(k): _coerce_for_json(v) for k, v in params.items()}


def build_run_context(
    *,
    app: str,
    metra_version: str,
    input_files: list[str],
    library_path: str | None = None,
    project_file: str | None = None,
    export_dir: str | None = None,
    params: dict | None = None,
    run_timestamp: str | None = None,
) -> RunContext:
    return RunContext(
        app=app,
        metra_version=metra_version,
        run_timestamp=run_timestamp or _dt.datetime.now().isoformat(timespec="seconds"),
        input_files=[str(p) for p in input_files],
        input_root=common_input_root(input_files),
        library_path=str(library_path) if library_path else None,
        project_file=str(project_file) if project_file else None,
        export_dir=str(export_dir) if export_dir else None,
        params=params_to_jsonable(params or {}),
    )


def feature_signature_from_components(*, mz: float, rt: float, mode: str) -> FeatureSignature:
    return FeatureSignature(mz=float(mz), rt=float(rt), mode=str(mode))
