"""Dataclasses for the feedback system. Pure Python, no Qt, no I/O."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FeatureSignature:
    mz: float
    rt: float
    mode: str

    def to_dict(self) -> dict:
        return {"mz": self.mz, "rt": self.rt, "mode": self.mode}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSignature":
        return cls(mz=float(d["mz"]), rt=float(d["rt"]), mode=str(d["mode"]))


@dataclass
class FeedbackEntry:
    feature_id_at_run: str
    feature_signature: FeatureSignature
    tags: list[str]
    verified_good: bool
    comment: str
    created_at: str
    updated_at: str
    run_timestamp_created: str

    def to_dict(self) -> dict:
        return {
            "feature_id_at_run": self.feature_id_at_run,
            "feature_signature": self.feature_signature.to_dict(),
            "tags": list(self.tags),
            "verified_good": self.verified_good,
            "comment": self.comment,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "run_timestamp_created": self.run_timestamp_created,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeedbackEntry":
        return cls(
            feature_id_at_run=str(d["feature_id_at_run"]),
            feature_signature=FeatureSignature.from_dict(d["feature_signature"]),
            tags=list(d.get("tags", [])),
            verified_good=bool(d.get("verified_good", False)),
            comment=str(d.get("comment", "")),
            created_at=str(d["created_at"]),
            updated_at=str(d["updated_at"]),
            run_timestamp_created=str(d["run_timestamp_created"]),
        )


@dataclass
class RunContext:
    app: str
    software_version: str
    run_timestamp: str
    input_files: list[str]
    input_root: str
    library_path: str | None
    project_file: str | None
    export_dir: str | None
    params: dict

    def to_dict(self) -> dict:
        return {
            "app": self.app,
            "software_version": self.software_version,
            "run_timestamp": self.run_timestamp,
            "input_files": list(self.input_files),
            "input_root": self.input_root,
            "library_path": self.library_path,
            "project_file": self.project_file,
            "export_dir": self.export_dir,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunContext":
        return cls(
            app=str(d["app"]),
            software_version=str(d["software_version"]),
            run_timestamp=str(d["run_timestamp"]),
            input_files=list(d.get("input_files", [])),
            input_root=str(d.get("input_root", "")),
            library_path=d.get("library_path"),
            project_file=d.get("project_file"),
            export_dir=d.get("export_dir"),
            params=dict(d.get("params", {})),
        )


@dataclass
class FeedbackStore:
    schema_version: int
    app: str
    software_version: str
    run_context: RunContext
    entries: list[FeedbackEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "app": self.app,
            "software_version": self.software_version,
            "run_context": self.run_context.to_dict(),
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeedbackStore":
        outer_app = str(d["app"])
        outer_version = str(d["software_version"])
        ctx = RunContext.from_dict(d["run_context"])
        if ctx.app != outer_app or ctx.software_version != outer_version:
            logger.warning(
                "FeedbackStore field mismatch: outer (app=%s, version=%s) vs "
                "run_context (app=%s, version=%s); outer wins",
                outer_app, outer_version, ctx.app, ctx.software_version,
            )
        return cls(
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            app=outer_app,
            software_version=outer_version,
            run_context=ctx,
            entries=[FeedbackEntry.from_dict(e) for e in d.get("entries", [])],
        )
