"""Write feedback.csv + feedback.json into the user's export directory."""
from __future__ import annotations

import csv
import datetime as _dt
import json
import os
from pathlib import Path

from .models import FeedbackStore

CSV_COLUMNS = (
    "feature_id", "mz", "rt", "mode",
    "tags", "verified_good", "comment",
    "created_at", "updated_at",
)


def dump_feedback_to_export_dir(export_dir: str | os.PathLike, store: FeedbackStore) -> None:
    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "feedback.csv", store)
    _write_json(out / "feedback.json", store)


def _write_csv(path: Path, store: FeedbackStore) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for e in store.entries:
            w.writerow([
                e.feature_id_at_run,
                e.feature_signature.mz,
                e.feature_signature.rt,
                e.feature_signature.mode,
                "|".join(e.tags),
                str(e.verified_good),
                e.comment,
                e.created_at,
                e.updated_at,
            ])


def _write_json(path: Path, store: FeedbackStore) -> None:
    payload = store.to_dict()
    payload["exported_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
