"""Tag catalog for feedback annotations.

Adding a new issue tag = append to ISSUE_TAGS + add labels.
"""
from __future__ import annotations

ISSUE_TAGS: tuple[str, ...] = (
    "peak_split",
    "noise",
    "missed_dup",
    "false_dup",
    "wrong_annot",
    "ms2_excess",
    "ms2_insufficient",
    "other",
)

VERIFIED_GOOD_TAG = "verified_good"

TAG_LABELS: dict[str, str] = {
    "peak_split": "Peak split",
    "noise": "Noise",
    "missed_dup": "Missed duplicate",
    "false_dup": "False duplicate",
    "wrong_annot": "Wrong annotation",
    "ms2_excess": "MS2 excess",
    "ms2_insufficient": "MS2 insufficient",
    "other": "Other",
    VERIFIED_GOOD_TAG: "Verified good",
}

STATUS_COLUMN_LABELS: dict[str, str] = {
    "peak_split": "split",
    "noise": "noise",
    "missed_dup": "dup-",
    "false_dup": "dup+",
    "wrong_annot": "annot",
    "ms2_excess": "ms2+",
    "ms2_insufficient": "ms2-",
    "other": "other",
}
