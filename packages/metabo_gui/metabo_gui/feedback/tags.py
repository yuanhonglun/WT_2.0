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
    "peak_split": "峰被拆开",
    "noise": "噪音被识别",
    "missed_dup": "漏标重复",
    "false_dup": "错标重复",
    "wrong_annot": "注释错误",
    "ms2_excess": "MS2 过多",
    "ms2_insufficient": "MS2 太少",
    "other": "其它",
    VERIFIED_GOOD_TAG: "已确认无误",
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
