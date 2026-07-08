"""ISSUE_TAGS catalog matches spec §5."""
from __future__ import annotations

from metabo_gui.feedback.tags import (
    ISSUE_TAGS,
    TAG_LABELS,
    VERIFIED_GOOD_TAG,
    STATUS_COLUMN_LABELS,
)


def test_v1_issue_tags_exact_set():
    assert set(ISSUE_TAGS) == {
        "peak_split",
        "noise",
        "missed_dup",
        "false_dup",
        "wrong_annot",
        "ms2_excess",
        "ms2_insufficient",
        "other",
    }


def test_issue_tags_have_stable_order():
    assert ISSUE_TAGS[0] == "peak_split"
    assert ISSUE_TAGS[-1] == "other"


def test_every_tag_has_label():
    for tag in ISSUE_TAGS:
        assert tag in TAG_LABELS
        assert TAG_LABELS[tag]


def test_verified_good_is_independent_constant():
    assert VERIFIED_GOOD_TAG == "verified_good"
    assert VERIFIED_GOOD_TAG not in ISSUE_TAGS


def test_status_column_has_short_label_for_every_issue_tag():
    for tag in ISSUE_TAGS:
        assert tag in STATUS_COLUMN_LABELS
        assert len(STATUS_COLUMN_LABELS[tag]) <= 8
