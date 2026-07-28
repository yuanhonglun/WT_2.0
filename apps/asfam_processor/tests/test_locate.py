"""Re-anchoring the absolute paths a project file records.

A run is normally produced on a fast local disk and then archived to another
one. Every path in the ``.asfam`` — ``_work/``, the directory ``alignment.eic``
went to — is absolute, so after the move they all name a machine that no longer
exists. That is not an error case; it is the ordinary way a project gets opened
a second time.
"""
from __future__ import annotations

from pathlib import Path

from asfam.io.locate import relocate, relocation_candidates


def test_the_recorded_path_is_offered_first(tmp_path):
    """Most runs never move; probing elsewhere first would be pure cost."""
    recorded = tmp_path / "result" / "_work"
    assert next(relocation_candidates(recorded, tmp_path)) == recorded


def test_no_anchor_means_nothing_to_re_anchor_on(tmp_path):
    recorded = tmp_path / "result" / "_work"
    assert list(relocation_candidates(recorded, None)) == [recorded]


def test_candidates_are_offered_once_each(tmp_path):
    seen = list(relocation_candidates(tmp_path / "a" / "b", tmp_path))
    assert len(seen) == len({str(p) for p in seen})


def test_a_project_inside_the_result_folder_finds_its_work_dir(tmp_path):
    """The project now auto-saves into the output directory, so a one-component
    tail is the common case: ``<project dir>/_work``."""
    moved = tmp_path / "result_0722" / "_work"
    moved.mkdir(parents=True)
    found = relocate(r"D:\gone\study_a\result_0722\_work",
                     anchor=tmp_path / "result_0722")
    assert found == moved


def test_a_whole_result_folder_that_moved_is_followed(tmp_path):
    """The archived-run case: the project sits beside the raw data and the
    result folder came along with it under a different root."""
    moved = tmp_path / "result_0722" / "_work"
    moved.mkdir(parents=True)
    found = relocate(r"D:\runs\study_a\result_0722\_work", anchor=tmp_path)
    assert found == moved


def test_the_anchor_s_parent_is_searched_too(tmp_path):
    """Raw mzML files sit one level up from a project saved in the output dir."""
    raw = tmp_path / "CK_75-104_P_1.mzML"
    raw.write_bytes(b"")
    found = relocate(r"D:\runs\study_a\CK_75-104_P_1.mzML",
                     anchor=tmp_path / "result_0722", is_match=Path.is_file)
    assert found == raw


def test_a_look_alike_can_be_rejected_by_the_predicate(tmp_path):
    """An empty ``_work`` beside the project is a directory too. Taking it would
    report zero samples instead of relocating to the one with the checkpoints."""
    (tmp_path / "_work").mkdir()
    real = tmp_path / "result_0722" / "_work"
    real.mkdir(parents=True)
    (real / "s1.json").write_text("{}")

    found = relocate(
        r"D:\gone\study_a\result_0722\_work", anchor=tmp_path,
        is_match=lambda p: p.is_dir() and any(p.glob("*.json")),
    )
    assert found == real


def test_nothing_anywhere_is_none_not_a_guess(tmp_path):
    assert relocate(r"D:\gone\study_a\result\_work", anchor=tmp_path) is None


def test_a_missing_recorded_path_is_not_an_error(tmp_path):
    assert relocate(None, anchor=tmp_path) is None
    assert list(relocation_candidates(None, tmp_path)) == []
