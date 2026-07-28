"""Finding a project's companion files after the data tree has been moved.

A ``.asfam`` file records absolute paths: ``<output_dir>/_work``, and the
directory ``alignment.eic`` was written to. Both die the moment the run is
archived somewhere else, which is the *normal* end of a run — the analysis
happens on a fast local disk and the result is copied to an external one. The
recorded path then names a machine that no longer exists::

    recorded:  D:\\runs\\study_a\\result_0722\\_work
    actually:  E:\\archive\\study_a\\result_0722\\_work

So a recorded directory is a hint, not an address. :func:`relocate` re-anchors
it on the directory the project file was actually opened from, trying
progressively longer tails of the old path — one component catches a project
sitting inside the result folder, two catch a whole result folder that moved
with its siblings.

Anchoring on the project file (rather than, say, the current working directory)
is what makes this safe: the answer only ever comes from a directory the user
themselves pointed the Open dialog at.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterator, Optional

#: How many trailing components of a recorded path to try re-anchoring.
#: 2 covers ``result_xxx/_work``; 3 leaves room for one more nesting level.
MAX_TAIL = 3


def relocation_candidates(recorded, anchor=None) -> Iterator[Path]:
    """Yield every place ``recorded`` could be now, best guess first.

    The recorded path itself comes first — most runs are opened on the machine
    that produced them and nothing needs relocating. ``anchor`` is the directory
    the project file was opened from; ``None`` means there is nothing to
    re-anchor on and only the recorded path is offered.

    Duplicates are suppressed, so a caller may probe the filesystem once per
    yielded path.
    """
    seen: set[str] = set()

    def _fresh(path: Path) -> bool:
        key = os.path.normcase(str(path))
        if key in seen:
            return False
        seen.add(key)
        return True

    recorded_path = Path(recorded) if recorded else None
    if recorded_path is not None and _fresh(recorded_path):
        yield recorded_path

    if anchor is None or recorded_path is None:
        return

    anchor_path = Path(anchor)
    parts = recorded_path.parts
    # parts[0] on Windows is the drive ("D:\\"); re-anchoring it would just
    # rebuild the recorded path, so tails stop one short of the full path.
    for k in range(1, min(MAX_TAIL, len(parts) - 1) + 1):
        tail = Path(*parts[-k:])
        for base in (anchor_path, anchor_path.parent):
            candidate = base / tail
            if _fresh(candidate):
                yield candidate


def relocate(
    recorded,
    anchor=None,
    is_match: Optional[Callable[[Path], bool]] = None,
) -> Optional[Path]:
    """First candidate for ``recorded`` that ``is_match`` accepts, else ``None``.

    ``is_match`` defaults to "is a directory". Pass something stricter when an
    empty look-alike would be worse than finding nothing — an empty ``_work``
    beside the project is a directory too, and picking it would report zero
    samples instead of relocating to the real one.
    """
    check = is_match or (lambda p: p.is_dir())
    for candidate in relocation_candidates(recorded, anchor):
        try:
            if check(candidate):
                return candidate
        except OSError:
            continue
    return None
