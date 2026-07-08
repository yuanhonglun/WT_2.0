"""Shared auto-worker count helper.

All three apps used to expose a "Workers" spin box in the GUI General
tab; defaults were 4 and users almost never touched it. Detecting the
machine's logical cores automatically is more reliable and removes a
parameter that was perpetual UI noise.

``auto_worker_count`` returns ``min(cap, max(1, os.cpu_count() - 1))``.
Cap defaults to 8 to keep memory pressure bounded on workstations with
many cores running on big mzML files.

Override via the ``METABO_WORKERS`` env var if you need to.
"""
from __future__ import annotations

import os


def auto_worker_count(cap: int = 8) -> int:
    """Return the recommended worker pool size for this machine."""
    env_override = os.environ.get("METABO_WORKERS")
    if env_override:
        try:
            n = int(env_override)
            if n >= 1:
                return n
        except ValueError:
            pass
    cores = os.cpu_count() or 1
    return min(cap, max(1, cores - 1))
