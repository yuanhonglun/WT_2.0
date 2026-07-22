"""Memory instrumentation for pipeline stage boundaries.

Pipelines log ``[mem]`` lines around every stage so that a crash log tells us
whether the process was climbing toward the machine's limit. The distinction
that matters when reading a crash log:

* traceback with ``MemoryError``  -> Python's allocator refused a request
* no traceback, ``[mem]`` line showing RSS near the machine limit -> the OS
  killed the process

Without the ``[mem]`` lines the second case is indistinguishable from any other
silent death.

``psutil`` is declared as a dependency of ``metabo-core``, but it carries a
native extension and can go missing from a PyInstaller bundle. Rather than
taking the pipeline down with it, the helpers below degrade to a no-op and warn
once, which is also how a broken frozen build announces itself.
"""
from __future__ import annotations

import logging
from math import nan

try:  # pragma: no cover - exercised only when the dependency is absent
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

_missing_warned = False


def _psutil_or_none():
    """Return the ``psutil`` module, warning once if it is unavailable."""
    global _missing_warned
    if psutil is None and not _missing_warned:
        _missing_warned = True
        logging.getLogger(__name__).warning(
            "[mem] psutil is unavailable; memory instrumentation is disabled. "
            "In a frozen build this usually means psutil is missing from "
            "hiddenimports."
        )
    return psutil


def rss_gib() -> float:
    """Resident set size of the current process, in GiB.

    Returns NaN when ``psutil`` is unavailable, so callers can format the value
    unconditionally.
    """
    ps = _psutil_or_none()
    if ps is None:
        return nan
    return ps.Process().memory_info().rss / 2 ** 30


def log_memory(logger: logging.Logger, label: str) -> None:
    """Log process RSS and system available memory at a stage boundary."""
    ps = _psutil_or_none()
    if ps is None:
        return
    vm = ps.virtual_memory()
    logger.info(
        "[mem] %s: RSS=%.2f GiB, available=%.2f GiB (%.0f%% used)",
        label,
        ps.Process().memory_info().rss / 2 ** 30,
        vm.available / 2 ** 30,
        vm.percent,
    )
