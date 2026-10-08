"""
Worker-count and memory defaults shared by the parallel sharding paths.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

_MEMORY_FRACTION = 0.5  # default memory limit, as a fraction of total (or cgroup-limited) RAM
_FALLBACK_MEMORY = 4 * 10**9
_MIN_BYTES_PER_JOB = 256 * 10**6  # don't pay worker start-up costs for smaller jobs
_ROW_OVERHEAD = 16  # per-row bytes beyond the values: destination position + indptr entry
_SIZE_PREFIXES = {"": 0, "K": 1, "M": 2, "G": 3, "T": 4}


def _available_cpus() -> int:
    """CPUs this process may use (respects affinity masks, unlike ``os.cpu_count``)."""
    affinity = getattr(os, "sched_getaffinity", None)  # Linux only
    return len(affinity(0)) if affinity is not None else (os.cpu_count() or 1)


def _default_memory_limit() -> int:
    """Half of physical RAM, or of the cgroup memory limit if one applies."""
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return _FALLBACK_MEMORY
    for cgroup_file in (
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    ):
        try:
            with open(cgroup_file) as fh:
                total = min(total, int(fh.read().strip()))
        except (OSError, ValueError):  # missing, or "max" (unlimited)
            continue
    return int(total * _MEMORY_FRACTION)


def _parse_size(value: int | float | str) -> int:
    """
    Parse a byte count such as ``8_000_000_000``, ``"8GB"``, ``"512MB"`` or ``"1.5G"``.

    K, M, G and T are powers of 1000 (``"8GB"`` is 8 × 10⁹ bytes); with an ``i`` (``"8GiB"``)
    they are powers of 1024.
    """
    if isinstance(value, (int, float)):
        return int(value)
    match = re.fullmatch(r"\s*([\d.]+)\s*([KMGT]?)(I?)B?\s*", value, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot parse memory size {value!r}; use e.g. '8GB' or '512MB'.")
    base = 1024 if match.group(3) else 1000
    return int(float(match.group(1)) * base ** _SIZE_PREFIXES[match.group(2).upper()])


def _row_bytes(mat: Any) -> float:
    """Approximate in-memory bytes per row of a lazy matrix (dense array or sparse dataset)."""
    n_rows, n_cols = mat.shape
    if not hasattr(mat, "group"):  # dense h5py / zarr array
        return n_cols * mat.dtype.itemsize + _ROW_OVERHEAD
    group = mat.group
    item_bytes = group["data"].dtype.itemsize + group["indices"].dtype.itemsize
    return int(group["indptr"][-1]) / max(n_rows, 1) * item_bytes + _ROW_OVERHEAD


def _resolve_resources(
    n_jobs: int | None, memory_limit: int | float | str | None, total_bytes: float
) -> tuple[int, int]:
    """
    Resolve the requested worker count and memory limit into concrete numbers.

    ``n_jobs=None`` picks one worker per available CPU, fewer when *total_bytes* (the in-memory
    size of the data to process) is too small to be worth the worker start-up cost; the planners
    then reduce it further to fit the memory limit.  ``memory_limit=None`` uses half of the
    available RAM.
    """
    if n_jobs is None:
        n_jobs = min(_available_cpus(), max(1, int(total_bytes // _MIN_BYTES_PER_JOB)))
    elif n_jobs < 1:
        raise ValueError(f"n_jobs must be at least 1, got {n_jobs}.")
    limit = _default_memory_limit() if memory_limit is None else _parse_size(memory_limit)
    return n_jobs, limit


def _release_workers() -> None:
    """Shut down joblib's idle worker processes so they don't hold on to memory."""
    try:
        from joblib.externals.loky import get_reusable_executor

        get_reusable_executor().shutdown(wait=True)
    except Exception:  # best-effort cleanup only
        logger.debug("Could not shut down joblib workers.", exc_info=True)
