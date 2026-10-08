"""
Parallel sharding into contiguous row ranges (no shuffle).

Shard ``j`` is rows ``[j * shard_size, (j + 1) * shard_size)`` of the input, so shards are
independent: each worker opens the input read-only, reads its shard with one slice per matrix
and writes its own output file.  Nothing is shared between workers and no scratch space is used.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from joblib import delayed

from annslicer._common import (
    LazyData,
    _lazy_matrix,
    _open_root,
    _read_rows,
    _take_rows,
    _write_h5ad_shard,
)
from annslicer._parallel import _Progress, run_parallel
from annslicer._resources import _release_workers, _resolve_resources, _row_bytes

logger = logging.getLogger(__name__)

_SHARD_MEM_FACTOR = 2  # the shard as read + the copies made while assembling and writing it


def _write_range(
    input_file: str,
    keys: list[str],
    start: int,
    stop: int,
    obs: pd.DataFrame,
    obsm: dict[str, Any],
    var: pd.DataFrame,
    uns: dict[str, Any],
    out_filename: str,
    compression: str | None,
) -> None:
    """Read input rows ``[start, stop)`` of the matrices *keys* and write them as one shard."""
    root = _open_root(input_file)
    try:
        mats = {key: _read_rows(_lazy_matrix(root, key), slice(start, stop)) for key in keys}
    finally:
        if hasattr(root, "close"):
            root.close()
    layers = {k[len("layers/") :]: v for k, v in mats.items() if k != "X"}
    _write_h5ad_shard(mats.get("X"), layers, obs, var, obsm, uns, out_filename, compression)


def contiguous_shards(
    input_file: str,
    data: LazyData,
    out_names: list[str],
    shard_size: int,
    compression: str | None = None,
    n_jobs: int | None = None,
    memory_limit: int | float | str | None = None,
) -> None:
    """
    Write ``out_names[j]`` as the ``j``-th run of ``shard_size`` consecutive cells of *data*.

    ``n_jobs=None`` picks ``min(CPUs, 8)`` (fewer for small inputs); ``memory_limit=None`` uses
    half of the available RAM.  The memory limit caps how many workers run at once, given that
    each holds about one shard; it is a sizing target, not a hard cap.
    """
    n_obs = data.n_obs
    n_shards = math.ceil(n_obs / shard_size)
    keys = list(data.matrices())
    row_bytes = sum(_row_bytes(mat) for mat in data.matrices().values())
    n_jobs, limit = _resolve_resources(n_jobs, memory_limit, row_bytes * n_obs)

    shard_bytes = _SHARD_MEM_FACTOR * shard_size * row_bytes
    if shard_bytes > limit:
        logger.warning(
            "A single shard needs about %.1f GB to write, more than the %.1f GB memory limit; "
            "use a smaller --size or a larger --memory-limit.",
            shard_bytes / 1024**3,
            limit / 1024**3,
        )
    n_jobs = max(1, min(n_jobs, n_shards, int(limit // max(shard_bytes, 1))))
    logger.info("Writing %d shards with %d worker(s).", n_shards, n_jobs)

    def tasks():
        for j, out_filename in enumerate(out_names):
            start, stop = j * shard_size, min((j + 1) * shard_size, n_obs)
            rows = np.arange(start, stop)
            obsm = {k: _take_rows(v, rows) for k, v in data.obsm.items()}
            yield delayed(_write_range)(
                input_file,
                keys,
                start,
                stop,
                data.obs.iloc[start:stop],
                obsm,
                data.var,
                data.uns,
                out_filename,
                compression,
            )

    try:
        run_parallel(tasks(), n_jobs, _Progress("Writing shards", n_shards, "shards"))
    finally:
        if n_jobs > 1:
            _release_workers()
