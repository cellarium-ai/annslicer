"""
Out-of-core shuffled sharding: a two-pass scatter / gather.

Fetching every shard's cells at random from the input costs a small read per row (and, for
compressed input, a chunk decompression per row), which adds up to many passes over the file.
Instead:

* **Pass 1 (scatter)** streams the input in contiguous row blocks.  Each block is reordered by
  destination *bucket* (a bucket is a run of consecutive output shards) and written to scratch
  as raw ``.npy`` files, one set per block.  Blocks are independent, so they run in parallel and
  never write to the same file.
* **Pass 2 (gather)** reads one bucket at a time from all the block files (a contiguous row
  range in each), puts its cells in their final order, and writes the output shards.  Buckets
  are independent, so they also run in parallel.

Every cell's output position comes from one global permutation, ``default_rng(seed).permutation``,
so the shards are the same cells in the same order as a direct gather of that permutation, no
matter how many workers, blocks or buckets are used.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
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

_MAX_BLOCK_ROWS = 50_000
_MIN_SEGMENT_ROWS = 64  # target minimum rows a block contributes to each bucket
_BLOCK_MEM_FACTOR = 3  # block + its reordered copy + write buffers
_BUCKET_MEM_FACTOR = 4  # gathered pieces + reordered copy + AnnData assembly + write


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Plan:
    n_jobs: int
    block_rows: int
    shards_per_bucket: int


def _plan(
    n_obs: int,
    shard_size: int,
    row_bytes: float,
    n_jobs: int,
    memory_limit: int,
    block_rows: int | None = None,
    shards_per_bucket: int | None = None,
) -> _Plan:
    """
    Choose the worker count, pass-1 block size and bucket width from a memory budget.

    ``row_bytes`` is the in-memory size of one cell across all matrices.  A bucket is a whole
    number of shards; narrower buckets need less memory in pass 2, wider ones make pass 1's
    per-bucket writes larger.  Explicit ``block_rows`` / ``shards_per_bucket`` override the
    heuristics (used by tests).
    """
    n_shards = math.ceil(n_obs / shard_size)
    shard_bytes = max(1, int(shard_size * row_bytes))
    jobs = max(1, min(n_jobs, n_shards))

    if block_rows is None:
        by_memory = int(memory_limit // (jobs * _BLOCK_MEM_FACTOR * max(row_bytes, 1)))
        block_rows = max(1, min(by_memory, _MAX_BLOCK_ROWS, math.ceil(n_obs / (2 * jobs))))

    if shards_per_bucket is None:
        wanted = math.ceil(n_shards * _MIN_SEGMENT_ROWS / block_rows)
        fits = max(1, int(memory_limit // (jobs * _BUCKET_MEM_FACTOR * shard_bytes)))
        shards_per_bucket = max(1, min(wanted, fits, n_shards))

    bucket_bytes = _BUCKET_MEM_FACTOR * shards_per_bucket * shard_bytes
    if bucket_bytes > memory_limit:
        logger.warning(
            "A single shard needs about %.1f GB to assemble, more than the %.1f GB memory limit; "
            "use a smaller --size or a larger --memory-limit.",
            bucket_bytes / 1024**3,
            memory_limit / 1024**3,
        )
    jobs = max(1, min(jobs, int(memory_limit // bucket_bytes)))
    return _Plan(jobs, block_rows, shards_per_bucket)


def _describe(key: str, mat: Any) -> tuple[bool, int, float]:
    """Return ``(is_sparse, n_cols, approx spill bytes per row)`` for a lazy matrix."""
    is_sparse = hasattr(mat, "group")  # otherwise a dense h5py / zarr array
    if is_sparse and mat.group.attrs.get("encoding-type") != "csr_matrix":
        raise ValueError(
            f"{key!r} is stored as {mat.group.attrs.get('encoding-type')!r}; shuffled sharding "
            f"needs row-compressed (CSR) or dense storage. Convert it to CSR first."
        )
    return is_sparse, mat.shape[1], _row_bytes(mat)


# ---------------------------------------------------------------------------
# Scratch-file helpers
# ---------------------------------------------------------------------------


def _scratch_file(scratch: str, block: int, name: str) -> str:
    return os.path.join(scratch, f"b{block}_{name}.npy")


def _read_into(path: str, out: np.ndarray, first: int) -> np.ndarray:
    """
    Fill the contiguous array *out* with elements ``first : first + out.size`` of the
    ``.npy`` file at *path* (flattened, C order) and return it.

    Plain sequential ``readinto`` calls into a preallocated array are several times faster
    than concatenating slices of memory-mapped files, which page-faults its way through.
    """
    if out.size == 0:
        return out
    buf = out.reshape(-1).data.cast("B")
    with open(path, "rb", buffering=0) as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            np.lib.format.read_array_header_1_0(f)
        else:
            np.lib.format.read_array_header_2_0(f)
        f.seek(first * out.itemsize, os.SEEK_CUR)
        done = 0
        while done < len(buf):
            n = f.readinto(buf[done:])
            if not n:
                raise EOFError(f"Scratch file {path} is truncated.")
            done += n
    return out


def _npy_dtype(path: str) -> np.dtype:
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        reader = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        return reader(f)[2]


# ---------------------------------------------------------------------------
# Pass 1: scatter a block of input rows into bucket order
# ---------------------------------------------------------------------------


def _scatter_block(
    path: str,
    keys: list[str],
    scratch: str,
    block: int,
    start: int,
    stop: int,
    bucket_rows: int,
    n_buckets: int,
) -> np.ndarray:
    """
    Read input rows ``[start, stop)`` and save them to scratch, sorted by destination bucket.

    Returns ``bounds`` of length ``n_buckets + 1``: after sorting, bucket ``b``'s rows are
    ``bounds[b]:bounds[b + 1]`` in every file written for this block.
    """
    dest = np.asarray(np.load(os.path.join(scratch, "dest.npy"), mmap_mode="r")[start:stop])
    bucket = dest // bucket_rows
    order = np.argsort(bucket, kind="stable")
    bounds = np.searchsorted(bucket[order], np.arange(n_buckets + 1))
    np.save(_scratch_file(scratch, block, "dest"), dest[order])

    root = _open_root(path)
    try:
        for m, key in enumerate(keys):
            rows = _read_rows(_lazy_matrix(root, key), slice(start, stop))[order]
            if sp.issparse(rows):
                np.save(_scratch_file(scratch, block, f"{m}_data"), rows.data)
                np.save(_scratch_file(scratch, block, f"{m}_indices"), rows.indices)
                np.save(_scratch_file(scratch, block, f"{m}_indptr"), rows.indptr)
            else:
                np.save(_scratch_file(scratch, block, f"{m}_dense"), rows)
            del rows
    finally:
        if hasattr(root, "close"):
            root.close()
    return bounds


# ---------------------------------------------------------------------------
# Pass 2: gather one bucket, order it, write its shards
# ---------------------------------------------------------------------------


def _gather(scratch: str, m: int, is_sparse: bool, n_cols: int, pieces: list) -> Any:
    """Read matrix *m*'s rows for one bucket from every block file (rows in file order)."""
    n_rows = sum(hi - lo for _, lo, hi in pieces)
    if not is_sparse:
        path = _scratch_file(scratch, pieces[0][0], f"{m}_dense")
        out = np.empty((n_rows, n_cols), dtype=_npy_dtype(path))
        row = 0
        for i, lo, hi in pieces:
            piece = out[row : row + hi - lo]
            _read_into(_scratch_file(scratch, i, f"{m}_dense"), piece, lo * n_cols)
            row += hi - lo
        return out

    # Row boundaries first (small), so the data / indices arrays can be allocated once.
    spans, row_nnz = [], []
    for i, lo, hi in pieces:
        path = _scratch_file(scratch, i, f"{m}_indptr")
        indptr = _read_into(path, np.empty(hi - lo + 1, dtype=_npy_dtype(path)), lo)
        spans.append((int(indptr[0]), int(indptr[-1])))
        row_nnz.append(np.diff(indptr))
    data_path = _scratch_file(scratch, pieces[0][0], f"{m}_data")
    indices_path = _scratch_file(scratch, pieces[0][0], f"{m}_indices")
    nnz = sum(last - first for first, last in spans)
    data = np.empty(nnz, dtype=_npy_dtype(data_path))
    indices = np.empty(nnz, dtype=_npy_dtype(indices_path))
    pos = 0
    for (i, _, _), (first, last) in zip(pieces, spans, strict=True):
        end = pos + last - first
        _read_into(_scratch_file(scratch, i, f"{m}_data"), data[pos:end], first)
        _read_into(_scratch_file(scratch, i, f"{m}_indices"), indices[pos:end], first)
        pos = end
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(np.concatenate(row_nnz), out=indptr[1:])
    return sp.csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))


def _write_bucket(
    scratch: str,
    specs: list[tuple[str, bool, int]],
    bucket_bounds: np.ndarray,
    base: int,
    payloads: list,
    shard_size: int,
    var: Any,
    uns: dict[str, Any],
    compression: str | None,
) -> int:
    """
    Assemble one bucket and write its shards; return how many were written.

    ``bucket_bounds`` has one ``(lo, hi)`` row range per block file; ``base`` is the output
    position of the bucket's first cell; ``payloads`` holds ``(filename, obs, obsm)`` per shard.
    """
    pieces = [(i, int(lo), int(hi)) for i, (lo, hi) in enumerate(bucket_bounds) if hi > lo]
    positions = np.empty(sum(hi - lo for _, lo, hi in pieces), dtype=np.int64)
    row = 0
    for i, lo, hi in pieces:
        _read_into(_scratch_file(scratch, i, "dest"), positions[row : row + hi - lo], lo)
        row += hi - lo
    order = np.argsort(positions - base)  # file order -> output order

    mats = {
        key: _gather(scratch, m, is_sparse, n_cols, pieces)[order]
        for m, (key, is_sparse, n_cols) in enumerate(specs)
    }
    for j, (filename, obs, obsm) in enumerate(payloads):
        rows = slice(j * shard_size, j * shard_size + len(obs))
        X = mats["X"][rows] if "X" in mats else None
        layers = {k[len("layers/") :]: v[rows] for k, v in mats.items() if k != "X"}
        _write_h5ad_shard(X, layers, obs, var, obsm, uns, filename, compression)
    return len(payloads)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def shuffled_shards(
    input_file: str,
    data: LazyData,
    out_names: list[str],
    shard_size: int,
    seed: int | None,
    compression: str | None = None,
    n_jobs: int | None = None,
    memory_limit: int | float | str | None = None,
    tmpdir: str | None = None,
    *,
    block_rows: int | None = None,
    shards_per_bucket: int | None = None,
) -> None:
    """
    Write ``out_names[j]`` as shard ``j`` of a random permutation of the cells of *data*.

    See the module docstring for the algorithm.  ``n_jobs=None`` picks one worker per CPU (fewer
    for small inputs); ``memory_limit=None`` uses half of the available RAM.  The memory limit
    is a sizing target for worker block sizes and counts, not a hard cap.  Scratch space of
    about the size of the uncompressed matrices is needed in *tmpdir* (default: the system
    temp directory).  ``block_rows`` and ``shards_per_bucket`` are test overrides.
    """
    n_obs = data.n_obs
    mats = data.matrices()
    keys = list(mats)
    specs, spill_bytes = [], 0.0
    for key, mat in mats.items():
        is_sparse, n_cols, row_bytes = _describe(key, mat)
        specs.append((key, is_sparse, n_cols))
        spill_bytes += row_bytes * n_obs
    row_bytes = spill_bytes / max(n_obs, 1)

    n_jobs, limit = _resolve_resources(n_jobs, memory_limit, spill_bytes)
    plan = _plan(n_obs, shard_size, row_bytes, n_jobs, limit, block_rows, shards_per_bucket)

    bucket_rows = plan.shards_per_bucket * shard_size
    n_buckets = math.ceil(n_obs / bucket_rows)
    n_shards = math.ceil(n_obs / shard_size)
    blocks = [
        (i, start, min(start + plan.block_rows, n_obs))
        for i, start in enumerate(range(0, n_obs, plan.block_rows))
    ]
    logger.info(
        "Two-pass shuffle: %d cells, %d shards, %d buckets, %d scatter blocks, %d worker(s), "
        "~%.1f GB scratch.",
        n_obs,
        n_shards,
        n_buckets,
        len(blocks),
        plan.n_jobs,
        spill_bytes / 1024**3,
    )

    # perm[p] is the input row that lands at output position p; dest is its inverse.
    perm = np.random.default_rng(seed).permutation(n_obs)
    dest = np.empty(n_obs, dtype=np.int64)
    dest[perm] = np.arange(n_obs)

    if tmpdir is not None:
        os.makedirs(tmpdir, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix="annslicer_", dir=tmpdir)
    try:
        free = shutil.disk_usage(scratch).free
        if free < spill_bytes * 1.05:
            raise OSError(
                f"Not enough scratch space in {os.path.dirname(scratch)}: need about "
                f"{spill_bytes / 1024**3:.1f} GB, {free / 1024**3:.1f} GB free. "
                f"Point --tmpdir / tmpdir at a larger local disk."
            )
        np.save(os.path.join(scratch, "dest.npy"), dest)
        del dest

        logger.info("Pass 1/2: scattering input rows into buckets...")
        bounds = np.stack(
            run_parallel(
                (
                    delayed(_scatter_block)(
                        input_file, keys, scratch, i, start, stop, bucket_rows, n_buckets
                    )
                    for i, start, stop in blocks
                ),
                plan.n_jobs,
                _Progress("Pass 1/2", len(blocks), "blocks"),
            )
        )

        def bucket_tasks():
            for b in range(n_buckets):
                payloads = []
                first = b * plan.shards_per_bucket
                for j in range(first, min(first + plan.shards_per_bucket, n_shards)):
                    cells = perm[j * shard_size : (j + 1) * shard_size]
                    obsm = {k: _take_rows(v, cells) for k, v in data.obsm.items()}
                    payloads.append((out_names[j], data.obs.iloc[cells], obsm))
                yield delayed(_write_bucket)(
                    scratch,
                    specs,
                    bounds[:, b : b + 2],
                    b * bucket_rows,
                    payloads,
                    shard_size,
                    data.var,
                    data.uns,
                    compression,
                )

        logger.info("Pass 2/2: writing %d shards...", n_shards)
        run_parallel(
            bucket_tasks(),
            plan.n_jobs,
            _Progress("Pass 2/2", n_shards, "shards"),
            weight=lambda shards_written: shards_written,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        if plan.n_jobs > 1:
            _release_workers()
