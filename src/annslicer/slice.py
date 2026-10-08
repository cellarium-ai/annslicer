"""
Core logic for annslicer: out-of-core sharding of .h5ad / .zarr files.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time

import numpy as np
import pandas as pd

from annslicer._common import (
    LazyData,
    _ensure_parent_dir,
    _merge_csv_into_obs,
    _open_lazy,
    _write_shard_from_indices,
)
from annslicer._contiguous import contiguous_shards
from annslicer._parallel import _fmt_duration
from annslicer._shuffle import shuffled_shards

logger = logging.getLogger(__name__)


def shard_h5ad(
    input_file: str,
    output_prefix: str,
    output_filenames: list[str] | None = None,
    shard_size: int = 10000,
    shuffle: bool = False,
    seed: int | None = None,
    compression: str | None = None,
    n_jobs: int | None = None,
    memory_limit: int | str | None = None,
    tmpdir: str | None = None,
) -> None:
    """
    Shard a large .h5ad or .zarr file into smaller files using minimal RAM.

    X and every layer stay on disk: the input is opened lazily (see
    :func:`annslicer._common._open_lazy`) and only the rows of the shard being written are
    ever read.  Without shuffling, each shard is one contiguous read and shards are written
    in parallel (see :mod:`annslicer._contiguous`).  With shuffling, a two-pass scatter /
    gather (see :mod:`annslicer._shuffle`) keeps all reads sequential at the cost of
    temporary scratch space.

    Parameters
    ----------
    input_file:
        Path to the source .h5ad or .zarr file.
    output_prefix:
        Prefix for output shard filenames, e.g. ``"dataset"`` produces
        ``dataset_shard_0.h5ad``, ``dataset_shard_1.h5ad``, etc.
    output_filenames:
        Optional list of output shard filenames to write, e.g.
        ``["shard_a.h5ad", "shard_b.h5ad", ...]``.  If provided, overrides the default naming scheme based on
        ``output_prefix`` and ``shard_size``.  Must be the same length as the number of shards needed to
        cover all cells in the input file.
    shard_size:
        Number of cells (rows) per shard. Defaults to 10 000.
    shuffle:
        When ``True``, cells are assigned to shards in a random order so
        that each shard contains a representative draw from the full dataset
        rather than a contiguous block of cells.
    seed:
        Random seed passed to :class:`numpy.random.Generator` when
        ``shuffle=True``.  Ignored when ``shuffle=False``.
    compression:
        HDF5 compression filter to use when writing shard ``.h5ad`` files,
        e.g. ``"gzip"`` or ``"lzf"``.  ``None`` (default) writes
        uncompressed files, which is fastest for downstream streaming reads.
    n_jobs:
        Number of worker processes.  ``None`` (default) uses one per available CPU, fewer
        for small inputs, and is further limited by ``memory_limit``.  When calling from a
        script with ``n_jobs > 1``, no ``if __name__ == "__main__":`` guard is needed.
    memory_limit:
        Memory budget used to limit how many workers run at once (and, when shuffling, to size
        their blocks), as bytes or a string such as ``"16GB"``.  Defaults to half of the
        available RAM.  It is a sizing target, not a hard cap.
    tmpdir:
        Directory for the temporary scratch files of shuffled sharding (about the size of the
        uncompressed matrices; unused without ``shuffle``).  Defaults to the system temp
        directory.
    """
    _ensure_parent_dir(output_prefix)

    logger.info("Opening %s lazily...", input_file)
    start = time.monotonic()
    data = _open_lazy(input_file)
    logger.info(
        "Read metadata for %d cells in %s.", data.n_obs, _fmt_duration(time.monotonic() - start)
    )
    try:
        _shard_store(
            input_file,
            data,
            output_prefix,
            output_filenames,
            shard_size,
            shuffle,
            seed,
            compression,
            n_jobs,
            memory_limit,
            tmpdir,
        )
    finally:
        data.close()


def _shard_store(
    input_file: str,
    data: LazyData,
    output_prefix: str,
    output_filenames: list[str] | None,
    shard_size: int,
    shuffle: bool,
    seed: int | None,
    compression: str | None = None,
    n_jobs: int | None = None,
    memory_limit: int | str | None = None,
    tmpdir: str | None = None,
) -> None:
    """
    Core sharding loop operating on an already-opened :class:`LazyData`.

    Unshuffled shards come from :func:`annslicer._contiguous.contiguous_shards`, shuffled
    shards from :func:`annslicer._shuffle.shuffled_shards`.
    """
    total_cells = data.n_obs
    n_shards = (total_cells + shard_size - 1) // shard_size
    if output_filenames is not None and len(output_filenames) < n_shards:
        raise ValueError(
            f"Not enough output filenames provided: expected at least "
            f"{n_shards}, got {len(output_filenames)}"
        )
    out_names = (
        output_filenames[:n_shards]
        if output_filenames is not None
        else [f"{output_prefix}_shard_{i}.h5ad" for i in range(n_shards)]
    )

    logger.info("Total cells: %d. Generating shards of %d...", total_cells, shard_size)
    start = time.monotonic()

    if shuffle:
        logger.info("Shuffle enabled (seed=%s).", seed)
        shuffled_shards(
            input_file,
            data,
            out_names,
            shard_size,
            seed,
            compression,
            n_jobs,
            memory_limit,
            tmpdir,
        )
    else:
        contiguous_shards(
            input_file, data, out_names, shard_size, compression, n_jobs, memory_limit
        )

    written = sum(os.path.getsize(name) for name in out_names)
    logger.info(
        "All %d shards successfully created in %s (%.1f GB written).",
        n_shards,
        _fmt_duration(time.monotonic() - start),
        written / 1024**3,
    )


def shard_by_obs_column(
    input_file: str,
    output_prefix: str,
    obs_column: str,
    csv_file: str | None = None,
    join_column: str | None = None,
    always_include: list[str] | None = None,
    compression: str | None = None,
) -> None:
    """
    Shard a large .h5ad or .zarr file by grouping cells according to a
    categorical ``adata.obs`` column.

    One output .h5ad file is produced per category (excluding any categories
    listed in ``always_include``).  Output filenames are derived from the
    category names rather than shard numbers:
    ``{output_prefix}_{safe_category_name}.h5ad``.

    Parameters
    ----------
    input_file:
        Path to the source .h5ad or .zarr file.
    output_prefix:
        Prefix for output shard filenames.
    obs_column:
        Name of the column in ``adata.obs`` (or in the auxiliary CSV) to
        partition on.  Must be (or be coercible to) a categorical dtype.
    csv_file:
        Optional path to a CSV file containing extra per-cell metadata.  The
        CSV is merged into ``adata.obs`` before partitioning.  Columns from
        the CSV that are not already categorical are automatically coerced to
        ``pd.CategoricalDtype``.
    join_column:
        Column in the CSV to use as the join key (cell barcode).  Defaults to
        the CSV's first column.
    always_include:
        One or more category values to append to *every* output shard.  Cells
        belonging to these categories are copied into each shard but do not
        produce a dedicated output file of their own.
    compression:
        HDF5 compression filter for output files (e.g. ``"gzip"``).
    """
    _ensure_parent_dir(output_prefix)

    logger.info("Opening %s lazily...", input_file)
    data = _open_lazy(input_file)
    try:
        _shard_by_obs_column_store(
            data,
            output_prefix,
            obs_column,
            csv_file,
            join_column,
            always_include,
            compression,
        )
    finally:
        data.close()


def _shard_by_obs_column_store(
    data: LazyData,
    output_prefix: str,
    obs_column: str,
    csv_file: str | None,
    join_column: str | None,
    always_include: list[str] | None,
    compression: str | None,
) -> None:
    """Core logic for :func:`shard_by_obs_column` operating on an open :class:`LazyData`."""
    # --- Merge auxiliary CSV into obs if provided ---
    if csv_file is not None:
        data.obs = _merge_csv_into_obs(data.obs, csv_file, obs_column, join_column)

    # --- Validate obs_column is categorical ---
    if obs_column not in data.obs.columns:
        raise KeyError(f"obs_column {obs_column!r} not found in adata.obs.")
    obs_col = data.obs[obs_column]
    if not isinstance(obs_col.dtype, pd.CategoricalDtype):
        raise ValueError(
            f"obs_column {obs_column!r} has dtype {obs_col.dtype!r}, expected a categorical. "
            f"Cast the column to a categorical before calling shard_by_obs_column, "
            f"or provide it via --csv-file (CSV columns are coerced automatically)."
        )

    categories = list(obs_col.cat.categories)
    always_include_set: set[str] = set(always_include) if always_include else set()

    # --- Validate always_include values ---
    if always_include_set:
        unknown = always_include_set - set(categories)
        if unknown:
            raise ValueError(
                f"always_include contains value(s) not found in category list: "
                f"{sorted(unknown)}. Valid categories are: {categories}."
            )

    # --- Compute always-include indices ---
    if always_include_set:
        always_idx = np.where(obs_col.isin(always_include_set))[0]
    else:
        always_idx = np.array([], dtype=np.intp)

    # --- Sanitize names and check for collisions ---
    shard_categories = [c for c in categories if c not in always_include_set]
    safe_names: dict[str, str] = {}  # category -> safe filename fragment
    seen_safe: dict[str, str] = {}  # safe name -> original category (for collision detection)
    for cat in shard_categories:
        safe = re.sub(r"[^\w.-]", "_", str(cat))
        if safe in seen_safe:
            raise ValueError(
                f"Category names {cat!r} and {seen_safe[safe]!r} both sanitize to the "
                f"same filename fragment {safe!r}. Rename one of the categories so that "
                f"their alphanumeric representations are distinct."
            )
        seen_safe[safe] = cat
        safe_names[cat] = safe

    # --- Write one shard per category ---
    shards_written = 0
    for cat in shard_categories:
        cat_idx = np.where(obs_col == cat)[0]
        if len(cat_idx) == 0:
            logger.warning(
                "Category %r has no cells — skipping (no output file will be written).", cat
            )
            continue

        indices = np.sort(np.concatenate([cat_idx, always_idx]))
        out_filename = f"{output_prefix}_{safe_names[cat]}.h5ad"
        logger.info(
            "  Writing %s (%d cells + %d always-include)...",
            out_filename,
            len(cat_idx),
            len(always_idx),
        )
        _write_shard_from_indices(data, indices, out_filename, compression)
        shards_written += 1

    logger.info(
        "shard_by_obs_column complete: %d shards written for column %r.",
        shards_written,
        obs_column,
    )


def register_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the ``slice`` subcommand on an existing subparsers action."""
    p = subparsers.add_parser(
        "slice",
        help="Shard a large .h5ad or .zarr file into smaller shards.",
        description=(
            "Safely shard large .h5ad or .zarr files out-of-core "
            "(includes X, layers, and obsm). Supports optional random shuffling."
        ),
    )
    p.add_argument("input_file", help="Path to the input .h5ad or .zarr file.")
    p.add_argument(
        "output_prefix",
        help="Prefix for output shard files (e.g. 'my_dataset').",
    )
    p.add_argument(
        "--size",
        type=int,
        default=10000,
        metavar="N",
        help="Number of cells per shard (default: 10000).",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
        help=(
            "Randomly assign cells to shards so each shard is representative "
            "of the full dataset rather than a contiguous block."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Random seed for reproducible shuffling (requires --shuffle).",
    )
    p.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Worker processes (default: one per available CPU, fewer for small inputs; "
            "further limited by --memory-limit)."
        ),
    )
    p.add_argument(
        "--memory-limit",
        default=None,
        metavar="SIZE",
        help=(
            "Memory budget for the workers, e.g. 16GB (default: half of available RAM). "
            "A sizing target, not a hard cap."
        ),
    )
    p.add_argument(
        "--tmpdir",
        default=None,
        metavar="PATH",
        help=(
            "Directory for --shuffle scratch files, about the size of the uncompressed "
            "matrices (default: the system temp directory)."
        ),
    )
    p.add_argument(
        "--compression",
        default=None,
        metavar="FILTER",
        help=(
            "HDF5 compression filter for output shard files "
            '(e.g. "gzip", "lzf"). Default: no compression.'
        ),
    )
    p.add_argument(
        "--obs-column",
        default=None,
        metavar="COLUMN",
        help=(
            "Partition cells by this categorical obs column instead of fixed-size shards. "
            "Each category produces one output file named {output_prefix}_{category}.h5ad."
        ),
    )
    p.add_argument(
        "--csv-file",
        default=None,
        metavar="PATH",
        help=(
            "Path to an auxiliary CSV file with extra per-cell metadata. "
            "Merged into obs before partitioning (columns coerced to categorical)."
        ),
    )
    p.add_argument(
        "--join-column",
        default=None,
        metavar="COLUMN",
        help=(
            "Column in the CSV to use as the cell-barcode join key. "
            "Defaults to the CSV's first column."
        ),
    )
    p.add_argument(
        "--always-include",
        nargs="+",
        default=None,
        metavar="VALUE",
        help=(
            "One or more category values to append to every output shard "
            "(e.g. non-targeting control cells). Requires --obs-column."
        ),
    )
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    """Dispatch function called by the CLI after argument parsing."""
    if args.obs_column is not None:
        shard_by_obs_column(
            args.input_file,
            args.output_prefix,
            args.obs_column,
            csv_file=args.csv_file,
            join_column=args.join_column,
            always_include=args.always_include,
            compression=args.compression,
        )
    else:
        shard_h5ad(
            args.input_file,
            args.output_prefix,
            shard_size=args.size,
            shuffle=args.shuffle,
            seed=args.seed,
            compression=args.compression,
            n_jobs=args.jobs,
            memory_limit=args.memory_limit,
            tmpdir=args.tmpdir,
        )
