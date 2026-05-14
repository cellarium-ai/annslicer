"""
Core logic for annslicer: out-of-core sharding of .h5ad / .zarr files.
"""

from __future__ import annotations

import argparse
import logging
import re
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from annslicer._common import (
    _ensure_parent_dir,
    _merge_csv_into_obs,
    _unwrap,
    _write_shard_from_indices,
)
from annslicer._store import _require_zarr

logger = logging.getLogger(__name__)


def _open_zarr_backed(input_file: str) -> ad.AnnData:
    """
    Open a zarr store in a backed-like mode without loading matrix data into RAM.

    X and sparse layers are wrapped as ``CSRDataset`` objects that support
    out-of-core slice and fancy indexing.  Small metadata (obs, var, obsm,
    uns, etc.) is loaded eagerly since it must fit in memory anyway.
    """
    from anndata.io import read_elem, sparse_dataset

    zarr_mod = _require_zarr()
    group = zarr_mod.open(input_file, mode="r")

    def _lazy_or_dense(grp, key: str) -> Any:
        """Return a CSRDataset if the key holds a sparse group, else read_elem."""
        try:
            return sparse_dataset(grp[key])
        except Exception:
            return read_elem(grp[key])

    layers: dict[str, Any] = (
        {k: _lazy_or_dense(group["layers"], k) for k in group["layers"]}
        if "layers" in group
        else {}
    )

    return ad.AnnData(
        X=_lazy_or_dense(group, "X"),
        **{
            k: read_elem(group[k]) if k in group else {}
            for k in ["obs", "var", "obsm", "varm", "uns", "obsp", "varp"]
        },
        layers=layers,
    )


def shard_h5ad(
    input_file: str,
    output_prefix: str,
    output_filenames: list[str] | None = None,
    shard_size: int = 10000,
    shuffle: bool = False,
    seed: int | None = None,
    compression: str | None = None,
) -> None:
    """
    Shard a large .h5ad or .zarr file into smaller files using minimal RAM.

    For .h5ad inputs, uses AnnData backed-mode reading so h5py streams each
    shard's rows without loading the full matrix into memory.

    For .zarr inputs, uses :func:`_open_zarr_backed` which wraps X and sparse
    layers as ``CSRDataset`` objects (``anndata.io.sparse_dataset``), giving
    the same out-of-core behaviour without requiring backed-mode support in
    AnnData's zarr reader.

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
    """
    _ensure_parent_dir(output_prefix)

    if input_file.endswith(".zarr"):
        logger.info("Opening zarr store %s in backed mode via sparse_dataset...", input_file)
        adata = _open_zarr_backed(input_file)
    else:
        logger.info("Opening %s in backed mode...", input_file)
        adata = ad.read_h5ad(input_file, backed="r")

    try:
        _shard_store(
            adata, output_prefix, output_filenames, shard_size, shuffle, seed, compression
        )
    finally:
        if hasattr(adata, "file") and adata.file.is_open:
            adata.file.close()


def _shard_store(
    adata: ad.AnnData,
    output_prefix: str,
    output_filenames: list[str] | None,
    shard_size: int,
    shuffle: bool,
    seed: int | None,
    compression: str | None = None,
) -> None:
    """
    Core sharding loop operating on an already-opened AnnData object.

    Reads each shard directly via h5py slice/fancy indexing and constructs an
    in-memory AnnData from the pieces before writing.  For shuffled output,
    indices are sorted prior to reading (sequential I/O), then reordered in
    memory into the target permutation order, avoiding random disk seeks.
    """
    if (
        output_filenames is not None
        and len(output_filenames) < (adata.n_obs + shard_size - 1) // shard_size
    ):
        raise ValueError(
            f"Not enough output filenames provided: expected at least "
            f"{(adata.n_obs + shard_size - 1) // shard_size}, got {len(output_filenames)}"
        )

    total_cells = adata.n_obs

    perm: np.ndarray | None = None
    if shuffle:
        perm = np.random.default_rng(seed).permutation(total_cells)
        logger.info("Shuffle enabled (seed=%s). Permutation generated.", seed)

    logger.info("Total cells: %d. Generating shards of %d...", total_cells, shard_size)

    for start_idx in range(0, total_cells, shard_size):
        end_idx = min(start_idx + shard_size, total_cells)
        shard_num = start_idx // shard_size
        out_filename = (
            output_filenames[shard_num]
            if output_filenames is not None
            else f"{output_prefix}_shard_{shard_num}.h5ad"
        )
        logger.info("  Writing %s (cells %d–%d)...", out_filename, start_idx, end_idx)

        if perm is not None:
            orig_idx = perm[start_idx:end_idx]
            sorted_idx = np.sort(orig_idx)
            restore = np.argsort(np.argsort(orig_idx))
            X = _unwrap(adata.X[sorted_idx, :])[restore]
            layers = {k: _unwrap(adata.layers[k][sorted_idx, :])[restore] for k in adata.layers}
            obsm = {k: np.asarray(adata.obsm[k][sorted_idx])[restore] for k in adata.obsm}
            obs = adata.obs.iloc[orig_idx]
            ad.AnnData(
                X=X,
                obs=obs.copy(),
                var=adata.var.copy(),
                obsm=obsm,
                layers=layers,
                uns=adata.uns.copy(),
            ).write_h5ad(out_filename, compression=compression)
        else:
            _write_shard_from_indices(
                adata,
                np.arange(start_idx, end_idx, dtype=np.intp),
                out_filename,
                compression,
            )

    logger.info("All shards successfully created.")


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

    if input_file.endswith(".zarr"):
        logger.info("Opening zarr store %s in backed mode via sparse_dataset...", input_file)
        adata = _open_zarr_backed(input_file)
    else:
        logger.info("Opening %s in backed mode...", input_file)
        adata = ad.read_h5ad(input_file, backed="r")

    try:
        _shard_by_obs_column_store(
            adata,
            output_prefix,
            obs_column,
            csv_file,
            join_column,
            always_include,
            compression,
        )
    finally:
        if hasattr(adata, "file") and adata.file.is_open:
            adata.file.close()


def _shard_by_obs_column_store(
    adata: ad.AnnData,
    output_prefix: str,
    obs_column: str,
    csv_file: str | None,
    join_column: str | None,
    always_include: list[str] | None,
    compression: str | None,
) -> None:
    """Core logic for :func:`shard_by_obs_column` operating on an open AnnData."""
    # --- Merge auxiliary CSV into obs if provided ---
    if csv_file is not None:
        adata.obs = _merge_csv_into_obs(adata.obs, csv_file, obs_column, join_column)

    # --- Validate obs_column is categorical ---
    if obs_column not in adata.obs.columns:
        raise KeyError(f"obs_column {obs_column!r} not found in adata.obs.")
    obs_col = adata.obs[obs_column]
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
        _write_shard_from_indices(adata, indices, out_filename, compression)
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
        )
