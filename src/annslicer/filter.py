"""
Core logic for annslicer: out-of-core filtering of .h5ad / .zarr files.

Produces a single output file containing only the cells for which a boolean
obs column (or auxiliary CSV column) evaluates to True.
"""

from __future__ import annotations

import argparse
import logging

import anndata as ad
import numpy as np
import pandas as pd

from annslicer._common import _merge_csv_into_obs, _write_shard_from_indices
from annslicer.slice import _open_zarr_backed

logger = logging.getLogger(__name__)


def filter_h5ad(
    input_file: str,
    output_file: str,
    obs_column: str,
    csv_file: str | None = None,
    join_column: str | None = None,
    compression: str | None = None,
) -> None:
    """
    Filter a large .h5ad or .zarr file to a single output file containing only
    the cells for which ``obs_column`` is truthy.

    The filter column is interpreted leniently:

    - ``bool`` dtype — used directly.
    - Numeric dtype — ``True`` wherever the value is non-zero.
    - Object / string dtype — ``"true"`` / ``"1"`` map to ``True``;
      ``"false"`` / ``"0"`` map to ``False``.  Any other value raises a
      ``ValueError`` listing the unrecognised entries.

    Parameters
    ----------
    input_file:
        Path to the source .h5ad or .zarr file.
    output_file:
        Destination path for the filtered .h5ad file.
    obs_column:
        Column in ``adata.obs`` (or in the auxiliary CSV) whose values
        determine which cells to keep.
    csv_file:
        Optional path to a CSV file with extra per-cell metadata.  Merged
        into ``adata.obs`` before filtering.  Columns from the CSV that are
        not already categorical are automatically coerced to
        ``pd.CategoricalDtype`` (useful when the CSV column is later used as
        an obs_column for :func:`shard_by_obs_column`).
    join_column:
        Column in the CSV to use as the cell-barcode join key.  Defaults to
        the CSV's first column.
    compression:
        HDF5 compression filter for the output file (e.g. ``"gzip"``).
    """
    if input_file.endswith(".zarr"):
        logger.info("Opening zarr store %s in backed mode via sparse_dataset...", input_file)
        adata = _open_zarr_backed(input_file)
    else:
        logger.info("Opening %s in backed mode...", input_file)
        adata = ad.read_h5ad(input_file, backed="r")

    try:
        _filter_store(adata, output_file, obs_column, csv_file, join_column, compression)
    finally:
        if hasattr(adata, "file") and adata.file.is_open:
            adata.file.close()


def _filter_store(
    adata: ad.AnnData,
    output_file: str,
    obs_column: str,
    csv_file: str | None,
    join_column: str | None,
    compression: str | None,
) -> None:
    """Core logic for :func:`filter_h5ad` operating on an open AnnData."""
    # --- Merge auxiliary CSV into obs if provided ---
    if csv_file is not None:
        adata.obs = _merge_csv_into_obs(adata.obs, csv_file, obs_column, join_column)

    col = adata.obs[obs_column]

    # --- Lenient boolean coercion ---
    if col.dtype == bool or pd.api.types.is_bool_dtype(col):
        bool_col = col
    elif pd.api.types.is_numeric_dtype(col):
        bool_col = col != 0
    else:
        # Object / string column: map known truthy/falsy strings.
        mapped = col.str.lower().map({"true": True, "false": False, "1": True, "0": False})
        bad = col[mapped.isna()]
        if len(bad) > 0:
            bad_values = sorted(bad.unique().tolist())[:10]
            raise ValueError(
                f"Column {obs_column!r} contains values that cannot be interpreted as boolean: "
                f"{bad_values}. Expected 'true', 'false', '1', or '0' (case-insensitive)."
            )
        bool_col = mapped.astype(bool)

    keep_idx = np.where(bool_col)[0]
    cells_in = adata.n_obs
    cells_out = len(keep_idx)
    logger.info(
        "Filtering %s: %d cells in → %d cells out → %s",
        "input",
        cells_in,
        cells_out,
        output_file,
    )
    _write_shard_from_indices(adata, keep_idx, output_file, compression)
    logger.info("Filter complete.")


def register_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the ``filter`` subcommand on an existing subparsers action."""
    p = subparsers.add_parser(
        "filter",
        help="Filter a large .h5ad or .zarr file to a single output file.",
        description=(
            "Out-of-core cell filtering: keep only cells for which a boolean "
            "obs column (or auxiliary CSV column) is True.  The filter column "
            "is interpreted leniently (bool, 0/1 int, or 'True'/'False' strings)."
        ),
    )
    p.add_argument("input_file", help="Path to the input .h5ad or .zarr file.")
    p.add_argument("output_file", help="Path for the filtered output .h5ad file.")
    p.add_argument(
        "--obs-column",
        required=True,
        metavar="COLUMN",
        help="Column in obs (or auxiliary CSV) with boolean keep/discard values.",
    )
    p.add_argument(
        "--csv-file",
        default=None,
        metavar="PATH",
        help=(
            "Path to an auxiliary CSV file with extra per-cell metadata. "
            "Merged into obs before filtering (columns coerced to categorical)."
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
        "--compression",
        default=None,
        metavar="FILTER",
        help=(
            "HDF5 compression filter for the output file "
            '(e.g. "gzip", "lzf"). Default: no compression.'
        ),
    )
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    """Dispatch function called by the CLI after argument parsing."""
    filter_h5ad(
        args.input_file,
        args.output_file,
        args.obs_column,
        csv_file=args.csv_file,
        join_column=args.join_column,
        compression=args.compression,
    )
