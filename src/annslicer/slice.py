"""
Core logic for annslicer: out-of-core sharding of .h5ad / .zarr files.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import anndata as ad
import numpy as np

from annslicer._store import _require_zarr

logger = logging.getLogger(__name__)


def shard_h5ad(
    input_file: str,
    output_prefix: str,
    shard_size: int = 10000,
    shuffle: bool = False,
    seed: int | None = None,
) -> None:
    """
    Shard a large .h5ad or .zarr file into smaller files using minimal RAM.

    Uses AnnData backed-mode reading for .h5ad inputs, delegating all row
    access to h5py's native indexing (which handles both dense and sparse CSR
    matrices efficiently for sequential and shuffled access).

    For .zarr inputs the store is read fully into memory before sharding,
    since AnnData does not support backed-mode for zarr.

    Parameters
    ----------
    input_file:
        Path to the source .h5ad or .zarr file.
    output_prefix:
        Prefix for output shard filenames, e.g. ``"dataset"`` produces
        ``dataset_shard001.h5ad``, ``dataset_shard002.h5ad``, etc.
    shard_size:
        Number of cells (rows) per shard. Defaults to 10 000.
    shuffle:
        When ``True``, cells are assigned to shards in a random order so
        that each shard contains a representative draw from the full dataset
        rather than a contiguous block of cells.
    seed:
        Random seed passed to :class:`numpy.random.Generator` when
        ``shuffle=True``.  Ignored when ``shuffle=False``.
    """
    if input_file.endswith(".zarr"):
        _require_zarr()
        logger.info("Opening zarr store %s (reads fully into memory)...", input_file)
        adata = ad.read_zarr(input_file)
    else:
        logger.info("Opening %s in backed mode...", input_file)
        adata = ad.read_h5ad(input_file, backed="r")

    try:
        _shard_store(adata, output_prefix, shard_size, shuffle, seed)
    finally:
        if hasattr(adata, "file") and adata.file.is_open:
            adata.file.close()


def _unwrap(arr: np.ndarray) -> Any:
    """Unwrap the 0-d object array that h5py sometimes returns for backed sparse layers."""
    return arr.item() if isinstance(arr, np.ndarray) and arr.ndim == 0 else arr


def _shard_store(
    adata: ad.AnnData,
    output_prefix: str,
    shard_size: int,
    shuffle: bool,
    seed: int | None,
) -> None:
    """
    Core sharding loop operating on an already-opened AnnData object.

    Reads each shard directly via h5py slice/fancy indexing and constructs an
    in-memory AnnData from the pieces before writing.  For shuffled output,
    indices are sorted prior to reading (sequential I/O), then reordered in
    memory into the target permutation order, avoiding random disk seeks.
    """
    total_cells = adata.n_obs

    perm: np.ndarray | None = None
    if shuffle:
        perm = np.random.default_rng(seed).permutation(total_cells)
        logger.info("Shuffle enabled (seed=%s). Permutation generated.", seed)

    logger.info("Total cells: %d. Generating shards of %d...", total_cells, shard_size)

    for start_idx in range(0, total_cells, shard_size):
        end_idx = min(start_idx + shard_size, total_cells)
        shard_num = (start_idx // shard_size) + 1
        out_filename = f"{output_prefix}_shard_{shard_num}.h5ad"
        logger.info("  Writing %s (cells %d–%d)...", out_filename, start_idx, end_idx)

        if perm is not None:
            orig_idx = perm[start_idx:end_idx]
            sorted_idx = np.sort(orig_idx)
            restore = np.argsort(np.argsort(orig_idx))
            X = _unwrap(adata.X[sorted_idx, :])[restore]
            layers = {k: _unwrap(adata.layers[k][sorted_idx, :])[restore] for k in adata.layers}
            obsm = {k: np.asarray(adata.obsm[k][sorted_idx])[restore] for k in adata.obsm}
            obs = adata.obs.iloc[orig_idx]
        else:
            s = slice(start_idx, end_idx)
            X = _unwrap(adata.X[s, :])
            layers = {k: _unwrap(adata.layers[k][s, :]) for k in adata.layers}
            obsm = {k: np.asarray(adata.obsm[k][s]) for k in adata.obsm}
            obs = adata.obs.iloc[start_idx:end_idx]

        ad.AnnData(
            X=X,
            obs=obs.copy(),
            var=adata.var.copy(),
            obsm=obsm,
            layers=layers,
            uns=adata.uns.copy(),
        ).write_h5ad(out_filename)

    logger.info("All shards successfully created.")


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
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    """Dispatch function called by the CLI after argument parsing."""
    shard_h5ad(
        args.input_file,
        args.output_prefix,
        args.size,
        shuffle=args.shuffle,
        seed=args.seed,
    )
