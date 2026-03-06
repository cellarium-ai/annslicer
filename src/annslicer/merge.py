"""
Core logic for annslicer: out-of-core merging of sharded .h5ad / .zarr files.
"""

from __future__ import annotations

import argparse
import logging

import anndata as ad
import numpy as np
import pandas as pd
from anndata.experimental import read_elem

from annslicer._store import _is_sparse_group, _require_zarr, open_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main merge function
# ---------------------------------------------------------------------------


def merge_out_of_core(input_files: list[str], output_file: str) -> None:
    """
    Merge multiple sharded .h5ad or .zarr files into a single large file
    using minimal RAM.

    Uses low-level HDF5/Zarr I/O to stream matrix data directly between files
    without loading full matrices into memory.

    Parameters
    ----------
    input_files:
        Ordered list of shard paths to merge (.h5ad or .zarr).
    output_file:
        Destination path for the merged file (.h5ad or .zarr).
        The format is inferred from the file extension.
    """
    is_zarr = output_file.endswith(".zarr")
    format_name = "Zarr" if is_zarr else "H5AD"

    # Validate zarr availability early — before doing any work.
    if is_zarr:
        _require_zarr()

    # -----------------------------------------------------------------------
    # Phase 1: gather metadata (obs, var, uns) with minimal memory
    # -----------------------------------------------------------------------
    logger.info("Phase 1: Building the metadata skeleton (%s format)...", format_name)

    store_first = open_store(input_files[0], "r")
    try:
        var: pd.DataFrame = read_elem(store_first["var"])
        uns: dict = read_elem(store_first["uns"]) if "uns" in store_first else {}
        num_genes: int = var.shape[0]
    finally:
        if hasattr(store_first, "close"):
            store_first.close()

    obs_list: list[pd.DataFrame] = []
    total_cells: int = 0

    for f in input_files:
        store = open_store(f, "r")
        try:
            obs: pd.DataFrame = read_elem(store["obs"])
            obs_list.append(obs)
            total_cells += obs.shape[0]
        finally:
            if hasattr(store, "close"):
                store.close()

    merged_obs = pd.concat(obs_list, axis=0)

    skeleton = ad.AnnData(obs=merged_obs, var=var, uns=uns)
    if is_zarr:
        skeleton.write_zarr(output_file)
    else:
        skeleton.write_h5ad(output_file)

    logger.info("Skeleton saved to %s. Total cells: %d", output_file, total_cells)

    # -----------------------------------------------------------------------
    # Phase 2: scan shards to calculate total non-zero counts per matrix
    # -----------------------------------------------------------------------
    logger.info("Phase 2: Scanning shards to calculate matrix sizes...")

    matrix_nnz: dict[str, int] = {}
    layer_nnz: dict[str, int] = {}
    obsm_shapes: dict[str, int] = {}

    for f in input_files:
        store = open_store(f, "r")
        try:
            if "X" in store:
                if _is_sparse_group(store, "X"):
                    matrix_nnz["X"] = matrix_nnz.get("X", 0) + store["X"]["data"].shape[0]
                else:
                    x_ds = store["X"]
                    matrix_nnz["X"] = matrix_nnz.get("X", 0) + int(
                        x_ds.shape[0] * x_ds.shape[1]  # type: ignore[index]
                    )

            if "layers" in store:
                for layer in store["layers"].keys():
                    lpath = f"layers/{layer}"
                    if _is_sparse_group(store, lpath):
                        layer_nnz[layer] = layer_nnz.get(layer, 0) + store[lpath]["data"].shape[0]

            if "obsm" in store:
                for key in store["obsm"].keys():
                    if key not in obsm_shapes:
                        obsm_shapes[key] = store["obsm"][key].shape[1]
        finally:
            if hasattr(store, "close"):
                store.close()

    # -----------------------------------------------------------------------
    # Phase 3: pre-allocate arrays and stream data shard-by-shard
    # -----------------------------------------------------------------------
    logger.info("Phase 3: Pre-allocating and streaming data to disk...")

    out_store = open_store(output_file, "a" if is_zarr else "r+")

    def allocate_sparse(group_name: str, total_nnz: int) -> None:
        """Pre-allocate a CSR sparse group in the output store."""
        # Remove any placeholder the AnnData skeleton may have written.
        if group_name in out_store:
            del out_store[group_name]

        grp = out_store.require_group(group_name)
        grp.attrs["encoding-type"] = "csr_matrix"
        grp.attrs["encoding-version"] = "0.1.0"
        grp.attrs["shape"] = [int(total_cells), int(num_genes)]

        chunk_size = (min(total_nnz, 1_000_000),)
        grp.create_dataset("data", shape=(total_nnz,), chunks=chunk_size, dtype=np.float32)
        grp.create_dataset("indices", shape=(total_nnz,), chunks=chunk_size, dtype=np.int32)
        indptr_chunk = (min(total_cells + 1, 100_000),)
        grp.create_dataset("indptr", shape=(total_cells + 1,), chunks=indptr_chunk, dtype=np.int64)

    if "X" in matrix_nnz:
        allocate_sparse("X", matrix_nnz["X"])

    if layer_nnz:
        layers_grp = out_store.require_group("layers")
        if is_zarr:
            layers_grp.attrs["encoding-type"] = "dict"
            layers_grp.attrs["encoding-version"] = "0.1.0"
        for layer, nnz in layer_nnz.items():
            allocate_sparse(f"layers/{layer}", nnz)

    if obsm_shapes:
        grp_obsm = out_store.require_group("obsm")
        if is_zarr:
            grp_obsm.attrs["encoding-type"] = "dict"
            grp_obsm.attrs["encoding-version"] = "0.1.0"
        for key, dim in obsm_shapes.items():
            if key in grp_obsm:
                del grp_obsm[key]
            obsm_chunk_rows = min(total_cells, 10_000)
            grp_obsm.create_dataset(
                key, shape=(total_cells, dim), chunks=(obsm_chunk_rows, dim), dtype=np.float32
            )

    # Streaming loop
    current_cell: int = 0
    current_nnz_X: int = 0
    current_nnz_layers: dict[str, int] = {k: 0 for k in layer_nnz}

    for i, (f, obs_df) in enumerate(zip(input_files, obs_list, strict=True)):
        num_cells_in_shard = obs_df.shape[0]
        logger.info(
            "Merging shard %d/%d: %s (%d cells)",
            i + 1,
            len(input_files),
            f,
            num_cells_in_shard,
        )

        in_store = open_store(f, "r")
        try:
            # Stream X
            if "X" in matrix_nnz and "X" in in_store and _is_sparse_group(in_store, "X"):
                shard_nnz = in_store["X"]["data"].shape[0]
                out_store["X"]["data"][current_nnz_X : current_nnz_X + shard_nnz] = in_store["X"][
                    "data"
                ][:]
                out_store["X"]["indices"][current_nnz_X : current_nnz_X + shard_nnz] = in_store[
                    "X"
                ]["indices"][:]
                shifted_indptr = in_store["X"]["indptr"][:] + current_nnz_X
                out_store["X"]["indptr"][current_cell : current_cell + num_cells_in_shard] = (
                    shifted_indptr[:-1]
                )
                current_nnz_X += shard_nnz

            # Stream layers
            if "layers" in in_store:
                for layer in layer_nnz:
                    lpath = f"layers/{layer}"
                    if layer in in_store["layers"] and _is_sparse_group(in_store, lpath):
                        shard_nnz = in_store[lpath]["data"].shape[0]
                        start_nnz = current_nnz_layers[layer]
                        out_store[lpath]["data"][start_nnz : start_nnz + shard_nnz] = in_store[
                            lpath
                        ]["data"][:]
                        out_store[lpath]["indices"][start_nnz : start_nnz + shard_nnz] = in_store[
                            lpath
                        ]["indices"][:]
                        shifted_indptr = in_store[lpath]["indptr"][:] + start_nnz
                        out_store[lpath]["indptr"][
                            current_cell : current_cell + num_cells_in_shard
                        ] = shifted_indptr[:-1]
                        current_nnz_layers[layer] += shard_nnz

            # Stream obsm
            if "obsm" in in_store:
                for key in obsm_shapes:
                    if key in in_store["obsm"]:
                        out_store["obsm"][key][
                            current_cell : current_cell + num_cells_in_shard, :
                        ] = in_store["obsm"][key][:]

        finally:
            if hasattr(in_store, "close"):
                in_store.close()

        current_cell += num_cells_in_shard

    # Cap off indptr arrays at total_cells position
    if "X" in matrix_nnz:
        out_store["X"]["indptr"][total_cells] = current_nnz_X
    for layer, final_nnz in current_nnz_layers.items():
        out_store[f"layers/{layer}"]["indptr"][total_cells] = final_nnz

    if hasattr(out_store, "close"):
        out_store.close()

    # For zarr output, update consolidated metadata (.zmetadata) so that
    # all newly-created groups and arrays are visible to zarr v2 readers
    # (including anndata's read_zarr).
    if is_zarr:
        zarr_mod = _require_zarr()
        zarr_mod.consolidate_metadata(output_file)

    logger.info("Done! All files merged successfully into %s", output_file)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def register_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the ``merge`` subcommand on an existing subparsers action."""
    p = subparsers.add_parser(
        "merge",
        help="Merge multiple shards into one large .h5ad or .zarr file.",
        description=(
            "Merge sharded .h5ad or .zarr files into a single large file "
            "using minimal RAM. Output format is inferred from the file extension."
        ),
    )
    p.add_argument(
        "output_file",
        help="Path for the merged output file (.h5ad or .zarr).",
    )
    p.add_argument(
        "input_files",
        nargs="+",
        help="Input shard files to merge, in order.",
    )
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    """Dispatch function called by the CLI after argument parsing."""
    merge_out_of_core(args.input_files, args.output_file)
