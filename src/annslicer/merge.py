"""
Core logic for annslicer: out-of-core merging of sharded .h5ad / .zarr files.
"""

from __future__ import annotations

import argparse
import logging

import anndata as ad
import numpy as np
import pandas as pd
from anndata.io import read_elem

from annslicer._store import _is_sparse_group, _require_zarr, open_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_merged_var(
    var_frames: list[pd.DataFrame], join: str
) -> tuple[pd.DataFrame, list[np.ndarray], bool]:
    """Compute the merged var DataFrame and per-shard column remaps.

    Parameters
    ----------
    var_frames:
        var DataFrames, one per shard, in shard order.
    join:
        ``"inner"`` or ``"outer"``.

    Returns
    -------
    merged_var : pd.DataFrame
        The merged var DataFrame (index is the merged gene set).
    col_remaps : list of np.ndarray
        ``col_remaps[i][j]`` is the column index in *merged_var* for gene *j*
        of shard *i*.  For an inner join this may be ``-1`` for genes outside
        the intersection; for an outer join every value is ``≥ 0``.
    is_identity : bool
        ``True`` when all shards share the identical var (fast path — no
        remapping required).
    """
    from functools import reduce

    indices = [df.index for df in var_frames]

    # Fast path: every shard has the same var in the same order.
    if all(idx.equals(indices[0]) for idx in indices[1:]):
        identity = np.arange(len(indices[0]), dtype=np.int32)
        return var_frames[0], [identity] * len(var_frames), True

    if join == "inner":
        merged_index = reduce(lambda a, b: a.intersection(b), indices)
    else:  # outer
        merged_index = reduce(lambda a, b: a.union(b), indices)

    if len(merged_index) == 0:
        raise ValueError(
            f"The {join!r} join of var indices resulted in an empty var. "
            "Check that your shards share at least one gene in common."
        )

    # Build merged var DataFrame: reindex the first shard's columns.
    merged_var = var_frames[0].reindex(merged_index)

    # get_indexer returns -1 for genes not present in merged_index (inner join
    # only — outer join is a union so every gene is always found).
    col_remaps = [merged_index.get_indexer(df.index).astype(np.int32) for df in var_frames]
    return merged_var, col_remaps, False


def _write_remapped_sparse(
    out_store: object,
    in_store: object,
    path: str,
    remap: np.ndarray,
    is_identity: bool,
    join: str,
    current_nnz: int,
    current_cell: int,
    num_cells: int,
) -> int:
    """Stream one shard's sparse CSR data with optional column remapping.

    Returns the updated ``current_nnz`` pointer.
    """
    old_data = in_store[path]["data"][:]  # type: ignore[index]
    old_indices = in_store[path]["indices"][:]  # type: ignore[index]
    # Cast to int64 immediately: on-disk indptr is int32 in h5ad files written by older
    # versions of anndata.  Adding a large current_nnz offset in int32 arithmetic silently
    # overflows once total nnz exceeds ~2.1 B, producing negative indptr values and a
    # structurally corrupt output CSR matrix.  Casting unconditionally is safe: it is a
    # no-op for files that already store int64, and protective for int32 files.
    old_indptr = in_store[path]["indptr"][:].astype(np.int64)  # type: ignore[index]

    if is_identity:
        shard_nnz = len(old_data)
        out_store[path]["data"][current_nnz : current_nnz + shard_nnz] = old_data  # type: ignore[index]
        out_store[path]["indices"][current_nnz : current_nnz + shard_nnz] = old_indices  # type: ignore[index]
        shifted_indptr = old_indptr + current_nnz
        out_store[path]["indptr"][current_cell : current_cell + num_cells] = shifted_indptr[:-1]  # type: ignore[index]
        return current_nnz + shard_nnz

    new_indices = remap[old_indices]

    if join == "outer":
        # All entries are kept; only column positions change.
        shard_nnz = len(old_data)
        out_store[path]["data"][current_nnz : current_nnz + shard_nnz] = old_data  # type: ignore[index]
        out_store[path]["indices"][current_nnz : current_nnz + shard_nnz] = new_indices  # type: ignore[index]
        shifted_indptr = old_indptr + current_nnz
        out_store[path]["indptr"][current_cell : current_cell + num_cells] = shifted_indptr[:-1]  # type: ignore[index]
        return current_nnz + shard_nnz

    # Inner join: drop entries for genes absent from the intersection.
    mask = new_indices >= 0
    surviving_data = old_data[mask]
    surviving_indices = new_indices[mask]

    # Recompute indptr: count survivors per row then cumsum.
    row_nnz = np.diff(old_indptr)  # shape: (num_cells,)
    if len(old_data) > 0:
        row_idx = np.repeat(np.arange(num_cells, dtype=np.int32), row_nnz)
        new_row_nnz = np.bincount(row_idx[mask], minlength=num_cells)
    else:
        new_row_nnz = np.zeros(num_cells, dtype=np.int64)

    new_local_indptr = np.concatenate([[0], np.cumsum(new_row_nnz)])
    shard_surviving = int(mask.sum())

    out_store[path]["data"][current_nnz : current_nnz + shard_surviving] = surviving_data  # type: ignore[index]
    out_store[path]["indices"][current_nnz : current_nnz + shard_surviving] = surviving_indices  # type: ignore[index]
    out_store[path]["indptr"][current_cell : current_cell + num_cells] = (  # type: ignore[index]
        new_local_indptr[:-1] + current_nnz
    )
    return current_nnz + shard_surviving


# ---------------------------------------------------------------------------
# Output integrity validation
# ---------------------------------------------------------------------------


def _validate_merge_indptr(out_store: object, expected_nnz_map: dict[str, int]) -> None:
    """Validate that CSR ``indptr`` arrays in the output store are non-corrupt.

    Checks two conditions for each sparse group path:

    1. No negative values — a negative entry is the signature of an int32
       overflow during the indptr-shift arithmetic in :func:`_write_remapped_sparse`.
    2. The final value equals the expected total nnz — guards against
       off-by-one errors or a truncated write.

    Only the compact ``indptr`` arrays are read (``n_cells + 1`` int64 values),
    so this check is cheap even for very large files.

    Parameters
    ----------
    out_store:
        An open h5py ``File`` or zarr ``Group`` for the output file.
    expected_nnz_map:
        Mapping of sparse group path to expected total non-zero count,
        e.g. ``{"X": 12_345_678, "layers/counts": 9_876_543}``.

    Raises
    ------
    RuntimeError
        If any ``indptr`` array contains negative values or its final entry
        does not match the expected nnz.
    """
    for path, expected_nnz in expected_nnz_map.items():
        indptr = out_store[path]["indptr"][:].astype(np.int64)  # type: ignore[index]
        if np.any(indptr < 0):
            raise RuntimeError(
                f"Corrupt output detected: '{path}/indptr' contains negative values "
                f"(likely caused by int32 overflow during the indptr shift). "
                f"Re-run the merge to obtain a correct output file."
            )
        actual_nnz = int(indptr[-1])
        if actual_nnz != expected_nnz:
            raise RuntimeError(
                f"Corrupt output detected: '{path}/indptr' ends at {actual_nnz}, "
                f"expected {expected_nnz}."
            )


# ---------------------------------------------------------------------------
# Main merge function
# ---------------------------------------------------------------------------


def merge_out_of_core(
    input_files: list[str],
    output_file: str,
    join: str = "outer",
    validate: bool = False,
) -> None:
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
    join:
        How to join the var (gene) axes when files differ.
        ``"outer"`` (default) takes the union of all gene sets and fills
        missing entries with zeros.  ``"inner"`` keeps only genes present in
        every shard.  Layers absent from any shard are always dropped.
    validate:
        When ``True``, checks every ``indptr`` array in the output file for
        negative values and the correct final nnz immediately after writing,
        raising :class:`RuntimeError` if corruption is detected.  The check
        reads only the compact ``indptr`` arrays so the overhead is negligible
        even for large files.  Defaults to ``False``.
    """
    if join not in ("inner", "outer"):
        raise ValueError(f"join must be 'inner' or 'outer', got {join!r}")

    is_zarr = output_file.endswith(".zarr")
    format_name = "Zarr" if is_zarr else "H5AD"

    # Validate zarr availability early — before doing any work.
    if is_zarr:
        _require_zarr()

    # -----------------------------------------------------------------------
    # Phase 1: gather metadata (obs, var, uns) with minimal memory
    # -----------------------------------------------------------------------
    logger.info("Phase 1: Building the metadata skeleton (%s format)...", format_name)

    var_frames: list[pd.DataFrame] = []
    obs_list: list[pd.DataFrame] = []
    layer_keys_per_shard: list[set[str]] = []
    total_cells: int = 0

    for f in input_files:
        store = open_store(f, "r")
        try:
            var_frames.append(read_elem(store["var"]))
            obs_df: pd.DataFrame = read_elem(store["obs"])
            obs_list.append(obs_df)
            total_cells += obs_df.shape[0]
            layer_keys_per_shard.append(
                set(store["layers"].keys()) if "layers" in store else set()
            )
        finally:
            if hasattr(store, "close"):
                store.close()

    store_first = open_store(input_files[0], "r")
    try:
        uns: dict = read_elem(store_first["uns"]) if "uns" in store_first else {}
    finally:
        if hasattr(store_first, "close"):
            store_first.close()

    merged_var, col_remaps, is_identity = _compute_merged_var(var_frames, join)
    num_genes: int = len(merged_var)

    # Only merge layers present in *every* shard; absent layers would produce
    # an inconsistent indptr and corrupt the output CSR matrix.
    active_layers: set[str] = (
        set.intersection(*layer_keys_per_shard) if layer_keys_per_shard else set()
    )

    merged_obs = pd.concat(obs_list, axis=0)
    skeleton = ad.AnnData(obs=merged_obs, var=merged_var, uns=uns)
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
    x_is_sparse: bool | None = None  # determined from first shard that has X
    x_dtype: np.dtype = np.dtype(np.float32)

    for i, f in enumerate(input_files):
        store = open_store(f, "r")
        try:
            if "X" in store:
                if x_is_sparse is None:
                    x_is_sparse = _is_sparse_group(store, "X")
                    if not x_is_sparse:
                        x_dtype = store["X"].dtype  # type: ignore[union-attr]
                if x_is_sparse:
                    if join == "inner" and not is_identity:
                        old_idx = store["X"]["indices"][:]
                        shard_nnz = int((col_remaps[i][old_idx] >= 0).sum())
                    else:
                        shard_nnz = int(store["X"]["data"].shape[0])
                    matrix_nnz["X"] = matrix_nnz.get("X", 0) + shard_nnz

            for layer in active_layers:
                lpath = f"layers/{layer}"
                if (
                    "layers" in store
                    and layer in store["layers"]
                    and _is_sparse_group(store, lpath)
                ):
                    if join == "inner" and not is_identity:
                        old_idx = store[lpath]["indices"][:]
                        lnnz = int((col_remaps[i][old_idx] >= 0).sum())
                    else:
                        lnnz = int(store[lpath]["data"].shape[0])
                    layer_nnz[layer] = layer_nnz.get(layer, 0) + lnnz

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
    elif x_is_sparse is False:
        # Dense X: allocate a plain 2-D dataset and stream row chunks.
        if "X" in out_store:
            del out_store["X"]
        chunk_rows = min(total_cells, 1_000)
        ds = out_store.create_dataset(
            "X",
            shape=(total_cells, num_genes),
            chunks=(chunk_rows, num_genes),
            dtype=x_dtype,
        )
        ds.attrs["encoding-type"] = "array"
        ds.attrs["encoding-version"] = "0.2.0"

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
        remap = col_remaps[i]
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
                current_nnz_X = _write_remapped_sparse(
                    out_store,
                    in_store,
                    "X",
                    remap,
                    is_identity,
                    join,
                    current_nnz_X,
                    current_cell,
                    num_cells_in_shard,
                )
            elif x_is_sparse is False and "X" in in_store:
                shard_X = in_store["X"][:]
                if is_identity:
                    out_store["X"][current_cell : current_cell + num_cells_in_shard, :] = shard_X
                elif join == "inner":
                    valid = remap >= 0
                    out_store["X"][  # type: ignore[index]
                        current_cell : current_cell + num_cells_in_shard, remap[valid]
                    ] = shard_X[:, valid]
                else:
                    out_store["X"][  # type: ignore[index]
                        current_cell : current_cell + num_cells_in_shard, remap
                    ] = shard_X

            # Stream layers
            for layer in layer_nnz:
                lpath = f"layers/{layer}"
                if (
                    "layers" in in_store
                    and layer in in_store["layers"]
                    and _is_sparse_group(in_store, lpath)
                ):
                    current_nnz_layers[layer] = _write_remapped_sparse(
                        out_store,
                        in_store,
                        lpath,
                        remap,
                        is_identity,
                        join,
                        current_nnz_layers[layer],
                        current_cell,
                        num_cells_in_shard,
                    )

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

    # Cap off indptr arrays at total_cells position (sparse matrices only)
    if "X" in matrix_nnz:
        out_store["X"]["indptr"][total_cells] = current_nnz_X
    for layer, final_nnz in current_nnz_layers.items():
        out_store[f"layers/{layer}"]["indptr"][total_cells] = final_nnz

    if validate:
        _nnz_map: dict[str, int] = {}
        if "X" in matrix_nnz:
            _nnz_map["X"] = current_nnz_X
        for layer, final_nnz in current_nnz_layers.items():
            _nnz_map[f"layers/{layer}"] = final_nnz
        _validate_merge_indptr(out_store, _nnz_map)

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
        help=(
            "Input shard files or glob patterns to merge, in order. "
            "Glob patterns are expanded lexicographically."
        ),
    )
    p.add_argument(
        "--join",
        choices=["inner", "outer"],
        default="outer",
        help="How to join var (gene) axes when files differ (default: outer).",
    )
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    """Dispatch function called by the CLI after argument parsing."""
    import glob as _glob_mod
    import sys

    expanded: list[str] = []
    for pattern in args.input_files:
        matches = sorted(_glob_mod.glob(pattern))
        if not matches:
            sys.exit(f"error: no files matched pattern: {pattern!r}")
        expanded.extend(matches)
    merge_out_of_core(expanded, args.output_file, join=args.join)
