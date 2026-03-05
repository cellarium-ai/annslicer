"""
Core logic for annslicer: out-of-core sharding of .h5ad files.
"""

from __future__ import annotations

import argparse
import logging

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def extract_matrix_slice(
    h5_obj: h5py.Dataset | h5py.Group,
    start_idx: int,
    end_idx: int,
    num_cols: int,
) -> np.ndarray | sp.csr_matrix:
    """
    Safely extracts a row slice from an HDF5 object (either dense or sparse CSR).

    Parameters
    ----------
    h5_obj:
        An open h5py Dataset (dense) or Group (sparse) representing a matrix.
    start_idx:
        First row index (inclusive).
    end_idx:
        Last row index (exclusive).
    num_cols:
        Total number of columns in the matrix (genes / features).

    Returns
    -------
    A numpy ndarray (dense) or scipy csr_matrix (sparse) for the requested rows.
    """
    if isinstance(h5_obj, h5py.Dataset):
        # Dense matrix — slice directly from disk.
        logger.debug("Extracting dense slice rows %d:%d", start_idx, end_idx)
        return h5_obj[start_idx:end_idx, :]

    elif isinstance(h5_obj, h5py.Group):
        # Sparse matrix — validate encoding and read only the needed data.
        encoding = h5_obj.attrs.get("encoding-type", b"")
        if isinstance(encoding, bytes):
            encoding = encoding.decode("utf-8")

        if encoding == "csc_matrix":
            raise NotImplementedError(
                "Encountered a CSC matrix. Row-slicing a CSC matrix out-of-core "
                "is highly inefficient. Please convert to CSR first."
            )

        if "indptr" not in h5_obj:
            raise ValueError("Sparse group is missing the 'indptr' array.")

        logger.debug("Extracting sparse (CSR) slice rows %d:%d", start_idx, end_idx)

        # Read only the indptr entries for this chunk of rows.
        indptr_chunk = h5_obj["indptr"][start_idx : end_idx + 1]
        start_ptr = int(indptr_chunk[0])
        end_ptr = int(indptr_chunk[-1])

        # Pull the exact slice of data and column indices into RAM.
        data_chunk = h5_obj["data"][start_ptr:end_ptr]
        indices_chunk = h5_obj["indices"][start_ptr:end_ptr]

        # Reset pointers so they start at 0 for the new matrix.
        new_indptr = indptr_chunk - start_ptr

        return sp.csr_matrix(
            (data_chunk, indices_chunk, new_indptr),
            shape=(end_idx - start_idx, num_cols),
        )

    else:
        raise ValueError(f"Unknown HDF5 object type encountered: {type(h5_obj)}")


def shard_h5ad(
    input_file: str,
    output_prefix: str,
    shard_size: int = 10000,
) -> None:
    """
    Shard a large .h5ad file into smaller files using minimal RAM.

    Uses h5py for surgical row-slicing of X and layers, and a backed AnnData
    for obs, var, obsm, and uns metadata.

    Parameters
    ----------
    input_file:
        Path to the source .h5ad file.
    output_prefix:
        Prefix for output shard filenames, e.g. ``"dataset"`` produces
        ``dataset_shard001.h5ad``, ``dataset_shard002.h5ad``, etc.
    shard_size:
        Number of cells (rows) per shard. Defaults to 10 000.
    """
    logger.info("Opening %s for metadata...", input_file)

    # Open backed AnnData only to read metadata (obs, var, obsm, uns).
    adata_backed = ad.read_h5ad(input_file, backed="r")
    total_cells = adata_backed.shape[0]
    num_genes = adata_backed.shape[1]

    # Open h5py in parallel to surgically extract matrix data.
    with h5py.File(input_file, "r") as h5_file:
        logger.info(
            "Total cells: %d. Generating shards of %d...", total_cells, shard_size
        )

        for start_idx in range(0, total_cells, shard_size):
            end_idx = min(start_idx + shard_size, total_cells)
            shard_num = (start_idx // shard_size) + 1

            out_filename = f"{output_prefix}_shard{shard_num:03d}.h5ad"
            logger.info(
                "  Building %s (cells %d–%d)...", out_filename, start_idx, end_idx
            )

            # --- Step A: Extract main X matrix ---
            X_shard = None
            if "X" in h5_file:
                X_shard = extract_matrix_slice(h5_file["X"], start_idx, end_idx, num_genes)

            # --- Step B: Extract all layers ---
            layers_shard: dict = {}
            if "layers" in h5_file:
                for layer_name in h5_file["layers"].keys():
                    logger.debug("    Extracting layer '%s'...", layer_name)
                    layers_shard[layer_name] = extract_matrix_slice(
                        h5_file["layers"][layer_name], start_idx, end_idx, num_genes
                    )

            # --- Step C: Extract obs metadata and obsm embeddings ---
            obs_shard = adata_backed.obs.iloc[start_idx:end_idx].copy()

            obsm_shard: dict = {}
            for key in adata_backed.obsm.keys():
                logger.debug("    Extracting obsm key '%s'...", key)
                obsm_shard[key] = np.array(adata_backed.obsm[key][start_idx:end_idx])

            # --- Step D: Assemble and write shard ---
            adata_shard = ad.AnnData(
                X=X_shard,
                obs=obs_shard,
                var=adata_backed.var.copy(),
                obsm=obsm_shard,
                layers=layers_shard,
                uns=adata_backed.uns.copy(),
            )

            adata_shard.write_h5ad(out_filename)

    logger.info("All shards successfully created.")


def main() -> None:
    """Command-line entry point for the ``annslice`` command."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="annslice",
        description="Safely shard large .h5ad files out-of-core (includes layers and obsm).",
    )
    parser.add_argument("input_file", help="Path to the input .h5ad file.")
    parser.add_argument(
        "output_prefix",
        help="Prefix for output shard files (e.g. 'my_dataset').",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=10000,
        metavar="N",
        help="Number of cells per shard (default: 10000).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging.",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    shard_h5ad(args.input_file, args.output_prefix, args.size)
