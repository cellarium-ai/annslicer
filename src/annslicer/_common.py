"""
Shared helpers for annslicer: out-of-core shard writing and CSV obs merging.

Used by both ``slice.py`` and ``filter.py`` to avoid code duplication.
"""

from __future__ import annotations

import logging
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _unwrap(arr: np.ndarray) -> Any:
    """Unwrap the 0-d object array that h5py sometimes returns for backed sparse layers."""
    return arr.item() if isinstance(arr, np.ndarray) and arr.ndim == 0 else arr


def _write_shard_from_indices(
    adata: ad.AnnData,
    indices: np.ndarray,
    out_filename: str,
    compression: str | None = None,
) -> None:
    """
    Write a subset of an AnnData object (identified by integer row indices) to a
    new .h5ad file.

    Indices are sorted before reading so that disk access is sequential (efficient
    for both HDF5 and zarr backends).  The output preserves the source order — cells
    appear in the same relative order as they do in the input file.

    Parameters
    ----------
    adata:
        An already-opened (backed or in-memory) AnnData object.
    indices:
        Integer row indices to include.  Need not be sorted; they will be sorted
        internally before reading and the output will be in ascending index order.
    out_filename:
        Destination .h5ad path.
    compression:
        HDF5 compression filter (e.g. ``"gzip"``).  ``None`` writes uncompressed.
    """
    sorted_idx = np.sort(indices)

    X = _unwrap(adata.X[sorted_idx, :])
    layers = {k: _unwrap(adata.layers[k][sorted_idx, :]) for k in adata.layers}
    obsm = {k: np.asarray(adata.obsm[k][sorted_idx]) for k in adata.obsm}
    obs = adata.obs.iloc[sorted_idx]

    # address dragen h5ad error issue #10
    if "_index" in obs.columns:
        obs = obs.drop(columns=["_index"])
        logger.warning(
            "Dropped '_index' column from obs before writing h5ad and assuming it is redundant with obs_names."
        )

    ad.AnnData(
        X=X,
        obs=obs.copy(),
        var=adata.var.copy(),
        obsm=obsm,
        layers=layers,
        uns=adata.uns.copy(),
    ).write_h5ad(out_filename, compression=compression)


def _merge_csv_into_obs(
    obs_df: pd.DataFrame,
    csv_file: str,
    join_column: str | None = None,
) -> pd.DataFrame:
    """
    Read an auxiliary CSV file and left-join its columns onto an obs DataFrame.

    The CSV is joined on the obs index (cell barcodes).  By default the CSV's
    first column is used as the join key (treated as the cell barcode index).
    Pass ``join_column`` to use a named column instead.

    After joining, any column whose dtype is not already categorical is coerced
    to ``pd.CategoricalDtype``.  This ensures that columns sourced from a CSV
    (which are typically read as plain strings or numbers) behave correctly when
    used as the ``obs_column`` argument to :func:`shard_by_obs_column`.

    Parameters
    ----------
    obs_df:
        The existing ``adata.obs`` DataFrame (index = cell barcodes).
    csv_file:
        Path to the CSV file containing additional per-cell metadata.
    join_column:
        Name of the column in the CSV to use as the join key.  If ``None``,
        the first column is used.

    Returns
    -------
    pd.DataFrame
        A new obs DataFrame with the CSV columns appended.

    Raises
    ------
    ValueError
        If any cell barcode present in ``obs_df`` is absent from the CSV.
    """
    csv_df = pd.read_csv(csv_file, low_memory=False)  # full read avoids dtype warning

    if join_column is not None:
        csv_df = csv_df.set_index(join_column)
    else:
        csv_df = csv_df.set_index(csv_df.columns[0])

    # Normalise the CSV index to plain Python strings.
    #
    # pd.read_csv may infer the join-key column as int64 when barcodes look
    # numeric (e.g. "1", "2", …), which would silently break a join against a
    # string obs index.  Stripping whitespace guards against trailing spaces in
    # either the CSV or the h5ad obs index.
    csv_df.index = csv_df.index.astype(str).str.strip()

    # Normalise the obs index to plain Python strings for comparison and joining.
    #
    # AnnData obs indices are almost always string-valued, but the backing dtype
    # can differ across anndata / pandas versions: "object" (Python str), or the
    # newer pandas StringDtype (arrow-backed).  Using .astype(str) produces a
    # consistent object-dtype index that joins correctly with the CSV index.
    obs_index_str = obs_df.index.astype(str).str.strip()

    # Validate: every obs barcode must appear in the CSV.
    missing = obs_index_str.difference(csv_df.index)
    if len(missing) > 0:
        missing_list = ", ".join(str(m) for m in sorted(missing)[:20])
        suffix = f" ... ({len(missing) - 20} more)" if len(missing) > 20 else ""
        raise ValueError(
            f"The auxiliary CSV is missing {len(missing)} cell barcode(s) that are "
            f"present in the h5ad obs index: {missing_list}{suffix}.\n"
            f"Ensure the CSV contains a row for every cell in the input file."
        )

    # Coerce all non-categorical columns in the CSV portion to CategoricalDtype so
    # that string/int columns from a CSV can be used directly as obs_column for
    # shard_by_obs_column without requiring the user to pre-cast the column.
    for col in csv_df.columns:
        if not isinstance(csv_df[col].dtype, pd.CategoricalDtype):
            csv_df[col] = csv_df[col].astype("category")

    # Join on the normalised string index; restore the original index object
    # afterwards so the returned DataFrame has the same index dtype as the input.
    result = obs_df.set_axis(obs_index_str).join(csv_df, how="left")
    result.index = obs_df.index
    return result
