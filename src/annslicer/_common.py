"""
Shared helpers for annslicer: lazy store opening, shard writing, and CSV obs merging.

Used by ``slice.py``, ``filter.py`` and ``_shuffle.py`` to avoid code duplication.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from annslicer._obs import normalize_obs
from annslicer._store import open_store

try:
    from anndata.io import read_elem, sparse_dataset
except ImportError:  # anndata < 0.11
    from anndata.experimental import read_elem, sparse_dataset

logger = logging.getLogger(__name__)

_SPARSE_ENCODINGS = ("csr_matrix", "csc_matrix")
_DROPPED_GROUPS = ("obsp", "varm", "varp", "raw")
_DROPPED_MESSAGE = (
    "%s contains %s, which annslicer does not carry over: output files hold only "
    "X, layers, obs, var, obsm and uns."
)


def _dropped_groups(root: Any) -> list[str]:
    """The non-empty groups of an opened store that annslicer does not write to its outputs."""
    # Newer anndata stores an absent element (e.g. ``raw=None``) as a scalar "null" array.
    return [
        key
        for key in _DROPPED_GROUPS
        if key in root and root[key].attrs.get("encoding-type") != "null" and len(root[key]) > 0
    ]


@dataclass
class LazyData:
    """
    The contents of an .h5ad / .zarr store, with the big matrices left on disk.

    ``X`` and ``layers`` hold lazy handles (``sparse_dataset`` objects for sparse matrices,
    raw h5py / zarr arrays for dense ones) that support row slicing and fancy indexing
    without loading the matrix.  Everything else is small and read eagerly.
    """

    root: Any
    X: Any = None
    layers: dict[str, Any] = field(default_factory=dict)
    obs: pd.DataFrame = field(default_factory=pd.DataFrame)
    var: pd.DataFrame = field(default_factory=pd.DataFrame)
    obsm: dict[str, Any] = field(default_factory=dict)
    uns: dict[str, Any] = field(default_factory=dict)

    @property
    def n_obs(self) -> int:
        return len(self.obs)

    def matrices(self) -> dict[str, Any]:
        """Lazy matrices keyed by their store path: ``"X"``, ``"layers/<name>"``."""
        mats = {} if self.X is None else {"X": self.X}
        mats.update({f"layers/{k}": v for k, v in self.layers.items()})
        return mats

    def close(self) -> None:
        if isinstance(self.root, h5py.File):
            self.root.close()


def _open_root(path: str) -> Any:
    """Open an .h5ad (h5py.File) or .zarr (zarr group) store read-only."""
    return open_store(path, "r") if path.endswith(".zarr") else h5py.File(path, "r")


def _lazy_matrix(root: Any, key: str) -> Any:
    """Return a lazy handle for the matrix at *key* (``sparse_dataset`` if sparse, else the array)."""
    item = root[key]
    if item.attrs.get("encoding-type") in _SPARSE_ENCODINGS:
        return sparse_dataset(item)
    return item


def _open_lazy(path: str) -> LazyData:
    """
    Open *path* without loading any matrix data into RAM.

    Unlike ``anndata.read_h5ad(backed="r")``, this never materialises layers (some anndata
    versions load them eagerly) and behaves identically for .h5ad and .zarr inputs.
    """
    root = _open_root(path)
    data = LazyData(root=root)
    if "X" in root:
        data.X = _lazy_matrix(root, "X")
    if "layers" in root:
        data.layers = {k: _lazy_matrix(root["layers"], k) for k in root["layers"]}
    dropped = _dropped_groups(root)
    if dropped:
        logger.warning(_DROPPED_MESSAGE, path, ", ".join(dropped))
    data.obs = normalize_obs(read_elem(root["obs"]))
    data.var = read_elem(root["var"])
    data.obsm = read_elem(root["obsm"]) if "obsm" in root else {}
    data.uns = read_elem(root["uns"]) if "uns" in root else {}
    return data


def _read_rows(mat: Any, idx: np.ndarray | slice) -> Any:
    """
    Read rows of a lazy matrix.  *idx* is a slice or an increasing array of row indices.
    A contiguous run of indices is turned into a slice, which is far cheaper to read than
    fancy indexing (a few large reads instead of several small reads per row).
    """
    if isinstance(idx, np.ndarray) and idx.size and idx[-1] - idx[0] + 1 == idx.size:
        idx = slice(int(idx[0]), int(idx[-1]) + 1)
    return mat[idx, :]


def _take_rows(a: Any, idx: np.ndarray) -> Any:
    """Select rows *idx* (in that order) from an in-memory array / DataFrame / sparse matrix."""
    if isinstance(a, pd.DataFrame):
        return a.iloc[idx]
    if sp.issparse(a):
        return a[idx]
    return np.asarray(a)[idx]


def _ensure_parent_dir(output_prefix: str) -> None:
    """Create the parent directory of *output_prefix* if it does not already exist."""
    parent = os.path.dirname(output_prefix)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_h5ad_shard(
    X: Any,
    layers: dict[str, Any],
    obs: pd.DataFrame,
    var: pd.DataFrame,
    obsm: dict[str, Any],
    uns: dict[str, Any],
    out_filename: str,
    compression: str | None = None,
) -> None:
    """Assemble an in-memory AnnData from already-read pieces and write it to *out_filename*."""
    # address dragen h5ad error issue #10
    if "_index" in obs.columns:
        obs = obs.drop(columns=["_index"])
        logger.warning(
            "Dropped '_index' column from obs before writing h5ad and assuming it is redundant with obs_names."
        )

    ad.AnnData(
        X=X,
        obs=obs.copy(),
        var=var.copy(),
        obsm=obsm,
        layers=layers,
        uns=uns.copy(),
    ).write_h5ad(out_filename, compression=compression, convert_strings_to_categoricals=False)


def _write_shard_from_indices(
    data: LazyData,
    indices: np.ndarray,
    out_filename: str,
    compression: str | None = None,
) -> None:
    """
    Write a subset of a :class:`LazyData` (identified by integer row indices) to a new
    .h5ad file.

    Indices are sorted before reading so that disk access is sequential (efficient
    for both HDF5 and zarr backends).  The output preserves the source order — cells
    appear in the same relative order as they do in the input file.

    Parameters
    ----------
    data:
        An opened store (see :func:`_open_lazy`).
    indices:
        Integer row indices to include.  Need not be sorted; they will be sorted
        internally before reading and the output will be in ascending index order.
    out_filename:
        Destination .h5ad path.
    compression:
        HDF5 compression filter (e.g. ``"gzip"``).  ``None`` writes uncompressed.
    """
    sorted_idx = np.sort(indices)

    X = None if data.X is None else _read_rows(data.X, sorted_idx)
    layers = {k: _read_rows(v, sorted_idx) for k, v in data.layers.items()}
    obsm = {k: _take_rows(v, sorted_idx) for k, v in data.obsm.items()}
    _write_h5ad_shard(
        X, layers, data.obs.iloc[sorted_idx], data.var, obsm, data.uns, out_filename, compression
    )


def _merge_csv_into_obs(
    obs_df: pd.DataFrame,
    csv_file: str,
    obs_column: str,
    join_column: str | None = None,
) -> pd.DataFrame:
    """
    Read a single column from an auxiliary CSV file and merge it into an obs DataFrame.

    Only ``obs_column`` is taken from the CSV — no other columns are touched.
    If ``obs_column`` already exists in ``obs_df`` it is overwritten with the
    CSV value, which allows the same CSV to be used across multiple commands
    (e.g. ``filter`` followed by ``slice``) without collision errors.

    The CSV is joined on the obs index (cell barcodes).  By default the CSV's
    first column is used as the join key (treated as the cell barcode index).
    Pass ``join_column`` to use a named column instead.

    The merged column is coerced to ``pd.CategoricalDtype`` so that it can be
    used directly as the ``obs_column`` argument to :func:`shard_by_obs_column`
    without requiring the user to pre-cast it.

    Parameters
    ----------
    obs_df:
        The existing ``adata.obs`` DataFrame (index = cell barcodes).
    csv_file:
        Path to the CSV file containing additional per-cell metadata.
    obs_column:
        The single column from the CSV to merge into obs.
    join_column:
        Column in the CSV to use as the cell-barcode join key.  If ``None``,
        the first column is used.

    Returns
    -------
    pd.DataFrame
        A new obs DataFrame with ``obs_column`` added (or overwritten).

    Raises
    ------
    KeyError
        If ``obs_column`` is not present as a column in the CSV.
    ValueError
        If any cell barcode present in ``obs_df`` is absent from the CSV.
    """
    csv_df = pd.read_csv(csv_file, low_memory=False)  # full read avoids dtype warning

    if join_column is not None:
        csv_df = csv_df.set_index(join_column)
    else:
        csv_df = csv_df.set_index(csv_df.columns[0])

    # Validate that the requested column exists in the CSV.
    if obs_column not in csv_df.columns:
        raise KeyError(
            f"Column {obs_column!r} not found in CSV file {csv_file!r}. "
            f"Available columns: {list(csv_df.columns)}."
        )

    # Restrict to only the one column we need.
    csv_df = csv_df[[obs_column]]

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

    # Coerce to CategoricalDtype so the column can be used directly by
    # shard_by_obs_column without requiring the caller to pre-cast it.
    if not isinstance(csv_df[obs_column].dtype, pd.CategoricalDtype):
        csv_df[obs_column] = csv_df[obs_column].astype("category")

    # Drop the column from obs_df if it already exists so the join does not
    # produce duplicate column names (e.g. when the same CSV is reused across
    # a filter run followed by a slice run on the resulting file).
    if obs_column in obs_df.columns:
        obs_df = obs_df.drop(columns=[obs_column])

    # Join on the normalised string index; restore the original index object
    # afterwards so the returned DataFrame has the same index dtype as the input.
    result = obs_df.set_axis(obs_index_str).join(csv_df, how="left")
    result.index = obs_df.index
    return result
