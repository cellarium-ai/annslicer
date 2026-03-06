"""
Shared low-level store utilities for opening and inspecting HDF5 / Zarr files.

Both ``slice.py`` and ``merge.py`` need the same helpers; keeping them here
avoids circular imports and lets mypy overrides be scoped to merge.py only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import h5py

if TYPE_CHECKING:
    import zarr as zarr_t

# Public type alias: a root store is either an open HDF5 file or a Zarr group.
Store = Union[h5py.File, "zarr_t.Group"]


def _require_zarr() -> zarr_t:
    """Import and return the zarr module, raising a helpful ImportError if absent."""
    try:
        import zarr  # type: ignore[import-untyped]

        return zarr  # type: ignore[return-value]
    except ImportError as exc:
        raise ImportError(
            "Zarr support requires the 'zarr' package, which is not installed.\n"
            "Install it with:  pip install zarr\n"
            "Or install annslicer with the zarr extra:  pip install annslicer[zarr]"
        ) from exc


def open_store(path: str, mode: str = "r") -> Store:
    """Open an HDF5 or Zarr store and return the root group."""
    if path.endswith(".zarr"):
        zarr = _require_zarr()
        # Zarr v3 reads consolidated metadata by default.  For mutable access
        # (append / write) the consolidated metadata becomes stale as soon as we
        # add new groups or arrays, causing subsequent key-lookups to fail.
        # Passing use_consolidated=False forces live store traversal instead.
        try:
            return zarr.open_group(path, mode=mode, use_consolidated=False)  # type: ignore[return-value]
        except TypeError:
            # zarr v2 — use_consolidated kwarg does not exist.
            return zarr.open(path, mode=mode)  # type: ignore[return-value]
    elif path.endswith(".h5ad"):
        return h5py.File(path, mode=mode)
    else:
        raise ValueError(f"Unsupported file format: '{path}'. Files must end in .h5ad or .zarr.")


def _is_sparse_group(store: Store, key: str) -> bool:
    """Return True if *store[key]* is a sparse CSR group rather than a dense dataset."""
    item = store[key]
    if isinstance(item, h5py.Dataset):
        return False
    # For both h5py.Group and zarr.Group the presence of 'data' + 'indptr'
    # sub-keys indicates a sparse CSR representation.
    return "data" in item and "indptr" in item
