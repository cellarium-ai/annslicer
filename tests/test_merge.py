"""
Tests for annslicer.merge — h5ad and zarr output formats.
"""

from __future__ import annotations

import math
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from annslicer.merge import merge_out_of_core
from annslicer.slice import shard_h5ad

N_CELLS = 150
N_GENES = 50
SHARD_SIZE = 50  # 3 shards from 150 cells


# ---------------------------------------------------------------------------
# Shared fixture: slice the synthetic h5ad into 3 shards
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sliced_shards(synthetic_h5ad: str, tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    """Shard the synthetic h5ad and return the sorted list of shard paths."""
    out_dir = tmp_path_factory.mktemp("shards")
    prefix = str(out_dir / "shard")
    shard_h5ad(synthetic_h5ad, prefix, shard_size=SHARD_SIZE)
    return sorted(str(p) for p in out_dir.glob("shard_shard*.h5ad"))


# ---------------------------------------------------------------------------
# H5AD merge tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def merged_h5ad(sliced_shards: list[str], tmp_path_factory: pytest.TempPathFactory) -> str:
    """Merge the shards into a single h5ad and return the output path."""
    out_dir = tmp_path_factory.mktemp("merged_h5ad")
    out_path = str(out_dir / "merged.h5ad")
    merge_out_of_core(sliced_shards, out_path)
    return out_path


def test_merge_h5ad_output_exists(merged_h5ad: str) -> None:
    assert Path(merged_h5ad).exists()


def test_merge_h5ad_cell_count(merged_h5ad: str) -> None:
    adata = ad.read_h5ad(merged_h5ad)
    assert adata.n_obs == N_CELLS


def test_merge_h5ad_gene_count(merged_h5ad: str) -> None:
    adata = ad.read_h5ad(merged_h5ad)
    assert adata.n_vars == N_GENES


def test_merge_h5ad_var_preserved(merged_h5ad: str, synthetic_h5ad: str) -> None:
    merged = ad.read_h5ad(merged_h5ad)
    original = ad.read_h5ad(synthetic_h5ad)
    assert merged.var.shape == original.var.shape
    assert list(merged.var_names) == list(original.var_names)


def test_merge_h5ad_obs_no_duplicates(merged_h5ad: str) -> None:
    adata = ad.read_h5ad(merged_h5ad)
    obs_names = list(adata.obs_names)
    assert len(obs_names) == N_CELLS
    assert len(set(obs_names)) == N_CELLS


def test_merge_h5ad_layer_present(merged_h5ad: str) -> None:
    adata = ad.read_h5ad(merged_h5ad)
    assert "counts" in adata.layers
    assert adata.layers["counts"].shape == (N_CELLS, N_GENES)


def test_merge_h5ad_obsm_present(merged_h5ad: str) -> None:
    adata = ad.read_h5ad(merged_h5ad)
    assert "X_pca" in adata.obsm
    assert adata.obsm["X_pca"].shape == (N_CELLS, 10)


def test_merge_h5ad_shard_count(sliced_shards: list[str]) -> None:
    """Sanity check: we produced the expected number of shards."""
    assert len(sliced_shards) == math.ceil(N_CELLS / SHARD_SIZE)


# ---------------------------------------------------------------------------
# Zarr merge tests — skipped automatically if zarr is not installed
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sliced_zarr_shards() -> list[str]:
    """Placeholder fixture — zarr merge tests use sliced_shards (h5ad) as inputs."""
    return []


@pytest.fixture(scope="module")
def merged_zarr(sliced_shards: list[str], tmp_path_factory: pytest.TempPathFactory) -> str:
    """Merge h5ad shards into a single zarr store and return its path."""
    pytest.importorskip("zarr", reason="zarr not installed; skipping zarr merge tests")
    out_dir = tmp_path_factory.mktemp("merged_zarr")
    out_path = str(out_dir / "merged.zarr")
    merge_out_of_core(sliced_shards, out_path)
    return out_path


def test_merge_zarr_output_exists(merged_zarr: str) -> None:
    assert Path(merged_zarr).exists()


def test_merge_zarr_cell_count(merged_zarr: str) -> None:
    adata = ad.read_zarr(merged_zarr)
    assert adata.n_obs == N_CELLS


def test_merge_zarr_gene_count(merged_zarr: str) -> None:
    adata = ad.read_zarr(merged_zarr)
    assert adata.n_vars == N_GENES


def test_merge_zarr_var_preserved(merged_zarr: str, synthetic_h5ad: str) -> None:
    merged = ad.read_zarr(merged_zarr)
    original = ad.read_h5ad(synthetic_h5ad)
    assert list(merged.var_names) == list(original.var_names)


def test_merge_zarr_obs_no_duplicates(merged_zarr: str) -> None:
    adata = ad.read_zarr(merged_zarr)
    obs_names = list(adata.obs_names)
    assert len(obs_names) == N_CELLS
    assert len(set(obs_names)) == N_CELLS


def test_merge_zarr_layer_present(merged_zarr: str) -> None:
    adata = ad.read_zarr(merged_zarr)
    assert "counts" in adata.layers
    assert adata.layers["counts"].shape == (N_CELLS, N_GENES)


def test_merge_zarr_obsm_present(merged_zarr: str) -> None:
    adata = ad.read_zarr(merged_zarr)
    assert "X_pca" in adata.obsm
    assert adata.obsm["X_pca"].shape == (N_CELLS, 10)


def test_merge_zarr_requires_zarr_installed(
    sliced_shards: list[str], tmp_path: pytest.TempPathFactory
) -> None:
    """merge_out_of_core to .zarr succeeds when zarr is available."""
    pytest.importorskip("zarr", reason="zarr not installed; skipping zarr merge tests")

    out_path = str(tmp_path / "check.zarr")  # type: ignore[operator]
    # Just check it doesn't raise ImportError
    merge_out_of_core(sliced_shards, out_path)
    assert Path(out_path).exists()


# ---------------------------------------------------------------------------
# Zarr slice → merge round-trip tests
# ---------------------------------------------------------------------------


def _to_dense(arr) -> np.ndarray:
    return arr.toarray() if sp.issparse(arr) else np.asarray(arr)


def test_zarr_slice_merge_roundtrip(synthetic_zarr: str, tmp_path) -> None:
    """Slice zarr → h5ad shards → merge back to zarr preserves all data."""
    pytest.importorskip("zarr", reason="zarr not installed")

    prefix = str(tmp_path / "sliced")
    shard_h5ad(synthetic_zarr, prefix, shard_size=SHARD_SIZE)
    shards = sorted(str(p) for p in tmp_path.glob("sliced_shard*.h5ad"))

    out_zarr = str(tmp_path / "merged.zarr")
    merge_out_of_core(shards, out_zarr)

    original = ad.read_zarr(synthetic_zarr)
    merged = ad.read_zarr(out_zarr)

    assert list(merged.obs_names) == list(original.obs_names)
    assert list(merged.var_names) == list(original.var_names)
    np.testing.assert_array_almost_equal(_to_dense(merged.X), _to_dense(original.X))
    np.testing.assert_array_almost_equal(
        _to_dense(merged.layers["counts"]), _to_dense(original.layers["counts"])
    )
    np.testing.assert_array_almost_equal(merged.obsm["X_pca"], original.obsm["X_pca"])


def test_zarr_slice_merge_to_h5ad(synthetic_zarr: str, tmp_path) -> None:
    """Zarr input → h5ad shards → merged h5ad demonstrates format conversion."""
    pytest.importorskip("zarr", reason="zarr not installed")

    prefix = str(tmp_path / "sliced")
    shard_h5ad(synthetic_zarr, prefix, shard_size=SHARD_SIZE)
    shards = sorted(str(p) for p in tmp_path.glob("sliced_shard*.h5ad"))

    out_h5ad = str(tmp_path / "merged.h5ad")
    merge_out_of_core(shards, out_h5ad)

    original = ad.read_zarr(synthetic_zarr)
    merged = ad.read_h5ad(out_h5ad)

    assert merged.n_obs == original.n_obs
    assert merged.n_vars == original.n_vars
    assert list(merged.obs_names) == list(original.obs_names)
    assert list(merged.var_names) == list(original.var_names)
    assert "counts" in merged.layers
    assert "X_pca" in merged.obsm
    np.testing.assert_array_almost_equal(_to_dense(merged.X), _to_dense(original.X))
