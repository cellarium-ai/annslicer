"""
Tests for annslicer.core
"""

from __future__ import annotations

import math
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pytest
import scipy.sparse as sp

from annslicer.core import extract_matrix_slice, shard_h5ad

# Match the constants from conftest.py
N_CELLS = 150
N_GENES = 50
SHARD_SIZE = 50  # Produces exactly 3 shards from N_CELLS=150


# ---------------------------------------------------------------------------
# shard_h5ad integration tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def shard_outputs(synthetic_h5ad, tmp_path):
    """Run shard_h5ad and return (shard_paths, tmp_path)."""
    prefix = str(tmp_path / "out")
    shard_h5ad(synthetic_h5ad, prefix, shard_size=SHARD_SIZE)
    shard_paths = sorted(tmp_path.glob("out_shard*.h5ad"))
    return shard_paths, tmp_path


def test_shard_count(shard_outputs):
    """Correct number of shards is created."""
    shard_paths, _ = shard_outputs
    expected = math.ceil(N_CELLS / SHARD_SIZE)
    assert len(shard_paths) == expected


def test_shard_cell_counts(shard_outputs):
    """Each shard has the expected number of cells."""
    shard_paths, _ = shard_outputs
    expected_counts = [SHARD_SIZE] * (N_CELLS // SHARD_SIZE)
    remainder = N_CELLS % SHARD_SIZE
    if remainder:
        expected_counts.append(remainder)

    for path, expected_n in zip(shard_paths, expected_counts):
        adata = ad.read_h5ad(path)
        assert adata.n_obs == expected_n, f"{path.name}: expected {expected_n} cells"


def test_shard_var_preserved(shard_outputs):
    """var DataFrame is identical across all shards."""
    shard_paths, _ = shard_outputs
    ref_var = ad.read_h5ad(shard_paths[0]).var
    for path in shard_paths[1:]:
        shard_var = ad.read_h5ad(path).var
        assert ref_var.equals(shard_var), f"{path.name}: var mismatch"


def test_shard_gene_count(shard_outputs):
    """Every shard has the right number of genes."""
    shard_paths, _ = shard_outputs
    for path in shard_paths:
        adata = ad.read_h5ad(path)
        assert adata.n_vars == N_GENES


def test_shard_obsm_preserved(shard_outputs):
    """obsm keys exist and have the correct shape in each shard."""
    shard_paths, _ = shard_outputs
    for path in shard_paths:
        adata = ad.read_h5ad(path)
        assert "X_pca" in adata.obsm
        assert adata.obsm["X_pca"].shape == (adata.n_obs, 10)


def test_shard_layer_present(shard_outputs):
    """'counts' layer is present in every shard."""
    shard_paths, _ = shard_outputs
    for path in shard_paths:
        adata = ad.read_h5ad(path)
        assert "counts" in adata.layers
        assert adata.layers["counts"].shape == (adata.n_obs, N_GENES)


def test_shard_obs_no_overlap(shard_outputs):
    """Cell indices across shards are disjoint and together cover all cells."""
    shard_paths, _ = shard_outputs
    all_indices: list[str] = []
    for path in shard_paths:
        adata = ad.read_h5ad(path)
        all_indices.extend(adata.obs_names.tolist())
    assert len(all_indices) == N_CELLS
    assert len(set(all_indices)) == N_CELLS  # no duplicates


# ---------------------------------------------------------------------------
# extract_matrix_slice unit tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_h5(tmp_path):
    """Create a small HDF5 file with a dense dataset and a CSR sparse group."""
    rng = np.random.default_rng(0)
    dense = rng.random((20, 10), dtype=np.float32)
    sparse_mat = sp.random(20, 10, density=0.3, format="csr", dtype=np.float32,
                           random_state=rng)

    h5_path = tmp_path / "temp.h5"
    with h5py.File(h5_path, "w") as f:
        # Dense dataset
        f.create_dataset("dense", data=dense)

        # CSR sparse group
        grp = f.create_group("sparse_csr")
        grp.attrs["encoding-type"] = "csr_matrix"
        grp.create_dataset("data", data=sparse_mat.data)
        grp.create_dataset("indices", data=sparse_mat.indices)
        grp.create_dataset("indptr", data=sparse_mat.indptr)

        # CSC sparse group (to test the rejection path)
        csc_mat = sparse_mat.tocsc()
        grp_csc = f.create_group("sparse_csc")
        grp_csc.attrs["encoding-type"] = "csc_matrix"
        grp_csc.create_dataset("data", data=csc_mat.data)
        grp_csc.create_dataset("indices", data=csc_mat.indices)
        grp_csc.create_dataset("indptr", data=csc_mat.indptr)

    return h5_path


def test_extract_dense_shape(temp_h5):
    """Dense extraction returns correct shape."""
    with h5py.File(temp_h5, "r") as f:
        result = extract_matrix_slice(f["dense"], 2, 8, 10)
    assert isinstance(result, np.ndarray)
    assert result.shape == (6, 10)


def test_extract_dense_values(temp_h5):
    """Dense extraction returns correct values."""
    with h5py.File(temp_h5, "r") as f:
        full = f["dense"][:]
        result = extract_matrix_slice(f["dense"], 3, 7, 10)
    np.testing.assert_array_equal(result, full[3:7])


def test_extract_sparse_shape(temp_h5):
    """Sparse (CSR) extraction returns correct shape."""
    with h5py.File(temp_h5, "r") as f:
        result = extract_matrix_slice(f["sparse_csr"], 5, 12, 10)
    assert sp.issparse(result)
    assert result.shape == (7, 10)


def test_extract_csc_raises(temp_h5):
    """CSC matrix raises NotImplementedError."""
    with h5py.File(temp_h5, "r") as f:
        with pytest.raises(NotImplementedError, match="CSC"):
            extract_matrix_slice(f["sparse_csc"], 0, 5, 10)
