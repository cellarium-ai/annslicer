"""
Tests for annslicer.slice
"""

from __future__ import annotations

import math

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from annslicer.slice import shard_h5ad

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

    for path, expected_n in zip(shard_paths, expected_counts, strict=True):
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
# Shuffle tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def shuffled_outputs(synthetic_h5ad, tmp_path):
    """Run shard_h5ad with shuffle=True and return shard paths."""
    prefix = str(tmp_path / "shuffled")
    shard_h5ad(synthetic_h5ad, prefix, shard_size=SHARD_SIZE, shuffle=True, seed=7)
    return sorted(tmp_path.glob("shuffled_shard*.h5ad"))


def test_shuffle_produces_all_cells(shuffled_outputs):
    """Shuffled shards collectively contain every cell exactly once."""
    all_indices: list[str] = []
    for path in shuffled_outputs:
        adata = ad.read_h5ad(path)
        all_indices.extend(adata.obs_names.tolist())
    assert len(all_indices) == N_CELLS
    assert len(set(all_indices)) == N_CELLS


def test_shuffle_order_differs_from_original(synthetic_h5ad, tmp_path):
    """Shuffled first shard has different cell order than the unshuffled version."""
    prefix_plain = str(tmp_path / "plain")
    prefix_shuf = str(tmp_path / "shuf")
    shard_h5ad(synthetic_h5ad, prefix_plain, shard_size=SHARD_SIZE, shuffle=False)
    shard_h5ad(synthetic_h5ad, prefix_shuf, shard_size=SHARD_SIZE, shuffle=True, seed=42)

    plain_names = ad.read_h5ad(sorted(tmp_path.glob("plain_shard*.h5ad"))[0]).obs_names.tolist()
    shuf_names = ad.read_h5ad(sorted(tmp_path.glob("shuf_shard*.h5ad"))[0]).obs_names.tolist()
    # With high probability (1 - 1/50! ≈ 1) the first shard differs.
    assert plain_names != shuf_names, "Shuffled order should differ from original"


def test_shuffle_reproducible_with_seed(synthetic_h5ad, tmp_path):
    """Same seed produces identical shard contents on repeated runs."""
    for run in ("run1", "run2"):
        shard_h5ad(
            synthetic_h5ad, str(tmp_path / run), shard_size=SHARD_SIZE, shuffle=True, seed=0
        )

    for shard_n in range(1, math.ceil(N_CELLS / SHARD_SIZE) + 1):
        p1 = tmp_path / f"run1_shard{shard_n:03d}.h5ad"
        p2 = tmp_path / f"run2_shard{shard_n:03d}.h5ad"
        a1 = ad.read_h5ad(p1)
        a2 = ad.read_h5ad(p2)
        assert a1.obs_names.tolist() == a2.obs_names.tolist(), f"Shard {shard_n} differs"
        np.testing.assert_array_equal(
            a1.X.toarray() if sp.issparse(a1.X) else a1.X,
            a2.X.toarray() if sp.issparse(a2.X) else a2.X,
        )


def test_shuffle_different_seeds_differ(synthetic_h5ad, tmp_path):
    """Different seeds produce different shuffles."""
    shard_h5ad(synthetic_h5ad, str(tmp_path / "s1"), shard_size=SHARD_SIZE, shuffle=True, seed=1)
    shard_h5ad(synthetic_h5ad, str(tmp_path / "s2"), shard_size=SHARD_SIZE, shuffle=True, seed=2)

    names1 = ad.read_h5ad(list(sorted(tmp_path.glob("s1_shard*.h5ad")))[0]).obs_names.tolist()
    names2 = ad.read_h5ad(list(sorted(tmp_path.glob("s2_shard*.h5ad")))[0]).obs_names.tolist()
    assert names1 != names2, "Different seeds should (almost certainly) produce different shuffles"


# ---------------------------------------------------------------------------
# Zarr input tests
# ---------------------------------------------------------------------------


def test_slice_zarr_input_cell_count(synthetic_zarr, tmp_path):
    """Slicing a zarr input produces the same number of shards with correct cell counts."""
    prefix = str(tmp_path / "zarr_in")
    shard_h5ad(synthetic_zarr, prefix, shard_size=SHARD_SIZE)

    shard_paths = sorted(tmp_path.glob("zarr_in_shard*.h5ad"))
    assert len(shard_paths) == math.ceil(N_CELLS / SHARD_SIZE)

    for path in shard_paths:
        adata = ad.read_h5ad(path)
        assert adata.n_vars == N_GENES


def test_slice_zarr_input_all_cells(synthetic_zarr, tmp_path):
    """Slicing a zarr input yields all cells with no duplicates."""
    prefix = str(tmp_path / "zarr_all")
    shard_h5ad(synthetic_zarr, prefix, shard_size=SHARD_SIZE)

    all_indices: list[str] = []
    for path in sorted(tmp_path.glob("zarr_all_shard*.h5ad")):
        all_indices.extend(ad.read_h5ad(path).obs_names.tolist())
    assert len(all_indices) == N_CELLS
    assert len(set(all_indices)) == N_CELLS


def test_slice_zarr_input_shuffle(synthetic_zarr, tmp_path):
    """Zarr input + shuffle still produces all cells exactly once."""
    prefix = str(tmp_path / "zarr_shuf")
    shard_h5ad(synthetic_zarr, prefix, shard_size=SHARD_SIZE, shuffle=True, seed=99)

    all_indices: list[str] = []
    for path in sorted(tmp_path.glob("zarr_shuf_shard*.h5ad")):
        all_indices.extend(ad.read_h5ad(path).obs_names.tolist())
    assert len(all_indices) == N_CELLS
    assert len(set(all_indices)) == N_CELLS
