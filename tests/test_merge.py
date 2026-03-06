"""
Tests for annslicer.merge — h5ad and zarr output formats.
"""

from __future__ import annotations

import math
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
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


# ---------------------------------------------------------------------------
# var join tests (inner / outer) — mismatched gene sets
# ---------------------------------------------------------------------------

# Two tiny shards with partially overlapping genes:
#   Shard A: genes 0-39  (40 genes)
#   Shard B: genes 10-49 (40 genes)
#   Overlap (inner join result): genes 10-39 (30 genes)
#   Union  (outer join result):  genes 0-49  (50 genes)

_JOIN_N_CELLS = 20
_JOIN_GENES_A = [f"g{i:04d}" for i in range(40)]  # g0000–g0039
_JOIN_GENES_B = [f"g{i:04d}" for i in range(10, 50)]  # g0010–g0049
_JOIN_OVERLAP = [f"g{i:04d}" for i in range(10, 40)]  # 30 genes
_JOIN_UNION = [f"g{i:04d}" for i in range(50)]  # 50 genes


def _make_shard(
    tmp_path: Path,
    name: str,
    gene_names: list[str],
    n_cells: int = _JOIN_N_CELLS,
    *,
    with_layer: bool = True,
) -> str:
    """Create a small synthetic h5ad shard with the given gene set."""
    n_genes = len(gene_names)
    X = sp.random(n_cells, n_genes, density=0.3, format="csr", random_state=42).astype(np.float32)
    obs = pd.DataFrame(index=[f"{name}_cell{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=gene_names)
    layers = {"counts": X.copy()} if with_layer else {}
    adata = ad.AnnData(X=X, obs=obs, var=var, layers=layers)
    path = str(tmp_path / f"{name}.h5ad")
    adata.write_h5ad(path)
    return path


@pytest.fixture()
def mismatched_pair(tmp_path: Path) -> tuple[str, str, ad.AnnData, ad.AnnData]:
    """Two shards with overlapping but non-identical gene sets. Returns (pathA, pathB, adataA, adataB)."""
    path_a = _make_shard(tmp_path, "A", _JOIN_GENES_A)
    path_b = _make_shard(tmp_path, "B", _JOIN_GENES_B)
    return path_a, path_b, ad.read_h5ad(path_a), ad.read_h5ad(path_b)


def test_outer_join_var_is_union(mismatched_pair: tuple, tmp_path: Path) -> None:
    path_a, path_b, _, _ = mismatched_pair
    out = str(tmp_path / "outer.h5ad")
    merge_out_of_core([path_a, path_b], out, join="outer")
    merged = ad.read_h5ad(out)
    assert merged.n_vars == len(_JOIN_UNION)
    assert list(merged.var_names) == _JOIN_UNION


def test_outer_join_cell_count(mismatched_pair: tuple, tmp_path: Path) -> None:
    path_a, path_b, _, _ = mismatched_pair
    out = str(tmp_path / "outer.h5ad")
    merge_out_of_core([path_a, path_b], out, join="outer")
    merged = ad.read_h5ad(out)
    assert merged.n_obs == _JOIN_N_CELLS * 2


def test_outer_join_X_values_for_overlap_genes(mismatched_pair: tuple, tmp_path: Path) -> None:
    """For genes present in shard A, merged rows for shard-A cells must match the original."""
    path_a, path_b, adata_a, _ = mismatched_pair
    out = str(tmp_path / "outer.h5ad")
    merge_out_of_core([path_a, path_b], out, join="outer")
    merged = ad.read_h5ad(out)

    # Rows 0:_JOIN_N_CELLS belong to shard A; columns for genes 0-39
    merged_X = _to_dense(merged.X)
    orig_X = _to_dense(adata_a.X)
    # Gene indices in merged var for genes g0000-g0039 are 0-39 (same positions)
    np.testing.assert_array_almost_equal(merged_X[:_JOIN_N_CELLS, :40], orig_X)


def test_outer_join_X_zeros_for_absent_genes(mismatched_pair: tuple, tmp_path: Path) -> None:
    """Cells from shard A should have zeros for genes g0040-g0049 (only in shard B)."""
    path_a, path_b, _, _ = mismatched_pair
    out = str(tmp_path / "outer.h5ad")
    merge_out_of_core([path_a, path_b], out, join="outer")
    merged = ad.read_h5ad(out)
    merged_X = _to_dense(merged.X)
    # genes g0040-g0049 are columns 40-49 in merged var
    np.testing.assert_array_equal(merged_X[:_JOIN_N_CELLS, 40:], 0)


def test_outer_join_is_default(mismatched_pair: tuple, tmp_path: Path) -> None:
    """Calling merge_out_of_core without join= should behave identically to join='outer'."""
    path_a, path_b, _, _ = mismatched_pair
    out_default = str(tmp_path / "default.h5ad")
    out_outer = str(tmp_path / "outer.h5ad")
    merge_out_of_core([path_a, path_b], out_default)
    merge_out_of_core([path_a, path_b], out_outer, join="outer")
    m_default = ad.read_h5ad(out_default)
    m_outer = ad.read_h5ad(out_outer)
    assert list(m_default.var_names) == list(m_outer.var_names)
    np.testing.assert_array_equal(_to_dense(m_default.X), _to_dense(m_outer.X))


def test_inner_join_var_is_intersection(mismatched_pair: tuple, tmp_path: Path) -> None:
    path_a, path_b, _, _ = mismatched_pair
    out = str(tmp_path / "inner.h5ad")
    merge_out_of_core([path_a, path_b], out, join="inner")
    merged = ad.read_h5ad(out)
    assert merged.n_vars == len(_JOIN_OVERLAP)
    assert list(merged.var_names) == _JOIN_OVERLAP


def test_inner_join_X_values_correct(mismatched_pair: tuple, tmp_path: Path) -> None:
    """Spot-check that inner-joined X has the expected values from each shard."""
    path_a, path_b, adata_a, adata_b = mismatched_pair
    out = str(tmp_path / "inner.h5ad")
    merge_out_of_core([path_a, path_b], out, join="inner")
    merged = ad.read_h5ad(out)
    merged_X = _to_dense(merged.X)

    # Shard A rows in merged: columns are genes g0010-g0039
    # In shard A, those are columns 10-39.
    orig_a = _to_dense(adata_a.X)
    np.testing.assert_array_almost_equal(merged_X[:_JOIN_N_CELLS, :], orig_a[:, 10:40])

    # Shard B rows in merged: columns are genes g0010-g0039
    # In shard B, those are columns 0-29.
    orig_b = _to_dense(adata_b.X)
    np.testing.assert_array_almost_equal(merged_X[_JOIN_N_CELLS:, :], orig_b[:, :30])


def test_inner_join_cell_count(mismatched_pair: tuple, tmp_path: Path) -> None:
    path_a, path_b, _, _ = mismatched_pair
    out = str(tmp_path / "inner.h5ad")
    merge_out_of_core([path_a, path_b], out, join="inner")
    merged = ad.read_h5ad(out)
    assert merged.n_obs == _JOIN_N_CELLS * 2


def test_layer_dropped_when_missing_from_shard(tmp_path: Path) -> None:
    """A layer present in only one shard must be absent from the merged output."""
    path_a = _make_shard(tmp_path, "A_layer", _JOIN_GENES_A, with_layer=True)
    path_b = _make_shard(tmp_path, "B_nolayer", _JOIN_GENES_B, with_layer=False)
    out = str(tmp_path / "no_layer.h5ad")
    merge_out_of_core([path_a, path_b], out)
    merged = ad.read_h5ad(out)
    assert "counts" not in merged.layers


def test_identical_var_fast_path_unchanged(tmp_path: Path) -> None:
    """When all shards share the same var, the merged X must be bit-identical to a simple concat."""
    path_a = _make_shard(tmp_path, "id_A", _JOIN_GENES_A)
    path_b = _make_shard(tmp_path, "id_B", _JOIN_GENES_A)  # same genes as A
    out = str(tmp_path / "identity.h5ad")
    merge_out_of_core([path_a, path_b], out)
    merged = ad.read_h5ad(out)
    assert merged.n_vars == len(_JOIN_GENES_A)
    adata_a = ad.read_h5ad(path_a)
    adata_b = ad.read_h5ad(path_b)
    expected = np.vstack([_to_dense(adata_a.X), _to_dense(adata_b.X)])
    np.testing.assert_array_equal(_to_dense(merged.X), expected)


def test_invalid_join_raises(tmp_path: Path) -> None:
    path_a = _make_shard(tmp_path, "inv_A", _JOIN_GENES_A)
    out = str(tmp_path / "bad.h5ad")
    with pytest.raises(ValueError, match="join must be"):
        merge_out_of_core([path_a], out, join="left")


# ---------------------------------------------------------------------------
# CLI glob expansion test
# ---------------------------------------------------------------------------


def test_cli_glob_expansion(tmp_path: Path) -> None:
    """CLI _run() expands glob patterns and passes expanded list to merge."""
    import argparse

    from annslicer.merge import _run

    _make_shard(tmp_path, "glob_shard_001", _JOIN_GENES_A)
    _make_shard(tmp_path, "glob_shard_002", _JOIN_GENES_A)

    out = str(tmp_path / "glob_merged.h5ad")
    pattern = str(tmp_path / "glob_shard_*.h5ad")

    args = argparse.Namespace(input_files=[pattern], output_file=out, join="outer")
    _run(args)

    merged = ad.read_h5ad(out)
    assert merged.n_obs == _JOIN_N_CELLS * 2
    assert merged.n_vars == len(_JOIN_GENES_A)


def test_cli_glob_no_match_exits(tmp_path: Path) -> None:
    """CLI _run() calls sys.exit when a pattern matches no files."""
    import argparse

    from annslicer.merge import _run

    out = str(tmp_path / "nowhere.h5ad")
    args = argparse.Namespace(
        input_files=[str(tmp_path / "nonexistent_*.h5ad")],
        output_file=out,
        join="outer",
    )
    with pytest.raises(SystemExit):
        _run(args)
