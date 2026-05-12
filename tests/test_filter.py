"""
Tests for annslicer.filter
"""

from __future__ import annotations

import csv
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from annslicer.filter import filter_h5ad

N_CELLS = 150
N_GENES = 50
N_KEEP = 100  # first 100 cells kept by the synthetic_csv_bool fixture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kept_cells() -> set[str]:
    return {f"cell_{i}" for i in range(N_KEEP)}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def filtered_h5ad(synthetic_h5ad, tmp_path_factory):
    """Filter the synthetic h5ad using the built-in 'keep' obs column and return the path."""
    import anndata as ad_mod

    # Add a boolean 'keep' column: True for first 100, False for last 50
    adata = ad_mod.read_h5ad(synthetic_h5ad)
    adata.obs["keep"] = [i < N_KEEP for i in range(N_CELLS)]
    out_dir = tmp_path_factory.mktemp("filter_data")
    with_keep = str(out_dir / "with_keep.h5ad")
    adata.write_h5ad(with_keep)

    out_path = str(out_dir / "filtered.h5ad")
    filter_h5ad(with_keep, out_path, obs_column="keep")
    return out_path


# ---------------------------------------------------------------------------
# Basic correctness tests
# ---------------------------------------------------------------------------


def test_filter_output_exists(filtered_h5ad):
    assert Path(filtered_h5ad).exists()


def test_filter_cell_count(filtered_h5ad):
    adata = ad.read_h5ad(filtered_h5ad)
    assert adata.n_obs == N_KEEP


def test_filter_correct_cells_kept(filtered_h5ad):
    adata = ad.read_h5ad(filtered_h5ad)
    assert set(adata.obs_names.tolist()) == _kept_cells()


def test_filter_var_preserved(filtered_h5ad, synthetic_h5ad):
    merged = ad.read_h5ad(filtered_h5ad)
    original = ad.read_h5ad(synthetic_h5ad)
    assert list(merged.var_names) == list(original.var_names)


def test_filter_layer_present(filtered_h5ad):
    adata = ad.read_h5ad(filtered_h5ad)
    assert "counts" in adata.layers
    assert adata.layers["counts"].shape == (N_KEEP, N_GENES)


def test_filter_obsm_present(filtered_h5ad):
    adata = ad.read_h5ad(filtered_h5ad)
    assert "X_pca" in adata.obsm
    assert adata.obsm["X_pca"].shape == (N_KEEP, 10)


def test_filter_all_false_writes_empty(synthetic_h5ad, tmp_path):
    """When all cells are False the output file has 0 cells."""
    import anndata as ad_mod

    adata = ad_mod.read_h5ad(synthetic_h5ad)
    adata.obs["keep"] = False
    alt = str(tmp_path / "all_false.h5ad")
    adata.write_h5ad(alt)
    out = str(tmp_path / "out.h5ad")
    filter_h5ad(alt, out, obs_column="keep")
    assert ad.read_h5ad(out).n_obs == 0


# ---------------------------------------------------------------------------
# CSV merge path
# ---------------------------------------------------------------------------


def test_filter_csv_merge(synthetic_h5ad, synthetic_csv_bool, tmp_path):
    """CSV merge path keeps the same cells as using a built-in obs column."""
    out = str(tmp_path / "filtered_csv.h5ad")
    filter_h5ad(synthetic_h5ad, out, obs_column="keep", csv_file=synthetic_csv_bool)
    adata = ad.read_h5ad(out)
    assert adata.n_obs == N_KEEP
    assert set(adata.obs_names.tolist()) == _kept_cells()


def test_filter_csv_missing_cells_raises(synthetic_h5ad, tmp_path):
    """ValueError when the CSV is missing some cell barcodes."""
    partial_csv = str(tmp_path / "partial.csv")
    with open(partial_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell_id", "keep"])
        for i in range(100):
            writer.writerow([f"cell_{i}", True])

    with pytest.raises(ValueError, match="missing"):
        filter_h5ad(synthetic_h5ad, str(tmp_path / "out.h5ad"), obs_column="keep",
                    csv_file=partial_csv)


# ---------------------------------------------------------------------------
# Zarr input
# ---------------------------------------------------------------------------


def test_filter_zarr_input(synthetic_zarr, tmp_path):
    """filter_h5ad works with a zarr input."""
    import anndata as ad_mod
    import anndata

    # zarr doesn't have a backed mode for writing only a subset of obs easily;
    # read the zarr, add a keep column, write a temp h5ad, then use that as input.
    adata = anndata.read_zarr(synthetic_zarr)
    adata.obs["keep"] = [i < N_KEEP for i in range(N_CELLS)]
    alt_zarr = str(tmp_path / "with_keep.zarr")
    pytest.importorskip("zarr")
    adata.write_zarr(alt_zarr)

    out = str(tmp_path / "filtered.h5ad")
    filter_h5ad(alt_zarr, out, obs_column="keep")
    result = ad.read_h5ad(out)
    assert result.n_obs == N_KEEP


# ---------------------------------------------------------------------------
# Lenient boolean coercion
# ---------------------------------------------------------------------------


def test_filter_bool_lenient_int(synthetic_h5ad, tmp_path):
    """0/1 integer column is accepted as boolean."""
    import anndata as ad_mod

    adata = ad_mod.read_h5ad(synthetic_h5ad)
    adata.obs["keep_int"] = [1 if i < N_KEEP else 0 for i in range(N_CELLS)]
    alt = str(tmp_path / "int_keep.h5ad")
    adata.write_h5ad(alt)

    out = str(tmp_path / "out.h5ad")
    filter_h5ad(alt, out, obs_column="keep_int")
    assert ad.read_h5ad(out).n_obs == N_KEEP


def test_filter_bool_lenient_string(synthetic_h5ad, tmp_path):
    """'True'/'False' string column is accepted as boolean."""
    import anndata as ad_mod

    adata = ad_mod.read_h5ad(synthetic_h5ad)
    adata.obs["keep_str"] = ["True" if i < N_KEEP else "False" for i in range(N_CELLS)]
    alt = str(tmp_path / "str_keep.h5ad")
    adata.write_h5ad(alt)

    out = str(tmp_path / "out.h5ad")
    filter_h5ad(alt, out, obs_column="keep_str")
    assert ad.read_h5ad(out).n_obs == N_KEEP


def test_filter_bool_invalid_strings_raise(synthetic_h5ad, tmp_path):
    """Unmappable string values in the filter column raise ValueError."""
    import anndata as ad_mod

    adata = ad_mod.read_h5ad(synthetic_h5ad)
    adata.obs["keep_bad"] = "yes"  # not a recognised boolean string
    alt = str(tmp_path / "bad_keep.h5ad")
    adata.write_h5ad(alt)

    with pytest.raises(ValueError, match="cannot be interpreted as boolean"):
        filter_h5ad(alt, str(tmp_path / "out.h5ad"), obs_column="keep_bad")
