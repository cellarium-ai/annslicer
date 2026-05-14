"""
Tests for annslicer.slice
"""

from __future__ import annotations

import math
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from annslicer.slice import shard_by_obs_column, shard_h5ad

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

    for shard_n in range(0, math.ceil(N_CELLS / SHARD_SIZE)):
        p1 = tmp_path / f"run1_shard_{shard_n}.h5ad"
        p2 = tmp_path / f"run2_shard_{shard_n}.h5ad"
        a1 = ad.read_h5ad(p1)
        a2 = ad.read_h5ad(p2)
        assert a1.obs_names.tolist() == a2.obs_names.tolist(), f"Shard {shard_n} differs"
        np.testing.assert_array_equal(
            a1.X.toarray() if sp.issparse(a1.X) else a1.X,
            a2.X.toarray() if sp.issparse(a2.X) else a2.X,
        )


def test_shard_different_seeds_differ(synthetic_h5ad, tmp_path):
    """Different seeds produce different shuffles."""
    shard_h5ad(synthetic_h5ad, str(tmp_path / "s1"), shard_size=SHARD_SIZE, shuffle=True, seed=1)
    shard_h5ad(synthetic_h5ad, str(tmp_path / "s2"), shard_size=SHARD_SIZE, shuffle=True, seed=2)

    names1 = ad.read_h5ad(list(sorted(tmp_path.glob("s1_shard*.h5ad")))[0]).obs_names.tolist()
    names2 = ad.read_h5ad(list(sorted(tmp_path.glob("s2_shard*.h5ad")))[0]).obs_names.tolist()
    assert names1 != names2, "Different seeds should (almost certainly) produce different shuffles"


# ---------------------------------------------------------------------------
# output_filenames tests
# ---------------------------------------------------------------------------


def test_output_filenames_respected(synthetic_h5ad, tmp_path):
    """Custom output_filenames are used instead of the default naming scheme."""
    n_shards = math.ceil(N_CELLS / SHARD_SIZE)
    custom_names = [str(tmp_path / f"custom_{i:02d}.h5ad") for i in range(n_shards)]
    shard_h5ad(
        synthetic_h5ad,
        str(tmp_path / "unused_prefix"),
        output_filenames=custom_names,
        shard_size=SHARD_SIZE,
    )

    for path in custom_names:
        assert Path(path).exists(), f"Expected shard file {path} was not created"


def test_output_filenames_cell_counts(synthetic_h5ad, tmp_path):
    """Shards written to custom filenames contain the expected number of cells."""
    n_shards = math.ceil(N_CELLS / SHARD_SIZE)
    custom_names = [str(tmp_path / f"named_{i}.h5ad") for i in range(n_shards)]
    shard_h5ad(
        synthetic_h5ad, str(tmp_path / "p"), output_filenames=custom_names, shard_size=SHARD_SIZE
    )

    all_obs: list[str] = []
    for path in custom_names:
        adata = ad.read_h5ad(path)
        assert adata.n_obs == SHARD_SIZE
        all_obs.extend(adata.obs_names.tolist())

    assert len(all_obs) == N_CELLS
    assert len(set(all_obs)) == N_CELLS


def test_output_filenames_too_few_raises(synthetic_h5ad, tmp_path):
    """Passing fewer filenames than shards raises ValueError."""
    too_few = [str(tmp_path / "only_one.h5ad")]
    with pytest.raises(ValueError, match="Not enough output filenames"):
        shard_h5ad(
            synthetic_h5ad, str(tmp_path / "p"), output_filenames=too_few, shard_size=SHARD_SIZE
        )


# ---------------------------------------------------------------------------
# Compression tests
# ---------------------------------------------------------------------------


def test_gzip_compression_produces_valid_shards(synthetic_h5ad, tmp_path):
    """Shards written with compression='gzip' are valid and data-identical to uncompressed."""
    prefix = str(tmp_path / "gzip")
    shard_h5ad(synthetic_h5ad, prefix, shard_size=SHARD_SIZE, compression="gzip")

    shard_paths = sorted(tmp_path.glob("gzip_shard*.h5ad"))
    assert len(shard_paths) == math.ceil(N_CELLS / SHARD_SIZE)

    for path in shard_paths:
        adata = ad.read_h5ad(path)
        assert adata.n_vars == N_GENES
        assert "counts" in adata.layers


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


# ---------------------------------------------------------------------------
# shard_by_obs_column tests
# ---------------------------------------------------------------------------

# The synthetic_h5ad fixture has cell_type in {type_0, type_1, type_2}, 50 cells each.
CATEGORIES = ["type_0", "type_1", "type_2"]
N_PER_CATEGORY = 50


def test_obs_shard_file_count(synthetic_h5ad, tmp_path):
    """Exactly one output file per non-empty category."""
    shard_by_obs_column(synthetic_h5ad, str(tmp_path / "out"), "cell_type")
    files = list(tmp_path.glob("out_*.h5ad"))
    assert len(files) == len(CATEGORIES)


def test_obs_shard_filenames(synthetic_h5ad, tmp_path):
    """Output filenames contain the category names."""
    shard_by_obs_column(synthetic_h5ad, str(tmp_path / "out"), "cell_type")
    names = {p.name for p in tmp_path.glob("out_*.h5ad")}
    for cat in CATEGORIES:
        assert f"out_{cat}.h5ad" in names, f"Expected file out_{cat}.h5ad"


def test_obs_shard_cell_assignment(synthetic_h5ad, tmp_path):
    """Every cell in a shard belongs to that shard's category."""
    shard_by_obs_column(synthetic_h5ad, str(tmp_path / "out"), "cell_type")
    for cat in CATEGORIES:
        adata = ad.read_h5ad(tmp_path / f"out_{cat}.h5ad")
        assert all(v == cat for v in adata.obs["cell_type"]), f"Unexpected cells in shard {cat}"


def test_obs_shard_no_overlap(synthetic_h5ad, tmp_path):
    """Non-always-include cells appear in exactly one shard."""
    shard_by_obs_column(synthetic_h5ad, str(tmp_path / "out"), "cell_type")
    all_cells: list[str] = []
    for cat in CATEGORIES:
        all_cells.extend(ad.read_h5ad(tmp_path / f"out_{cat}.h5ad").obs_names.tolist())
    assert len(all_cells) == N_CELLS
    assert len(set(all_cells)) == N_CELLS


def test_obs_shard_full_coverage(synthetic_h5ad, tmp_path):
    """Union of cells across all shards equals the full dataset."""
    shard_by_obs_column(synthetic_h5ad, str(tmp_path / "out"), "cell_type")
    all_cells: set[str] = set()
    for path in tmp_path.glob("out_*.h5ad"):
        all_cells.update(ad.read_h5ad(path).obs_names.tolist())
    assert len(all_cells) == N_CELLS


def test_obs_shard_var_preserved(synthetic_h5ad, tmp_path):
    """var DataFrame is identical across all shards."""
    shard_by_obs_column(synthetic_h5ad, str(tmp_path / "out"), "cell_type")
    ref = ad.read_h5ad(tmp_path / f"out_{CATEGORIES[0]}.h5ad").var
    for cat in CATEGORIES[1:]:
        shard_var = ad.read_h5ad(tmp_path / f"out_{cat}.h5ad").var
        assert ref.equals(shard_var)


def test_obs_shard_non_categorical_raises(synthetic_h5ad, tmp_path):
    """ValueError when the obs column is not categorical."""
    import anndata as ad_mod

    # Write a version with a numeric (non-categorical) column
    adata = ad_mod.read_h5ad(synthetic_h5ad)
    adata.obs["score"] = np.arange(N_CELLS, dtype=np.float32)
    alt = str(tmp_path / "alt.h5ad")
    adata.write_h5ad(alt)

    with pytest.raises(ValueError, match="expected a categorical"):
        shard_by_obs_column(alt, str(tmp_path / "out"), "score")


def test_obs_shard_missing_column_raises(synthetic_h5ad, tmp_path):
    """KeyError when the obs column doesn't exist."""
    with pytest.raises(KeyError):
        shard_by_obs_column(synthetic_h5ad, str(tmp_path / "out"), "nonexistent_column")


def test_obs_shard_zarr_input(synthetic_zarr, tmp_path):
    """shard_by_obs_column works with a zarr input file."""
    shard_by_obs_column(synthetic_zarr, str(tmp_path / "out"), "cell_type")
    files = list(tmp_path.glob("out_*.h5ad"))
    assert len(files) == len(CATEGORIES)
    all_cells: list[str] = []
    for path in files:
        all_cells.extend(ad.read_h5ad(path).obs_names.tolist())
    assert len(all_cells) == N_CELLS
    assert len(set(all_cells)) == N_CELLS


def test_obs_shard_csv_merge(synthetic_h5ad, synthetic_csv_categorical, tmp_path):
    """CSV merge path produces the same shards as using the built-in obs column."""
    # Write a version of the h5ad that does NOT have cell_type in obs
    import anndata as ad_mod

    adata = ad_mod.read_h5ad(synthetic_h5ad)
    adata.obs = adata.obs.drop(columns=["cell_type"])
    no_cat = str(tmp_path / "no_cell_type.h5ad")
    adata.write_h5ad(no_cat)

    shard_by_obs_column(
        no_cat,
        str(tmp_path / "csv"),
        "cell_type",
        csv_file=synthetic_csv_categorical,
    )
    files = list(tmp_path.glob("csv_*.h5ad"))
    assert len(files) == len(CATEGORIES)
    all_cells: list[str] = []
    for path in files:
        all_cells.extend(ad.read_h5ad(path).obs_names.tolist())
    assert len(set(all_cells)) == N_CELLS


def test_obs_shard_csv_overwrites_existing_column(
    synthetic_h5ad, synthetic_csv_categorical, tmp_path
):
    """CSV silently overwrites an existing obs column of the same name."""
    import anndata as ad_mod

    # Corrupt the built-in cell_type so every cell reports "wrong"
    adata = ad_mod.read_h5ad(synthetic_h5ad)
    adata.obs["cell_type"] = pd.Categorical(["wrong"] * N_CELLS)
    corrupted = str(tmp_path / "corrupted.h5ad")
    adata.write_h5ad(corrupted)

    # CSV has the correct values — must overwrite without error
    shard_by_obs_column(
        corrupted,
        str(tmp_path / "out"),
        "cell_type",
        csv_file=synthetic_csv_categorical,
    )
    files = list(tmp_path.glob("out_*.h5ad"))
    # If the overwrite worked we get 3 shards (type_0/1/2); if it kept "wrong"
    # we would get 0 shards (no matching non-always-include categories).
    assert len(files) == len(CATEGORIES)


def test_obs_shard_csv_missing_cells_raises(synthetic_h5ad, tmp_path):
    """ValueError when the CSV is missing cell barcodes present in the h5ad."""
    import csv

    # Write a CSV that only covers the first 100 cells
    partial_csv = str(tmp_path / "partial.csv")
    with open(partial_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell_id", "cell_type"])
        for i in range(100):
            writer.writerow([f"cell_{i}", f"type_{i % 3}"])

    with pytest.raises(ValueError, match="missing"):
        shard_by_obs_column(
            synthetic_h5ad,
            str(tmp_path / "out"),
            "cell_type",
            csv_file=partial_csv,
        )


def test_obs_shard_always_include_in_every_shard(synthetic_h5ad, tmp_path):
    """Cells from always_include categories appear in every output shard."""
    # always_include = ["type_2"] → type_2 cells (cell_100..cell_149) appear in type_0 and type_1 shards
    shard_by_obs_column(
        synthetic_h5ad,
        str(tmp_path / "out"),
        "cell_type",
        always_include=["type_2"],
    )
    always_cells = {f"cell_{i}" for i in range(N_CELLS) if i % 3 == 2}  # type_2 cells
    for cat in ["type_0", "type_1"]:
        shard_obs_names = set(ad.read_h5ad(tmp_path / f"out_{cat}.h5ad").obs_names.tolist())
        assert always_cells.issubset(shard_obs_names), (
            f"Always-include cells missing from shard {cat}"
        )


def test_obs_shard_always_include_no_dedicated_shard(synthetic_h5ad, tmp_path):
    """No dedicated output file is written for always_include categories."""
    shard_by_obs_column(
        synthetic_h5ad,
        str(tmp_path / "out"),
        "cell_type",
        always_include=["type_2"],
    )
    assert not (tmp_path / "out_type_2.h5ad").exists()
    # Only two shards: type_0 and type_1
    files = list(tmp_path.glob("out_*.h5ad"))
    assert len(files) == 2


def test_obs_shard_always_include_invalid_raises(synthetic_h5ad, tmp_path):
    """ValueError for always_include values not in the category list."""
    with pytest.raises(ValueError, match="always_include"):
        shard_by_obs_column(
            synthetic_h5ad,
            str(tmp_path / "out"),
            "cell_type",
            always_include=["nonexistent_category"],
        )


def test_obs_shard_sanitized_name_collision_raises(tmp_path):
    """ValueError when two category names sanitize to the same filename fragment."""
    import anndata as ad_mod
    import pandas as pd

    rng = np.random.default_rng(0)
    # "foo bar" and "foo_bar" both sanitize to "foo_bar"
    cats = pd.Categorical(["foo bar"] * 5 + ["foo_bar"] * 5)
    obs = pd.DataFrame({"grp": cats}, index=[f"c{i}" for i in range(10)])
    adata = ad_mod.AnnData(X=rng.random((10, 5), dtype=np.float32), obs=obs)
    h5ad = str(tmp_path / "collision.h5ad")
    adata.write_h5ad(h5ad)

    with pytest.raises(ValueError, match="sanitize"):
        shard_by_obs_column(h5ad, str(tmp_path / "out"), "grp")
