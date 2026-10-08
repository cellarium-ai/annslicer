"""
Tests for the parallel, unshuffled sharding in annslicer._contiguous.

The oracle is plain slicing of the fully loaded input: shard ``j`` must be rows
``[j * shard_size, (j + 1) * shard_size)``, whatever the worker count.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from annslicer import _contiguous
from annslicer.slice import shard_h5ad

N_CELLS = 150


def _dense(m):
    return m.toarray() if sp.issparse(m) else np.asarray(m)


def _assert_contiguous_shards(path: str, prefix: str, shard_size: int) -> None:
    ref = ad.read_zarr(path) if path.endswith(".zarr") else ad.read_h5ad(path)
    n_shards = math.ceil(ref.n_obs / shard_size)

    for j in range(n_shards):
        got = ad.read_h5ad(f"{prefix}_shard_{j}.h5ad")
        want = ref[j * shard_size : (j + 1) * shard_size]
        assert got.obs_names.tolist() == want.obs_names.tolist(), f"shard {j} cells differ"
        np.testing.assert_array_equal(_dense(got.X), _dense(want.X))
        assert got.X.dtype == want.X.dtype
        assert sp.issparse(got.X) == sp.issparse(want.X)
        assert set(got.layers) == set(want.layers)
        for k in want.layers:
            np.testing.assert_array_equal(_dense(got.layers[k]), _dense(want.layers[k]))
            assert got.layers[k].dtype == want.layers[k].dtype
        np.testing.assert_array_equal(got.obsm["X_pca"], want.obsm["X_pca"])
    assert len(list(Path(prefix).parent.glob("*_shard_*.h5ad"))) == n_shards


INPUTS = ["synthetic_h5ad", "synthetic_sparse_h5ad", "synthetic_zarr", "synthetic_sparse_zarr"]


@pytest.mark.parametrize("fixture", INPUTS)
@pytest.mark.parametrize("shard_size", [50, 32, 200, 1])  # 1: one-cell shards
def test_matches_slicing_reference(request, tmp_path, fixture, shard_size):
    path = request.getfixturevalue(fixture)
    prefix = str(tmp_path / "out")
    shard_h5ad(path, prefix, shard_size=shard_size, n_jobs=1)
    _assert_contiguous_shards(path, prefix, shard_size)


def test_parallel_workers_match_reference(synthetic_sparse_h5ad, tmp_path):
    """Real worker processes (no ``__main__`` guard needed) give the same shards."""
    prefix = str(tmp_path / "par")
    shard_h5ad(synthetic_sparse_h5ad, prefix, shard_size=32, n_jobs=3)
    _assert_contiguous_shards(synthetic_sparse_h5ad, prefix, 32)


def test_parallel_output_equals_serial_output(synthetic_sparse_h5ad, tmp_path):
    shard_h5ad(synthetic_sparse_h5ad, str(tmp_path / "one"), shard_size=40, n_jobs=1)
    shard_h5ad(synthetic_sparse_h5ad, str(tmp_path / "two"), shard_size=40, n_jobs=2)
    for j in range(math.ceil(N_CELLS / 40)):
        one = ad.read_h5ad(tmp_path / f"one_shard_{j}.h5ad")
        two = ad.read_h5ad(tmp_path / f"two_shard_{j}.h5ad")
        assert one.obs.equals(two.obs)
        assert (one.X != two.X).nnz == 0


def test_large_obsm_survives_worker_transfer(tmp_path):
    """obsm payloads over joblib's 1 MB threshold must reach workers as plain arrays."""
    n_cells, shard_size = 4000, 2000
    rng = np.random.default_rng(0)
    adata = ad.AnnData(
        X=sp.random(n_cells, 20, density=0.3, format="csr", dtype=np.float32, random_state=rng),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)]),
        obsm={"X_pca": rng.random((n_cells, 80))},  # 2000 cells x 80 x 8 B = 1.3 MB per shard
    )
    path = str(tmp_path / "big_obsm.h5ad")
    adata.write_h5ad(path)

    prefix = str(tmp_path / "out")
    shard_h5ad(path, prefix, shard_size=shard_size, n_jobs=2)
    _assert_contiguous_shards(path, prefix, shard_size)


def test_csc_input_is_supported_without_shuffle(tmp_path):
    adata = ad.AnnData(
        X=sp.random(20, 10, density=0.3, format="csc", dtype=np.float32, random_state=0),
        obsm={"X_pca": np.random.default_rng(0).random((20, 3))},
    )
    path = str(tmp_path / "csc.h5ad")
    adata.write_h5ad(path)
    prefix = str(tmp_path / "out")
    shard_h5ad(path, prefix, shard_size=10, n_jobs=2)
    _assert_contiguous_shards(path, prefix, 10)


def test_memory_limit_caps_workers(synthetic_sparse_h5ad, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger=_contiguous.logger.name):
        shard_h5ad(
            synthetic_sparse_h5ad, str(tmp_path / "o"), shard_size=10, n_jobs=4, memory_limit="1KB"
        )
    assert "with 1 worker(s)" in caplog.text
    assert "more than" in caplog.text  # a single shard exceeds the limit: warn, but still run
    _assert_contiguous_shards(synthetic_sparse_h5ad, str(tmp_path / "o"), 10)


def test_workers_never_exceed_shard_count(synthetic_sparse_h5ad, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger=_contiguous.logger.name):
        shard_h5ad(synthetic_sparse_h5ad, str(tmp_path / "o"), shard_size=100, n_jobs=8)
    assert "2 shards with 2 worker(s)" in caplog.text


def test_invalid_n_jobs_raises(synthetic_h5ad, tmp_path):
    with pytest.raises(ValueError, match="n_jobs"):
        shard_h5ad(synthetic_h5ad, str(tmp_path / "o"), shard_size=50, n_jobs=0)
