"""
Tests for the two-pass shuffled sharding in annslicer._shuffle.

The oracle is the obvious in-memory implementation: load everything, take the global
permutation ``default_rng(seed).permutation(n)``, and cut it into shards.  The out-of-core
result must match it exactly, however the work is split into blocks, buckets and workers.
"""

from __future__ import annotations

import math
from collections import namedtuple
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from annslicer import _resources, _shuffle
from annslicer._common import _open_lazy
from annslicer.slice import shard_h5ad

N_CELLS = 150


def _dense(m):
    return m.toarray() if sp.issparse(m) else np.asarray(m)


def _assert_matches_reference(path: str, prefix: str, shard_size: int, seed: int) -> None:
    ref = ad.read_zarr(path) if path.endswith(".zarr") else ad.read_h5ad(path)
    perm = np.random.default_rng(seed).permutation(ref.n_obs)
    n_shards = math.ceil(ref.n_obs / shard_size)

    for j in range(n_shards):
        got = ad.read_h5ad(f"{prefix}_shard_{j}.h5ad")
        want = ref[perm[j * shard_size : (j + 1) * shard_size]]
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
@pytest.mark.parametrize(
    "shard_size, block_rows, shards_per_bucket",
    [
        (50, None, None),  # defaults: one block, one bucket
        (50, 7, 1),  # many blocks, one shard per bucket
        (50, 7, 2),  # bucket wider than one shard, last bucket partial
        (32, 13, 3),  # shard size does not divide the cell count
        (200, 40, None),  # a single shard holding every cell
        (1, 64, 40),  # one-cell shards, wide buckets
    ],
)
def test_matches_in_memory_reference(
    request, tmp_path, fixture, shard_size, block_rows, shards_per_bucket
):
    path = request.getfixturevalue(fixture)
    prefix = str(tmp_path / "out")
    _shuffle.shuffled_shards(
        path,
        _open_lazy(path),
        [f"{prefix}_shard_{j}.h5ad" for j in range(math.ceil(N_CELLS / shard_size))],
        shard_size,
        seed=11,
        n_jobs=1,
        block_rows=block_rows,
        shards_per_bucket=shards_per_bucket,
    )
    _assert_matches_reference(path, prefix, shard_size, seed=11)


def test_parallel_workers_match_reference(synthetic_sparse_h5ad, tmp_path):
    """Real worker processes (no ``__main__`` guard needed) give the same shards."""
    prefix = str(tmp_path / "par")
    shard_h5ad(
        synthetic_sparse_h5ad,
        prefix,
        shard_size=32,
        shuffle=True,
        seed=5,
        n_jobs=2,
        tmpdir=str(tmp_path),
    )
    _assert_matches_reference(synthetic_sparse_h5ad, prefix, 32, seed=5)


def test_large_obsm_survives_worker_transfer(tmp_path):
    """
    obsm payloads over joblib's 1 MB threshold must reach the workers as plain arrays
    (joblib would otherwise pass them as read-only np.memmap objects that anndata can't write).
    """
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
    shard_h5ad(path, prefix, shard_size=shard_size, shuffle=True, seed=1, n_jobs=2)
    _assert_matches_reference(path, prefix, shard_size, seed=1)


def test_public_api_matches_reference(synthetic_h5ad, tmp_path):
    prefix = str(tmp_path / "api")
    shard_h5ad(synthetic_h5ad, prefix, shard_size=50, shuffle=True, seed=3, n_jobs=1)
    _assert_matches_reference(synthetic_h5ad, prefix, 50, seed=3)


def test_scratch_removed_after_success_and_failure(synthetic_sparse_h5ad, tmp_path, monkeypatch):
    scratch_root = tmp_path / "scratch"
    shard_h5ad(
        synthetic_sparse_h5ad, str(tmp_path / "ok"), shard_size=50, shuffle=True, seed=0,
        n_jobs=1, tmpdir=str(scratch_root),
    )  # fmt: skip
    assert list(scratch_root.iterdir()) == []

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(_shuffle, "_write_bucket", boom)
    with pytest.raises(RuntimeError, match="boom"):
        shard_h5ad(
            synthetic_sparse_h5ad, str(tmp_path / "bad"), shard_size=50, shuffle=True, seed=0,
            n_jobs=1, tmpdir=str(scratch_root),
        )  # fmt: skip
    assert list(scratch_root.iterdir()) == []


def test_insufficient_scratch_space_raises(synthetic_sparse_h5ad, tmp_path, monkeypatch):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(_shuffle.shutil, "disk_usage", lambda _: usage(10, 10, 0))
    scratch_root = tmp_path / "scratch"
    with pytest.raises(OSError, match="scratch space"):
        shard_h5ad(
            synthetic_sparse_h5ad, str(tmp_path / "out"), shard_size=50, shuffle=True,
            n_jobs=1, tmpdir=str(scratch_root),
        )  # fmt: skip
    assert list(scratch_root.iterdir()) == []
    assert not list(tmp_path.glob("out_shard_*"))


def test_csc_input_is_rejected_for_shuffle(tmp_path):
    adata = ad.AnnData(X=sp.random(20, 10, density=0.3, format="csc", dtype=np.float32))
    path = str(tmp_path / "csc.h5ad")
    adata.write_h5ad(path)
    with pytest.raises(ValueError, match="CSR"):
        shard_h5ad(path, str(tmp_path / "out"), shard_size=10, shuffle=True, n_jobs=1)


def test_invalid_n_jobs_raises(synthetic_h5ad, tmp_path):
    with pytest.raises(ValueError, match="n_jobs"):
        shard_h5ad(synthetic_h5ad, str(tmp_path / "o"), shard_size=50, shuffle=True, n_jobs=0)


# ---------------------------------------------------------------------------
# Planning and resource helpers
# ---------------------------------------------------------------------------


def test_plan_memory_limit_caps_workers():
    shard_bytes = 100 * 1000  # 100-cell shards at 1000 bytes per cell
    roomy = _shuffle._plan(10_000, 100, 1000, n_jobs=8, memory_limit=10**9)
    tight = _shuffle._plan(
        10_000,
        100,
        1000,
        n_jobs=8,
        memory_limit=int(2.5 * _shuffle._BUCKET_MEM_FACTOR * shard_bytes),
    )
    assert roomy.n_jobs == 8
    assert tight.n_jobs == 2
    assert tight.block_rows <= roomy.block_rows


def test_plan_always_has_one_worker_and_valid_sizes():
    plan = _shuffle._plan(1000, 500, 10**6, n_jobs=8, memory_limit=1)  # absurdly small budget
    assert plan.n_jobs == 1 and plan.block_rows >= 1 and plan.shards_per_bucket == 1


def test_plan_wide_buckets_for_many_small_shards():
    plan = _shuffle._plan(1_000_000, 100, 100, n_jobs=4, memory_limit=10**10)
    assert plan.shards_per_bucket > 1
    assert plan.shards_per_bucket * 100 <= plan.block_rows * 100  # bucket fits in a block's worth


@pytest.mark.parametrize(
    "text, expected",
    [("512", 512), ("1KB", 1024), ("2MiB", 2 * 1024**2), ("1.5G", int(1.5 * 1024**3)), (7, 7)],
)
def test_parse_size(text, expected):
    assert _resources._parse_size(text) == expected


def test_parse_size_rejects_garbage():
    with pytest.raises(ValueError, match="memory size"):
        _resources._parse_size("lots")


def test_default_resources_are_sane():
    assert _resources._available_cpus() >= 1
    assert _resources._default_memory_limit() > 0


# ---------------------------------------------------------------------------
# Lazy opening: nothing big may be read into memory
# ---------------------------------------------------------------------------


def test_open_lazy_keeps_matrices_on_disk(synthetic_sparse_h5ad, synthetic_h5ad):
    sparse = _open_lazy(synthetic_sparse_h5ad)
    try:
        for mat in (sparse.X, sparse.layers["counts"]):
            assert hasattr(mat, "group") and not sp.issparse(mat)  # lazy sparse_dataset
        assert isinstance(sparse.layers["dense"], h5py.Dataset)
        assert set(sparse.matrices()) == {"X", "layers/counts", "layers/dense"}
    finally:
        sparse.close()

    dense = _open_lazy(synthetic_h5ad)
    try:
        assert isinstance(dense.X, h5py.Dataset)
        assert not sp.issparse(dense.layers["counts"])
    finally:
        dense.close()


def test_open_lazy_zarr_keeps_dense_arrays_lazy(synthetic_zarr):
    data = _open_lazy(synthetic_zarr)
    assert not isinstance(data.X, np.ndarray)
    assert data.X.shape == (N_CELLS, 50)
