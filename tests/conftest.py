"""
Pytest fixtures for annslicer tests.

Creates a small, synthetic .h5ad file on disk so tests run with zero large
file dependencies. The fixture uses tmp_path (built-in pytest fixture) so
everything is cleaned up automatically after each test session.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

N_CELLS = 150
N_GENES = 50


@pytest.fixture(scope="session")
def synthetic_h5ad(tmp_path_factory: pytest.TempPathFactory) -> str:
    """
    Write a small .h5ad file (150 cells × 50 genes) and return its path.

    Includes:
    - Dense X matrix
    - One sparse CSR layer ("counts")
    - One obsm embedding ("X_pca", shape N_CELLS × 10)
    - obs DataFrame with a "cell_type" column
    - var DataFrame with a "gene_name" column
    """
    rng = np.random.default_rng(42)

    # Dense X (float32)
    X = rng.random((N_CELLS, N_GENES), dtype=np.float32)

    # Sparse counts layer
    counts_dense = rng.integers(0, 10, size=(N_CELLS, N_GENES))
    counts_sparse = sp.csr_matrix(counts_dense.astype(np.float32))

    obs = pd.DataFrame(
        {"cell_type": [f"type_{i % 3}" for i in range(N_CELLS)]},
        index=[f"cell_{i}" for i in range(N_CELLS)],
    )
    var = pd.DataFrame(
        {"gene_name": [f"gene_{j}" for j in range(N_GENES)]},
        index=[f"gene_{j}" for j in range(N_GENES)],
    )
    obsm = {"X_pca": rng.random((N_CELLS, 10), dtype=np.float64)}

    adata = ad.AnnData(
        X=X,
        obs=obs,
        var=var,
        obsm=obsm,
        layers={"counts": counts_sparse},
    )

    out_dir = tmp_path_factory.mktemp("data")
    h5ad_path = str(out_dir / "synthetic.h5ad")
    adata.write_h5ad(h5ad_path)
    return h5ad_path


@pytest.fixture(scope="session")
def synthetic_zarr(tmp_path_factory: pytest.TempPathFactory) -> str:
    """
    Write the same synthetic data as a .zarr store and return its path.
    Skipped automatically if zarr is not installed.
    """
    zarr = pytest.importorskip("zarr", reason="zarr not installed; skipping zarr tests")
    _ = zarr  # importorskip returns the module; we only need the side-effect

    rng = np.random.default_rng(42)

    X = rng.random((N_CELLS, N_GENES), dtype=np.float32)
    counts_sparse = sp.csr_matrix(rng.integers(0, 10, size=(N_CELLS, N_GENES)).astype(np.float32))
    obs = pd.DataFrame(
        {"cell_type": [f"type_{i % 3}" for i in range(N_CELLS)]},
        index=[f"cell_{i}" for i in range(N_CELLS)],
    )
    var = pd.DataFrame(
        {"gene_name": [f"gene_{j}" for j in range(N_GENES)]},
        index=[f"gene_{j}" for j in range(N_GENES)],
    )
    obsm = {"X_pca": rng.random((N_CELLS, 10), dtype=np.float64)}

    adata = ad.AnnData(
        X=X,
        obs=obs,
        var=var,
        obsm=obsm,
        layers={"counts": counts_sparse},
    )

    out_dir = tmp_path_factory.mktemp("zarr_data")
    zarr_path = str(out_dir / "synthetic.zarr")
    adata.write_zarr(zarr_path)
    return zarr_path
