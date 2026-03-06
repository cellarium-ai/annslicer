"""
Pytest fixtures for annslicer benchmarks.

Generates a single, large synthetic .h5ad file that is shared across the
entire benchmark session.  Adjust N_CELLS_BENCH / N_GENES_BENCH to trade
benchmark realism against setup time.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

# Adjust these to control benchmark scale.
N_CELLS_BENCH = 100_000
N_GENES_BENCH = 5_000


@pytest.fixture(scope="session")
def large_h5ad(tmp_path_factory: pytest.TempPathFactory) -> str:
    """
    Write a large .h5ad file (N_CELLS_BENCH × N_GENES_BENCH) and return its path.

    Includes:
    - Dense float32 X matrix
    - One sparse CSR layer ("counts")
    - One obsm embedding ("X_pca", 10 dims)
    """
    rng = np.random.default_rng(0)

    X = sp.random(
        N_CELLS_BENCH,
        N_GENES_BENCH,
        density=0.01,
        format="csr",
        dtype=np.float32,
        random_state=rng,
    )
    counts_sparse = sp.random(
        N_CELLS_BENCH,
        N_GENES_BENCH,
        density=0.01,
        format="csr",
        dtype=np.float32,
        random_state=rng,
    )

    obs = pd.DataFrame(
        {"cell_type": [f"type_{i % 10}" for i in range(N_CELLS_BENCH)]},
        index=[f"cell_{i}" for i in range(N_CELLS_BENCH)],
    )
    var = pd.DataFrame(
        {"gene_name": [f"gene_{j}" for j in range(N_GENES_BENCH)]},
        index=[f"gene_{j}" for j in range(N_GENES_BENCH)],
    )
    obsm = {"X_pca": rng.random((N_CELLS_BENCH, 10), dtype=np.float64)}

    adata = ad.AnnData(
        X=X,
        obs=obs,
        var=var,
        obsm=obsm,
        layers={"counts": counts_sparse},
    )

    out_dir = tmp_path_factory.mktemp("bench_data")
    h5ad_path = str(out_dir / "large.h5ad")
    adata.write_h5ad(h5ad_path)
    return h5ad_path


@pytest.fixture(scope="session")
def bench_output_dir(tmp_path_factory: pytest.TempPathFactory):
    """
    Single shared output directory for all benchmark shard writes.

    All benchmarks write to the same filename prefix, overwriting files on
    each round, so total disk usage is bounded to one full set of shards at
    a time (~500 MB for 50 k × 2 k).  Using per-benchmark mktemp() dirs
    would accumulate N × that amount and exhaust the macOS temp partition.
    """
    out_dir = tmp_path_factory.getbasetemp() / "bench_shards"
    out_dir.mkdir(exist_ok=True)
    return out_dir
