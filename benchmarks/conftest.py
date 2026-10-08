"""
Pytest fixtures and options for annslicer benchmarks.

By default a synthetic .h5ad file with realistic sparsity (about 1,500 non-zeros per cell
over 30,000 genes) is generated once per session.  Adjust N_CELLS_BENCH / N_GENES_BENCH /
NNZ_PER_CELL to trade benchmark realism against setup and run time.

To benchmark your own file instead, pass ``--bench-input`` (or ``make benchmark INPUT=...``);
see ``pytest benchmarks/ --help`` (the "annslicer benchmarks" group) for the other options.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

# Adjust these to control the scale of the synthetic dataset.
N_CELLS_BENCH = 40_000
N_GENES_BENCH = 30_000
NNZ_PER_CELL = 1_500

DEFAULT_SHARD_SIZE = 5_000
# Worker counts benchmarked for annslicer.  The synthetic suite uses an explicit 4 workers;
# with --bench-input the second run is annslicer's default worker count ("auto", n_jobs=None),
# so a run on your own data reflects real-world default usage.
DEFAULT_JOBS = "1,4"
DEFAULT_JOBS_CUSTOM_INPUT = "1,auto"


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("annslicer benchmarks")
    group.addoption(
        "--bench-input",
        default=None,
        metavar="FILE.h5ad",
        help="Benchmark this .h5ad file instead of generating a synthetic dataset. Only the "
        "shuffled h5ad benchmarks run (gzip output), each once.",
    )
    group.addoption(
        "--bench-workdir",
        default=None,
        metavar="DIR",
        help="Directory for the shard outputs and the shuffle scratch files (default: pytest's "
        "temp directory). Use a large local disk for large inputs.",
    )
    group.addoption(
        "--bench-jobs",
        default=None,
        metavar="N,N,...",
        help="Worker counts to benchmark for annslicer, e.g. 1,2,4,8,auto. "
        "'auto' is annslicer's default worker count (n_jobs=None). 1 is always included, since "
        "it is the run whose memory is comparable with the single-process baselines. "
        f"Default: {DEFAULT_JOBS} for the synthetic dataset, {DEFAULT_JOBS_CUSTOM_INPUT} for "
        "--bench-input.",
    )
    group.addoption(
        "--bench-memory-limit",
        default=None,
        metavar="SIZE",
        help="Pass this memory_limit (e.g. 16GB) to annslicer's shuffled sharding and report it "
        "next to the measured peak memory.",
    )
    group.addoption(
        "--bench-shard-size",
        type=int,
        default=None,
        metavar="N",
        help=f"Cells per shard (default: {DEFAULT_SHARD_SIZE} for the synthetic dataset, "
        "10000 for --bench-input).",
    )
    group.addoption(
        "--bench-drop-caches",
        action="store_true",
        default=False,
        help="Drop the Linux page cache before each benchmark so that earlier benchmarks do "
        "not warm the cache for later ones (needs root; ignored elsewhere).",
    )


@dataclass(frozen=True)
class BenchConfig:
    input_file: str | None
    workdir: Path | None
    jobs: list[int | None]
    memory_limit: str | None
    shard_size: int

    @property
    def custom_input(self) -> bool:
        return self.input_file is not None


def _parse_jobs(text: str) -> list[int | None]:
    """Parse ``1,4,auto`` into ``[1, 4, None]``: 1 always first, ``auto`` (the default) last."""
    counts: set[int] = set()
    auto = False
    for token in (t.strip().lower() for t in text.split(",") if t.strip()):
        if token == "auto":
            auto = True
        elif token.isdigit() and int(token) >= 1:
            counts.add(int(token))
        else:
            raise pytest.UsageError(
                f"--bench-jobs expects positive integers and/or 'auto', like 1,4,auto; got {text!r}"
            )
    jobs: list[int | None] = sorted(counts | {1})
    return jobs + [None] if auto else jobs


def _jobs_option(config: pytest.Config) -> list[int | None]:
    """The worker counts to benchmark: ``--bench-jobs``, or the default for the input kind."""
    default = DEFAULT_JOBS_CUSTOM_INPUT if config.getoption("--bench-input") else DEFAULT_JOBS
    return _parse_jobs(config.getoption("--bench-jobs") or default)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Run the annslicer benchmarks once per requested worker count (``auto`` = default)."""
    if "n_jobs" in metafunc.fixturenames:
        jobs = _jobs_option(metafunc.config)
        ids = ["jobsauto" if j is None else f"jobs{j}" for j in jobs]
        metafunc.parametrize("n_jobs", jobs, ids=ids)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """With ``--bench-input`` run only the shuffled .h5ad benchmarks (no unshuffled, no zarr)."""
    if not config.getoption("--bench-input"):
        return
    keep = [i for i in items if "shuffle" in i.originalname and "zarr" not in i.originalname]
    config.hook.pytest_deselected(items=[i for i in items if i not in keep])
    items[:] = keep


@pytest.fixture(scope="session")
def bench_config(request: pytest.FixtureRequest) -> BenchConfig:
    opt = request.config.getoption
    input_file = opt("--bench-input")
    if input_file is not None:
        if not input_file.endswith(".h5ad"):
            raise pytest.UsageError(f"--bench-input must be an .h5ad file, got {input_file!r}")
        if not os.path.isfile(input_file):
            raise pytest.UsageError(f"--bench-input file not found: {input_file}")
    shard_size = opt("--bench-shard-size") or (10_000 if input_file else DEFAULT_SHARD_SIZE)
    workdir = opt("--bench-workdir")
    return BenchConfig(
        input_file=input_file,
        workdir=Path(workdir) if workdir else None,
        jobs=_jobs_option(request.config),
        memory_limit=opt("--bench-memory-limit"),
        shard_size=shard_size,
    )


@pytest.fixture(autouse=True)
def _drop_page_cache(request: pytest.FixtureRequest) -> None:
    """With --bench-drop-caches, start every benchmark with a cold Linux page cache."""
    if not request.config.getoption("--bench-drop-caches"):
        return
    try:
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w") as fh:
            fh.write("3")
    except OSError as exc:
        warnings.warn(f"--bench-drop-caches could not drop the page cache: {exc}", stacklevel=1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _random_csr(
    rng: np.random.Generator, n_cells: int, n_genes: int, nnz_per_cell: int, chunk: int = 20_000
) -> sp.csr_matrix:
    """
    A CSR matrix with exactly *nnz_per_cell* distinct genes per cell and count-like values.

    Built directly (sorted columns from random gaps) because ``scipy.sparse.random`` samples
    without replacement over the whole matrix, which is slow and memory-hungry at this size.
    """
    max_gap = int(1.6 * n_genes / nnz_per_cell)  # mean column span ~0.8 * n_genes
    data, indices = [], []
    for start in range(0, n_cells, chunk):
        n = min(chunk, n_cells - start)
        gaps = rng.integers(1, max_gap + 1, size=(n, nnz_per_cell), dtype=np.int32)
        columns = np.cumsum(gaps, axis=1, dtype=np.int32) - 1
        assert columns.max() < n_genes, "increase n_genes or decrease nnz_per_cell"
        indices.append(columns.ravel())
        data.append(rng.geometric(0.4, size=n * nnz_per_cell).astype(np.float32))
    indptr = np.arange(0, n_cells * nnz_per_cell + 1, nnz_per_cell, dtype=np.int64)
    return sp.csr_matrix(
        (np.concatenate(data), np.concatenate(indices), indptr), shape=(n_cells, n_genes)
    )


@pytest.fixture(scope="session")
def large_h5ad(bench_config: BenchConfig, tmp_path_factory: pytest.TempPathFactory) -> str:
    """
    The .h5ad benchmark input: ``--bench-input`` if given, otherwise a synthetic file.

    The synthetic file (N_CELLS_BENCH × N_GENES_BENCH, NNZ_PER_CELL non-zeros per cell) has:
    - A sparse CSR X matrix
    - One sparse CSR layer ("counts")
    - One obsm embedding ("X_pca", 10 dims)
    """
    if bench_config.input_file is not None:
        return bench_config.input_file

    rng = np.random.default_rng(0)
    adata = ad.AnnData(
        X=_random_csr(rng, N_CELLS_BENCH, N_GENES_BENCH, NNZ_PER_CELL),
        obs=pd.DataFrame(
            {"cell_type": [f"type_{i % 10}" for i in range(N_CELLS_BENCH)]},
            index=[f"cell_{i}" for i in range(N_CELLS_BENCH)],
        ),
        var=pd.DataFrame(
            {"gene_name": [f"gene_{j}" for j in range(N_GENES_BENCH)]},
            index=[f"gene_{j}" for j in range(N_GENES_BENCH)],
        ),
        obsm={"X_pca": rng.random((N_CELLS_BENCH, 10), dtype=np.float64)},
        layers={"counts": _random_csr(rng, N_CELLS_BENCH, N_GENES_BENCH, NNZ_PER_CELL)},
    )
    out_dir = tmp_path_factory.mktemp("bench_data")
    h5ad_path = str(out_dir / "large.h5ad")
    adata.write_h5ad(h5ad_path)
    return h5ad_path


@pytest.fixture(scope="session")
def large_zarr(large_h5ad: str, tmp_path_factory: pytest.TempPathFactory) -> str:
    """
    Write the synthetic benchmark dataset as a .zarr store and return its path.

    Re-uses the already-generated large_h5ad data so the two fixtures are
    guaranteed to have identical contents, keeping h5ad vs zarr comparisons
    apples-to-apples.  Skipped if zarr is not installed.
    """
    pytest.importorskip("zarr", reason="zarr not installed; skipping zarr benchmarks")
    adata = ad.read_h5ad(large_h5ad)
    out_dir = tmp_path_factory.mktemp("bench_zarr_data")
    zarr_path = str(out_dir / "large.zarr")
    adata.write_zarr(zarr_path)
    return zarr_path


@pytest.fixture(scope="session")
def dataset_info(large_h5ad: str) -> dict:
    """Facts about the benchmark input, recorded in every result's ``extra_info``."""
    with h5py.File(large_h5ad, "r") as f:
        x = f["X"]
        if isinstance(x, h5py.Group):  # sparse
            n_obs, n_vars = (int(v) for v in x.attrs["shape"])
            nnz_per_cell = round(int(x["indptr"][-1]) / max(n_obs, 1))
            compression = x["data"].compression
        else:
            n_obs, n_vars = x.shape
            nnz_per_cell = None
            compression = x.compression
        n_layers = len(f["layers"]) if "layers" in f else 0
    return {
        "n_obs": n_obs,
        "n_vars": n_vars,
        "nnz_per_cell": nnz_per_cell,
        "n_layers": n_layers,
        "input_MB": round(os.path.getsize(large_h5ad) / 1e6, 1),
        "input_compression": compression,
    }


# ---------------------------------------------------------------------------
# Working directories
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def bench_output_dir(bench_config: BenchConfig, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Single shared output directory for all benchmark shard writes.

    All benchmarks write to the same filename prefix, overwriting files on
    each round, so total disk usage is bounded to one full set of shards at
    a time.  Using per-benchmark mktemp() dirs would accumulate N × that amount
    and exhaust the macOS temp partition.  With --bench-workdir it is ``<workdir>/out``.
    """
    if bench_config.workdir is not None:
        out_dir = bench_config.workdir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    out_dir = tmp_path_factory.getbasetemp() / "bench_shards"
    out_dir.mkdir(exist_ok=True)
    return out_dir


@pytest.fixture(scope="session")
def bench_scratch_dir(bench_config: BenchConfig) -> str | None:
    """Scratch directory for shuffled sharding: ``<workdir>/scratch``, or the system default."""
    if bench_config.workdir is None:
        return None
    scratch = bench_config.workdir / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    return str(scratch)
