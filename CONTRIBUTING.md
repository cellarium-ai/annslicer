# Contributing to annslicer

## Development setup

We recommend working inside a conda environment to keep dependencies isolated.

```bash
conda create -n annslicer python=3.10
conda activate annslicer
```

Clone the repo:

```bash
git clone https://github.com/cellarium-ai/annslicer.git
cd annslicer
```

And then install in editable mode with development dependencies. You can then either install using the Makefile command

```bash
make install
```

or instead, you can equivalently run

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

The `annslicer` command will now point to your local source, and changes take effect immediately without reinstalling.

## Running tests

```bash
pytest
```

Zarr-related tests (zarr output merging, zarr input slicing, zarr shuffle) are skipped automatically if `zarr` is not installed. Installing `[dev]` as above does install `zarr`.

## Linting, formatting, and type-checking

Before committing, run two commands from the root of the repo:

```bash
make lint
make typecheck
```

Or if you want to type things manually:

```bash
# Check for lint errors
ruff check src/ tests/

# Auto-fix where possible
ruff check --fix src/ tests/

# Check formatting
ruff format --check src/ tests/

# Apply formatting
ruff format src/ tests/

# Type-check
mypy src/annslicer
```

All three checks run automatically on every push and pull request via GitHub Actions.

## Running benchmarks

Benchmarks live in `benchmarks/` and are excluded from the normal `pytest` run so that CI stays fast. Run them locally with:

```bash
make benchmark
```

Or directly:

```bash
pytest benchmarks/ --benchmark-only -v
```

The benchmark suite (`benchmarks/bench_slice.py`) compares:

| Benchmark | What it measures |
|---|---|
| `bench_annslicer_slice[jobsN]` | Full out-of-core sharding pipeline (no shuffle) with `N` worker processes (same job counts as below) |
| `bench_annslicer_slice_shuffle[jobsN]` | Shuffled sharding (two-pass scatter / gather) with `N` worker processes (default runs: 1 and 4); `[jobsauto]` uses annslicer's default worker count, i.e. real-world default usage, and is the second run when you pass `INPUT=` (the resolved count is recorded as `workers`) |
| `bench_anndata_backed_iterate` | Baseline: backed AnnData row iteration |
| `bench_anndata_backed_shuffle` | Baseline: backed AnnData fancy-indexed shuffle |

as well as the same things for `.zarr` files, where the baseline is `read_zarr` (anndata has no backed mode for zarr, so it loads the whole store). All outputs are written gzip-compressed, for annslicer and the baselines alike.

By default the benchmarks generate a synthetic `.h5ad` with realistic sparsity (about 1,500 non-zeros per cell over 30,000 genes, `X` plus one layer). Adjust `N_CELLS_BENCH`, `N_GENES_BENCH` and `NNZ_PER_CELL` in `benchmarks/conftest.py` to scale it.

### Benchmarking your own file

Pass an `.h5ad` file with the `INPUT` make variable (a positional `make benchmark file.h5ad` would be read by make as another target):

```bash
make benchmark INPUT=/data/my_file.h5ad
```

or, equivalently, `pytest benchmarks/ --benchmark-only -v -s --bench-input /data/my_file.h5ad`. With your own input only the shuffled `.h5ad` benchmarks run (gzip output, as always): `bench_annslicer_slice_shuffle` at 1 job and at the default worker count, plus the anndata backed shuffle baseline. The unshuffled and zarr benchmarks are left out. Each runs once (time and peak memory from the same run), and the backed shuffle can take very long on large files. Select what to run with `-k`, for example `PYTEST_ARGS="-k annslicer"` to skip the baseline.

Extra options, passed through `PYTEST_ARGS` or directly to pytest (`pytest benchmarks/ --help` lists them under "annslicer benchmarks"):

| Option | Meaning |
|---|---|
| `--bench-input FILE.h5ad` | Benchmark this file instead of the synthetic dataset (what `INPUT=` sets) |
| `--bench-workdir DIR` | Directory for the shard outputs and the shuffle scratch files; use a large local disk |
| `--bench-jobs 1,2,4,8,auto` | Worker counts to benchmark for annslicer (default `1,4`, or `1,auto` with `--bench-input`; `auto` is annslicer's default worker count; `1` is always included) |
| `--bench-memory-limit 16GB` | Pass this `memory_limit` to annslicer and report it next to the measured peak memory |
| `--bench-shard-size N` | Cells per shard (default 5000 for synthetic data, 10000 for `--bench-input`) |
| `--bench-drop-caches` | Drop the Linux page cache before each benchmark (needs root), so earlier benchmarks don't warm the cache for later ones |

For example:

```bash
make benchmark INPUT=/data/my_file.h5ad \
    PYTEST_ARGS="--bench-workdir /mnt/scratch --bench-jobs 1,4,8,16,auto --bench-memory-limit 32GB --benchmark-json=results.json"
```

Memory is reported as `peak_memory_MiB` (tracemalloc, single-process synthetic runs) or `peak_tree_RSS_MiB` (the summed resident memory of the process and its workers, sampled during the run: an upper-bound estimate that includes each process's imported libraries). The dataset shape and output compression are recorded in each result's `extra_info`, so `--benchmark-json` files are self-describing.

## Releasing a new version and pushing to PyPI

Version is derived automatically from git tags — there is no version string to update in code.

1. Ensure all tests and lint checks pass on `main`.
2. Tag the release commit:
   ```bash
   git tag v0.2.0
   git push --tags
   ```
3. Create a new release on GitHub based on this new tag. The creation of the versioned release triggers the **Publish to PyPI** workflow automatically.

That's it. `setuptools-scm` picks the version from the tag, builds the sdist and wheel, and publishes to PyPI using OIDC Trusted Publishing (no API token required).

(Also possible: In GitHub Actions tab, manually trigger the **Publish to PyPI** workflow.)
