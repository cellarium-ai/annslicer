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
| `bench_annslicer_slice` | Full out-of-core sharding pipeline (no shuffle) |
| `bench_annslicer_slice_shuffle` | Overhead of random shuffling via sort-read-reorder |
| `bench_anndata_backed_iterate` | Baseline: backed AnnData row iteration |

as well as the same things for `.zarr` files.

Adjust `N_CELLS_BENCH` and `N_GENES_BENCH` in `benchmarks/conftest.py` to scale the dataset up or down.

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
