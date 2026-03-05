# Contributing to annslicer

## Development setup

We recommend working inside a conda environment to keep dependencies isolated.

```bash
conda create -n annslicer-dev python=3.11
conda activate annslicer-dev
```

Clone the repo and install in editable mode with development dependencies:

```bash
git clone https://github.com/sfleming/annslicer.git
cd annslicer
pip install -e ".[dev]"
```

The `annslice` command will now point to your local source, and changes take effect immediately without reinstalling.

## Running tests

```bash
pytest
```

## Linting and formatting

```bash
# Check for lint errors
ruff check src/ tests/

# Auto-fix where possible
ruff check --fix src/ tests/

# Check formatting
ruff format --check src/ tests/

# Apply formatting
ruff format src/ tests/
```

Both checks run automatically on every push and pull request via GitHub Actions.

## Releasing a new version

Version is derived automatically from git tags — there is no version string to update in code.

1. Ensure all tests and lint checks pass on `main`.
2. Tag the release commit:
   ```bash
   git tag v0.2.0
   git push --tags
   ```
3. In the GitHub Actions tab, manually trigger the **Publish to PyPI** workflow.

That's it. `setuptools-scm` picks the version from the tag, builds the sdist and wheel, and publishes to PyPI using OIDC Trusted Publishing (no API token required).

## Project layout

```
src/annslicer/
    __init__.py   — public API surface and __version__
    core.py       — all logic: extract_matrix_slice, shard_h5ad, main (CLI)
tests/
    conftest.py   — synthetic .h5ad fixture
    test_core.py  — unit and integration tests
```
