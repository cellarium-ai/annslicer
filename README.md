# annslicer

**Out-of-core sharding of large `.h5ad` AnnData files with minimal memory usage.**

Large single-cell datasets stored as `.h5ad` files can easily exceed available RAM. `annslicer` slices them into manageable shards without loading the full matrix into memory. It uses `h5py` directly to read only the rows it needs from disk, and lets `anndata` handle metadata.

## Features

- Shards `X`, all `layers`, `obs`, `var`, `obsm`, and `uns`
- Handles both dense and sparse (CSR) matrices
- Constant, low memory footprint regardless of file size
- Simple CLI and Python API

## Installation

```bash
pip install annslicer
```

## CLI Usage

```bash
annslice input.h5ad output_prefix --size 10000
```

| Argument | Description |
|---|---|
| `input.h5ad` | Path to the source `.h5ad` file |
| `output_prefix` | Prefix for output files (e.g. `my_dataset` → `my_dataset_shard001.h5ad`, …) |
| `--size N` | Number of cells per shard (default: `10000`) |
| `--debug` | Enable verbose debug logging |

**Example:**

```bash
annslice /data/large_atlas.h5ad /outputs/atlas --size 20000
```

Produces: `atlas_shard001.h5ad`, `atlas_shard002.h5ad`, …

## Python API

```python
from annslicer import shard_h5ad

shard_h5ad(
    input_file="large_atlas.h5ad",
    output_prefix="atlas",
    shard_size=20000,
)
```

## How it works

1. Opens the `.h5ad` file twice: once with `anndata` in backed mode (metadata only), once with `h5py` directly (matrix data).
2. For each shard, reads only the relevant rows from `X` and each layer using `h5py` slice indexing — no full matrix is ever loaded.
3. Reassembles a valid `AnnData` object per shard and writes it to disk.

> **Note:** CSC (column-compressed) sparse matrices are not supported for out-of-core row-slicing. Convert to CSR before sharding.

## License

MIT
