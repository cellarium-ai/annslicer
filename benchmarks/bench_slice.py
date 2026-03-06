"""
Benchmarks comparing annslicer out-of-core sharding against backed AnnData.

Each annslicer benchmark is paired with an equivalent backed-AnnData baseline
that performs the identical work (same shard size, same output files written to
disk) so wall-time and peak-memory figures are directly comparable.

Pairs:
  bench_annslicer_slice          vs  bench_anndata_backed_iterate
  bench_annslicer_slice_shuffle  vs  bench_anndata_backed_shuffle

Run with:
    pytest benchmarks/ --benchmark-only -v

Peak RSS memory is captured via tracemalloc and reported in
benchmark.extra_info["peak_memory_MiB"] as well as printed to stdout.
"""

from __future__ import annotations

import tracemalloc

import anndata as ad
import numpy as np

from annslicer.slice import shard_h5ad

BENCH_SHARD_SIZE = 10_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _peak_mb(snapshot: tracemalloc.Snapshot) -> float:
    """Return total bytes allocated (MiB) from a tracemalloc snapshot."""
    return sum(s.size for s in snapshot.statistics("lineno")) / (1024**2)


def _unwrap(arr):
    """
    Strip the 0-d numpy object-array wrapper that some anndata/h5py versions
    return when fancy-indexing a backed sparse layer, without densifying.

    Backed sparse layers can come back as ``array(<CSR matrix>, dtype=object)``
    instead of the matrix itself.  Calling ``.item()`` recovers the actual
    sparse (or dense) matrix so that downstream indexing works correctly and
    sparsity is preserved — keeping the baseline fair against annslicer, which
    also writes sparse layers to the shard files.
    """
    if isinstance(arr, np.ndarray) and arr.ndim == 0:
        return arr.item()  # 0-d object array → contained sparse / dense matrix
    return arr


def _run_with_memory(fn, *args, **kwargs) -> float:
    """Run *fn*, return peak Python heap allocation in MiB."""
    tracemalloc.start()
    try:
        fn(*args, **kwargs)
        snapshot = tracemalloc.take_snapshot()
    finally:
        tracemalloc.stop()
    return _peak_mb(snapshot)


def _backed_shard(input_file: str, output_prefix: str, shard_size: int) -> None:
    """
    Backed-AnnData equivalent of shard_h5ad (no shuffle).

    Opens the file with backed=True and reads each shard's rows via h5py
    slice indexing, then writes them as individual .h5ad files — the same
    end-result as annslicer but using the backed AnnData API directly.
    """
    adata = ad.read_h5ad(input_file, backed="r")
    total_cells = adata.n_obs

    for start in range(0, total_cells, shard_size):
        end = min(start + shard_size, total_cells)
        shard_num = (start // shard_size) + 1
        out_path = f"{output_prefix}_shard{shard_num:03d}.h5ad"

        X_shard = _unwrap(adata.X[start:end, :])

        layers_shard = {
            name: _unwrap(adata.layers[name][start:end, :]) for name in adata.layers.keys()
        }
        obsm_shard = {k: np.asarray(adata.obsm[k][start:end]) for k in adata.obsm.keys()}

        ad.AnnData(
            X=X_shard,
            obs=adata.obs.iloc[start:end].copy(),
            var=adata.var.copy(),
            obsm=obsm_shard,
            layers=layers_shard,
            uns=adata.uns.copy(),
        ).write_h5ad(out_path)

    adata.file.close()


def _backed_shard_shuffle(input_file: str, output_prefix: str, shard_size: int, seed: int) -> None:
    """
    Backed-AnnData equivalent of shard_h5ad(shuffle=True).

    Uses the same sort-read-reorder strategy as annslicer: generate a global
    permutation, sort each shard's indices for sequential I/O, read, then
    reorder in-memory to the desired shuffled order before writing.
    """
    adata = ad.read_h5ad(input_file, backed="r")
    total_cells = adata.n_obs

    perm = np.random.default_rng(seed).permutation(total_cells)

    for start in range(0, total_cells, shard_size):
        end = min(start + shard_size, total_cells)
        shard_num = (start // shard_size) + 1
        out_path = f"{output_prefix}_shard{shard_num:03d}.h5ad"

        orig_indices = perm[start:end]
        sorted_idx = np.sort(orig_indices)
        restore_order = np.argsort(np.argsort(orig_indices))

        X_shard = _unwrap(adata.X[sorted_idx, :])
        X_shard = X_shard[restore_order]

        layers_shard = {}
        for name in adata.layers.keys():
            arr = _unwrap(adata.layers[name][sorted_idx, :])
            layers_shard[name] = arr[restore_order]  # sparse fancy-row indexing works on CSR

        obsm_shard = {}
        for k in adata.obsm.keys():
            arr = np.asarray(adata.obsm[k][sorted_idx])
            obsm_shard[k] = arr[restore_order]

        ad.AnnData(
            X=X_shard,
            obs=adata.obs.iloc[orig_indices].copy(),
            var=adata.var.copy(),
            obsm=obsm_shard,
            layers=layers_shard,
            uns=adata.uns.copy(),
        ).write_h5ad(out_path)

    adata.file.close()


# ---------------------------------------------------------------------------
# Benchmark pair 1: sequential sharding
# ---------------------------------------------------------------------------


def bench_annslicer_slice(benchmark, large_h5ad, bench_output_dir):
    """
    annslicer — sequential sharding.

    Uses h5py direct slice indexing with no full matrix ever loaded into RAM.
    Writes one .h5ad file per shard to the shared bench output directory
    (files are overwritten on every round, keeping disk usage bounded).
    """
    prefix = str(bench_output_dir / "shard")

    peak_mb = _run_with_memory(shard_h5ad, large_h5ad, prefix, BENCH_SHARD_SIZE)

    def _fn():
        shard_h5ad(large_h5ad, prefix, BENCH_SHARD_SIZE)

    benchmark(_fn)
    benchmark.extra_info["peak_memory_MiB"] = round(peak_mb, 1)
    print(f"\n  [annslicer/slice]   peak RAM: {peak_mb:.1f} MiB")


def bench_anndata_backed_iterate(benchmark, large_h5ad, bench_output_dir):
    """
    Baseline — backed AnnData sequential sharding.

    Reads each shard via backed AnnData row slicing and writes the same .h5ad
    output files as bench_annslicer_slice, making the comparison apples-to-apples.
    """
    prefix = str(bench_output_dir / "shard")

    peak_mb = _run_with_memory(_backed_shard, large_h5ad, prefix, BENCH_SHARD_SIZE)

    def _fn():
        _backed_shard(large_h5ad, prefix, BENCH_SHARD_SIZE)

    benchmark(_fn)
    benchmark.extra_info["peak_memory_MiB"] = round(peak_mb, 1)
    print(f"\n  [backed/iterate]    peak RAM: {peak_mb:.1f} MiB")


# ---------------------------------------------------------------------------
# Benchmark pair 2: shuffled sharding
# ---------------------------------------------------------------------------


def bench_annslicer_slice_shuffle(benchmark, large_h5ad, bench_output_dir):
    """
    annslicer — shuffled sharding.

    Same pipeline as bench_annslicer_slice but with --shuffle enabled.
    Uses sort-read-reorder: read indices in sorted order (sequential I/O),
    then permute in-memory to the desired random order before writing.
    """
    prefix = str(bench_output_dir / "shard")

    peak_mb = _run_with_memory(
        shard_h5ad, large_h5ad, prefix, BENCH_SHARD_SIZE, shuffle=True, seed=0
    )

    def _fn():
        shard_h5ad(large_h5ad, prefix, BENCH_SHARD_SIZE, shuffle=True, seed=0)

    benchmark(_fn)
    benchmark.extra_info["peak_memory_MiB"] = round(peak_mb, 1)
    print(f"\n  [annslicer/shuffle] peak RAM: {peak_mb:.1f} MiB")


def bench_anndata_backed_shuffle(benchmark, large_h5ad, bench_output_dir):
    """
    Baseline — backed AnnData shuffled sharding.

    Applies the identical global permutation and sort-read-reorder logic as
    bench_annslicer_slice_shuffle, implemented directly with backed AnnData.
    Writes the same .h5ad output files for a fair comparison.
    """
    prefix = str(bench_output_dir / "shard")

    peak_mb = _run_with_memory(_backed_shard_shuffle, large_h5ad, prefix, BENCH_SHARD_SIZE, 0)

    def _fn():
        _backed_shard_shuffle(large_h5ad, prefix, BENCH_SHARD_SIZE, 0)

    benchmark(_fn)
    benchmark.extra_info["peak_memory_MiB"] = round(peak_mb, 1)
    print(f"\n  [backed/shuffle]    peak RAM: {peak_mb:.1f} MiB")
