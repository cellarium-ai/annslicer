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

Peak memory is captured via tracemalloc's high-water mark
(tracemalloc.get_traced_memory()[1]) and reported in
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


def _run_with_memory(fn, *args, **kwargs) -> float:
    """Run *fn*, return peak Python heap allocation in MiB (tracemalloc high-water mark)."""
    tracemalloc.start()
    try:
        fn(*args, **kwargs)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak_bytes / (1024**2)


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

        adata[start:end].to_memory().write_h5ad(out_path)
        # the following will be allowable after https://github.com/scverse/anndata/issues/2077
        # adata[start:end].write_h5ad(out_path)

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

        adata[perm[start:end]].to_memory().write_h5ad(out_path)
        # the following will be allowable after https://github.com/scverse/anndata/issues/2077
        # adata[start:end].write_h5ad(out_path)

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
    benchmark.group = "sequential"

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
    benchmark.group = "sequential"

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
    benchmark.group = "shuffle"

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
    benchmark.group = "shuffle"

    peak_mb = _run_with_memory(_backed_shard_shuffle, large_h5ad, prefix, BENCH_SHARD_SIZE, 0)

    def _fn():
        _backed_shard_shuffle(large_h5ad, prefix, BENCH_SHARD_SIZE, 0)

    benchmark(_fn)
    benchmark.extra_info["peak_memory_MiB"] = round(peak_mb, 1)
    print(f"\n  [backed/shuffle]    peak RAM: {peak_mb:.1f} MiB")
