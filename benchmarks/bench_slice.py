"""
Benchmarks comparing annslicer out-of-core sharding against plain AnnData.

Each annslicer benchmark is paired with an equivalent AnnData baseline that
performs the identical work (same shard size, same gzip-compressed output files
written to disk) so wall-time and peak-memory figures are directly comparable.
The baselines use only what anndata itself offers: ``read_h5ad(backed="r")`` for
h5ad, and ``read_zarr`` (which loads the whole store; anndata has no backed
mode for zarr) for zarr.

Pairs:
  bench_annslicer_slice                vs  bench_anndata_backed_iterate
  bench_annslicer_slice_shuffle        vs  bench_anndata_backed_shuffle
  bench_annslicer_zarr_slice           vs  bench_anndata_zarr_iterate
  bench_annslicer_zarr_slice_shuffle   vs  bench_anndata_zarr_shuffle

The annslicer benchmarks (shuffled or not) run once per ``--bench-jobs`` entry (default ``1,4``, or
``1,auto`` with ``--bench-input``).  The 1-job run is single-process, so its memory is
comparable with the baselines.  "auto" is annslicer's default worker count
(``n_jobs=None``), i.e. real-world default usage; the number of workers a run resolved to
is recorded in ``extra_info["workers"]``.

Run with:
    make benchmark                         # synthetic dataset
    make benchmark INPUT=my_file.h5ad      # your own file (see CONTRIBUTING.md)
    pytest benchmarks/ --benchmark-only -v

Memory: single-process runs report tracemalloc's high-water mark
(``extra_info["peak_memory_MB"]``).  tracemalloc only sees the current process
and only Python-level allocations, so runs with several worker processes, and all
runs on a ``--bench-input`` file, instead report ``extra_info["peak_tree_RSS_MB"]``:
the peak summed resident memory of the process and all its workers.  That is an
upper-bound estimate, since pages shared between processes are counted once per
process, and it includes the memory of the imported libraries (a few hundred MB
per process).  With ``--bench-input`` every benchmark runs once and time and memory
come from that same run.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import tracemalloc

import anndata as ad
import numpy as np

from annslicer.slice import shard_h5ad

# All outputs are gzip-compressed, which is the realistic use case and keeps the comparison fair.
BENCH_COMPRESSION = "gzip"
BENCH_ROUNDS = 2


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def _run_with_memory(fn, *args, **kwargs) -> float:
    """Run *fn*, return peak Python heap allocation in MB (tracemalloc high-water mark)."""
    tracemalloc.start()
    try:
        fn(*args, **kwargs)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak_bytes / 1e6


def _tree_rss_mb() -> float:
    """Summed resident memory (MB) of this process and all its descendants, via ``ps``."""
    rows = subprocess.run(
        ["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    children: dict[int, list[int]] = {}
    rss_kib: dict[int, int] = {}
    for row in rows:
        fields = row.split()
        if len(fields) == 3:
            pid, ppid, kib = map(int, fields)
            children.setdefault(ppid, []).append(pid)
            rss_kib[pid] = kib
    tree, todo = {os.getpid()}, [os.getpid()]
    while todo:
        for child in children.get(todo.pop(), []):
            if child not in tree:
                tree.add(child)
                todo.append(child)
    return sum(rss_kib[p] for p in tree) * 1024 / 1e6


def _run_with_tree_rss(fn, *args, **kwargs) -> float:
    """Run *fn*, return the peak of :func:`_tree_rss_mb`, sampled every 100 ms, in MB."""
    peak = _tree_rss_mb()
    done = threading.Event()

    def _sample():
        nonlocal peak
        while not done.wait(0.1):
            peak = max(peak, _tree_rss_mb())

    sampler = threading.Thread(target=_sample, daemon=True)
    sampler.start()
    try:
        fn(*args, **kwargs)
    finally:
        done.set()
        sampler.join()
    return peak


def _run_benchmark(
    benchmark, fn, label, bench_config, dataset_info, *, multiprocess=False, memory_limit=None
) -> None:
    """
    Time *fn* with pytest-benchmark and record its peak memory in ``extra_info``.

    Synthetic dataset: one untimed-by-the-table measurement run for memory (tracemalloc,
    or tree RSS for multi-process runs), then BENCH_ROUNDS timed rounds.  ``--bench-input``:
    a single run that is both timed and memory-sampled (tree RSS), since the input may be
    far too large to run several times.
    """
    benchmark.extra_info.update(dataset_info)
    benchmark.extra_info["shard_size"] = bench_config.shard_size
    benchmark.extra_info["output_compression"] = BENCH_COMPRESSION
    if memory_limit is not None:
        benchmark.extra_info["memory_limit"] = memory_limit

    if bench_config.custom_input:
        peak = _run_with_tree_rss(
            lambda: benchmark.pedantic(fn, rounds=1, iterations=1, warmup_rounds=0)
        )
        tree = True
    else:
        tree = multiprocess
        peak = _run_with_tree_rss(fn) if tree else _run_with_memory(fn)
        benchmark.pedantic(fn, rounds=BENCH_ROUNDS, iterations=1, warmup_rounds=0)

    benchmark.extra_info["peak_tree_RSS_MB" if tree else "peak_memory_MB"] = round(peak, 1)
    kind = "peak tree RSS (est.)" if tree else "peak RAM"
    suffix = f"  (memory limit {memory_limit})" if memory_limit else ""
    workers = benchmark.extra_info.get("workers")
    suffix += f"  [{workers} worker(s)]" if workers else ""
    print(f"\n  [{label}] {kind}: {peak:.1f} MB{suffix}")


class _WorkerCount(logging.Handler):
    """Captures how many workers annslicer resolved ``n_jobs`` to, from its log."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.count: int | None = None

    def emit(self, record: logging.LogRecord) -> None:
        match = re.search(r"(\d+) worker\(s\)", record.getMessage())
        if match:
            self.count = int(match.group(1))


def _jobs_label(n_jobs: int | None) -> str:
    return "auto" if n_jobs is None else str(n_jobs)


def _annslicer(
    input_file, prefix, bench_config, scratch_dir, *, shuffle=False, n_jobs=None, record=None
) -> None:
    """
    Run annslicer with the benchmark settings.  ``n_jobs=None`` is annslicer's default.

    If *record* (a dict) is given, the resolved number of workers is stored in it.
    """
    logger = logging.getLogger("annslicer")
    handler, level = _WorkerCount(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        shard_h5ad(
            input_file,
            prefix,
            shard_size=bench_config.shard_size,
            shuffle=shuffle,
            seed=0,
            compression=BENCH_COMPRESSION,
            n_jobs=n_jobs,
            memory_limit=bench_config.memory_limit,
            tmpdir=scratch_dir,
        )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
    if record is not None and handler.count is not None:
        record["workers"] = handler.count


# ---------------------------------------------------------------------------
# AnnData baselines
# ---------------------------------------------------------------------------


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
        shard_num = start // shard_size
        out_path = f"{output_prefix}_shard_{shard_num}.h5ad"

        adata[start:end].to_memory().write_h5ad(out_path, compression=BENCH_COMPRESSION)
        # the following will be allowable after https://github.com/scverse/anndata/issues/2077
        # adata[start:end].write_h5ad(out_path)

    adata.file.close()


def _backed_shard_shuffle(input_file: str, output_prefix: str, shard_size: int, seed: int) -> None:
    """
    Backed-AnnData equivalent of shard_h5ad(shuffle=True).

    Generates a global permutation and reads each shard's (random) cells from the
    backed file with fancy indexing, as an anndata user would.
    """
    adata = ad.read_h5ad(input_file, backed="r")
    total_cells = adata.n_obs

    perm = np.random.default_rng(seed).permutation(total_cells)

    for start in range(0, total_cells, shard_size):
        end = min(start + shard_size, total_cells)
        shard_num = start // shard_size
        out_path = f"{output_prefix}_shard_{shard_num}.h5ad"

        adata[perm[start:end]].to_memory().write_h5ad(out_path, compression=BENCH_COMPRESSION)
        # the following will be allowable after https://github.com/scverse/anndata/issues/2077
        # adata[start:end].write_h5ad(out_path)

    adata.file.close()


def _anndata_zarr_shard(input_file: str, output_prefix: str, shard_size: int) -> None:
    """
    AnnData equivalent of shard_h5ad for zarr input (no shuffle).

    anndata has no backed mode for zarr, so this reads the entire store into
    memory upfront — the inevitable cost any AnnData user would pay — then
    slices and writes shards just as the h5ad baseline does.
    """
    adata = ad.read_zarr(input_file)
    total_cells = adata.n_obs

    for start in range(0, total_cells, shard_size):
        end = min(start + shard_size, total_cells)
        shard_num = start // shard_size
        out_path = f"{output_prefix}_shard_{shard_num}.h5ad"
        adata[start:end].to_memory().write_h5ad(out_path, compression=BENCH_COMPRESSION)


def _anndata_zarr_shard_shuffle(
    input_file: str, output_prefix: str, shard_size: int, seed: int
) -> None:
    """
    AnnData equivalent of shard_h5ad(shuffle=True) for zarr input.

    Reads the entire zarr store into memory then applies a global permutation.
    Since data is already in RAM, no sort-read-reorder optimisation is needed.
    """
    adata = ad.read_zarr(input_file)
    total_cells = adata.n_obs
    perm = np.random.default_rng(seed).permutation(total_cells)

    for start in range(0, total_cells, shard_size):
        end = min(start + shard_size, total_cells)
        shard_num = start // shard_size
        out_path = f"{output_prefix}_shard_{shard_num}.h5ad"
        adata[perm[start:end]].to_memory().write_h5ad(out_path, compression=BENCH_COMPRESSION)


# ---------------------------------------------------------------------------
# Benchmark pair 1: sequential sharding
# ---------------------------------------------------------------------------


def bench_annslicer_slice(
    benchmark, large_h5ad, bench_output_dir, bench_scratch_dir, bench_config, dataset_info, n_jobs
):
    """
    annslicer — sequential sharding with ``n_jobs`` worker processes (None = the default).

    Reads each shard as one contiguous slice with no full matrix ever loaded into RAM.
    Writes one .h5ad file per shard to the shared bench output directory
    (files are overwritten on every round, keeping disk usage bounded).  The jobs1 run is
    single-process, so its memory is comparable with the single-process baseline.
    """
    prefix = str(bench_output_dir / "shard")
    benchmark.group = "h5ad-sequential"

    def _fn():
        _annslicer(
            large_h5ad,
            prefix,
            bench_config,
            bench_scratch_dir,
            n_jobs=n_jobs,
            record=benchmark.extra_info,
        )

    _run_benchmark(
        benchmark,
        _fn,
        f"annslicer/slice jobs={_jobs_label(n_jobs)}",
        bench_config,
        dataset_info,
        multiprocess=n_jobs != 1,
        memory_limit=bench_config.memory_limit,
    )


def bench_anndata_backed_iterate(
    benchmark, large_h5ad, bench_output_dir, bench_config, dataset_info
):
    """
    Baseline — backed AnnData sequential sharding.

    Reads each shard via backed AnnData row slicing and writes the same .h5ad
    output files as bench_annslicer_slice, making the comparison apples-to-apples.
    """
    prefix = str(bench_output_dir / "shard")
    benchmark.group = "h5ad-sequential"

    def _fn():
        _backed_shard(large_h5ad, prefix, bench_config.shard_size)

    _run_benchmark(benchmark, _fn, "backed/iterate", bench_config, dataset_info)


# ---------------------------------------------------------------------------
# Benchmark pair 2: shuffled sharding
# ---------------------------------------------------------------------------


def bench_annslicer_slice_shuffle(
    benchmark, large_h5ad, bench_output_dir, bench_scratch_dir, bench_config, dataset_info, n_jobs
):
    """
    annslicer — shuffled sharding with ``n_jobs`` worker processes (None = the default).

    Same pipeline as bench_annslicer_slice but with --shuffle enabled, using the
    two-pass scatter / gather.  The jobs1 run is single-process, so its memory
    is comparable with the single-process baseline.
    """
    prefix = str(bench_output_dir / "shard")
    benchmark.group = "h5ad-shuffle"

    def _fn():
        _annslicer(
            large_h5ad,
            prefix,
            bench_config,
            bench_scratch_dir,
            shuffle=True,
            n_jobs=n_jobs,
            record=benchmark.extra_info,
        )

    _run_benchmark(
        benchmark,
        _fn,
        f"annslicer/shuffle jobs={_jobs_label(n_jobs)}",
        bench_config,
        dataset_info,
        multiprocess=n_jobs != 1,
        memory_limit=bench_config.memory_limit,
    )


def bench_anndata_backed_shuffle(
    benchmark, large_h5ad, bench_output_dir, bench_config, dataset_info
):
    """
    Baseline — backed AnnData shuffled sharding.

    Applies the identical global permutation as bench_annslicer_slice_shuffle,
    implemented directly with backed AnnData.  Writes the same .h5ad output files
    for a fair comparison.
    """
    prefix = str(bench_output_dir / "shard")
    benchmark.group = "h5ad-shuffle"

    def _fn():
        _backed_shard_shuffle(large_h5ad, prefix, bench_config.shard_size, 0)

    _run_benchmark(benchmark, _fn, "backed/shuffle", bench_config, dataset_info)


# ---------------------------------------------------------------------------
# Benchmark pair 3: zarr sequential sharding (synthetic dataset only)
# ---------------------------------------------------------------------------


def bench_annslicer_zarr_slice(
    benchmark, large_zarr, bench_output_dir, bench_scratch_dir, bench_config, dataset_info, n_jobs
):
    """annslicer — sequential sharding from a zarr store with ``n_jobs`` workers (None = default)."""
    prefix = str(bench_output_dir / "shard")
    benchmark.group = "zarr-sequential"

    def _fn():
        _annslicer(
            large_zarr,
            prefix,
            bench_config,
            bench_scratch_dir,
            n_jobs=n_jobs,
            record=benchmark.extra_info,
        )

    _run_benchmark(
        benchmark,
        _fn,
        f"annslicer/zarr jobs={_jobs_label(n_jobs)}",
        bench_config,
        dataset_info,
        multiprocess=n_jobs != 1,
        memory_limit=bench_config.memory_limit,
    )


def bench_anndata_zarr_iterate(
    benchmark, large_zarr, bench_output_dir, bench_config, dataset_info
):
    """Baseline — read_zarr + AnnData sequential sharding."""
    prefix = str(bench_output_dir / "shard")
    benchmark.group = "zarr-sequential"

    def _fn():
        _anndata_zarr_shard(large_zarr, prefix, bench_config.shard_size)

    _run_benchmark(benchmark, _fn, "anndata/zarr", bench_config, dataset_info)


# ---------------------------------------------------------------------------
# Benchmark pair 4: zarr shuffled sharding (synthetic dataset only)
# ---------------------------------------------------------------------------


def bench_annslicer_zarr_slice_shuffle(
    benchmark, large_zarr, bench_output_dir, bench_scratch_dir, bench_config, dataset_info, n_jobs
):
    """annslicer — shuffled sharding from a zarr store with ``n_jobs`` workers (None = default)."""
    prefix = str(bench_output_dir / "shard")
    benchmark.group = "zarr-shuffle"

    def _fn():
        _annslicer(
            large_zarr,
            prefix,
            bench_config,
            bench_scratch_dir,
            shuffle=True,
            n_jobs=n_jobs,
            record=benchmark.extra_info,
        )

    _run_benchmark(
        benchmark,
        _fn,
        f"annslicer/zarr-s jobs={_jobs_label(n_jobs)}",
        bench_config,
        dataset_info,
        multiprocess=n_jobs != 1,
        memory_limit=bench_config.memory_limit,
    )


def bench_anndata_zarr_shuffle(
    benchmark, large_zarr, bench_output_dir, bench_config, dataset_info
):
    """Baseline — read_zarr + AnnData shuffled sharding."""
    prefix = str(bench_output_dir / "shard")
    benchmark.group = "zarr-shuffle"

    def _fn():
        _anndata_zarr_shard_shuffle(large_zarr, prefix, bench_config.shard_size, 0)

    _run_benchmark(benchmark, _fn, "anndata/zarr-s", bench_config, dataset_info)
