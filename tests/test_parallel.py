"""
Tests for annslicer._parallel: progress logging and ordered parallel execution.
"""

from __future__ import annotations

import logging
import re

import pytest
from joblib import delayed

from annslicer import _parallel
from annslicer._parallel import _fmt_duration, _Progress, run_parallel
from annslicer.slice import shard_h5ad


@pytest.mark.parametrize("shuffle", [False, True])
def test_shard_h5ad_logs_phases_and_summary(synthetic_sparse_h5ad, tmp_path, caplog, shuffle):
    with caplog.at_level(logging.INFO, logger="annslicer"):
        shard_h5ad(synthetic_sparse_h5ad, str(tmp_path / "o"), shard_size=50, shuffle=shuffle)
    text = caplog.text
    assert "Read metadata for 150 cells in" in text
    if shuffle:
        assert re.search(r"Pass 1/2: done, \d+ blocks in", text)
        assert "Pass 2/2: done, 3 shards in" in text
    else:
        assert "Writing shards: done, 3 shards in" in text
    assert "All 3 shards successfully created in" in text
    assert "GB written" in text


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.parametrize(
    "seconds, expected",
    [(8.24, "8.2s"), (59.9, "59.9s"), (60, "1m 00s"), (242, "4m 02s"), (3780, "1h 03m")],
)
def test_fmt_duration(seconds, expected):
    assert _fmt_duration(seconds) == expected


def test_progress_logs_at_most_once_per_interval(caplog):
    clock = _Clock()
    progress = _Progress("Pass 1/2", 10, "blocks", interval=30, clock=clock)
    with caplog.at_level(logging.INFO, logger=_parallel.logger.name):
        for _ in range(4):  # 4 blocks in 20 s: below the interval, so silent
            clock.now += 5
            progress.update()
        assert caplog.messages == []

        clock.now += 20  # 5th block at 40 s: logs, with ETA = 40 s / 5 * 5 remaining
        progress.update()
        assert caplog.messages == ["Pass 1/2: 5/10 blocks (50%), elapsed 40.0s, ETA 40.0s"]

        clock.now += 5  # throttled again
        progress.update()
        assert len(caplog.messages) == 1


def test_progress_finish_reports_total(caplog):
    clock = _Clock()
    progress = _Progress("Pass 2/2", 3, "shards", clock=clock)
    with caplog.at_level(logging.INFO, logger=_parallel.logger.name):
        for _ in range(3):
            clock.now += 100
            progress.update()
        progress.finish()
    assert caplog.messages[-1] == "Pass 2/2: done, 3 shards in 5m 00s"


def _double(x):
    return 2 * x


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_run_parallel_returns_results_in_task_order(n_jobs):
    progress = _Progress("t", 6, "tasks")
    results = run_parallel((delayed(_double)(i) for i in range(6)), n_jobs, progress)
    assert results == [0, 2, 4, 6, 8, 10]
    assert progress.done == 6


def test_run_parallel_weights_progress_by_result():
    progress = _Progress("t", 9, "shards")
    run_parallel((delayed(_double)(i) for i in (1, 2, 3)), 1, progress, weight=lambda r: r)
    assert progress.done == 2 + 4 + 6  # each result (2, 4, 6) counts as that many units
