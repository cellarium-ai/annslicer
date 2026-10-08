"""
Running tasks on worker processes with throttled progress logging.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from typing import Any

from joblib import Parallel

logger = logging.getLogger(__name__)

_LOG_INTERVAL = 30.0  # seconds between progress lines


def _fmt_duration(seconds: float) -> str:
    """Format a duration like ``8.2s``, ``4m 02s`` or ``1h 03m``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds + 0.5), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


class _Progress:
    """Logs ``label: done/total unit (pct), elapsed, ETA`` at most once per *interval* seconds."""

    def __init__(
        self,
        label: str,
        total: int,
        unit: str,
        interval: float = _LOG_INTERVAL,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.label, self.total, self.unit = label, total, unit
        self.interval, self.clock = interval, clock
        self.done = 0
        self.start = self.last_log = clock()

    def update(self, n: int = 1) -> None:
        self.done += n
        now = self.clock()
        if self.done < self.total and now - self.last_log >= self.interval:
            self.last_log = now
            elapsed = now - self.start
            eta = elapsed / self.done * (self.total - self.done)
            logger.info(
                "%s: %d/%d %s (%d%%), elapsed %s, ETA %s",
                self.label,
                self.done,
                self.total,
                self.unit,
                100 * self.done // self.total,
                _fmt_duration(elapsed),
                _fmt_duration(eta),
            )

    def finish(self) -> None:
        logger.info(
            "%s: done, %d %s in %s",
            self.label,
            self.done,
            self.unit,
            _fmt_duration(self.clock() - self.start),
        )


def run_parallel(
    tasks: Iterable[Any],
    n_jobs: int,
    progress: _Progress,
    weight: Callable[[Any], int] = lambda _: 1,
) -> list[Any]:
    """
    Run joblib ``delayed`` *tasks* on *n_jobs* workers and return their results in task order.

    Each finished task advances *progress* by ``weight(result)``.
    """
    results = []
    # max_nbytes=None: pass arrays by value; joblib would otherwise hand large ones to the
    # workers as read-only np.memmap objects, which anndata cannot write.
    for result in Parallel(n_jobs=n_jobs, max_nbytes=None, return_as="generator")(tasks):
        results.append(result)
        progress.update(weight(result))
    progress.finish()
    return results
