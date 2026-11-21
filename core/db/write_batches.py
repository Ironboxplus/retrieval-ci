"""Cut a long SQLite write into transactions that give the write lock back in between.

SQLite has one write lock per file. A repository that wrote a whole scope (tens of thousands of rows)
in one transaction held it for tens of seconds (journal bulk create: ~27 s for 79,264 chunks; catalog
report: ~25 s for 100k new rows) and every other writer died of ``database is locked`` once
``busy_timeout`` (30 s) ran out. ``paced_batches`` is the one place both repositories cut their writes.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

# SQLite's lock is not fair: a waiter polls with sleeps of up to 100 ms, and a writer that opens its
# next transaction right after its commit wins every time, so batches alone starve the others. The
# pause is longer than the longest poll sleep, so every waiter sees the lock free at least once.
YIELD_S = 0.15


def batches_by_weight(
    items: Sequence[Any], limit: int, weight: Callable[[Any], int] = lambda item: 1,
) -> list[list[Any]]:
    """``items`` in order, cut into runs of about ``limit`` total weight (always at least one batch,
    so a caller that must write something, even for an empty input, runs once)."""
    batches: list[list[Any]] = [[]]
    total = 0
    for item in items:
        if batches[-1] and total >= limit:
            batches.append([])
            total = 0
        batches[-1].append(item)
        total += weight(item)
    return batches


def paced_batches(
    items: Sequence[Any], limit: int, weight: Callable[[Any], int] = lambda item: 1,
) -> Iterator[list[Any]]:
    """The batches of ``batches_by_weight``, pausing ``YIELD_S`` before every batch but the first.
    The caller opens and commits one transaction per batch."""
    for number, batch in enumerate(batches_by_weight(items, limit, weight)):
        if number:
            time.sleep(YIELD_S)
        yield batch
