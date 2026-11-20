"""Stale-while-revalidate cache: the one shared cache behind the WebUI statistics endpoints.

Real threads, real SQLite for the last test; the only injected seam is the clock, so the
TTL boundary is exact instead of sleep-based.
"""
from __future__ import annotations

import threading
import time

import pytest
from sqlalchemy import text

from core.cache import StaleWhileRevalidateCache


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _join_refresh_threads(timeout: float = 5.0) -> None:
    for thread in threading.enumerate():
        if thread.name.startswith("swr-test-refresh"):
            thread.join(timeout)


def test_fresh_value_is_served_without_recompute():
    clock = Clock()
    cache = StaleWhileRevalidateCache("swr-test", ttl_seconds=60, clock=clock)
    calls = []

    def compute():
        calls.append(1)
        return {"n": len(calls)}

    first = cache.get("k", compute)
    clock.now += 59
    assert cache.get("k", compute) is first
    assert len(calls) == 1


def test_stale_value_is_returned_at_once_and_refreshed_exactly_once():
    clock = Clock()
    cache = StaleWhileRevalidateCache("swr-test", ttl_seconds=60, clock=clock)
    release = threading.Event()
    started = threading.Event()
    calls = []

    def compute():
        calls.append(1)
        if len(calls) == 1:
            return "old"
        started.set()
        assert release.wait(5)
        return "new"

    assert cache.get("k", compute) == "old"
    clock.now += 61

    results = []
    threads = [threading.Thread(target=lambda: results.append(cache.get("k", compute))) for _ in range(8)]
    began = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join(2)
    # Every caller got the stale value without waiting for the blocked refresh.
    assert results == ["old"] * 8
    assert time.monotonic() - began < 2
    assert started.wait(2)
    assert len(calls) == 2, "eight stale callers must trigger a single refresh"

    release.set()
    _join_refresh_threads()
    assert cache.get("k", compute) == "new"
    assert len(calls) == 2


def test_refresh_runs_on_a_daemon_thread():
    clock = Clock()
    cache = StaleWhileRevalidateCache("swr-test", ttl_seconds=1, clock=clock)
    seen = []
    cache.get("k", lambda: seen.append(threading.current_thread()) or 1)
    clock.now += 2
    cache.get("k", lambda: seen.append(threading.current_thread()) or 2)
    _join_refresh_threads()
    assert seen[0] is threading.current_thread()
    assert seen[1] is not threading.current_thread()
    assert seen[1].daemon


def test_concurrent_cold_callers_wait_for_one_computation():
    cache = StaleWhileRevalidateCache("swr-test", ttl_seconds=60)
    calls = []
    gate = threading.Event()

    def compute():
        calls.append(1)
        gate.wait(5)
        return object()

    results = []
    threads = [threading.Thread(target=lambda: results.append(cache.get("k", compute))) for _ in range(6)]
    for t in threads:
        t.start()
    time.sleep(0.2)  # let every caller reach the cache while the first compute is blocked
    gate.set()
    for t in threads:
        t.join(5)
    assert len(calls) == 1
    assert len(results) == 6 and all(r is results[0] for r in results)


def test_refresh_failure_keeps_the_stale_value_and_retries_on_next_call():
    clock = Clock()
    cache = StaleWhileRevalidateCache("swr-test", ttl_seconds=60, clock=clock)
    outcomes = iter(["old", RuntimeError("db locked"), "new"])

    def compute():
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert cache.get("k", compute) == "old"
    clock.now += 61
    assert cache.get("k", compute) == "old"  # triggers the failing refresh
    _join_refresh_threads()
    # The failure neither evicted nor poisoned the entry: it is still served, and the next
    # call retries the refresh (this time it succeeds).
    assert cache.get("k", compute) == "old"
    _join_refresh_threads()
    assert cache.get("k", compute) == "new"


def test_cold_failure_propagates_and_is_not_cached():
    cache = StaleWhileRevalidateCache("swr-test", ttl_seconds=60)
    attempts = []

    def compute():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("boom")
        return 7

    with pytest.raises(RuntimeError, match="boom"):
        cache.get("k", compute)
    assert cache.get("k", compute) == 7
    assert len(attempts) == 2


def test_default_makes_a_cold_call_non_blocking_and_fills_in_the_background():
    cache = StaleWhileRevalidateCache("swr-test", ttl_seconds=60)
    gate = threading.Event()

    def compute():
        gate.wait(5)
        return 123

    began = time.monotonic()
    assert cache.get("k", compute, default=0) == 0
    assert time.monotonic() - began < 1
    assert cache.get("k", compute, default=0) == 0  # still computing: no second flight
    gate.set()
    _join_refresh_threads()
    assert cache.get("k", compute, default=0) == 123


def test_keys_are_independent_and_ttl_can_be_set_per_call():
    clock = Clock()
    cache = StaleWhileRevalidateCache("swr-test", ttl_seconds=600, clock=clock)
    calls = {"a": 0, "b": 0}

    def make(key):
        def compute():
            calls[key] += 1
            return calls[key]
        return compute

    cache.get("a", make("a"), ttl=60)
    cache.get("b", make("b"))
    clock.now += 100
    assert cache.get("a", make("a"), ttl=60) == 1  # stale value served ...
    _join_refresh_threads()
    assert calls == {"a": 2, "b": 1}  # ... and only the short-TTL key refreshed
    assert cache.get("b", make("b")) == 1


def test_clear_forgets_a_key_or_everything():
    cache = StaleWhileRevalidateCache("swr-test", ttl_seconds=60)
    cache.get("a", lambda: 1)
    cache.get("b", lambda: 2)
    cache.clear("a")
    assert cache.get("a", lambda: 10) == 10
    assert cache.get("b", lambda: 20) == 2
    cache.clear()
    assert cache.get("b", lambda: 30) == 30


def test_caches_a_real_sqlite_aggregate_and_serves_it_while_the_table_changes(tmp_path):
    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{tmp_path / 'c.db'}", connect_args={"check_same_thread": False})
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE t (v INTEGER)"))
        conn.execute(text("INSERT INTO t VALUES (1), (2)"))

    def count():
        with engine.connect() as conn:
            return conn.execute(text("SELECT COUNT(*) FROM t")).scalar()

    clock = Clock()
    cache = StaleWhileRevalidateCache("swr-test", ttl_seconds=60, clock=clock)
    assert cache.get("n", count) == 2
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO t VALUES (3)"))
    assert cache.get("n", count) == 2  # inside TTL: the COUNT(*) is not re-run
    clock.now += 61
    assert cache.get("n", count) == 2  # stale value now, refresh in the background
    _join_refresh_threads()
    assert cache.get("n", count) == 3


def test_clear_discards_the_result_of_a_computation_already_running():
    cache = StaleWhileRevalidateCache("swr-test", ttl_seconds=60)
    started = threading.Event()
    release = threading.Event()
    calls = []

    def compute():
        calls.append(1)
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
            return "computed-before-clear"
        return "computed-after-clear"

    first = []
    worker = threading.Thread(target=lambda: first.append(cache.get("k", compute)))
    worker.start()
    assert started.wait(2)
    cache.clear("k")
    release.set()
    worker.join(5)
    assert first == ["computed-before-clear"]  # its own caller still gets the answer
    # ... but it was not stored, so the cleared key is recomputed rather than served stale.
    assert cache.get("k", compute) == "computed-after-clear"
