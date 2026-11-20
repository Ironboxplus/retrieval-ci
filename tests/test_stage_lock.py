"""TDD: StageLockSet — per-resource mutex so concurrent pipelines serialize
per-stage but otherwise run side-by-side.

A "resource" is whatever's behind the stage. For local stages (preindex, ingest)
the resource is the local DB / Qdrant. For remote-GPU stages (extract, embed)
the resource is the GPU box at a given ssh_target: extract on 4090 and embed on
4090 share the *same* lock so they can't double-book a GPU, but extract on 4090
and extract on kss are independent.

Tests use TINY timeouts so failure modes (deadlock, leak, missing release) fail
fast instead of hanging the suite.
"""
from __future__ import annotations

import threading
import time

import pytest

from app.services.pipeline.stage_lock import StageLockSet


@pytest.fixture
def locks():
    return StageLockSet()


@pytest.fixture
def abort():
    return threading.Event()


def test_acquire_release_single_resource(locks, abort):
    assert locks.acquire("ingest", abort, poll_s=0.01) is True
    # second acquire must NOT succeed while held — but with timeout, returns False
    e2 = threading.Event()
    e2.set()  # immediately aborted so we don't hang
    assert locks.acquire("ingest", e2, poll_s=0.01) is False
    locks.release("ingest")
    # now free → next acquire succeeds
    assert locks.acquire("ingest", abort, poll_s=0.01) is True
    locks.release("ingest")


def test_distinct_resources_do_not_contend(locks, abort):
    """Different resource keys → independent locks, both held simultaneously."""
    assert locks.acquire("gpu-4090", abort) is True
    assert locks.acquire("gpu-kss", abort) is True
    locks.release("gpu-4090")
    locks.release("gpu-kss")


def test_acquire_blocks_until_released_by_other_thread(locks, abort):
    assert locks.acquire("preindex", abort) is True

    second_got = threading.Event()

    def waiter():
        if locks.acquire("preindex", abort, poll_s=0.01):
            second_got.set()
            locks.release("preindex")

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    # second thread should be blocked while we hold the lock
    assert not second_got.wait(timeout=0.1)
    locks.release("preindex")
    # now the waiter should finish
    assert second_got.wait(timeout=1.0)
    t.join(timeout=1.0)


def test_abort_unblocks_pending_acquire(locks):
    abort = threading.Event()
    assert locks.acquire("gpu-4090", abort) is True

    result: list[bool] = []

    def waiter():
        result.append(locks.acquire("gpu-4090", abort, poll_s=0.01))

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    time.sleep(0.05)  # waiter is in acquire loop
    abort.set()
    t.join(timeout=1.0)
    assert result == [False]  # aborted → False, not deadlocked
    locks.release("gpu-4090")


def test_release_unknown_key_is_safe(locks):
    """Calling release on a never-acquired resource must not raise (defensive).
    Without this, a finally-block on an early-aborted acquire would crash."""
    locks.release("never-acquired")  # must not raise


def test_holders_reports_currently_held_resources(locks, abort):
    """Inspection hook for UI / events: who is in each resource."""
    locks.acquire("preindex", abort, owner="task-7")
    locks.acquire("gpu-4090", abort, owner="task-9")
    held = locks.holders()
    assert held["preindex"] == "task-7"
    assert held["gpu-4090"] == "task-9"
    locks.release("preindex")
    locks.release("gpu-4090")
    assert locks.holders() == {}


def test_hold_context_manager_releases_on_normal_exit(locks, abort):
    with locks.hold("sql", abort) as acquired:
        assert acquired is True
        assert "sql" in locks.holders()
    assert "sql" not in locks.holders()  # released


def test_hold_context_manager_releases_on_exception(locks, abort):
    with pytest.raises(RuntimeError, match="boom"):
        with locks.hold("sql", abort):
            raise RuntimeError("boom")
    assert "sql" not in locks.holders()  # released despite exception


def test_hold_yields_false_when_aborted_before_acquire(locks):
    # Pre-take the lock so the second hold() must wait.
    held = threading.Event()
    holder_done = threading.Event()

    def holder():
        with locks.hold("sql", threading.Event()):
            held.set()
            holder_done.wait(timeout=2.0)

    threading.Thread(target=holder, daemon=True).start()
    held.wait(timeout=1.0)

    abort = threading.Event()
    abort.set()  # already aborted → hold() should yield False fast
    with locks.hold("sql", abort, poll_s=0.01) as ok:
        assert ok is False
    holder_done.set()


def test_all_waiters_eventually_served(locks):
    """3 threads queue on the same lock — all eventually drained, no deadlock.
    Order is NOT guaranteed (stdlib threading.Lock is unfair); we only assert
    completeness."""
    abort = threading.Event()
    assert locks.acquire("ingest", abort, owner="A") is True

    served: list[str] = []
    served_lock = threading.Lock()

    def waiter(name: str):
        if locks.acquire("ingest", abort, poll_s=0.005, owner=name):
            with served_lock:
                served.append(name)
            locks.release("ingest")

    threads = []
    for name in ("B", "C", "D"):
        t = threading.Thread(target=waiter, args=(name,), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.02)

    time.sleep(0.05)
    locks.release("ingest")
    for t in threads:
        t.join(timeout=2.0)
    assert sorted(served) == ["B", "C", "D"]
