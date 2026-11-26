"""Per-resource mutex so concurrent pipelines serialize per-stage.

Resource model
--------------
Each StageSpec carries a `resource_key`. Anything in the same key serializes;
distinct keys run in parallel. Typical mapping:

  download   → "net"            (HTTP rate limit; single global slot)
  extract    → "gpu-{ssh_target}"  → e.g. "gpu-4090", "gpu-kss"
  preindex   → "sql"            (SQLite WAL writer)
  embed      → "gpu-{ssh_target}"  → SAME key as extract on the same box,
                                      so extract+embed on one box don't OOM
  ingest     → "qdrant"         (Qdrant commit-charge headroom)

Why this design
---------------
- Tasks A and B on DIFFERENT ssh_targets (4090 vs kss) run extract concurrently.
- Tasks A and B on the SAME ssh_target serialize at the GPU resource — even if
  one is in extract and the other in embed, because both map to "gpu-{target}".
- Local stages (preindex, ingest) are global single-slot — protects SQL writes
  and Qdrant commit charge.

Implementation
--------------
One `threading.Condition` per resource key, the current holder (thread, re-entry
count, owner) and a heap of waiters ordered by `(rank, seq)`. A free resource is
granted only to the head of the heap:

  - rank is `(0, priority)` when the caller passed a priority, else `(1,)`, so
    prioritized waiters go before unprioritized ones and a smaller priority goes
    first. `priority` must be comparable among callers of the same key (the
    orchestrator always passes a tuple).
  - seq is a process-wide arrival counter, so equal priorities and unprioritized
    callers are FIFO.

Waiters sleep on the condition for at most `poll_s` and re-check `abort`; an
aborted waiter leaves the heap and wakes the rest. The same thread may re-enter
a key it already holds (count, not queue).

Why not an unordered lock: with a plain `threading.Lock` whoever polls first
after a release wins. On 2026-11-13 twenty-three slice tasks queued their
extract on one `gpu-local`; each release went to an arbitrary extract and three
embeds (waiting 8-13 h, their papers already in SQL but without vectors) lost
every time. Priority order lets the orchestrator grant a stage that may still
be running on the resource first, then the oldest task, so pipelines finish in
the order they started. This is the usual condition-variable + ordered wait
queue (CPython's `queue.PriorityQueue` is built the same way).
"""
from __future__ import annotations

import heapq
import itertools
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional

# Process-wide arrival counter: the FIFO tie-break among equal ranks.
_ARRIVALS = itertools.count()


class _ResourceState:
    __slots__ = ("cond", "waiters", "count", "thread", "owner")

    def __init__(self) -> None:
        self.cond = threading.Condition(threading.Lock())
        self.waiters: list[tuple] = []  # heap of (rank, seq)
        self.count = 0  # re-entry depth of the current holder; 0 = free
        self.thread: Optional[int] = None  # holding thread's ident
        self.owner: object = None


class StageLockSet:
    def __init__(self) -> None:
        self._states: dict[str, _ResourceState] = {}
        self._meta = threading.Lock()  # guards _states creation

    def _state(self, resource: str) -> _ResourceState:
        with self._meta:
            state = self._states.get(resource)
            if state is None:
                state = self._states[resource] = _ResourceState()
            return state

    def acquire(
        self,
        resource: str,
        abort: threading.Event,
        *,
        poll_s: float = 0.25,
        owner: Optional[object] = None,
        priority: Optional[Any] = None,
    ) -> bool:
        """Block until `resource` is granted to this caller or `abort` is set.

        Waiters are granted in `(priority, arrival)` order, prioritized before
        unprioritized (see module docstring). Returns True on acquisition,
        False on abort. Empty `resource` short-circuits to True (free-for-all,
        no serialization)."""
        if not resource:
            return True
        if abort.is_set():
            return False  # aborted callers never take the resource, even re-entrantly
        state = self._state(resource)
        me = threading.get_ident()
        with state.cond:
            if state.count and state.thread == me:
                state.count += 1  # re-entry from the holding thread
                return True
            rank = (0, priority) if priority is not None else (1,)
            ticket = (rank, next(_ARRIVALS))
            heapq.heappush(state.waiters, ticket)
            granted = False
            try:
                while not abort.is_set():
                    if state.count == 0 and state.waiters[0] == ticket:
                        heapq.heappop(state.waiters)
                        state.count = 1
                        state.thread = me
                        state.owner = owner if owner is not None else True
                        granted = True
                        return True
                    state.cond.wait(timeout=poll_s)
                return False
            finally:
                if not granted:
                    state.waiters.remove(ticket)
                    heapq.heapify(state.waiters)
                    state.cond.notify_all()  # the next head may now be grantable

    def release(self, resource: str) -> None:
        """Release `resource`. Safe on a key never acquired in this process."""
        if not resource:
            return
        with self._meta:
            state = self._states.get(resource)
        if state is None:
            return
        with state.cond:
            if state.count == 0:
                return  # nothing to release; defensive against finally-blocks
            state.count -= 1
            if state.count == 0:
                state.thread = None
                state.owner = None
                state.cond.notify_all()

    def holders(self) -> dict[str, object]:
        """Snapshot of current owners (resource_key → owner). For UI / events."""
        with self._meta:
            states = list(self._states.items())
        held: dict[str, object] = {}
        for resource, state in states:
            with state.cond:
                if state.count:
                    held[resource] = state.owner
        return held

    def _waiting(self, resource: str) -> int:
        """Number of callers queued on `resource` (test introspection)."""
        state = self._state(resource)
        with state.cond:
            return len(state.waiters)

    @contextmanager
    def hold(
        self,
        resource: str,
        abort: threading.Event,
        *,
        owner: Optional[object] = None,
        poll_s: float = 0.25,
        priority: Optional[Any] = None,
    ) -> Iterator[bool]:
        """Context manager: acquire on enter, release on exit (always).

        Yields True iff acquired; False means abort was set while queued.
        Exception-safe: release fires even if the critical section raises.

            with locks.hold("sql", abort, owner=task_id) as acquired:
                if not acquired:
                    return "cancelled"
                ... critical section ...
        """
        acquired = self.acquire(
            resource, abort, poll_s=poll_s, owner=owner, priority=priority,
        )
        try:
            yield acquired
        finally:
            if acquired:
                self.release(resource)
