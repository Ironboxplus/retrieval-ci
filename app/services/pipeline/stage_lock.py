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
This is a thin wrapper over `threading.Lock` (one lock per resource key,
lazily created). Acquire polls with a small timeout so we stay responsive to
the abort event without spinning. `threading.Lock` is not strictly FIFO under
contention; for our load (≤ a handful of concurrent pipelines) the difference
is invisible. If strict FIFO becomes a requirement we can switch to a
Semaphore + ordered queue without changing the call sites.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator, Optional


class StageLockSet:
    def __init__(self) -> None:
        # defaultdict gives lazy lock creation on first reference — no
        # registration step needed for new resource keys. RLock instead of
        # Lock so a refactor that accidentally re-enters a stage from the
        # same thread won't self-deadlock (negligible overhead vs Lock).
        self._locks: dict[str, threading.RLock] = defaultdict(threading.RLock)
        self._owners: dict[str, object] = {}
        self._meta = threading.Lock()  # guards _owners reads/writes

    def acquire(
        self,
        resource: str,
        abort: threading.Event,
        *,
        poll_s: float = 0.25,
        owner: Optional[object] = None,
    ) -> bool:
        """Block until `resource` is free or `abort` is set.

        Returns True on acquisition, False on abort. Empty `resource` short-
        circuits to True (free-for-all, no serialization)."""
        if not resource:
            return True
        lock = self._locks[resource]
        while not abort.is_set():
            if lock.acquire(timeout=poll_s):
                with self._meta:
                    self._owners[resource] = owner if owner is not None else True
                return True
        return False

    def release(self, resource: str) -> None:
        """Release `resource`. Safe on a key never acquired in this process."""
        if not resource:
            return
        with self._meta:
            had_owner = self._owners.pop(resource, None) is not None
        if not had_owner:
            return  # nothing to release; defensive against finally-blocks
        try:
            self._locks[resource].release()
        except RuntimeError:
            # Already released (e.g. double release) — log-worthy but not fatal.
            pass

    def holders(self) -> dict[str, object]:
        """Snapshot of current owners (resource_key → owner). For UI / events."""
        with self._meta:
            return dict(self._owners)

    @contextmanager
    def hold(
        self,
        resource: str,
        abort: threading.Event,
        *,
        owner: Optional[object] = None,
        poll_s: float = 0.25,
    ) -> Iterator[bool]:
        """Context manager: acquire on enter, release on exit (always).

        Yields True iff acquired; False means abort was set while queued.
        Exception-safe: release fires even if the critical section raises.

            with locks.hold("sql", abort, owner=task_id) as acquired:
                if not acquired:
                    return "cancelled"
                ... critical section ...
        """
        acquired = self.acquire(resource, abort, poll_s=poll_s, owner=owner)
        try:
            yield acquired
        finally:
            if acquired:
                self.release(resource)
