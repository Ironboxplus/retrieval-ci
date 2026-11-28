"""One stale-while-revalidate cache for expensive read-only aggregates.

The WebUI statistics endpoints are COUNT(*) / GROUP BY scans over multi-million-row
tables (19 s for `chunks` by level on the production host). They change only when an
ingest lands, so a request must never pay for the scan once any value exists:

- fresh value: returned as is;
- stale value: returned at once, and ONE background daemon thread refreshes it
  (single flight per key); a failed refresh keeps the stale value and is retried by
  the next call;
- no value yet: the first caller computes synchronously and concurrent callers wait for
  that same computation (a cold failure propagates and caches nothing). Passing
  `default=` makes the cold path non-blocking instead (the default is returned and the
  value is filled in by the background refresh) for sources that can block for a minute,
  such as a Qdrant collection probe during HNSW indexing.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)

_MISSING = object()


class _Flight:
    """One in-progress computation that other callers may wait on."""

    __slots__ = ("done", "value", "error")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.value: Any = None
        self.error: BaseException | None = None


class StaleWhileRevalidateCache:
    def __init__(
        self,
        name: str,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self._ttl = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: Dict[Any, tuple[Any, float]] = {}
        self._flights: Dict[Any, _Flight] = {}
        self._generation = 0

    def get(
        self,
        key: Any,
        compute: Callable[[], Any],
        *,
        ttl: float | None = None,
        default: Any = _MISSING,
    ) -> Any:
        max_age = self._ttl if ttl is None else ttl
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                value, stamp = entry
                if self._clock() - stamp >= max_age:
                    self._refresh_in_background_locked(key, compute)
                return value
            if default is not _MISSING:
                self._refresh_in_background_locked(key, compute)
                return default
            flight = self._flights.get(key)
            leader = flight is None
            if leader:
                flight = self._flights[key] = _Flight()
        if leader:
            self._run(key, compute, flight, background=False)
        else:
            flight.done.wait()
        if flight.error is not None:
            raise flight.error
        return flight.value

    def clear(self, key: Any = _MISSING) -> None:
        """Forget one key, or everything.

        A computation already running keeps serving the callers waiting on it, but its
        result is dropped: it may have read the state this call was made to forget.
        """
        with self._lock:
            self._generation += 1
            if key is _MISSING:
                self._entries.clear()
                self._flights.clear()
            else:
                self._entries.pop(key, None)
                self._flights.pop(key, None)

    def _refresh_in_background_locked(self, key: Any, compute: Callable[[], Any]) -> None:
        if key in self._flights:
            return
        flight = self._flights[key] = _Flight()
        threading.Thread(
            target=self._run,
            args=(key, compute, flight, True),
            daemon=True,
            name=f"{self.name}-refresh",
        ).start()

    def _run(self, key: Any, compute: Callable[[], Any], flight: _Flight, background: bool) -> None:
        generation = self._generation
        try:
            value = compute()
        except BaseException as exc:  # noqa: BLE001 - delivered to waiters; background callers log it
            flight.error = exc
            if background:
                logger.warning("%s refresh failed for %r; keeping the stale value: %s", self.name, key, exc)
        else:
            flight.value = value
            with self._lock:
                if generation == self._generation:
                    self._entries[key] = (value, self._clock())
        finally:
            with self._lock:
                if self._flights.get(key) is flight:
                    del self._flights[key]
            flight.done.set()
