"""Bounded, leak-safe runner for streaming a blocking generator over async.

W60 fix: the previous /api/search/stream spawned an unbounded raw
``threading.Thread`` per request feeding a ``queue.Queue(maxsize=8)``. Under
concurrency each thread's first CUDA call committed multi-GB host-side WDDM
backing that never released; on client disconnect the producer blocked
forever at ``q.put`` (queue full) and leaked. ~18 concurrent requests pushed
private commit to ~97GB and OOM-killed Qdrant.

This module routes every streaming search through a single process-wide
``ThreadPoolExecutor`` capped at ``MAX_STREAM_WORKERS`` so the number of
CUDA-touching threads is bounded regardless of request concurrency, and the
producer is signalled to stop (and never blocks indefinitely) the moment the
consumer abandons the stream.
"""
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, AsyncIterator, Callable, Iterator

from core.logging import get_logger

logger = get_logger(__name__)

# Cap concurrent CUDA-touching producer threads. 2 keeps the GPU busy without
# letting per-thread host backing accumulate unbounded (W60).
MAX_STREAM_WORKERS = 2

_EXECUTOR = ThreadPoolExecutor(
    max_workers=MAX_STREAM_WORKERS, thread_name_prefix="search-stream"
)

_SENTINEL = object()

# Bound on items buffered between producer and consumer. The producer never
# blocks on this indefinitely — it polls with a timeout and checks the stop
# flag, so an abandoned consumer cannot wedge the producer.
_QUEUE_MAX = 8
_PUT_POLL_SEC = 0.2


async def stream_blocking(
    gen_factory: Callable[[], Iterator[Any]],
    *,
    queue_max: int = _QUEUE_MAX,
) -> AsyncIterator[Any]:
    """Run ``gen_factory()`` (a blocking generator) in the bounded executor and
    yield its items as they arrive.

    Guarantees:
      * At most ``MAX_STREAM_WORKERS`` producers run concurrently (shared
        executor), bounding CUDA-touching threads.
      * If the consumer stops early (``aclose`` / cancellation / disconnect),
        the producer is signalled via ``stop`` and exits promptly instead of
        blocking forever on a full queue.
      * A producer exception is surfaced as a final ``{"stage": "error", ...}``
        event rather than crashing the worker thread silently.
    """
    loop = asyncio.get_event_loop()
    events: asyncio.Queue[Any] = asyncio.Queue(maxsize=queue_max)
    stop = threading.Event()

    def enqueue_from_producer(item: Any) -> bool:
        """Apply async-queue backpressure without consuming a default-executor thread."""
        while not stop.is_set():
            pending = asyncio.run_coroutine_threadsafe(events.put(item), loop)
            try:
                pending.result(timeout=_PUT_POLL_SEC)
                return True
            except FutureTimeoutError:
                # Cancellation is safe while Queue.put() is waiting for capacity. If the
                # event landed at the timeout boundary, cancel() returns False and result()
                # confirms success so we never enqueue a duplicate on the next iteration.
                if pending.cancel():
                    continue
                try:
                    pending.result()
                    return True
                except FutureCancelledError:
                    continue
            except (FutureCancelledError, RuntimeError):
                return False
        return False

    def produce() -> None:
        iterator: Iterator[Any] | None = None
        try:
            if stop.is_set():
                return
            iterator = iter(gen_factory())
            while not stop.is_set():
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                if not enqueue_from_producer(item):
                    break
        except Exception as e:  # producer failure -> error event, not a dead thread
            logger.exception("stream producer failed")
            if not stop.is_set():
                enqueue_from_producer(
                    {"stage": "error", "error": f"{type(e).__name__}: {e}"})
        finally:
            if iterator is not None and stop.is_set():
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
            if not stop.is_set():
                enqueue_from_producer(_SENTINEL)

    producer_future = _EXECUTOR.submit(produce)
    try:
        while True:
            item = await events.get()
            if item is _SENTINEL:
                break
            yield item
    finally:
        # Consumer is done or was abandoned. A producer waiting for async queue
        # capacity polls this flag and cancels its pending Queue.put promptly.
        stop.set()
        # If the bounded executor has not started this producer yet, remove it from the
        # queue so an abandoned HTTP stream cannot later begin expensive embed work.
        producer_future.cancel()
