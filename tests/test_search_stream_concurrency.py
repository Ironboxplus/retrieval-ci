"""W60 regression: /api/search/stream must bound concurrent CUDA-touching
producer threads and must not leak a producer thread when the consumer
abandons the stream (client disconnect).

Root cause (W60): main.py spawned an unbounded raw threading.Thread per
request feeding a queue.Queue(maxsize=8). Under concurrency, each thread's
CUDA work committed ~4.7GB host-side WDDM backing; on client disconnect the
producer blocked forever at q.put (queue full) and leaked. ~18 concurrent
requests -> +84GB private commit -> Qdrant OOM.
"""
import asyncio
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress


class TestStreamConcurrencyBound(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_producers_bounded_to_max_workers(self):
        from core.retrieval.stream_runner import stream_blocking, MAX_STREAM_WORKERS

        active = 0
        peak = 0
        lock = threading.Lock()
        release = threading.Event()

        def gen_factory():
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                release.wait(timeout=5)
                yield {"stage": "done"}
            finally:
                with lock:
                    active -= 1

        async def consume():
            out = []
            async for ev in stream_blocking(gen_factory):
                out.append(ev)
            return out

        tasks = [asyncio.create_task(consume()) for _ in range(6)]
        await asyncio.sleep(1.0)  # let as many producers start as allowed
        with lock:
            observed_peak = peak
        release.set()
        await asyncio.gather(*tasks)

        self.assertGreaterEqual(observed_peak, 1, "no producer ever started")
        self.assertLessEqual(
            observed_peak, MAX_STREAM_WORKERS,
            f"peak concurrent producers {observed_peak} exceeded "
            f"MAX_STREAM_WORKERS={MAX_STREAM_WORKERS} (W60: unbounded CUDA threads)",
        )

    async def test_producer_stops_when_consumer_abandons(self):
        from core.retrieval.stream_runner import stream_blocking

        started = threading.Event()
        producer_finished = threading.Event()

        def gen_factory():
            started.set()
            try:
                for i in range(100000):  # effectively unbounded stream
                    yield {"stage": "ev", "i": i}
            finally:
                producer_finished.set()

        gen = stream_blocking(gen_factory)
        first = await gen.__anext__()
        self.assertEqual(first["stage"], "ev")

        await gen.aclose()  # simulate client disconnect / early consumer exit

        self.assertTrue(
            producer_finished.wait(timeout=5),
            "producer thread leaked: did not terminate after consumer abandoned "
            "the stream (W60 leak: q.put blocks forever on full queue)",
        )

    async def test_queued_producer_is_cancelled_before_generator_advances(self):
        from core.retrieval.stream_runner import MAX_STREAM_WORKERS, stream_blocking

        release = threading.Event()
        started_lock = threading.Lock()
        started_count = 0
        all_workers_started = threading.Event()

        def blocking_factory():
            nonlocal started_count
            with started_lock:
                started_count += 1
                if started_count == MAX_STREAM_WORKERS:
                    all_workers_started.set()
            release.wait(timeout=5)
            yield {"stage": "done"}

        async def consume_blocker():
            return [event async for event in stream_blocking(blocking_factory)]

        blockers = [
            asyncio.create_task(consume_blocker())
            for _ in range(MAX_STREAM_WORKERS)
        ]
        deadline = asyncio.get_running_loop().time() + 2
        while not all_workers_started.is_set() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        self.assertTrue(all_workers_started.is_set())

        advanced = threading.Event()

        def queued_factory():
            advanced.set()
            yield {"stage": "should-not-run"}

        queued_stream = stream_blocking(queued_factory)
        queued_next = asyncio.create_task(queued_stream.__anext__())
        await asyncio.sleep(0.05)
        queued_next.cancel()
        with suppress(asyncio.CancelledError):
            await queued_next
        await queued_stream.aclose()

        release.set()
        await asyncio.gather(*blockers)
        await asyncio.sleep(0.1)
        self.assertFalse(
            advanced.is_set(),
            "a cancelled stream still advanced its queued generator after a worker freed",
        )

    async def test_yields_all_events_in_order_normal_path(self):
        from core.retrieval.stream_runner import stream_blocking

        def gen_factory():
            for i in range(5):
                yield {"stage": "ev", "i": i}

        out = []
        async for ev in stream_blocking(gen_factory):
            out.append(ev)

        self.assertEqual([e["i"] for e in out], [0, 1, 2, 3, 4])

    async def test_stream_bridge_does_not_depend_on_default_executor_capacity(self):
        """Long-lived stream consumers must not consume the same executor used by
        request handlers via asyncio.to_thread. A saturated default executor must not
        prevent an already-running stream producer from delivering its next event."""
        from core.retrieval.stream_runner import stream_blocking

        loop = asyncio.get_running_loop()
        release_default_worker = threading.Event()
        default_worker_started = threading.Event()
        default_executor = ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(default_executor)

        def occupy_default_worker():
            default_worker_started.set()
            release_default_worker.wait(timeout=5)

        blocker = loop.run_in_executor(None, occupy_default_worker)
        self.assertTrue(default_worker_started.wait(timeout=1))

        def gen_factory():
            yield {"stage": "ev", "i": 0}

        async def consume_one():
            async for event in stream_blocking(gen_factory):
                return event
            self.fail("stream ended without yielding an event")

        try:
            event = await asyncio.wait_for(consume_one(), timeout=0.5)
            self.assertEqual(event, {"stage": "ev", "i": 0})
        finally:
            release_default_worker.set()
            await blocker
            default_executor.shutdown(wait=True)

    async def test_producer_exception_propagates_as_error_event(self):
        from core.retrieval.stream_runner import stream_blocking

        def gen_factory():
            yield {"stage": "ev", "i": 0}
            raise RuntimeError("boom")

        out = []
        async for ev in stream_blocking(gen_factory):
            out.append(ev)

        self.assertEqual(out[0], {"stage": "ev", "i": 0})
        self.assertEqual(out[-1]["stage"], "error")
        self.assertIn("boom", out[-1]["error"])


if __name__ == "__main__":
    unittest.main()
