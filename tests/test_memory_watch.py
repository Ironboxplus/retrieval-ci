"""Memory watch: burst detection logic and an end-to-end run on real process memory."""

from __future__ import annotations

import logging
import threading
import time
import tracemalloc

from core.memory_watch import BurstDetector, committed_bytes, run_memory_watch

GIB = 1 << 30


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.messages: list[str] = []
        self._guard = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        text = record.getMessage()
        if record.exc_info:
            text += "\n" + logging.Formatter().formatException(record.exc_info)
        with self._guard:
            self.messages.append(text)

    def snapshot(self) -> list[str]:
        with self._guard:
            return list(self.messages)


def _logger(name: str) -> tuple[logging.Logger, _Collect]:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = _Collect()
    logger.addHandler(handler)
    return logger, handler


def _wait_for(predicate, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def test_burst_detector_ignores_a_flat_series():
    d = BurstDetector(threshold_bytes=GIB, window_s=60.0, cooldown_s=600.0)
    assert [d.observe(t, 5 * GIB) for t in range(0, 300, 5)] == [None] * 60


def test_burst_detector_fires_once_then_cools_down_then_fires_again():
    d = BurstDetector(threshold_bytes=GIB, window_s=60.0, cooldown_s=600.0)
    base = 2 * GIB
    assert d.observe(0, base) is None
    assert d.observe(30, base + int(0.5 * GIB)) is None
    grown = d.observe(55, base + int(1.1 * GIB))
    assert grown == int(1.1 * GIB)
    # still growing inside the cooldown: silent
    assert d.observe(70, base + int(2.5 * GIB)) is None
    assert d.observe(300, base + int(3.0 * GIB)) is None
    # after the cooldown a fresh rise fires again
    assert d.observe(700, base + int(3.0 * GIB)) is None
    assert d.observe(730, base + int(4.2 * GIB)) == int(1.2 * GIB)


def test_committed_bytes_returns_nonnegative_ints():
    got = committed_bytes()
    assert set(got) >= {"data", "swap", "rss"}
    assert all(isinstance(got[k], int) and got[k] >= 0 for k in ("data", "swap", "rss"))
    assert got["data"] > 0


def test_real_allocation_triggers_stacks_and_tracemalloc_report():
    assert not tracemalloc.is_tracing()
    logger, handler = _logger("test_memory_watch.e2e")
    stop = threading.Event()
    release = threading.Event()
    held: list[bytearray] = []

    def _allocate_for_burst_test():
        held.append(bytearray(300 * 2**20))
        junk = []
        while not release.is_set():  # keep allocating so tracemalloc has something to attribute
            junk.append(bytes(64 * 1024))
            if len(junk) > 50:
                junk.clear()
            time.sleep(0.01)

    watcher = threading.Thread(
        target=run_memory_watch,
        args=(logger,),
        kwargs=dict(
            sample_s=0.2,
            trace_s=1.0,
            log_every_s=0.5,
            stop=stop,
            detector=BurstDetector(threshold_bytes=200 * 2**20, window_s=60.0, cooldown_s=600.0),
        ),
        daemon=True,
    )
    watcher.start()
    time.sleep(0.6)  # let the detector record a baseline before the allocation
    allocator = threading.Thread(target=_allocate_for_burst_test, daemon=True)
    allocator.start()
    try:
        assert _wait_for(lambda: any(m.startswith("mem.burst.alloc size=") for m in handler.snapshot()), 15.0)
    finally:
        release.set()
        allocator.join(5)
        stop.set()
        watcher.join(5)
    msgs = handler.snapshot()
    assert not watcher.is_alive()
    assert any(m.startswith("mem.watchdog vmdata=") for m in msgs)
    burst = [m for m in msgs if m.startswith("mem.burst ")]
    assert burst, msgs
    assert any("_allocate_for_burst_test" in m for m in burst)
    assert any(m.startswith("mem.burst.alloc size=") for m in msgs)
    assert not tracemalloc.is_tracing()
    held.clear()


def test_read_failure_is_logged_and_sampling_continues():
    logger, handler = _logger("test_memory_watch.readfail")
    stop = threading.Event()
    calls = {"n": 0}

    def flaky_read():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return committed_bytes()

    watcher = threading.Thread(
        target=run_memory_watch,
        args=(logger,),
        kwargs=dict(sample_s=0.1, log_every_s=0.2, trace_s=0.5, stop=stop, read=flaky_read),
        daemon=True,
    )
    watcher.start()
    try:
        assert _wait_for(lambda: calls["n"] >= 4, 5.0)
        assert watcher.is_alive()
    finally:
        stop.set()
        watcher.join(5)
    msgs = handler.snapshot()
    assert any("boom" in m for m in msgs)
    assert any(m.startswith("mem.watchdog vmdata=") for m in msgs)
    assert not watcher.is_alive()


def test_gil_stall_dumps_all_thread_stacks_without_the_gil(tmp_path):
    logger, handler = _logger("test_memory_watch.stall")
    stop = threading.Event()
    dump_path = tmp_path / "stall.txt"

    probe = time.perf_counter()
    sum(range(2_000_000))
    per_item = (time.perf_counter() - probe) / 2_000_000
    n = int(3.0 / per_item)

    def _hold_gil_for_test():
        sum(range(n))  # one C call: the GIL is not released for ~3 s

    with open(dump_path, "w") as stall_file:
        watcher = threading.Thread(
            target=run_memory_watch,
            args=(logger,),
            kwargs=dict(
                sample_s=0.1,
                log_every_s=10.0,
                trace_s=0.5,
                stop=stop,
                stall_s=1.0,
                stall_file=stall_file,
                stall_cooldown_s=120.0,
                detector=BurstDetector(threshold_bytes=1 << 40),
            ),
            daemon=True,
        )
        watcher.start()
        time.sleep(0.5)
        holder = threading.Thread(target=_hold_gil_for_test, daemon=True)
        holder.start()
        holder.join(30)
        assert _wait_for(lambda: any(m.startswith("mem.stall") for m in handler.snapshot()), 10.0)
        stop.set()
        watcher.join(5)
        assert not watcher.is_alive()
        text = dump_path.read_text()
        assert "_hold_gil_for_test" in text
        size = dump_path.stat().st_size
        time.sleep(2.0)  # timer must be cancelled: nothing more is written
        assert dump_path.stat().st_size == size
