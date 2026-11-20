"""Process memory watch: periodic committed-memory log plus on-the-spot capture of a burst.

The old watchdog logged psutil ``uss`` (resident only), which on Linux hides memory that
is committed but swapped out. This module reads VmData/VmSwap/VmRSS from /proc and, when
VmData jumps, records every thread's Python stack and a short tracemalloc window so the
allocating code can be named.
"""

from __future__ import annotations

import faulthandler
import logging
import sys
import threading
import time
import traceback
import tracemalloc
from collections import deque
from typing import Callable

_GB = 1024**3
_STATUS_FIELDS = {"VmData": "data", "VmSwap": "swap", "VmRSS": "rss"}


def committed_bytes() -> dict[str, int]:
    """Return ``{"data", "swap", "rss"}`` in bytes for the current process."""
    if sys.platform.startswith("linux"):
        out = {"data": 0, "swap": 0, "rss": 0}
        with open("/proc/self/status", "r", encoding="ascii", errors="replace") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                name = _STATUS_FIELDS.get(key)
                if name:
                    out[name] = int(rest.split()[0]) * 1024
        return out
    import psutil

    info = psutil.Process().memory_info()
    data = getattr(info, "private", None)
    if data is None:
        data = info.vms
    return {"data": int(data), "swap": 0, "rss": int(info.rss)}


class BurstDetector:
    """Flags a rise of ``threshold_bytes`` within ``window_s``, then stays quiet for ``cooldown_s``."""

    def __init__(self, threshold_bytes: int = 1 << 30, window_s: float = 60.0, cooldown_s: float = 600.0):
        self.threshold_bytes = threshold_bytes
        self.window_s = window_s
        self.cooldown_s = cooldown_s
        self._samples: deque[tuple[float, int]] = deque()
        self._quiet_until = float("-inf")

    def observe(self, t: float, data_bytes: int) -> int | None:
        self._samples.append((t, data_bytes))
        while self._samples and self._samples[0][0] < t - self.window_s:
            self._samples.popleft()
        if t < self._quiet_until:
            return None
        growth = data_bytes - min(v for _, v in self._samples)
        if growth >= self.threshold_bytes:
            self._quiet_until = t + self.cooldown_s
            return growth
        return None


def thread_stacks() -> str:
    """Python stack (last 25 frames) of every live thread, labelled with its name."""
    names = {t.ident: t.name for t in threading.enumerate()}
    parts = []
    for ident, frame in sys._current_frames().items():
        stack = "".join(traceback.format_stack(frame, limit=25))
        parts.append(f"--- thread {names.get(ident, '?')} ({ident}) ---\n{stack}")
    return "\n".join(parts)


def _report_allocations(logger: logging.Logger) -> None:
    snapshot = tracemalloc.take_snapshot()
    for stat in snapshot.statistics("traceback")[:15]:
        logger.warning(
            "mem.burst.alloc size=%.1fMB count=%d\n%s",
            stat.size / 2**20,
            stat.count,
            "\n".join(stat.traceback.format()),
        )


def run_memory_watch(
    logger: logging.Logger,
    *,
    sample_s: float = 5.0,
    log_every_s: float = 60.0,
    trace_s: float = 30.0,
    extra: Callable[[], str] | None = None,
    stop: threading.Event | None = None,
    read: Callable[[], dict[str, int]] = committed_bytes,
    clock: Callable[[], float] = time.monotonic,
    detector: BurstDetector | None = None,
    stall_s: float = 20.0,
    stall_file=None,
    stall_cooldown_s: float = 120.0,
) -> None:
    """Log committed memory and capture bursts.

    When an allocation stalls the GIL, this thread cannot run either, so each loop re-arms
    ``faulthandler.dump_traceback_later``: its C thread prints every thread's stack without
    the GIL if the loop does not come back within ``stall_s``. (One such timer per process.)
    """
    stop = stop or threading.Event()
    detector = detector or BurstDetector()
    last_log = float("-inf")
    trace_report_at: float | None = None
    started_tracing = False
    last_tick: float | None = None
    stall_quiet_until = float("-inf")
    try:
        while not stop.is_set():
            try:
                now = clock()
                mem = read()
                if last_tick is not None and now - last_tick > stall_s:
                    stall_quiet_until = now + stall_cooldown_s
                    logger.warning(
                        "mem.stall gap=%.1fs vmdata=%.2fGB vmswap=%.2fGB rss=%.2fGB; "
                        "faulthandler wrote all thread stacks to %s",
                        now - last_tick,
                        mem["data"] / _GB,
                        mem["swap"] / _GB,
                        mem["rss"] / _GB,
                        getattr(stall_file, "name", "stderr"),
                    )
                if now - last_log >= log_every_s:
                    last_log = now
                    tail = ""
                    if extra is not None:
                        try:
                            tail = " " + extra()
                        except Exception:
                            logger.exception("mem.watchdog extra failed")
                    logger.info(
                        "mem.watchdog vmdata=%.2fGB vmswap=%.2fGB rss=%.2fGB threads=%d%s",
                        mem["data"] / _GB,
                        mem["swap"] / _GB,
                        mem["rss"] / _GB,
                        threading.active_count(),
                        tail,
                    )
                grown = detector.observe(now, mem["data"])
                if grown is not None:
                    logger.warning(
                        "mem.burst vmdata=%.2fGB grew %.2fGB within %ds; thread stacks:\n%s",
                        mem["data"] / _GB,
                        grown / _GB,
                        int(detector.window_s),
                        thread_stacks(),
                    )
                    if not tracemalloc.is_tracing():
                        tracemalloc.start(10)
                        started_tracing = True
                        trace_report_at = now + trace_s
                if trace_report_at is not None and now >= trace_report_at:
                    trace_report_at = None
                    try:
                        _report_allocations(logger)
                    finally:
                        if started_tracing:
                            tracemalloc.stop()
                            started_tracing = False
            except Exception:
                logger.exception("mem.watchdog iteration failed")
                if started_tracing and trace_report_at is None and tracemalloc.is_tracing():
                    tracemalloc.stop()
                    started_tracing = False
            try:
                if clock() >= stall_quiet_until:
                    faulthandler.dump_traceback_later(
                        stall_s, repeat=False, file=stall_file if stall_file is not None else sys.stderr, exit=False
                    )
                else:
                    faulthandler.cancel_dump_traceback_later()
                last_tick = clock()
            except Exception:
                logger.exception("mem.watchdog could not arm the stall dump")
            stop.wait(sample_s)
    finally:
        faulthandler.cancel_dump_traceback_later()
        if started_tracing and tracemalloc.is_tracing():
            tracemalloc.stop()


def start_memory_watch(logger: logging.Logger, *, extra: Callable[[], str] | None = None) -> threading.Thread:
    thread = threading.Thread(
        target=run_memory_watch, args=(logger,), kwargs={"extra": extra}, daemon=True, name="mem-watch"
    )
    thread.start()
    return thread
