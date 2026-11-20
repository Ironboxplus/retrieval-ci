r"""Rate limiter vendored for the online reranker.

Source: .233:E:\JETB\pythons\playground\homeassitant\deploy\camera-vlm\live_narrator.py:585-680
(local copy C:\Users\Arc\AppData\Local\Temp\csd\ha233\live_narrator.py), copied 2026-11-21.
The class body is unchanged except for the added ``try_acquire``.
"""
from __future__ import annotations

import heapq
import itertools
import threading
import time


class TokenBucket:
    """The account's per-minute request/token budget as a bucket that drains
    continuously (GCRA): a request goes as soon as enough has drained to hold
    it. The sliding 60 s window this replaced only freed budget a whole
    request at a time, so with ~13k-token segments it held five of them --
    65k of the 72k it allowed -- and that gap repeated every minute.
    Every request is charged its estimate up front and settled to its real
    usage once known, so the account's hard TPM is never what stops one.

    This is the only place that decides whether a cloud request may go --
    segment, merge or daily, from whichever thread. There is no other pacing,
    counting or waiting anywhere.

    A request bigger than the whole budget (it cannot "fit" however long it
    waits) goes once the bucket is empty, and is charged like any other. That
    is also all that is needed to cover the 2026-10-07 probe, where after one
    150-frame / 132,346-token request the account held every request for
    about 60 s: charged 132,346 against 80,000 a minute (1,333 a second), the
    bucket owes 52,346 more than it holds, and a 25,000-token request fits
    only once that has drained to 55,000 -- (132,346 - 55,000) / 1,333 = 58 s
    later, a 500-token one after 40 s. The same order as the probe's ~60 s,
    from the one calculation every request goes through, with no special case
    for the big one (an earlier version froze everything while it ran and for
    65 s after; with a first token allowed 600 s that could freeze the whole
    cloud for ten minutes)."""

    def __init__(self, tokens_per_minute: int, requests_per_minute: int, window_seconds: float = 60.0):
        self.tokens_per_minute = tokens_per_minute
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self.tokens = 0.0  # charged and not yet drained
        self.requests = 0.0
        self.updated = None
        self.charged = 0  # requests charged so far
        self.settled_tokens = 0  # real usage reported so far
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        self.waiting = []  # heap of (-priority, arrival)
        self.arrivals = itertools.count()

    def acquire(self, tokens: int, reserve: int = 0, priority: float = 0.0, clock=time.monotonic) -> list:
        """Blocks until `tokens` fit, then charges them; safe from any number
        of request threads at once. `reserve` tokens of the bucket are left
        free by this request, for callers that pass a smaller reserve.
        Waiters go strictly highest `priority` first (then first come): with
        every worker queued on the budget, a fresh recording must not wait
        behind all of them."""
        with self.changed:
            ticket = (-priority, next(self.arrivals))
            heapq.heappush(self.waiting, ticket)
            try:
                while True:
                    wait = None
                    if self.waiting[0] == ticket:
                        now = clock()
                        wait = self.wait_seconds(tokens + reserve, now)
                        if wait <= 0:
                            return self.charge(tokens, now)
                    self.changed.wait(wait)
            finally:
                self.waiting.remove(ticket)
                heapq.heapify(self.waiting)
                self.changed.notify_all()

    def _drain(self, now: float) -> None:
        if self.updated is not None:
            elapsed = max(0.0, now - self.updated)
            self.tokens = max(0.0, self.tokens - elapsed * self.tokens_per_minute / self.window_seconds)
            self.requests = max(0.0, self.requests - elapsed * self.requests_per_minute / self.window_seconds)
        self.updated = max(now, self.updated or now)

    def wait_seconds(self, tokens: int, now: float) -> float:
        """0 when a request of `tokens` fits now, else how long until it does."""
        self._drain(now)
        # More than the whole bucket fits only an empty one.
        excess = self.tokens if tokens > self.tokens_per_minute else self.tokens + tokens - self.tokens_per_minute
        token_wait = excess * self.window_seconds / self.tokens_per_minute
        request_wait = (self.requests + 1 - self.requests_per_minute) * self.window_seconds / self.requests_per_minute
        return max(0.0, token_wait, request_wait)

    def charge(self, tokens: int, now: float) -> list:
        """[charged at, tokens]."""
        self._drain(now)
        self.tokens += tokens
        self.requests += 1
        self.charged += 1
        return [now, tokens]

    def settle(self, entry: list, actual_tokens: int) -> None:
        with self.changed:
            self.tokens = max(0.0, self.tokens - (entry[1] - actual_tokens))
            self.settled_tokens += actual_tokens
            entry[1] = actual_tokens
            self.changed.notify_all()

    def try_acquire(self, tokens: int, max_wait_s: float, priority: float = 0.0,
                    clock=time.monotonic):
        """Like ``acquire`` but never blocks longer than ``max_wait_s``: returns the charge
        entry, or None -- without charging -- when the request cannot fit within that time
        (its computed wait is longer, or it queued behind higher-priority waiters past it)."""
        deadline = clock() + max_wait_s
        with self.changed:
            ticket = (-priority, next(self.arrivals))
            heapq.heappush(self.waiting, ticket)
            try:
                while True:
                    now = clock()
                    remaining = deadline - now
                    wait = remaining
                    if self.waiting[0] == ticket:
                        needed = self.wait_seconds(tokens, now)
                        if needed <= 0:
                            return self.charge(tokens, now)
                        if needed > remaining:
                            return None
                        wait = needed
                    elif remaining <= 0:
                        return None
                    self.changed.wait(wait)
            finally:
                self.waiting.remove(ticket)
                heapq.heapify(self.waiting)
                self.changed.notify_all()
