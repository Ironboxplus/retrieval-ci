"""``core.net.token_bucket.TokenBucket``: the vendored GCRA limiter and its added ``try_acquire``."""
from __future__ import annotations

import time

import pytest

from core.net.token_bucket import TokenBucket

NOW = 1000.0


def _fixed():
    return NOW


def test_try_acquire_grants_and_charges_when_the_request_fits():
    bucket = TokenBucket(tokens_per_minute=6000, requests_per_minute=6000)

    entry = bucket.try_acquire(100, 0.0, clock=_fixed)

    assert entry == [NOW, 100]
    assert bucket.tokens == 100 and bucket.requests == 1 and bucket.charged == 1


def test_try_acquire_refuses_without_charging_when_the_wait_exceeds_the_limit():
    bucket = TokenBucket(tokens_per_minute=60, requests_per_minute=6000)   # 1 token a second
    assert bucket.acquire(50, clock=_fixed) == [NOW, 50]

    started = time.perf_counter()
    refused = bucket.try_acquire(50, 0.2, clock=_fixed)                    # needs 40 s of draining

    assert refused is None
    assert time.perf_counter() - started < 0.2
    assert bucket.tokens == 50 and bucket.charged == 1 and bucket.requests == 1
    assert bucket.waiting == []                                            # the refused ticket left the queue


def test_settle_lowers_the_charge_to_the_real_usage():
    bucket = TokenBucket(tokens_per_minute=6000, requests_per_minute=6000)
    entry = bucket.try_acquire(100, 0.0, clock=_fixed)

    bucket.settle(entry, 30)

    assert bucket.tokens == 30 and entry[1] == 30 and bucket.settled_tokens == 30


def test_a_request_bigger_than_the_whole_budget_goes_once_the_bucket_is_empty():
    bucket = TokenBucket(tokens_per_minute=1000, requests_per_minute=6000)

    assert bucket.acquire(5000, clock=_fixed) == [NOW, 5000]               # empty bucket: it goes, charged in full
    assert bucket.tokens == 5000

    assert bucket.wait_seconds(10, NOW) == pytest.approx((5000 + 10 - 1000) * 60 / 1000)
    assert bucket.try_acquire(10, 1.0, clock=_fixed) is None               # 240 s away
    assert bucket.wait_seconds(5000, NOW + 300) == pytest.approx(0.0, abs=1e-9)   # drained: another big one goes


def test_try_acquire_waits_for_the_budget_that_drains_within_the_limit():
    bucket = TokenBucket(tokens_per_minute=600_000, requests_per_minute=600)  # 10 requests a second
    for _ in range(600):
        bucket.charge(1, time.monotonic())

    started = time.perf_counter()
    entry = bucket.try_acquire(1, 1.0)

    waited = time.perf_counter() - started
    assert entry is not None
    assert 0.03 < waited < 1.0                                             # about 0.1 s of drain, not the limit


def test_acquire_still_blocks_until_the_request_fits():
    bucket = TokenBucket(tokens_per_minute=600_000, requests_per_minute=600)
    for _ in range(600):
        bucket.charge(1, time.monotonic())

    started = time.perf_counter()
    assert bucket.acquire(1) is not None
    assert time.perf_counter() - started > 0.03
