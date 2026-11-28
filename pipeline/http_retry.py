"""Retry vocabulary shared by every HTTP client in the corpus pipelines.

``retry_after_seconds`` / ``retry_after_from`` are the single reading of a
Retry-After header (RFC 9110 section 10.2.3: ``delay-seconds`` or an
``HTTP-date``), and ``exponential_backoff_seconds`` is the download backoff
curve, computed by tenacity's ``wait_exponential`` so no client carries its own
``min(cap, base * 2 ** n)``. What a client does while it waits (abort events,
cooldown queues, proxy rotation) stays with the client.
"""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from tenacity import RetryCallState, Retrying, wait_exponential

# 1.5 s, 3, 6, 12, 24, then 30 s: the curve the PDF download loops already used.
DOWNLOAD_BACKOFF = wait_exponential(multiplier=1.5, max=30)


def retry_after_seconds(value: str | None, *, default: float | None = 30.0,
                        now: datetime | None = None) -> float | None:
    """Seconds named by a Retry-After header (RFC 9110 section 10.2.3):
    ``delay-seconds`` or an ``HTTP-date``. Missing or unparsable -> ``default``
    (``None`` lets the caller tell "no answer" from "wait 0")."""
    text = (value or "").strip()
    if not text:
        return default
    if text.isdigit():
        return float(text)
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return default
    if when.tzinfo is None:  # HTTP-date is always GMT
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - (now or datetime.now(timezone.utc))).total_seconds())


def retry_after_from(source: Any, *, default: float | None = 30.0,
                     now: datetime | None = None) -> float | None:
    """``retry_after_seconds`` of the Retry-After header carried by ``source``:
    an httpx / requests / aiohttp response, or an exception holding one
    (``httpx.HTTPStatusError`` and ``requests.HTTPError`` expose ``.response``,
    ``aiohttp.ClientResponseError`` carries ``.headers``)."""
    for carrier in (getattr(source, "response", None), source):
        headers = getattr(carrier, "headers", None)
        if headers is not None:
            return retry_after_seconds(headers.get("Retry-After"), default=default, now=now)
    return default


def exponential_backoff_seconds(attempt: int, wait: Any = DOWNLOAD_BACKOFF) -> float:
    """Delay after failed attempt number ``attempt`` (1-based) under a tenacity wait strategy."""
    state = RetryCallState(retry_object=Retrying(), fn=None, args=(), kwargs={})
    state.attempt_number = attempt
    return float(wait(state))
