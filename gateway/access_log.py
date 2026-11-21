"""Structured per-request access log (W128/W131) — one JSON line per request
(authenticated OR anonymous) on the `gateway.access` logger → stdout.

EVERY request is recorded — operator requirement "所有的请求都要记录". Anonymous
hits (401s, public endpoints, pre-auth failures) log with `username=null` and the
client `ip` as the only identifier; authenticated hits add `user_id`/`username`/
`tier`. This is a complete edge access log, not a per-user-only billing trail.

Storage lives on the 4090 (Loki); BWH stays STATELESS — it only emits. The line
carries `username` (operator accepted PII on the shared box) + `tier` so Grafana
LogQL can count calls per user for ANY tier, including billing-exempt
internal/admin which never touch the transactions ledger.

Own handler + `propagate=False` so the line is bare JSON (no logging prefix),
clean for Loki/promtail to parse; an `"evt":"access"` marker selects these lines.
"""
from __future__ import annotations

import json
import logging
import sys

import httpx

_logger = logging.getLogger("gateway.access")
# Level/propagate set UNCONDITIONALLY — guarding these behind a handler check
# means a pre-attached handler (e.g. a test capture handler) leaves the logger at
# the default WARNING level and INFO lines get silently dropped.
_logger.setLevel(logging.INFO)
_logger.propagate = False
if not any(getattr(h, "_gateway_access", False) for h in _logger.handlers):
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    _h._gateway_access = True  # tag so we add our stdout handler exactly once
    _logger.addHandler(_h)


def emit(user, *, method: str, path: str, status: int,
         duration_ms: float, cost=0, ip: str | None = None) -> dict:
    """Log one access line to stdout and RETURN the record, so the middleware can
    ship the very same dict to Loki without rebuilding it (one source of truth).

    `user` may be None — anonymous requests are logged too (W131). All `getattr`s
    are None-safe so a missing user yields null user_id/username/tier while still
    capturing method/path/status/duration and the client `ip`.
    """
    tier = getattr(getattr(user, "tier", None), "value", None)
    record = {
        "evt": "access",
        "user_id": getattr(user, "id", None),
        "username": getattr(user, "username", None),
        "tier": tier,
        "ip": ip,
        "method": method,
        "path": path,
        "status": status,
        "cost": float(cost or 0),  # cost may be a Decimal (0.1) — JSON needs a number
        "duration_ms": duration_ms,
    }
    _logger.info(json.dumps(record, ensure_ascii=False))
    return record


async def ship(record: dict, *, base_url: str, token: str,
               timeout: float = 2.0, client: httpx.AsyncClient | None = None) -> bool:
    """Best-effort POST of one access record to the 4090's relay over the existing
    internal tunnel (`base_url` = the gateway's internal_base_url). Attaches the
    shared `X-Internal-Token`. NEVER raises and never blocks the user's request —
    a logging hop must not affect serving. Returns True on 2xx, else False.
    """
    if not base_url:
        return False
    owns = client is None
    client = client or httpx.AsyncClient(timeout=timeout)
    try:
        resp = await client.post(
            base_url.rstrip("/") + "/internal/ingest-log",
            json=record, headers={"X-Internal-Token": token or ""})
        return 200 <= resp.status_code < 300
    except Exception:
        _logger.debug("access-log ship failed (best-effort)", exc_info=True)
        return False
    finally:
        if owns:
            await client.aclose()
