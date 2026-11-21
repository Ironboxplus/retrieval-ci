"""Prometheus instrumentation for the FastAPI compute backend (W128).

Attaches the prometheus-fastapi-instrumentator to the app and exposes a
`/metrics` endpoint (Prometheus exposition format) carrying per-handler request
counts + latency histograms. Prometheus on the 4090 scrapes this locally.

IMPORTANT: `setup_metrics(app)` MUST be called BEFORE the SPA catch-all route
(`GET /{spa_path:path}`) is registered in main.py — otherwise the catch-all
shadows `/metrics` and returns the SPA HTML with a 200 (the W114 pattern), and
Prometheus parses garbage.

This is compute-side only (no user/billing data — that lives in the BWH gateway).
Per-user audit lives in the gateway's structured access logs → Loki, not here.
"""
from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, Request

logger = logging.getLogger("core.observability")


def setup_metrics(app, *, endpoint: str = "/metrics", registry=None):
    """Instrument `app` and expose `endpoint` in Prometheus format.

    Returns the Instrumentator (handy for registering extra custom metrics).
    Best-effort: if the dependency is missing we log and no-op rather than break
    the backend boot — metrics are observability, never a hard dependency.

    `registry` defaults to the global prometheus registry (what the real backend
    wants). Pass an isolated ``CollectorRegistry`` to keep an app's metrics out of
    the shared registry — e.g. so tests are deterministic regardless of run order.
    """
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
    except ImportError:
        logger.warning("prometheus-fastapi-instrumentator not installed — /metrics disabled")
        return None

    kwargs = {"registry": registry} if registry is not None else {}
    instrumentator = Instrumentator(
        should_group_status_codes=True,      # 2xx/4xx/5xx buckets, bounded cardinality
        should_ignore_untemplated=True,      # don't explode labels on unmatched paths
        excluded_handlers=[endpoint],        # don't count scrapes of /metrics itself
        **kwargs,
    )
    instrumentator.instrument(app).expose(app, endpoint=endpoint, include_in_schema=False)
    logger.info("prometheus /metrics exposed at %s", endpoint)
    return instrumentator


# --- Per-user audit relay (W128 Phase 2) -----------------------------------
# The BWH gateway emits one JSON access line per authenticated request and ships
# it inward over the EXISTING internal tunnel (retrieval-xray:18000 → here). This
# relay forwards the line into the 4090's local Loki, so the per-user audit lives
# next to Prometheus/Grafana while BWH stays stateless (it only emits). `username`
# stays in the log LINE (high cardinality) — never a Loki stream label.

def build_loki_payload(line: str, *, labels: dict, ts_ns: int) -> dict:
    """Shape one pre-serialized log line into a Loki push-API body."""
    return {"streams": [{"stream": labels, "values": [[str(ts_ns), line]]}]}


# Distinct (target, exception type, message) triples already reported. Unbounded
# in principle, bounded in practice by how many ways one URL can fail.
_warned_loki_failures: set = set()


async def push_to_loki(line: str, *, loki_url: str, labels: dict,
                       timeout: float = 2.0) -> bool:
    """Best-effort push of one line to Loki's /loki/api/v1/push.

    Returns True on 2xx, False on any failure. NEVER raises — an audit hop must
    not break the relay request (which the stateless gateway will not retry).
    """
    import httpx

    payload = build_loki_payload(line, labels=labels, ts_ns=time.time_ns())
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(loki_url.rstrip("/") + "/loki/api/v1/push",
                                     json=payload)
        if 200 <= resp.status_code < 300:
            return True
        logger.warning("loki push non-2xx: %s %s", resp.status_code, resp.text[:200])
        return False
    except Exception as exc:
        # One line, once per distinct failure. This hop runs on every relayed
        # log record, so `exc_info=True` here meant a full traceback per record:
        # on `.88` that was ~9,367 error-shaped lines in two days, all of them
        # this, and it buried the real production errors underneath. The push
        # failing is expected and harmless; making the log unreadable is not.
        # Deduplication is by (target, exception type, message) so a genuinely
        # new failure still gets through — the full stack stays at DEBUG.
        signature = (loki_url, type(exc).__name__, str(exc))
        if signature not in _warned_loki_failures:
            _warned_loki_failures.add(signature)
            logger.warning(
                "loki push failed (best-effort, further identical failures suppressed): "
                "%s: %s: %s", loki_url, type(exc).__name__, exc)
        logger.debug("loki push failure detail", exc_info=True)
        return False


def make_log_relay_router(*, loki_url: str, labels: dict,
                          path: str = "/internal/ingest-log"):
    """Router exposing `path` (POST) that relays a gateway access record into Loki.

    Gated by the same `INTERNAL_TOKEN` middleware as `/api/*` (reachable only over
    the internal tunnel / LAN). Always returns 200 — a Loki failure surfaces as
    `{"ok": false}` but is never an error, so the gateway never retries the hop.
    """
    router = APIRouter()

    @router.post(path, include_in_schema=False)
    async def ingest_log(request: Request):
        try:
            body = await request.json()
        except Exception:
            return {"ok": False, "reason": "bad json"}
        line = (json.dumps(body, ensure_ascii=False)
                if isinstance(body, (dict, list)) else str(body))
        ok = await push_to_loki(line, loki_url=loki_url, labels=labels)
        return {"ok": ok}

    return router
