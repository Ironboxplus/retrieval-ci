"""Faithful audit of every index REQUEST at receipt (W172).

The BWH gateway's per-user access log (gateway/access_log.py) covers PUBLIC traffic,
but LAN / MCP index calls hit this compute backend directly and bypass the gateway,
so they never reached the queryable audit log DB — proven: uvicorn's systemd journal
had them, Loki had nothing. And IndexCatalog only records on job COMPLETION (W159), so
a submitted-then-failed / crashed index left no trace at all. This records each index
request the MOMENT it arrives, independent of downstream success, so no request is
silently dropped.

Mirrors gateway/access_log.emit(): a dedicated ``index.audit`` logger, ``propagate=
False``, one bare-JSON line to stdout — which the systemd journal captures durably and
which ALWAYS fires (synchronous, in-process). That stdout line is the no-drop guarantee.
A best-effort push into the local Loki (``push_index_audit``) puts the same record next
to Prometheus/Grafana for LogQL; a Loki outage can neither drop the record nor break the
index request. The ``"evt":"index_request"`` marker selects these lines in LogQL.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any

from core.observability import push_to_loki

INDEX_AUDIT_LOGGER_NAME = "index.audit"

_logger = logging.getLogger(INDEX_AUDIT_LOGGER_NAME)
# Set level/propagate UNCONDITIONALLY (a pre-attached test handler must not leave the
# logger at the default WARNING and drop INFO lines) — same reasoning as gateway.access.
_logger.setLevel(logging.INFO)
_logger.propagate = False
if not any(getattr(h, "_index_audit", False) for h in _logger.handlers):
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    _h._index_audit = True  # tag so the bare-JSON stdout handler is added exactly once
    _logger.addHandler(_h)


def build_index_audit_record(
    *,
    event: str,
    kind: str,
    owner_user_id: str,
    project: str,
    host: str,
    files: int,
    chunks: int | None = None,
    job_id: str | None = None,
    replace: bool | None = None,
) -> dict[str, Any]:
    """Shape one index-request audit record. ``evt`` is the fixed LogQL selector;
    ``event`` names the specific op (code_index_submitted / doc_index_reported).
    Optionals are omitted when None so the line carries no null noise."""
    record: dict[str, Any] = {
        "evt": "index_request",
        "event": event,
        "kind": kind,
        "owner_user_id": owner_user_id,
        "project": project,
        "host": host,
        "files": files,
    }
    if chunks is not None:
        record["chunks"] = chunks
    if job_id is not None:
        record["job_id"] = job_id
    if replace is not None:
        record["replace"] = replace
    return record


def emit_index_audit(record: dict[str, Any]) -> dict[str, Any]:
    """Write one bare-JSON audit line to the ``index.audit`` logger and RETURN the
    record (so a caller can push the very same dict to Loki without rebuilding it).

    NEVER raises — an audit hop must not break the index request it is recording. A
    record that fails to serialise is logged as a best-effort repr instead of blowing up.
    """
    try:
        line = json.dumps(record, ensure_ascii=False)
    except Exception:
        line = json.dumps({"evt": "index_request", "event": record.get("event"),
                            "serialize_error": True}, ensure_ascii=False)
    _logger.info(line)
    return record


def _loki_url() -> str:
    return os.environ.get("LOKI_PUSH_URL", "http://127.0.0.1:3100")


async def push_index_audit(record: dict[str, Any], *, loki_url: str | None = None) -> bool:
    """Best-effort push of one audit record into Loki under the index-audit stream.
    Reuses core.observability.push_to_loki (never raises). Returns True on 2xx."""
    try:
        line = json.dumps(record, ensure_ascii=False)
    except Exception:
        return False
    return await push_to_loki(
        line, loki_url=loki_url or _loki_url(),
        labels={"job": "index-audit", "host": record.get("host") or "unknown"})


# Hold in-flight fire-and-forget Loki pushes so the running loop can't GC them mid-flight
# (durability is the synchronous stdout emit in emit_index_audit, not this push).
_bg_tasks: set[asyncio.Task] = set()


def push_index_audit_bg(record: dict[str, Any]) -> None:
    """Schedule ``push_index_audit`` on the running loop without awaiting it — the index
    route stays fast and a Loki outage can't stall or break the request. Shared by every
    index route so the fire-and-forget bookkeeping lives in one place. NEVER raises: with
    no running loop (a sync caller) the push is dropped, upholding the module's contract
    that an audit hop must not break the index request it records."""
    try:
        task = asyncio.ensure_future(push_index_audit(record))
    except RuntimeError:
        return  # no running event loop — the durable stdout emit already fired
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
