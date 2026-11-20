"""TDD for the index-request audit (W172).

Every index request must be recorded faithfully at RECEIPT — LAN/MCP index calls
bypass the BWH gateway's access log, so without this they only existed in uvicorn's
ephemeral stdout and never reached the queryable audit log DB (Loki). The durable
guarantee is the synchronous bare-JSON log line (mirrors gateway/access_log.emit);
the Loki push is best-effort convenience on top.
"""
from __future__ import annotations

import json
import logging

import pytest

from app.services.index_audit import (
    INDEX_AUDIT_LOGGER_NAME,
    build_index_audit_record,
    emit_index_audit,
)


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def cap():
    logger = logging.getLogger(INDEX_AUDIT_LOGGER_NAME)
    h = _Capture()
    logger.addHandler(h)
    try:
        yield h
    finally:
        logger.removeHandler(h)


def test_build_record_marks_and_carries_fields():
    rec = build_index_audit_record(
        event="code_index_submitted", kind="code", owner_user_id="u1",
        project="/repo", host="h1", files=3, chunks=42, job_id="j1")
    assert rec["evt"] == "index_request"       # the marker LogQL selects on
    assert rec["event"] == "code_index_submitted"
    assert rec["kind"] == "code"
    assert rec["owner_user_id"] == "u1"
    assert rec["project"] == "/repo"
    assert rec["host"] == "h1"
    assert rec["files"] == 3
    assert rec["chunks"] == 42
    assert rec["job_id"] == "j1"


def test_build_record_omits_none_optionals():
    rec = build_index_audit_record(
        event="doc_index_reported", kind="doc", owner_user_id="u1",
        project="/docs", host="h1", files=2)
    # Required fields always present; unspecified optionals absent (not null noise).
    assert rec["files"] == 2
    assert "chunks" not in rec and "job_id" not in rec


def test_emit_writes_single_bare_json_line(cap):
    rec = build_index_audit_record(
        event="doc_index_reported", kind="doc", owner_user_id="u1",
        project="/docs", host="h1", files=2)
    out = emit_index_audit(rec)
    assert out is rec                          # returns the record (ship without rebuild)
    assert len(cap.lines) == 1
    parsed = json.loads(cap.lines[0])          # a bare JSON line, no logging prefix
    assert parsed["evt"] == "index_request" and parsed["kind"] == "doc"


def test_emit_logger_does_not_propagate():
    # propagate=False so the bare JSON line isn't duplicated up the logging tree
    # (mirrors gateway.access) — clean for Loki/promtail to parse.
    assert logging.getLogger(INDEX_AUDIT_LOGGER_NAME).propagate is False


def test_emit_never_raises_on_unserialisable(cap):
    # A record that can't be JSON-serialised must not blow up the caller (audit is
    # never allowed to break the index request it is recording).
    bad = build_index_audit_record(
        event="code_index_submitted", kind="code", owner_user_id="u1",
        project="/repo", host="h1", files=1)
    bad["oops"] = {1, 2, 3}                     # a set is not JSON-serialisable
    emit_index_audit(bad)                       # must not raise
