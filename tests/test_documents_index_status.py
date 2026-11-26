"""TDD: GET /api/documents/{doc_id}/index-status contract.

Mirrors the paper version (`/api/papers/{paper_id}/index-status`) but keyed by
SQLite Document.id (integer PK) rather than the business paper_id. Used by the
frontend Documents page to render an "indexed" badge.

Contract:
    GET /api/documents/{doc_id}/index-status
    →  {document_id: int, sql_chunks_count: int, qdrant_points_count: int, indexed: bool}
    404 if document_id not in SQLite documents table.

Real system test: requires live FastAPI at http://localhost:8000.
"""
from __future__ import annotations

import os
import sqlite3
import pytest
import requests


API = os.environ.get("RETRIVAL_API", "http://localhost:8000")
DB_PATH = os.environ.get("RETRIVAL_DB", "data/db/retrieval.db")
TIMEOUT = 60


def _pick_doc_id_with_chunks() -> int:
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT doc_id FROM chunks GROUP BY doc_id ORDER BY COUNT(*) DESC LIMIT 1"
        ).fetchone()
        if not row:
            pytest.skip("no chunks in DB")
        return int(row[0])
    finally:
        conn.close()


def _max_doc_id() -> int:
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute("SELECT MAX(id) FROM documents").fetchone()
        return int(row[0]) if row and row[0] else 0
    finally:
        conn.close()


def test_server_reachable():
    r = requests.get(f"{API}/api/stats", timeout=5)
    r.raise_for_status()


def test_endpoint_exists():
    doc_id = _pick_doc_id_with_chunks()
    r = requests.get(f"{API}/api/documents/{doc_id}/index-status", timeout=TIMEOUT)
    assert r.status_code == 200, f"got {r.status_code}: {r.text[:200]}"


def test_response_shape():
    doc_id = _pick_doc_id_with_chunks()
    r = requests.get(f"{API}/api/documents/{doc_id}/index-status", timeout=TIMEOUT)
    r.raise_for_status()
    body = r.json()
    for key in ("document_id", "sql_chunks_count", "qdrant_points_count", "indexed"):
        assert key in body, f"missing key {key!r} in {body}"
    assert body["document_id"] == doc_id
    assert isinstance(body["sql_chunks_count"], int)
    assert isinstance(body["qdrant_points_count"], int)
    assert isinstance(body["indexed"], bool)


def test_chunked_doc_reports_nonzero_sql_chunks():
    doc_id = _pick_doc_id_with_chunks()
    r = requests.get(f"{API}/api/documents/{doc_id}/index-status", timeout=TIMEOUT)
    r.raise_for_status()
    body = r.json()
    assert body["sql_chunks_count"] > 0, body


def test_nonexistent_document_returns_404():
    bogus = _max_doc_id() + 999_999
    r = requests.get(f"{API}/api/documents/{bogus}/index-status", timeout=TIMEOUT)
    assert r.status_code == 404, f"expected 404, got {r.status_code}: {r.text[:200]}"


def test_indexed_flag_matches_qdrant_count():
    """Contract: indexed == (qdrant_points_count > 0)."""
    doc_id = _pick_doc_id_with_chunks()
    r = requests.get(f"{API}/api/documents/{doc_id}/index-status", timeout=TIMEOUT)
    r.raise_for_status()
    body = r.json()
    assert body["indexed"] == (body["qdrant_points_count"] > 0), body
