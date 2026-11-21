"""TDD: /api/stats schema contract.

The frontend Search page stats widget consumes this. Key contracts:
- total_chunks must be true COUNT(*), NOT MAX(id) — PK gaps from deletes
  make MAX wildly overestimate.
- total_papers (separate from total_documents).
- coverage_ratio: how well Qdrant covers SQL chunks (0..1).

Real system test: requires live FastAPI at http://localhost:8000.
"""
from __future__ import annotations

import os
import sqlite3
import pytest
import requests


API = os.environ.get("RETRIVAL_API", "http://localhost:8000")
DB_PATH = os.environ.get("RETRIVAL_DB", "data/db/retrieval.db")
TIMEOUT = 30


def _get_stats() -> dict:
    r = requests.get(f"{API}/api/stats", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def test_server_reachable():
    _get_stats()


def test_required_keys_present():
    body = _get_stats()
    for key in ("total_documents", "total_papers", "total_chunks",
                "vector_index_size", "coverage_ratio",
                "embedding_model", "embedding_dimension", "supports_sparse"):
        assert key in body, f"missing {key!r}: {body}"


def test_total_chunks_is_true_count_not_max_id():
    """Regression: /api/stats.total_chunks must equal COUNT(*), not MAX(id).

    PK gaps from delete cascades historically made MAX(id) overshoot by ~16%.
    """
    body = _get_stats()
    conn = sqlite3.connect(DB_PATH)
    try:
        sql_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        sql_max = conn.execute("SELECT MAX(id) FROM chunks").fetchone()[0] or 0
    finally:
        conn.close()
    api_chunks = body["total_chunks"]
    assert api_chunks == sql_count, (
        f"total_chunks={api_chunks:,} but true COUNT={sql_count:,} (MAX(id)={sql_max:,}). "
        f"Regression: stats endpoint must report COUNT."
    )


def test_total_papers_matches_sql():
    body = _get_stats()
    conn = sqlite3.connect(DB_PATH)
    try:
        sql_papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    finally:
        conn.close()
    assert body["total_papers"] == sql_papers, body


def test_coverage_ratio_within_bounds():
    body = _get_stats()
    cov = body["coverage_ratio"]
    assert isinstance(cov, (int, float))
    assert 0 <= cov <= 1.5, f"coverage_ratio={cov} out of plausible range"
    expected = body["vector_index_size"] / body["total_chunks"] if body["total_chunks"] else 0
    # Backend rounds to 4 decimal places, so tolerate that quantum.
    assert abs(cov - expected) < 1e-3, f"got {cov}, expected {expected}"
