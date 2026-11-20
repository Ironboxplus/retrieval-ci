"""TDD: paper index status + index-on-demand endpoints.

Real system test against live server + Qdrant + SQLite. No mocks.

Fixtures (from real DB at test-authoring time):
  - INDEXED_PAPER_ID:   aaai-2025-39567 (doc_id=39598, 110 Qdrant points)
  - UNINDEXED_PAPER_ID: aaai-2022-1     (doc_id=32, 98 SQL chunks, 0 Qdrant)
"""
from __future__ import annotations

import os

import pytest
import requests


# Avoid HTTP_PROXY routing for localhost (env may set proxy 127.0.0.1:10808 for OpenReview downloads)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

API = os.environ.get("RETRIVAL_API", "http://127.0.0.1:8000")
TIMEOUT = 30
INDEX_TIMEOUT = 180  # re-index runs real chunk+embed pipeline


def _session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False  # bypass HTTP_PROXY env vars
    return s

# post-方案-X (2026-05-17): paper_id is now PRIMARY KEY (string) — no more
# rsplit hack. Fixtures are real papers.paper_id column values, verified
# against Qdrant payload.paper_id at authoring time.
INDEXED_PAPER_ID = "aaai-2025-31974"   # 122 Qdrant points
UNINDEXED_PAPER_ID = "aaai-2022-1"     # legacy: pre-Stage-C this had 98 SQL chunks / 0 Qdrant; post-Stage-C all papers indexed (test skipped)
NONEXISTENT_PAPER_ID = "fake-9999-9999999"


def test_server_reachable():
    # cold indexer lazy-init can take 7-10s (journal 2026-05-17 09:30)
    r = _session().get(f"{API}/api/stats", timeout=30)
    assert r.status_code == 200


def test_paper_index_status_endpoint_exists():
    """GET /api/papers/{paper_id}/index-status must respond 200."""
    r = _session().get(f"{API}/api/papers/{INDEXED_PAPER_ID}/index-status", timeout=TIMEOUT)
    assert r.status_code == 200, f"got {r.status_code}: {r.text[:300]}"


def test_paper_index_status_shape():
    r = _session().get(f"{API}/api/papers/{INDEXED_PAPER_ID}/index-status", timeout=TIMEOUT)
    body = r.json()
    for key in ("indexed", "sql_chunks_count", "qdrant_points_count", "doc_id", "paper_id"):
        assert key in body, f"missing key {key!r} in response: {body}"
    assert isinstance(body["indexed"], bool)
    assert isinstance(body["sql_chunks_count"], int)
    assert isinstance(body["qdrant_points_count"], int)


def test_indexed_paper_reports_nonzero_qdrant():
    r = _session().get(f"{API}/api/papers/{INDEXED_PAPER_ID}/index-status", timeout=TIMEOUT)
    body = r.json()
    assert body["qdrant_points_count"] > 0, (
        f"{INDEXED_PAPER_ID} should have Qdrant points, got {body}"
    )
    assert body["indexed"] is True


@pytest.mark.skip(reason="Stage C 100% coverage (2026-05-17) — no unindexed paper available")
def test_unindexed_paper_reports_zero_qdrant_but_sql_chunks():
    r = _session().get(f"{API}/api/papers/{UNINDEXED_PAPER_ID}/index-status", timeout=TIMEOUT)
    body = r.json()
    assert body["qdrant_points_count"] == 0, (
        f"{UNINDEXED_PAPER_ID} should have 0 Qdrant points, got {body}"
    )
    assert body["sql_chunks_count"] > 0, (
        f"{UNINDEXED_PAPER_ID} should still have SQL chunks, got {body}"
    )
    assert body["indexed"] is False


def test_nonexistent_paper_returns_404():
    r = _session().get(f"{API}/api/papers/{NONEXISTENT_PAPER_ID}/index-status", timeout=TIMEOUT)
    assert r.status_code == 404, f"expected 404 for nonexistent paper, got {r.status_code}"


def test_paper_index_endpoint_triggers_index():
    """POST /api/papers/{paper_id}/index must trigger (re)indexing and return status."""
    r = _session().post(f"{API}/api/papers/{INDEXED_PAPER_ID}/index", timeout=INDEX_TIMEOUT)
    assert r.status_code in (200, 202), f"got {r.status_code}: {r.text[:300]}"
    body = r.json()
    assert any(k in body for k in ("status", "task_id", "indexed", "ok", "chunks_indexed")), (
        f"unexpected shape: {body}"
    )


def test_paper_index_endpoint_404_for_nonexistent():
    r = _session().post(f"{API}/api/papers/{NONEXISTENT_PAPER_ID}/index", timeout=TIMEOUT)
    assert r.status_code == 404
