"""TDD API 契约：方案 X 后 paper_id 字符串 PK 直查，删 rsplit hack。

P0 锁定的回归：
  1. hash 类型 paper_id (NeurIPS) 不再被 _parse_paper_db_id 400 拒绝
  2. browse 返回的 paper_id 跟 papers.paper_id 列原值完全一致（不再拼接 venue-year-id）
  3. 数字尾段 paper_id (AAAI) 行为不退化

迁移前所有 hash 用例 RED，迁移后全 GREEN。
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
import requests

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

API = os.environ.get("RETRIVAL_API", "http://127.0.0.1:8000")
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "db" / "retrieval.db"
TIMEOUT = 30

# Real fixtures verified at test-authoring time (2026-05-17):
#   neurips hash key: doc_id=25698, Qdrant 106 points (indexed)
#   aaai numeric key: doc_id=35385, Qdrant 122 points (indexed)
HASH_PAPER_ID = "neurips-2022-002262941c9edfd472a79298b2ac5e17"
NUMERIC_PAPER_ID = "aaai-2025-31974"
NONEXISTENT_PAPER_ID = "fake-9999-9999999"


def _session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False
    return s


def test_server_reachable():
    # cold indexer lazy-init can take 7-10s on first call (journal 2026-05-17 09:30)
    r = _session().get(f"{API}/api/stats", timeout=30)
    assert r.status_code == 200


# ---------- index-status endpoint ----------

def test_index_status_accepts_hash_paper_id():
    """Hash paper_id (NeurIPS-style) must not be rejected by _parse hack."""
    r = _session().get(f"{API}/api/papers/{HASH_PAPER_ID}/index-status", timeout=TIMEOUT)
    assert r.status_code == 200, (
        f"hash paper_id must not 400; got {r.status_code}: {r.text[:300]}"
    )
    body = r.json()
    assert body["paper_id"] == HASH_PAPER_ID
    assert body["qdrant_points_count"] > 0
    assert body["indexed"] is True


def test_index_status_accepts_numeric_paper_id():
    r = _session().get(f"{API}/api/papers/{NUMERIC_PAPER_ID}/index-status", timeout=TIMEOUT)
    assert r.status_code == 200, f"got {r.status_code}: {r.text[:300]}"
    body = r.json()
    assert body["paper_id"] == NUMERIC_PAPER_ID
    assert body["qdrant_points_count"] > 0


def test_index_status_404_for_nonexistent():
    r = _session().get(f"{API}/api/papers/{NONEXISTENT_PAPER_ID}/index-status", timeout=TIMEOUT)
    assert r.status_code == 404


# ---------- POST /index endpoint ----------

def test_post_index_accepts_hash_paper_id():
    r = _session().post(f"{API}/api/papers/{HASH_PAPER_ID}/index", timeout=TIMEOUT)
    assert r.status_code in (200, 202), (
        f"hash paper_id POST must not 400; got {r.status_code}: {r.text[:300]}"
    )


def test_post_index_404_for_nonexistent():
    r = _session().post(f"{API}/api/papers/{NONEXISTENT_PAPER_ID}/index", timeout=TIMEOUT)
    assert r.status_code == 404


# ---------- GET /pdf endpoint ----------

def test_pdf_endpoint_accepts_hash_paper_id():
    """Even if PDF file missing on disk (404), endpoint must not 400 on hash key."""
    r = _session().get(f"{API}/api/papers/{HASH_PAPER_ID}/pdf", timeout=TIMEOUT, allow_redirects=False)
    assert r.status_code != 400, (
        f"hash paper_id PDF endpoint must not 400; got {r.status_code}: {r.text[:300]}"
    )


# ---------- /browse endpoint: paper_id 必须是 papers.paper_id 列原值 ----------

def test_browse_returns_canonical_paper_id_not_composed():
    """browse 不应再拼 venue-year-id；直接返回 papers.paper_id 列值。"""
    r = _session().get(f"{API}/api/papers/browse?venue=NeurIPS&year=2022&page_size=10", timeout=TIMEOUT)
    assert r.status_code == 200, f"browse failed: {r.text[:300]}"
    body = r.json()
    results = body.get("results", [])
    assert results, "browse should return at least one NeurIPS 2022 paper"

    # Validate every returned paper_id matches the SQL papers.paper_id column
    conn = sqlite3.connect(DB_PATH)
    try:
        for item in results:
            api_pid = item["paper_id"]
            api_doc_id = item["doc_id"]
            db_row = conn.execute(
                "SELECT paper_id FROM papers WHERE doc_id = ?", (api_doc_id,)
            ).fetchone()
            assert db_row, f"doc_id={api_doc_id} missing from SQL papers"
            sql_pid = db_row[0]
            assert api_pid == sql_pid, (
                f"browse returned composed paper_id {api_pid!r} but SQL truth is {sql_pid!r} "
                "(doc_id=" + str(api_doc_id) + ")"
            )
    finally:
        conn.close()


def test_browse_paper_id_roundtrip_via_index_status():
    """browse → paper_id → GET /papers/{paper_id}/index-status 必须能查到（拉通契约）。"""
    r = _session().get(f"{API}/api/papers/browse?venue=NeurIPS&year=2022&page_size=3", timeout=TIMEOUT)
    assert r.status_code == 200
    results = r.json().get("results", [])
    assert results
    for item in results:
        pid = item["paper_id"]
        r2 = _session().get(f"{API}/api/papers/{pid}/index-status", timeout=TIMEOUT)
        assert r2.status_code == 200, (
            f"browse → index-status roundtrip broken: paper_id={pid!r} → {r2.status_code}"
        )
        assert r2.json()["paper_id"] == pid
