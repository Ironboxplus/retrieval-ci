"""TDD: /api/dashboard/* operations endpoints.

Real HTTP test against live FastAPI + Qdrant + SQLite. No mocks — verifies
the actual contract that the React dashboard will consume.

Six endpoints under test:
  - /overview      KPI roll-up (papers/chunks/vectors/coverage)
  - /qdrant        Qdrant deep state (status, segments, payload_schema, search probe)
  - /sqlite        Per-table rows, db size, PRAGMA
  - /distribution  Aggregations for charts (chunk_levels, venues, years)
  - /system        FastAPI uptime + Qdrant ping + disk usage
  - /activity      Active task counts + recent task feed
"""
from __future__ import annotations

import os

import pytest
import requests


os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

API = os.environ.get("RETRIVAL_API", "http://127.0.0.1:8000")
TIMEOUT = 30


def _session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False
    return s


@pytest.fixture(scope="module")
def session():
    s = _session()
    # liveness probe before any test runs — fail fast if server down
    try:
        r = s.get(f"{API}/api/papers/stats", timeout=TIMEOUT)
        if r.status_code != 200:
            pytest.skip(f"server unhealthy: /api/papers/stats -> {r.status_code}")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"server unreachable: {e}")
    return s


# ---------- overview ----------
def test_overview_shape(session):
    r = session.get(f"{API}/api/dashboard/overview", timeout=TIMEOUT)
    assert r.status_code == 200, r.text
    body = r.json()
    assert {"ts", "uptime_s", "sql", "qdrant"}.issubset(body.keys())
    assert {"papers", "documents", "chunks"} == set(body["sql"].keys())
    for v in body["sql"].values():
        assert isinstance(v, int) and v >= 0
    # qdrant may carry error subkey if down — but in healthy state must have these
    if "error" not in body["qdrant"]:
        assert {"status", "points", "indexed_vectors", "segments"}.issubset(body["qdrant"].keys())
        assert body["qdrant"]["status"] in {"green", "yellow", "red", "grey"}


def test_overview_coverage_in_unit_range(session):
    body = session.get(f"{API}/api/dashboard/overview", timeout=TIMEOUT).json()
    cov = body.get("coverage")
    if cov is None:
        # qdrant down — acceptable
        return
    assert 0.0 <= cov <= 1.5  # allow >1 transiently during ingest replay


# ---------- vector_delta (F1-b) ----------
def test_overview_has_vector_delta_key(session):
    """vector_delta must always be present, even when null (Qdrant down)."""
    body = session.get(f"{API}/api/dashboard/overview", timeout=TIMEOUT).json()
    assert "vector_delta" in body


def test_overview_vector_delta_equals_points_minus_chunks(session):
    """vector_delta = qdrant.points - sql.chunks — the raw drift the F1 status
    capsule's 索引同步 card surfaces directly, instead of only a ratio."""
    body = session.get(f"{API}/api/dashboard/overview", timeout=TIMEOUT).json()
    if "error" in body["qdrant"]:
        pytest.skip("qdrant down for this run — vector_delta must be null, covered separately")
    assert body["vector_delta"] == body["qdrant"]["points"] - body["sql"]["chunks"]


def test_overview_vector_delta_null_when_qdrant_down(monkeypatch):
    """Unit-level check on the real route function (no live server needed):
    when Qdrant is unreachable, vector_delta must be null, not raise or default
    to 0 (0 would silently claim perfect sync)."""
    from app.api.routes import dashboard as dash

    def _boom(*_a, **_kw):
        raise RuntimeError("qdrant unreachable (simulated)")

    monkeypatch.setattr(dash, "_qdrant", _boom)
    body = dash.overview()
    assert body["vector_delta"] is None
    assert body["coverage"] is None


# ---------- qdrant ----------
def test_qdrant_detail_shape(session):
    r = session.get(f"{API}/api/dashboard/qdrant", timeout=TIMEOUT)
    assert r.status_code == 200, r.text
    body = r.json()
    expected = {
        "status", "points_count", "indexed_vectors_count", "segments_count",
        "payload_schema", "hnsw_config", "optimizer_config", "vectors",
    }
    assert expected.issubset(body.keys()), f"missing: {expected - body.keys()}"
    assert body["status"] in {"green", "yellow", "red", "grey"}
    assert isinstance(body["payload_schema"], dict)
    assert isinstance(body["points_count"], int)


def test_qdrant_on_disk_flags_present(session):
    """W22 audit: on_disk should be reported per-vector (dense/sparse/payload)."""
    body = session.get(f"{API}/api/dashboard/qdrant", timeout=TIMEOUT).json()
    v = body["vectors"]
    # All three keys must exist (value can be True/False/None, but key present)
    assert {"dense_on_disk", "sparse_on_disk", "payload_on_disk"}.issubset(v.keys())


def test_qdrant_search_probe_returns_latency(session):
    body = session.get(f"{API}/api/dashboard/qdrant", timeout=TIMEOUT).json()
    # probe may fail (None) but key must exist
    assert "search_probe_ms" in body
    if body["search_probe_ms"] is not None:
        assert body["search_probe_ms"] >= 0


# ---------- sqlite ----------
def test_sqlite_detail_shape(session):
    r = session.get(f"{API}/api/dashboard/sqlite", timeout=TIMEOUT)
    assert r.status_code == 200, r.text
    body = r.json()
    assert {"db_path", "db_size_bytes", "tables", "indexes", "pragma"}.issubset(body.keys())
    assert body["db_size_bytes"] > 0
    table_names = {t["name"] for t in body["tables"]}
    # core tables must exist
    assert {"papers", "documents", "chunks", "tasks"}.issubset(table_names)


def test_sqlite_pragma_journal_mode_is_wal(session):
    body = session.get(f"{API}/api/dashboard/sqlite", timeout=TIMEOUT).json()
    assert body["pragma"]["journal_mode"].lower() == "wal", \
        "WAL mode required for concurrent ingest + read"


def test_sqlite_indexes_include_compound(session):
    """Stage D added ix_chunks_doc_id_level (35× GROUP BY speedup)."""
    body = session.get(f"{API}/api/dashboard/sqlite", timeout=TIMEOUT).json()
    idx_names = {i["name"] for i in body["indexes"]}
    assert "ix_chunks_doc_id_level" in idx_names


# ---------- distribution ----------
def test_distribution_shape(session):
    r = session.get(f"{API}/api/dashboard/distribution", timeout=TIMEOUT)
    assert r.status_code == 200, r.text
    body = r.json()
    assert {"chunk_levels", "venues", "years"}.issubset(body.keys())


def test_distribution_chunk_levels_sum_matches_total(session):
    """chunks GROUP BY level should sum to overview.sql.chunks (consistency)."""
    dist = session.get(f"{API}/api/dashboard/distribution", timeout=TIMEOUT).json()
    overview = session.get(f"{API}/api/dashboard/overview", timeout=TIMEOUT).json()
    sum_levels = sum(dist["chunk_levels"].values())
    # Allow small drift since two separate snapshots
    assert abs(sum_levels - overview["sql"]["chunks"]) < 1000


def test_distribution_venues_nonempty(session):
    body = session.get(f"{API}/api/dashboard/distribution", timeout=TIMEOUT).json()
    assert len(body["venues"]) > 0
    # Each entry must have venue+papers fields
    for v in body["venues"][:3]:
        assert {"venue", "papers"} == set(v.keys())
        assert isinstance(v["papers"], int)


# ---------- system ----------
def test_system_detail_shape(session):
    r = session.get(f"{API}/api/dashboard/system", timeout=TIMEOUT)
    assert r.status_code == 200, r.text
    body = r.json()
    assert {"fastapi", "qdrant", "disk", "filesystem"}.issubset(body.keys())
    assert body["fastapi"]["alive"] is True
    assert isinstance(body["disk"], list) and len(body["disk"]) > 0


def test_system_disk_includes_qdrant_path(session):
    body = session.get(f"{API}/api/dashboard/system", timeout=TIMEOUT).json()
    paths = [d["path"] for d in body["disk"]]
    assert any("qdrant_server" in p for p in paths)
    assert any("/db" in p or p == "db" for p in paths)


def test_system_filesystem_has_free_bytes(session):
    body = session.get(f"{API}/api/dashboard/system", timeout=TIMEOUT).json()
    fs = body["filesystem"]
    if "error" in fs:
        pytest.skip(f"statvfs/disk_usage unavailable: {fs['error']}")
    assert fs["free_bytes"] > 0


# ---------- activity ----------
def test_activity_shape(session):
    r = session.get(f"{API}/api/dashboard/activity", timeout=TIMEOUT)
    assert r.status_code == 200, r.text
    body = r.json()
    assert {"active", "recent_tasks"}.issubset(body.keys())
    assert {"running", "pending"} == set(body["active"].keys())
    assert isinstance(body["recent_tasks"], list)


def test_activity_limit_param(session):
    body = session.get(f"{API}/api/dashboard/activity?limit=5", timeout=TIMEOUT).json()
    assert len(body["recent_tasks"]) <= 5
