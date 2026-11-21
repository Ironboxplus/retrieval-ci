"""Operations dashboard endpoints — multi-dimensional system observability.

Each endpoint focuses on one slice of the system so the frontend can poll them
at different cadences (e.g. qdrant/overview at 10s, distribution at 10 min).
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import requests
from fastapi import APIRouter, HTTPException

from core.cache import DISTRIBUTION_TTL_SECONDS, TOTAL_COUNT_TTL_SECONDS, stats_cache
from core.timestamps import local_naive_iso, utc_naive_iso

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "papers")
DB_PATH = os.getenv("RETRIEVAL_DB_PATH", "data/db/retrieval.db")
DATA_DIR = Path(os.getenv("RETRIEVAL_DATA_DIR", "data"))
NO_PROXY = {"http": None, "https": None}

_PROCESS_START = time.time()


def _qdrant(method: str, path: str, **kw) -> dict:
    """Tight wrapper — Qdrant local server, no proxy."""
    r = requests.request(
        method,
        f"{QDRANT_URL}{path}",
        proxies=NO_PROXY,
        timeout=kw.pop("timeout", 15),
        **kw,
    )
    r.raise_for_status()
    return r.json()


def _open_sqlite() -> sqlite3.Connection:
    if not Path(DB_PATH).exists():
        raise HTTPException(503, f"SQLite DB not found: {DB_PATH}")
    return sqlite3.connect(DB_PATH)


_dir_size_cache: dict[str, tuple[float, int]] = {}
_DIR_SIZE_TTL = 300  # 5 min cache
_dir_size_pending: set[str] = set()

def _dir_size_scan(p: Path) -> None:
    """Background scan — updates cache when done."""
    key = str(p)
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    _dir_size_cache[key] = (time.time(), total)
    _dir_size_pending.discard(key)

def _dir_size_bytes(p: Path) -> int:
    """Return cached dir size. Triggers async refresh if stale. Never blocks."""
    import threading
    key = str(p)
    now = time.time()
    cached = _dir_size_cache.get(key)
    if cached and now - cached[0] < _DIR_SIZE_TTL:
        return cached[1]
    if not p.exists():
        return 0
    if key not in _dir_size_pending:
        _dir_size_pending.add(key)
        threading.Thread(target=_dir_size_scan, args=(p,), daemon=True).start()
    return cached[1] if cached else -1
    return total


def _sql_totals() -> dict:
    with _open_sqlite() as conn:
        c = conn.cursor()
        return {
            "papers": c.execute("SELECT COUNT(*) FROM papers").fetchone()[0],
            "documents": c.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            "chunks": c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        }


def _table_row_counts() -> list[dict]:
    with _open_sqlite() as conn:
        c = conn.cursor()
        tables = [
            r[0]
            for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        return [
            {"name": t, "rows": c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]}
            for t in tables
        ]


# ---------- overview (high-level KPIs) ----------
@router.get("/overview")
def overview() -> dict:
    """Single-shot KPI card data — papers/chunks/vectors/coverage + service state."""
    out: dict = {
        "ts": time.time(),
        "uptime_s": round(time.time() - _PROCESS_START, 1),
    }
    # COUNT(*) over the multi-million-row chunks table is a 1-14 s scan: served from the
    # shared stale-while-revalidate cache, never recomputed on the request path.
    out["sql"] = stats_cache.get(
        ("dashboard.sql_totals", DB_PATH), _sql_totals, ttl=TOTAL_COUNT_TTL_SECONDS
    )
    # Qdrant collection meta (1 call) + corpus point counts. SQL chunks cover
    # papers, arXiv and reviews, so `points` is the distinct total over the
    # papers/arxiv/reviews collections (just `papers` before the split).
    try:
        from core.storage.corpus_store import paper_store

        r = _qdrant("GET", f"/collections/{QDRANT_COLLECTION}")["result"]
        counts = paper_store().point_counts()
        legacy = counts.pop("legacy_copies")
        points = sum(counts.values()) - legacy
        out["qdrant"] = {
            "status": r["status"],
            "points": points,
            "collections": counts,
            "legacy_copies": legacy,
            "indexed_vectors": r["indexed_vectors_count"],
            "segments": r["segments_count"],
            "optimizer_status": r.get("optimizer_status"),
        }
        out["coverage"] = (
            round(points / out["sql"]["chunks"], 4)
            if out["sql"]["chunks"]
            else 0.0
        )
        # F1-b: the raw drift, not just the ratio — the 索引同步 status card shows
        # this number directly (e.g. "+1,337") instead of making the reader do
        # points/chunks math in their head. `coverage` stays for compatibility;
        # the frontend no longer renders it.
        out["vector_delta"] = points - out["sql"]["chunks"]
    except Exception as e:  # noqa: BLE001 — surface Qdrant outage in payload
        out["qdrant"] = {"error": str(e)}
        out["coverage"] = None
        out["vector_delta"] = None
    return out


# ---------- qdrant deep state ----------
@router.get("/qdrant")
def qdrant_detail() -> dict:
    """Qdrant internals: config, segments meta, search latency probe."""
    try:
        meta = _qdrant("GET", f"/collections/{QDRANT_COLLECTION}")["result"]
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"Qdrant unreachable: {e}") from e

    cfg = meta.get("config", {})
    params = cfg.get("params", {})
    out = {
        "status": meta["status"],
        "points_count": meta["points_count"],
        "indexed_vectors_count": meta["indexed_vectors_count"],
        "segments_count": meta["segments_count"],
        "vectors_count": meta.get("vectors_count"),
        "optimizer_status": meta.get("optimizer_status"),
        "payload_schema": {
            k: {"type": v.get("data_type"), "points": v.get("points")}
            for k, v in (meta.get("payload_schema") or {}).items()
        },
        # W39: dashboard 须展示 per-vector HNSW override 而非顶层默认。
        # Qdrant 多 named-vector collection（dense+sparse）下顶层
        # `hnsw_config` 只是 fallback default — 真正生效的是
        # `params.vectors.dense.hnsw_config` per-vector override。
        # effective_hnsw_config = 顶层 + dense override 合并。
        "hnsw_config": cfg.get("hnsw_config"),
        "effective_hnsw_config": {
            **(cfg.get("hnsw_config") or {}),
            **(params.get("vectors", {}).get("dense", {}).get("hnsw_config") or {}),
        },
        "optimizer_config": cfg.get("optimizer_config"),
        "quantization": cfg.get("quantization_config"),
        "vectors": {
            "dense_on_disk": params.get("vectors", {}).get("dense", {}).get("on_disk"),
            "sparse_on_disk": (
                params.get("sparse_vectors", {})
                .get("sparse", {})
                .get("index", {})
                .get("on_disk")
            ),
            "payload_on_disk": params.get("on_disk_payload"),
        },
    }
    # search latency probe — dense top_k=1 single vector
    try:
        probe_vec = [0.0] * 1024
        t0 = time.time()
        _qdrant(
            "POST",
            f"/collections/{QDRANT_COLLECTION}/points/query",
            json={"query": probe_vec, "using": "dense", "limit": 1, "with_payload": False},
            timeout=10,
        )
        out["search_probe_ms"] = round((time.time() - t0) * 1000, 1)
    except Exception as e:  # noqa: BLE001
        out["search_probe_ms"] = None
        out["search_probe_error"] = str(e)
    return out


# ---------- sqlite deep state ----------
@router.get("/sqlite")
def sqlite_detail() -> dict:
    """Per-table row counts, db file size, index list."""
    db_path = Path(DB_PATH)
    out: dict = {
        "db_path": str(db_path),
        "db_size_bytes": db_path.stat().st_size if db_path.exists() else 0,
        "wal_size_bytes": 0,
        "shm_size_bytes": 0,
    }
    for ext, key in [("-wal", "wal_size_bytes"), ("-shm", "shm_size_bytes")]:
        side = Path(str(db_path) + ext)
        if side.exists():
            out[key] = side.stat().st_size

    out["tables"] = stats_cache.get(
        ("dashboard.table_rows", DB_PATH), _table_row_counts, ttl=DISTRIBUTION_TTL_SECONDS
    )
    with _open_sqlite() as conn:
        c = conn.cursor()
        # index list
        indexes = c.execute(
            "SELECT name, tbl_name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY tbl_name, name"
        ).fetchall()
        out["indexes"] = [{"name": n, "table": t} for n, t in indexes]
        # PRAGMA stats
        out["pragma"] = {
            "journal_mode": c.execute("PRAGMA journal_mode").fetchone()[0],
            "synchronous": c.execute("PRAGMA synchronous").fetchone()[0],
            "page_size": c.execute("PRAGMA page_size").fetchone()[0],
            "cache_size": c.execute("PRAGMA cache_size").fetchone()[0],
        }
    return out


# ---------- distribution charts ----------
def _compute_distribution() -> dict:
    with _open_sqlite() as conn:
        c = conn.cursor()
        chunk_levels = {
            str(lvl): n
            for lvl, n in c.execute(
                "SELECT level, COUNT(*) FROM chunks GROUP BY level ORDER BY level"
            ).fetchall()
        }
        venues = [
            {"venue": v, "papers": n}
            for v, n in c.execute(
                "SELECT venue, COUNT(*) FROM papers GROUP BY venue ORDER BY COUNT(*) DESC"
            ).fetchall()
        ]
        years = [
            {"year": y, "papers": n}
            for y, n in c.execute(
                "SELECT year, COUNT(*) FROM papers GROUP BY year ORDER BY year"
            ).fetchall()
        ]
        venue_year = [
            {"venue": v, "year": y, "papers": n}
            for v, y, n in c.execute(
                "SELECT venue, year, COUNT(*) FROM papers GROUP BY venue, year ORDER BY venue, year"
            ).fetchall()
        ]
        data_sources = [
            {"data_source": ds or "(none)", "documents": n}
            for ds, n in c.execute(
                "SELECT data_source, COUNT(*) FROM documents GROUP BY data_source ORDER BY COUNT(*) DESC"
            ).fetchall()
        ]
    return {
        "chunk_levels": chunk_levels,
        "venues": venues,
        "years": years,
        "venue_year": venue_year,
        "data_sources": data_sources,
    }


@router.get("/distribution")
def distribution() -> dict:
    """Aggregations for charts: chunks-by-level, papers-by-venue, papers-by-year.

    The GROUP BY scans take 2-19 s on the production tables, so the result is served from
    the shared stale-while-revalidate cache (10 min TTL, background refresh).
    """
    return stats_cache.get(
        ("dashboard.distribution", DB_PATH), _compute_distribution, ttl=DISTRIBUTION_TTL_SECONDS
    )


# ---------- system health (service / disk) ----------
@router.get("/system")
def system_detail() -> dict:
    """Service uptime, disk usage by data subdir, simple liveness."""
    out: dict = {
        "process_uptime_s": round(time.time() - _PROCESS_START, 1),
        "ts": time.time(),
    }
    # FastAPI self-ping (always healthy — we're answering)
    out["fastapi"] = {"alive": True, "uptime_s": out["process_uptime_s"]}
    # Qdrant readyz with latency
    try:
        t0 = time.time()
        r = requests.get(
            f"{QDRANT_URL}/readyz", proxies=NO_PROXY, timeout=5
        )
        out["qdrant"] = {
            "alive": r.status_code == 200,
            "ping_ms": round((time.time() - t0) * 1000, 1),
            "status_code": r.status_code,
        }
    except Exception as e:  # noqa: BLE001
        out["qdrant"] = {"alive": False, "error": str(e)}
    # disk usage by key data subdirs
    subdirs = [
        "db",
        "indices/qdrant_server",
        "indices/whoosh",
        "papers/ccf_a_main_conference_corpus/pdf",
        "papers/ccf_a_main_conference_corpus/markdown",
        "uploads",
        "crawled",
        "remote_embed/vectors",
        "remote_embed/chunks",
    ]
    out["disk"] = [
        {
            "path": s,
            "bytes": _dir_size_bytes(DATA_DIR / s),
            "exists": (DATA_DIR / s).exists(),
        }
        for s in subdirs
    ]
    # free space on data dir's drive
    try:
        usage = os.statvfs(str(DATA_DIR)) if hasattr(os, "statvfs") else None
        if usage:
            out["filesystem"] = {
                "free_bytes": usage.f_bavail * usage.f_frsize,
                "total_bytes": usage.f_blocks * usage.f_frsize,
            }
        else:
            import shutil

            total, used, free = shutil.disk_usage(str(DATA_DIR.resolve()))
            out["filesystem"] = {"free_bytes": free, "total_bytes": total, "used_bytes": used}
    except Exception as e:  # noqa: BLE001
        out["filesystem"] = {"error": str(e)}
    return out


# ---------- activity feed ----------
@router.get("/activity")
def activity(limit: int = 20) -> dict:
    """Recent task events + active tasks summary (lightweight)."""
    with _open_sqlite() as conn:
        c = conn.cursor()
        # active / recent tasks
        tasks_running = c.execute(
            "SELECT COUNT(*) FROM tasks WHERE status='running'"
        ).fetchone()[0]
        tasks_pending = c.execute(
            "SELECT COUNT(*) FROM tasks WHERE status='pending'"
        ).fetchone()[0]
        # Schema: id, task_type, status, ..., created_at, started_at, completed_at, ..., last_heartbeat
        # "Recent activity" ranks by COALESCE(completed_at, started_at, created_at)
        # The raw columns are naive strings in two conventions (core/timestamps.py):
        # created_at is UTC, the rest are server-local. They leave here with offsets.
        recent = [
            {
                **dict(zip(["id", "task_type", "status", "progress", "total"], row[:5])),
                "created_at": utc_naive_iso(row[5]),
                **{
                    name: local_naive_iso(value)
                    for name, value in zip(["started_at", "completed_at", "last_heartbeat"], row[6:])
                },
            }
            for row in c.execute(
                """SELECT id, task_type, status, progress, total,
                          created_at, started_at, completed_at, last_heartbeat
                   FROM tasks
                   ORDER BY COALESCE(completed_at, started_at, created_at) DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        ]
    return {
        "active": {"running": tasks_running, "pending": tasks_pending},
        "recent_tasks": recent,
    }
