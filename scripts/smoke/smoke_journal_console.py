#!/usr/bin/env python3
"""Real end-to-end smoke for the ACE Journal 管理台 + 关键词检索 + 索引清单 feature,
driven with THIS project's own Markdown as the test corpus.

It exercises, over an in-process ASGI transport (no GPU, no live Qdrant, no port):
  1. INCREMENTAL journal recording — every `##` section of journal.md/plan.md/… is
     recorded one entry at a time (via the MCP backend client — "MCP 后端").
  2. Keyword (FTS5) search — embedding-free, works while the embed endpoint is down.
  3. Index-inventory reporting — the repo's real .py files are reported, then read
     back as a user→host→project→dir tree ("MCP 前端" formatter included).
  4. Admin-console / dashboard stats — overview / users / cross-user search /
     inventory, asserting the counts match exactly what was ingested.
  5. Auth matrix — LAN internal-admin (200), gateway admin header (200),
     gateway non-admin header (403), off-LAN anonymous (401).

Run:
  .venv\\Scripts\\python.exe scripts\\smoke\\smoke_journal_console.py
Exit 0 = all checks passed.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from core.db.database import Base  # noqa: E402
from app.services.journal.repository import JournalRepository  # noqa: E402
from app.services.journal.service import JournalService  # noqa: E402
from app.services.index_catalog.repository import IndexCatalogRepository  # noqa: E402
from app.services.index_catalog.service import IndexCatalogService  # noqa: E402
from mcp_server.backend import (  # noqa: E402
    RetrievalClient, format_journal_search, format_journal_recent, format_inventory,
)

MD_FILES = ["journal.md", "plan.md", "INDEX.md", "CLAUDE.md", "env.md", "README.md"]
KEYWORDS = ["Qdrant", "gateway", "observability", "journal", "embedding"]

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)


def _fake_embed(texts, include_sparse):
    # Deterministic, GPU-free. Keyword path never calls this; record()'s vector path does.
    dense = [[float((len(t) % 13) + 1), 1.0, 0.5] for t in texts]
    sparse = [{"1": float(len(t.split()) or 1)} for t in texts] if include_sparse else None
    return dense, sparse


class _FakeVectorStore:
    """No live Qdrant — record()'s upsert is captured; semantic search returns nothing."""
    def __init__(self):
        self.upserts = 0

    def upsert_chunks(self, chunks, **kwargs):
        self.upserts += len(chunks)
        return len(chunks)

    def search(self, **kwargs):
        return []

    def delete_by_entry_id(self, entry_id):
        return 0


def _split_sections(text: str, fname: str):
    """Yield (title, body) per top-level `##` section; whole file if none."""
    parts = re.split(r"^(##\s+.+)$", text, flags=re.M)
    out = []
    if len(parts) == 1:
        body = text.strip()
        if body:
            out.append((fname, body))
        return out
    # parts = [pre, heading1, body1, heading2, body2, ...]
    pre = parts[0].strip()
    if pre:
        out.append((f"{fname} (前言)", pre))
    for i in range(1, len(parts), 2):
        title = parts[i].lstrip("#").strip()[:180] or fname
        body = (parts[i + 1] if i + 1 < len(parts) else "").strip()
        if body:
            out.append((title, body))
    return out


def _build_app():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_factory():
        session = session_local()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    journal = JournalService(
        repository=JournalRepository(session_factory=session_factory),
        vector_store=_FakeVectorStore(), embed_fn=_fake_embed,
    )
    index = IndexCatalogService(repository=IndexCatalogRepository(session_factory=session_factory))

    from app.api.routes import admin_console, index_catalog, journal as journal_routes
    journal_routes.get_journal_service = lambda: journal
    admin_console.get_journal_service = lambda: journal
    admin_console.get_index_catalog_service = lambda: index
    index_catalog.get_index_catalog_service = lambda: index

    app = FastAPI()
    app.include_router(journal_routes.router, prefix="/api/journal")
    app.include_router(index_catalog.router, prefix="/api/index")
    app.include_router(admin_console.router, prefix="/api/admin/console")
    return app


def _repo_py_inventory(limit=60):
    files = []
    for base in ("app", "core", "mcp_server"):
        for p in sorted((REPO / base).rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            rel = p.relative_to(REPO).as_posix()
            files.append({"rel_path": rel, "kind": "code", "collection": "code_chunks",
                          "language": "python", "chunk_count": max(1, p.stat().st_size // 400)})
            if len(files) >= limit:
                return files
    return files


async def main() -> int:
    print("== Building in-process app (temp SQLite, fake embed, fake vector store) ==")
    app = _build_app()
    lan = httpx.ASGITransport(app=app, client=("10.100.100.9", 5555))   # direct-4090 / plugin path
    wan = httpx.ASGITransport(app=app, client=("8.8.8.8", 5555))        # off-LAN anonymous

    admin_hdr = {"X-Retrival-User-Id": "gw-admin", "X-Retrival-Username": "gwadmin",
                 "X-Retrival-User-Tier": "admin"}
    paying_hdr = {"X-Retrival-User-Id": "gw-user", "X-Retrival-User-Tier": "paying"}

    # ---- 1. INCREMENTAL recording via the MCP backend client (direct-LAN internal-admin) ----
    print("\n== 1. Incremental journal recording from project MD (MCP backend) ==")
    mcp = RetrievalClient(client=httpx.AsyncClient(transport=lan, base_url="http://smoke"))
    recorded = 0
    per_file = {}
    for fname in MD_FILES:
        fp = REPO / fname
        if not fp.exists():
            continue
        sections = _split_sections(fp.read_text(encoding="utf-8", errors="ignore"), fname)
        for title, body in sections:
            resp = await mcp.record_journal(project_slug="retrival", client="smoke",
                                            source_kind="md", title=title, body=body,
                                            tags=["md", fname])
            if resp.get("entry", {}).get("entry_id"):
                recorded += 1
        per_file[fname] = len(sections)
    check("incremental record produced entries", recorded > 0, f"{recorded} entries from {len(per_file)} files")

    # ---- 2. Keyword (FTS5) search — embedding-free ----
    print("\n== 2. Keyword (FTS5) search ==")
    async with httpx.AsyncClient(transport=lan, base_url="http://smoke") as http:
        for term in KEYWORDS:
            r = await http.post("/api/journal/search",
                                json={"project_slug": "retrival", "query": term,
                                      "mode": "keyword", "top_k": 5, "include_chunks": True})
            total = r.json().get("total", 0)
            check(f"keyword '{term}' hits", r.status_code == 200 and total > 0, f"{total} hits")
        # MCP front formatter over a keyword search
        ks = await mcp.search_journal(project_slug="retrival", query="Qdrant", mode="keyword", top_k=3)
        check("MCP journal_search formatter", "journal entries" in format_journal_search(ks).lower())
        rec = await mcp.recent_journal_entries(project_slug="retrival", limit=5)
        check("MCP journal_recent formatter", "Recent journal" in format_journal_recent(rec))

    # ---- 3. Index-inventory report + read-back tree ----
    print("\n== 3. Index inventory report + tree (MCP front/back) ==")
    inv_files = _repo_py_inventory()
    rep = await mcp.report_index(host="smoke-host", project="retrival",
                                 root_dir=str(REPO), files=inv_files)
    check("report_index upserted", rep.get("upserted") == len(inv_files),
          f"upserted={rep.get('upserted')} of {len(inv_files)}")
    my_inv = await mcp.index_inventory()
    fmt = format_inventory(my_inv)
    check("MCP index_inventory formatter shows host", "smoke-host" in fmt)
    check("inventory totals.files matches report",
          my_inv.get("totals", {}).get("files") == len(inv_files),
          f"{my_inv.get('totals', {}).get('files')} vs {len(inv_files)}")

    # ---- 4. Admin console / dashboard stats ----
    print("\n== 4. Admin console dashboard stats (cross-user) ==")
    async with httpx.AsyncClient(transport=lan, base_url="http://smoke") as http:
        ov = (await http.get("/api/admin/console/overview")).json()
        check("overview journal.entries == recorded",
              ov["journal"]["entries"] == recorded, f"{ov['journal']['entries']} vs {recorded}")
        check("overview index.files == reported",
              ov["index"]["files"] == len(inv_files), f"{ov['index']['files']} vs {len(inv_files)}")
        users = (await http.get("/api/admin/console/users")).json()
        owners = {u["owner_user_id"] for u in users["users"]}
        check("users lists internal-admin owner", "internal-admin" in owners, str(owners))
        srch = (await http.post("/api/admin/console/journal/search",
                                json={"query": "gateway", "mode": "keyword", "top_k": 10})).json()
        check("admin cross-user keyword search", srch["total"] > 0, f"{srch['total']} hits")
        inv = (await http.get("/api/admin/console/index/inventory")).json()
        check("admin inventory tree totals.files", inv["totals"]["files"] == len(inv_files))

    # ---- 5. Auth matrix ----
    print("\n== 5. Auth matrix ==")
    async with httpx.AsyncClient(transport=lan, base_url="http://smoke") as http:
        check("LAN anonymous → 200 (internal-admin)",
              (await http.get("/api/admin/console/overview")).status_code == 200)
        check("gateway admin header → 200",
              (await http.get("/api/admin/console/overview", headers=admin_hdr)).status_code == 200)
        check("gateway non-admin header → 403",
              (await http.get("/api/admin/console/overview", headers=paying_hdr)).status_code == 403)
    async with httpx.AsyncClient(transport=wan, base_url="http://smoke") as http:
        check("off-LAN anonymous → 401",
              (await http.get("/api/admin/console/overview")).status_code == 401)

    await mcp.aclose()

    print("\n== RESULT ==")
    if _failures:
        print(f"FAILED ({len(_failures)}): {_failures}")
        return 1
    print("ALL CHECKS PASSED  (real project MD ingested incrementally; MCP front/back + "
          "dashboard stats + auth matrix verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
