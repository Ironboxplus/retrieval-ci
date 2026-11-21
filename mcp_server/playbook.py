"""Project-local ACE playbook store (client-side, inside the MCP server).

The playbook is a small, per-project evolving set of "bullets" — reusable lessons:
strategies, domain facts, failure modes, conventions. It lives in a local SQLite
file (default `.ace/playbook.db` under the project), NOT in the shared paper
backend, keeping memory local and
per-project (project dir = natural isolation).

Embeddings come from the backend's BGE-M3 (`POST /api/embed`) via an injected async
embed function, so there is a single embedding model and none is loaded here.
Retrieval is brute-force cosine over the (small) bullet set — instant for the
hundreds-of-bullets scale a playbook lives at.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, List, Optional

logger = logging.getLogger("mcp_server.playbook")

# embed_fn(texts) -> one vector per text. Async because it crosses HTTP to the backend.
EmbedFn = Callable[[List[str]], Awaitable[List[List[float]]]]

DEFAULT_DEDUP_THRESHOLD = 0.92


@dataclass
class Bullet:
    """One evolving-playbook entry."""
    id: str
    content: str
    section: str = "insights"   # patterns / best_practices / insights / failure_mode
    helpful: int = 0
    harmful: int = 0
    domain: Optional[str] = None
    created_at: Optional[str] = None
    score: Optional[float] = None   # similarity; set only on search results
    deduped: bool = False           # True when add() merged into an existing bullet


def _bid(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class PlaybookStore:
    """SQLite-backed playbook. ``embed_fn`` is injected (backend BGE-M3 in prod,
    a fake in tests) — the store itself loads no model."""

    def __init__(self, db_path: str, embed_fn: EmbedFn,
                 dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD):
        self.db_path = db_path
        self.embed_fn = embed_fn
        self.dedup_threshold = dedup_threshold
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        d = os.path.dirname(self.db_path)
        if d:
            os.makedirs(d, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS bullets(
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    section TEXT DEFAULT 'insights',
                    helpful INTEGER DEFAULT 0,
                    harmful INTEGER DEFAULT 0,
                    domain TEXT,
                    created_at TEXT,
                    embedding TEXT
                )"""
            )

    async def _embed_one(self, text: str) -> List[float]:
        vecs = await self.embed_fn([text])
        if not vecs:
            raise RuntimeError("embed_fn returned no vectors")
        return vecs[0]

    def _row_to_bullet(self, row: sqlite3.Row, score: Optional[float] = None,
                       deduped: bool = False) -> Bullet:
        return Bullet(
            id=row["id"], content=row["content"], section=row["section"],
            helpful=row["helpful"], harmful=row["harmful"], domain=row["domain"],
            created_at=row["created_at"], score=score, deduped=deduped,
        )

    def _rows(self, domain: Optional[str] = None) -> List[sqlite3.Row]:
        with self._conn() as c:
            if domain:
                return c.execute("SELECT * FROM bullets WHERE domain=?", (domain,)).fetchall()
            return c.execute("SELECT * FROM bullets").fetchall()

    async def add_bullet(self, content: str, section: str = "insights",
                         domain: Optional[str] = None) -> Bullet:
        content = (content or "").strip()
        if not content:
            raise ValueError("bullet content must be non-empty")
        vec = await self._embed_one(content)

        # grow-and-refine: a near-duplicate reinforces the existing bullet
        best_row, best_score = None, -1.0
        for row in self._rows(domain=domain):
            emb = json.loads(row["embedding"]) if row["embedding"] else None
            if not emb:
                continue
            s = _cosine(vec, emb)
            if s > best_score:
                best_score, best_row = s, row
        if best_row is not None and best_score >= self.dedup_threshold:
            self.mark(best_row["id"], helpful=True)
            b = self._row_to_bullet(best_row, deduped=True)
            b.helpful += 1
            logger.info("playbook dedup-merge id=%s sim=%.3f helpful=%d",
                        b.id, best_score, b.helpful)
            return b

        bid = _bid(content)
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO bullets"
                "(id,content,section,helpful,harmful,domain,created_at,embedding)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (bid, content, section, 0, 0, domain, _now(), json.dumps(vec)),
            )
        logger.info("playbook add id=%s section=%s domain=%s", bid, section, domain)
        return Bullet(id=bid, content=content, section=section, domain=domain,
                      created_at=_now())

    async def search(self, query: str, top_k: int = 5,
                     domain: Optional[str] = None) -> List[Bullet]:
        if not query or top_k <= 0:
            return []
        qv = await self._embed_one(query)
        scored = []
        for row in self._rows(domain=domain):
            emb = json.loads(row["embedding"]) if row["embedding"] else None
            if not emb:
                continue
            scored.append((_cosine(qv, emb), row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._row_to_bullet(r, score=s) for s, r in scored[:top_k]]

    def mark(self, bid: str, helpful: bool = True) -> None:
        col = "helpful" if helpful else "harmful"
        with self._conn() as c:
            c.execute(f"UPDATE bullets SET {col}={col}+1 WHERE id=?", (bid,))

    def count(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM bullets").fetchone()[0]

    def list_bullets(self, domain: Optional[str] = None, limit: int = 100) -> List[Bullet]:
        return [self._row_to_bullet(r) for r in self._rows(domain=domain)[:limit]]
