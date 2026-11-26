"""
SQLite FTS5 全文搜索存储，替代 Whoosh。

直接使用 retrieval.db（与 ORM 共库），零额外依赖。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional

from app.models.schemas import Chunk

_DEFAULT_DB = "data/db/retrieval.db"


class FtsStore:
    """SQLite FTS5 full-text search store with BM25 ranking.

    Shares the same retrieval.db as the ORM layer. The FTS5 virtual table
    is a lightweight shadow of the chunks table.
    """

    def __init__(self, db_path: str = _DEFAULT_DB):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        c = self._conn
        c.execute("""
            CREATE TABLE IF NOT EXISTS chunks_meta (
                chunk_id   TEXT PRIMARY KEY,
                doc_id     TEXT NOT NULL,
                level      INTEGER DEFAULT 2,
                title      TEXT,
                source_type TEXT,
                data_source TEXT
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_meta_doc ON chunks_meta(doc_id)")
        c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id,
                doc_id,
                content,
                tokenize='unicode61'
            )
        """)
        c.commit()

    def add_document(
        self,
        doc_id: str,
        chunks: list[Chunk],
        title: str | None = None,
        source_type: str | None = None,
        data_source: str | None = None,
    ) -> None:
        self.delete_document(doc_id)
        c = self._conn
        for chunk in chunks:
            chunk_id = chunk.id or f"{doc_id}_c{chunk.start_idx}"
            c.execute(
                "INSERT INTO chunks_meta (chunk_id, doc_id, level, title, source_type, data_source) VALUES (?,?,?,?,?,?)",
                (chunk_id, doc_id, chunk.level, title, source_type, data_source),
            )
            c.execute(
                "INSERT INTO chunks_fts (chunk_id, doc_id, content) VALUES (?,?,?)",
                (chunk_id, doc_id, chunk.content),
            )
        c.commit()

    def delete_document(self, doc_id: str) -> None:
        c = self._conn
        chunk_ids = [r[0] for r in c.execute("SELECT chunk_id FROM chunks_meta WHERE doc_id=?", (doc_id,)).fetchall()]
        if chunk_ids:
            placeholders = ",".join("?" * len(chunk_ids))
            c.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})", chunk_ids)
            c.execute("DELETE FROM chunks_meta WHERE doc_id=?", (doc_id,))
            c.commit()

    def search(
        self,
        query: str,
        top_k: int = 20,
        source_type: str | None = None,
        data_source: str | None = None,
        level: int | None = None,
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []

        tokens = [t for t in query.strip().split() if t]
        fts_query = " ".join(f"{t}*" for t in tokens)

        sql = """
            SELECT f.chunk_id, f.doc_id, f.content, bm25(chunks_fts) AS score,
                   m.title, m.level, m.source_type, m.data_source
            FROM chunks_fts f
            JOIN chunks_meta m ON f.chunk_id = m.chunk_id
            WHERE chunks_fts MATCH ?
        """
        params: list[Any] = [fts_query]

        if source_type is not None:
            sql += " AND m.source_type = ?"
            params.append(source_type)
        if data_source is not None:
            sql += " AND m.data_source = ?"
            params.append(data_source)
        if level is not None:
            sql += " AND m.level = ?"
            params.append(level)

        sql += " ORDER BY score LIMIT ?"
        params.append(top_k)

        rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "content": r["content"],
                "score": -r["score"],
                "title": r["title"],
                "level": r["level"],
                "source_type": r["source_type"],
                "data_source": r["data_source"],
            }
            for r in rows
        ]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks_meta").fetchone()[0]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
