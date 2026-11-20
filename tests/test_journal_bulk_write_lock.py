"""The journal bulk index must not hold SQLite's write lock for long (2026-11-16 incident).

SQLite has one write lock per database file. A journal_index task failed with
``database is locked`` on ``INSERT INTO journal_entries`` because another connection kept the
lock for longer than ``busy_timeout`` (30 s). These tests run the real repository/service on a
real WAL file with a second real connection (``tests/sqlite_lock_probe.py``) asking for the write
lock all the time; the longest wait of that connection is the longest hold of any other writer.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from app.services.journal.chunker import JournalTextChunk
from app.services.journal.repository import JournalRepository
from app.services.journal.schemas import JournalBulkDocItem
from app.services.journal.service import JournalService
from core.db.models import JournalChunk, JournalEntry
from tests.sqlite_lock_probe import WriteLockProbe, make_wal_session_factory


class SleepingEmbedServer:
    """A real local HTTP embedding endpoint that takes ``delay`` seconds per request."""

    def __init__(self, delay: float):
        outer = self
        self.requests = 0

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - http.server API
                texts = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["texts"]
                outer.requests += 1
                time.sleep(delay)
                body = json.dumps({"dense": [[float(len(t)), 1.0] for t in texts]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/embed"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def embed(self, texts, include_sparse):
        request = urllib.request.Request(
            self.url, data=json.dumps({"texts": list(texts)}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            dense = json.loads(response.read())["dense"]
        return dense, ([{"t": 1.0} for _ in texts] if include_sparse else None)


class RecordingVectorStore:
    def __init__(self):
        self.written = 0

    def upsert_chunk_groups(self, groups, on_batch=None):
        self.written += sum(len(group.chunks) for group in groups)
        return sum(len(group.chunks) for group in groups)

    def delete_by_entry_ids(self, entry_ids):
        return 0

    def delete_by_entry_id(self, entry_id):
        return 0


def _documents(count: int, chunks_per_doc: int) -> list[dict]:
    return [{
        "title": f"docs/d{i}.md",
        "body": f"# d{i}\n" + "\n".join(f"line {j} of document {i}" for j in range(chunks_per_doc)),
        "template_key": f"t{i}",
        "chunks": [JournalTextChunk(
            chunk_id=f"c{i}-{j}", entry_id="template", chunk_index=j, level=2,
            content=f"document {i} chunk {j} " + "word " * 80, heading=f"h{j}", token_count=90)
            for j in range(chunks_per_doc)],
    } for i in range(count)]


def test_bulk_doc_creation_gives_other_writers_the_lock_between_batches(tmp_path: Path):
    """900 documents x 20 chunks (18,000 chunk rows + FTS rows) are written in one pass; before the
    fix that was a single transaction holding the write lock for the whole insert (~0.3 ms per chunk,
    ~27 s for the 79,264-chunk JMLR scope), so a concurrent writer waited the whole time."""
    db = tmp_path / "journal.db"
    factory = make_wal_session_factory(db)
    repository = JournalRepository(session_factory=factory)
    documents = _documents(900, 20)

    with WriteLockProbe(db) as probe:
        started = time.monotonic()
        project_id, slug, rows = repository.create_doc_entries_bulk(
            project_slug="C:/proj/lock", owner_user_id="owner", client="mcp", documents=documents)
        took = time.monotonic() - started

    assert len(rows) == 900 and all(not row["deduped"] for row in rows)
    with factory() as session:
        assert session.query(JournalEntry).count() == 900
        assert session.query(JournalChunk).count() == 18_000
    assert probe.longest_wait < 2.0, (
        f"another writer waited {probe.longest_wait:.2f}s for the write lock while the bulk "
        f"create ran for {took:.2f}s")
    assert len(probe.waits) > 5, "the second writer must have been able to run during the write"


def test_bulk_doc_creation_is_idempotent_across_batches(tmp_path: Path):
    """Committing between batches must not change what a second identical request sees: every
    document is already there, deduped or (still pending) retried, never inserted twice."""
    factory = make_wal_session_factory(tmp_path / "journal.db")
    repository = JournalRepository(session_factory=factory)
    documents = _documents(150, 20)
    first = repository.create_doc_entries_bulk(
        project_slug="C:/proj/idem", owner_user_id="owner", client="mcp", documents=documents)[2]
    repository.mark_index_results_bulk([row["entry_id"] for row in first[:75]], success=True)

    again = repository.create_doc_entries_bulk(
        project_slug="C:/proj/idem", owner_user_id="owner", client="mcp", documents=documents)[2]

    assert [row["entry_id"] for row in again] == [row["entry_id"] for row in first]
    assert [row["deduped"] for row in again] == [True] * 75 + [False] * 75
    with factory() as session:
        assert session.query(JournalEntry).count() == 150


def test_embedding_phase_holds_no_write_lock(tmp_path: Path):
    """The embedding service is slow (a real HTTP server sleeping 3 s per request); no SQL write
    transaction of the bulk index may stay open across that call."""
    db = tmp_path / "journal.db"
    factory = make_wal_session_factory(db)
    repository = JournalRepository(session_factory=factory)
    with SleepingEmbedServer(delay=3.0) as embed_server:
        service = JournalService(
            repository=repository, vector_store=RecordingVectorStore(), embed_fn=embed_server.embed)
        files = []
        for i in range(12):
            body = f"# doc {i}\n\nsome text of document {i}\n\n## part\n\nmore text {i}"
            files.append(JournalBulkDocItem(
                title=f"d{i}.md", body=body,
                content_hash=hashlib.sha256(body.encode()).hexdigest(), size_bytes=len(body)))
        with WriteLockProbe(db) as probe:
            result = service.index_docs_bulk(
                project_slug="C:/proj/embed", owner_user_id="owner", files=files)
        assert embed_server.requests >= 1

    assert result["files"] == 12
    assert len(probe.waits) > 20, "the probe must have run during the 3 s embedding call"
    assert probe.longest_wait < 1.0, f"another writer waited {probe.longest_wait:.2f}s"
