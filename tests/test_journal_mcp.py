"""TDD for MCP journal HTTP client helpers."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from mcp_server.backend import RetrievalClient
from mcp_server.backend import format_journal_search
import mcp_server.server as server


def _make_client(handler):
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://test")
    return RetrievalClient(base_url="http://test", client=http)


def test_record_journal_posts_expected_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "entry": {
                "entry_id": "journal_entry_test",
                "project_id": 1,
                "project_slug": "retrival",
                "client": "codex",
                "source_kind": "memory",
                "content_hash": "abc",
            },
            "chunk_count": 2,
            "indexed": True,
            "deduped": False,
        })

    client = _make_client(handler)
    response = asyncio.run(client.record_journal(
        project_slug="retrival",
        body="Work log",
        client="codex",
        source_kind="memory",
        title="Today",
        tags=["ace"],
    ))

    assert captured["path"] == "/api/journal/entries"
    assert captured["body"]["project_slug"] == "retrival"
    assert captured["body"]["tags"] == ["ace"]
    assert response["entry"]["entry_id"] == "journal_entry_test"


def test_search_journal_posts_expected_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "query": "storage",
            "project_slug": "retrival",
            "mode": "hybrid",
            "total": 0,
            "hits": [],
        })

    client = _make_client(handler)
    asyncio.run(client.search_journal(
        project_slug="retrival",
        query="storage",
        top_k=3,
        dedup_by_entry=False,
    ))

    assert captured["path"] == "/api/journal/search"
    assert captured["body"]["top_k"] == 3
    assert captured["body"]["dedup_by_entry"] is False


def test_repo_journal_search_schema_bounds_chunk_count():
    tool = {
        item.name: item for item in asyncio.run(server.mcp.list_tools())
    }["journal_search"]
    chunk_schema = tool.inputSchema["properties"]["chunk_limit"]

    assert chunk_schema["minimum"] == 1
    assert chunk_schema["maximum"] == 20
    assert "number of chunks" in chunk_schema["description"].lower()


def test_repo_journal_search_rejects_invalid_chunk_count_before_http(monkeypatch):
    class FailIfCalled:
        async def search_journal(self, **_kwargs):
            raise AssertionError("invalid chunk_limit must not reach the backend")

    monkeypatch.setattr(server, "_client", FailIfCalled())

    with pytest.raises(ValueError, match="number of chunks.*1 through 20"):
        asyncio.run(server.journal_search(query="cscc hook TODO", chunk_limit=16000))


def test_repo_journal_search_empty_guides_to_the_exact_scope(tmp_path, monkeypatch):
    class EmptyJournal:
        async def search_journal(self, **_kwargs):
            return {"hits": []}

    monkeypatch.setattr(server, "_client", EmptyJournal())
    out = asyncio.run(server.journal_search(query="q", scope_dir=str(tmp_path)))
    scope = server._scope_path(str(tmp_path))

    assert f"scope {scope!r}" in out
    assert "does not prove that scope_dir is wrong" in out
    assert f"journal_search(query='q', scope_dir={scope!r})" in out
    assert "index_inventory()" in out
    assert f"journal_index_dir(dir={scope!r})" in out


def test_journal_search_formatter_includes_returned_chunks_not_only_summary():
    """Journal MCP output must show the matched body chunk, not just entry metadata.

    Indexed Markdown docs always have a chunk-0 summary such as
    ``journal.md source_kind: doc client: mcp``. If the formatter only displays
    ``best_chunk`` and that chunk is the summary, the MCP caller cannot see the
    concrete paragraph that matched. The backend already returns ``chunks`` when
    ``include_chunks`` is true, so the formatter must surface them.
    """
    out = format_journal_search({
        "query": "ICML 2026",
        "hits": [{
            "score": 0.9,
            "entry": {
                "entry_id": "journal_entry_doc",
                "project_slug": "C:/Users/Arc/Prd/retrival",
                "client": "mcp",
                "source_kind": "doc",
                "title": "journal.md",
                "content_hash": "abc",
            },
            "best_chunk": {
                "chunk_id": "summary",
                "chunk_index": 0,
                "content": "journal.md source_kind: doc client: mcp",
            },
            "chunks": [
                {
                    "chunk_id": "summary",
                    "chunk_index": 0,
                    "content": "journal.md source_kind: doc client: mcp",
                },
                {
                    "chunk_id": "body",
                    "chunk_index": "body-12",
                    "heading": "ICML 2026",
                    "content": "ICML 2026 CDP manifest PDF download paragraph with concrete steps.",
                },
            ],
        }],
    })

    assert "ICML 2026 CDP manifest PDF download paragraph" in out


def test_journal_search_formatter_keeps_real_summary_heading_chunk():
    out = format_journal_search({
        "query": "real summary",
        "hits": [{
            "score": 0.9,
            "entry": {
                "entry_id": "journal_entry_doc",
                "project_slug": "C:/Users/Arc/Prd/retrival",
                "client": "mcp",
                "source_kind": "doc",
                "title": "summary-section.md",
                "content_hash": "abc",
            },
            "best_chunk": {
                "chunk_id": "body-summary",
                "chunk_index": 1,
                "heading": "Summary",
                "content": "REAL SUMMARY MATCH from the user's Markdown section.",
            },
            "chunks": [
                {
                    "chunk_id": "body-summary",
                    "chunk_index": 1,
                    "heading": "Summary",
                    "content": "REAL SUMMARY MATCH from the user's Markdown section.",
                },
                {
                    "chunk_id": "details",
                    "chunk_index": 2,
                    "heading": "Details",
                    "content": "Unrelated details should not replace the matched summary section.",
                },
            ],
        }],
    })

    assert "REAL SUMMARY MATCH" in out


def test_recent_journal_entries_uses_query_params():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"entries": [], "total": 0})

    client = _make_client(handler)
    asyncio.run(client.recent_journal_entries(project_slug="retrival", limit=7))

    assert seen["path"] == "/api/journal/entries"
    assert seen["query"]["project_slug"] == "retrival"
    assert seen["query"]["limit"] == "7"


class FakeJournalClient:
    def __init__(self):
        self.recorded = None
        self.searched = None
        self.recent = None

    async def record_journal(self, **kwargs):
        self.recorded = kwargs
        return {
            "entry": {
                "entry_id": "journal_entry_test",
                "project_slug": kwargs["project_slug"],
                "client": kwargs["client"],
                "source_kind": kwargs["source_kind"],
                "title": kwargs.get("title"),
                "content_hash": "abc",
            },
            "chunk_count": 2,
            "indexed": True,
        }

    async def search_journal(self, **kwargs):
        self.searched = kwargs
        return {
            "query": kwargs["query"],
            "project_slug": kwargs["project_slug"],
            "mode": "hybrid",
            "total": 1,
            "hits": [{
                "score": 0.9,
                "entry": {
                    "entry_id": "journal_entry_test",
                    "project_slug": kwargs["project_slug"],
                    "client": "codex",
                    "source_kind": "memory",
                    "title": "Storage decision",
                    "content_hash": "abc",
                },
                "best_chunk": {"content": "Use SQL as fact source.", "chunk_id": "c1"},
            }],
        }

    async def recent_journal_entries(self, **kwargs):
        self.recent = kwargs
        return {
            "entries": [{
                "entry_id": "journal_entry_test",
                "title": "Recent note",
                "client": "codex",
                "source_kind": "memory",
                "content_hash": "abc",
            }],
            "total": 1,
        }


def test_journal_mcp_tools_delegate_to_client(monkeypatch):
    fake = FakeJournalClient()
    monkeypatch.setattr(server, "_client", fake)

    async def run():
        recorded = await server.journal_record(
            body="Detailed work log",
            title="Work log",
            source_kind="memory",
            tags=["ace"],
        )
        searched = await server.journal_search(
            query="storage decision",
            top_k=3,
        )
        recent = await server.journal_recent(limit=5)
        return recorded, searched, recent

    recorded, searched, recent = asyncio.run(run())

    # scope is FORCED to the cwd abspath — the journal namespace is never hand-typed.
    scope = server._scope_path(".")
    assert fake.recorded["project_slug"] == scope
    assert fake.searched["project_slug"] == scope
    assert fake.recent["project_slug"] == scope
    assert fake.recorded["body"] == "Detailed work log"
    assert fake.recorded["tags"] == ["ace"]
    assert "journal_entry_test" in recorded
    assert fake.searched["query"] == "storage decision"
    assert fake.searched["include_chunks"] is True
    assert fake.searched["chunk_limit"] == 3
    assert "Storage decision" in searched
    assert fake.recent["limit"] == 5
    assert "Recent note" in recent


def test_journal_mcp_tools_accept_explicit_scope_dir(tmp_path, monkeypatch):
    fake = FakeJournalClient()
    monkeypatch.setattr(server, "_client", fake)
    scope_dir = str(tmp_path / "indexed-docs")

    async def run():
        await server.journal_record(body="Scoped note", scope_dir=scope_dir)
        await server.journal_search(query="scoped docs", scope_dir=scope_dir)
        await server.journal_recent(scope_dir=scope_dir)

    asyncio.run(run())

    scope = server._scope_path(scope_dir)
    assert fake.recorded["project_slug"] == scope
    assert fake.searched["project_slug"] == scope
    assert fake.recent["project_slug"] == scope


def test_journal_search_renders_more_than_default_chunks_when_requested(monkeypatch):
    fake = FakeJournalClient()

    async def search_journal(**kwargs):
        fake.searched = kwargs
        chunks = [
            {"chunk_id": f"c{idx}", "chunk_index": idx, "content": f"body chunk {idx}"}
            for idx in range(1, 7)
        ]
        return {
            "query": kwargs["query"],
            "project_slug": kwargs["project_slug"],
            "mode": "keyword",
            "total": 1,
            "hits": [{
                "score": 0.9,
                "entry": {
                    "entry_id": "journal_entry_test",
                    "project_slug": kwargs["project_slug"],
                    "client": "mcp",
                    "source_kind": "doc",
                    "title": "large.md",
                    "content_hash": "abc",
                },
                "best_chunk": chunks[0],
                "chunks": chunks,
            }],
        }

    fake.search_journal = search_journal
    monkeypatch.setattr(server, "_client", fake)

    out = asyncio.run(server.journal_search(
        query="body chunk",
        top_k=1,
        chunk_limit=5,
    ))

    assert "body chunk 5" in out
    assert "body chunk 6" not in out
    assert fake.searched["include_chunks"] is True
    assert fake.searched["chunk_limit"] == 5
