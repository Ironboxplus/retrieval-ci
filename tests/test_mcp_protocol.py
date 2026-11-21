"""End-to-end test of the MCP protocol layer (client <-> server <-> backend).

This is the real thing: it launches the explicit direct MCP debug server as a subprocess and
talks to it with the MCP SDK's ClientSession over stdio JSON-RPC (the same way
Claude Code / Cursor do — we do NOT ship a client of our own). The FastAPI backend
is replaced by a tiny in-process stub HTTP server so the test is hermetic (no 4090,
no BGE-M3) yet still exercises the full chain:

    MCP ClientSession ──stdio/JSON-RPC──▶ mcp_server.server ──HTTP──▶ stub backend

Async calls are driven via asyncio.run() so no pytest-asyncio dependency is needed.
"""
from __future__ import annotations

import asyncio
import hashlib
import http.server
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp_plugin import retrieval_mcp_daemon as daemon

REPO_ROOT = Path(__file__).resolve().parent.parent

# 10 paper + 5 ACE + 6 journal/inventory + 7 private Code KB + 7 public-code tools.
EXPECTED_TOOLS = {
    "search_papers", "search_arxiv", "search_within_paper", "match_document", "match_paper",
    "match_paper_status",
    "get_paper_markdown", "download_paper_pdf", "download_papers",
    "list_conferences", "corpus_stats",
    "journal_record", "journal_search", "journal_recent", "journal_index_dir",
    "journal_index_status", "delete_journal_docs",
    "ace_context_aware", "ace_enhance_prompt", "ace_smart_generate",
    "ace_smart_reflect", "ace_playbook",
    "index_code", "index_code_status", "search_code",
    "find_duplicate_code", "delete_code", "index_inventory",
    "list_public_code_sources", "add_public_code_source", "check_public_code_updates",
    "index_public_code_sources", "set_public_code_update_policy",
    "public_code_job_status", "import_public_code_repo",
}


def test_plugin_and_repository_server_publish_the_same_full_tool_schema() -> None:
    """Names alone are insufficient: callers cache descriptions and argument schemas."""
    from mcp_plugin import retrieval_mcp as plugin
    from mcp_server import server as repository_server

    plugin_surface = daemon.canonical_tool_surface(asyncio.run(plugin.mcp.list_tools()))
    repository_surface = daemon.canonical_tool_surface(
        asyncio.run(repository_server.mcp.list_tools())
    )

    assert {entry["name"] for entry in plugin_surface} == EXPECTED_TOOLS
    assert plugin_surface == repository_surface
    assert daemon.canonical_tool_surface_digest(plugin_surface) == daemon.TOOL_SURFACE_DIGEST
    assert daemon.tool_surface_field_digests(plugin_surface) == daemon.TOOL_SURFACE_FIELD_DIGESTS


def _stub_vec(text: str, dim: int = 16) -> list[float]:
    """Deterministic bag-of-words embedding so the stub /api/embed is stable."""
    v = [0.0] * dim
    for w in text.lower().split():
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    return v


# In-memory playbook shared across stub requests (ACE tools are now server-backed).
_STUB_PLAYBOOK: list[dict] = []


class _StubHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence per-request logging
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/stats":
            self._send({"total_papers": 7, "total_chunks": 70,
                        "vector_index_size": 70, "embedding_model": "stub-model"})
        elif self.path == "/api/conferences":
            self._send({"conferences": [
                {"title": "StubConf", "venue_tier": "CCF-A", "years": [2024, 2025]}],
                "categories": {}})
        elif self.path == "/api/papers/stub-1/fulltext":
            self._send({"paper_id": "stub-1", "title": "Stub Paper",
                        "content": "# Stub Paper\n\nComplete Markdown body.\n"})
        elif self.path == "/api/papers/stub-1/markdown":
            self._send_bytes(
                b"# Stub Paper\n\nComplete Markdown body.\n",
                "text/markdown; charset=utf-8",
            )
        elif self.path == "/api/papers/stub-1/pdf":
            self._send_bytes(b"%PDF-1.7\nstub-pdf-body\n", "application/pdf")
        elif self.path.startswith("/api/journal/entries"):
            self._send({"entries": [{
                "entry_id": "journal_entry_stub",
                "project_id": 1,
                "project_slug": "retrival",
                "client": "mcp",
                "source_kind": "memory",
                "title": "Stub journal",
                "content_hash": "abc",
            }], "total": 1})
        elif self.path.startswith("/api/playbook/bullets"):
            self._send({"bullets": list(_STUB_PLAYBOOK), "total": len(_STUB_PLAYBOOK)})
        elif self.path == "/api/public-code/sources":
            self._send([{
                "source_id": "pion", "domain": "networking",
                "relative_path": "networking/pion", "active_revision": "abcdef123456",
                "active_manifest_file_count": 80, "active_chunk_count": 713,
                "active_vector_count": 713, "update_policy": "manual",
                "update_interval_minutes": None, "last_check_error": None,
            }])
        else:
            self._send({})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        if self.path == "/api/search":
            self._send({"papers": [{
                "paper_id": "stub-1", "doc_title": "Stub Paper",
                "max_score": 0.5, "metadata": {"venue": "StubConf", "year": 2024},
                "chunks": [{"content": "stub snippet about retrieval"}],
            }], "results": []})
        elif self.path == "/api/match-doc":
            self._send({"input_type": "text", "papers": [{
                "paper_id": "stub-1", "title": "Stub Paper", "venue": "StubConf",
                "year": 2024, "score": 0.6, "coverage": 2,
                "matched": [{"paper_chunk": "stub matched chunk", "sim": 0.6,
                             "query_chunk_idx": 0}],
            }]})
        elif self.path == "/api/embed":
            data = json.loads(raw or b"{}")
            texts = data.get("texts", [])
            self._send({"model": "stub", "dim": 16,
                        "dense": [_stub_vec(t) for t in texts], "sparse": None})
        elif self.path == "/api/journal/entries":
            self._send({
                "entry": {
                    "entry_id": "journal_entry_stub",
                    "project_id": 1,
                    "project_slug": "retrival",
                    "client": "mcp",
                    "source_kind": "memory",
                    "title": "Stub journal",
                    "content_hash": "abc",
                },
                "chunk_count": 2,
                "indexed": True,
                "deduped": False,
            })
        elif self.path == "/api/playbook/bullets":
            data = json.loads(raw or b"{}")
            row = {"id": len(_STUB_PLAYBOOK) + 1, "content": data.get("content", ""),
                   "section": data.get("section", "insights"), "domain": data.get("domain"),
                   "project_slug": data.get("project_slug", "retrival"),
                   "helpful": 0, "harmful": 0, "deduped": False}
            _STUB_PLAYBOOK.append(row)
            self._send(row)
        elif self.path == "/api/playbook/search":
            data = json.loads(raw or b"{}")
            token = (data.get("query", "").split() or [""])[0]
            hits = [dict(b, score=1.0) for b in _STUB_PLAYBOOK if token in b["content"]]
            self._send({"bullets": hits, "total": len(hits)})
        elif self.path == "/api/journal/search":
            self._send({
                "query": "storage",
                "project_slug": "retrival",
                "mode": "hybrid",
                "total": 1,
                "hits": [{
                    "score": 0.9,
                    "entry": {
                        "entry_id": "journal_entry_stub",
                        "project_slug": "retrival",
                        "client": "mcp",
                        "source_kind": "memory",
                        "title": "Stub journal",
                        "content_hash": "abc",
                    },
                    "best_chunk": {"chunk_id": "c1", "content": "journal storage snippet"},
                }],
            })
        else:
            self._send({})


def _start_stub():
    _STUB_PLAYBOOK.clear()  # fresh playbook per run
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _server_params(backend_url: str, playbook_db: str) -> StdioServerParameters:
    # Point the playbook at a temp DB so the protocol test never writes into the repo.
    env = {**os.environ, "RETRIVAL_API_URL": backend_url, "ACE_PLAYBOOK_DB": playbook_db}
    return StdioServerParameters(
        # Normal command-mode clients intentionally enter the lightweight
        # singleton proxy.  This protocol test exercises the explicit direct
        # FastMCP compatibility/debug server instead.
        command=sys.executable, args=["-m", "mcp_server.server", "stdio-server"],
        env=env, cwd=str(REPO_ROOT),
    )


async def _drive(backend_url: str, playbook_db: str, pdf_path: str, batch_dir: str):
    async with stdio_client(_server_params(backend_url, playbook_db)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = [t.name for t in (await session.list_tools()).tools]

            def text(res) -> str:
                return "\n".join(getattr(c, "text", "") for c in (res.content or []))

            search = text(await session.call_tool("search_papers", {"query": "retrieval", "top_k": 3}))
            stats = text(await session.call_tool("corpus_stats", {}))
            match = text(await session.call_tool("match_document",
                                                 {"text": "abstract about retrieval", "top_k": 3}))
            confs = text(await session.call_tool("list_conferences", {}))
            markdown = text(await session.call_tool(
                "get_paper_markdown", {"paper_id": "stub-1"}))
            pdf = text(await session.call_tool(
                "download_paper_pdf", {"paper_id": "stub-1", "save_path": pdf_path}))
            batch = text(await session.call_tool("download_papers", {
                "paper_ids": ["stub-1"],
                "save_dir": batch_dir,
                "artifacts": "both",
            }))
            # ACE path: reflect a lesson (writes playbook via stub /api/embed), then read it back.
            reflect = text(await session.call_tool(
                "ace_smart_reflect", {"code": "Always wait=True on Qdrant upsert"}))
            playbook = text(await session.call_tool("ace_playbook", {}))
            journal_record = text(await session.call_tool(
                "journal_record", {"project_slug": "retrival", "body": "stub journal body"}))
            journal_search = text(await session.call_tool(
                "journal_search", {"project_slug": "retrival", "query": "storage"}))
            journal_recent = text(await session.call_tool(
                "journal_recent", {"project_slug": "retrival"}))
            public_code_sources = text(await session.call_tool(
                "list_public_code_sources", {}))
            return (names, search, stats, match, confs, markdown, pdf, batch, reflect,
                    playbook, journal_record, journal_search, journal_recent,
                    public_code_sources)


def test_mcp_protocol_end_to_end(tmp_path):
    server, base_url = _start_stub()
    db = str(tmp_path / "playbook.db")
    pdf_path = str(tmp_path / "stub.pdf")
    batch_dir = str(tmp_path / "batch")
    try:
        (names, search, stats, match, confs, markdown, pdf, batch, reflect, playbook,
         journal_record, journal_search, journal_recent, public_code_sources) = asyncio.run(
            _drive(base_url, db, pdf_path, batch_dir))

        # MCP handshake + tool discovery worked — full surface present
        assert set(names) == EXPECTED_TOOLS

        # A real tool call round-tripped through the protocol to the stub backend
        assert "Stub Paper" in search
        assert "stub-1" in search
        assert "stub snippet" in search

        assert "7 papers" in stats
        assert "stub-model" in stats
        assert "Stub Paper" in match
        assert "StubConf" in confs
        assert markdown == "# Stub Paper\n\nComplete Markdown body.\n"
        assert "Downloaded PDF" in pdf
        assert Path(pdf_path).read_bytes().startswith(b"%PDF-")
        assert "2 artifact(s)" in batch
        assert (Path(batch_dir) / "stub-1.pdf").read_bytes().startswith(b"%PDF-")
        assert (Path(batch_dir) / "stub-1.md").read_text(
            encoding="utf-8") == "# Stub Paper\n\nComplete Markdown body.\n"

        # ACE loop round-tripped over stdio: reflect curated a bullet, playbook shows it
        assert "added new" in reflect
        assert "wait=True" in playbook
        assert "1 configured, 1 indexed" in public_code_sources
        assert "713 chunks" in public_code_sources
        assert "journal_entry_stub" in journal_record
        assert "journal storage snippet" in journal_search
        assert "Stub journal" in journal_recent
    finally:
        server.shutdown()
        server.server_close()
