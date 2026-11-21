"""Tests for the standalone MCP server's HTTP client layer (mcp_server/client.py).

The MCP server is a thin process that wraps the running FastAPI backend over
HTTP — it touches no core code. These tests pin the request/response contract
(endpoints, payload shape, compact formatting) using httpx.MockTransport, so they
run hermetically without a live backend or the BGE-M3 model.

Async calls are driven via asyncio.run() so no pytest-asyncio dependency is needed.
"""
from __future__ import annotations

import asyncio
import json
import re

import httpx
import pytest

import mcp_server.backend as backend_module
from mcp_server.backend import (
    RetrievalClient, format_inventory, format_journal_search, format_match_results,
    format_public_code_job, format_public_code_search, format_public_code_sources,
    format_search_results,
)


def test_format_journal_search_surfaces_notice_when_empty():
    """An empty journal search with a server notice (path_glob 0-of-N) surfaces the notice
    instead of the bare 'No matching' — so the user learns their glob is scope-relative."""
    out = format_journal_search({"hits": [], "notice": "path_glob 'x/*.md' matched 0 of 3 doc(s)."})
    assert "matched 0 of 3 doc(s)" in out


def test_format_journal_search_empty_without_notice_is_plain():
    assert format_journal_search({"hits": []}) == "No matching journal entries found."


def _make_client(handler):
    """Build a RetrievalClient whose httpx layer is a MockTransport."""
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://test")
    return RetrievalClient(base_url="http://test", client=http)


def test_public_code_client_forwards_source_control_contracts():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, body))
        if request.url.path == "/api/public-code/sources":
            if request.method == "GET":
                return httpx.Response(200, json=[])
            return httpx.Response(201, json={"source": {"source_id": "pion"}, "index_job": None})
        if request.url.path.endswith("/update-policy"):
            return httpx.Response(200, json={"source_id": "pion", "update_policy": "scheduled"})
        return httpx.Response(202, json={"job_id": "job-1", "status": "queued"})

    client = _make_client(handler)
    asyncio.run(client.public_code_sources())
    asyncio.run(client.add_public_code_source(
        source_id="pion",
        domain="networking",
        upstream_url="https://github.com/pion/webrtc.git",
        relative_path="networking/pion",
        proxy_profile="local-10808",
        credential_profile="github-readonly",
        update_policy="scheduled",
        update_interval_minutes=1440,
        index_now=True,
    ))
    asyncio.run(client.check_public_code_updates(source_ids=["pion"], domains=[]))
    asyncio.run(client.index_public_code_sources(source_ids=["pion"], domains=[]))
    asyncio.run(client.set_public_code_update_policy(
        source_id="pion", update_policy="scheduled", update_interval_minutes=1440))

    assert calls[0] == ("GET", "/api/public-code/sources", None)
    assert calls[1][0:2] == ("POST", "/api/public-code/sources")
    assert calls[1][2]["proxy_profile"] == "local-10808"
    assert calls[1][2]["credential_profile"] == "github-readonly"
    assert "token" not in calls[1][2]
    assert calls[2] == ("POST", "/api/public-code/sources/check-updates", {"source_ids": ["pion"], "domains": []})
    assert calls[3] == ("POST", "/api/public-code/sources/index", {"source_ids": ["pion"], "domains": []})
    assert calls[4] == ("PUT", "/api/public-code/sources/pion/update-policy", {
        "update_policy": "scheduled",
        "update_interval_minutes": 1440,
    })


def test_public_code_client_forwards_search_and_job_queries():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, dict(request.url.params),
                      json.loads(request.content) if request.content else None))
        if request.url.path == "/api/public-code/search":
            return httpx.Response(200, json={"query": "rtp", "selected_source_ids": ["pion"], "hits": []})
        if request.url.path == "/api/public-code/jobs":
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={"job_id": "job-1", "status": "completed"})

    client = _make_client(handler)
    asyncio.run(client.search_public_code(
        query="rtp packetizer", source_ids=["pion"], domains=["networking"],
        top_k=7, mode="hybrid", language="go", rerank=True))
    asyncio.run(client.public_code_job("job-1"))
    asyncio.run(client.public_code_jobs(limit=17))

    assert calls[0][1] == "/api/public-code/search"
    assert calls[0][3] == {
        "query": "rtp packetizer", "source_ids": ["pion"], "domains": ["networking"],
        "top_k": 7, "mode": "hybrid", "language": "go", "rerank": True,
    }
    assert calls[1][1] == "/api/public-code/jobs/job-1"
    assert calls[2][1] == "/api/public-code/jobs"
    assert calls[2][2] == {"limit": "17"}


def test_public_code_formatters_return_dense_agent_friendly_context():
    sources = format_public_code_sources([{
        "source_id": "pion", "domain": "networking", "active_revision": "abcdef123456",
        "active_manifest_file_count": 80, "active_chunk_count": 713,
        "active_vector_count": 713, "update_policy": "scheduled",
        "update_interval_minutes": 1440, "relative_path": "networking/pion",
        "last_check_error": None,
    }])
    job = format_public_code_job({
        "job_id": "job-1", "operation": "index", "status": "running",
        "completed_sources": 0, "total_sources": 1, "progress": 0.25,
        "phase": "chunking", "current_source_id": "pion", "current_path": "rtp/packet.go",
        "files_done": 20, "files_total": 80, "chunks_done": 200, "errors": {},
    })
    search = format_public_code_search({
        "query": "rtp packetizer", "selected_source_ids": ["pion"],
        "requested_mode": "hybrid", "effective_mode": "hybrid",
        "rerank_applied": True, "candidate_count": 20, "rerank_candidate_count": 12,
        "hits": [{
            "score": 0.91, "source_id": "pion", "domain": "networking",
            "retrieval_score": 0.42, "reranker_score": 0.91,
            "revision": "abcdef123456", "path": "rtp/packet.go", "start_line": 10,
            "end_line": 30, "language": "go", "label": "Packetize", "code": "func Packetize() {}",
        }],
    })

    assert "1 configured, 1 indexed" in sources
    assert "713 chunks" in sources
    assert "job-1 [running]" in job
    assert "rtp/packet.go" in job
    assert "pion@abcdef1234" in search
    assert "func Packetize()" in search
    assert "retrieval_score=0.420" in search
    assert "reranker_score=0.910" in search
    assert "requested='hybrid', effective='hybrid'" in search
    assert "applied to 12 of 20" in search

    non_reranked = format_public_code_search({
        "query": "rtp packetizer",
        "hits": [{
            "score": 0.42, "retrieval_score": 0.42, "source_id": "pion",
            "revision": "abcdef123456", "path": "rtp/packet.go", "start_line": 10,
            "end_line": 30, "language": "go", "code": "func Packetize() {}",
        }],
    })
    assert "score=0.420" in non_reranked
    assert "retrieval_score=0.420" in non_reranked

    unavailable = format_public_code_search({
        "query": "Depends", "hits": [], "requested_mode": "keyword",
        "effective_mode": "unavailable", "rerank_applied": False,
        "candidate_count": 0, "rerank_candidate_count": 0,
        "notice": "keyword search is unavailable because sparse lexical retrieval is unavailable; it does not substitute dense results.",
    })
    assert "requested='keyword', effective='unavailable'" in unavailable
    assert "does not substitute dense" in unavailable


def test_search_papers_posts_expected_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "query": "retrieval augmented generation",
            "method": "hybrid",
            "total_results": 1,
            "papers": [{
                "paper_id": "neurips-2023-rag",
                "doc_id": "42",
                "doc_title": "Retrieval-Augmented Generation",
                "max_score": 0.91,
                "chunk_count": 1,
                "metadata": {"venue": "NeurIPS", "year": 2023},
                "chunks": [{"doc_id": "42", "chunk_id": "c1", "content": "RAG combines...",
                            "score": 0.91, "doc_title": "Retrieval-Augmented Generation",
                            "source": "", "metadata": {}}],
            }],
            "results": [],
        })

    client = _make_client(handler)
    resp = asyncio.run(client.search_papers(
        "retrieval augmented generation", top_k=5,
        venue="NeurIPS", year=2023, title_only=True))

    assert captured["path"] == "/api/search"
    body = captured["body"]
    assert body["query"] == "retrieval augmented generation"
    assert body["top_k"] == 5
    assert body["venue"] == "NeurIPS"
    assert body["year"] == 2023
    assert body["title_only"] is True
    assert resp["papers"][0]["paper_id"] == "neurips-2023-rag"


def test_search_within_paper_scopes_and_expands():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"papers": [], "results": []})

    client = _make_client(handler)
    asyncio.run(client.search_within_paper("acl-2024-foo", "ablation study", top_k=50))

    body = captured["body"]
    assert body["paper_ids"] == ["acl-2024-foo"]
    # In-paper search must expand all matched chunks, not collapse to best-per-paper.
    assert body["dedup_by_paper"] is False
    assert body["top_k"] == 50


def test_paper_artifact_client_fetches_full_markdown_and_downloads_pdf(tmp_path):
    full_text = "# Complete paper\n\n" + ("full body\n" * 300) + "END-OF-PAPER"
    pdf_bytes = b"%PDF-1.7\n" + (b"pdf-body\n" * 100)
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.url.path == "/api/papers/neurips-2025-artifact/fulltext":
            return httpx.Response(200, json={
                "paper_id": "neurips-2025-artifact",
                "title": "Artifact Paper",
                "content": full_text,
                "size": len(full_text),
            })
        if request.url.path == "/api/papers/neurips-2025-artifact/markdown":
            return httpx.Response(200, content=full_text.encode("utf-8"))
        if request.url.path == "/api/papers/neurips-2025-artifact/pdf":
            return httpx.Response(
                200,
                headers={"content-type": "application/pdf"},
                content=pdf_bytes,
            )
        return httpx.Response(404)

    client = _make_client(handler)
    destination = tmp_path / "artifact.pdf"
    markdown_destination = tmp_path / "artifact.md"

    async def scenario():
        markdown = await client.get_paper_markdown("neurips-2025-artifact")
        saved_markdown = await client.save_paper_markdown(
            "neurips-2025-artifact", save_path=str(markdown_destination))
        saved = await client.download_paper_pdf(
            "neurips-2025-artifact", save_path=str(destination))
        await client.aclose()
        return markdown, saved_markdown, saved

    markdown, saved_markdown, saved = asyncio.run(scenario())

    assert markdown["content"] == full_text
    assert markdown_destination.read_text(encoding="utf-8") == full_text
    assert saved_markdown["size"] == len(full_text.encode("utf-8"))
    assert destination.read_bytes() == pdf_bytes
    assert saved["path"] == str(destination.resolve())
    assert saved["size"] == len(pdf_bytes)
    assert ("GET", "/api/papers/neurips-2025-artifact/fulltext") in seen
    assert ("GET", "/api/papers/neurips-2025-artifact/markdown") in seen
    assert ("GET", "/api/papers/neurips-2025-artifact/pdf") in seen


def test_paper_artifact_client_reuses_idempotency_key_across_markdown_ranges(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(backend_module, "_PAPER_ARTIFACT_RANGE_BYTES", 16)
    content = ("# Ranged paper\n\n" + "range body\n" * 20).encode("utf-8")
    idempotency_keys: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        idempotency_keys.append(request.headers["x-idempotency-key"])
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers["range"])
        assert match is not None
        start, requested_end = map(int, match.groups())
        end = min(requested_end, len(content) - 1)
        return httpx.Response(206, headers={
            "content-range": f"bytes {start}-{end}/{len(content)}",
            "content-length": str(end - start + 1),
        }, content=content[start:end + 1])

    client = _make_client(handler)
    destination = tmp_path / "ranged.md"

    async def scenario():
        try:
            return await client.save_paper_markdown(
                "ranged-paper", save_path=str(destination))
        finally:
            await client.aclose()

    saved = asyncio.run(scenario())

    assert saved["size"] == len(content)
    assert destination.read_bytes() == content
    assert len(idempotency_keys) > 2
    assert len(set(idempotency_keys)) == 1
    assert idempotency_keys[0].startswith("paper-markdown-")


def test_paper_artifact_client_rejects_a_file_version_change(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_module, "_PAPER_ARTIFACT_RANGE_BYTES", 8)
    version_a = b"%PDF-A1" + b"a" * 24
    version_b = b"%PDF-B2" + b"b" * 24
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers["range"])
        assert match is not None
        start, requested_end = map(int, match.groups())
        content = version_a if requests == 1 else version_b
        etag = '"version-a"' if requests == 1 else '"version-b"'
        if requests > 1:
            assert request.headers["if-range"] == '"version-a"'
        end = min(requested_end, len(content) - 1)
        return httpx.Response(206, headers={
            "content-range": f"bytes {start}-{end}/{len(content)}",
            "content-length": str(end - start + 1),
            "etag": etag,
        }, content=content[start:end + 1])

    client = _make_client(handler)
    destination = tmp_path / "mixed-version.pdf"

    async def scenario() -> None:
        try:
            with pytest.raises(ValueError, match="changed during ranged download"):
                await client.download_paper_pdf(
                    "versioned-paper", save_path=str(destination))
        finally:
            await client.aclose()

    asyncio.run(scenario())

    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_paper_artifact_client_retries_an_interrupted_markdown_response(monkeypatch):
    monkeypatch.setattr(backend_module, "_PAPER_ARTIFACT_RETRY_BASE_SECONDS", 0)
    full_text = "# Retried paper\n\nComplete body after reconnect.\n"
    requests = 0
    idempotency_keys: list[str] = []

    class InterruptedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"paper_id":"retry","content":"partial'
            raise httpx.RemoteProtocolError("peer closed incomplete response")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        idempotency_keys.append(request.headers["x-idempotency-key"])
        if requests == 1:
            return httpx.Response(200, stream=InterruptedStream())
        return httpx.Response(200, json={
            "paper_id": "retry",
            "title": "Retried Paper",
            "content": full_text,
        })

    client = _make_client(handler)

    async def scenario():
        try:
            return await client.get_paper_markdown("retry")
        finally:
            await client.aclose()

    payload = asyncio.run(scenario())

    assert requests == 2
    assert len(set(idempotency_keys)) == 1
    assert idempotency_keys[0].startswith("paper-fulltext-")
    assert payload["content"] == full_text


def test_paper_artifact_client_retries_an_interrupted_pdf_stream(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_module, "_PAPER_ARTIFACT_RETRY_BASE_SECONDS", 0)
    pdf_bytes = b"%PDF-1.7\n" + (b"complete-body\n" * 100)
    requests = 0

    class InterruptedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield pdf_bytes[:128]
            raise httpx.RemoteProtocolError("peer closed incomplete response")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(200, stream=InterruptedStream())
        return httpx.Response(200, content=pdf_bytes)

    client = _make_client(handler)
    destination = tmp_path / "retried.pdf"

    async def scenario():
        try:
            return await client.download_paper_pdf(
                "retry-paper", save_path=str(destination))
        finally:
            await client.aclose()

    saved = asyncio.run(scenario())

    assert requests == 2
    assert saved["size"] == len(pdf_bytes)
    assert destination.read_bytes() == pdf_bytes
    assert not list(tmp_path.glob("*.part"))


def test_paper_artifact_client_retries_a_transient_server_error(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_module, "_PAPER_ARTIFACT_RETRY_BASE_SECONDS", 0)
    pdf_bytes = b"%PDF-1.7\n" + (b"healthy-after-retry\n" * 100)
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(502, text="compute restarting")
        return httpx.Response(200, content=pdf_bytes)

    client = _make_client(handler)
    destination = tmp_path / "server-retry.pdf"

    async def scenario():
        try:
            return await client.download_paper_pdf(
                "server-retry", save_path=str(destination))
        finally:
            await client.aclose()

    saved = asyncio.run(scenario())

    assert requests == 2
    assert saved["size"] == len(pdf_bytes)
    assert destination.read_bytes() == pdf_bytes


def test_paper_artifact_client_downloads_a_batch_via_single_artifact_routes(tmp_path):
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        paper_id = request.url.path.split("/")[-2]
        if request.url.path.endswith("/markdown"):
            return httpx.Response(200, content=f"# {paper_id}\n".encode())
        return httpx.Response(200, content=f"%PDF-1.7\n{paper_id}\n".encode())

    client = _make_client(handler)

    async def scenario():
        try:
            return await client.download_papers(
                ["p1", "p2"], save_dir=str(tmp_path))
        finally:
            await client.aclose()

    result = asyncio.run(scenario())

    assert len(result["paths"]) == 2
    assert result["errors"] == []
    assert (tmp_path / "p1.pdf").read_bytes().startswith(b"%PDF-")
    assert (tmp_path / "p2.pdf").read_bytes().startswith(b"%PDF-")
    assert set(seen) == {
        ("GET", "/api/papers/p1/pdf"),
        ("GET", "/api/papers/p2/pdf"),
    }


def test_match_document_uses_text_input_type():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"input_type": "text", "papers": [], "query_chunks": []})

    client = _make_client(handler)
    asyncio.run(client.match_document(
        "Some abstract text about graph neural networks.", top_k=10))

    assert captured["path"] == "/api/match-doc"
    assert captured["body"]["input_type"] == "text"
    assert captured["body"]["text"].startswith("Some abstract")
    assert captured["body"]["top_k"] == 10


def test_corpus_stats_and_conferences_are_get():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.url.path == "/api/stats":
            return httpx.Response(200, json={"total_papers": 94924, "total_chunks": 12753418})
        return httpx.Response(200, json={"conferences": [], "categories": {}})

    client = _make_client(handler)

    async def _run():
        return await client.corpus_stats(), await client.list_conferences()

    stats, confs = asyncio.run(_run())

    assert ("GET", "/api/stats") in seen
    assert ("GET", "/api/conferences") in seen
    assert stats["total_papers"] == 94924
    assert "conferences" in confs


def test_format_search_results_is_compact_and_cites():
    """Formatter must emit compact, agent-friendly citations (title + id + score),
    echoing FastContext's 'return compact evidence, not the whole payload' idea."""
    resp = {
        "papers": [{
            "paper_id": "neurips-2023-rag",
            "doc_title": "Retrieval-Augmented Generation",
            "max_score": 0.913,
            "metadata": {"venue": "NeurIPS", "year": 2023},
            "chunks": [{"content": "RAG combines a retriever with a generator to ground outputs."}],
        }],
    }
    text = format_search_results(resp)
    assert "Retrieval-Augmented Generation" in text
    assert "neurips-2023-rag" in text
    assert "NeurIPS" in text
    assert "2023" in text
    # snippet present
    assert "RAG combines" in text


def test_format_match_results_lists_papers_with_coverage():
    resp = {
        "input_type": "text",
        "papers": [
            {"paper_id": "p1", "title": "Paper One", "venue": "ICML", "year": 2024,
             "score": 0.8, "coverage": 3},
        ],
    }
    text = format_match_results(resp)
    assert "Paper One" in text
    assert "p1" in text
    assert "ICML" in text


def test_format_inventory_shows_project_freshness_and_commit():
    # W158: the inventory surfaces per-project freshness (newest file's indexed_at) and,
    # when the client recorded it, the indexed git commit.
    resp = {
        "totals": {"users": 1, "hosts": 1, "projects": 1, "files": 2, "chunks": 9},
        "users": [{"owner_user_id": "u1", "file_count": 2, "chunk_count": 9, "hosts": [
            {"host": "box", "file_count": 2, "chunk_count": 9, "projects": [
                {"project": "retrival", "root_dir": "/repo", "file_count": 2, "chunk_count": 9,
                 "dirs": [{"dir": ".", "file_count": 2, "chunk_count": 9, "files": [
                     {"rel_path": "a.py", "kind": "code", "chunk_count": 4,
                      "indexed_at": "2026-07-02T10:00:00", "metadata": {}},
                     {"rel_path": "b.py", "kind": "code", "chunk_count": 5,
                      "indexed_at": "2026-07-03T12:34:56", "metadata": {"git_commit": "abc1234"}},
                 ]}]}]}]}],
    }
    text = format_inventory(resp)
    assert "last indexed 2026-07-03 12:34:56" in text   # newest timestamp wins
    assert "@abc1234" in text                            # indexed revision surfaced


def test_format_inventory_uses_server_freshness_without_file_nodes():
    """Summary responses carry per-project last_indexed_at/last_commit on the project node
    (no per-file tree). The formatter must surface freshness from those server fields so it
    never needs the 10MB file listing just to show 'last indexed'."""
    project = {"project": "retrival", "root_dir": "/repo", "file_count": 2, "chunk_count": 9,
               "last_indexed_at": "2026-07-03T12:34:56", "last_commit": "abc1234", "dirs": []}
    host = {"host": "box", "file_count": 2, "chunk_count": 9, "projects": [project]}
    user = {"owner_user_id": "u1", "file_count": 2, "chunk_count": 9, "hosts": [host]}
    resp = {
        "totals": {"users": 1, "hosts": 1, "projects": 1, "files": 2, "chunks": 9},
        "users": [user],
    }
    text = format_inventory(resp)
    assert "last indexed 2026-07-03 12:34:56" in text
    assert "@abc1234" in text


def test_format_inventory_surfaces_preparing_code_scope_instead_of_calling_it_indexed():
    resp = {
        "totals": {"users": 1, "hosts": 1, "projects": 1, "files": 0, "chunks": 0},
        "users": [{"owner_user_id": "u1", "file_count": 0, "chunk_count": 0,
                   "hosts": [{"host": "desktop", "file_count": 0, "chunk_count": 0,
                              "projects": [{"project": "/repo/sing-box", "root_dir": "/repo/sing-box",
                                            "file_count": 0, "chunk_count": 0,
                                            "code_snapshot_state": "preparing",
                                            "code_active_index_job_id": "sing-box-job",
                                            "code_previous_completed_snapshot_available": False,
                                            "dirs": []}]}]}],
    }

    text = format_inventory(resp)

    assert "code snapshot: preparing" in text
    assert "job=sing-box-job" in text
    assert "completed snapshot unavailable" in text


def test_format_inventory_marks_interrupted_code_write_as_recovery_required():
    resp = {
        "totals": {"users": 1, "hosts": 1, "projects": 1, "files": 1, "chunks": 1},
        "users": [{"owner_user_id": "u1", "file_count": 1, "chunk_count": 1,
                   "hosts": [{"host": "desktop", "file_count": 1, "chunk_count": 1,
                              "projects": [{"project": "/repo/sing-box", "root_dir": "/repo/sing-box",
                                            "file_count": 1, "chunk_count": 1,
                                            "code_snapshot_state": "preparing",
                                            "code_active_index_job_id": "failed-write-job",
                                            "code_recovery_required": True,
                                            "dirs": []}]}]}],
    }

    text = format_inventory(resp)

    assert "recovery index required" in text
    assert "failed-write-job" in text


def test_index_inventory_client_forwards_detail_param():
    """RetrievalClient.index_inventory forwards detail=true only when asked (incremental
    baseline needs the file tree); the default summary call omits it."""
    urls: list[str] = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json={"users": [], "totals": {}})

    client = _make_client(handler)
    asyncio.run(client.index_inventory(project="p", kind="code", detail=True))
    assert "detail=true" in urls[-1]
    asyncio.run(client.index_inventory(project="p"))
    assert "detail" not in urls[-1]


def test_format_inventory_empty():
    assert "No indexed files" in format_inventory({"users": [], "totals": {}})
