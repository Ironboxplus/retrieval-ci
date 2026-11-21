"""Live acceptance for one-call/many-query Retrieval MCP searches.

This is deliberately not a hermetic contract test.  With
``RETRIEVAL_MCP_REAL_SMOKE=1`` it launches the current plugin through the official
MCP stdio client and sends every request to the current Retrieval backend.  It
therefore proves that a list query is only an MCP-round-trip optimization: every
per-query result must equal the existing single-query result byte-for-byte.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

import httpx
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN = REPO_ROOT / "mcp_plugin" / "retrieval_mcp.py"
REAL_BACKEND = os.environ.get("RETRIEVAL_MCP_REAL_BACKEND", "http://10.100.100.111:8000")
# `index_inventory` on 2026-09-15 reports this completed WSH-LAPTOP scope.  It is
# intentionally not mounted on this test host: search_code must exercise remote
# stored-snapshot lookup without starting a filesystem watcher or index job.
REMOTE_UNMOUNTED_CODE_SCOPE = "D:/survey"


def _require_real_smoke() -> None:
    if os.environ.get("RETRIEVAL_MCP_REAL_SMOKE") != "1":
        pytest.skip("set RETRIEVAL_MCP_REAL_SMOKE=1 to run the live MCP/backend acceptance")


def _text(reply: object) -> str:
    content = getattr(reply, "content", None) or []
    return "\n".join(getattr(part, "text", "") for part in content)


def _batch_segment(text: str, *, index: int, total: int, query: str) -> str:
    marker = f"Query {index + 1}/{total}: {query!r}\n"
    assert marker in text
    start = text.index(marker) + len(marker)
    if index + 1 == total:
        return text[start:]
    next_marker = f"\n\nQuery {index + 2}/{total}:"
    end = text.index(next_marker, start)
    return text[start:end]


def _stable_result_text(tool_name: str, text: str) -> str:
    """Remove only live watcher status that is appended after formatting code hits."""
    if tool_name != "search_code":
        return text
    return re.sub(r"\n\n(?:Search diagnostic:.*\n\n)?Index freshness:.*\Z", "", text, flags=re.S)


async def _remote_scope_job_ids() -> set[str]:
    """Read recent real backend jobs so the smoke proves it did not enqueue one."""
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{REAL_BACKEND}/api/code/jobs", params={"limit": 100})
        response.raise_for_status()
    return {
        str(job["job_id"])
        for job in response.json().get("jobs") or []
        if job.get("project") == REMOTE_UNMOUNTED_CODE_SCOPE
    }


async def _call_many_and_single() -> None:
    assert not os.path.isdir(REMOTE_UNMOUNTED_CODE_SCOPE), (
        "live smoke scope became locally mounted; refusing to start an auto watcher"
    )
    job_ids_before = await _remote_scope_job_ids()
    environment = dict(os.environ)
    environment["RETRIEVAL_API_URL"] = REAL_BACKEND
    environment["RETRIVAL_API_URL"] = REAL_BACKEND
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(PLUGIN), "stdio-server"],
        env=environment,
        cwd=str(REPO_ROOT),
    )
    repository_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server", "stdio-server"],
        env=environment,
        cwd=str(REPO_ROOT),
    )
    # These are intentionally current, non-fixture corpus queries.  The duplicate
    # ``FastMCP`` query catches accidental cross-query de-duplication as well.
    cases = (
        ("search_code", {
            "query": ["survey", "survey"], "repo_root": REMOTE_UNMOUNTED_CODE_SCOPE,
            "mode": "keyword", "top_k": 2,
        }),
        ("journal_search", {
            "query": ["match_doc", "match_doc"], "scope_dir": str(REPO_ROOT),
            "mode": "keyword", "top_k": 2,
        }),
        ("search_code", {
            "query": ["Hysteria", "Hysteria"], "sources": "public", "mode": "keyword",
            "top_k": 2,
        }),
        ("search_code", {
            "query": ["Hysteria", "Hysteria"], "repo_root": "*", "mode": "keyword",
            "top_k": 2,
        }),
        ("search_code", {
            "query": ["Hysteria", "Hysteria"], "sources": "all",
            "repo_root": str(REPO_ROOT), "mode": "keyword", "top_k": 2,
        }),
        ("search_papers", {
            "query": ["retrieval augmented generation", "retrieval augmented generation"],
            "venue": "ACL", "year": 2026, "top_k": 2,
        }),
        ("search_within_paper", {
            "query": ["retrieval", "retrieval"], "paper_id": "acl-2026-long-1", "top_k": 2,
        }),
        ("search_arxiv", {
            "query": ["delayed generalization grokking", "delayed generalization grokking"],
            "year": 2026, "top_k": 2,
        }),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            initialized = await session.initialize()
            instructions = getattr(initialized, "instructions", "") or ""
            assert "Retrieval-first rule" in instructions
            assert "your FIRST search must be a Retrieval MCP call" in instructions
            assert "search_code" in instructions
            assert "search_papers" in instructions
            assert "search_arxiv" in instructions
            assert "search_within_paper" in instructions
            assert "journal_search" in instructions
            assert "Batch independent queries as one list" in instructions
            invalid = await session.call_tool("search_code", {
                "query": ["survey", "   "], "repo_root": REMOTE_UNMOUNTED_CODE_SCOPE,
                "mode": "keyword", "top_k": 2,
            })
            assert invalid.isError
            assert "search_code.query[1] must not be blank" in _text(invalid)
            advertised = {tool.name: tool for tool in (await session.list_tools()).tools}
            assert "search_public_code" not in advertised
            assert "search_all_code" not in advertised
            # search_code routes by sources/repo_root to the three backend searches
            routed = {
                "public": {"sources": "public"},
                "every scope": {"repo_root": "*"},
                "private+public": {"sources": "all", "repo_root": str(REPO_ROOT)},
            }
            for label, extra in routed.items():
                reply = await session.call_tool("search_code", {
                    "query": "Hysteria", "mode": "keyword", "top_k": 2, **extra})
                assert not reply.isError, (label, _text(reply))
                routed[label] = _text(reply)
            assert routed["public"].startswith("Public code matches for 'Hysteria'")
            assert routed["every scope"].startswith("Federated code matches for 'Hysteria'")
            assert "public: not included" in routed["every scope"]
            assert "- public: 20 candidate(s)" in routed["private+public"]
            # A rejected request carries the backend's reason, not only "422" and an
            # MDN link: in a batch the failing item says which argument was wrong.
            rejected = _text(await session.call_tool("search_code", {
                "query": ["LLaDA", "DiffusionScheduler"], "sources": "public",
                "source_ids": ["no-such-source"], "top_k": 2,
            }))
            # ...and an unknown public name comes back with the names that exist.
            assert rejected.count(
                "ERROR: ValueError: search_code: Unknown public-code source IDs: "
                "no-such-source. Valid source_ids: ") == 2, rejected
            # only indexed sources are offered (pytorch is configured but not indexed)
            assert "llama-cpp" in rejected and "flash-attention" in rejected
            bad_domain = _text(await session.call_tool("search_code", {
                "query": "LLaDA", "sources": "all", "repo_root": str(REPO_ROOT),
                "domains": ["diffusion-llm"], "top_k": 2,
            }))
            assert "Unknown public-code domains: diffusion-llm. Valid domains: " in bad_domain
            assert "machine-learning" in bad_domain
            out_of_range = await session.call_tool("search_code", {
                "query": "LLaDA", "sources": "public", "top_k": 60})
            assert out_of_range.isError
            assert "top_k: Input should be less than or equal to 50" in _text(out_of_range)
            for tool_name, many_args in cases:
                schema = advertised[tool_name].inputSchema
                query_schema = schema["properties"]["query"]
                assert "anyOf" in query_schema, (tool_name, query_schema)
                assert "list of independent queries" in query_schema["description"]
                many_reply = await session.call_tool(tool_name, many_args)
                assert not many_reply.isError, _text(many_reply)
                # text only: no {"result": <same text>} copy of every reply
                assert many_reply.structuredContent is None, tool_name
                assert advertised[tool_name].outputSchema is None, tool_name
                many_text = _text(many_reply)
                for index, query in enumerate(many_args["query"]):
                    one_args = {**many_args, "query": query}
                    one_reply = await session.call_tool(tool_name, one_args)
                    assert not one_reply.isError, _text(one_reply)
                    one_text = _text(one_reply)
                    assert f"Query {index + 1}/2: {query!r}" in many_text
                    segment = _batch_segment(many_text, index=index, total=2, query=query)
                    assert _stable_result_text(tool_name, segment) == _stable_result_text(
                        tool_name, one_text)
    async with stdio_client(repository_params) as (read, write):
        async with ClientSession(read, write) as session:
            repository_initialized = await session.initialize()
            assert getattr(repository_initialized, "instructions", "") == instructions
    assert await _remote_scope_job_ids() == job_ids_before, (
        "the remote/unmounted search smoke must not enqueue a code-index job"
    )


def test_real_mcp_stdio_multi_query_is_per_query_equivalent() -> None:
    _require_real_smoke()
    asyncio.run(_call_many_and_single())
