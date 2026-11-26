"""Official MCP stdio acceptance for durable real match-paper jobs."""
from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN = REPO_ROOT / "mcp_plugin" / "retrieval_mcp.py"
REAL_BACKEND = os.environ.get(
    "RETRIEVAL_MCP_MATCH_JOB_BACKEND",
    "http://127.0.0.1:28080",
)
PAPER_ID = "test-real-acl-source"


def _require_real_smoke() -> None:
    if os.environ.get("RETRIEVAL_MCP_MATCH_JOB_REAL_SMOKE") != "1":
        pytest.skip("set RETRIEVAL_MCP_MATCH_JOB_REAL_SMOKE=1 for the real job gate")


def _text(reply: object) -> str:
    return "\n".join(
        getattr(part, "text", "")
        for part in (getattr(reply, "content", None) or [])
    )


async def _exercise(params: StdioServerParameters) -> None:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            receipt = await session.call_tool(
                "match_paper",
                {"paper_id": PAPER_ID, "top_k": 3},
            )
            assert not receipt.isError, _text(receipt)
            receipt_text = _text(receipt)
            match = re.search(r"job\s+(\d+)", receipt_text, flags=re.I)
            assert match, receipt_text
            task_id = int(match.group(1))
            for _ in range(120):
                status = await session.call_tool(
                    "match_paper_status",
                    {"task_id": task_id},
                )
                assert not status.isError, _text(status)
                status_text = _text(status)
                if "test-real-acl-neighbour" in status_text:
                    assert "test-real-kdd-neighbour" in status_text
                    return
                assert "failed" not in status_text.lower(), status_text
                await asyncio.sleep(0.1)
            raise AssertionError(f"match-paper job {task_id} did not complete")


async def _run() -> None:
    environment = dict(os.environ)
    environment["RETRIEVAL_API_URL"] = REAL_BACKEND
    environment["RETRIVAL_API_URL"] = REAL_BACKEND
    plugin_params = StdioServerParameters(
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
    await _exercise(plugin_params)
    await _exercise(repository_params)


def test_real_mcp_match_paper_job_completes_with_exact_results() -> None:
    _require_real_smoke()
    asyncio.run(_run())
