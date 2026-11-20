"""Isolated stdio smoke for a retrieval-mcp release candidate against the real backend.

Usage: python scripts/maintenance/smoke_mcp_release.py <wheel path or package spec>
uvx runs the artifact with its own state dir and test ports (SMOKE_PORT, default 48101),
so the session daemon is untouched. Indexes a throwaway directory as a Journal doc scope
and a Code scope, checks no-op/incremental/removal/inventory filtering, then deletes both.
"""
import asyncio
import os
import re
import sys
import tempfile
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Tool output carries non-GBK glyphs; a print must never abort before the cleanup.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SOURCE = sys.argv[1]
work = Path(tempfile.mkdtemp(prefix="rmcp-release-smoke-"))
scope = work / "scope"
scope.mkdir()
(scope / "a.md").write_text("# alpha\nlean baseline smoke alpha\n", encoding="utf-8")
(scope / "b.md").write_text("# beta\nlean baseline smoke beta\n", encoding="utf-8")
(scope / "x.py").write_text("def smoke_alpha_release():\n    return 1\n", encoding="utf-8")
(scope / "y.py").write_text("def smoke_beta_release():\n    return 2\n", encoding="utf-8")
env = dict(os.environ)
env.update({
    "RETRIEVAL_API_URL": "http://10.100.100.88:18080",
    "RETRIEVAL_MCP_STATE_DIR": str(work / "state"),
    "RETRIEVAL_MCP_TEST_PORT": os.environ.get("SMOKE_PORT", "48101"),
    "RETRIEVAL_MCP_TEST_RAW_PORT": str(int(os.environ.get("SMOKE_PORT", "48101")) + 1),
    "NO_PROXY": "10.100.100.88,127.0.0.1,localhost",
    "PYTHONIOENCODING": "utf-8",
})
env.pop("RETRIEVAL_JOURNAL_CONVERSATIONS", None)
UVX = ["uvx", "--python", "3.12", "--no-cache", "--from", SOURCE]
params = StdioServerParameters(command=UVX[0], args=[*UVX[1:], "retrieval-mcp"], env=env, cwd=str(work))


def text(reply) -> str:
    return "\n".join(getattr(p, "text", "") for p in (getattr(reply, "content", None) or []))


async def wait_journal(session, out: str) -> None:
    if "job '" not in out:
        return
    job = out.split("job '", 1)[1].split("'", 1)[0]
    for _ in range(240):
        status = text(await session.call_tool("journal_index_status", {"job_id": job}))
        if "completed" in status or "failed" in status:
            assert "failed" not in status, status
            return
        await asyncio.sleep(1)
    raise AssertionError("journal job did not finish")


async def wait_code(session, out: str) -> None:
    m = re.search(r"job_id='([^']+)'", out)
    if not m:
        print("no job id in:", out[:300])
        return
    # Judge the bracketed job states, not words: since 0.5.13 the status also carries the
    # doc job's summary, whose counts read "(2 unchanged, 0 failed)" on success.
    for _ in range(240):
        st = text(await session.call_tool("index_code_status", {"job_id": m.group(1)}))
        if "not found" in st.lower():
            print("code job:", st[:200])
            return
        code = re.search(r"Job \S+ \[(\w+)\]", st)
        docs = re.search(r"docs: \[(\w+)\]", st)
        states = [s.group(1) for s in (code, docs) if s]
        if states and all(s in ("completed", "failed") for s in states):
            print("code job:", st[:300])
            assert "failed" not in states, st
            return
        await asyncio.sleep(1)
    raise AssertionError("code job did not finish")


async def main() -> int:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            print("tools", len((await s.list_tools()).tools))
            try:
                await exercise(s)
            finally:
                print(text(await s.call_tool("delete_journal_docs", {"dir": str(scope)}))[:160])
                print(text(await s.call_tool("delete_code", {"repo_root": str(scope)}))[:160])
    return 0


async def exercise(s) -> None:
    first = text(await s.call_tool("journal_index_dir", {"dir": str(scope)}))
    print("journal first:", first.splitlines()[0][:160])
    await wait_journal(s, first)
    second = text(await s.call_tool("journal_index_dir", {"dir": str(scope)}))
    print("journal no-op:", second.splitlines()[0][:160])
    assert "job '" not in second, second
    hits = text(await s.call_tool("journal_search", {"query": "lean baseline smoke beta", "scope_dir": str(scope)}))
    assert "b.md" in hits, hits[:800]
    out = text(await s.call_tool("index_code", {"repo_root": str(scope)}))
    print("index_code full:", out.splitlines()[0][:160])
    await wait_code(s, out)
    # index_code indexes the scope's documents too (0.5.13): a doc that only it has seen
    # must be searchable in the journal.
    (scope / "c.rst").write_text("gamma\n=====\n\nlean baseline smoke gamma rst\n", encoding="utf-8")
    out = text(await s.call_tool("index_code", {"repo_root": str(scope), "incremental": True}))
    print("index_code with a new doc:", out.splitlines()[0][:160])
    await wait_code(s, out)
    hits = text(await s.call_tool("journal_search", {"query": "lean baseline smoke gamma rst", "scope_dir": str(scope)}))
    assert "c.rst" in hits, hits[:800]
    (scope / "y.py").unlink()
    (scope / "x.py").write_text("def smoke_alpha_release_changed():\n    return 3\n", encoding="utf-8")
    out = text(await s.call_tool("index_code", {"repo_root": str(scope), "incremental": True}))
    print("index_code incremental:", out.splitlines()[0][:160])
    await wait_code(s, out)
    time.sleep(2)
    gone = text(await s.call_tool("search_code", {"query": "smoke_beta_release", "mode": "keyword", "repo_root": str(scope)}))
    changed = text(await s.call_tool("search_code", {"query": "smoke_alpha_release_changed", "mode": "keyword", "repo_root": str(scope)}))
    assert "y.py" not in gone, gone[:600]
    assert "x.py" in changed, changed[:600]
    inv = text(await s.call_tool("index_inventory", {"project": "rmcp-release-smoke"}))
    print("inventory filtered:", inv[:1200])
    assert "rmcp-release-smoke" in inv.replace("\\", "/").lower(), inv[:600]


sys.exit(asyncio.run(main()))
