"""Conversation sync end to end: real deja -> real plugin tools -> real Uvicorn/SQLite/Qdrant.

Opt-in like ``test_journal_bulk_real_lifecycle.py`` (it starts qdrant.exe, Uvicorn and the
local BGE-M3 on CPU)::

    $env:RUN_REAL_CONVERSATION_SYNC='1'
    .venv/Scripts/python.exe -m pytest -q -s tests/test_mcp_conversations_real_backend.py

The plugin runs in a child process (``conversation_sync_mcp_client.py``) with deja pointed at
temp transcript roots.  HTTP traffic is counted from the SERVER's access log, never from
a client-side hook, so "zero requests" means the backend really saw none.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from mcp_plugin import retrieval_mcp_conversations as conv
from tests.conversation_transcripts import ClaudeSession, CodexRollout
from tests.test_journal_bulk_real_lifecycle import PYTHON, REPO, RealJournalBulkHarness

from tests.test_mcp_conversations import DEJA  # noqa: E402  pinned install
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_CONVERSATION_SYNC") != "1" or DEJA is None,
    reason="set RUN_REAL_CONVERSATION_SYNC=1 and install deja-vu for the real-backend conversation sync")

HOST = "convtest-host"
ACCESS_LINE = re.compile(r'"(GET|POST|PUT|PATCH|DELETE) (\S+) HTTP/[\d.]+"')


def _requests(harness: RealJournalBulkHarness) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2).split("?")[0]) for m in ACCESS_LINE.finditer(harness.api.text())]


def _conversation_rows(db: Path) -> dict[str, int]:
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT client, count(*) FROM journal_entries "
            "WHERE source_kind='doc' AND deleted_at IS NULL AND title LIKE '.conversations/%' "
            "GROUP BY client").fetchall()
    return dict(rows)


def _conversation_body(db: Path, title_suffix: str, client: str) -> str:
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT body FROM journal_entries WHERE source_kind='doc' AND deleted_at IS NULL "
            "AND client=? AND title LIKE ?", (client, f".conversations/%{title_suffix}")).fetchone()
    return row[0] if row else ""


def test_conversation_sync_against_real_backend(tmp_path: Path) -> None:
    harness = RealJournalBulkHarness(tmp_path)
    scope = tmp_path / "work" / "convproj"  # deja project "work/convproj"
    scope.mkdir(parents=True)
    claude_root, codex_root, state_dir = tmp_path / "claude" / "projects", tmp_path / "codex", tmp_path / "state"
    for directory in (claude_root, codex_root, state_dir):
        directory.mkdir(parents=True)
    claude = ClaudeSession(claude_root, str(scope), "claude-sess").turns(28)  # 7 segments
    CodexRollout(codex_root, str(scope), "codex-sess").turns(28)              # 7 segments

    env = {**harness.env,
           "RETRIEVAL_JOURNAL_CONVERSATIONS": "1", "RETRIEVAL_DEJA_BIN": DEJA,
           "RETRIEVAL_MCP_STATE_DIR": str(state_dir), "RETRIEVAL_HOST": HOST,
           "DEJA_CLAUDE_ROOT": str(claude_root), "DEJA_CODEX_ROOT": str(codex_root)}

    def drive(op: str, *extra: str) -> dict:
        done = subprocess.run(
            [str(PYTHON), "tests/conversation_sync_mcp_client.py", op, str(scope), *extra],
            cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=1500, check=False)
        assert done.returncode == 0, f"{op} rc={done.returncode}\nstdout={done.stdout}\nstderr={done.stderr[-4000:]}"
        return json.loads([line for line in done.stdout.splitlines() if line.startswith("{")][-1])

    def inventory_paths(project: str) -> set[str]:
        response = httpx.get(f"{harness.api_url}/api/index/inventory", timeout=30.0, params={
            "host": HOST, "project": project, "kind": "doc", "detail": "true"})
        response.raise_for_status()
        return {f["rel_path"] for u in response.json().get("users", []) for h in u.get("hosts", [])
                for p in h.get("projects", []) for d in p.get("dirs", []) for f in d.get("files", [])}

    try:
        harness.start()

        # 1. First sync: every segment of both harnesses, one bulk request per harness.
        before = len(_requests(harness))
        first = drive("sync")
        assert first["kick_line"] is None
        assert first["summary"]["segments_uploaded"] == 14 and first["summary"]["errors"] == []
        sent = _requests(harness)[before:]
        assert [r for r in sent if r == ("POST", "/api/journal/bulk-index")] == [("POST", "/api/journal/bulk-index")] * 2
        assert {p for _, p in sent if not p.startswith("/api/journal/bulk-index")} == set()
        assert all(p in {"/api/journal/bulk-index"} or p.startswith("/api/journal/bulk-index/jobs/")
                   for _, p in sent), sent
        assert _conversation_rows(harness.db_path) == {"claude": 7, "codex": 7}
        assert harness.sql_counts()["indexed_live_doc_entries"] == 14
        assert first["final_line"] is None

        # 2. Nothing changed: the backend must see no request at all.
        before = len(_requests(harness))
        second = drive("sync")
        assert _requests(harness)[before:] == []
        assert second["summary"]["segments_uploaded"] == 0 and second["summary"]["requests"] == 0
        assert second["final_line"] is None

        # 3. One appended message re-uploads only the last claude segment.
        claude.user("appended-marker question").assistant("appended-marker answer")
        before = len(_requests(harness))
        third = drive("sync")
        sent = _requests(harness)[before:]
        assert sent.count(("POST", "/api/journal/bulk-index")) == 1
        assert third["summary"]["segments_uploaded"] == 1
        assert "appended-marker" in _conversation_body(harness.db_path, "/007.md", "claude")
        assert _conversation_rows(harness.db_path) == {"claude": 7, "codex": 7}

        # 4. Search shows both harnesses and the status line; the sync itself stays silent.
        before = len(_requests(harness))
        searched = drive("search", "--query", "alpha3_5")
        sent = _requests(harness)[before:]
        assert sent == [("POST", "/api/journal/search")], sent
        assert "[doc/claude]" in searched["result"] and "[doc/codex]" in searched["result"]
        assert "Conversations" not in searched["result"]  # healthy sync: no extra context

        # 5. A recursive *.md journal_index_dir must not sweep .conversations/ away.
        (scope / "README.md").write_text("# readme\nrecursive sweep canary", encoding="utf-8")
        (scope / "docs").mkdir()
        (scope / "docs" / "guide.md").write_text("# guide\nanother canary", encoding="utf-8")
        assert _conversation_rows(harness.db_path) == {"claude": 7, "codex": 7}
        indexed = drive("index_dir")
        job = re.search(r"job '([^']+)'", indexed["result"])
        assert job, indexed["result"]
        deadline = time.monotonic() + 600
        while True:
            state = httpx.get(f"{harness.api_url}/api/journal/bulk-index/jobs/{job.group(1)}", timeout=30.0).json()
            if state["status"] in {"completed", "failed"} or time.monotonic() > deadline:
                break
            time.sleep(0.5)
        assert state["status"] == "completed", state
        assert _conversation_rows(harness.db_path) == {"claude": 7, "codex": 7}
        paths = inventory_paths(searched["scope"])
        assert {"README.md", "docs/guide.md"} <= paths
        assert sum(path.startswith(".conversations/") for path in paths) == 14, sorted(paths)
        after = drive("search", "--query", "alpha3_5")
        assert "[doc/claude]" in after["result"] and "[doc/codex]" in after["result"]
    finally:
        harness.stop()
