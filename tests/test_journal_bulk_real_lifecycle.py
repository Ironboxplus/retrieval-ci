"""No-mock, subprocess black-box lifecycle for Journal's directory bulk job.

The test is intentionally opt-in because it starts ``qdrant.exe``, Uvicorn, three
MCP client subprocesses, a real local BGE-M3 model, and performs a real Qdrant
outage/recovery.  Run it with::

    $env:RUN_REAL_JOURNAL_BULK='1'
    .venv/Scripts/python.exe -m pytest -q -s tests/test_journal_bulk_real_lifecycle.py

All state lives below pytest's temporary directory.  It never opens the repository
database, the ordinary Qdrant storage root, ``papers``, or ``journal_chunks``.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest


REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"
QDRANT = REPO / "tools" / "qdrant" / "qdrant.exe"
RUN = os.environ.get("RUN_REAL_JOURNAL_BULK") == "1"
pytestmark = pytest.mark.skipif(not RUN, reason="set RUN_REAL_JOURNAL_BULK=1 for real local model/Qdrant lifecycle")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _private_bind_host() -> str:
    """Use a real LAN address so normal internal-owner resolution applies.

    The black-box must not forge an ``X-Retrival-User-Id`` header; that is exactly
    what production callers are prohibited from doing.  This Windows runner normally
    owns a 10.100.100.x overlay address; the environment override makes the check
    explicit on a different lab host.
    """
    return os.environ.get("REAL_JOURNAL_BULK_BIND_HOST", "10.100.100.99")


def _wait_http(url: str, *, timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=1.0)
            if response.status_code < 500:
                return
        except Exception as exc:  # noqa: BLE001 -- diagnostic retained in failure.
            last = exc
        time.sleep(0.2)
    raise AssertionError(f"service did not become reachable at {url}: {last!r}")


@dataclass
class Child:
    name: str
    command: list[str]
    env: dict[str, str]
    cwd: Path
    log_path: Path
    process: subprocess.Popen[str] | None = None
    exits: list[int | None] = field(default_factory=list)

    @property
    def stdout_path(self) -> Path:
        return self.log_path.with_name(f"{self.log_path.stem}.stdout.log")

    @property
    def stderr_path(self) -> Path:
        return self.log_path.with_name(f"{self.log_path.stem}.stderr.log")

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        stdout = self.stdout_path.open("a", encoding="utf-8")
        stderr = self.stderr_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            self.command, cwd=self.cwd, env=self.env, text=True,
            stdout=stdout, stderr=stderr,
        )
        # Popen owns duplicate descriptors; close only the parent copies.
        stdout.close()
        stderr.close()

    def stop(self, *, required: bool = False) -> int | None:
        proc = self.process
        if proc is None:
            return None
        if self.exits:
            return self.exits[-1]
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=15)
        rc = proc.returncode
        self.exits.append(rc)
        if required:
            assert rc == 0, f"{self.name} exited {rc}; log={self.log_path}"
        return rc

    def stdout_text(self) -> str:
        return self.stdout_path.read_text(encoding="utf-8", errors="replace") if self.stdout_path.exists() else ""

    def stderr_text(self) -> str:
        return self.stderr_path.read_text(encoding="utf-8", errors="replace") if self.stderr_path.exists() else ""

    def text(self) -> str:
        return f"[stdout]\n{self.stdout_text()}\n[stderr]\n{self.stderr_text()}"


class RealJournalBulkHarness:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.logs = root / "child-logs"
        self.bind_host = _private_bind_host()
        self.http_port = _free_port()
        self.grpc_port = _free_port()
        self.api_port = _free_port()
        self.collection = "journal_bulk_blackbox_isolated"
        self.qdrant_url = f"http://127.0.0.1:{self.http_port}"
        self.api_url = f"http://{self.bind_host}:{self.api_port}"
        self.env = os.environ.copy()
        self.env.update({
            "PYTHONPATH": str(REPO),
            "PYTHONIOENCODING": "utf-8",
            "NO_PROXY": "localhost,127.0.0.1,::1,10.100.100.99",
            "no_proxy": "localhost,127.0.0.1,::1,10.100.100.99",
            # Use the authentic local BGE-M3 on CPU: this runner is not .88 and
            # AGENTS.md prohibits consuming its GPU0.
            "EMBEDDING__DEVICE": "cpu",
            "EMBEDDING__LOCAL_FILES_ONLY": "true",
            "DATA_DIR": str(root / "data"),
            "QDRANT_URL": self.qdrant_url,
            "QDRANT_GRPC_PORT": str(self.grpc_port),
            "QDRANT_PREFER_GRPC": "false",
            "JOURNAL_QDRANT_COLLECTION": self.collection,
            "INTERNAL_API_PORT": str(self.api_port),
            "PUBLIC_API_PORT": str(_free_port()),
            "RETRIEVAL_API_URL": self.api_url,
            "RETRIVAL_API_URL": self.api_url,
        })
        self.qdrant_children: list[Child] = []
        self.api = Child(
            "uvicorn", [str(PYTHON), "-m", "uvicorn", "--host", self.bind_host,
            "--port", str(self.api_port), "--log-level", "info", "--app-dir", "tests",
            "journal_bulk_blackbox_app:app"], self.env, REPO, self.logs / "uvicorn.log")
        self.children: list[Child] = [self.api]

    @property
    def db_path(self) -> Path:
        return self.root / "data" / "db" / "retrieval.db"

    def start_qdrant(self) -> Child:
        child = Child(
            f"qdrant-{len(self.qdrant_children) + 1}", [str(QDRANT), "--disable-telemetry"],
            {**self.env,
             "QDRANT__SERVICE__HTTP_PORT": str(self.http_port),
             "QDRANT__SERVICE__GRPC_PORT": str(self.grpc_port),
             "QDRANT__STORAGE__STORAGE_PATH": str(self.root / "qdrant-storage")},
            REPO, self.logs / f"qdrant-{len(self.qdrant_children) + 1}.log")
        child.start()
        _wait_http(f"{self.qdrant_url}/readyz")
        self.qdrant_children.append(child)
        self.children.append(child)
        return child

    def start(self) -> None:
        assert PYTHON.exists(), PYTHON
        assert QDRANT.exists(), QDRANT
        self.start_qdrant()
        self.api.start()
        _wait_http(f"{self.api_url}/openapi.json")

    def mcp_index(self, directory: Path, label: str) -> tuple[dict[str, Any], Child]:
        child = Child(
            f"mcp-{label}", [str(PYTHON), "tests/journal_bulk_mcp_client.py", str(directory),
            "--timeout-seconds", "1200"], self.env, REPO, self.logs / f"mcp-{label}.log")
        child.start()
        self.children.append(child)
        deadline = time.monotonic() + 1250
        while child.process is not None and child.process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)
        assert child.process is not None
        assert child.process.poll() is not None, f"{child.name} timed out; log={child.log_path}"
        rc = child.stop()
        output = child.stdout_text()
        assert rc == 0, f"{child.name} rc={rc}; transcript:\n{output}"
        lines = [line for line in output.splitlines() if line.strip().startswith("{")]
        assert lines, f"{child.name} emitted no JSON result; transcript:\n{output}"
        payload = json.loads(lines[-1])
        return payload, child

    def sql_counts(self) -> dict[str, int]:
        with sqlite3.connect(self.db_path) as db:
            return {
                "live_doc_entries": db.execute(
                    "SELECT count(*) FROM journal_entries WHERE source_kind='doc' AND deleted_at IS NULL"
                ).fetchone()[0],
                "indexed_live_doc_entries": db.execute(
                    "SELECT count(*) FROM journal_entries WHERE source_kind='doc' AND deleted_at IS NULL AND indexed_at IS NOT NULL"
                ).fetchone()[0],
                "live_doc_chunks": db.execute(
                    "SELECT count(*) FROM journal_chunks c JOIN journal_entries e ON e.entry_id=c.entry_id "
                    "WHERE e.source_kind='doc' AND e.deleted_at IS NULL"
                ).fetchone()[0],
                "fts_rows": db.execute("SELECT count(*) FROM journal_fts").fetchone()[0],
            }

    def qdrant_count(self) -> int:
        response = httpx.post(
            f"{self.qdrant_url}/collections/{self.collection}/points/count",
            json={"exact": True}, timeout=20.0)
        response.raise_for_status()
        return int(response.json()["result"]["count"])

    def inventory(self, project: str) -> dict[str, Any]:
        response = httpx.get(f"{self.api_url}/api/index/inventory", params={
            "host": os.environ.get("COMPUTERNAME") or socket.gethostname(),
            "project": project, "kind": "doc", "detail": "true",
        }, timeout=20.0)
        response.raise_for_status()
        return response.json()

    def search(self, project: str, query: str, mode: str) -> dict[str, Any]:
        response = httpx.post(f"{self.api_url}/api/journal/search", json={
            "project_slug": project, "query": query, "mode": mode,
            "source_kind": "doc", "top_k": 10, "include_chunks": True,
        }, timeout=120.0)
        response.raise_for_status()
        return response.json()

    def stop(self) -> None:
        self.api.stop()
        for child in reversed(self.qdrant_children):
            child.stop()
        manifest = {"children": [
            {
                "name": child.name,
                "command": child.command,
                "stdout": str(child.stdout_path),
                "stderr": str(child.stderr_path),
                "exit_codes": child.exits,
            }
            for child in self.children
        ]}
        (self.root / "child-process-manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")


def _body(section_count: int, marker: str) -> str:
    # 63 + 64 bodies plus each document's summary chunk = exactly 129 Journal chunks.
    return "\n\n".join(
        f"## {marker} section {index}\n{marker} lifecycle token {index} deterministic evidence"
        for index in range(section_count)
    )


def _find_inventory_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for user in payload.get("users", []):
        for host in user.get("hosts", []):
            for project in host.get("projects", []):
                for directory in project.get("dirs", []):
                    out.update({row["rel_path"]: row for row in directory.get("files", [])})
    return out


def _deny_read(path: Path) -> str:
    user = os.environ.get("USERNAME")
    assert user, "USERNAME is required for Windows ACL unreadable lifecycle"
    result = subprocess.run(["icacls", str(path), "/deny", f"{user}:(R)"], text=True,
                            capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    try:
        with path.open("rb"):
            pass
    except PermissionError:
        return user
    # Explicitly reject a cosmetic ACL operation.  The test needs a real failed read.
    subprocess.run(["icacls", str(path), "/remove:d", user], text=True,
                   capture_output=True, check=False)
    raise AssertionError("icacls deny did not make the current process unable to read the file")


def _restore_read(path: Path, user: str) -> None:
    # The explicit deny intentionally prevents a normal ACL edit by this same
    # user.  Reclaiming ownership is limited to the pytest-owned temp file and
    # makes teardown reliable even when Windows applies the deny aggressively.
    ownership = subprocess.run(["takeown", "/F", str(path)], text=True,
                               capture_output=True, check=False)
    assert ownership.returncode == 0, ownership.stdout + ownership.stderr
    result = subprocess.run(["icacls", str(path), "/remove:d", user], text=True,
                            capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert path.read_text(encoding="utf-8")


def _job(payload: dict[str, Any], expected: str) -> dict[str, Any]:
    job = payload.get("job")
    assert isinstance(job, dict), payload
    assert job.get("status") == expected, job
    return job


def test_real_journal_bulk_lifecycle_no_mock(tmp_path: Path) -> None:
    """Exercise MCP -> HTTP -> Uvicorn -> SQLite/FTS -> BGE-M3 -> Qdrant lifecycle."""
    harness = RealJournalBulkHarness(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    first = docs / "alpha.md"
    second = docs / "beta.md"
    first.write_text(_body(63, "alphaseedtoken"), encoding="utf-8")
    second.write_text(_body(64, "betaseedtoken"), encoding="utf-8")
    unreadable_user: str | None = None
    keep = docs / "keep.md"
    try:
        harness.start()

        initial, initial_client = harness.mcp_index(docs, "initial-129")
        initial_job = _job(initial, "completed")
        # The real MCP owns Windows scope canonicalization; use its returned
        # value for all direct API/database observations instead of guessing the
        # filesystem's casing from the parent pytest process.
        project = str(initial_job["project_slug"])
        assert initial_job["files_total"] == 2
        assert initial_job["chunks_total"] == 129
        assert initial_job["indexed"] == 129
        # These assertions are intentionally RED until production emits the observable
        # real-service batches.  A unit fake cannot satisfy this process log contract.
        api_log = harness.api.text()
        assert re.search(r"journal\.embed_batch.*size=128", api_log), api_log[-8000:]
        assert re.search(r"journal\.embed_batch.*size=1", api_log), api_log[-8000:]
        assert harness.sql_counts() == {
            "live_doc_entries": 2, "indexed_live_doc_entries": 2,
            "live_doc_chunks": 129, "fts_rows": 2,
        }
        assert harness.qdrant_count() == 129

        before_unchanged_sql = harness.sql_counts()
        before_unchanged_qdrant = harness.qdrant_count()
        before_unchanged_api_stdout = harness.api.stdout_text()
        unchanged, unchanged_client = harness.mcp_index(docs, "unchanged")
        assert unchanged.get("job") is None, unchanged
        transcript = unchanged_client.text()
        assert "POST /api/journal/bulk-index" not in transcript
        assert "POST /api/journal/entries" not in transcript
        unchanged_api_delta = harness.api.stdout_text()[len(before_unchanged_api_stdout):]
        assert "journal.embed_batch" not in unchanged_api_delta, unchanged_api_delta
        assert harness.sql_counts() == before_unchanged_sql
        assert harness.qdrant_count() == before_unchanged_qdrant

        keep.write_text("## keep\nreal unreadable_keep lifecycle evidence", encoding="utf-8")
        keep_index, _ = harness.mcp_index(docs, "add-keep")
        _job(keep_index, "completed")
        before_unreadable_inventory = _find_inventory_rows(harness.inventory(project))
        assert set(before_unreadable_inventory) == {"alpha.md", "beta.md", "keep.md"}
        unreadable_user = _deny_read(keep)
        unreadable, unreadable_client = harness.mcp_index(docs, "unreadable-keep")
        assert unreadable.get("job") is None
        assert "unreadable files" in unreadable_client.text()
        after_unreadable_inventory = _find_inventory_rows(harness.inventory(project))
        assert after_unreadable_inventory["keep.md"]["content_hash"] == before_unreadable_inventory["keep.md"]["content_hash"]
        _restore_read(keep, unreadable_user)
        unreadable_user = None

        # Remove one file.  The synchronous no-change MCP completion path must execute
        # latest-wins reconcile in SQL/FTS and Qdrant rather than merely rewrite catalog.
        second.unlink()
        deleted, _ = harness.mcp_index(docs, "delete-beta")
        assert deleted.get("job") is None
        deleted_inventory = _find_inventory_rows(harness.inventory(project))
        assert set(deleted_inventory) == {"alpha.md", "keep.md"}
        beta_keyword = harness.search(project, "betaseedtoken", "keyword")
        assert beta_keyword["total"] == 0, beta_keyword
        assert harness.qdrant_count() == 66

        # First normal edit establishes a known old revision for the Qdrant outage test.
        first.write_text(_body(63, "alphamidtoken"), encoding="utf-8")
        edited, _ = harness.mcp_index(docs, "edit-alpha")
        _job(edited, "completed")
        old_live = harness.search(project, "alphamidtoken", "keyword")
        assert old_live["total"] == 1, old_live
        stable_catalog = _find_inventory_rows(harness.inventory(project))
        stable_sql = harness.sql_counts()
        stable_qdrant = harness.qdrant_count()

        # Stop only this isolated Qdrant.  The subsequent bulk job must fail and must
        # not publish a new catalog snapshot/reconcile away the old readable revision.
        harness.qdrant_children[-1].stop()
        first.write_text(_body(63, "alpharecoverytoken"), encoding="utf-8")
        failed, _ = harness.mcp_index(docs, "qdrant-down")
        failed_job = _job(failed, "failed")
        # Windows localises the real connection-refused message, so the durable
        # contract is a nonempty job error after the isolated Qdrant was stopped,
        # not an English spelling of the service name.
        assert str(failed_job.get("error", "")).strip()
        assert _find_inventory_rows(harness.inventory(project)) == stable_catalog
        # Qdrant is down here; prove the old searchable SQL/FTS fact remains first.
        assert harness.sql_counts()["indexed_live_doc_entries"] == stable_sql["indexed_live_doc_entries"]
        assert harness.search(project, "alphamidtoken", "keyword")["total"] == 1

        # Restart the same isolated storage and retry the SAME changed file through MCP.
        # Correct recovery promotes it and leaves only this latest revision searchable.
        harness.start_qdrant()
        recovered, _ = harness.mcp_index(docs, "qdrant-retry")
        _job(recovered, "completed")
        final_keyword = harness.search(project, "alpharecoverytoken", "keyword")
        final_vector = harness.search(project, "alpharecoverytoken", "dense")
        assert final_keyword["total"] == 1, final_keyword
        assert final_vector["total"] == 1, final_vector
        assert harness.search(project, "alphamidtoken", "keyword")["total"] == 0
        final_sql = harness.sql_counts()
        assert final_sql == {
            "live_doc_entries": 2, "indexed_live_doc_entries": 2,
            "live_doc_chunks": 66, "fts_rows": 2,
        }
        assert harness.qdrant_count() == 66
        final_rows = _find_inventory_rows(harness.inventory(project))
        assert set(final_rows) == {"alpha.md", "keep.md"}
        assert final_rows["alpha.md"]["content_hash"] != stable_catalog["alpha.md"]["content_hash"]
        (tmp_path / "blackbox-summary.json").write_text(json.dumps({
            "collection": harness.collection,
            "project": project,
            "embedding_batches": [128, 1],
            "final_sql": final_sql,
            "final_qdrant_points": harness.qdrant_count(),
            "live_catalog_paths": sorted(final_rows),
        }, indent=2), encoding="utf-8")
    finally:
        if unreadable_user is not None:
            _restore_read(keep, unreadable_user)
        harness.stop()
