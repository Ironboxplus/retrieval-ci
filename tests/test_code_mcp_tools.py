"""Tests for the code-KB MCP tools (index_code / search_code) — REST-backed.

The code KB is chunked CLIENT-side (this process walks + AST-chunks the repo); the
chunks are POSTed to the backend, which embeds + stores them in the shared 4090
`code_chunks` collection. These tests are hermetic: `server._client` is replaced
with a fake capturing the REST calls (no backend, no GPU, no local Qdrant). Async
calls use asyncio.run() (no pytest-asyncio).
"""
from __future__ import annotations

import asyncio
import re
import threading

import mcp_server.server as server

from conftest import git_init_repo

def test_known_code_hashes_requests_detail_tree(monkeypatch):
    """The incremental diff baseline MUST request detail=True: the inventory endpoint now
    defaults to a SUMMARY (no per-file nodes), so without detail the baseline would be empty
    and index_code(incremental=True) would silently re-embed every file every run."""
    fake = _FakeCodeClient(inventory={"users": []})
    monkeypatch.setattr(server, "_client", fake)
    asyncio.run(server._known_code_hashes("C:/repo"))
    assert fake.inv_detail is True


# A relative scope resolves against the CALLING client's directory (`_session_cwd`),
# not the shared singleton daemon's process directory — that is the base these vary.
def test_scope_path_canonicalises_windows_drive_case(monkeypatch):
    """All Windows components share one case-insensitive canonical scope identity."""
    monkeypatch.setattr(server, "_session_cwd", lambda: "c:/Users/Arc/repo")
    assert server._scope_path(".") == "C:/users/arc/repo"
    assert server._scope_path("Sub\\Dir") == "C:/users/arc/repo/sub/dir"


def test_scope_path_leaves_posix_paths_untouched(monkeypatch):
    monkeypatch.setattr(server, "_session_cwd", lambda: "/home/arc/repo")
    assert server._scope_path(".") == "/home/arc/repo"
    # POSIX casing identifies different directories, so it must survive the join.
    assert server._scope_path("Sub/Dir") == "/home/arc/repo/Sub/Dir"


SAMPLE = '''\
import time


def upsert_with_retry(client, points):
    """upsert with exponential backoff retry"""
    delay = 1.0
    for attempt in range(5):
        try:
            client.upsert(points)
            return
        except Exception:
            time.sleep(delay)
            delay *= 2


def render_template(name, context):
    """render an html template with the given context"""
    return f"<html>{name}:{context}</html>"
'''


# --------------------------- _format_code_hits (pure, dict hits) --------------------------- #

def test_format_code_hits_cites_path_line_and_snippet():
    hits = [{"path": "core/storage/qdrant_store.py", "start_line": 10, "end_line": 25,
             "label": "upsert_with_retry", "score": 0.812, "language": "python",
             "code": "def upsert_with_retry(client, points):\n    ..."}]
    text = server._format_code_hits(hits, "Code matches:")
    assert "core/storage/qdrant_store.py:10-25" in text
    assert "upsert_with_retry" in text
    assert "0.812" in text


def test_format_code_hits_returns_full_code_fenced_not_truncated():
    # Wave-1 #2: the full chunk comes back in a fenced block with real newlines/indent,
    # NOT a whitespace-flattened 200-char preview (closes the "Read again" round-trip).
    # Normal AST-sized chunks stay complete; only pathological giants are hard-capped.
    body = "\n".join(f"    line_{i} = compute({i})" for i in range(40))
    code = f"def big():\n{body}\n    return done"
    hits = [{"path": "x.py", "start_line": 1, "end_line": 42, "label": "big",
             "score": 0.5, "language": "python", "code": code}]
    text = server._format_code_hits(hits, "Code matches:")
    assert "```python" in text and "\n```" in text
    assert "line_39 = compute(39)" in text          # full body, not truncated at 200
    assert "    line_0 = compute(0)" in text          # indentation preserved
    assert "truncated" not in text.lower()


def test_format_code_hits_caps_giant_single_chunk():
    # Hang forensics 2026-07-23: one methviz-style hit can be multi-KB; unbounded full
    # chunk dump bloated MCP stdio and left the host tool call "in progress" for minutes.
    giant = "x" * (server._CODE_HIT_CODE_CHARS + 5000)
    hits = [{"path": "methviz/index.html", "start_line": 1, "end_line": 1,
             "label": "", "score": 0.9, "language": "html", "code": giant}]
    text = server._format_code_hits(hits, "Code matches:")
    assert "methviz/index.html:1-1" in text
    assert "```html" in text
    assert len(text) < len(giant)
    assert "truncated" in text.lower()
    assert "methviz/index.html:1" in text  # recovery citation


def test_format_code_hits_caps_total_output_across_hits():
    # Each hit stays under the per-chunk cap but many of them would exceed the
    # total MCP stdio budget without omission.
    hits = []
    for i in range(20):
        hits.append({
            "path": f"f{i}.py", "start_line": 1, "end_line": 50, "label": f"fn{i}",
            "score": 0.5, "language": "python",
            "code": f"def fn{i}():\n" + ("    value = compute(payload)\n" * 120),
        })
    text = server._format_code_hits(hits, "Code matches:")
    assert len(text) <= server._CODE_HIT_TOTAL_CHARS + 400  # header + omit note slack
    assert "omitted" in text.lower() or "more hit" in text.lower()
    assert "f0.py:1-50" in text  # first hits still present


def test_format_code_hits_empty_guides_to_index():
    text = server._format_code_hits([], "Code matches:")
    assert "index_code" in text


# --------------------------- index_code / search_code (REST) --------------------------- #

def _inventory(files):
    """A minimal inventory response (the diff baseline). files: (rel_path, hash, chunks, lang)."""
    return {"users": [{"hosts": [{"projects": [{"dirs": [{"files": [
        {"rel_path": r, "kind": "code", "content_hash": h, "chunk_count": cc, "language": lang}
        for (r, h, cc, lang) in files]}]}]}]}]}


def _child_dir_with_file(tmp_path, filename, content):
    child = tmp_path / "child"
    child.mkdir()
    (child / filename).write_text(content, encoding="utf-8")
    return child


class _FakeCodeClient:
    """Captures code_index / code_search / report_index / index_inventory calls."""

    def __init__(self, hits=None, inventory=None, overlap=None):
        self.indexed = None
        self.reported = None
        self.deleted = None
        self._hits = hits or []
        self._inventory = inventory or {"users": []}
        self._overlap = overlap or {"ancestor_blocks": [], "child_warnings": []}
        self.policy = None

    async def code_index(self, *, project, host, root_dir, files,
                         incremental=False, manifest=None, removed_paths=None, report=None,
                         client_job_id=None):
        # Async contract (W156/W159): POST returns a 202 job view; the catalog `report`
        # is bundled in so the backend writes it on completion — the client never polls.
        self.indexed = {"project": project, "host": host, "root_dir": root_dir, "files": files,
                        "incremental": incremental, "manifest": manifest,
                        "removed_paths": removed_paths, "report": report,
                        "client_job_id": client_job_id}
        n = sum(len(f["chunks"]) for f in files)
        return {"job_id": "j1", "status": "queued", "indexed": n,
                "progress": 0.0, "queue_position": 0}

    async def code_index_job(self, job_id):
        return {"job_id": job_id, "status": "completed", "indexed": 0, "progress": 1.0}

    async def code_index_jobs(self, **kwargs):
        return {"jobs": []}

    async def index_inventory(self, *, host=None, project=None, kind=None, detail=False):
        self.inv_detail = detail
        return self._inventory

    async def scope_overlap(self, *, project, host=None, kind=None,
                            respect_gitignore=None):
        # Recorded, not ignored: the backend needs the candidate's filter to tell a
        # redundant child from one reaching files the parent's gitignore skipped.
        self.overlap_respect_gitignore = respect_gitignore
        return self._overlap

    async def upsert_index_policy(self, **policy):
        self.policy = policy
        return policy

    async def get_index_policy(self, *, host, project, kind="code"):
        return self.policy or {"auto_refresh": False}

    async def code_search(self, *, query, project=None, top_k=8, mode="hybrid",
                          rerank=False, language=None, path_glob=None, max_per_file=None):
        self.searched = {"query": query, "project": project, "top_k": top_k,
                         "mode": mode, "rerank": rerank, "language": language,
                         "path_glob": path_glob, "max_per_file": max_per_file}
        return {"hits": self._hits, "total": len(self._hits)}

    async def report_index(self, *, host, project, files, root_dir=None, replace=True):
        self.reported = {"host": host, "project": project, "files": files,
                         "root_dir": root_dir, "replace": replace}
        return {"total_files": len(files), "upserted": len(files), "removed": 0}

    async def code_delete(self, *, project):
        self.deleted = {"project": project}
        return {"project": project, "deleted": 3, "catalog_removed": 1}


def _index_code_to_completion(*args, **kwargs) -> str:
    async def scenario() -> str:
        reply = await server.index_code(*args, **kwargs)
        match = re.search(r"job ([0-9a-f]+)", reply)
        if not match:
            return reply
        local_job_id = match.group(1)
        task = server._CODE_PREPARATION_TASKS.get(local_job_id)
        if task is not None:
            await task
        return reply + "\n" + await server.index_code_status(local_job_id)

    return asyncio.run(scenario())


def test_index_code_chunks_locally_and_uploads(tmp_path, monkeypatch):
    (tmp_path / "sample.py").write_text(SAMPLE, encoding="utf-8")
    fake = _FakeCodeClient()
    monkeypatch.setattr(server, "_client", fake)

    out = _index_code_to_completion(str(tmp_path))
    # W159 true-async: returns immediately with the job, never blocks/polls.
    assert "Indexing started" in out and "backend job j1" in out
    # scope is FORCED to the indexed dir's abspath (no hand-typed project) — the dedup
    # fix: the same directory always maps to one scope, never name-variant duplicates.
    assert fake.indexed is not None
    assert fake.indexed["project"] == server._scope_path(str(tmp_path))
    # default is the full wipe-and-rebuild path (incremental opt-in)
    assert fake.indexed["incremental"] is False
    files = fake.indexed["files"]
    assert {f["rel_path"] for f in files} == {"sample.py"}
    assert files[0]["language"] == "python"
    assert files[0]["chunks"] and files[0]["chunks"][0]["content"].startswith("# path:")
    # each file carries a content_hash so a later incremental pass can diff it
    assert files[0].get("content_hash")
    # catalog records are BUNDLED into the submit (report=), written by the backend on
    # completion — the client never makes a separate blocking report call.
    report = fake.indexed["report"]
    assert report is not None and report["replace"] is True
    assert {f["rel_path"] for f in report["files"]} == {"sample.py"}
    assert report["files"][0]["kind"] == "code"
    assert fake.reported is None  # no separate /api/index/report round-trip


def test_index_code_returns_while_local_chunking_is_still_running(tmp_path, monkeypatch):
    """The repo-local MCP server must background the local walk/chunker too."""
    fake = _FakeCodeClient()
    monkeypatch.setattr(server, "_client", fake)
    collection_started = threading.Event()
    release_collection = threading.Event()

    def collect(*_args, **_kwargs):
        collection_started.set()
        release_collection.wait(timeout=5)
        return [{
            "rel_path": "sample.py", "language": "python", "chunks": [{
                "start_line": 1, "end_line": 1, "label": "",
                "content": "# path: sample.py\nx", "code": "x",
            }],
        }]

    monkeypatch.setattr(server, "_collect_code_files", collect)

    async def scenario():
        call = asyncio.create_task(server.index_code(str(tmp_path), auto_refresh=False))
        while not collection_started.is_set():
            await asyncio.sleep(0.01)
        try:
            reply = await asyncio.wait_for(asyncio.shield(call), timeout=0.1)
            local_job_id = re.search(r"job ([0-9a-f]+)", reply).group(1)
            status = await server.index_code_status(local_job_id)
            assert "[preparing]" in status
            assert fake.indexed is None
        finally:
            release_collection.set()
            await call

        while fake.indexed is None:
            await asyncio.sleep(0.01)
        while "[completed]" not in await server.index_code_status(local_job_id):
            await asyncio.sleep(0.01)

    asyncio.run(scenario())


def test_index_code_returns_while_scope_preflight_is_still_running(tmp_path, monkeypatch):
    fake = _FakeCodeClient()
    monkeypatch.setattr(server, "_client", fake)
    preflight_started = asyncio.Event()
    release_preflight = asyncio.Event()

    async def guard(*_args, **_kwargs):
        preflight_started.set()
        await release_preflight.wait()
        return None, ""

    monkeypatch.setattr(server, "_nested_scope_guard", guard)
    monkeypatch.setattr(server, "_collect_code_files", lambda *_a, **_k: [])

    async def scenario():
        reply = await asyncio.wait_for(
            server.index_code(str(tmp_path), auto_refresh=False), timeout=0.1)
        local_job_id = re.search(r"job ([0-9a-f]+)", reply).group(1)
        await preflight_started.wait()
        assert "checking scope overlap" in await server.index_code_status(local_job_id)
        release_preflight.set()
        await server._CODE_PREPARATION_TASKS[local_job_id]

    asyncio.run(scenario())


def test_index_code_status_refreshes_live_backend_progress_for_local_job(monkeypatch):
    fake = _FakeCodeClient()

    async def code_index_job(job_id):
        assert job_id == "backend-job"
        return {
            "job_id": job_id, "status": "running", "project": "p",
            "chunks_done": 384, "chunks_total": 1024, "progress": 0.375,
            "queue_position": 0, "indexed": 0,
            # Backend files_total = files in this POST (changed set), NOT full tree.
            "files_total": 1347,
        }

    fake.code_index_job = code_index_job
    monkeypatch.setattr(server, "_client", fake)
    monkeypatch.setitem(server._CODE_PREPARATION_STATUS, "local-job", {
        "job_id": "local-job", "backend_job_id": "backend-job",
        "status": "running", "project": "p", "phase": "backend embedding and upsert",
        "chunks_done": 0, "chunks_total": 1024, "progress": 0.0,
        # Local prepare saw the full tree; must survive backend merge.
        "files_done": 3135, "files_total": 3135, "files_changed": 1347,
        "chunks_ready": 21851,
    })

    out = asyncio.run(server.index_code_status(job_id="local-job"))

    assert "384/1024 chunks" in out
    assert "(38%)" in out
    assert "prepared 1347/3135 files" in out
    st = server._CODE_PREPARATION_STATUS["local-job"]
    assert st["chunks_done"] == 384
    assert st["files_total"] == 3135  # not clobbered by backend's 1347
    assert st["files_changed"] == 1347
    assert st["chunks_ready"] == 21851
    assert st["phase"] == "backend embedding and upsert"


def test_update_preparation_from_backend_retains_local_file_counters():
    status = {
        "job_id": "local", "backend_job_id": "backend", "project": "p",
        "status": "running", "progress": 0.1, "chunks_done": 0, "chunks_total": 100,
        "files_done": 50, "files_total": 100, "files_changed": 20, "chunks_ready": 400,
        "phase": "backend embedding and upsert", "summary": "incremental: 20 changed",
    }
    backend = {
        "job_id": "backend", "status": "running", "project": "p",
        "progress": 0.5, "chunks_done": 50, "chunks_total": 100,
        "files_total": 20,  # changed-file count from the request body
        "indexed": 0, "queue_position": 0,
    }
    server._update_preparation_from_backend(status, backend)
    assert status["chunks_done"] == 50
    assert status["progress"] == 0.5
    assert status["files_total"] == 100
    assert status["files_done"] == 50
    assert status["files_changed"] == 20
    assert status["chunks_ready"] == 400
    assert status["summary"] == "incremental: 20 changed"
    assert status["job_id"] == "local"
    assert status["backend_job_id"] == "backend"


def test_index_code_status_list_merges_backend_progress_into_local_jobs(monkeypatch):
    fake = _FakeCodeClient()

    async def code_index_jobs(**_kwargs):
        assert _kwargs == {"limit": 100}
        return {"jobs": [{
            "job_id": "backend-job", "status": "running", "project": "p",
            "chunks_done": 640, "chunks_total": 1024, "progress": 0.625,
            "queue_position": 0, "indexed": 0,
        }]}

    fake.code_index_jobs = code_index_jobs
    monkeypatch.setattr(server, "_client", fake)
    monkeypatch.setitem(server._CODE_PREPARATION_STATUS, "local-job", {
        "job_id": "local-job", "backend_job_id": "backend-job",
        "status": "running", "project": "p", "phase": "backend embedding and upsert",
        "chunks_done": 0, "chunks_total": 1024, "progress": 0.0,
    })

    out = asyncio.run(server.index_code_status())

    assert "640/1024 chunks (62%)" in out
    assert out.count("backend-job") == 1


def test_index_code_status_keeps_cached_local_progress_when_backend_job_is_lost(monkeypatch):
    fake = _FakeCodeClient()
    response = server.httpx.Response(
        404, request=server.httpx.Request("GET", "http://backend/api/code/jobs/backend-job"))

    async def missing_job(_job_id):
        raise server.httpx.HTTPStatusError(
            "job not found", request=response.request, response=response)

    fake.code_index_job = missing_job
    monkeypatch.setattr(server, "_client", fake)
    monkeypatch.setitem(server._CODE_PREPARATION_STATUS, "local-job", {
        "job_id": "local-job", "backend_job_id": "backend-job",
        "status": "running", "project": "p", "phase": "backend embedding and upsert",
        "chunks_done": 384, "chunks_total": 1024, "progress": 0.375,
    })

    out = asyncio.run(server.index_code_status(job_id="local-job"))

    assert "384/1024 chunks (38%)" in out
    assert "backend restarted" in out
    assert "showing cached local status" in out


def test_index_code_status_keeps_cached_progress_during_retryable_gateway_error(monkeypatch):
    fake = _FakeCodeClient()
    response = server.httpx.Response(
        503, request=server.httpx.Request("GET", "http://backend/api/code/jobs/backend-job"))

    async def unavailable_job(_job_id):
        raise server.httpx.HTTPStatusError(
            "service unavailable", request=response.request, response=response)

    fake.code_index_job = unavailable_job
    monkeypatch.setattr(server, "_client", fake)
    monkeypatch.setitem(server._CODE_PREPARATION_STATUS, "local-job", {
        "job_id": "local-job", "backend_job_id": "backend-job",
        "status": "running", "project": "p", "phase": "backend embedding and upsert",
        "chunks_done": 384, "chunks_total": 1024, "progress": 0.375,
    })

    out = asyncio.run(server.index_code_status(job_id="local-job"))

    assert "384/1024 chunks (38%)" in out
    assert "progress refresh unavailable" in out
    assert "showing cached local status" in out


def test_index_code_status_list_keeps_cached_progress_during_backend_outage(monkeypatch):
    fake = _FakeCodeClient()
    request = server.httpx.Request("GET", "http://backend/api/code/jobs")

    async def unavailable_jobs(**_kwargs):
        raise server.httpx.ConnectError("backend offline", request=request)

    fake.code_index_jobs = unavailable_jobs
    monkeypatch.setattr(server, "_client", fake)
    monkeypatch.setitem(server._CODE_PREPARATION_STATUS, "local-job", {
        "job_id": "local-job", "backend_job_id": "backend-job",
        "status": "running", "project": "p", "phase": "backend embedding and upsert",
        "chunks_done": 384, "chunks_total": 1024, "progress": 0.375,
    })

    out = asyncio.run(server.index_code_status())

    assert "384/1024 chunks (38%)" in out
    assert "progress refresh unavailable" in out
    assert "showing cached local status" in out


def test_index_code_status_list_keeps_cached_progress_during_gateway_restart(monkeypatch):
    fake = _FakeCodeClient()
    response = server.httpx.Response(
        503, request=server.httpx.Request("GET", "http://backend/api/code/jobs"))

    async def unavailable_jobs(**_kwargs):
        raise server.httpx.HTTPStatusError(
            "service unavailable", request=response.request, response=response)

    fake.code_index_jobs = unavailable_jobs
    monkeypatch.setattr(server, "_client", fake)
    monkeypatch.setitem(server._CODE_PREPARATION_STATUS, "local-job", {
        "job_id": "local-job", "backend_job_id": "backend-job",
        "status": "running", "project": "p", "phase": "backend embedding and upsert",
        "chunks_done": 384, "chunks_total": 1024, "progress": 0.375,
    })

    out = asyncio.run(server.index_code_status())

    assert "384/1024 chunks (38%)" in out
    assert "progress refresh unavailable" in out
    assert "showing cached local status" in out


def test_index_code_status_list_warns_when_active_backend_job_disappears(monkeypatch):
    fake = _FakeCodeClient()
    monkeypatch.setattr(server, "_client", fake)
    monkeypatch.setitem(server._CODE_PREPARATION_STATUS, "local-job", {
        "job_id": "local-job", "backend_job_id": "backend-job",
        "status": "running", "project": "p", "phase": "backend embedding and upsert",
        "chunks_done": 384, "chunks_total": 1024, "progress": 0.375,
    })

    out = asyncio.run(server.index_code_status())

    assert "384/1024 chunks (38%)" in out
    assert "backend may have restarted" in out
    assert "showing cached local status" in out


def test_backend_progress_refresh_cannot_regress_a_terminal_local_job(monkeypatch):
    fake = _FakeCodeClient()
    status = {
        "job_id": "local-job", "backend_job_id": "backend-job",
        "status": "running", "project": "p", "chunks_done": 384,
        "chunks_total": 1024, "progress": 0.375,
    }

    async def stale_running(_job_id):
        status.update({"status": "completed", "chunks_done": 1024, "progress": 1.0})
        return {
            "job_id": "backend-job", "status": "running", "project": "p",
            "chunks_done": 512, "chunks_total": 1024, "progress": 0.5,
        }

    fake.code_index_job = stale_running
    monkeypatch.setattr(server, "_client", fake)

    asyncio.run(server._refresh_preparation_from_backend(status))

    assert status["status"] == "completed"
    assert status["chunks_done"] == 1024
    assert status["progress"] == 1.0


def test_backend_progress_refresh_cannot_regress_running_progress(monkeypatch):
    fake = _FakeCodeClient()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    call_count = 0
    status = {
        "job_id": "local-job", "backend_job_id": "backend-job",
        "status": "running", "project": "p", "chunks_done": 128,
        "chunks_total": 1024, "progress": 0.125,
    }

    async def out_of_order(_job_id):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            first_started.set()
            await release_first.wait()
            done = 384
        else:
            done = 640
        return {
            "job_id": "backend-job", "status": "running", "project": "p",
            "chunks_done": done, "chunks_total": 1024, "progress": done / 1024,
        }

    fake.code_index_job = out_of_order
    monkeypatch.setattr(server, "_client", fake)

    async def scenario():
        first = asyncio.create_task(server._refresh_preparation_from_backend(status))
        await first_started.wait()
        await server._refresh_preparation_from_backend(status)
        release_first.set()
        await first

    asyncio.run(scenario())

    assert status["status"] == "running"
    assert status["chunks_done"] == 640
    assert status["progress"] == 0.625


def test_wait_code_job_retries_transient_network_and_gateway_failures(monkeypatch):
    fake = _FakeCodeClient()
    request = server.httpx.Request("GET", "http://backend/api/code/jobs/backend-job")
    response = server.httpx.Response(503, request=request)
    outcomes = [
        server.httpx.ConnectError("backend offline", request=request),
        server.httpx.HTTPStatusError(
            "service unavailable", request=request, response=response),
        {"job_id": "backend-job", "status": "completed", "progress": 1.0},
    ]

    async def get_job(_job_id):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    fake.code_index_job = get_job
    monkeypatch.setattr(server, "_client", fake)
    monkeypatch.setattr(server, "_CODE_JOB_POLL_SECONDS", 0.0)
    monkeypatch.setattr(server, "_CODE_JOB_RETRY_MAX_SECONDS", 0.0, raising=False)

    job = asyncio.run(server._wait_code_job("backend-job"))

    assert job["status"] == "completed"
    assert outcomes == []


def test_active_preparation_deduplicates_index_calls_and_suppresses_watcher(
        tmp_path, monkeypatch):
    fake = _FakeCodeClient()
    monkeypatch.setattr(server, "_client", fake)
    collection_started = threading.Event()
    release_collection = threading.Event()
    collection_count = 0
    watcher_count = 0

    def collect(*_args, **_kwargs):
        nonlocal collection_count
        collection_count += 1
        collection_started.set()
        release_collection.wait(timeout=5)
        return [{
            "rel_path": "sample.py", "language": "python", "chunks": [{
                "start_line": 1, "end_line": 1, "label": "",
                "content": "# path: sample.py\nx", "code": "x",
            }],
        }]

    async def watch_scope(**_kwargs):
        nonlocal watcher_count
        watcher_count += 1

    monkeypatch.setattr(server, "_collect_code_files", collect)
    monkeypatch.setattr(server, "_watch_code_scope", watch_scope)

    async def scenario():
        first = await server.index_code(str(tmp_path), auto_refresh=False)
        first_job_id = re.search(r"job ([0-9a-f]+)", first).group(1)
        while not collection_started.is_set():
            await asyncio.sleep(0.01)
        second = await server.index_code(str(tmp_path), auto_refresh=False)
        assert first_job_id in second
        assert "no duplicate scan" in second.lower()
        await server.search_code(query="probe", repo_root=str(tmp_path))
        assert collection_count == 1
        assert watcher_count == 0
        release_collection.set()
        await server._CODE_PREPARATION_TASKS[first_job_id]

    asyncio.run(scenario())


def test_delete_waits_for_inflight_index_submission(tmp_path, monkeypatch):
    submission_started = asyncio.Event()
    release_submission = asyncio.Event()
    operation_order = []

    class BlockingClient(_FakeCodeClient):
        async def code_index(self, **kwargs):
            operation_order.append("submit-started")
            submission_started.set()
            await release_submission.wait()
            operation_order.append("submit-finished")
            return {"job_id": kwargs["client_job_id"], "status": "completed",
                    "progress": 1.0, "indexed": 1}

        async def code_delete(self, *, project):
            operation_order.append("delete-sent")
            return {"project": project, "deleted": 1, "catalog_removed": 1}

    monkeypatch.setattr(server, "_client", BlockingClient())
    monkeypatch.setattr(server, "_collect_code_files", lambda *_a, **_k: [{
        "rel_path": "sample.py", "language": "python", "chunks": [{
            "start_line": 1, "end_line": 1, "label": "",
            "content": "# path: sample.py\nx", "code": "x",
        }],
    }])

    async def scenario():
        reply = await server.index_code(str(tmp_path), auto_refresh=False)
        local_job_id = re.search(r"job ([0-9a-f]+)", reply).group(1)
        await submission_started.wait()
        deletion = asyncio.create_task(server.delete_code(str(tmp_path)))
        await asyncio.sleep(0.05)
        assert "delete-sent" not in operation_order
        release_submission.set()
        await deletion
        task = server._CODE_PREPARATION_TASKS.get(local_job_id)
        if task is not None:
            await task

    asyncio.run(scenario())
    assert operation_order == ["submit-started", "submit-finished", "delete-sent"]


def test_lost_submit_response_recovers_by_durable_receipt(tmp_path, monkeypatch):
    receipt_queries = []

    class LostResponseClient(_FakeCodeClient):
        async def code_index(self, **_kwargs):
            raise server.httpx.ReadTimeout(
                "response lost after backend acceptance",
                request=server.httpx.Request("POST", "http://backend/api/code/index"),
            )

        async def code_index_job(self, job_id):
            receipt_queries.append(job_id)
            return {"job_id": job_id, "status": "completed", "progress": 1.0,
                    "indexed": 1}

    monkeypatch.setattr(server, "_client", LostResponseClient())
    monkeypatch.setattr(server, "_collect_code_files", lambda *_a, **_k: [{
        "rel_path": "sample.py", "language": "python", "chunks": [{
            "start_line": 1, "end_line": 1, "label": "",
            "content": "# path: sample.py\nx", "code": "x",
        }],
    }])

    out = _index_code_to_completion(str(tmp_path), auto_refresh=False)

    local_job_id = re.search(r"job ([0-9a-f]+)", out).group(1)
    assert receipt_queries == [local_job_id]
    assert "[completed]" in out and "response lost" not in out


def test_index_code_blocks_child_when_parent_scope_overlap_covers_it(tmp_path, monkeypatch):
    child = _child_dir_with_file(tmp_path, "sample.py", SAMPLE)
    parent_scope = server._scope_path(str(tmp_path))
    fake = _FakeCodeClient(overlap={
        "ancestor_blocks": [{"project": parent_scope, "host": "h1"}], "child_warnings": []})
    monkeypatch.setattr(server, "_client", fake)

    out = _index_code_to_completion(str(child))

    assert "already covered by ancestor scope" in out
    assert "not indexed" in out
    assert fake.indexed is None


def test_index_code_allows_parent_when_child_scope_exists_with_warning(tmp_path, monkeypatch):
    child = _child_dir_with_file(tmp_path, "sample.py", SAMPLE)
    (tmp_path / "root.py").write_text(SAMPLE, encoding="utf-8")
    child_scope = server._scope_path(str(child))
    fake = _FakeCodeClient(overlap={
        "ancestor_blocks": [], "child_warnings": [{"project": child_scope, "host": "h1"}]})
    monkeypatch.setattr(server, "_client", fake)

    out = _index_code_to_completion(str(tmp_path))

    assert "Indexing started" in out
    assert "warning" in out.lower()
    assert "child scope" in out
    assert fake.indexed is not None
    assert fake.indexed["project"] == server._scope_path(str(tmp_path))


def test_index_code_incremental_first_pass_indexes_all(tmp_path, monkeypatch):
    # No prior inventory → incremental degrades to a full pass, but flagged incremental
    # with a manifest so the server reconciles rather than wipes.
    (tmp_path / "sample.py").write_text(SAMPLE, encoding="utf-8")
    fake = _FakeCodeClient(inventory={"users": []})
    monkeypatch.setattr(server, "_client", fake)

    out = _index_code_to_completion(str(tmp_path), incremental=True)
    assert "Indexing started" in out
    assert fake.indexed["incremental"] is True
    assert fake.indexed["manifest"] == ["sample.py"]
    assert {f["rel_path"] for f in fake.indexed["files"]} == {"sample.py"}
    assert all(f.get("content_hash") for f in fake.indexed["files"])


def test_index_code_incremental_skips_unchanged(tmp_path, monkeypatch):
    # Inventory already has sample.py at its CURRENT hash → nothing to do, no re-embed.
    from core.code_kb.chunker import file_content_hash
    (tmp_path / "sample.py").write_text(SAMPLE, encoding="utf-8")
    h = file_content_hash(str(tmp_path / "sample.py"))
    fake = _FakeCodeClient(inventory=_inventory([("sample.py", h, 2, "python")]))
    monkeypatch.setattr(server, "_client", fake)

    out = _index_code_to_completion(str(tmp_path), incremental=True)
    assert "up to date" in out
    assert fake.indexed is None  # never submitted a job — nothing changed


def test_index_code_incremental_sends_only_changed(tmp_path, monkeypatch):
    # a.py unchanged (real hash), b.py changed (stale hash) → only b.py re-embedded,
    # but the manifest lists BOTH so a.py is kept (not reconciled away).
    from core.code_kb.chunker import file_content_hash
    (tmp_path / "a.py").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "b.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    ah = file_content_hash(str(tmp_path / "a.py"))
    fake = _FakeCodeClient(inventory=_inventory([("a.py", ah, 2, "python"),
                                                 ("b.py", "stale", 1, "python")]))
    monkeypatch.setattr(server, "_client", fake)

    _index_code_to_completion(str(tmp_path), incremental=True)
    assert {f["rel_path"] for f in fake.indexed["files"]} == {"b.py"}  # only changed re-embedded
    assert fake.indexed["incremental"] is True
    assert set(fake.indexed["manifest"]) == {"a.py", "b.py"}          # both kept
    # full snapshot bundled in report so the catalog stays in sync + removed files swept
    report = fake.indexed["report"]
    assert report["replace"] is True
    assert {f["rel_path"] for f in report["files"]} == {"a.py", "b.py"}


def test_index_code_incremental_reconciles_deleted_file(tmp_path, monkeypatch):
    # Inventory has gone.py but it's no longer on disk → manifest omits it → the server
    # reconciles it away. The manifest must still be sent even with zero changed files.
    from core.code_kb.chunker import file_content_hash
    (tmp_path / "keep.py").write_text(SAMPLE, encoding="utf-8")
    kh = file_content_hash(str(tmp_path / "keep.py"))
    fake = _FakeCodeClient(inventory=_inventory([("keep.py", kh, 2, "python"),
                                                 ("gone.py", "h", 1, "python")]))
    monkeypatch.setattr(server, "_client", fake)

    _index_code_to_completion(str(tmp_path), incremental=True)
    assert fake.indexed is not None                                    # not "up to date"
    assert fake.indexed["files"] == []                                # nothing to re-embed
    assert fake.indexed["manifest"] == ["keep.py"]                    # gone.py dropped


def test_index_code_incremental_keeps_transiently_unreadable_file(tmp_path, monkeypatch):
    # CF1: b.py is already indexed but momentarily unreadable this run (AV / IDE autosave /
    # git checkout lock) → its hash reads None. It must NOT be mistaken for a deletion:
    # it stays in the manifest (vectors survive) and its stored hash is preserved in the
    # catalog report (so next run re-checks it instead of force-re-embedding).
    import core.code_kb.chunker as chunker
    from core.code_kb.chunker import file_content_hash as real_hash
    (tmp_path / "a.py").write_text(SAMPLE, encoding="utf-8")           # will be edited → changed
    (tmp_path / "b.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    ah = real_hash(str(tmp_path / "a.py"))

    def flaky_hash(path):
        return None if path.replace("\\", "/").endswith("/b.py") else real_hash(path)
    monkeypatch.setattr(chunker, "file_content_hash", flaky_hash)

    # inventory: a.py at a STALE hash (so it's "changed"), b.py already indexed
    fake = _FakeCodeClient(inventory=_inventory([("a.py", "stale", 2, "python"),
                                                 ("b.py", "bhash", 1, "python")]))
    monkeypatch.setattr(server, "_client", fake)

    _index_code_to_completion(str(tmp_path), incremental=True)
    # only a.py re-embedded (b.py unreadable, never chunked)…
    assert {f["rel_path"] for f in fake.indexed["files"]} == {"a.py"}
    # …but the unreadable-yet-known b.py stays in the manifest so it isn't reconciled away
    assert set(fake.indexed["manifest"]) == {"a.py", "b.py"}
    # bundled catalog report preserves b.py's stored hash (not overwritten with None)
    reported = {f["rel_path"]: f for f in fake.indexed["report"]["files"]}
    assert reported["b.py"]["content_hash"] == "bhash"


def test_git_commit_uses_devnull_stdin():
    # W159: git spawned from the MCP stdio server must not inherit the JSON-RPC stdin
    # pipe, or it wedges for the full 5s timeout on every index_code call.
    import inspect
    src = inspect.getsource(server._git_commit)
    assert "stdin=subprocess.DEVNULL" in src, "git subprocess must not inherit the stdio pipe"


def test_index_code_respects_gitignore_by_default(tmp_path, monkeypatch):
    # A repo where secret.py is gitignored: index_code (full pass) must skip it by
    # default — indexing gitignored files (secrets, build output) is search noise.
    git_init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("secret.py\n", encoding="utf-8")
    (tmp_path / "app.py").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "secret.py").write_text(SAMPLE, encoding="utf-8")
    fake = _FakeCodeClient()
    monkeypatch.setattr(server, "_client", fake)

    _index_code_to_completion(str(tmp_path))
    assert {f["rel_path"] for f in fake.indexed["files"]} == {"app.py"}  # secret.py skipped


def test_index_code_gitignore_can_be_disabled(tmp_path, monkeypatch):
    git_init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("secret.py\n", encoding="utf-8")
    (tmp_path / "app.py").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "secret.py").write_text(SAMPLE, encoding="utf-8")
    fake = _FakeCodeClient()
    monkeypatch.setattr(server, "_client", fake)

    _index_code_to_completion(str(tmp_path), respect_gitignore=False)
    assert {f["rel_path"] for f in fake.indexed["files"]} == {"app.py", "secret.py"}


def test_index_code_incremental_respects_gitignore(tmp_path, monkeypatch):
    # The incremental path must apply the same gitignore filter as the full pass, or a
    # gitignored file would be indexed on the first incremental pass.
    git_init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("secret.py\n", encoding="utf-8")
    (tmp_path / "app.py").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "secret.py").write_text(SAMPLE, encoding="utf-8")
    fake = _FakeCodeClient(inventory={"users": []})
    monkeypatch.setattr(server, "_client", fake)

    _index_code_to_completion(str(tmp_path), incremental=True)
    assert {f["rel_path"] for f in fake.indexed["files"]} == {"app.py"}
    assert fake.indexed["manifest"] == ["app.py"]  # secret.py not in the tree manifest


def test_search_code_formats_rest_hits(monkeypatch):
    fake = _FakeCodeClient(hits=[{"path": "a.py", "start_line": 1, "end_line": 9,
                                  "label": "f", "score": 0.7, "code": "def f(): ..."}])
    monkeypatch.setattr(server, "_client", fake)
    out = asyncio.run(server.search_code("do something", top_k=3))
    assert "a.py:1-9" in out and "[f]" in out


def test_search_code_empty_guides_to_the_exact_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_client", _FakeCodeClient(hits=[]))
    out = asyncio.run(server.search_code("anything", top_k=3, repo_root=str(tmp_path)))
    scope = server._scope_path(str(tmp_path))

    assert f"scope {scope!r}" in out
    assert "does not prove that repo_root is wrong" in out
    assert f"search_code(query='anything', repo_root={scope!r})" in out
    assert "index_inventory()" in out
    assert f"index_code(repo_root={scope!r})" in out


def test_search_code_preparing_scope_is_not_rendered_as_no_match(tmp_path, monkeypatch):
    class PreparingClient(_FakeCodeClient):
        async def code_search(self, **kwargs):
            self.searched = kwargs
            return {
                "hits": [], "total": 0,
                "scope_snapshot_state": "preparing",
                "active_index_job_id": "sing-box-job",
                "notice": "Code index write is active; partial scope withheld.",
            }

    monkeypatch.setattr(server, "_client", PreparingClient())
    out = asyncio.run(server.search_code("Hysteria-CC-RX SendBPS", repo_root=str(tmp_path)))

    assert "still preparing" in out
    assert "not a no-match result" in out
    assert "sing-box-job" in out
    assert "No matching code" not in out


def test_search_code_recovery_required_guides_full_reindex_without_polling_old_job(
        tmp_path, monkeypatch):
    class RecoveryClient(_FakeCodeClient):
        async def code_search(self, **kwargs):
            self.searched = kwargs
            return {
                "hits": [], "total": 0,
                "scope_snapshot_state": "recovery_required",
                "recovery_required": True,
                "active_index_job_id": "failed-write-job",
                "notice": "Full recovery index required after interrupted write.",
            }

    monkeypatch.setattr(server, "_client", RecoveryClient())
    out = asyncio.run(server.search_code("SendBPS", repo_root=str(tmp_path)))

    assert "full recovery index" in out.lower()
    assert "index_code(repo_root=" in out
    assert "Poll index_code_status" not in out
    assert "failed-write-job" not in out
    assert "No matching code" not in out


def test_search_code_catalog_unavailable_guides_retry_without_polling(tmp_path, monkeypatch):
    class UnavailableClient(_FakeCodeClient):
        async def code_search(self, **kwargs):
            self.searched = kwargs
            return {
                "hits": [], "total": 0,
                "scope_snapshot_state": "unavailable",
                "recovery_required": False,
                "notice": "Index Catalog visibility is unavailable; retry shortly.",
            }

    monkeypatch.setattr(server, "_client", UnavailableClient())
    out = asyncio.run(server.search_code("SendBPS", repo_root=str(tmp_path)))

    assert "catalog" in out.lower()
    assert "retry" in out.lower()
    assert "Poll index_code_status" not in out
    assert "No matching code" not in out


def test_server_watched_paths_submit_only_touched_files_and_explicit_deletions(
        tmp_path, monkeypatch):
    changed = tmp_path / "changed.py"
    changed.write_text("value = 2\n", encoding="utf-8")
    deleted = tmp_path / "deleted.py"
    project = server._scope_path(str(tmp_path))
    fake = _FakeCodeClient()
    monkeypatch.setattr(server, "_client", fake)
    known = {
        "changed.py": {"content_hash": "old", "chunk_count": 1, "language": "python"},
        "deleted.py": {"content_hash": "old", "chunk_count": 1, "language": "python"},
        "untouched.py": {"content_hash": "same", "chunk_count": 1, "language": "python"},
    }

    result = asyncio.run(server._submit_watched_code_paths(
        project=project,
        host="desktop",
        repo_root=str(tmp_path),
        changed_paths={str(changed), str(deleted)},
        known=known,
        respect_gitignore=False,
        exclude_globs=[],
    ))

    assert result.status["state"] == "refresh_started"
    assert {item["rel_path"] for item in fake.indexed["files"]} == {"changed.py"}
    assert fake.indexed["removed_paths"] == ["deleted.py"]
    assert fake.indexed["manifest"] is None
    assert fake.indexed["report"]["replace"] is False


def test_server_watched_paths_ignore_files_outside_selected_repo_root(tmp_path):
    selected = tmp_path / "selected"
    sibling = tmp_path / "sibling"
    selected.mkdir()
    sibling.mkdir()
    inside = selected / "inside.py"
    outside = sibling / "outside.py"
    inside.write_text("inside = True\n", encoding="utf-8")
    outside.write_text("outside = True\n", encoding="utf-8")

    payloads, removed, _current, reconcile = server._collect_watched_code_changes(
        str(selected), {str(inside), str(outside)}, {}, False, [])

    assert {payload["rel_path"] for payload in payloads} == {"inside.py"}
    assert removed == []
    assert reconcile is False


def test_server_search_code_reuses_one_watcher_and_never_runs_full_scan(
        tmp_path, monkeypatch):
    fake = _FakeCodeClient()
    monkeypatch.setattr(server, "_client", fake)
    started = 0
    release = asyncio.Event()

    async def watch_scope(**_kwargs):
        nonlocal started
        started += 1
        await release.wait()

    monkeypatch.setattr(server, "_watch_code_scope", watch_scope)
    monkeypatch.setattr(
        server, "_collect_incremental",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("search must not scan")),
    )

    async def scenario():
        await server.search_code(query="first", repo_root=str(tmp_path))
        await server.search_code(query="second", repo_root=str(tmp_path))
        await asyncio.sleep(0)
        assert started == 1
        release.set()
        await asyncio.sleep(0)

    asyncio.run(scenario())


def test_search_code_forwards_filter_params(monkeypatch):
    # Wave-1 #3/#6 + Wave-2 mode/rerank: all reach the backend client.
    fake = _FakeCodeClient(hits=[])
    monkeypatch.setattr(server, "_client", fake)
    asyncio.run(server.search_code("q", top_k=5, mode="keyword", rerank=True,
                                   language="rust", path_glob="src/peers/", max_per_file=2))
    assert fake.searched["mode"] == "keyword"
    assert fake.searched["rerank"] is True
    assert fake.searched["language"] == "rust"
    assert fake.searched["path_glob"] == "src/peers/"
    assert fake.searched["max_per_file"] == 2


def test_search_code_defaults_to_hybrid(monkeypatch):
    fake = _FakeCodeClient(hits=[])
    monkeypatch.setattr(server, "_client", fake)
    asyncio.run(server.search_code("q"))
    assert fake.searched["mode"] == "hybrid" and fake.searched["rerank"] is False


def test_search_code_times_out_instead_of_hanging(monkeypatch):
    """Interactive search_code must fail-fast; a stuck HTTP must not leave the MCP
    tool call open for the host's multi-minute tool_timeout_sec."""
    class _SlowClient(_FakeCodeClient):
        async def code_search(self, **kwargs):
            await asyncio.sleep(5)
            return {"hits": []}

    monkeypatch.setattr(server, "_client", _SlowClient())
    monkeypatch.setattr(server, "_SEARCH_CODE_TIMEOUT_S", 0.05)
    out = asyncio.run(server.search_code("q", top_k=3))
    assert "timed out" in out.lower()
    assert "0.05" in out or "seconds" in out.lower()


def test_search_papers_times_out_instead_of_hanging(monkeypatch):
    class _SlowClient(_FakeCodeClient):
        async def search_papers(self, **kwargs):
            await asyncio.sleep(5)
            return {"papers": [], "results": []}

    monkeypatch.setattr(server, "_client", _SlowClient())
    monkeypatch.setattr(server, "_SEARCH_PAPERS_TIMEOUT_S", 0.05)
    out = asyncio.run(server.search_papers(
        "masked diffusion language model unmasking order", top_k=20))
    assert "timed out" in out.lower()


def test_http_timeout_is_layered_not_single_float():
    # Connect/pool fail fast; long bulk reads still allowed. A bare float hid
    # "stuck connect via proxy" for the full 180s.
    timeout = server._http_timeout()
    assert isinstance(timeout, server.httpx.Timeout)
    assert timeout.connect <= 10.0
    assert timeout.pool <= 10.0
    assert timeout.read >= 30.0


# --------------------------- delete_code / 0-file no-op hint (W165) --------------------------- #

def test_delete_code_forces_abspath_scope_and_clears(tmp_path, monkeypatch):
    # W165: the explicit cleanup path. delete_code forces the scope to the dir's abspath
    # (same identity index_code used) and clears BOTH vectors and inventory rows.
    fake = _FakeCodeClient()
    monkeypatch.setattr(server, "_client", fake)
    out = asyncio.run(server.delete_code(str(tmp_path)))
    assert fake.deleted == {"project": server._scope_path(str(tmp_path))}
    assert "Deleted" in out and "3" in out and "1" in out


def test_delete_code_reports_nothing_when_already_clear(monkeypatch):
    fake = _FakeCodeClient()

    async def _empty(*, project):
        fake.deleted = {"project": project}
        return {"project": project, "deleted": 0, "catalog_removed": 0}

    fake.code_delete = _empty
    monkeypatch.setattr(server, "_client", fake)
    out = asyncio.run(server.delete_code("."))
    assert "Nothing to delete" in out


def test_index_code_full_zero_files_keeps_index_and_hints_delete(tmp_path, monkeypatch):
    # W165: a full reindex of a source-less dir must NOT submit a wipe job — the existing
    # index is kept by design (a broken walk can't silently erase it). The tool tells the
    # user the index was kept and to call delete_code to clear it explicitly.
    fake = _FakeCodeClient()
    monkeypatch.setattr(server, "_client", fake)
    out = _index_code_to_completion(str(tmp_path))  # empty dir → 0 chunkable files
    assert fake.indexed is None                          # no pointless job submitted
    assert "delete_code" in out
    assert "kept" in out.lower()


# --------------------------- delete_journal_docs (doc-scope clear) --------------------------- #

def test_delete_journal_docs_forces_abspath_scope_and_clears(tmp_path, monkeypatch):
    # Symmetric to delete_code: scope forced to the dir's abspath, clears doc entries+vectors
    # + inventory rows; keeps ACE memories (handled server-side, kind="doc").
    fake = _FakeJournalClient()
    monkeypatch.setattr(server, "_client", fake)
    out = asyncio.run(server.delete_journal_docs(str(tmp_path)))
    assert fake.deleted_scope == server._scope_path(str(tmp_path))
    assert "Deleted doc scope" in out and "3" in out and "kept" in out.lower()


def test_delete_journal_docs_reports_nothing_when_already_clear(monkeypatch):
    fake = _FakeJournalClient()

    async def _empty(*, project_slug):
        fake.deleted_scope = project_slug
        return {"project_slug": project_slug, "removed_entries": 0,
                "deleted_vectors": 0, "catalog_removed": 0}

    fake.delete_journal_docs = _empty
    monkeypatch.setattr(server, "_client", fake)
    out = asyncio.run(server.delete_journal_docs("."))
    assert "Nothing to delete" in out


# --------------------------- journal_index_dir (dedup) --------------------------- #


class _FakeJournalClient:
    """Captures record_journal / report_index calls for journal_index_dir tests.

    record_journal's dedupe_policy DEFAULTS to "append" here on purpose: the tool must
    explicitly opt into "return_existing", so a tool that forgets to pass it fails the
    assertion (this is the RED guard against re-index duplication).
    """

    def __init__(self, deduped_titles=(), inventory=None, fail_titles=(), dup_per_record=0,
                 overlap=None):
        self.records = []
        self.reported = None
        self.reconciled = None
        self.deleted_scope = None
        self._deduped_titles = set(deduped_titles)
        self._inventory = inventory or {"users": []}
        self._overlap = overlap or {"ancestor_blocks": [], "child_warnings": []}
        self._fail_titles = set(fail_titles)
        self._dup_per_record = dup_per_record

    async def record_journal(self, *, project_slug, body, client="mcp",
                             source_kind="memory", title=None, summary=None,
                             tags=None, metadata=None, dedupe_policy="append"):
        if title in self._fail_titles:
            raise RuntimeError(f"simulated indexing failure for {title}")
        self.records.append({"title": title, "dedupe_policy": dedupe_policy,
                             "source_kind": source_kind, "project_slug": project_slug})
        return {"deduped": title in self._deduped_titles, "chunk_count": 2,
                "duplicates_updated": self._dup_per_record,
                "entry": {"entry_id": f"e_{len(self.records)}"}}

    async def report_index(self, *, host, project, files, root_dir=None, replace=True,
                           patterns=None, max_depth=None):
        self.reported = {"files": files, "replace": replace, "project": project,
                         "root_dir": root_dir, "patterns": patterns, "max_depth": max_depth}
        return {"total_files": len(files)}

    async def index_inventory(self, *, host=None, project=None, kind=None, detail=False):
        self.inv_detail = detail
        return self._inventory

    async def scope_overlap(self, *, project, host=None, kind=None,
                            respect_gitignore=None):
        # Recorded, not ignored: the backend needs the candidate's filter to tell a
        # redundant child from one reaching files the parent's gitignore skipped.
        self.overlap_respect_gitignore = respect_gitignore
        return self._overlap

    async def delete_journal_docs(self, *, project_slug):
        self.deleted_scope = project_slug
        return {"project_slug": project_slug, "removed_entries": 3,
                "deleted_vectors": 9, "catalog_removed": 2}

    async def reconcile_journal_docs(self, *, project_slug, live_titles, max_depth=None,
                                     patterns=None):
        self.reconciled = {"project_slug": project_slug, "live_titles": live_titles,
                           "max_depth": max_depth, "patterns": patterns}
        return {"project_slug": project_slug, "live_titles": len(live_titles),
                "removed_entries": 1, "removed_entry_ids": ["stale"],
                "deleted_vectors": 3, "skipped": False}


def test_journal_index_dir_scope_is_forced_to_dir_abspath(tmp_path, monkeypatch):
    # Dedup fix: the journal namespace is the indexed dir's abspath, never a hand-typed
    # slug — so re-indexing the same dir (under any agent-invented name) hits one scope.
    (tmp_path / "a.md").write_text("# alpha\n", encoding="utf-8")
    fake = _FakeJournalClient()
    monkeypatch.setattr(server, "_client", fake)

    asyncio.run(server.journal_index_dir(str(tmp_path)))
    scope = server._scope_path(str(tmp_path))
    assert {r["project_slug"] for r in fake.records} == {scope}
    assert fake.reported["project"] == scope


def test_search_code_scope_defaults_to_cwd_abspath(monkeypatch):
    # search_code with no repo_root scopes to the cwd's abspath — matching what
    # index_code(repo_root=".") wrote, so index-then-search from the same workspace agrees.
    fake = _FakeCodeClient(hits=[])
    monkeypatch.setattr(server, "_client", fake)
    asyncio.run(server.search_code("q"))
    assert fake.searched["project"] == server._scope_path(".")


def test_journal_index_dir_reports_cross_scope_synced(tmp_path, monkeypatch):
    # 方案2: sum duplicates_updated across entry records and surface the total.
    (tmp_path / "a.md").write_text("# alpha\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# beta\n", encoding="utf-8")
    fake = _FakeJournalClient(dup_per_record=1)
    monkeypatch.setattr(server, "_client", fake)
    out = asyncio.run(server.journal_index_dir(str(tmp_path)))
    assert "2 cross-scope synced" in out


def test_fmt_job_line_surfaces_cross_scope_synced():
    line = server._fmt_job_line({
        "job_id": "abc", "status": "completed", "project": "/p",
        "chunks_done": 10, "chunks_total": 10, "progress": 1.0,
        "indexed": 10, "duplicates_updated": 3,
    })
    assert "3 cross-scope" in line
    quiet = server._fmt_job_line({
        "job_id": "abc", "status": "completed", "project": "/p",
        "chunks_done": 10, "chunks_total": 10, "progress": 1.0,
        "indexed": 10, "duplicates_updated": 0,
    })
    assert "cross-scope" not in quiet


def test_journal_index_dir_dedupes_and_reports_only_nonempty_md(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("# alpha\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# beta\n", encoding="utf-8")
    (tmp_path / "empty.md").write_text("   \n", encoding="utf-8")  # skipped (blank)
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")  # skipped (pattern)
    fake = _FakeJournalClient()
    monkeypatch.setattr(server, "_client", fake)

    out = asyncio.run(server.journal_index_dir(str(tmp_path)))

    # every journal write must opt into return_existing so re-runs don't duplicate
    assert [r["dedupe_policy"] for r in fake.records] == ["return_existing", "return_existing"]
    assert {r["title"] for r in fake.records} == {"a.md", "b.md"}
    assert all(r["source_kind"] == "doc" for r in fake.records)
    # snapshot inventory upsert (replace=True: the reported set IS the scope's truth now)
    assert fake.reported["replace"] is True
    assert {f["rel_path"] for f in fake.reported["files"]} == {"a.md", "b.md"}
    assert "2 file(s)" in out


def test_journal_index_dir_allows_scope_when_parent_does_not_cover_it(tmp_path, monkeypatch):
    # No overlap conflict (a parent whose indexed files sit outside this child yields
    # neither an ancestor_block nor a child_warning) → the child index proceeds and
    # reconciles its own snapshot. The overlap DECISION itself is covered server-side by
    # tests/test_index_catalog.py::test_scope_overlap_no_block_when_parent_has_no_file_under_child.
    child = _child_dir_with_file(tmp_path, "a.md", "# child\n")
    fake = _FakeJournalClient(overlap={"ancestor_blocks": [], "child_warnings": []})
    monkeypatch.setattr(server, "_client", fake)

    out = asyncio.run(server.journal_index_dir(str(child)))

    assert "1 file(s)" in out
    assert {r["title"] for r in fake.records} == {"a.md"}
    assert fake.reported["replace"] is True
    assert fake.reconciled == {
        "project_slug": server._scope_path(str(child)),
        "live_titles": ["a.md"],
        "max_depth": 0,  # non-recursive default → shallow reconcile envelope
        "patterns": ["*.md"],  # default pattern → sweep only *.md (W169b envelope)
    }


def test_journal_index_dir_blocks_child_when_parent_scope_overlap_covers_it(tmp_path, monkeypatch):
    # The server-side overlap check reports an ancestor_block (a parent doc scope already
    # covers this child) → re-indexing the child adds no coverage and would duplicate a
    # namespace, so it is refused (no records, no inventory report, no reconcile).
    child = _child_dir_with_file(tmp_path, "a.md", "# child\n")
    parent_scope = server._scope_path(str(tmp_path))
    fake = _FakeJournalClient(overlap={
        "ancestor_blocks": [{"project": parent_scope, "host": "h1"}], "child_warnings": []})
    monkeypatch.setattr(server, "_client", fake)

    out = asyncio.run(server.journal_index_dir(str(child)))

    assert "already covered by ancestor scope" in out
    assert fake.records == []
    assert fake.reported is None
    assert fake.reconciled is None


def test_journal_index_dir_allows_parent_over_child_scope_with_warning(tmp_path, monkeypatch):
    child = _child_dir_with_file(tmp_path, "a.md", "# child\n")
    (tmp_path / "root.md").write_text("# root\n", encoding="utf-8")
    child_scope = server._scope_path(str(child))
    fake = _FakeJournalClient(overlap={
        "ancestor_blocks": [], "child_warnings": [{"project": child_scope, "host": "h1"}]})
    monkeypatch.setattr(server, "_client", fake)

    out = asyncio.run(server.journal_index_dir(str(tmp_path)))

    assert "warning" in out.lower()
    assert "child scope" in out
    assert {r["title"] for r in fake.records} == {"root.md"}
    assert fake.reported is not None


def test_journal_index_dir_reconciles_only_within_its_pattern_envelope(tmp_path, monkeypatch):
    # W169b data-loss valve: indexing a dir with "*.txt" must tell the backend to sweep only
    # within the *.txt envelope, so a prior "*.md" index in the same scope is never treated as
    # a "disappeared file". The tool passes the scan's pattern(s) to reconcile.
    (tmp_path / "current.txt").write_text("txt body\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# md the txt scan never looks for\n", encoding="utf-8")
    fake = _FakeJournalClient()
    monkeypatch.setattr(server, "_client", fake)

    asyncio.run(server.journal_index_dir(str(tmp_path), pattern="*.txt"))

    assert {r["title"] for r in fake.records} == {"current.txt"}  # md not globbed
    assert fake.reconciled["patterns"] == ["*.txt"]
    assert fake.reconciled["live_titles"] == ["current.txt"]


def test_journal_index_dir_brace_pattern_covers_multiple_extensions(tmp_path, monkeypatch):
    # "*.{md,txt}" expands to both globs in a single call (glob has no native brace support),
    # and the reconcile envelope carries the expanded pattern list.
    (tmp_path / "a.md").write_text("# md\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("txt\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("print(1)\n", encoding="utf-8")  # outside both globs
    fake = _FakeJournalClient()
    monkeypatch.setattr(server, "_client", fake)

    asyncio.run(server.journal_index_dir(str(tmp_path), pattern="*.{md,txt}"))

    assert {r["title"] for r in fake.records} == {"a.md", "b.txt"}
    assert fake.reconciled["patterns"] == ["*.md", "*.txt"]


def test_expand_brace_patterns():
    assert server._expand_brace_patterns("*.md") == ["*.md"]
    assert server._expand_brace_patterns("*.{md,txt}") == ["*.md", "*.txt"]
    assert server._expand_brace_patterns("{a,b}.md") == ["a.md", "b.md"]
    assert server._expand_brace_patterns("*") == ["*"]
    # malformed / nested / multiple groups fall back to the literal pattern (unchanged behavior)
    assert server._expand_brace_patterns("*.{md") == ["*.{md"]
    assert server._expand_brace_patterns("{a,{b,c}}.md") == ["{a,{b,c}}.md"]
    assert server._expand_brace_patterns("{a,b}.{c,d}") == ["{a,b}.{c,d}"]
    assert server._expand_brace_patterns("*.{}") == ["*.{}"]


def test_journal_index_dir_keeps_catalog_and_skips_reconcile_when_a_file_fails(tmp_path, monkeypatch):
    # Finding #3: if a file fails to read/index this run, the catalog report must NOT do a
    # destructive replace (it would drop a still-live doc's row), and the stale-doc reconcile
    # must be skipped — mirroring the code-KB unreadable-keep guard.
    (tmp_path / "a.md").write_text("# alpha\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# beta\n", encoding="utf-8")
    fake = _FakeJournalClient(fail_titles={"b.md"})
    monkeypatch.setattr(server, "_client", fake)

    out = asyncio.run(server.journal_index_dir(str(tmp_path)))

    assert "1 failed" in out
    assert fake.reported["replace"] is False
    assert fake.reconciled is None


def test_journal_index_dir_default_pattern_matches_markdown_case_insensitively(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("# readme\n", encoding="utf-8")
    (tmp_path / "ENV.MD").write_text("# env\n", encoding="utf-8")
    (tmp_path / "Plan.Md").write_text("# plan\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore me\n", encoding="utf-8")

    fake = _FakeJournalClient()
    monkeypatch.setattr(server, "_client", fake)

    out = asyncio.run(server.journal_index_dir(str(tmp_path)))

    assert {r["title"] for r in fake.records} == {"README.md", "ENV.MD", "Plan.Md"}
    assert {f["rel_path"] for f in fake.reported["files"]} == {"README.md", "ENV.MD", "Plan.Md"}
    assert "3 file(s)" in out


def test_journal_index_dir_skips_image_and_binary_even_with_broad_pattern(tmp_path, monkeypatch):
    """W161: a broad pattern ('*') must NEVER index rendered-image/binary files (SVG
    plots, PNGs, pickles) as prose — their bytes pollute semantic memory and swamp
    browse. Text data files (.json) are NOT blocked (narrow image/binary blocklist)."""
    (tmp_path / "notes.md").write_text("# real doc\n", encoding="utf-8")
    (tmp_path / "data.json").write_text('{"k": 1}\n', encoding="utf-8")
    (tmp_path / "plot.svg").write_text('<svg><path d="M0 0 L9 9"/></svg>', encoding="utf-8")
    (tmp_path / "chart.PNG").write_text("\x89PNG not really", encoding="utf-8")  # case-insensitive
    (tmp_path / "model.pkl").write_text("\x80\x04 fake", encoding="utf-8")
    fake = _FakeJournalClient()
    monkeypatch.setattr(server, "_client", fake)

    out = asyncio.run(server.journal_index_dir(str(tmp_path), pattern="*"))

    # image/binary (svg/png/pkl) skipped regardless of case; markdown + text-data kept
    assert {r["title"] for r in fake.records} == {"notes.md", "data.json"}
    assert {f["rel_path"] for f in fake.reported["files"]} == {"notes.md", "data.json"}
    assert "3 binary/image skipped" in out  # surfaced to the caller


def test_journal_index_dir_counts_unchanged_files(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("# alpha\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# beta\n", encoding="utf-8")
    # simulate a re-run where a.md is unchanged (backend returns deduped=True)
    fake = _FakeJournalClient(deduped_titles={"a.md"})
    monkeypatch.setattr(server, "_client", fake)

    out = asyncio.run(server.journal_index_dir(str(tmp_path)))

    assert "1 unchanged" in out


def _make_depth_tree(root):
    (root / "a.md").write_text("# a\n", encoding="utf-8")          # depth 0
    (root / "sub").mkdir()
    (root / "sub" / "b.md").write_text("# b\n", encoding="utf-8")  # depth 1
    (root / "sub" / "deep").mkdir()
    (root / "sub" / "deep" / "c.md").write_text("# c\n", encoding="utf-8")  # depth 2


def _index_rel_paths(tmp_path, monkeypatch, **kwargs):
    fake = _FakeJournalClient()
    monkeypatch.setattr(server, "_client", fake)
    asyncio.run(server.journal_index_dir(str(tmp_path), **kwargs))
    # Return the RAW reported paths — do NOT normalize here. Normalizing in the test
    # is exactly what masked W145 (Windows backslash paths reached the store while the
    # suite stayed green). rel_paths must already be forward-slash at the source.
    return {f["rel_path"] for f in (fake.reported["files"] if fake.reported else [])}


def test_journal_index_dir_non_recursive_is_shallow(tmp_path, monkeypatch):
    _make_depth_tree(tmp_path)
    assert _index_rel_paths(tmp_path, monkeypatch) == {"a.md"}


def test_journal_index_dir_recursive_unbounded_walks_all_depths(tmp_path, monkeypatch):
    _make_depth_tree(tmp_path)
    assert _index_rel_paths(tmp_path, monkeypatch, recursive=True) == {"a.md", "sub/b.md", "sub/deep/c.md"}


def test_journal_index_dir_max_depth_caps_recursion(tmp_path, monkeypatch):
    _make_depth_tree(tmp_path)
    # depth 0 (root) + depth 1 (immediate subdirs), but NOT depth 2
    assert _index_rel_paths(tmp_path, monkeypatch, recursive=True, max_depth=1) == {"a.md", "sub/b.md"}


def test_journal_index_dir_max_depth_zero_is_root_only(tmp_path, monkeypatch):
    _make_depth_tree(tmp_path)
    assert _index_rel_paths(tmp_path, monkeypatch, recursive=True, max_depth=0) == {"a.md"}


def test_journal_index_dir_rel_paths_are_forward_slash(tmp_path, monkeypatch):
    """W145: nested files must be reported with '/' separators, never OS backslashes,
    so the inventory tree / basename lookup works regardless of the client's OS."""
    _make_depth_tree(tmp_path)
    paths = _index_rel_paths(tmp_path, monkeypatch, recursive=True)
    assert all("\\" not in p for p in paths), paths
    assert "sub/deep/c.md" in paths


def test_journal_index_dir_titles_nested_docs_by_relative_path(tmp_path, monkeypatch):
    # Recursive docs can contain same-named files (README.md in many directories).
    # The doc title is the latest-wins identity, so it must use the relative path,
    # not basename, otherwise indexing one README supersedes the other.
    (tmp_path / "api").mkdir()
    (tmp_path / "user").mkdir()
    (tmp_path / "api" / "README.md").write_text("# API\n", encoding="utf-8")
    (tmp_path / "user" / "README.md").write_text("# User\n", encoding="utf-8")
    fake = _FakeJournalClient()
    monkeypatch.setattr(server, "_client", fake)

    asyncio.run(server.journal_index_dir(str(tmp_path), recursive=True))

    assert {r["title"] for r in fake.records} == {"api/README.md", "user/README.md"}


def test_journal_index_dir_respects_gitignore_by_default(tmp_path, monkeypatch):
    # A gitignored doc (a generated / draft .md) must not be indexed into the journal
    # by default when the directory is a git repo.
    git_init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("draft.md\n", encoding="utf-8")
    (tmp_path / "readme.md").write_text("# readme\n", encoding="utf-8")
    (tmp_path / "draft.md").write_text("# draft\n", encoding="utf-8")
    fake = _FakeJournalClient()
    monkeypatch.setattr(server, "_client", fake)

    out = asyncio.run(server.journal_index_dir(str(tmp_path)))
    assert {r["title"] for r in fake.records} == {"readme.md"}   # draft.md gitignored
    assert "1 gitignored" in out                                 # surfaced to the caller


def test_journal_index_dir_gitignore_can_be_disabled(tmp_path, monkeypatch):
    git_init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("draft.md\n", encoding="utf-8")
    (tmp_path / "readme.md").write_text("# readme\n", encoding="utf-8")
    (tmp_path / "draft.md").write_text("# draft\n", encoding="utf-8")
    fake = _FakeJournalClient()
    monkeypatch.setattr(server, "_client", fake)

    asyncio.run(server.journal_index_dir(str(tmp_path),
                                         respect_gitignore=False))
    assert {r["title"] for r in fake.records} == {"readme.md", "draft.md"}


# --------------------------- local chunking parallelism (async-ification) --------------------------- #

def _make_py_repo(tmp_path, n):
    for i in range(n):
        (tmp_path / f"f{i}.py").write_text(f"def f{i}(x):\n    return x + {i}\n", encoding="utf-8")


def _concurrency_probe():
    """Returns (probe_factory, state): probe wraps the real fn and records peak concurrency."""
    import threading
    import time
    state = {"cur": 0, "max": 0}
    lock = threading.Lock()

    def make(real):
        def probe(abs_path, rel, repo_root, content_hash=None):
            with lock:
                state["cur"] += 1
                state["max"] = max(state["max"], state["cur"])
            time.sleep(0.1)  # hold the slot so peers can overlap if run in parallel
            try:
                return real(abs_path, rel, repo_root, content_hash)
            finally:
                with lock:
                    state["cur"] -= 1
        return probe

    return make, state


def test_collect_code_files_chunks_files_concurrently(tmp_path, monkeypatch):
    """Local AST chunking runs files in parallel, not one-at-a-time (async-ification)."""
    _make_py_repo(tmp_path, 4)
    monkeypatch.setattr(server, "_CHUNK_WORKERS", 4)
    make, state = _concurrency_probe()
    monkeypatch.setattr(server, "_chunk_file_payload", make(server._chunk_file_payload))
    files = server._collect_code_files(str(tmp_path))
    assert len(files) == 4
    assert state["max"] >= 2, f"expected concurrent chunking, saw peak concurrency {state['max']}"


def test_collect_code_files_output_matches_serial(tmp_path):
    """Parallel collect stays byte-identical to a serial walk (order + content)."""
    from core.code_kb.chunker import iter_repo_files
    _make_py_repo(tmp_path, 5)
    got = server._collect_code_files(str(tmp_path))
    expected = []
    for abs_path, rel in iter_repo_files(str(tmp_path), respect_gitignore=True):
        f = server._chunk_file_payload(abs_path, rel, str(tmp_path))
        if f is not None:
            expected.append(f)
    assert got == expected


def test_collect_incremental_chunks_changed_files_concurrently(tmp_path, monkeypatch):
    """Incremental collect chunks CHANGED files in parallel too."""
    _make_py_repo(tmp_path, 4)
    monkeypatch.setattr(server, "_CHUNK_WORKERS", 4)
    make, state = _concurrency_probe()
    monkeypatch.setattr(server, "_chunk_file_payload", make(server._chunk_file_payload))
    payloads, manifest, current = server._collect_incremental(str(tmp_path), {})  # all changed
    assert len(payloads) == 4
    assert sorted(manifest) == [f"f{i}.py" for i in range(4)]
    assert state["max"] >= 2, f"incremental chunking not concurrent, peak {state['max']}"


def test_collect_code_files_emits_file_level_progress(tmp_path):
    """Observability: full collect reports files_done/files_total while walking."""
    _make_py_repo(tmp_path, 6)
    ticks: list[dict] = []
    files = server._collect_code_files(str(tmp_path), progress_cb=ticks.append)
    assert len(files) == 6
    assert ticks, "expected at least one progress tick"
    assert ticks[0]["files_total"] == 6
    assert ticks[0]["files_done"] == 0
    assert ticks[-1]["files_done"] == 6
    assert ticks[-1]["files_changed"] == 6
    assert ticks[-1]["chunks_ready"] > 0
    # Monotonic files_done
    done_seq = [t["files_done"] for t in ticks]
    assert done_seq == sorted(done_seq)


def test_collect_incremental_emits_file_level_progress(tmp_path):
    """Observability: incremental collect reports progress across all candidates."""
    _make_py_repo(tmp_path, 5)
    # Mark two files as already-known (unchanged) so only 3 are "changed".
    from core.code_kb.chunker import file_content_hash
    known = {
        "f0.py": file_content_hash(str(tmp_path / "f0.py")),
        "f1.py": file_content_hash(str(tmp_path / "f1.py")),
    }
    ticks: list[dict] = []
    payloads, manifest, current = server._collect_incremental(
        str(tmp_path), known, progress_cb=ticks.append)
    assert len(payloads) == 3
    assert ticks[0]["files_total"] == 5
    assert ticks[-1]["files_done"] == 5
    assert ticks[-1]["files_changed"] == 3
    assert ticks[-1]["chunks_ready"] > 0


def test_fmt_preparing_work_includes_file_and_chunk_counters():
    line = server._fmt_job_line({
        "job_id": "abc", "status": "preparing", "project": "/tmp/p",
        "phase": "scanning and chunking local files",
        "progress": 0.42,
        "files_done": 1347, "files_total": 3135, "files_changed": 900,
        "chunks_ready": 12000, "chunks_done": 0, "chunks_total": 0,
    })
    assert "[preparing]" in line
    assert "scanning and chunking local files" in line
    assert "files 1347/3135" in line
    assert "900 changed" in line
    assert "12000 chunks ready" in line
    assert "(42%)" in line


def test_index_code_status_shows_live_local_file_progress(tmp_path, monkeypatch):
    """While local AST walk runs, index_code_status must show files_done, not stuck 0%."""
    fake = _FakeCodeClient()
    monkeypatch.setattr(server, "_client", fake)
    collection_started = threading.Event()
    release_collection = threading.Event()

    def slow_collect(repo_root, respect_gitignore=True, exclude_globs=None, progress_cb=None):
        collection_started.set()
        # Mid-walk tick so status can observe non-zero files_done during preparing.
        if progress_cb is not None:
            progress_cb({
                "phase": "scanning and chunking local files",
                "files_total": 4, "files_done": 2, "files_changed": 2, "chunks_ready": 6,
            })
        release_collection.wait(timeout=5)
        if progress_cb is not None:
            progress_cb({
                "phase": "scanning and chunking local files",
                "files_total": 4, "files_done": 4, "files_changed": 4, "chunks_ready": 12,
            })
        return [{
            "rel_path": "sample.py", "language": "python", "chunks": [{
                "start_line": 1, "end_line": 1, "label": "",
                "content": "# path: sample.py\nx", "code": "x",
            }],
        }]

    monkeypatch.setattr(server, "_collect_code_files", slow_collect)

    async def scenario():
        call = asyncio.create_task(server.index_code(str(tmp_path), auto_refresh=False))
        while not collection_started.is_set():
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)  # let progress_cb land on status dict
        try:
            reply = await asyncio.wait_for(asyncio.shield(call), timeout=0.2)
            local_job_id = re.search(r"job ([0-9a-f]+)", reply).group(1)
            status = await server.index_code_status(local_job_id)
            assert "[preparing]" in status
            assert "files 2/4" in status
            assert "2 changed" in status
            assert "6 chunks ready" in status
            assert "(50%)" in status
            st = server._CODE_PREPARATION_STATUS[local_job_id]
            assert st["files_done"] == 2
            assert st["files_total"] == 4
            assert st["progress"] == 0.5
        finally:
            release_collection.set()
            await call

    asyncio.run(scenario())


def test_journal_index_dir_records_entries_concurrently(tmp_path, monkeypatch):
    """journal_index_dir (server mirror) records entries concurrently (bounded)."""
    import asyncio
    for i in range(4):
        (tmp_path / f"n{i}.md").write_text(f"# doc {i}\n\nbody {i}\n", encoding="utf-8")
    monkeypatch.setattr(server, "_JOURNAL_INDEX_CONCURRENCY", 4)
    state = {"cur": 0, "max": 0}

    class _ConcClient(_FakeJournalClient):
        async def record_journal(self, **kw):
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
            await asyncio.sleep(0.05)
            state["cur"] -= 1
            return await super().record_journal(**kw)

    monkeypatch.setattr(server, "_client", _ConcClient())
    out = asyncio.run(server.journal_index_dir(str(tmp_path), pattern="*.md",
                                               respect_gitignore=False))
    assert "Indexed 4 file(s)" in out
    assert state["max"] >= 2, f"entries must be recorded concurrently, peak {state['max']}"
def test_job_line_describes_delete_only_refresh():
    line = server._fmt_job_line({
        "job_id": "delete-job", "status": "completed", "project": "C:/repo",
        "chunks_done": 0, "chunks_total": 0, "progress": 1.0,
        "indexed": 0, "removed_total": 1,
    })

    assert "removed 1 path" in line


class TestIndexingAGitignoredSubtreeUnderAnIndexedParent:
    """Server mirror of the plugin's guard test: a child that widens the gitignore
    filter under a filtered parent must proceed with a warning, not be refused.

    Real FastAPI route, real IndexCatalogService, real SQLite; only the network hop
    is in-process ASGI. Kept in step with the same class in test_mcp_plugin.py —
    mcp_server/server.py and mcp_plugin/retrieval_mcp.py carry parallel copies of
    this guard and a fix applied to one alone would silently diverge.
    """

    @staticmethod
    def _kb_app(files, *, parent_respects_gitignore=True):
        from contextlib import contextmanager

        from fastapi import FastAPI
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool

        from app.api.routes import index_catalog as ic_routes
        from app.services.index_catalog.repository import IndexCatalogRepository
        from app.services.index_catalog.schemas import (
            IndexFileRecord, IndexReportRequest, IndexScopePolicyUpsert,
        )
        from app.services.index_catalog.service import IndexCatalogService
        from core.db.database import Base

        engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                               poolclass=StaticPool)
        Base.metadata.create_all(engine)
        session_local = sessionmaker(bind=engine, expire_on_commit=False)

        @contextmanager
        def session_factory():
            session = session_local()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        svc = IndexCatalogService(
            repository=IndexCatalogRepository(session_factory=session_factory))
        svc.report(IndexReportRequest(
            host="h1", project="/repo/erp", root_dir="/repo/erp", replace=True,
            files=[IndexFileRecord(rel_path=p, kind="code", collection="code_chunks",
                                   chunk_count=3) for p in files],
        ), owner_user_id="u1")
        svc.upsert_policy(IndexScopePolicyUpsert(
            host="h1", project="/repo/erp", root_dir="/repo/erp", kind="code",
            respect_gitignore=parent_respects_gitignore), owner_user_id="u1")

        ic_routes.get_index_catalog_service = lambda: svc
        app = FastAPI()
        app.include_router(ic_routes.router, prefix="/api/index")
        return app

    def _guard(self, monkeypatch, app, *, respect_gitignore):
        import httpx

        from mcp_server.backend import RetrievalClient

        async def scenario():
            async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://kb",
                    headers={"X-Retrival-User-Id": "u1"}) as http:
                monkeypatch.setattr(server, "_client", RetrievalClient(client=http))
                return await server._nested_scope_guard(
                    "/repo/erp/data", "h1", "code", respect_gitignore)

        return asyncio.run(scenario())

    def test_a_partly_ignored_subtree_indexes_with_a_warning(self, monkeypatch):
        block, warning = self._guard(
            monkeypatch, self._kb_app(["src/main.py", "data/README.md"]),
            respect_gitignore=False)
        assert block is None
        assert "/repo/erp" in warning
        assert "respect_gitignore=False" in warning

    def test_keeping_the_filter_on_is_still_refused(self, monkeypatch):
        block, warning = self._guard(
            monkeypatch, self._kb_app(["src/main.py", "data/README.md"]),
            respect_gitignore=True)
        assert block is not None
        assert "already covered by ancestor scope '/repo/erp'" in block
        assert warning == ""

    def test_a_wholly_ignored_subtree_indexes_without_a_word(self, monkeypatch):
        block, warning = self._guard(
            monkeypatch, self._kb_app(["src/main.py", "README.md"]),
            respect_gitignore=False)
        assert block is None
        assert warning == ""
