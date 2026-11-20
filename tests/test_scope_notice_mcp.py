"""MCP tools show the backend's deliver-once scope notice verbatim, at the very top.

Both clients (the plugin and the repository server) are driven through the real tool
functions, over real HTTP semantics (an ASGI transport into the real routes), against a
real SQLite catalog and a real Qdrant scratch collection. The client keeps no state: the
backend decides who gets the message, so what is checked here is only placement, that a
list call shows it once, and that rendering costs no extra request.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI

import mcp_server.server as server
from app.api.routes import code_kb as code_routes
from app.api.routes import journal as journal_routes
from app.services.code_kb.jobs import CodeIndexJobQueue, JobStatus
from mcp_server.backend import RetrievalClient
from tests.code_analysis_support import OWNER, make_request, vector_store  # noqa: F401
from tests.test_mcp_plugin import _load
# Fixtures and helpers of the backend-side tests: one scratch Qdrant + SQLite harness.
from tests.test_scope_notice_delivery import (  # noqa: F401
    _index, _journal_doc, _publish, code_env, journal_env)

NOTICE = "Duplicate code found: app/a.py <-> app/b.py (2 chunks)"


class _CountingTransport(httpx.ASGITransport):
    def __init__(self, app) -> None:
        super().__init__(app=app)
        self.requests: list[tuple[str, str, str]] = []

    async def handle_async_request(self, request):
        self.requests.append((request.method, request.url.path, request.url.query.decode()))
        return await super().handle_async_request(request)


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(code_routes.router, prefix="/api/code")
    app.include_router(journal_routes.router, prefix="/api/journal")
    return app


class _Front:
    """One MCP client under test, wired to the real backend app."""

    def __init__(self, kind: str, monkeypatch) -> None:
        self.kind = kind
        self.transport = _CountingTransport(_app())
        http = httpx.AsyncClient(base_url="http://test", transport=self.transport)
        if kind == "plugin":
            self.mod = _load()
            self.mod._CLIENT = http
        else:
            self.mod = server
            monkeypatch.setattr(
                server, "_client", RetrievalClient(base_url="http://test", client=http))

    @property
    def requests(self) -> list[tuple[str, str, str]]:
        return self.transport.requests

    def scope(self, tmp_path) -> tuple[str, str]:
        """(argument, canonical scope) for a directory that does not exist locally, so
        no file watcher starts and the path names only a backend scope."""
        argument = str(tmp_path / "repo")
        return argument, self.mod._scope_path(argument)

    def search_code(self, query, repo_root: str) -> str:
        return asyncio.run(self.mod.search_code(query, repo_root=repo_root, mode="dense"))

    def journal_search(self, query, scope_dir: str) -> str:
        return asyncio.run(self.mod.journal_search(query, scope_dir=scope_dir))

    def index_code_status(self, job_id: str) -> str:
        return asyncio.run(self.mod.index_code_status(job_id))


@pytest.fixture(params=["plugin", "server"])
def front(request, monkeypatch):
    return _Front(request.param, monkeypatch)


def _calls(front: _Front, path: str) -> list[tuple[str, str, str]]:
    return [call for call in front.requests if call[1] == path]


# ------------------------------------------------------------------------- search_code

def test_search_code_puts_the_notice_first_and_only_once(front, code_env, tmp_path):
    argument, proj = front.scope(tmp_path)
    _index(code_env, proj, {"app/a.py": [0, 1]})
    _publish(code_env.store, proj, NOTICE)

    first = front.search_code("basis 0", argument)
    assert first.startswith(NOTICE + "\n\nCode matches for: 'basis 0'"), first[:200]
    assert first.count(NOTICE) == 1
    second = front.search_code("basis 0", argument)
    assert NOTICE not in second and second.startswith("Code matches for: 'basis 0'")


def test_search_code_without_a_notice_renders_exactly_as_before(front, code_env, tmp_path):
    argument, proj = front.scope(tmp_path)
    _index(code_env, proj, {"app/a.py": [0, 1]})
    out = front.search_code("basis 0", argument)
    assert out.startswith("Code matches for: 'basis 0'")


def test_a_notice_costs_no_extra_request(front, code_env, tmp_path):
    argument, proj = front.scope(tmp_path)
    _index(code_env, proj, {"app/a.py": [0, 1]})
    _publish(code_env.store, proj, NOTICE)

    front.search_code("basis 0", argument)   # carries the notice
    front.search_code("basis 0", argument)   # carries none
    assert [(m, p) for m, p, _q in front.requests] == [("POST", "/api/code/search")] * 2


def test_a_list_search_shows_the_notice_once_above_every_query(front, code_env, tmp_path):
    argument, proj = front.scope(tmp_path)
    _index(code_env, proj, {"app/a.py": [0, 1]})
    _publish(code_env.store, proj, NOTICE)

    out = front.search_code(["basis 0", "basis 1", "basis 0"], argument)
    assert out.startswith(NOTICE + "\n\nQuery 1/3: 'basis 0'"), out[:200]
    assert out.count(NOTICE) == 1
    assert out.index("Query 1/3") < out.index("Query 2/3") < out.index("Query 3/3")
    assert len(_calls(front, "/api/code/search")) == 3

    again = front.search_code(["basis 0", "basis 1"], argument)
    assert NOTICE not in again and again.startswith("Query 1/2: 'basis 0'")


def test_concurrent_tool_calls_deliver_the_notice_to_exactly_one(front, code_env, tmp_path):
    argument, proj = front.scope(tmp_path)
    _index(code_env, proj, {"app/a.py": [0, 1]})
    _publish(code_env.store, proj, NOTICE)

    async def together() -> list[str]:
        return list(await asyncio.gather(*(
            front.mod.search_code("basis 0", repo_root=argument, mode="dense")
            for _ in range(6))))

    outputs = asyncio.run(together())
    assert sum(NOTICE in out for out in outputs) == 1
    assert sum(out.startswith(NOTICE) for out in outputs) == 1


def test_a_new_analysis_version_is_shown_again(front, code_env, tmp_path):
    argument, proj = front.scope(tmp_path)
    _index(code_env, proj, {"app/a.py": [0, 1]})
    _publish(code_env.store, proj, NOTICE)
    assert front.search_code("basis 0", argument).startswith(NOTICE)
    assert NOTICE not in front.search_code("basis 0", argument)

    newer = "New duplicate code: app/c.py <-> app/d.py"
    _publish(code_env.store, proj, newer)
    assert front.search_code("basis 0", argument).startswith(newer)


# ----------------------------------------------------------------------- journal_search

def test_journal_search_puts_the_notice_first_and_only_once(front, journal_env, tmp_path):
    argument, proj = front.scope(tmp_path)
    _journal_doc(journal_env, proj, "docs/a.md")
    _publish(journal_env.store, proj, NOTICE)

    first = front.journal_search("marker", argument)
    assert first.startswith(NOTICE + "\n\nFound "), first[:200]
    assert first.count(NOTICE) == 1
    second = front.journal_search("marker", argument)
    assert NOTICE not in second and second.startswith("Found ")
    assert len(_calls(front, "/api/journal/search")) == 2


def test_a_journal_miss_still_carries_the_notice_first(front, journal_env, tmp_path):
    argument, proj = front.scope(tmp_path)
    _journal_doc(journal_env, proj, "docs/a.md")
    _publish(journal_env.store, proj, NOTICE)

    out = front.journal_search("zzzznomatchzzzz", argument)
    assert out.startswith(NOTICE + "\n\n"), out[:200]
    assert "Found " not in out


def test_a_list_journal_search_shows_the_notice_once_at_the_top(front, journal_env, tmp_path):
    argument, proj = front.scope(tmp_path)
    _journal_doc(journal_env, proj, "docs/a.md")
    _publish(journal_env.store, proj, NOTICE)

    out = front.journal_search(["marker", "shared"], argument)
    assert out.startswith(NOTICE + "\n\nQuery 1/2: 'marker'"), out[:200]
    assert out.count(NOTICE) == 1


# ------------------------------------------------------------------------ index_code_status

@pytest.fixture()
def finished_job(code_env, monkeypatch):
    queue = CodeIndexJobQueue(service=code_env.service, catalog_service=code_env.catalog.service)
    monkeypatch.setattr(code_routes, "get_existing_code_index_queue", lambda: queue)

    def run(project: str) -> str:
        import time
        job = queue.submit(make_request(project, {"app/a.py": [0, 1]}, []),
                           owner_user_id=OWNER)
        deadline = time.time() + 20
        while queue.get(job.job_id).status is not JobStatus.COMPLETED and time.time() < deadline:
            time.sleep(0.02)
        assert queue.get(job.job_id).status is JobStatus.COMPLETED
        return job.job_id

    try:
        yield run
    finally:
        queue.shutdown(timeout=5)


def test_index_code_status_puts_the_notice_first_and_only_once(
        front, code_env, finished_job, tmp_path):
    _argument, proj = front.scope(tmp_path)
    job_id = finished_job(proj)
    _publish(code_env.store, proj, NOTICE)

    first = front.index_code_status(job_id)
    assert first.startswith(NOTICE + "\n\nJob " + job_id), first[:200]
    assert first.count(NOTICE) == 1
    second = front.index_code_status(job_id)
    assert NOTICE not in second and second.startswith("Job " + job_id)
    assert [q for _m, p, q in front.requests if p == f"/api/code/jobs/{job_id}"] == [
        "scope_notice=true", "scope_notice=true"]
