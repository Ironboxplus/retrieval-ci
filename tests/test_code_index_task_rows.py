"""Code-KB index jobs are mirrored onto Tasks rows without changing their queue semantics.

Same harness as tests/test_code_kb_jobs.py (in-memory vector store, deterministic
embed_fn) plus a real SQLite file for the Tasks tables and the real Tasks router.
"""
from __future__ import annotations

import asyncio
import threading
import time

import httpx
import pytest
from fastapi import FastAPI

import app.api.routes.tasks as tasks_routes
import app.services.code_kb.jobs as code_jobs
import app.services.code_kb.service as code_service_module
import app.services.task_runtime as task_runtime
from app.services.code_kb.jobs import CodeIndexJobQueue, JobStatus
from app.services.code_kb.schemas import CodeChunkIn, CodeFileIn, CodeIndexRequest
from app.services.code_kb.service import CodeKBService
from app.services.pipeline import runner_registry
from app.services.pipeline.queue_tasks import QueueTasks
from app.services.pipeline.runner_registry import register_runner
from app.services.pipeline.store import TaskStore
from core.db.database import get_db
from tests.sqlite_task_db import make_session_factory
from tests.test_code_kb_api import FakeStore, _embed
from tests.test_code_kb_jobs import _report, _req, _SpyCatalog

TYPE = "code_index"


@pytest.fixture
def env(tmp_path, monkeypatch):
    saved_getters = dict(runner_registry._getters)
    factory = make_session_factory(tmp_path / "tasks.db")
    store = TaskStore(factory)
    monkeypatch.setattr(code_service_module, "EMBED_BATCH", 2)
    monkeypatch.setattr(tasks_routes, "get_db_session", factory)
    monkeypatch.setattr(task_runtime, "get_db_session", factory)
    register_runner(TYPE, lambda: code_jobs.build_task_runner(store))

    def override_db():
        with factory() as session:
            yield session

    app = FastAPI()
    app.include_router(tasks_routes.router, prefix="/api/tasks")
    app.dependency_overrides[get_db] = override_db

    queues: list[CodeIndexJobQueue] = []

    def build(*, embed_fn=_embed, vector_store=None) -> CodeIndexJobQueue:
        service = CodeKBService(embed_fn=embed_fn, vector_store=vector_store or FakeStore())
        queue = CodeIndexJobQueue(
            service=service, catalog_service=_SpyCatalog(),
            tasks=QueueTasks(store, min_interval=0.0))
        monkeypatch.setattr(code_jobs, "_QUEUE", queue)
        queues.append(queue)
        return queue

    yield build, store, app
    for queue in queues:
        queue.shutdown(timeout=5)
    runner_registry._getters.clear()
    runner_registry._getters.update(saved_getters)


def _three_file_req(project: str) -> CodeIndexRequest:
    """Three 2-chunk files: with EMBED_BATCH=2 each file is its own embed batch."""
    files = [
        CodeFileIn(rel_path=f"f{i}.py", language="python", chunks=[
            CodeChunkIn(start_line=j, end_line=j + 5, label=f"f{i}_{j}",
                        content=f"qdrant upsert vector {i} {j}", code=f"def f{i}_{j}(): ...")
            for j in range(2)])
        for i in range(3)]
    return CodeIndexRequest(project=project, host="h1", root_dir="/repo", files=files)


def wait_for(predicate, what: str, timeout: float = 15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


def row_for(store: TaskStore, job_id: str) -> dict | None:
    with store._sf() as s:
        from core.db.models import Task
        ids = [t.id for t in s.query(Task).filter(Task.task_type == TYPE).order_by(Task.id)
               if (t.params_json or {}).get("job_id") == job_id]
    return store.get_task(ids[0]) if ids else None


def client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def test_submitted_job_becomes_a_running_then_completed_task_row(env):
    build, store, _app = env
    reached, release = threading.Event(), threading.Event()
    calls = []

    def gated_embed(texts):
        calls.append(len(texts))
        if len(calls) == 2:
            reached.set()
            assert release.wait(15)
        return _embed(texts)

    queue = build(embed_fn=gated_embed)
    request = _three_file_req("proj-live").model_copy(
        update={"report": _report(("f0.py", "f1.py", "f2.py")), "removed_paths": ["gone.py"]})
    job = queue.submit(request, owner_user_id="u1")

    # The row exists at submit time, before the worker may have picked the job up.
    assert row_for(store, job.job_id) is not None
    assert reached.wait(15)
    row = row_for(store, job.job_id)
    assert row["status"] == "running"
    assert row["params"] == {
        "job_id": job.job_id, "project": "proj-live", "host": "h1",
        "files_total": 3, "chunks_total": 6, "removed_total": 1}
    assert row["owner_user_id"] == "u1"
    assert row["total"] == 6 and 0 < row["progress"] < 6

    release.set()
    done = wait_for(lambda: (r := row_for(store, job.job_id))["status"] == "completed" and r,
                    "row completed")
    assert done["progress"] == done["total"] == 6
    assert done["result"] == {"indexed": 6, "removed_total": 1, "duplicates_updated": 0}
    assert queue.get(job.job_id).status is JobStatus.COMPLETED


def test_a_cancelled_queued_job_settles_its_row(env):
    build, store, _app = env
    gate = threading.Event()
    queue = build(embed_fn=lambda texts: (gate.wait(15), _embed(texts))[1])
    try:
        blocker = queue.submit(_req(project="p-block"), owner_user_id="u1")
        wait_for(lambda: queue.get(blocker.job_id).status is JobStatus.RUNNING, "blocker running")
        queued = queue.submit(_req(project="p-queued"), owner_user_id="u1")
        assert row_for(store, queued.job_id)["status"] == "pending"

        queue.cancel(queued.job_id, owner_user_id="u1")
        assert row_for(store, queued.job_id)["status"] == "cancelled"
        assert row_for(store, blocker.job_id)["status"] == "running"
    finally:
        gate.set()
    wait_for(lambda: row_for(store, blocker.job_id)["status"] == "completed", "blocker done")


def test_a_failed_job_settles_its_row_with_the_error(env):
    build, store, _app = env

    def broken_embed(texts):
        raise RuntimeError("embedder exploded")

    queue = build(embed_fn=broken_embed)
    job = queue.submit(_req(project="p-fail"), owner_user_id="u1")
    row = wait_for(lambda: (r := row_for(store, job.job_id))["status"] == "failed" and r,
                   "row failed")
    assert "embedder exploded" in row["error"]


@pytest.mark.asyncio
async def test_cancelling_a_committing_job_through_the_tasks_api_is_409(env):
    build, store, app = env
    write_started, release = threading.Event(), threading.Event()

    class BlockingStore(FakeStore):
        def upsert(self, **kwargs):
            write_started.set()
            assert release.wait(15)
            return super().upsert(**kwargs)

    queue = build(vector_store=BlockingStore())
    try:
        job = queue.submit(_req(project="p-commit"), owner_user_id="u1")
        assert await asyncio.to_thread(write_started.wait, 15)
        task_id = row_for(store, job.job_id)["id"]

        async with client(app) as http:
            resp = await http.post(f"/api/tasks/{task_id}/cancel")
        assert resp.status_code == 409
        assert "写入" in resp.text
        assert row_for(store, job.job_id)["status"] == "running"
    finally:
        release.set()
    wait_for(lambda: row_for(store, job.job_id)["status"] == "completed", "row completed")


@pytest.mark.asyncio
async def test_cancelling_an_embedding_job_through_the_tasks_api_settles_the_row(env):
    build, store, app = env
    reached, release = threading.Event(), threading.Event()
    calls = []

    def gated_embed(texts):
        calls.append(len(texts))
        if len(calls) == 1:
            reached.set()
            assert release.wait(15)
        return _embed(texts)

    queue = build(embed_fn=gated_embed)
    try:
        job = queue.submit(_req(project="p-embed", n_chunks=6), owner_user_id="u1")
        assert await asyncio.to_thread(reached.wait, 15)
        task_id = row_for(store, job.job_id)["id"]

        async with client(app) as http:
            resp = await http.post(f"/api/tasks/{task_id}/cancel")
        assert resp.status_code == 200, resp.text
    finally:
        release.set()
    wait_for(lambda: row_for(store, job.job_id)["status"] == "cancelled", "row cancelled")
    assert queue.get(job.job_id).status is JobStatus.CANCELLED


def test_a_job_superseded_by_project_deletion_fails_its_queued_row(env):
    build, store, _app = env
    gate = threading.Event()
    queue = build(embed_fn=lambda texts: (gate.wait(15), _embed(texts))[1])
    try:
        blocker = queue.submit(_req(project="p-other"), owner_user_id="u1")
        wait_for(lambda: queue.get(blocker.job_id).status is JobStatus.RUNNING, "blocker running")
        queued = queue.submit(_req(project="p-gone"), owner_user_id="u1")

        assert queue.cancel_project(owner_user_id="u1", project="p-gone") == 1
        row = row_for(store, queued.job_id)
        assert row["status"] == "failed"
        assert "project deletion" in row["error"]
    finally:
        gate.set()
