"""Journal bulk-index jobs are visible, live and cancellable on the Tasks page.

Real SQLite file shared by Journal, the index catalog and the Tasks tables (as in
production's single retrieval.db). The embedder is a deterministic local spy behind
``core.indexing.embedders.get_embedder`` - the same seam the existing Journal tests
use - and the vector store records groups instead of calling Qdrant. Nothing between
the HTTP routes, the queue, JournalService and SQLite is faked.
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass

import httpx
import numpy as np
import pytest
from fastapi import FastAPI

import app.api.routes.journal as journal_routes
import app.api.routes.tasks as tasks_routes
import app.services.journal.jobs as journal_jobs
import app.services.journal.service as journal_service_module
import app.services.task_runtime as task_runtime
from app.services.index_catalog.repository import IndexCatalogRepository
from app.services.index_catalog.service import IndexCatalogService
from app.services.journal.jobs import JournalBulkIndexQueue
from app.services.journal.repository import JournalRepository
from app.services.journal.schemas import JournalBulkDocItem, JournalBulkIndexRequest
from app.services.journal.service import JournalService
from app.services.journal.vector_store import JournalVectorStore
from app.services.pipeline import runner_registry
from app.services.pipeline.queue_tasks import QueueTasks
from app.services.pipeline.runner_registry import register_runner
from app.services.pipeline.store import TaskStore
from core.db.database import get_db
from core.db.models import Task
from core.indexing.embedders import EmbeddingResult
from tests.sqlite_task_db import make_session_factory

TYPE = "journal_index"
OWNER = "user-1"
BODY = "\n\n".join(
    f"## Section {i}\n\n" + " ".join(f"word{i}_{j}" for j in range(160)) for i in range(8))


class SpyEmbedder:
    def __init__(self, on_call=None):
        self.calls = 0
        self.on_call = on_call

    def embed_texts(self, texts, *, priority="interactive"):
        self.calls += 1
        if self.on_call is not None:
            self.on_call(self.calls)
        return EmbeddingResult(
            dense=np.asarray([[float(len(t)), 1.0] for t in texts]),
            sparse=[{"1": 1.0} for _ in texts],
        )


class RecordingVectorStore:
    def __init__(self, on_write=None):
        self.groups = []
        self.deleted = []
        self.on_write = on_write

    def upsert_chunk_groups(self, groups, *, on_batch=None):
        if self.on_write is not None:
            self.on_write()
        self.groups.extend(groups)
        written = sum(len(group.chunks) for group in groups)
        if on_batch is not None:
            on_batch(written, written)
        return written

    def delete_by_entry_ids(self, entry_ids):
        self.deleted.extend(entry_ids)
        return 0

    def delete_by_entry_id(self, entry_id):
        self.deleted.append(entry_id)
        return 0


def _item(title: str, body: str = BODY) -> JournalBulkDocItem:
    return JournalBulkDocItem(
        title=title, body=body, size_bytes=len(body.encode()),
        content_hash=hashlib.sha256(body.encode()).hexdigest())


def _request(title="notes/a.md", client_job_id=None, body=BODY) -> JournalBulkIndexRequest:
    return JournalBulkIndexRequest(
        project_slug="retrival", host="h1", root_dir="/repo", replace=False,
        files=[_item(title, body)], client_job_id=client_job_id,
        report_files=[{"rel_path": title, "kind": "doc", "collection": "journal_chunks",
                       "content_hash": _item(title, body).content_hash}])


@dataclass
class Env:
    queue: JournalBulkIndexQueue
    store: TaskStore
    embedder: SpyEmbedder
    vectors: RecordingVectorStore
    app: FastAPI


@pytest.fixture
def make_env(tmp_path, monkeypatch):
    saved_getters = dict(runner_registry._getters)

    def build(*, on_embed=None, on_write=None) -> Env:
        factory = make_session_factory(tmp_path / "app.db")
        monkeypatch.setattr(journal_service_module, "JOURNAL_EMBED_BATCH", 2)
        embedder = SpyEmbedder(on_embed)
        monkeypatch.setattr("core.indexing.embedders.get_embedder", lambda: embedder)
        vectors = RecordingVectorStore(on_write)
        service = JournalService(
            repository=JournalRepository(session_factory=factory), vector_store=vectors)
        catalog = IndexCatalogService(repository=IndexCatalogRepository(session_factory=factory))
        store = TaskStore(factory)
        queue = JournalBulkIndexQueue(
            service=service, catalog_service=catalog,
            tasks=QueueTasks(store, min_interval=0.0))

        monkeypatch.setattr(journal_jobs, "_QUEUE", queue)
        monkeypatch.setattr(journal_routes, "get_journal_index_queue", lambda: queue)
        monkeypatch.setattr(journal_routes, "resolve_owner", lambda *_a: OWNER)
        monkeypatch.setattr(tasks_routes, "get_db_session", factory)
        monkeypatch.setattr(task_runtime, "get_db_session", factory)
        register_runner(TYPE, lambda: journal_jobs.build_task_runner(store))

        def override_db():
            with factory() as session:
                yield session

        app = FastAPI()
        app.include_router(journal_routes.router, prefix="/api/journal")
        app.include_router(tasks_routes.router, prefix="/api/tasks")
        app.dependency_overrides[get_db] = override_db
        return Env(queue, store, embedder, vectors, app)

    yield build
    runner_registry._getters.clear()
    runner_registry._getters.update(saved_getters)


def http(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def wait_for(predicate, what: str, timeout: float = 15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


def only_row(store: TaskStore) -> dict:
    rows = store.list_active(TYPE)
    assert len(rows) == 1, rows
    return store.get_task(rows[0]["id"])


JOB_FIELDS = {
    "job_id", "project_slug", "host", "status", "files_total", "files_done", "chunks_total",
    "chunks_done", "indexed", "progress", "queue_position", "error", "created_at",
    "started_at", "finished_at", "skipped_binary",
}


@pytest.mark.asyncio
async def test_running_job_is_a_live_task_row_and_the_job_endpoint_keeps_its_fields(make_env):
    seen: list[tuple[str, int, int]] = []
    reached, release = threading.Event(), threading.Event()
    env_box: dict[str, Env] = {}

    def on_embed(call: int) -> None:
        row = only_row(env_box["env"].store)
        seen.append((row["status"], row["progress"], row["total"]))
        if call == 2:
            reached.set()
            assert release.wait(15)

    env = make_env(on_embed=on_embed)
    env_box["env"] = env
    try:
        async with http(env.app) as client:
            resp = await client.post(
                "/api/journal/bulk-index", json=_request(client_job_id="jid-live").model_dump())
            assert resp.status_code == 202
            assert await _wait(reached)

            row = only_row(env.store)
            assert row["status"] == "running"
            assert row["params"] == {
                "job_id": "jid-live", "project": "retrival", "host": "h1", "files_total": 1}
            assert row["owner_user_id"] == OWNER
            assert 0 < row["progress"] < row["total"]

            listed = (await client.get("/api/tasks", params={"task_type": TYPE})).json()
            assert [t["id"] for t in listed["tasks"]] == [row["id"]]
            assert listed["tasks"][0]["status"] == "running"

            job = (await client.get("/api/journal/bulk-index/jobs/jid-live")).json()
            assert set(job) == JOB_FIELDS
            assert job["status"] == "running"
            assert job["chunks_total"] == row["total"]
            assert 0 < job["chunks_done"] < job["chunks_total"]
            assert 0.0 < job["progress"] < 1.0
    finally:
        release.set()

    final = wait_for(
        lambda: (r := env.store.get_task(row["id"]))["status"] == "completed" and r,
        "task completed")
    assert final["progress"] == final["total"] == row["total"]
    assert final["result"]["files"] == 1
    assert final["result"]["chunks"] == final["result"]["indexed"] == row["total"]

    statuses = {s for s, _, _ in seen}
    assert statuses == {"running"}
    progress = [p for _, p, _ in seen]
    assert progress == sorted(progress) and progress[0] == 0 and progress[-1] > 0
    assert {t for _, _, t in seen} == {row["total"]}


async def _wait(event: threading.Event, timeout: float = 15.0) -> bool:
    import asyncio
    return await asyncio.to_thread(event.wait, timeout)


@pytest.mark.asyncio
async def test_cancel_through_the_tasks_api_stops_a_job_that_is_still_embedding(make_env):
    reached, release = threading.Event(), threading.Event()

    def on_embed(call: int) -> None:
        if call == 2:
            reached.set()
            assert release.wait(15)

    env = make_env(on_embed=on_embed)
    try:
        async with http(env.app) as client:
            await client.post(
                "/api/journal/bulk-index", json=_request(client_job_id="jid-cancel").model_dump())
            assert await _wait(reached)
            task_id = only_row(env.store)["id"]

            resp = await client.post(f"/api/tasks/{task_id}/cancel")
            assert resp.status_code == 200, resp.text
            release.set()

            row = wait_for(
                lambda: (r := env.store.get_task(task_id))["status"] == "cancelled" and r,
                "task cancelled")
            job = wait_for(
                lambda: (j := env.queue.get("jid-cancel", owner_user_id=OWNER))
                and j.status.value == "cancelled" and j, "job cancelled")
            # Embedding stopped at the batch boundary, nothing reached the vector store.
            assert env.embedder.calls == 2
            assert env.vectors.groups == []
            assert row["result"] is None and job.finished_at is not None

            body = (await client.get("/api/journal/bulk-index/jobs/jid-cancel")).json()
            assert body["status"] == "cancelled"
    finally:
        release.set()


@pytest.mark.asyncio
async def test_cancel_is_refused_with_409_once_the_job_started_writing_to_qdrant(make_env):
    write_started, release = threading.Event(), threading.Event()

    def on_write() -> None:
        write_started.set()
        assert release.wait(15)

    env = make_env(on_write=on_write)
    try:
        async with http(env.app) as client:
            await client.post(
                "/api/journal/bulk-index", json=_request(client_job_id="jid-commit").model_dump())
            assert await _wait(write_started)
            task_id = only_row(env.store)["id"]

            resp = await client.post(f"/api/tasks/{task_id}/cancel")
            assert resp.status_code == 409
            assert "Qdrant" in resp.text
            assert env.store.get_task(task_id)["status"] == "running"
            release.set()

            row = wait_for(
                lambda: (r := env.store.get_task(task_id))["status"] == "completed" and r,
                "task completed")
            assert row["result"]["indexed"] > 0
    finally:
        release.set()


def test_a_queued_job_has_a_pending_row_and_cancelling_it_leaves_the_running_job_alone(make_env):
    reached, release = threading.Event(), threading.Event()

    def on_embed(call: int) -> None:
        if call == 1:
            reached.set()
            assert release.wait(15)

    env = make_env(on_embed=on_embed)
    try:
        first = env.queue.submit(_request("notes/a.md", "first"), owner_user_id=OWNER)
        assert reached.wait(15)
        second = env.queue.submit(_request("notes/b.md", "second"), owner_user_id=OWNER)

        rows = {r["params"]["job_id"]: r for r in map(
            lambda x: env.store.get_task(x["id"]), env.store.list_active(TYPE))}
        assert rows[first.job_id]["status"] == "running"
        assert rows[second.job_id]["status"] == "pending"

        cancelled = env.queue.cancel(second.job_id, owner_user_id=OWNER)
        assert cancelled.status.value == "cancelled"
        assert env.store.get_task(rows[second.job_id]["id"])["status"] == "cancelled"
        assert env.store.get_task(rows[first.job_id]["id"])["status"] == "running"
    finally:
        release.set()

    wait_for(lambda: env.store.get_task(rows[first.job_id]["id"])["status"] == "completed",
             "first job completed")
    titles = {chunk.content for group in env.vectors.groups for chunk in group.chunks}
    assert env.store.get_task(rows[second.job_id]["id"])["status"] == "cancelled"
    assert not any("second" in text for text in titles)


def test_a_failed_job_row_carries_the_error(make_env):
    class Boom:
        def upsert_chunk_groups(self, groups, *, on_batch=None):
            raise RuntimeError("qdrant down")

        def delete_by_entry_ids(self, entry_ids):
            return 0

    env = make_env()
    env.queue.service.vector_store = Boom()
    job = env.queue.submit(_request(client_job_id="jid-fail"), owner_user_id=OWNER)

    row = wait_for(
        lambda: (r := env.store.get_task(1)) and r["status"] == "failed" and r,
        "task failed")
    assert "qdrant down" in row["error"]
    assert row["params"]["job_id"] == job.job_id


@pytest.mark.asyncio
async def test_a_binary_file_is_skipped_alone_and_the_rest_of_the_batch_is_indexed(make_env):
    """One NUL-carrying body used to fail the whole job (attpatt 2758/2760/2764: 8-12
    files lost to one paper text). It is now skipped by itself and named in the job and
    the task result; every other file of the request is indexed."""
    env = make_env()
    binary = "\x89PNG\r\n\x00\x00\x00\rIHDR\x00\x00binary"
    request = JournalBulkIndexRequest(
        project_slug="retrival", host="h1", root_dir="/repo", replace=False,
        client_job_id="jid-binary",
        files=[_item("notes/a.md"), _item("plots/p.png.md", binary), _item("notes/b.md")],
        report_files=[{"rel_path": t, "kind": "doc", "collection": "journal_chunks",
                       "content_hash": _item(t, b).content_hash}
                      for t, b in (("notes/a.md", BODY), ("plots/p.png.md", binary),
                                   ("notes/b.md", BODY))])
    async with http(env.app) as client:
        resp = await client.post("/api/journal/bulk-index", json=request.model_dump())
        assert resp.status_code == 202
        row = wait_for(
            lambda: (r := env.store.get_task(1)) and r["status"] in ("completed", "failed") and r,
            "task ended")
        assert row["status"] == "completed", row.get("error")
        assert row["result"]["files"] == 2
        assert row["result"]["skipped_binary"] == ["plots/p.png.md"]
        job = (await client.get("/api/journal/bulk-index/jobs/jid-binary")).json()
        assert job["status"] == "completed"
        assert job["skipped_binary"] == ["plots/p.png.md"]
    written = {group.entry_id for group in env.vectors.groups}
    assert len(written) == 2
    assert not any("\x00" in chunk.content for group in env.vectors.groups for chunk in group.chunks)


class _HeldQdrantBoundary:
    """Qdrant-shaped boundary that parks inside the second wait=True batch."""

    supports_sparse = False

    def __init__(self, store: TaskStore) -> None:
        self.task_store = store
        self.batches = 0
        self.heartbeat_in_first_batch = None
        self.in_second_batch = threading.Event()
        self.release = threading.Event()

    def _heartbeat(self):
        with self.task_store._sf() as session:
            return session.query(Task).one().last_heartbeat

    def _upsert_batch(self, points, *, wait):
        assert wait is True
        self.batches += 1
        if self.batches == 1:
            self.heartbeat_in_first_batch = self._heartbeat()
        if self.batches == 2:
            self.in_second_batch.set()
            assert self.release.wait(15)


def test_write_phase_keeps_heartbeating_and_progress_moving_between_qdrant_batches(make_env):
    """601 chunks are three 256-point waiting batches; a big job must not look dead."""
    env = make_env()
    boundary = _HeldQdrantBoundary(env.store)
    env.queue.service.vector_store = JournalVectorStore(
        collection_name="journal_chunks_test", store=boundary)
    big = "\n\n".join(
        # ~400 model tokens (5 per word): one chunk per section under the 520-token window
        f"## Section {i}\n\n" + " ".join(f"word{i}_{j}" for j in range(80)) for i in range(600))
    try:
        env.queue.submit(_request(client_job_id="jid-write", body=big), owner_user_id=OWNER)
        assert boundary.in_second_batch.wait(30)

        job = env.queue.get("jid-write", owner_user_id=OWNER)
        assert job.chunks_total > 512
        assert 0.95 < job.progress < 1.0
        assert boundary._heartbeat() > boundary.heartbeat_in_first_batch
        assert only_row(env.store)["status"] == "running"
    finally:
        boundary.release.set()

    wait_for(lambda: env.store.get_task(1)["status"] == "completed", "task completed")
    assert boundary.batches == 3
    assert env.queue.get("jid-write", owner_user_id=OWNER).progress == 1.0
