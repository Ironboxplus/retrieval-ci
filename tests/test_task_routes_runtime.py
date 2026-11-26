"""Tasks API: runner-registry dispatch for cancel/retry, and SSE per WHATWG HTML 9.2.

Real SQLite file; the app is the real tasks router driven through
httpx.ASGITransport. The route module's session factory is pointed at the test
database; nothing in the request path is faked. The runner registered below is a
small real class that writes real Task rows.
"""
from __future__ import annotations

import asyncio
import threading
import time

import httpx
import pytest
from fastapi import FastAPI

import app.api.routes.tasks as tasks_routes
import app.services.task_runtime as task_runtime
from app.services.pipeline import runner_registry
from app.services.pipeline.runner_registry import cancel_run, get_runner, register_runner
from app.services.pipeline.store import TaskStore
from core.db.database import get_db
from tests.sqlite_task_db import make_session_factory

TYPE = "unit_recording_runner"


class RecordingRunner:
    """A TaskRunner that really creates/cancels/retries Task rows in the test DB."""

    task_type = TYPE

    def __init__(self, store: TaskStore) -> None:
        self.store = store
        self.retried: list[int] = []
        self.retry_error: Exception | None = None
        self.retry_result: dict | None = None

    def start(self, params):
        task_id = self.store.create_task(task_type=TYPE, params=params)
        self.store.set_task_status(task_id, "running")
        return {"status": "started", "task_id": task_id}

    def cancel(self, task_id):
        return cancel_run(self.store, task_id, None)

    def retry(self, task_id):
        if self.retry_error is not None:
            raise self.retry_error
        if self.retry_result is not None:
            return self.retry_result
        self.retried.append(task_id)
        self.store.reset_pipeline_for_retry(task_id)
        return {"status": "started", "task_id": task_id}

    def resume_running(self):
        return []


@pytest.fixture
def env(tmp_path, monkeypatch):
    factory = make_session_factory(tmp_path / "tasks.db")
    store = TaskStore(factory)
    runner = RecordingRunner(store)
    register_runner(TYPE, lambda: runner)
    monkeypatch.setattr(tasks_routes, "get_db_session", factory)
    monkeypatch.setattr(task_runtime, "get_db_session", factory)

    def override_db():
        with factory() as session:
            yield session

    app = FastAPI()
    app.include_router(tasks_routes.router, prefix="/api/tasks")
    app.dependency_overrides[get_db] = override_db
    yield app, store, runner
    runner_registry._getters.pop(TYPE, None)


def client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


# --------------------------------------------------------------- registry
def test_registry_resolves_the_builtin_runners_by_task_type():
    assert {"ingest_pipeline", "upload_pipeline", "medical_corpus_download"} <= set(
        runner_registry.all_runners())
    assert get_runner("crawl_urls") is None


@pytest.mark.asyncio
async def test_cancel_goes_through_the_registered_runner(env):
    app, store, runner = env
    task_id = runner.start({"x": 1})["task_id"]
    async with client(app) as http:
        resp = await http.post(f"/api/tasks/{task_id}/cancel")
    assert resp.status_code == 200
    assert store.get_task(task_id)["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_of_a_finished_task_is_rejected(env):
    app, store, runner = env
    task_id = runner.start({})["task_id"]
    store.set_task_status(task_id, "completed")
    async with client(app) as http:
        resp = await http.post(f"/api/tasks/{task_id}/cancel")
    assert resp.status_code == 400
    assert store.get_task(task_id)["status"] == "completed"


@pytest.mark.asyncio
async def test_retry_goes_through_the_registered_runner(env):
    app, store, runner = env
    task_id = runner.start({})["task_id"]
    store.set_task_status(task_id, "failed", error="boom")
    async with client(app) as http:
        resp = await http.post(f"/api/tasks/{task_id}/retry")
    assert resp.status_code == 200
    assert runner.retried == [task_id]
    assert store.get_task(task_id)["status"] == "pending"


@pytest.mark.parametrize("error,code", [
    (ValueError("only failed tasks"), 400),
    (NotImplementedError("not retryable"), 400),
    (RuntimeError("host not designated"), 503),
])
@pytest.mark.asyncio
async def test_retry_errors_map_to_http_status(env, error, code):
    app, store, runner = env
    task_id = runner.start({})["task_id"]
    store.set_task_status(task_id, "failed")
    runner.retry_error = error
    async with client(app) as http:
        resp = await http.post(f"/api/tasks/{task_id}/retry")
    assert resp.status_code == code
    assert str(error) in resp.text


@pytest.mark.asyncio
async def test_retry_conflict_is_409_with_the_runner_message(env):
    app, store, runner = env
    task_id = runner.start({})["task_id"]
    store.set_task_status(task_id, "failed")
    runner.retry_result = {"status": "conflict", "task_id": 99, "error": "already running (#99)"}
    async with client(app) as http:
        resp = await http.post(f"/api/tasks/{task_id}/retry")
    assert resp.status_code == 409
    assert "already running (#99)" in resp.text


@pytest.mark.asyncio
async def test_unregistered_task_type_keeps_the_legacy_cancel_path(env):
    app, store, _ = env
    task_id = store.create_task(task_type="crawl_urls", params={"urls": []})
    async with client(app) as http:
        resp = await http.post(f"/api/tasks/{task_id}/cancel")
    assert resp.status_code == 200
    assert store.get_task(task_id)["status"] == "cancelled"


# ------------------------------------------------------------ terminal listener
def test_terminal_listener_fires_after_commit_for_both_writers(tmp_path):
    store = TaskStore(make_session_factory(tmp_path / "t.db"))
    seen = []

    def listener(task_id, task_type, status):
        # Called after the commit: the new status must already be readable.
        seen.append((task_id, task_type, status, store.get_task(task_id)["status"]))

    TaskStore.add_terminal_listener(listener)
    try:
        a = store.create_task(task_type="t")
        b = store.create_task(task_type="t")
        store.set_task_status(a, "running")                       # not terminal
        store.set_task_status(a, "failed")
        assert store.compare_and_set_status(b, expect={"pending"}, new="completed")
        assert not store.compare_and_set_status(b, expect={"pending"}, new="cancelled")  # lost: no event
    finally:
        TaskStore.remove_terminal_listener(listener)
    assert seen == [(a, "t", "failed", "failed"), (b, "t", "completed", "completed")]


def test_a_failing_terminal_listener_does_not_break_the_status_write(tmp_path):
    store = TaskStore(make_session_factory(tmp_path / "t.db"))

    def broken(*_):
        raise RuntimeError("listener bug")

    TaskStore.add_terminal_listener(broken)
    try:
        task_id = store.create_task(task_type="t")
        store.set_task_status(task_id, "failed")
    finally:
        TaskStore.remove_terminal_listener(broken)
    assert store.get_task(task_id)["status"] == "failed"


def test_list_by_origin_and_update_result(tmp_path):
    store = TaskStore(make_session_factory(tmp_path / "t.db"))
    mine = [store.create_task(task_type="t", params={"origin": {"schedule_id": 7}}) for _ in range(3)]
    store.create_task(task_type="t", params={"origin": {"schedule_id": 8}})
    store.create_task(task_type="t", params={})
    store.set_task_status(mine[0], "completed")
    store.set_task_status(mine[1], "running")

    assert [t["id"] for t in store.list_by_origin(7)] == mine[::-1]
    assert [t["id"] for t in store.list_by_origin(7, 2)] == mine[::-1][:2]
    assert [t["id"] for t in store.list_by_origin(7, statuses=("pending", "running"))] == [mine[2], mine[1]]
    assert store.list_by_origin(99) == []

    store.update_result(mine[1], {"new": 4})
    got = store.get_task(mine[1])
    assert got["result"] == {"new": 4} and got["status"] == "running"


# ------------------------------------------------------------------- SSE
def _seed_finished_task(store, events=5):
    task_id = store.create_task(task_type="t")
    store.set_task_status(task_id, "running")
    for i in range(events):
        store.emit_event(task_id, None, f"event {i}")
    store.set_task_status(task_id, "completed")
    return task_id


def _blocks(body: str) -> list[dict]:
    """Parse an event stream into [{field: value}] per blank-line-separated block."""
    out = []
    for raw in body.split("\n\n"):
        if not raw.strip():
            continue
        block = {}
        for line in raw.split("\n"):
            if line.startswith(":"):
                block.setdefault("comment", []).append(line[1:].strip())
                continue
            name, _, value = line.partition(": ")
            block[name] = value
        out.append(block)
    return out


@pytest.mark.parametrize("path", ["stream", "workflow/stream"])
@pytest.mark.asyncio
async def test_stream_declares_retry_and_ids_only_task_events(env, path):
    app, store, _ = env
    task_id = _seed_finished_task(store)
    async with client(app) as http:
        resp = await http.get(f"/api/tasks/{task_id}/{path}")
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.text.startswith("retry: 3000\n\n")

    blocks = _blocks(resp.text)
    task_events = [b for b in blocks if b.get("event") == "task-event"]
    assert len(task_events) == 5 and all(b["id"].isdigit() for b in task_events)
    assert [int(b["id"]) for b in task_events] == sorted(int(b["id"]) for b in task_events)
    # snapshot / final must not move the client's last-event-ID (HTML 9.2.6).
    assert all("id" not in b for b in blocks if b.get("event") in ("snapshot", "final"))
    assert blocks[-1]["event"] == "final"


@pytest.mark.parametrize("path", ["stream", "workflow/stream"])
@pytest.mark.asyncio
async def test_stream_resumes_after_last_event_id(env, path):
    app, store, _ = env
    task_id = _seed_finished_task(store)
    async with client(app) as http:
        everything = await http.get(f"/api/tasks/{task_id}/{path}")
        ids = [int(b["id"]) for b in _blocks(everything.text) if b.get("event") == "task-event"]
        resumed = await http.get(f"/api/tasks/{task_id}/{path}", headers={"Last-Event-ID": str(ids[2])})
    got = [int(b["id"]) for b in _blocks(resumed.text) if b.get("event") == "task-event"]
    assert got == ids[3:]


@pytest.mark.asyncio
async def test_stream_ignores_an_invalid_last_event_id_and_honours_the_larger_cursor(env):
    app, store, _ = env
    task_id = _seed_finished_task(store)
    async with client(app) as http:
        everything = await http.get(f"/api/tasks/{task_id}/stream")
        ids = [int(b["id"]) for b in _blocks(everything.text) if b.get("event") == "task-event"]
        bad = await http.get(f"/api/tasks/{task_id}/stream", headers={"Last-Event-ID": "not-a-number"})
        both = await http.get(
            f"/api/tasks/{task_id}/stream?after_event_id={ids[3]}", headers={"Last-Event-ID": str(ids[1])})
    assert [int(b["id"]) for b in _blocks(bad.text) if b.get("event") == "task-event"] == ids
    assert [int(b["id"]) for b in _blocks(both.text) if b.get("event") == "task-event"] == ids[4:]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["stream", "workflow/stream"])
async def test_catch_up_streams_a_long_backlog_without_one_second_pauses(env, path):
    """1,000 stored events used to take one sleep(1.0) per 200-event batch, and the
    `final` marker went out after the first batch while 800 events were still unsent."""
    from core.db.models import TaskEvent

    app, store, _ = env
    task_id = store.create_task(task_type="t")
    store.set_task_status(task_id, "running")
    with store._sf() as session:
        session.add_all(TaskEvent(task_id=task_id, event_type="stage.log", message=f"e{i}") for i in range(1000))
    store.set_task_status(task_id, "completed")

    started = time.monotonic()
    async with client(app) as http:
        resp = await asyncio.wait_for(http.get(f"/api/tasks/{task_id}/{path}"), timeout=10)
    elapsed = time.monotonic() - started

    blocks = _blocks(resp.text)
    ids = [int(b["id"]) for b in blocks if b.get("event") == "task-event"]
    assert len(ids) == 1000 and ids == sorted(ids)
    assert blocks[-1]["event"] == "final"            # only after the whole backlog
    assert elapsed < 2.0, f"catch-up took {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_idle_stream_sends_keepalive_comments_until_final(env, monkeypatch):
    app, store, _ = env
    monkeypatch.setattr(tasks_routes, "SSE_KEEPALIVE_SECONDS", 0.5)
    task_id = store.create_task(task_type="t")
    store.set_task_status(task_id, "running")

    def finish_later():
        time.sleep(2.2)
        store.set_task_status(task_id, "completed")

    threading.Thread(target=finish_later, daemon=True).start()
    async with client(app) as http:
        resp = await asyncio.wait_for(http.get(f"/api/tasks/{task_id}/stream"), timeout=8)
    blocks = _blocks(resp.text)
    assert any(b.get("comment") == ["keepalive"] for b in blocks)
    assert blocks[-1]["event"] == "final"


@pytest.mark.asyncio
async def test_stream_of_a_missing_task_is_404(env):
    app, _, _ = env
    async with client(app) as http:
        resp = await http.get("/api/tasks/424242/stream")
    assert resp.status_code == 404
