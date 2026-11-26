"""Shared adapter that mirrors the in-process job queues onto Tasks rows.

Real SQLite file (tests/sqlite_task_db.py); the Tasks API is the real router driven
through httpx.ASGITransport. Only the clock is injected, to make the progress
throttle deterministic.
"""
from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from loguru import logger

import app.api.routes.tasks as tasks_routes
import app.services.task_runtime as task_runtime
from app.services.pipeline import runner_registry
from app.services.pipeline.queue_tasks import (
    INTERRUPTED_ERROR,
    QUEUE_TASK_KEEP_COMPLETED,
    QueueTaskRunner,
    QueueTasks,
)
from app.services.pipeline.runner_registry import TaskNotCancellable, register_runner
from app.services.pipeline.store import TaskStore
from core.db.database import get_db
from tests.sqlite_task_db import make_session_factory

TYPE = "unit_queue_job"


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def store(tmp_path):
    return TaskStore(make_session_factory(tmp_path / "tasks.db"))


def _events(store: TaskStore, task_id: int) -> list[str]:
    with store._sf() as s:
        from core.db.models import TaskEvent
        return [e.event_type for e in
                s.query(TaskEvent).filter(TaskEvent.task_id == task_id).order_by(TaskEvent.id)]


def test_open_creates_a_pending_row_with_params_total_owner_and_created_event(store):
    tasks = QueueTasks(store)
    task_id = tasks.open(TYPE, {"job_id": "j1", "project": "p"}, 40, "user-1")

    row = store.get_task(task_id)
    assert row["task_type"] == TYPE
    assert row["status"] == "pending"
    assert row["params"] == {"job_id": "j1", "project": "p"}
    assert (row["progress"], row["total"]) == (0, 40)
    assert row["owner_user_id"] == "user-1"
    assert _events(store, task_id) == ["task.created"]


def test_started_then_finished_completed_publishes_result_and_fills_progress(store):
    tasks = QueueTasks(store)
    task_id = tasks.open(TYPE, {}, 10, None)
    tasks.started(task_id)
    assert store.get_task(task_id)["status"] == "running"
    assert store.get_task(task_id)["started_at"] is not None

    tasks.finished(task_id, "completed", result={"indexed": 10})
    row = store.get_task(task_id)
    assert row["status"] == "completed"
    assert row["result"] == {"indexed": 10}
    assert row["progress"] == row["total"] == 10
    assert row["finished_at"] is not None


def test_finished_failed_and_cancelled_record_error_and_do_not_fill_progress(store):
    tasks = QueueTasks(store)
    failed = tasks.open(TYPE, {}, 10, None)
    tasks.started(failed)
    tasks.progress(failed, 4, 10)
    tasks.finished(failed, "failed", error="boom")
    assert store.get_task(failed)["status"] == "failed"
    assert store.get_task(failed)["error"] == "boom"
    assert store.get_task(failed)["progress"] == 4

    cancelled = tasks.open(TYPE, {}, 10, None)
    tasks.finished(cancelled, "cancelled")  # cancelled while still queued
    assert store.get_task(cancelled)["status"] == "cancelled"


def test_progress_is_throttled_to_one_write_per_interval_but_final_value_always_lands(store):
    clock = Clock()
    tasks = QueueTasks(store, clock=clock)
    task_id = tasks.open(TYPE, {}, 100, None)
    tasks.started(task_id)

    tasks.progress(task_id, 10, 100)          # first write always lands
    tasks.progress(task_id, 20, 100)          # same instant: dropped
    clock.now += 0.4
    tasks.progress(task_id, 30, 100)          # inside the interval: dropped
    assert store.get_task(task_id)["progress"] == 10

    clock.now += 0.7                          # 1.1s since the last write
    tasks.progress(task_id, 40, 100)
    assert store.get_task(task_id)["progress"] == 40

    tasks.progress(task_id, 100, 100)         # done == total bypasses the throttle
    assert store.get_task(task_id)["progress"] == 100


def _last_heartbeat(store: TaskStore, task_id: int):
    with store._sf() as s:
        from core.db.models import Task
        return s.get(Task, task_id).last_heartbeat


def test_heartbeat_refreshes_only_last_heartbeat_and_is_throttled_like_progress(store):
    clock = Clock()
    tasks = QueueTasks(store, clock=clock)
    task_id = tasks.open(TYPE, {}, 100, None)
    tasks.started(task_id)
    tasks.progress(task_id, 40, 100)
    before = _last_heartbeat(store, task_id)

    clock.now += 2.0                          # past the interval since the progress write
    tasks.heartbeat(task_id)
    first = _last_heartbeat(store, task_id)
    assert first > before
    row = store.get_task(task_id)
    assert (row["status"], row["progress"], row["total"]) == ("running", 40, 100)

    clock.now += 0.4                          # inside the interval: dropped
    tasks.heartbeat(task_id)
    assert _last_heartbeat(store, task_id) == first

    clock.now += 0.7
    tasks.heartbeat(task_id)
    assert _last_heartbeat(store, task_id) > first
    tasks.heartbeat(None)                     # an unrecorded job is a silent no-op


def test_progress_sets_total_once_the_queue_learns_it(store):
    tasks = QueueTasks(store)
    task_id = tasks.open(TYPE, {}, 0, None)   # chunk count unknown until chunking ran
    tasks.started(task_id)
    tasks.progress(task_id, 0, 15082)
    row = store.get_task(task_id)
    assert (row["progress"], row["total"]) == (0, 15082)


def test_finished_never_overwrites_a_terminal_row(store):
    tasks = QueueTasks(store)
    task_id = tasks.open(TYPE, {}, 5, None)
    tasks.started(task_id)
    assert store.compare_and_set_status(
        task_id, expect={"pending", "running"}, new="cancelled")

    tasks.finished(task_id, "completed", result={"indexed": 5})
    row = store.get_task(task_id)
    assert row["status"] == "cancelled"
    assert row["result"] is None


def test_completed_rows_are_pruned_to_the_newest_twenty_per_type_only(store):
    tasks = QueueTasks(store)
    kept_other_states = []
    for state in ("failed", "cancelled"):
        for _ in range(3):
            task_id = tasks.open(TYPE, {}, 1, None)
            tasks.finished(task_id, state, error="x" if state == "failed" else None)
            kept_other_states.append(task_id)
    other_type = tasks.open("unit_other_job", {}, 1, None)
    tasks.finished(other_type, "completed")

    completed = []
    for _ in range(QUEUE_TASK_KEEP_COMPLETED + 5):
        task_id = tasks.open(TYPE, {}, 1, None)
        tasks.finished(task_id, "completed", result={})
        completed.append(task_id)

    with store._sf() as s:
        from core.db.models import Task
        alive = {t.id for t in s.query(Task).filter(Task.task_type == TYPE)}
        other_alive = s.get(Task, other_type) is not None
    assert QUEUE_TASK_KEEP_COMPLETED == 20
    assert alive == set(kept_other_states) | set(completed[-20:])
    assert other_alive


def test_tasks_adapter_logs_a_store_failure_once_and_never_raises(tmp_path):
    working = make_session_factory(tmp_path / "tasks.db")
    broken = {"on": False}

    from contextlib import contextmanager

    @contextmanager
    def flaky():
        if broken["on"]:
            raise RuntimeError("database is locked")
        with working() as session:
            yield session

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        tasks = QueueTasks(TaskStore(flaky))
        task_id = tasks.open(TYPE, {}, 3, None)
        broken["on"] = True
        tasks.started(task_id)
        tasks.progress(task_id, 1, 3)
        tasks.heartbeat(task_id)
        tasks.finished(task_id, "completed")
        assert tasks.open(TYPE, {}, 3, None) is None
        # Every later call on an unrecorded job is a silent no-op.
        tasks.started(None)
        tasks.progress(None, 1, 3)
        tasks.finished(None, "completed")
    finally:
        logger.remove(sink)
    assert len([m for m in messages if "queue_tasks" in m]) == 1


def test_runner_resume_running_fails_leftover_rows_of_its_own_type_only(store):
    tasks = QueueTasks(store)
    pending = tasks.open(TYPE, {"job_id": "a"}, 1, None)
    running = tasks.open(TYPE, {"job_id": "b"}, 1, None)
    tasks.started(running)
    done = tasks.open(TYPE, {"job_id": "c"}, 1, None)
    tasks.finished(done, "completed")
    other = tasks.open("unit_other_job", {}, 1, None)
    tasks.started(other)

    runner = QueueTaskRunner(TYPE, cancel_job=lambda job_id, owner: True, store=store)
    assert runner.resume_running() == []

    for task_id in (pending, running):
        row = store.get_task(task_id)
        assert row["status"] == "failed"
        assert row["error"] == INTERRUPTED_ERROR
    assert store.get_task(done)["status"] == "completed"
    assert store.get_task(other)["status"] == "running"


def test_runner_start_and_retry_are_not_offered(store):
    runner = QueueTaskRunner(TYPE, cancel_job=lambda job_id, owner: True, store=store)
    with pytest.raises(NotImplementedError):
        runner.start({})
    with pytest.raises(NotImplementedError, match="客户端"):
        runner.retry(1)


def test_runner_cancel_delegates_by_job_id_and_row_owner(store):
    calls: list[tuple[str, str | None]] = []

    def cancel_job(job_id, owner):
        calls.append((job_id, owner))
        return True

    task_id = QueueTasks(store).open(TYPE, {"job_id": "j9"}, 1, "user-7")
    runner = QueueTaskRunner(TYPE, cancel_job=cancel_job, store=store)
    assert runner.cancel(task_id) is True
    assert calls == [("j9", "user-7")]


def test_runner_cancel_of_a_job_the_queue_no_longer_knows_settles_the_orphan_row(store):
    def cancel_job(job_id, owner):
        raise KeyError(job_id)

    task_id = QueueTasks(store).open(TYPE, {"job_id": "gone"}, 1, "user-7")
    runner = QueueTaskRunner(TYPE, cancel_job=cancel_job, store=store)
    assert runner.cancel(task_id) is True
    assert store.get_task(task_id)["status"] == "cancelled"


def test_runner_cancel_propagates_not_cancellable(store):
    def cancel_job(job_id, owner):
        raise TaskNotCancellable("已开始写入向量库")

    task_id = QueueTasks(store).open(TYPE, {"job_id": "j"}, 1, None)
    runner = QueueTaskRunner(TYPE, cancel_job=cancel_job, store=store)
    with pytest.raises(TaskNotCancellable):
        runner.cancel(task_id)
    assert store.get_task(task_id)["status"] == "pending"


def test_registry_resolves_the_three_queue_task_types():
    assert {"journal_index", "code_index", "public_code_job"} <= set(
        runner_registry.all_runners())


@pytest.fixture
def api(tmp_path, monkeypatch):
    factory = make_session_factory(tmp_path / "tasks.db")
    store = TaskStore(factory)
    monkeypatch.setattr(tasks_routes, "get_db_session", factory)
    monkeypatch.setattr(task_runtime, "get_db_session", factory)

    def override_db():
        with factory() as session:
            yield session

    app = FastAPI()
    app.include_router(tasks_routes.router, prefix="/api/tasks")
    app.dependency_overrides[get_db] = override_db
    saved = dict(runner_registry._getters)
    yield app, store
    runner_registry._getters.clear()
    runner_registry._getters.update(saved)


def client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_cancel_endpoint_maps_not_cancellable_to_409_with_the_reason(api):
    app, store = api

    def cancel_job(job_id, owner):
        raise TaskNotCancellable("代码索引已开始写入向量库，无法取消")

    register_runner(TYPE, lambda: QueueTaskRunner(TYPE, cancel_job=cancel_job, store=store))
    task_id = QueueTasks(store).open(TYPE, {"job_id": "j"}, 1, None)
    QueueTasks(store).started(task_id)

    async with client(app) as http:
        resp = await http.post(f"/api/tasks/{task_id}/cancel")
    assert resp.status_code == 409
    assert "代码索引已开始写入向量库" in resp.text
    assert store.get_task(task_id)["status"] == "running"


@pytest.mark.asyncio
async def test_retry_endpoint_tells_the_client_to_resubmit(api):
    app, store = api
    register_runner(TYPE, lambda: QueueTaskRunner(
        TYPE, cancel_job=lambda job_id, owner: True, store=store))
    tasks = QueueTasks(store)
    task_id = tasks.open(TYPE, {"job_id": "j"}, 1, None)
    tasks.finished(task_id, "failed", error="x")

    async with client(app) as http:
        resp = await http.post(f"/api/tasks/{task_id}/retry")
    assert resp.status_code == 400
    assert "客户端" in resp.text
