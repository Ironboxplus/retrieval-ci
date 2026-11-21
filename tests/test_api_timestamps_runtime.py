"""Every timestamp the task/schedule APIs return is RFC 3339 with an explicit offset
and denotes the right instant.

The stored columns are naive in two conventions (see core/timestamps.py): the
database default (`created_at`, `updated_at`) is UTC, Python's `datetime.now()`
(`started_at`, `completed_at`, `last_heartbeat`) is server-local. Each test stores a
value *the way its real writer would have* for a known instant and asserts the
endpoint returns that instant, independent of the machine's timezone.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import httpx
import pytest
from fastapi import FastAPI

import app.api.routes.tasks as tasks_routes
import app.services.task_runtime as task_runtime
from app.services.pipeline.store import TaskStore
from core.db.database import get_db
from core.db.models import (
    Task,
    TaskEvent,
    TaskResourceSample,
    TaskStatus,
    TaskSubtask,
    TaskSubtaskStatus,
    TaskWorkflowEdge,
)
from tests.sqlite_task_db import make_session_factory

# One known instant per column kind, so a wrong convention shows up as a wrong hour.
CREATED = datetime(2026, 11, 3, 3, 35, 12, 250000, tzinfo=timezone.utc)
STARTED = datetime(2026, 11, 3, 3, 40, 0, tzinfo=timezone.utc)
FINISHED = datetime(2026, 11, 3, 3, 50, 30, tzinfo=timezone.utc)
HEARTBEAT = datetime(2026, 11, 3, 3, 49, 59, tzinfo=timezone.utc)
SAMPLED = datetime(2026, 11, 3, 3, 41, 0, tzinfo=timezone.utc)

OFFSET = re.compile(r"(Z|[+-]\d{2}:\d{2})$")


def as_db_default(instant: datetime) -> datetime:
    """What CURRENT_TIMESTAMP stores: naive UTC."""
    return instant.astimezone(timezone.utc).replace(tzinfo=None)


def as_datetime_now(instant: datetime) -> datetime:
    """What datetime.now() would have stored at that instant: naive server-local."""
    return instant.astimezone().replace(tzinfo=None)


def assert_instant(text: str, instant: datetime) -> None:
    assert isinstance(text, str) and OFFSET.search(text), f"no explicit offset: {text!r}"
    parsed = datetime.fromisoformat(text)
    assert parsed.tzinfo is not None
    assert parsed == instant, f"{text} is {parsed.astimezone(timezone.utc)}, expected {instant}"


@pytest.fixture
def env(tmp_path, monkeypatch):
    factory = make_session_factory(tmp_path / "ts.db")
    monkeypatch.setattr(tasks_routes, "get_db_session", factory)
    monkeypatch.setattr(task_runtime, "get_db_session", factory)

    def override_db():
        with factory() as session:
            yield session

    app = FastAPI()
    app.include_router(tasks_routes.router, prefix="/api/tasks")
    app.dependency_overrides[get_db] = override_db

    with factory() as s:
        task = Task(
            task_type="ts_probe", status=TaskStatus.completed, total=2, progress=2,
            created_at=as_db_default(CREATED),
            started_at=as_datetime_now(STARTED),
            completed_at=as_datetime_now(FINISHED),
            last_heartbeat=as_datetime_now(HEARTBEAT),
        )
        s.add(task)
        s.flush()
        sub = TaskSubtask(
            task_id=task.id, subtask_key="download", subtask_type="pipeline_stage", name="download",
            status=TaskSubtaskStatus.completed,
            created_at=as_db_default(CREATED), updated_at=as_db_default(FINISHED),
            started_at=as_datetime_now(STARTED), completed_at=as_datetime_now(FINISHED),
        )
        s.add(sub)
        s.add(TaskEvent(task_id=task.id, event_type="stage.log", level="info", message="hello",
                        created_at=as_db_default(CREATED)))
        s.add(TaskResourceSample(task_id=task.id, phase="run", pid=1, cpu_percent=1.0,
                                 created_at=as_db_default(SAMPLED)))
        s.add(TaskWorkflowEdge(task_id=task.id, edge_key="a->b", from_key="a", to_key="b",
                               created_at=as_db_default(CREATED), updated_at=as_db_default(FINISHED)))
        s.flush()
        ids = {"task": task.id, "subtask": sub.id}
    return app, TaskStore(factory), ids


def client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def assert_task_times(body: dict) -> None:
    assert_instant(body["created_at"], CREATED)          # database default: UTC
    assert_instant(body["started_at"], STARTED)          # datetime.now(): local
    assert_instant(body["completed_at"], FINISHED)
    assert_instant(body["last_heartbeat"], HEARTBEAT)


@pytest.mark.asyncio
async def test_task_list_and_detail_times(env):
    app, _, ids = env
    async with client(app) as http:
        listing = (await http.get("/api/tasks")).json()["tasks"]
        detail = (await http.get(f"/api/tasks/{ids['task']}")).json()
    assert_task_times(next(t for t in listing if t["id"] == ids["task"]))
    assert_task_times(detail)
    # Embedded pieces of the detail view carry offsets too.
    assert_instant(detail["recent_events"][0]["created_at"], CREATED)
    assert_instant(detail["resource_summary"]["latest"]["created_at"], SAMPLED)
    # duration is computed from two same-convention values, so it is unaffected.
    assert detail["duration_seconds"] == pytest.approx((FINISHED - STARTED).total_seconds())


@pytest.mark.asyncio
async def test_event_resource_and_subtask_times(env):
    app, _, ids = env
    async with client(app) as http:
        events = (await http.get(f"/api/tasks/{ids['task']}/events")).json()["events"]
        samples = (await http.get(f"/api/tasks/{ids['task']}/resources")).json()["samples"]
        subtasks = (await http.get(f"/api/tasks/{ids['task']}/subtasks")).json()["subtasks"]
        one = (await http.get(f"/api/tasks/{ids['task']}/subtasks/{ids['subtask']}")).json()
    assert_instant(events[0]["created_at"], CREATED)
    assert_instant(samples[0]["created_at"], SAMPLED)
    for sub in (subtasks[0], one):
        assert_instant(sub["created_at"], CREATED)
        assert_instant(sub["updated_at"], FINISHED)
        assert_instant(sub["started_at"], STARTED)
        assert_instant(sub["completed_at"], FINISHED)


@pytest.mark.asyncio
async def test_workflow_graph_times(env):
    app, _, ids = env
    async with client(app) as http:
        workflow = (await http.get(f"/api/tasks/{ids['task']}/workflow")).json()
    [node] = workflow["nodes"]
    assert_instant(node["created_at"], CREATED)
    assert_instant(node["updated_at"], FINISHED)
    assert_instant(node["started_at"], STARTED)
    assert_instant(node["completed_at"], FINISHED)
    [edge] = workflow["edges"]
    assert_instant(edge["created_at"], CREATED)
    assert_instant(edge["updated_at"], FINISHED)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["stream", "workflow/stream"])
async def test_stream_payloads_times(env, path):
    import json

    app, _, ids = env
    async with client(app) as http:
        body = (await http.get(f"/api/tasks/{ids['task']}/{path}")).text
    blocks = [b for b in body.split("\n\n") if b.startswith(("id:", "event:"))]
    data = {}
    for block in blocks:
        lines = dict(line.split(": ", 1) for line in block.split("\n") if ": " in line)
        data.setdefault(lines["event"], []).append(json.loads(lines["data"]))
    snapshot = data["snapshot"][0]
    task = snapshot["task"] if "task" in snapshot else snapshot
    assert_task_times(task)
    assert_instant(data["task-event"][0]["created_at"], CREATED)
    assert_instant(data["resource"][0]["created_at"], SAMPLED)
    final = data["final"][0]
    assert_task_times(final["task"] if path.startswith("workflow") else final)


def test_task_store_epochs_follow_the_column_conventions(env):
    _, store, ids = env
    task = store.get_task(ids["task"])
    assert task["created_at"] == pytest.approx(CREATED.timestamp())      # was off by the UTC offset
    assert task["started_at"] == pytest.approx(STARTED.timestamp())
    assert task["finished_at"] == pytest.approx(FINISHED.timestamp())


@pytest.mark.asyncio
async def test_dashboard_activity_recent_tasks_carry_offsets(env, monkeypatch):
    """/api/dashboard/activity reads the raw columns with SQL; it must apply the same
    per-column conventions as the Tasks API."""
    import app.api.routes.dashboard as dashboard

    _, store, ids = env
    monkeypatch.setattr(dashboard, "DB_PATH", store._sf.engine.url.database)
    app = FastAPI()
    app.include_router(dashboard.router)
    async with client(app) as http:
        resp = await http.get("/api/dashboard/activity")
    assert resp.status_code == 200
    recent = next(t for t in resp.json()["recent_tasks"] if t["id"] == ids["task"])
    assert_task_times(recent)
    assert (recent["status"], recent["task_type"]) == ("completed", "ts_probe")
