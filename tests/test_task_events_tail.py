"""`GET /api/tasks/{id}/events?tail=true` returns the newest events of a long task.

The task page reads one page of events. Without ``tail`` the page is the oldest
``limit`` events, so a task with hundreds of events showed only its first hour
(user, 2026-11-14: "这个都卡好几个小时了" while the stage was at shard 31 of 54).
"""
from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

import app.api.routes.tasks as tasks_routes
from core.db.database import get_db
from core.db.models import Task, TaskEvent, TaskStatus
from tests.sqlite_task_db import make_session_factory

N_EVENTS = 150


@pytest.fixture
def env(tmp_path, monkeypatch):
    factory = make_session_factory(tmp_path / "events.db")
    monkeypatch.setattr(tasks_routes, "get_db_session", factory)

    def override_db():
        with factory() as session:
            yield session

    app = FastAPI()
    app.include_router(tasks_routes.router, prefix="/api/tasks")
    app.dependency_overrides[get_db] = override_db

    with factory() as s:
        task = Task(task_type="events_probe", status=TaskStatus.running, total=1, progress=0)
        s.add(task)
        s.flush()
        for i in range(N_EVENTS):
            s.add(TaskEvent(task_id=task.id, event_type="stage.log", level="info", message=f"event {i}"))
        s.flush()
        task_id = task.id
    return app, task_id


def client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_tail_returns_the_newest_events_in_ascending_order(env):
    app, task_id = env
    async with client(app) as http:
        body = (await http.get(f"/api/tasks/{task_id}/events", params={"tail": "true", "limit": 100})).json()
    messages = [e["message"] for e in body["events"]]
    assert messages == [f"event {i}" for i in range(N_EVENTS - 100, N_EVENTS)]
    assert body["total"] == N_EVENTS


@pytest.mark.asyncio
async def test_without_tail_the_page_is_still_the_oldest_events(env):
    app, task_id = env
    async with client(app) as http:
        body = (await http.get(f"/api/tasks/{task_id}/events", params={"limit": 100})).json()
    assert [e["message"] for e in body["events"]] == [f"event {i}" for i in range(100)]


@pytest.mark.asyncio
async def test_tail_with_after_id_is_the_newest_events_after_that_id(env):
    app, task_id = env
    async with client(app) as http:
        everything = (await http.get(f"/api/tasks/{task_id}/events", params={"limit": 500})).json()["events"]
        cut = everything[139]["id"]                               # events 140..149 are after it
        body = (await http.get(f"/api/tasks/{task_id}/events",
                               params={"tail": "true", "limit": 5, "after_id": cut})).json()
    assert [e["message"] for e in body["events"]] == [f"event {i}" for i in range(145, 150)]
    assert body["total"] == 10
