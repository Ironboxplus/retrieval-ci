"""Public-code control/index jobs are mirrored onto Tasks rows.

Reuses the fixture catalog and control/index doubles of tests/test_public_code_jobs.py;
the Tasks tables are a real SQLite file and the cancel path goes through the real Tasks
router.
"""
from __future__ import annotations

import asyncio
import threading
import time

import httpx
import pytest
from fastapi import FastAPI

import app.api.routes.tasks as tasks_routes
import app.services.public_code.jobs as public_code_jobs
import app.services.task_runtime as task_runtime
from app.services.pipeline import runner_registry
from app.services.pipeline.queue_tasks import QueueTasks
from app.services.pipeline.runner_registry import register_runner
from app.services.pipeline.store import TaskStore
from app.services.public_code.jobs import PublicCodeJobService
from core.db.database import get_db
from tests.sqlite_task_db import make_session_factory
from tests.test_public_code_jobs import ControlService, IndexService, catalog, wait_done

TYPE = "public_code_job"


@pytest.fixture
def env(tmp_path, monkeypatch):
    saved_getters = dict(runner_registry._getters)
    factory = make_session_factory(tmp_path / "tasks.db")
    store = TaskStore(factory)
    monkeypatch.setattr(tasks_routes, "get_db_session", factory)
    monkeypatch.setattr(task_runtime, "get_db_session", factory)
    register_runner(TYPE, lambda: public_code_jobs.build_task_runner(store))

    def override_db():
        with factory() as session:
            yield session

    app = FastAPI()
    app.include_router(tasks_routes.router, prefix="/api/tasks")
    app.dependency_overrides[get_db] = override_db

    services: list[PublicCodeJobService] = []

    def build(**kwargs) -> PublicCodeJobService:
        service = PublicCodeJobService(
            source_registry=catalog(),
            control_service=kwargs.pop("control_service", ControlService()),
            index_service=kwargs.pop("index_service", IndexService()),
            tasks=QueueTasks(store, min_interval=0.0),
            **kwargs,
        )
        services.append(service)
        return service

    yield build, store, app
    for service in services:
        service.shutdown(wait=False)
    runner_registry._getters.clear()
    runner_registry._getters.update(saved_getters)


def row_for(store: TaskStore, job_id: str) -> dict | None:
    with store._sf() as s:
        from core.db.models import Task
        ids = [t.id for t in s.query(Task).filter(Task.task_type == TYPE).order_by(Task.id)
               if (t.params_json or {}).get("job_id") == job_id]
    return store.get_task(ids[0]) if ids else None


def wait_for(predicate, what: str, timeout: float = 10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


def test_an_update_check_produces_a_public_code_job_row_that_completes(env):
    build, store, _app = env
    gate = threading.Event()
    service = build(control_service=ControlService(gate=gate))

    job = service.submit_update_check(source_ids=["ffmpeg", "pion"])
    row = row_for(store, job.job_id)
    assert row is not None and row["status"] in {"pending", "running"}
    assert row["params"] == {
        "job_id": job.job_id, "operation": "check_updates",
        "source_ids": ["ffmpeg", "pion"], "total_sources": 2}
    assert row["owner_user_id"] is None
    assert row["total"] == 100

    wait_for(lambda: row_for(store, job.job_id)["status"] == "running", "row running")
    gate.set()
    wait_done(service, job.job_id)
    done = wait_for(lambda: (r := row_for(store, job.job_id))["status"] == "completed" and r,
                    "row completed")
    assert done["progress"] == done["total"] == 100
    assert done["result"] == {"completed_sources": 2, "errors": {}}


def test_an_index_job_row_advances_per_source_and_a_failure_fails_the_row(env):
    build, store, _app = env
    second_started, release = threading.Event(), threading.Event()

    class TwoStepIndex(IndexService):
        def index_source(self, source_id, progress_cb=None, *, refresh_mirror=True):
            if source_id == "pion":
                second_started.set()
                assert release.wait(5)
                raise RuntimeError("cannot index pion")
            return super().index_source(source_id, progress_cb, refresh_mirror=refresh_mirror)

    service = build(index_service=TwoStepIndex())
    job = service.submit_index(source_ids=["ffmpeg", "pion"])
    assert second_started.wait(5)

    row = row_for(store, job.job_id)
    assert row["status"] == "running"
    assert row["params"]["operation"] == "index"
    assert 0 < row["progress"] < 100

    release.set()
    wait_done(service, job.job_id)
    failed = wait_for(lambda: (r := row_for(store, job.job_id))["status"] == "failed" and r,
                      "row failed")
    assert "cannot index pion" in failed["error"]
    assert failed["result"]["completed_sources"] == 2
    assert failed["result"]["errors"] == {"pion": "cannot index pion"}


def test_an_import_job_row_carries_the_operation_and_the_private_owner(env):
    build, store, _app = env

    class PrivateAwareIndex(IndexService):
        def index_source(self, source_id, progress_cb=None, *, refresh_mirror=True,
                         private_owner_user_id=None, private_project=None):
            return super().index_source(source_id, progress_cb, refresh_mirror=refresh_mirror)

    service = build(index_service=PrivateAwareIndex())

    job = service.submit_import(
        source_id="ffmpeg", prepare=lambda progress: progress("bundle published"),
        private_owner_user_id="user-9", private_project="proj")
    wait_done(service, job.job_id)

    row = wait_for(lambda: (r := row_for(store, job.job_id))["status"] == "completed" and r,
                   "row completed")
    assert row["params"]["operation"] == "import"
    assert row["params"]["source_ids"] == ["ffmpeg"]
    assert row["owner_user_id"] == "user-9"


@pytest.mark.asyncio
async def test_cancelling_a_public_code_task_is_refused_with_a_clear_reason(env):
    build, store, app = env
    gate = threading.Event()
    service = build(control_service=ControlService(gate=gate))
    try:
        job = service.submit_update_check(source_ids=["ffmpeg"])
        task_id = row_for(store, job.job_id)["id"]

        async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            resp = await http.post(f"/api/tasks/{task_id}/cancel")
        assert resp.status_code == 409
        assert "不支持取消" in resp.text
    finally:
        gate.set()
    wait_done(service, job.job_id)
    await asyncio.sleep(0)
