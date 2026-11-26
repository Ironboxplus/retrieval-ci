"""TDD: UploadPipelineRunner — create / launch / conflict-guard / resume.

Mirrors test_pipeline_service.py but for single-document upload tasks. Offline +
deterministic: fake pipeline_builder, in-memory SQLite store, wait=True runs the
orchestrator synchronously.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.database import Base
from core.db import models  # noqa: F401
from app.services.pipeline.spec import StageSpec, Progress
from app.services.pipeline.store import TaskStore
from app.services.pipeline.upload_service import UploadPipelineRunner, UPLOAD_PIPELINE


@pytest.fixture
def store():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def factory():
        s = Local()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    return TaskStore(factory)


def _fake_builder(ctx):
    def mk(key):
        return StageSpec(
            key=key, title=key,
            run=lambda c, s: s(f"{key} ran for doc {c.doc_id}"),
            probe=lambda c: Progress(0, 1),
            poll_interval_s=0.0,
        )
    return [mk("extract"), mk("index")]


def test_start_upload_creates_and_runs(store):
    runner = UploadPipelineRunner(store=store, pipeline_builder=_fake_builder)
    res = runner.start_upload(42, "/data/uploads/x.pdf", wait=True)

    assert res["status"] == "started"
    tid = res["task_id"]
    t = store.get_task(tid)
    assert t["task_type"] == UPLOAD_PIPELINE
    assert t["params"]["doc_id"] == 42
    assert t["params"]["file_path"] == "/data/uploads/x.pdf"
    assert t["status"] == "completed"
    assert store.get_subtask(tid, "extract")["status"] == "completed"
    assert store.get_subtask(tid, "index")["status"] == "completed"


def test_duplicate_doc_blocked(store):
    runner = UploadPipelineRunner(store=store, pipeline_builder=_fake_builder)
    rid = store.create_task(task_type=UPLOAD_PIPELINE, params={"doc_id": 42, "file_path": "/x"})
    store.set_task_status(rid, "running")

    res = runner.start_upload(42, "/x", wait=True)
    assert res["status"] == "conflict"
    assert res["task_id"] == rid
    assert len(store.list_running(UPLOAD_PIPELINE)) == 1


def test_resume_relaunches_running_upload(store):
    seen = []

    def builder(ctx):
        return [StageSpec(
            key="extract", title="extract",
            run=lambda c, s: seen.append(c.doc_id),
            probe=lambda c: Progress(0, 1), poll_interval_s=0.0,
        )]

    runner = UploadPipelineRunner(store=store, pipeline_builder=builder)
    rid = store.create_task(task_type=UPLOAD_PIPELINE, params={"doc_id": 7, "file_path": "/data/uploads/u.pdf"})
    store.set_task_status(rid, "running")

    resumed = runner.resume_running(wait=True)
    assert resumed == [rid]
    assert seen == [7]
    assert store.get_task(rid)["status"] == "completed"
