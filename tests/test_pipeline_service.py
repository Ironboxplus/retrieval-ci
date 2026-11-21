"""TDD: PipelineRunner — create / launch / conflict-guard / resume.

Offline + deterministic: a fake pipeline_builder supplies trivial stages, the
store is a private in-memory SQLite (StaticPool), and `wait=True` runs the
orchestrator synchronously so there are no thread races.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.database import Base
from core.db import models  # noqa: F401 — register tables
from app.services.pipeline.spec import StageSpec, Progress
from app.services.pipeline.store import TaskStore
from app.services.pipeline.service import PipelineRunner, INGEST_PIPELINE


@pytest.fixture
def session_factory():
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

    return factory


@pytest.fixture
def store(session_factory):
    return TaskStore(session_factory)


def _fake_builder(ctx):
    def mk(key):
        return StageSpec(
            key=key, title=key,
            run=lambda c, s: s(f"{key} ran"),
            probe=lambda c: Progress(0, 1),
            poll_interval_s=0.0,
        )

    return [mk("download"), mk("ingest")]


def test_start_ingest_creates_and_runs(store):
    runner = PipelineRunner(store=store, pipeline_builder=_fake_builder)
    res = runner.start_ingest("CVPR", 2026, wait=True)

    assert res["status"] == "started"
    tid = res["task_id"]
    t = store.get_task(tid)
    assert t["task_type"] == INGEST_PIPELINE
    assert t["params"]["venue"] == "CVPR" and t["params"]["year"] == 2026
    assert t["status"] == "completed"
    assert store.get_subtask(tid, "download")["status"] == "completed"
    assert store.get_subtask(tid, "ingest")["status"] == "completed"


def test_concurrent_pipelines_for_different_venues_both_accepted(store):
    """Two distinct (venue, year) pipelines must BOTH be accepted — the global
    "only one pipeline at a time" guard is gone; serialization happens per-stage
    via StageLockSet, not at submit time."""
    runner = PipelineRunner(store=store, pipeline_builder=_fake_builder)
    rid = store.create_task(task_type=INGEST_PIPELINE, params={"venue": "AAAI", "year": 2026})
    store.set_task_status(rid, "running")

    res = runner.start_ingest("CVPR", 2026, wait=True)
    assert res["status"] == "started"  # NOT conflict — different venue
    assert res["task_id"] != rid


def test_duplicate_venue_year_blocked(store):
    """A second submission for the SAME (venue, year) while one is in-flight
    returns conflict — duplicate work serves no purpose."""
    runner = PipelineRunner(store=store, pipeline_builder=_fake_builder)
    rid = store.create_task(task_type=INGEST_PIPELINE, params={"venue": "WWW", "year": 2026})
    store.set_task_status(rid, "running")

    res = runner.start_ingest("WWW", 2026, wait=True)
    assert res["status"] == "conflict"
    assert res["task_id"] == rid
    assert len(store.list_running(INGEST_PIPELINE)) == 1


def test_resource_key_serializes_concurrent_starts(tmp_path):
    """Two pipelines using the SAME resource_key run sequentially even when
    submitted simultaneously. wait=False so threads race for the lock."""
    import threading
    import time
    from sqlalchemy.pool import NullPool

    db_path = tmp_path / "svc.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
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

    store = TaskStore(factory)
    inside_count = {"n": 0, "max": 0}
    lock = threading.Lock()

    def builder(ctx):
        def run(c, sink):
            with lock:
                inside_count["n"] += 1
                inside_count["max"] = max(inside_count["max"], inside_count["n"])
            time.sleep(0.05)
            with lock:
                inside_count["n"] -= 1

        return [StageSpec(
            key="ingest", title="ingest",
            run=run, probe=lambda c: Progress(0, 1),
            poll_interval_s=0.0, resource_key="qdrant",
        )]

    runner = PipelineRunner(store=store, pipeline_builder=builder)
    r1 = runner.start_ingest("CVPR", 2026, ssh_target="kss")
    r2 = runner.start_ingest("WWW", 2026, ssh_target="kss")

    # Wait briefly for both tasks to finish
    for _ in range(40):
        if store.get_task(r1["task_id"])["status"] == "completed" \
                and store.get_task(r2["task_id"])["status"] == "completed":
            break
        time.sleep(0.05)

    assert inside_count["max"] == 1, "shared resource_key must serialize"
    assert r1["status"] == "started" and r2["status"] == "started"


def test_resume_relaunches_running_pipeline(store):
    seen = []

    def builder(ctx):
        return [
            StageSpec(
                key="download", title="download",
                run=lambda c, s: seen.append(c.task_id),
                probe=lambda c: Progress(0, 1),
                poll_interval_s=0.0,
            )
        ]

    runner = PipelineRunner(store=store, pipeline_builder=builder)
    rid = store.create_task(task_type=INGEST_PIPELINE, params={"venue": "CVPR", "year": 2026})
    store.set_task_status(rid, "running")

    resumed = runner.resume_running(wait=True)
    assert resumed == [rid]
    assert seen == [rid]  # the stage actually ran under the resumed task id
    assert store.get_task(rid)["status"] == "completed"


def test_resume_orders_inflight_tasks_first(store):
    """W78: tasks with an in-flight (started_at set, non-terminal) subtask
    launch before fresh peers, so the in-flight task grabs its resource lock
    first and fresh peers naturally queue behind it instead of racing into
    preflight against a busy GPU/SQL/Qdrant."""
    seen_order: list[int] = []

    def builder(ctx):
        return [
            StageSpec(
                key="extract", title="extract",
                run=lambda c, s: seen_order.append(c.task_id),
                probe=lambda c: Progress(0, 1),
                poll_interval_s=0.0,
            )
        ]

    runner = PipelineRunner(store=store, pipeline_builder=builder)

    # Three tasks. Create in order: fresh A, in-flight B, fresh C.
    a = store.create_task(task_type=INGEST_PIPELINE, params={"venue": "CVPR", "year": 2024})
    b = store.create_task(task_type=INGEST_PIPELINE, params={"venue": "CVPR", "year": 2025})
    c = store.create_task(task_type=INGEST_PIPELINE, params={"venue": "CVPR", "year": 2026})
    for tid in (a, b, c):
        store.set_task_status(tid, "running")
        store.ensure_subtask(tid, "extract", subtask_type="pipeline_stage", name="extract", order=0)
    # Mark only B's extract as in-flight (started_at set, still pending status).
    store.start_subtask(b, "extract", message="running", stage="extract")
    # Roll B back to "pending" so the test still runs its stage; we keep started_at.
    sub_b = store.get_subtask(b, "extract")
    assert sub_b["started_at"] is not None  # sanity: started_at survives

    resumed = runner.resume_running(wait=True)
    # B (in-flight) must come first; fresh A/C come after in id order.
    assert resumed[0] == b, f"expected in-flight task {b} first, got {resumed}"
