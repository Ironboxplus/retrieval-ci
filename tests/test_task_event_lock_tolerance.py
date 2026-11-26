"""A stage's log line must not be able to fail the stage (2026-11-16 incident).

task 1309's extract ran 4.5 h and then failed with ``database is locked ... INSERT INTO
task_events``: the stage's own log line raised. Real SQLite WAL file, a second real connection
holding the write lock, a real orchestrator run. A short busy timeout makes "locked" visible within a
fraction of a second, which is what the production 30 s timeout looks like after 30 s.
"""
from __future__ import annotations

import sqlite3
import threading
import time

import pytest
from loguru import logger
from sqlalchemy.exc import OperationalError

from app.services.pipeline.service import PipelineRunner
from app.services.pipeline.spec import Progress, StageSpec
from app.services.pipeline.store import TaskStore
from core.db.models import TaskEvent
from tests.sqlite_lock_probe import HeldWriteLock, make_wal_session_factory


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "tasks.db"
    factory = make_wal_session_factory(path, busy_timeout_ms=100)
    store = TaskStore(factory)
    store.event_retry_window_s = 1.0
    return path, factory, store


@pytest.fixture
def warnings():
    seen: list[str] = []
    sink_id = logger.add(lambda message: seen.append(str(message)), level="WARNING", format="{message}")
    yield seen
    logger.remove(sink_id)


def _messages(factory, task_id):
    with factory() as session:
        return [event.message for event in session.query(TaskEvent).filter_by(task_id=task_id)]


def test_a_log_line_survives_a_lock_that_ends_inside_the_retry_window(db):
    path, factory, store = db
    task_id = store.create_task(task_type="ingest_pipeline", params={})
    lock = HeldWriteLock(path)
    threading.Timer(0.5, lock.release).start()

    store.emit_event(task_id, None, "written after the lock was released")

    assert "written after the lock was released" in _messages(factory, task_id)


def test_a_log_line_is_dropped_not_raised_when_the_lock_outlasts_the_retry_window(db, warnings):
    path, factory, store = db
    task_id = store.create_task(task_type="ingest_pipeline", params={})
    with HeldWriteLock(path):
        started = time.monotonic()
        store.emit_event(task_id, "extract", "pvl progress 12/40", level="info")   # raised OperationalError before
        waited = time.monotonic() - started

    assert 0.9 < waited < 3.0, f"retry window is 1 s, waited {waited:.2f}s"
    assert "pvl progress 12/40" not in _messages(factory, task_id)
    assert any("pvl progress 12/40" in line and "dropped" in line for line in warnings), warnings


def test_a_stage_completes_when_its_log_lines_hit_a_locked_database(tmp_path, warnings):
    """The incident end to end: the stage logs while another connection holds the write lock; the
    stage, its subtask and the task must still complete, and only the unwritable lines are lost."""
    path = tmp_path / "tasks.db"
    factory = make_wal_session_factory(path, busy_timeout_ms=100)
    store = TaskStore(factory)
    store.event_retry_window_s = 0.5
    locked, released = threading.Event(), threading.Event()
    holder = {}

    def hold():
        holder["lock"] = HeldWriteLock(path)
        locked.set()
        released.wait(10)
        holder["lock"].release()

    def run(ctx, sink):
        threading.Thread(target=hold, daemon=True).start()
        assert locked.wait(5)
        sink("line while locked 1")
        sink("line while locked 2")
        released.set()
        time.sleep(0.3)
        sink("line after release")

    spec = StageSpec(key="extract", title="extract", run=run, probe=lambda c: Progress(0, 1), poll_interval_s=0.0)
    runner = PipelineRunner(store=store, pipeline_builder=lambda ctx: [spec])

    task_id = runner.start_ingest("CVPR", 2026, wait=True)["task_id"]

    assert store.get_task(task_id)["status"] == "completed"
    assert store.get_subtask(task_id, "extract")["status"] == "completed"
    messages = _messages(factory, task_id)
    assert "line after release" in messages
    assert "line while locked 1" not in messages and "line while locked 2" not in messages
    assert sum("line while locked" in line and "dropped" in line for line in warnings) == 2


def test_a_real_error_in_the_event_write_is_still_raised(tmp_path):
    """Only lock contention is tolerated: a missing table is a bug, not a busy database."""
    path = tmp_path / "broken.db"
    store = TaskStore(make_wal_session_factory(path, busy_timeout_ms=100))
    store.event_retry_window_s = 0.2
    with sqlite3.connect(path) as conn:                 # the table the write needs is gone
        conn.execute("DROP TABLE task_events")

    with pytest.raises(OperationalError, match="no such table"):
        store.emit_event(1, None, "x")


def test_status_writes_stay_fatal_under_a_lock(db):
    """Progress and status are state, not log lines: they are not retried or dropped."""
    path, factory, store = db
    task_id = store.create_task(task_type="ingest_pipeline", params={})
    with HeldWriteLock(path):
        with pytest.raises(OperationalError, match="locked"):
            store.set_task_progress(task_id, 1, 2)
