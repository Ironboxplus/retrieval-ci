"""Time-weighted pipeline progress + ETA, against a real SQLite task store.

Shapes come from .88 on 2026-09-25: task 365 (arXiv 2026-06 #0) mid-embed at
shard 4/8 while earlier batches had finished all five stages.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.database import Base
from core.db import models  # noqa: F401 — register all tables
from core.db.models import Task, TaskStatus, TaskSubtask, TaskSubtaskStatus
from app.services.pipeline.progress import estimate, pipeline_progress

KEYS = ["download", "extract", "preindex", "embed", "ingest"]
T0 = datetime(2026, 9, 25, 14, 0, 0)
# Typical minutes per stage: download 1, extract 3, preindex 2, embed 20, ingest 10.
MINUTES = {"download": 1, "extract": 3, "preindex": 2, "embed": 20, "ingest": 10}


@pytest.fixture
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Local = sessionmaker(bind=engine, expire_on_commit=False)
    s = Local()
    yield s
    s.close()


def _task(db, status, stages):
    """stages: {key: (status, progress, total, started_at, completed_at)}"""
    t = Task(task_type="ingest_pipeline", status=status, params_json={"venue": "arXiv", "year": 20260600})
    db.add(t)
    db.flush()
    for order, key in enumerate(KEYS):
        st, done, total, started, completed = stages.get(key, ("pending", 0, 0, None, None))
        db.add(TaskSubtask(
            task_id=t.id, subtask_key=key, subtask_type="pipeline_stage", name=key,
            status=TaskSubtaskStatus(st), progress=done, total=total,
            started_at=started, completed_at=completed, metadata_json={"order": order},
        ))
    db.commit()
    return t.id


def _finished(db, n):
    for i in range(n):
        start = T0 - timedelta(hours=i + 1)
        stages, cursor = {}, start
        for key in KEYS:
            end = cursor + timedelta(minutes=MINUTES[key])
            stages[key] = ("completed", 1, 1, cursor, end)
            cursor = end
        _task(db, TaskStatus.completed, stages)


def test_mid_embed_weighs_by_typical_stage_time_and_projects_eta(db):
    _finished(db, 3)
    now = T0 + timedelta(minutes=40)
    tid = _task(db, TaskStatus.running, {
        "download": ("completed", 2456, 2456, T0, T0 + timedelta(minutes=1)),
        "extract": ("completed", 2337, 2337, T0 + timedelta(minutes=1), T0 + timedelta(minutes=4)),
        "preindex": ("completed", 2335, 2335, T0 + timedelta(minutes=4), T0 + timedelta(minutes=6)),
        # waited for the GPU lock, then 4 of 8 shards in 10 minutes
        "embed": ("running", 4, 8, now - timedelta(minutes=10), None),
    })
    got = pipeline_progress(db, [tid], now=now)["tasks"][tid]
    # (1 + 3 + 2 + 20 * 0.5) / 36
    assert got["fraction"] == pytest.approx(16 / 36, abs=1e-3)
    # 10 more minutes of embed at the observed rate, then a typical 10-minute ingest
    assert got["eta_seconds"] == 20 * 60
    assert got["stage"] == {"key": "embed", "name": "embed", "status": "running", "progress": 4, "total": 8}
    assert got["waiting"] is False


def test_task_parked_on_a_stage_lock_has_no_eta(db):
    _finished(db, 3)
    tid = _task(db, TaskStatus.running, {
        "download": ("completed", 1, 1, T0, T0 + timedelta(minutes=1)),
        "extract": ("completed", 1, 1, T0, T0 + timedelta(minutes=3)),
        "preindex": ("completed", 1, 1, T0, T0 + timedelta(minutes=2)),
    })
    got = pipeline_progress(db, [tid], now=T0 + timedelta(hours=2))["tasks"][tid]
    assert got["fraction"] == pytest.approx(6 / 36, abs=1e-3)
    assert got["eta_seconds"] is None
    assert got["waiting"] is True
    assert got["stage"]["key"] == "embed"


def test_typical_duration_uses_recent_runs_not_old_slow_ones(db):
    # Five old embeds took 60 min; the five most recent took 20.
    for i, minutes in enumerate([60] * 5 + [20] * 5):
        start = T0 - timedelta(days=1) + timedelta(hours=i)
        stages = {k: ("completed", 1, 1, start, start + timedelta(minutes=MINUTES[k])) for k in KEYS}
        stages["embed"] = ("completed", 1, 1, start, start + timedelta(minutes=minutes))
        _task(db, TaskStatus.completed, stages)
    tid = _task(db, TaskStatus.running, {
        k: ("completed", 1, 1, T0, T0 + timedelta(minutes=MINUTES[k])) for k in KEYS[:3]
    } | {"embed": ("running", 0, 0, T0, None)})
    got = pipeline_progress(db, [tid], now=T0 + timedelta(minutes=5))["tasks"][tid]
    # no shard counts yet: 15 of a typical 20 minutes left, plus 10 of ingest
    assert got["eta_seconds"] == 25 * 60


def test_no_history_falls_back_to_equal_weights_and_no_eta():
    now = T0 + timedelta(minutes=5)
    stages = [
        {"key": "a", "name": "a", "status": "completed", "progress": 1, "total": 1, "started_at": T0},
        {"key": "b", "name": "b", "status": "running", "progress": 1, "total": 4, "started_at": T0},
        {"key": "c", "name": "c", "status": "pending", "progress": 0, "total": 0, "started_at": None},
    ]
    got = estimate(stages, {}, now)
    assert got["fraction"] == pytest.approx((1 + 0.25) / 3, abs=1e-3)
    assert got["eta_seconds"] is None


def test_non_pipeline_tasks_get_nothing(db):
    t = Task(task_type="crawl_urls", status=TaskStatus.running)
    db.add(t)
    db.commit()
    assert pipeline_progress(db, [t.id]) == {"tasks": {}, "eta_seconds": None}


def test_board_eta_is_the_busiest_lane_backlog_plus_its_tail(db):
    """One task mid-embed (10 min left) and two parked before embed: the GPU lane
    holds 10 + 20 + 20 minutes, then the last one still needs a 10-minute ingest."""
    _finished(db, 3)
    now = T0 + timedelta(minutes=40)
    pre = {k: ("completed", 1, 1, T0, T0 + timedelta(minutes=MINUTES[k])) for k in KEYS[:3]}
    ids = [
        _task(db, TaskStatus.running, pre | {"embed": ("running", 4, 8, now - timedelta(minutes=10), None)}),
        _task(db, TaskStatus.running, pre),
        _task(db, TaskStatus.running, pre),
    ]
    got = pipeline_progress(db, ids, now=now)
    assert got["eta_seconds"] == (10 + 20 + 20 + 10) * 60
    assert all("_backlog" not in r for r in got["tasks"].values())
