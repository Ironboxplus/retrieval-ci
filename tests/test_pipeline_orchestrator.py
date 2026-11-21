"""TDD: generic pipeline orchestrator engine (stage-agnostic core).

The orchestrator drives an ordered list of `StageSpec` callbacks against a
parent Task + per-stage TaskSubtask rows. It must be:
  - ordered: run stages in sequence
  - resumable: skip stages whose subtask already completed (crash recovery)
  - idempotent-on-satisfied: skip a stage whose probe() already shows complete
    (e.g. CVPR PDFs already on disk → download stage is a no-op)
  - gated: preflight halts BEFORE run, gate halts AFTER run; either fails the pipeline
  - observable: writes subtask status/progress + a TaskEvent stream

These tests use a private in-memory SQLite (StaticPool so every session shares
one connection) and FAKE stages — zero real scripts / KSS / GPU. The poller is
disabled (poll_interval_s=0) for determinism; it is exercised separately.
"""
from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool, StaticPool

from core.db.database import Base
from core.db import models  # noqa: F401  — register all tables on Base.metadata
from core.db.models import TaskEvent
from app.services.pipeline.spec import StageSpec, Progress, GateResult
from app.services.pipeline.stage_lock import StageLockSet
from app.services.pipeline.store import TaskStore
from app.services.pipeline.orchestrator import PipelineOrchestrator


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
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

    return factory


@pytest.fixture
def store(session_factory):
    return TaskStore(session_factory)


@pytest.fixture
def concurrent_store(tmp_path):
    """File-backed SQLite + NullPool so multiple threads can each commit without
    racing on a single shared connection (in-memory StaticPool only allows one
    writer at a time → "SQL statements in progress" under concurrency tests)."""
    db_path = tmp_path / "orchestrator.db"
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

    return TaskStore(factory)


def _ctx():
    return SimpleNamespace(venue="CVPR", year=2026, state={})


def _probe(done, total):
    return lambda ctx: Progress(done, total)


def _spec(key, *, run=None, probe=None, gate=None, preflight=None, resource_key=""):
    return StageSpec(
        key=key,
        title=key.title(),
        run=run or (lambda ctx, sink: None),
        probe=probe or _probe(0, 10),
        gate=gate,
        preflight=preflight,
        poll_interval_s=0.0,  # disable background poller → deterministic
        resource_key=resource_key,
    )


def _events(session_factory, task_id):
    with session_factory() as s:
        return (
            s.query(TaskEvent)
            .filter(TaskEvent.task_id == task_id)
            .order_by(TaskEvent.id)
            .all()
        )


def test_runs_stages_in_order_and_marks_complete(store):
    task_id = store.create_task(task_type="ingest_pipeline", params={"venue": "CVPR"})
    calls = []
    specs = [
        _spec("download", run=lambda c, s: calls.append("download"), probe=_probe(0, 10)),
        _spec("extract", run=lambda c, s: calls.append("extract"), probe=_probe(0, 10)),
        _spec("ingest", run=lambda c, s: calls.append("ingest"), probe=_probe(0, 10)),
    ]
    final = PipelineOrchestrator(task_id, specs, _ctx(), store).run()

    assert final == "completed"
    assert calls == ["download", "extract", "ingest"]  # strict order
    t = store.get_task(task_id)
    assert t["status"] == "completed"
    assert t["progress"] == 3 and t["total"] == 3
    for key in ("download", "extract", "ingest"):
        assert store.get_subtask(task_id, key)["status"] == "completed"


def test_already_satisfied_stage_is_skipped_without_running(store):
    """CVPR download already on disk → probe complete → skip run, mark completed."""
    task_id = store.create_task(task_type="ingest_pipeline")
    ran = []
    specs = [
        _spec("download", run=lambda c, s: ran.append("download"), probe=_probe(4071, 4071)),
        _spec("extract", run=lambda c, s: ran.append("extract"), probe=_probe(0, 10)),
    ]
    final = PipelineOrchestrator(task_id, specs, _ctx(), store).run()

    assert final == "completed"
    assert ran == ["extract"]  # download skipped — artifacts already present
    assert store.get_subtask(task_id, "download")["status"] == "completed"


def test_preflight_failure_halts_pipeline_before_run(store):
    task_id = store.create_task(task_type="ingest_pipeline")
    ran = []
    specs = [
        _spec("download", run=lambda c, s: ran.append("download"), probe=_probe(0, 10)),
        _spec(
            "extract",
            run=lambda c, s: ran.append("extract"),
            probe=_probe(0, 10),
            preflight=lambda c: GateResult(False, "all GPUs bricked"),
        ),
        _spec("ingest", run=lambda c, s: ran.append("ingest"), probe=_probe(0, 10)),
    ]
    final = PipelineOrchestrator(task_id, specs, _ctx(), store).run()

    assert final == "failed"
    assert ran == ["download"]  # extract never runs; ingest never reached
    assert store.get_subtask(task_id, "extract")["status"] == "failed"
    assert "bricked" in store.get_subtask(task_id, "extract")["message"]
    assert store.get_subtask(task_id, "ingest")["status"] == "pending"
    assert store.get_task(task_id)["status"] == "failed"


def test_gate_failure_after_run_halts_pipeline(store):
    task_id = store.create_task(task_type="ingest_pipeline")
    ran = []
    specs = [
        _spec(
            "extract",
            run=lambda c, s: ran.append("extract"),
            probe=_probe(5, 10),
            gate=lambda c: GateResult(False, "MD sample empty — extraction broken"),
        ),
        _spec("preindex", run=lambda c, s: ran.append("preindex"), probe=_probe(0, 10)),
    ]
    final = PipelineOrchestrator(task_id, specs, _ctx(), store).run()

    assert final == "failed"
    assert ran == ["extract"]  # extract ran, gate rejected, preindex not reached
    assert store.get_subtask(task_id, "extract")["status"] == "failed"
    assert store.get_subtask(task_id, "preindex")["status"] == "pending"


def test_run_exception_marks_stage_failed(store):
    task_id = store.create_task(task_type="ingest_pipeline")

    def boom(c, s):
        raise RuntimeError("subprocess exit 2")

    specs = [_spec("download", run=boom, probe=_probe(0, 10))]
    final = PipelineOrchestrator(task_id, specs, _ctx(), store).run()

    assert final == "failed"
    sub = store.get_subtask(task_id, "download")
    assert sub["status"] == "failed"
    assert "subprocess exit 2" in (sub["message"] or "")


def test_events_and_progress_written(store, session_factory):
    task_id = store.create_task(task_type="ingest_pipeline")

    def run(c, sink):
        sink("downloading shard 1")
        sink("downloading shard 2")

    specs = [_spec("download", run=run, probe=_probe(0, 10))]
    PipelineOrchestrator(task_id, specs, _ctx(), store).run()

    msgs = [e.message for e in _events(session_factory, task_id)]
    assert any("downloading shard 1" in m for m in msgs)
    assert any("started" in m.lower() for m in msgs)
    assert any("completed" in m.lower() for m in msgs)

    sub = store.get_subtask(task_id, "download")
    assert sub["status"] == "completed"
    assert sub["total"] == 10 and sub["progress"] == 10  # complete bumps to total


def test_a_gate_that_passes_with_a_note_shows_it_on_the_stage(store, session_factory):
    # task 1302: the extract gate passes 259/297 and must say that 38 were refused, not just "completed"
    task_id = store.create_task(task_type="ingest_pipeline")
    specs = [
        _spec("extract", probe=_probe(259, 297), gate=lambda c: GateResult(True, "38/297 refused verification")),
        _spec("preindex", probe=_probe(0, 10), gate=lambda c: GateResult(True)),
    ]
    assert PipelineOrchestrator(task_id, specs, _ctx(), store).run() == "completed"
    assert "38/297 refused verification" in store.get_subtask(task_id, "extract")["message"]
    assert store.get_subtask(task_id, "preindex")["message"] == "completed"
    msgs = [e.message for e in _events(session_factory, task_id)]
    assert any("Extract: completed" in m and "38/297 refused verification" in m for m in msgs)


def test_two_orchestrators_on_same_resource_serialize(concurrent_store):
    store = concurrent_store
    """Two pipelines sharing a resource_key must run sequentially, not interleave.
    Verifies the per-stage mutex.
    """
    import threading as _thr
    locks = StageLockSet()
    timeline: list[str] = []
    inside_count = {"n": 0, "max": 0}
    inside_lock = _thr.Lock()

    def run(name: str):
        def _run(c, sink):
            with inside_lock:
                inside_count["n"] += 1
                inside_count["max"] = max(inside_count["max"], inside_count["n"])
                timeline.append(f"+{name}")
            time.sleep(0.05)
            with inside_lock:
                inside_count["n"] -= 1
                timeline.append(f"-{name}")
        return _run

    tid_a = store.create_task(task_type="ingest_pipeline", params={"venue": "A"})
    tid_b = store.create_task(task_type="ingest_pipeline", params={"venue": "B"})

    def drive(tid, name):
        spec = _spec("ingest", run=run(name), resource_key="qdrant")
        PipelineOrchestrator(tid, [spec], _ctx(), store, locks=locks).run()

    import time
    ta = _thr.Thread(target=drive, args=(tid_a, "A"), daemon=True)
    tb = _thr.Thread(target=drive, args=(tid_b, "B"), daemon=True)
    ta.start(); time.sleep(0.005); tb.start()
    ta.join(timeout=3.0); tb.join(timeout=3.0)

    assert inside_count["max"] == 1, f"expected serialization, got max={inside_count['max']}"
    # Each task's "+" must precede its own "-" (no interleaving inside the
    # critical section). Order BETWEEN tasks is NOT guaranteed (stdlib
    # threading.Lock is unfair).
    a_in = timeline.index("+A"); a_out = timeline.index("-A")
    b_in = timeline.index("+B"); b_out = timeline.index("-B")
    assert a_in < a_out and b_in < b_out
    # And the two intervals must not overlap.
    assert a_out < b_in or b_out < a_in


def test_different_resource_keys_run_concurrently(concurrent_store):
    store = concurrent_store
    """A pipeline at preindex(resource=sql) and B pipeline at extract(resource=gpu)
    are independent — they MUST overlap, not serialize."""
    import threading as _thr
    import time
    locks = StageLockSet()
    started = {"A": _thr.Event(), "B": _thr.Event()}
    both_inside = _thr.Event()
    inside = {"A": False, "B": False}

    def run(name: str):
        def _run(c, sink):
            inside[name] = True
            started[name].set()
            # Wait for peer to also be inside — proves concurrent execution.
            peer = "B" if name == "A" else "A"
            for _ in range(50):
                if inside[peer]:
                    both_inside.set()
                    break
                time.sleep(0.01)
            inside[name] = False
        return _run

    tid_a = store.create_task(task_type="ingest_pipeline", params={"venue": "A"})
    tid_b = store.create_task(task_type="ingest_pipeline", params={"venue": "B"})

    def drive(tid, name, rkey):
        spec = _spec("any", run=run(name), resource_key=rkey)
        PipelineOrchestrator(tid, [spec], _ctx(), store, locks=locks).run()

    ta = _thr.Thread(target=drive, args=(tid_a, "A", "sql"), daemon=True)
    tb = _thr.Thread(target=drive, args=(tid_b, "B", "gpu-4090"), daemon=True)
    ta.start(); tb.start()
    ta.join(timeout=3.0); tb.join(timeout=3.0)

    assert both_inside.is_set(), "different resource_keys should run concurrently"


def test_resume_with_running_subtask_skips_preflight(store):
    """A subtask with status=running and started_at set = a previous orchestrator
    crashed mid-stage; the remote worker (marker tmux / embed tmux) is likely
    still alive. Preflight on resume would see that worker's own resource use
    and reject — instead, skip preflight and re-attach via probe poll. W67."""
    task_id = store.create_task(task_type="ingest_pipeline")
    # Seed a started-but-not-completed subtask, mimicking a crashed run.
    store.ensure_subtask(task_id, "extract", subtask_type="pipeline_stage", name="extract", order=0)
    store.start_subtask(task_id, "extract", message="running", stage="extract")

    preflight_called = []
    ran = []
    specs = [
        _spec(
            "extract",
            run=lambda c, s: ran.append("extract"),
            probe=_probe(3, 10),
            preflight=lambda c: (preflight_called.append(1), GateResult(False, "GPU busy"))[1],
        )
    ]
    final = PipelineOrchestrator(task_id, specs, _ctx(), store).run()
    assert final == "completed"
    assert preflight_called == []  # skipped on resume
    assert ran == ["extract"]


def test_resume_skips_completed_stage_on_rerun(store):
    """Simulate a crash: stage 1 completes, stage 2 raises. A fresh orchestrator
    over the SAME task must skip stage 1 (resume) and re-run only stage 2."""
    task_id = store.create_task(task_type="ingest_pipeline")

    ran1 = []
    specs1 = [
        _spec("download", run=lambda c, s: ran1.append("download"), probe=_probe(0, 10)),
        _spec("extract", run=lambda c, s: (_ for _ in ()).throw(RuntimeError("crash")), probe=_probe(0, 10)),
    ]
    assert PipelineOrchestrator(task_id, specs1, _ctx(), store).run() == "failed"
    assert ran1 == ["download"]

    ran2 = []
    specs2 = [
        _spec("download", run=lambda c, s: ran2.append("download"), probe=_probe(0, 10)),
        _spec("extract", run=lambda c, s: ran2.append("extract"), probe=_probe(0, 10)),
    ]
    assert PipelineOrchestrator(task_id, specs2, _ctx(), store).run() == "completed"
    assert ran2 == ["extract"]  # download skipped via resume (subtask already completed)


def test_cancel_while_queued_for_resource_is_not_reported_as_failure(store):
    """A user cancellation during lock wait is terminal ``cancelled``, not failed."""
    import threading
    import time

    task_id = store.create_task(task_type="ingest_pipeline")
    locks = StageLockSet()
    blocker_abort = threading.Event()
    acquired = locks.acquire("shared-download", blocker_abort, owner="other-task")
    assert acquired

    abort = threading.Event()
    final: list[str] = []
    worker = threading.Thread(
        target=lambda: final.append(PipelineOrchestrator(
            task_id,
            [_spec("download", resource_key="shared-download")],
            _ctx(), store, abort=abort, locks=locks,
        ).run()),
        daemon=True,
    )
    worker.start()
    time.sleep(0.05)  # the real StageLockSet is now waiting on the occupied lock
    abort.set()
    worker.join(timeout=2)
    locks.release("shared-download")

    assert final == ["cancelled"]
    assert store.get_task(task_id)["status"] == "cancelled"
    assert store.get_subtask(task_id, "download")["status"] == "cancelled"


def test_cancel_racing_with_stage_completion_cannot_flip_parent_to_completed(store):
    """A cancel arriving as a stage returns wins over the normal completion path."""
    import threading

    task_id = store.create_task(task_type="ingest_pipeline")
    abort = threading.Event()

    def run_then_cancel(ctx, sink):
        abort.set()

    final = PipelineOrchestrator(
        task_id,
        [_spec("download", run=run_then_cancel)],
        _ctx(), store, abort=abort,
    ).run()

    assert final == "cancelled"
    assert store.get_task(task_id)["status"] == "cancelled"
    assert store.get_subtask(task_id, "download")["status"] == "cancelled"


def _wait_queued_on(locks, resource, n, timeout=5.0):
    import time

    deadline = time.monotonic() + timeout
    while locks._waiting(resource) < n:
        assert time.monotonic() < deadline, f"only {locks._waiting(resource)} of {n} queued"
        time.sleep(0.002)


def _run_gpu_queue(store, locks, holder, first_in_line, second_in_line):
    """Real orchestrators on one shared StageLockSet. `holder` runs a stage that
    holds "gpu" until released; then `first_in_line` and `second_in_line`
    ((task_id, stage_key) pairs) queue on "gpu" in that order. Returns the order
    in which the three stages actually ran."""
    import threading

    order: list = []
    holding, release = threading.Event(), threading.Event()

    def held_run(ctx, sink):
        order.append(holder)
        holding.set()
        release.wait(timeout=10)

    def start(task_id, key, run):
        spec = _spec(key, run=run, resource_key="gpu")
        t = threading.Thread(
            target=lambda: PipelineOrchestrator(
                task_id, [spec], _ctx(), store, locks=locks,
            ).run(),
            daemon=True,
        )
        t.start()
        return t

    threads = [start(holder[0], holder[1], held_run)]
    assert holding.wait(timeout=5)
    for i, (task_id, key) in enumerate((first_in_line, second_in_line), start=1):
        threads.append(start(task_id, key, lambda ctx, sink, tk=(task_id, key): order.append(tk)))
        _wait_queued_on(locks, "gpu", i)
    release.set()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()
    return order


def test_older_tasks_embed_is_granted_before_younger_tasks_extract(concurrent_store):
    """2026-11-13: task 1305's embed queued behind younger tasks' extracts and
    lost every release. Grant order follows task id, not who polls first."""
    store = concurrent_store
    t1 = store.create_task(task_type="ingest_pipeline")
    t2 = store.create_task(task_type="ingest_pipeline")
    t3 = store.create_task(task_type="ingest_pipeline")

    order = _run_gpu_queue(
        store, StageLockSet(),
        holder=(t2, "extract"),
        first_in_line=(t3, "extract"),  # arrives first ...
        second_in_line=(t1, "embed"),   # ... but is the younger task
    )

    assert order == [(t2, "extract"), (t1, "embed"), (t3, "extract")]


def test_resuming_stage_is_granted_before_older_fresh_stage(concurrent_store):
    """W78: a stage whose subtask already started (a remote worker may still be
    on the resource) goes ahead of an older task's fresh stage."""
    store = concurrent_store
    fresh = store.create_task(task_type="ingest_pipeline")
    holder = store.create_task(task_type="ingest_pipeline")
    resumed = store.create_task(task_type="ingest_pipeline")
    store.ensure_subtask(resumed, "embed", subtask_type="pipeline_stage", name="Embed", order=0)
    store.start_subtask(resumed, "embed", message="running", stage="embed")

    order = _run_gpu_queue(
        store, StageLockSet(),
        holder=(holder, "extract"),
        first_in_line=(fresh, "extract"),
        second_in_line=(resumed, "embed"),
    )

    assert order == [(holder, "extract"), (resumed, "embed"), (fresh, "extract")]
