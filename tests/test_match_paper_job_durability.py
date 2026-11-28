"""Concurrency invariants of the durable match-paper job.

These are the six review blockers that kept 0.4.13 unreleased. What is REAL here:
the SQLite database, the Task rows, TaskStore's conditional UPDATEs, the runner's
threads and locks — i.e. everything the defects actually lived in.

What is substituted: the match computation itself. A real match is ~40s of GPU and
its timing cannot be steered, so it could not express "cancel arrives exactly
between the status read and the status write". The substitute is a real function
gated on threading.Events, which is what makes each race deterministic rather than
hopeful. Retrieval quality/latency and the rerank cancel-vs-score equivalence are
NOT covered here — those need the real model and belong to the .88 gates.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.services.match_doc.jobs import MATCH_PAPER_JOB, MatchPaperJobRunner
from app.services.match_doc.schemas import MatchDocRequest, MatchDocResponse
from app.services.pipeline.store import TaskStore
from core.db.models import Base


def _sqlite_store() -> TaskStore:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_factory():
        session = session_local()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return TaskStore(session_factory=session_factory)


def _request() -> MatchDocRequest:
    return MatchDocRequest(input_type="paper_id", paper_id="cvpr-2024-00042", top_k=3)


class _ControlledMatch:
    """Stands in for MatchDocService.match with steerable timing."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.cancel_observed = False

    def __call__(self, request, *, progress_callback=None, should_cancel=None):
        self.calls += 1
        if progress_callback is not None:
            progress_callback({"phase": "rerank", "completed": 1, "total": 2})
        self.entered.set()
        self.release.wait(timeout=10)
        if should_cancel is not None and should_cancel():
            self.cancel_observed = True
            from core.storage.qdrant_store import QdrantRetrievalCancelledError
            raise QdrantRetrievalCancelledError("cancelled in controlled match")
        return MatchDocResponse(input_type="paper_id", query_chunks=[], papers=[])


@pytest.fixture()
def runner_with_match(monkeypatch):
    store = _sqlite_store()
    controlled = _ControlledMatch()

    class _Service:
        match = staticmethod(controlled)

    monkeypatch.setattr(
        "app.services.match_doc.jobs.get_match_doc_service", lambda _pipeline: _Service,
    )
    runner = MatchPaperJobRunner(object(), store=store, worker_id="worker-A")

    yield runner, store, controlled

    # Join the worker threads. These are daemon threads blocked on an Event, so a
    # test that returns without releasing them leaves them running for the rest of
    # the session — which is not merely untidy: it made three unrelated,
    # timing-sensitive crawl tests fail later in a full run, with no connection to
    # this file in the traceback.
    controlled.release.set()
    for thread in threading.enumerate():
        if thread.name.startswith("match-paper-"):
            thread.join(timeout=10)
            assert not thread.is_alive(), f"leaked worker thread {thread.name}"


def test_a_cancel_landing_before_the_worker_starts_prevents_the_run(runner_with_match):
    """The original bug: set_task_status('running') overwrote a cancel and ran anyway."""
    runner, store, controlled = runner_with_match
    task_id = store.create_task(task_type=MATCH_PAPER_JOB, params=_request().model_dump(mode="json"), total=100)

    assert runner.cancel(task_id) is True
    runner._launch(task_id, wait=True)

    assert controlled.calls == 0, "cancelled job must never reach the match computation"
    assert store.get_task(task_id)["status"] == "cancelled"


def test_a_cancel_during_the_run_never_publishes_a_result(runner_with_match):
    """plan.md: 任何 partial result 都不得标为 completed."""
    runner, store, controlled = runner_with_match
    receipt = runner.start(_request(), wait=False)
    task_id = receipt["task_id"]

    assert controlled.entered.wait(timeout=10), "worker never started"
    assert runner.cancel(task_id) is True
    controlled.release.set()

    deadline = time.time() + 10
    while time.time() < deadline and store.get_task(task_id)["status"] == "running":
        time.sleep(0.02)

    task = store.get_task(task_id)
    assert task["status"] == "cancelled"
    assert task["result"] is None, "a cancelled job must not carry a result"


def test_a_live_lease_stops_a_second_process_claiming_the_same_row(runner_with_match):
    """Two backend processes both resume_active() the same durable row."""
    runner, store, controlled = runner_with_match
    receipt = runner.start(_request(), wait=False)
    task_id = receipt["task_id"]
    assert controlled.entered.wait(timeout=10)

    # A different process: same store, different worker id, its own in-process set.
    other = MatchPaperJobRunner(object(), store=store, worker_id="worker-B")
    assert other.store.claim_task(task_id, "worker-B", lease_seconds=120) is False

    controlled.release.set()


def test_an_expired_lease_is_reclaimable_so_a_crash_does_not_strand_work(runner_with_match):
    runner, store, _ = runner_with_match
    task_id = store.create_task(task_type=MATCH_PAPER_JOB, params=_request().model_dump(mode="json"), total=100)

    assert store.claim_task(task_id, "worker-dead", lease_seconds=120) is True
    assert store.claim_task(task_id, "worker-B", lease_seconds=120) is False
    # lease_seconds=0 makes every heartbeat older than the cutoff: the owner is
    # presumed dead, which is what a real takeover after a crash looks like.
    assert store.claim_task(task_id, "worker-B", lease_seconds=0) is True
    assert store.get_task(task_id)["worker_id"] == "worker-B"


def test_losing_the_lease_stops_the_worker(runner_with_match):
    runner, store, controlled = runner_with_match
    receipt = runner.start(_request(), wait=False)
    task_id = receipt["task_id"]
    assert controlled.entered.wait(timeout=10)

    # Another process takes the row over while we are mid-match.
    store.claim_task(task_id, "worker-B", lease_seconds=0)
    controlled.release.set()

    deadline = time.time() + 10
    while time.time() < deadline and store.get_task(task_id)["status"] == "running":
        time.sleep(0.02)
    assert controlled.cancel_observed, "worker must observe the lost lease as a stop signal"
    assert store.get_task(task_id)["result"] is None


def test_exceeding_the_deadline_reports_timeout_and_the_phase(runner_with_match):
    """plan.md: 超时必须返回 phase、已完成 batch 和明确错误."""
    runner, store, controlled = runner_with_match
    receipt = runner.start(_request(), wait=False, deadline_seconds=0)
    task_id = receipt["task_id"]

    assert controlled.entered.wait(timeout=10)
    controlled.release.set()

    deadline = time.time() + 10
    while time.time() < deadline and store.get_task(task_id)["status"] in {"running", "pending"}:
        time.sleep(0.02)

    task = store.get_task(task_id)
    assert task["status"] == "timeout", f"expected timeout, got {task['status']}"
    assert task["result"] is None
    assert "rerank" in (task["error"] or ""), f"error must name the phase: {task['error']!r}"


def test_the_submitter_is_recorded_so_routes_can_scope_by_owner(runner_with_match):
    runner, store, controlled = runner_with_match
    receipt = runner.start(_request(), wait=False, owner_user_id="user-1")
    controlled.release.set()
    assert store.get_task(receipt["task_id"])["owner_user_id"] == "user-1"


def test_compare_and_set_status_rejects_a_transition_from_a_terminal_state(runner_with_match):
    runner, store, _ = runner_with_match
    task_id = store.create_task(task_type=MATCH_PAPER_JOB, params={}, total=100)

    assert store.compare_and_set_status(task_id, expect={"pending"}, new="cancelled") is True
    # The losing writer: this is exactly the overwrite the old code performed.
    assert store.compare_and_set_status(task_id, expect={"pending", "running"}, new="running") is False
    assert store.get_task(task_id)["status"] == "cancelled"
