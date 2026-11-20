"""A busy GPU is a wait, not a failure (2026-11-16 incident).

An untracked worker held the card, so the local extract preflight said ``no allowed local GPU has
>=6657MB free``; the 18 other ingest tasks queued on ``gpu-local`` took the lock in turn and each
failed on that one transient condition. Real orchestrators on a real SQLite file sharing one
``StageLockSet``, the repository's real ``extract_local_preflight``. The only stand-in is the card's
free-memory reader (``nvidia-smi``), a real function whose answer changes over time.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

from app.services.pipeline import orchestrator as orch
from app.services.pipeline import stages as st
from app.services.pipeline.context import StageContext
from app.services.pipeline.orchestrator import PipelineOrchestrator
from app.services.pipeline.spec import Progress, StageSpec
from app.services.pipeline.stage_lock import StageLockSet
from app.services.pipeline.store import TaskStore
from tests.sqlite_task_db import make_session_factory

BUSY_MIB, FREE_MIB = 1000, 20000          # the registered extractor's floor is 16,384 MiB
MESSAGE = "no allowed local GPU has ≥{floor}MB free; retry once memory is free or widen PIPELINE_LOCAL_GPU_IDS"


class Card:
    """What nvidia-smi reports for the one card: free MiB, changed by the test."""

    def __init__(self, free_mib: int):
        self.free_mib = free_mib
        self.reads = 0

    def __call__(self, target, *, min_free_mb: int = 0, min_free_ratio: float = 0.0, **kw) -> list[int]:
        self.reads += 1
        return [3] if self.free_mib >= min_free_mb else []


@pytest.fixture
def card(monkeypatch, tmp_path):
    extractor = tmp_path / "extractor.json"
    extractor.write_text(json.dumps({"python": Path(sys.executable).as_posix(),
                                     "worker": "scripts/remote_pvl/pvl_worker.py"}), encoding="utf-8")
    monkeypatch.setenv("PIPELINE_LOCAL_EXTRACTOR", str(extractor))
    monkeypatch.delenv("PIPELINE_LOCAL_GPU_IDS", raising=False)
    reader = Card(BUSY_MIB)
    monkeypatch.setattr(st, "gpu_health", reader)
    # seconds instead of minutes; raising=False so a build without the wait fails on behaviour, not on a name
    monkeypatch.setattr(orch, "PREFLIGHT_WAIT_INITIAL_S", 0.1, raising=False)
    monkeypatch.setattr(orch, "PREFLIGHT_WAIT_MAX_S", 0.2, raising=False)
    monkeypatch.setattr(orch, "PREFLIGHT_WAIT_DEADLINE_S", 30.0, raising=False)
    return reader


@pytest.fixture
def store(tmp_path):
    return TaskStore(make_session_factory(tmp_path / "tasks.db"))


class Pipelines:
    """N ingest tasks whose one stage is the real local extract preflight on the shared ``gpu-local``."""

    def __init__(self, store: TaskStore, tmp_path: Path, count: int):
        corpus = tmp_path / "corpus"
        (corpus / "manifests").mkdir(parents=True)
        (corpus / "manifests" / "paper_manifest.json").write_text("[]", encoding="utf-8")
        self.store, self.locks = store, StageLockSet()
        self.ran: list[int] = []
        self.results: dict[int, str] = {}
        self.orchestrators: dict[int, PipelineOrchestrator] = {}
        self.threads: list[threading.Thread] = []
        self.task_ids = [store.create_task(task_type="ingest_pipeline") for _ in range(count)]
        for task_id in self.task_ids:
            ctx = StageContext(venue="CVPR", year=2026, store=store, task_id=task_id,
                               corpus_root=corpus, project_root=tmp_path / "proj")
            spec = StageSpec(key="extract", title="extract", run=lambda c, sink: self.ran.append(c.task_id),
                             probe=lambda c: Progress(0, 1), preflight=st.extract_local_preflight,
                             resource_key="gpu-local", poll_interval_s=0.0)
            self.orchestrators[task_id] = PipelineOrchestrator(task_id, [spec], ctx, store, locks=self.locks)

    def _launch(self, task_id: int) -> None:
        orchestrator = self.orchestrators[task_id]
        thread = threading.Thread(target=lambda: self.results.__setitem__(task_id, orchestrator.run()))
        thread.start()
        self.threads.append(thread)

    def start(self) -> None:
        """The first task takes ``gpu-local``; the others queue behind it in reverse id order, so the
        grant order (resumed first, then oldest task) is what puts them back in id order."""
        first, *rest = self.task_ids
        self._launch(first)
        assert _wait_until(lambda: "gpu-local" in self.locks.holders()), "the first task never got the card"
        for queued, task_id in enumerate(reversed(rest), start=1):
            self._launch(task_id)
            assert _wait_until(lambda: self.locks._waiting("gpu-local") == queued), "a task did not queue"

    def join(self, timeout: float = 20.0) -> None:
        for thread in self.threads:
            thread.join(timeout)
        assert not any(thread.is_alive() for thread in self.threads), "an orchestrator is still running"

    def statuses(self) -> dict[int, str]:
        return {t: self.store.get_task(t)["status"] for t in self.task_ids}

    def events(self, task_id: int) -> list[str]:
        with self.store._sf() as session:
            from core.db.models import TaskEvent
            return [e.message for e in session.query(TaskEvent).filter_by(task_id=task_id)]


def _wait_until(predicate, timeout=10.0, step=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


def test_a_busy_card_makes_every_queued_task_wait_and_none_fail(card, store, tmp_path):
    pipelines = Pipelines(store, tmp_path, count=3)
    pipelines.start()

    # the card stays busy for a while: before the fix all three had failed within milliseconds
    time.sleep(1.2)
    assert pipelines.results == {}, f"tasks ended while the card was busy: {pipelines.results}"
    assert set(pipelines.statuses().values()) == {"running"}
    for task_id in pipelines.task_ids:
        assert store.get_subtask(task_id, "extract")["status"] != "failed"
    assert pipelines.ran == []
    assert card.reads > 3, "the preflight must be asked again, not once"
    assert any("waiting" in message for message in pipelines.events(pipelines.task_ids[0])), \
        "the task at the head of the queue says it is waiting for the card"

    card.free_mib = FREE_MIB                  # memory frees: the queue drains

    pipelines.join()
    assert pipelines.results == {t: "completed" for t in pipelines.task_ids}
    assert sorted(pipelines.ran) == pipelines.task_ids
    assert pipelines.ran == pipelines.task_ids, "the queue must still drain in task order"
    # the head of the queue did the asking: one preflight poller, not one per queued task
    assert card.reads < 40, f"{card.reads} reads: the queued tasks must not each poll the card"
    assert set(pipelines.locks.holders()) == set(), "gpu-local must be released"


def test_the_wait_ends_with_the_preflights_own_message_after_the_deadline(card, store, tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "PREFLIGHT_WAIT_DEADLINE_S", 0.6, raising=False)
    pipelines = Pipelines(store, tmp_path, count=1)
    started = time.monotonic()
    pipelines.start()
    pipelines.join()
    waited = time.monotonic() - started

    task_id = pipelines.task_ids[0]
    assert pipelines.results == {task_id: "failed"}
    assert waited >= 0.6, f"failed after {waited:.2f}s, before the deadline"
    error = store.get_subtask(task_id, "extract")["message"]
    assert error == "preflight: " + MESSAGE.format(floor=st.extract_local_min_free_mb())
    assert pipelines.ran == []


def test_cancelling_while_the_task_waits_for_the_card(card, store, tmp_path):
    pipelines = Pipelines(store, tmp_path, count=1)
    pipelines.start()
    task_id = pipelines.task_ids[0]
    assert _wait_until(lambda: card.reads >= 2)

    pipelines.orchestrators[task_id].abort.set()

    # the default 20 s, far under the 60 s first wait a transient verdict would start: a fixed 5 s
    # failed once under a 400-test run's load (2026-11-16) without the code being wrong
    pipelines.join()
    assert pipelines.results == {task_id: "cancelled"}
    assert store.get_task(task_id)["status"] == "cancelled"
    assert set(pipelines.locks.holders()) == set()


def test_a_preflight_that_is_not_transient_still_fails_at_once(card, store, tmp_path, monkeypatch):
    """No registered extractor is a configuration error: waiting would never fix it."""
    monkeypatch.delenv("PIPELINE_LOCAL_EXTRACTOR")
    monkeypatch.setattr(st, "require_local_extractor", lambda: (_ for _ in ()).throw(ValueError("no extractor registered")))
    pipelines = Pipelines(store, tmp_path, count=1)
    pipelines.start()
    # the default 20 s, far under the 60 s first wait a transient verdict would start: a fixed 5 s
    # failed once under a 400-test run's load (2026-11-16) without the code being wrong
    pipelines.join()

    task_id = pipelines.task_ids[0]
    assert pipelines.results == {task_id: "failed"}
    assert store.get_subtask(task_id, "extract")["message"] == "preflight: no extractor registered"
