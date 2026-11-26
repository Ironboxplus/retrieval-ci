"""A backend shutdown interrupts running pipelines; it does not judge or fail them.

2026-11-21 15:41: a deploy restart sent SIGTERM to the PaddleOCR-VL extract worker along with the
backend (systemd KillMode=control-group). The extract stage took the dead worker for "some PDFs
failed", its gate judged the partial Markdown and TPAMI 2026 (task 2577) failed. Now the unit
signals only the backend and the lifespan shutdown calls ``orchestrator.request_shutdown``: the
workers are stopped through the cancel path, the tasks stay `running`, and ``resume_running`` at
the next start finishes them.

Real SQLite file, real orchestrator threads, real child processes. The stage is a stand-in with
the extract stage's error handling at its worst (it swallows the worker's failure without
looking at the abort) and a gate that refuses partial output, so the orchestrator itself must
keep the gate from judging.
"""
from __future__ import annotations

import sys
import threading
import time

import psutil
import pytest

from app.services.pipeline import orchestrator
from app.services.pipeline.orchestrator import request_shutdown, wait_for_interrupted
from app.services.pipeline.service import PipelineRunner
from app.services.pipeline.spec import GateResult, Progress, StageSpec
from app.services.pipeline.store import TaskStore
from app.services.pipeline.runners import run_local
from core.db.models import TaskEvent
from tests.sqlite_task_db import make_session_factory

# argv: pid file, output dir, "hang" | "finish". Writes one of two pages, then hangs or writes the other.
WORKER = (
    "import os, sys, time, pathlib\n"
    "pid, out, mode = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]\n"
    "out.mkdir(parents=True, exist_ok=True)\n"
    "(out / 'a.md').write_text('first')\n"
    "open(pid, 'w').write(str(os.getpid()))\n"
    "if mode == 'hang':\n"
    "    time.sleep(60)\n"
    "(out / 'b.md').write_text('second')\n"
)


@pytest.fixture(autouse=True)
def fresh_shutdown_state(monkeypatch):
    """The shutdown flag is process-wide: each test gets its own."""
    monkeypatch.setattr(orchestrator, "_shutdown", threading.Event())
    monkeypatch.setattr(orchestrator, "_running_aborts", set())
    monkeypatch.setattr(orchestrator, "_running_lock", threading.Condition())


@pytest.fixture
def store(tmp_path):
    return TaskStore(make_session_factory(tmp_path / "tasks.db"))


def _messages(store, task_id) -> list[str]:
    with store._sf() as s:
        return [e.message for e in s.query(TaskEvent).filter(TaskEvent.task_id == task_id).order_by(TaskEvent.id)]


def _wait(predicate, timeout=5.0, step=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


class Pipeline:
    """download (instant) -> extract (a worker process on the shared 'gpu-local' resource)."""

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.mode: dict[int, str] = {}
        self.gate_calls: list[int] = []
        self.preflight_calls: list[int] = []
        self.extract_runs: list[int] = []

    def pid_file(self, task_id):
        return self.tmp_path / f"worker-{task_id}.pid"

    def out(self, task_id):
        return self.tmp_path / f"md-{task_id}"

    def build(self, ctx):
        def extract(c, sink):
            self.extract_runs.append(c.task_id)
            cmd = [sys.executable, "-c", WORKER, str(self.pid_file(c.task_id)), str(self.out(c.task_id)),
                   self.mode.get(c.task_id, "hang")]
            try:
                run_local(cmd, self.tmp_path, sink, abort=c.abort)
            except RuntimeError as exc:
                sink(f"worker did not finish ({exc}); keeping what it wrote, the gate judges the total")

        def gate(c):
            self.gate_calls.append(c.task_id)
            written = len(list(self.out(c.task_id).glob("*.md")))
            return GateResult(written == 2, f"{written}/2 pages")

        def preflight(c):
            self.preflight_calls.append(c.task_id)
            return GateResult(True)

        return [
            StageSpec(key="download", title="download", run=lambda c, s: None,
                      probe=lambda c: Progress(0, 1), poll_interval_s=0.0),
            StageSpec(key="extract", title="extract", run=extract, probe=lambda c: Progress(0, 2),
                      gate=gate, preflight=preflight, poll_interval_s=0.0, resource_key="gpu-local"),
        ]


def test_shutdown_stops_the_worker_leaves_the_tasks_running_and_the_next_start_finishes_them(store, tmp_path):
    pipeline = Pipeline(tmp_path)
    runner = PipelineRunner(store=store, pipeline_builder=pipeline.build)
    first = runner.start_ingest("TPAMI", 2026)["task_id"]
    assert _wait(lambda: pipeline.pid_file(first).exists() and pipeline.pid_file(first).read_text() != ""), \
        "the extract worker never started"
    second = runner.start_ingest("TPAMI", 2022)["task_id"]      # queued behind the first on gpu-local
    assert _wait(lambda: any(m.startswith("extract: queued") for m in _messages(store, second)))
    worker = int(pipeline.pid_file(first).read_text())
    assert psutil.pid_exists(worker)

    assert request_shutdown() == 2
    assert wait_for_interrupted(10.0) == 0

    assert _wait(lambda: not psutil.pid_exists(worker), timeout=2.0), "the worker survived the shutdown"
    assert pipeline.gate_calls == []                             # nothing judged the partial output
    for task_id in (first, second):
        assert store.get_task(task_id)["status"] == "running"
        assert "interrupted by backend shutdown; resumes at the next start" in _messages(store, task_id)
    assert store.get_subtask(first, "download")["status"] == "completed"
    assert store.get_subtask(first, "extract")["status"] == "running"   # started, not failed or cancelled
    assert store.get_subtask(second, "extract")["status"] == "pending"  # was waiting for the resource
    assert pipeline.preflight_calls == [first]

    # The next start: a fresh process (fresh shutdown flag and runner) resumes both and they finish.
    orchestrator._shutdown.clear()
    pipeline.mode.update({first: "finish", second: "finish"})
    restarted = PipelineRunner(store=store, pipeline_builder=pipeline.build)
    assert sorted(restarted.resume_running(wait=True)) == [first, second]

    for task_id in (first, second):
        assert store.get_task(task_id)["status"] == "completed"
        assert store.get_subtask(task_id, "extract")["status"] == "completed"
    assert pipeline.extract_runs == [first, first, second]
    assert pipeline.gate_calls == [first, second]
    assert pipeline.preflight_calls == [first, second]           # the resumed stage skips its preflight


def test_a_pipeline_started_during_shutdown_runs_no_stage(store, tmp_path):
    pipeline = Pipeline(tmp_path)
    runner = PipelineRunner(store=store, pipeline_builder=pipeline.build)
    request_shutdown()

    task_id = runner.start_ingest("TPAMI", 2026, wait=True)["task_id"]

    assert pipeline.extract_runs == [] and pipeline.gate_calls == []
    assert store.get_task(task_id)["status"] == "running"        # picked up by the next start
    assert store.get_subtask(task_id, "download")["status"] == "pending"


def test_a_user_cancel_is_still_a_cancel(store, tmp_path):
    pipeline = Pipeline(tmp_path)
    runner = PipelineRunner(store=store, pipeline_builder=pipeline.build)
    task_id = runner.start_ingest("TPAMI", 2026)["task_id"]
    assert _wait(lambda: pipeline.pid_file(task_id).exists() and pipeline.pid_file(task_id).read_text() != "")

    assert runner.cancel(task_id) is True
    assert _wait(lambda: task_id not in runner._active)

    assert store.get_task(task_id)["status"] == "cancelled"
    assert store.get_subtask(task_id, "extract")["status"] == "cancelled"
    assert pipeline.gate_calls == []
