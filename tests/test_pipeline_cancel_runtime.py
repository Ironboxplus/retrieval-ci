"""Cancelling a pipeline stops its subprocess and the terminal status is atomic.

Real SQLite file, real orchestrator thread, real child process (a Python sleeper).
Nothing here is faked: the only injected piece is the stage list, which is how every
PipelineRunner test supplies work.
"""
from __future__ import annotations

import sys
import threading
import time

import psutil
import pytest

from app.services.pipeline.service import INGEST_PIPELINE, PipelineRunner
from app.services.pipeline.spec import Progress, StageSpec
from app.services.pipeline.store import TaskStore
from app.services.pipeline.runners import run_local
from tests.sqlite_task_db import make_session_factory

SLEEPER = (
    "import os, sys, time\n"
    "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
    "time.sleep(60)\n"
)


@pytest.fixture
def store(tmp_path):
    return TaskStore(make_session_factory(tmp_path / "tasks.db"))


def _wait(predicate, timeout=5.0, step=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


def _sleeper_runner(store, tmp_path):
    pid_file = tmp_path / "child.pid"

    def run(c, sink):
        run_local([sys.executable, "-c", SLEEPER, str(pid_file)], tmp_path, sink, abort=c.abort)

    spec = StageSpec(key="download", title="sleep", run=run, probe=lambda c: Progress(0, 1),
                     poll_interval_s=0.0)
    return PipelineRunner(store=store, pipeline_builder=lambda ctx: [spec]), pid_file


def test_cancel_terminates_the_stage_subprocess_and_status_stays_cancelled(store, tmp_path):
    runner, pid_file = _sleeper_runner(store, tmp_path)
    task_id = runner.start_ingest("CVPR", 2026)["task_id"]
    assert _wait(lambda: pid_file.exists() and pid_file.read_text().strip() != ""), "the stage never launched its subprocess"
    pid = int(pid_file.read_text())
    assert psutil.pid_exists(pid)

    started = time.monotonic()
    assert runner.cancel(task_id) is True

    assert _wait(lambda: not psutil.pid_exists(pid), timeout=2.0), "child survived cancel"
    assert time.monotonic() - started < 2.5
    assert _wait(lambda: task_id not in runner._active), "orchestrator thread did not finish"
    # The orchestrator lost every terminal write to the cancel, so nothing can have
    # turned it into failed/completed afterwards.
    assert store.get_task(task_id)["status"] == "cancelled"
    time.sleep(0.3)
    assert store.get_task(task_id)["status"] == "cancelled"
    assert store.get_subtask(task_id, "download")["status"] == "cancelled"
    assert runner._aborts == {}


def test_cancel_after_the_task_finished_changes_nothing(store, tmp_path):
    spec = StageSpec(key="download", title="quick", run=lambda c, s: None,
                     probe=lambda c: Progress(0, 1), poll_interval_s=0.0)
    runner = PipelineRunner(store=store, pipeline_builder=lambda ctx: [spec])
    task_id = runner.start_ingest("CVPR", 2026, wait=True)["task_id"]
    assert store.get_task(task_id)["status"] == "completed"

    assert runner.cancel(task_id) is False
    assert store.get_task(task_id)["status"] == "completed"


def test_cancel_before_the_orchestrator_runs_is_not_resurrected(store, tmp_path):
    """A cancel committed between task creation and the orchestrator's first write."""
    ran = []
    spec = StageSpec(key="download", title="never", run=lambda c, s: ran.append(1),
                     probe=lambda c: Progress(0, 1), poll_interval_s=0.0)
    runner = PipelineRunner(store=store, pipeline_builder=lambda ctx: [spec])
    task_id = store.create_task(task_type=INGEST_PIPELINE, params={"venue": "CVPR", "year": 2026})
    assert runner.cancel(task_id) is True

    from app.services.pipeline.orchestrator import PipelineOrchestrator
    assert PipelineOrchestrator(task_id, [spec], object(), store).run() == "cancelled"
    assert ran == []
    assert store.get_task(task_id)["status"] == "cancelled"


def test_completion_does_not_overwrite_a_cancel_that_won_the_race(store):
    """The orchestrator's final write is compare-and-set: if the task was cancelled
    while the last stage returned, `completed` must lose."""
    gate = threading.Event()
    release = threading.Event()

    def run(c, sink):
        gate.set()
        release.wait(5)

    spec = StageSpec(key="download", title="slow", run=run, probe=lambda c: Progress(0, 1),
                     poll_interval_s=0.0)
    runner = PipelineRunner(store=store, pipeline_builder=lambda ctx: [spec])
    task_id = runner.start_ingest("CVPR", 2026)["task_id"]
    assert gate.wait(5)

    # Cancel through the store only (another backend process would do exactly this):
    # the runner's own abort Event is deliberately NOT set.
    assert store.compare_and_set_status(task_id, expect={"running"}, new="cancelled")
    release.set()
    assert _wait(lambda: task_id not in runner._active)

    assert store.get_task(task_id)["status"] == "cancelled"


def test_cancelling_an_upload_task_terminates_its_stage_subprocess(store, tmp_path):
    """Same guarantee for upload pipelines: UploadStageContext carries the abort Event
    the orchestrator swaps in, so run_local() in an upload stage is killed on cancel."""
    from app.services.pipeline.upload_service import UPLOAD_PIPELINE, UploadPipelineRunner

    pid_file = tmp_path / "upload-child.pid"

    def run(c, sink):
        run_local([sys.executable, "-c", SLEEPER, str(pid_file)], tmp_path, sink, abort=c.abort)

    spec = StageSpec(key="extract", title="sleep", run=run, probe=lambda c: Progress(0, 1),
                     poll_interval_s=0.0)
    runner = UploadPipelineRunner(store=store, pipeline_builder=lambda ctx: [spec])
    task_id = runner.start_upload(7, str(tmp_path / "doc.pdf"))["task_id"]
    assert store.get_task(task_id)["task_type"] == UPLOAD_PIPELINE
    assert _wait(lambda: pid_file.exists() and pid_file.read_text().strip() != ""), "the upload stage never launched its subprocess"
    pid = int(pid_file.read_text())

    assert runner.cancel(task_id) is True

    assert _wait(lambda: not psutil.pid_exists(pid), timeout=2.0), "child survived cancel"
    assert _wait(lambda: task_id not in runner._active)
    time.sleep(0.3)
    assert store.get_task(task_id)["status"] == "cancelled"
    assert store.get_subtask(task_id, "extract")["status"] == "cancelled"
