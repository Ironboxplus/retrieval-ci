"""Runners behind the registry: start(params) with options/origin, cancel, retry limits.

Real SQLite file and the real PipelineRunner / UploadPipelineRunner /
MedicalDownloadPipelineRunner classes; stages are small callables, which is how
every runner test supplies work.
"""
from __future__ import annotations

import threading
import time

import pytest

from app.services.medical_corpus.service import MEDICAL_DOWNLOAD_PIPELINE, MedicalDownloadPipelineRunner
from app.services.pipeline.service import INGEST_PIPELINE, PipelineRunner
from app.services.pipeline.spec import Progress, StageSpec
from app.services.pipeline.store import TaskStore
from app.services.pipeline.upload_service import UPLOAD_PIPELINE, UploadPipelineRunner
from pipeline.medical_corpus.proxy import MedicalProxyConfig
from tests.sqlite_task_db import make_session_factory


@pytest.fixture
def store(tmp_path):
    return TaskStore(make_session_factory(tmp_path / "tasks.db"))


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def quick_stage(run):
    return StageSpec(key="download", title="work", run=run, probe=lambda c: Progress(0, 1), poll_interval_s=0.0)


def test_start_params_carry_options_and_origin_into_the_task_and_the_stage_context(store):
    seen = []
    runner = PipelineRunner(
        store=store,
        pipeline_builder=lambda ctx: [quick_stage(lambda c, s: seen.append(dict(c.options)))],
    )
    origin = {"schedule_id": 3, "trigger": "cron", "scheduled_for": "2026-11-03T00:00:00+00:00", "attempt": 0}

    receipt = runner.start({"venue": "CVPR", "year": 2026, "options": {"limit": 50}, "origin": origin})

    assert receipt["status"] == "started"
    task_id = receipt["task_id"]
    assert wait_for(lambda: store.get_task(task_id)["status"] == "completed")
    assert seen == [{"limit": 50}]
    params = store.get_task(task_id)["params"]
    assert params["options"] == {"limit": 50} and params["origin"] == origin
    assert [t["id"] for t in store.list_by_origin(3)] == [task_id]


def test_a_task_without_options_keeps_the_context_default(store):
    seen = []
    runner = PipelineRunner(
        store=store, pipeline_builder=lambda ctx: [quick_stage(lambda c, s: seen.append(c.options))])
    runner.start_ingest("CVPR", 2026, wait=True)
    assert seen == [{}]


def test_retry_and_resume_restore_the_stored_options(store):
    attempts = []

    def run(c, sink):
        attempts.append(dict(c.options))
        if len(attempts) == 1:
            raise RuntimeError("first run fails")

    runner = PipelineRunner(store=store, pipeline_builder=lambda ctx: [quick_stage(run)])
    task_id = runner.start_ingest("CVPR", 2026, options={"limit": 7}, wait=True)["task_id"]
    assert store.get_task(task_id)["status"] == "failed"

    assert runner.retry(task_id)["status"] == "started"
    assert wait_for(lambda: store.get_task(task_id)["status"] == "completed")
    assert attempts == [{"limit": 7}, {"limit": 7}]

    # A task a dead process left `running` resumes with its options too.
    orphan = store.create_task(task_type=INGEST_PIPELINE, params={"venue": "CVPR", "year": 2027, "options": {"limit": 9}})
    store.set_task_status(orphan, "running")
    assert runner.resume_running(wait=True) == [orphan]
    assert attempts[-1] == {"limit": 9}


def test_retry_stops_at_the_task_max_retries(store):
    runner = PipelineRunner(
        store=store, pipeline_builder=lambda ctx: [quick_stage(lambda c, s: (_ for _ in ()).throw(RuntimeError("no")))])
    task_id = runner.start_ingest("CVPR", 2026, wait=True)["task_id"]
    max_retries = store.get_task(task_id)["max_retries"]

    for _ in range(max_retries):
        assert runner.retry_failed(task_id, wait=True)["status"] == "started"
        assert store.get_task(task_id)["status"] == "failed"
    assert store.get_task(task_id)["retry_count"] == max_retries
    with pytest.raises(ValueError, match="exceeded max retries"):
        runner.retry_failed(task_id, wait=True)


def test_retry_rejects_a_task_that_is_not_failed(store):
    runner = PipelineRunner(store=store, pipeline_builder=lambda ctx: [quick_stage(lambda c, s: None)])
    task_id = runner.start_ingest("CVPR", 2026, wait=True)["task_id"]
    with pytest.raises(ValueError, match="only failed/cancelled"):
        runner.retry(task_id)


def test_conflicting_start_reports_the_running_task(store):
    release = threading.Event()
    runner = PipelineRunner(store=store, pipeline_builder=lambda ctx: [quick_stage(lambda c, s: release.wait(5))])
    first = runner.start({"venue": "CVPR", "year": 2026})
    assert wait_for(lambda: store.get_task(first["task_id"])["status"] == "running")
    again = runner.start({"venue": "CVPR", "year": 2026})
    release.set()
    assert again["status"] == "conflict" and again["task_id"] == first["task_id"]
    assert wait_for(lambda: store.get_task(first["task_id"])["status"] == "completed")


def test_upload_runner_start_cancel_and_unsupported_retry(store):
    release = threading.Event()
    spec = StageSpec(key="extract", title="x", run=lambda c, s: release.wait(5),
                     probe=lambda c: Progress(0, 1), poll_interval_s=0.0)
    runner = UploadPipelineRunner(store=store, pipeline_builder=lambda ctx: [spec])

    receipt = runner.start({"doc_id": 11, "file_path": "/data/uploads/u.pdf"})
    task_id = receipt["task_id"]
    assert receipt["status"] == "started" and store.get_task(task_id)["task_type"] == UPLOAD_PIPELINE
    assert wait_for(lambda: store.get_task(task_id)["status"] == "running")

    assert runner.cancel(task_id) is True
    release.set()
    assert wait_for(lambda: task_id not in runner._active)
    assert store.get_task(task_id)["status"] == "cancelled"
    assert runner.cancel(task_id) is False
    with pytest.raises(NotImplementedError):
        runner.retry(task_id)


@pytest.fixture
def medical(store, tmp_path):
    proxy = "http://127.0.0.1:1"
    return MedicalDownloadPipelineRunner(
        store=store, data_root=tmp_path / "medical",
        proxy=MedicalProxyConfig(http_proxy=proxy, https_proxy=proxy),
        execution_guard=lambda: None, enforce_designated_proxy=False,
    )


def test_medical_runner_cancel_and_retry_validation_go_through_the_common_entry_points(medical, store):
    running = store.create_task(task_type=MEDICAL_DOWNLOAD_PIPELINE, params={"data_root": str(medical.data_root)})
    store.set_task_status(running, "running")

    with pytest.raises(ValueError, match="Cannot retry task in status: running"):
        medical.retry(running)
    with pytest.raises(ValueError, match="not found"):
        medical.retry(987654)

    assert medical.cancel(running) is True            # no live worker: status flips, nothing to signal
    assert store.get_task(running)["status"] == "cancelled"
    assert medical.cancel(running) is False

    exhausted = store.create_task(task_type=MEDICAL_DOWNLOAD_PIPELINE, params={"data_root": str(medical.data_root)})
    store.set_task_status(exhausted, "failed")
    for _ in range(store.get_task(exhausted)["max_retries"]):
        store.reset_pipeline_for_retry(exhausted)
    store.set_task_status(exhausted, "failed")
    with pytest.raises(ValueError, match="exceeded max retries"):
        medical.retry(exhausted)
