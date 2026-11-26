"""reembed_pipeline as a durable Task type: registry, start/conflict/cancel/retry/resume, endpoint.

Real SQLite task store and the real orchestrator; the stages are small real StageSpecs (the
real stages are exercised against Qdrant in test_reembed_stages_real_qdrant.py).
"""
from __future__ import annotations

import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.pipeline import runner_registry
from app.services.pipeline.reembed_pipeline import REEMBED_PIPELINE, ReembedPipelineRunner
from app.services.pipeline.reembed_scopes import SCOPES, get_scope
from app.services.pipeline.spec import GateResult, Progress, StageSpec
from app.services.pipeline.store import TaskStore
from tests.reembed_support import new_store
from tests.sqlite_task_db import make_session_factory

ADMIN = {"X-Retrival-User-Id": "t", "X-Retrival-User-Tier": "admin"}


def noop_stage(key="select", run=lambda c, sink: None, resource_key=""):
    return StageSpec(key=key, title=key, run=run, probe=lambda c: Progress(0, 0), gate=lambda c: GateResult(True),
                     skip_when_satisfied=False, resource_key=resource_key)


def runner_with(builder, store=None):
    return ReembedPipelineRunner(store=store or new_store()[0], pipeline_builder=builder)


def test_the_task_type_is_registered_with_the_runner_registry():
    assert REEMBED_PIPELINE == "reembed_pipeline"
    assert REEMBED_PIPELINE in runner_registry.all_runners()


def test_the_presets_are_exactly_the_ten_named_scopes_with_their_collections():
    assert set(SCOPES) == {
        "code_chunks:truncated", "public_code_chunks:truncated", "code_chunks:align",
        "public_code_chunks:align", "journal_chunks:rebuild", "papers:uploads", "papers:pipeline", "arxiv:all",
        "papers:rebuild", "arxiv:rebuild"}
    with pytest.raises(ValueError, match="unknown re-embed scope"):
        get_scope("papers:everything")


def test_start_refuses_unknown_scope_mismatched_collection_and_unknown_host_before_a_task_exists(monkeypatch):
    runner = runner_with(lambda ctx: [noop_stage()])
    for kwargs, message in (
        (dict(collection="papers", scope="papers:everything"), "unknown re-embed scope"),
        (dict(collection="code_chunks", scope="papers:uploads"), "does not belong to collection"),
        (dict(collection="papers", scope="papers:uploads", ssh_target="no-such-host"), "unknown embed host"),
    ):
        with pytest.raises(ValueError, match=message):
            runner.start_reembed(**kwargs, wait=True)
    assert runner.store.list_running(REEMBED_PIPELINE) == [] and runner.store.get_task(1) is None


def test_start_runs_the_stages_through_the_orchestrator_and_the_task_shows_them():
    seen = []
    runner = runner_with(lambda ctx: [
        noop_stage("select", lambda c, sink: seen.append((type(c).__name__, c.scope_name, c.ssh_target))),
        noop_stage("verify")])

    result = runner.start_reembed("papers", "papers:uploads", wait=True)

    assert result["status"] == "started"
    task = runner.store.get_task(result["task_id"])
    assert task["task_type"] == REEMBED_PIPELINE and task["status"] == "completed"
    assert task["params"]["scope"] == "papers:uploads" and task["params"]["ssh_target"] == "arc99"
    assert seen == [("ReembedContext", "papers:uploads", "arc99")]
    assert runner.store.get_subtask(result["task_id"], "verify")["status"] == "completed"


def test_a_second_start_for_the_same_scope_conflicts_while_one_runs_but_another_scope_may_run(tmp_path):
    gate, release = threading.Event(), threading.Event()
    runner = runner_with(lambda ctx: [noop_stage("select", lambda c, sink: (gate.set(), release.wait(30)))],
                         store=TaskStore(make_session_factory(tmp_path / "tasks.db")))      # two live task threads
    first = runner.start_reembed("papers", "papers:uploads")
    assert gate.wait(10)

    again = runner.start_reembed("papers", "papers:uploads")
    other = runner.start_reembed("papers", "papers:pipeline")

    assert again["status"] == "conflict" and again["task_id"] == first["task_id"]
    assert other["status"] == "started"
    release.set()


def test_cancel_stops_a_running_task_and_retry_redrives_it_in_place():
    gate, release = threading.Event(), threading.Event()
    calls = []

    def stage(c, sink):
        calls.append(1)
        gate.set()
        release.wait(30)
        if c.abort.is_set():
            raise RuntimeError("cancelled")

    runner = runner_with(lambda ctx: [noop_stage("select", stage)])
    started = runner.start_reembed("papers", "papers:uploads")
    assert gate.wait(10)
    assert runner.cancel(started["task_id"]) is True
    release.set()
    for _ in range(100):
        if runner.store.get_task(started["task_id"])["status"] == "cancelled" and not runner._active:
            break
        threading.Event().wait(0.1)
    assert runner.store.get_task(started["task_id"])["status"] == "cancelled"

    release.set()
    retried = runner.retry(started["task_id"])
    assert retried == {"status": "started", "task_id": started["task_id"]}
    for _ in range(100):
        if runner.store.get_task(started["task_id"])["status"] == "completed":
            break
        threading.Event().wait(0.1)
    assert runner.store.get_task(started["task_id"])["status"] == "completed" and len(calls) == 2


def test_resume_running_relaunches_a_task_a_previous_process_left_running():
    store = new_store()[0]
    task_id = store.create_task(task_type=REEMBED_PIPELINE, params={
        "collection": "papers", "scope": "papers:uploads", "ssh_target": "arc99"}, total=2)
    store.set_task_status(task_id, "running")
    ran = []
    runner = ReembedPipelineRunner(store=store, pipeline_builder=lambda ctx: [
        noop_stage("select", lambda c, sink: ran.append(c.task_id))])

    assert runner.resume_running(wait=True) == [task_id]

    assert ran == [task_id] and store.get_task(task_id)["status"] == "completed"
    assert runner.resume_running(wait=True) == []                  # nothing is left running


def test_the_endpoint_starts_a_task_and_rejects_bad_requests(monkeypatch, tmp_path):
    from app.api.routes import pipeline as routes

    # A file database: the tasks the endpoint starts run on their own threads, which the shared
    # in-memory connection cannot serve at once.
    runner = runner_with(lambda ctx: [noop_stage()], store=TaskStore(make_session_factory(tmp_path / "tasks.db")))
    monkeypatch.setattr("app.services.pipeline.reembed_pipeline.get_reembed_runner", lambda: runner)
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)

    body = {"collection": "arxiv", "scope": "arxiv:all"}
    assert client.post("/api/pipeline/reembed", json=body).status_code == 401           # admin only
    ok = client.post("/api/pipeline/reembed", json=body, headers=ADMIN)
    assert ok.status_code == 200 and ok.json()["status"] == "started"
    task = runner.store.get_task(ok.json()["task_id"])
    assert task["params"] == {**task["params"], "collection": "arxiv", "scope": "arxiv:all", "ssh_target": "arc99"}
    papers = client.post("/api/pipeline/reembed", json={"collection": "papers", "scope": "papers:uploads",
                                                        "ssh_target": "local"}, headers=ADMIN)
    assert papers.status_code == 200 and papers.json()["status"] == "started"       # this backend's own GPU
    assert runner.store.get_task(papers.json()["task_id"])["params"]["ssh_target"] == "local"
    assert client.post("/api/pipeline/reembed", json={**body, "ssh_target": "no-such-host"},
                       headers=ADMIN).status_code == 422
    bad = client.post("/api/pipeline/reembed", json={"collection": "arxiv", "scope": "papers:uploads"}, headers=ADMIN)
    assert bad.status_code == 422 and "does not belong" in bad.text
    assert client.post("/api/pipeline/reembed", json={"collection": "arxiv", "scope": "arxiv:all", "filter": {"x": 1}},
                       headers=ADMIN).status_code == 422                                 # no free-form filters


def test_ssh_target_local_runs_the_task_on_this_backend_and_needs_no_embed_host_registry(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPELINE_EMBED_HOSTS", str(tmp_path / "no_such_registry.json"))
    seen = []
    runner = runner_with(lambda ctx: [noop_stage("select", lambda c, sink: seen.append(c.execution_mode))])

    result = runner.start_reembed("papers", "papers:uploads", "local", wait=True)

    assert result["status"] == "started" and seen == ["local"]
    assert runner.store.get_task(result["task_id"])["params"]["ssh_target"] == "local"
    with pytest.raises(ValueError, match="registry unreadable"):                  # a host name still reads the registry
        runner.start_reembed("papers", "papers:pipeline", wait=True)


def test_a_host_task_and_a_local_task_embed_at_once_while_their_qdrant_stages_take_turns(tmp_path):
    """The real orchestrator and the runner's shared StageLockSet, with the real stage specs'
    resource keys: only the stage bodies are replaced by small timed ones."""
    from dataclasses import replace

    from app.services.pipeline.reembed_pipeline import build_reembed_pipeline

    both_embedding = threading.Barrier(2)
    guard = threading.Lock()
    in_qdrant = {"now": 0, "max": 0}
    keys: dict[str, str] = {}

    def body(spec):
        def run(c, sink):
            keys[f"{c.execution_mode}:{spec.key}"] = spec.resource_key
            if spec.key == "embed":
                both_embedding.wait(timeout=20)            # raises BrokenBarrierError unless both are inside
                return
            if spec.resource_key != "qdrant":
                return
            with guard:
                in_qdrant["now"] += 1
                in_qdrant["max"] = max(in_qdrant["max"], in_qdrant["now"])
            threading.Event().wait(0.3)
            with guard:
                in_qdrant["now"] -= 1
        return run

    def builder(ctx):
        return [replace(spec, run=body(spec), preflight=lambda c: GateResult(True), gate=lambda c: GateResult(True),
                        probe=lambda c: Progress(0, 0))
                for spec in build_reembed_pipeline(ctx) if spec.key in ("select", "embed", "write", "resplit")]

    # A file database: two orchestrator threads need two connections (the in-memory one is shared).
    runner = runner_with(builder, store=TaskStore(make_session_factory(tmp_path / "tasks.db")))
    host = runner.start_reembed("code_chunks", "code_chunks:truncated")
    local = runner.start_reembed("public_code_chunks", "public_code_chunks:truncated", "local")
    for _ in range(300):
        statuses = {runner.store.get_task(t["task_id"])["status"] for t in (host, local)}
        if statuses <= {"completed", "failed", "cancelled"}:
            break
        threading.Event().wait(0.1)

    assert statuses == {"completed"}, [runner.store.get_subtask(t["task_id"], "embed") for t in (host, local)]
    assert keys["host:embed"] != keys["local:embed"] and keys["local:embed"] == "embed-service"
    assert keys["host:select"] == keys["local:select"] == ""                 # select takes no lock
    assert keys["host:write"] == keys["local:write"] == keys["host:resplit"] == "qdrant"
    assert in_qdrant["max"] == 1                           # the two tasks' write/resplit never overlapped
