"""An externally started pipeline (start_stage) is checked against what is on disk.

`start_stage` says "everything before this stage was done elsewhere". The runner
now probes the stage right before the start point and refuses (ValueError, which
the route answers as 422) before a task exists when that stage has produced
nothing, and the prefix subtasks record the numbers the probes saw instead of an
asserted 1/1.

Real everything: real manifest and Markdown files, the real production stage
builders and probes, a real SQLite task table and data database, the real
orchestrator and the real preindex script in a temporary DATA_DIR.
"""
from __future__ import annotations

import dataclasses
import json

import pytest
from sqlalchemy import func

from app.services.pipeline import stages
from app.services.pipeline.registry import build_ingest_pipeline
from app.services.pipeline.service import INGEST_PIPELINE, PipelineRunner
from core.db.models import Paper as DBPaper, TaskEvent
from tests.test_pipeline_external_md_runtime import _record, supplied_slice  # noqa: F401 - fixture


def _runner(store, ctx, data_factory, *, stage_keys=None) -> PipelineRunner:
    """Production stages for the slice of `ctx`, preindex bound to the temporary data DB.

    `stage_keys` trims the list so a run stops before the GPU stages."""
    def build(c):
        specs = [
            stages.make_preindex_stage(c, session_factory=data_factory) if s.key == "preindex" else s
            for s in build_ingest_pipeline(c)
        ]
        return [s for s in specs if stage_keys is None or s.key in stage_keys]

    def context_factory(**kw):
        return dataclasses.replace(PipelineRunner._default_ctx(**kw), corpus_root=ctx.corpus_root)

    return PipelineRunner(store=store, pipeline_builder=build, context_factory=context_factory)


def test_start_at_preindex_is_refused_when_extract_produced_no_markdown(supplied_slice):
    ctx, store, data_factory, _papers = supplied_slice
    for md in ctx.local_md_dir.glob("*.md"):
        md.unlink()
    runner = _runner(store, ctx, data_factory)

    with pytest.raises(ValueError, match=r"start_stage='preindex'.*extract.*0/2"):
        runner.start_ingest("CVPR", 2026, start_stage="preindex", wait=True)

    assert store.list_running(INGEST_PIPELINE) == []
    assert store.get_task(ctx.task_id + 1) is None  # no task row was created


def test_start_at_extract_is_refused_without_pdfs(supplied_slice):
    ctx, store, data_factory, _papers = supplied_slice
    assert ctx.pdf_count() == 0 and ctx.md_count_local() == 2  # Markdown alone is not a download
    runner = _runner(store, ctx, data_factory)

    with pytest.raises(ValueError, match=r"start_stage='extract'.*download.*0/2"):
        runner.start_ingest("CVPR", 2026, start_stage="extract", wait=True)
    assert store.get_task(ctx.task_id + 1) is None


def test_start_after_preindex_is_refused_while_sql_holds_no_paper_of_the_slice(supplied_slice):
    ctx, store, data_factory, _papers = supplied_slice
    runner = _runner(store, ctx, data_factory)

    with pytest.raises(ValueError, match=r"start_stage='embed'.*preindex.*found 0/"):
        runner.start_ingest("CVPR", 2026, start_stage="embed", chunk_id_before=0, wait=True)
    # embed's own output is task-private, so an ingest/cleanup start is judged by preindex too
    with pytest.raises(ValueError, match=r"start_stage='ingest'.*preindex"):
        runner.start_ingest("CVPR", 2026, start_stage="ingest", shard_start=0, shard_end=1, wait=True)
    with pytest.raises(ValueError, match=r"start_stage='cleanup'.*preindex"):
        runner.start_ingest("CVPR", 2026, start_stage="cleanup", wait=True)
    assert store.get_task(ctx.task_id + 1) is None


def test_start_at_preindex_records_what_the_probes_saw_and_runs(supplied_slice):
    ctx, store, data_factory, papers = supplied_slice
    runner = _runner(store, ctx, data_factory, stage_keys={"download", "extract", "preindex"})

    started = runner.start_ingest("CVPR", 2026, start_stage="preindex", wait=True)

    task = store.get_task(started["task_id"])
    assert task["status"] == "completed", task
    download = store.get_subtask(started["task_id"], "download")
    extract = store.get_subtask(started["task_id"], "extract")
    # Real numbers, not an asserted 1/1: no PDF was downloaded here, both Markdown files exist.
    assert (download["status"], download["progress"], download["total"]) == ("completed", 0, 2)
    assert (extract["status"], extract["progress"], extract["total"]) == ("completed", 2, 2)
    assert "externally supplied" in extract["message"] and "2/2" in extract["message"]
    assert store.get_subtask(started["task_id"], "preindex")["status"] == "completed"
    with data_factory() as db:
        assert {r[0] for r in db.query(DBPaper.paper_id)} == set(papers)
    manifest = json.loads(ctx.manifest_path.read_text(encoding="utf-8"))
    assert all(r["markdown_path"] for r in manifest)  # C1 still holds on the external-start path


def test_start_at_embed_is_accepted_once_preindex_has_filled_sql(supplied_slice):
    ctx, store, data_factory, papers = supplied_slice
    first = _runner(store, ctx, data_factory, stage_keys={"download", "extract", "preindex"})
    assert first.start_ingest("CVPR", 2026, start_stage="preindex", wait=True)["status"] == "started"
    with data_factory() as db:
        assert db.query(func.count(DBPaper.paper_id)).scalar() == len(papers)

    # The verifiable prefix is now complete, so the start is accepted; run only the
    # prefix bookkeeping (the GPU stages are outside this test).
    runner = _runner(store, ctx, data_factory, stage_keys={"download", "extract", "preindex", "embed"})
    probe_ctx = dataclasses.replace(PipelineRunner._default_ctx(
        venue="CVPR", year=2026, store=None, task_id=0, concurrency=8, world_size=8), corpus_root=ctx.corpus_root)
    runner._verify_external_prefix(probe_ctx, runner._build(probe_ctx), "embed")  # does not raise

    task_id = store.create_task(task_type=INGEST_PIPELINE, params={"venue": "CVPR", "year": 2026}, total=6)
    real_ctx = dataclasses.replace(probe_ctx, store=store, task_id=task_id)
    runner._complete_external_stage_prefix(task_id, runner._build(real_ctx), "embed", real_ctx)
    preindex = store.get_subtask(task_id, "preindex")
    assert (preindex["status"], preindex["progress"], preindex["total"]) == ("completed", len(papers), 2)
    assert store.get_subtask(task_id, "embed") is None


def test_exempt_prefix_stages_keep_their_asserted_completion(supplied_slice):
    """embed/ingest write to the task's own directory, which cannot exist before the
    task does, so there is nothing to probe and they stay asserted as supplied."""
    ctx, store, data_factory, _papers = supplied_slice
    runner = _runner(store, ctx, data_factory)
    task_id = store.create_task(task_type=INGEST_PIPELINE, params={"venue": "CVPR", "year": 2026}, total=6)
    real_ctx = dataclasses.replace(ctx, task_id=task_id)

    runner._complete_external_stage_prefix(task_id, runner._build(real_ctx), "cleanup", real_ctx)

    for key in ("embed", "ingest"):
        sub = store.get_subtask(task_id, key)
        assert (sub["status"], sub["progress"], sub["total"]) == ("completed", 1, 1)
        assert "externally supplied" in sub["message"]
    assert store.get_subtask(task_id, "cleanup") is None


# ------------------------------------------- the stage the start consumes is judged
def _slice_of_ten(ctx, pdfs_on_disk: int) -> None:
    """Ten papers in the manifest and `pdfs_on_disk` of their PDFs in the download directory."""
    ctx.manifest_path.write_text(
        json.dumps([_record(f"cvpr-2026-{n:02d}", f"Paper {n}") for n in range(10)]), encoding="utf-8")
    ctx.pdf_dir.mkdir(parents=True, exist_ok=True)
    for n in range(pdfs_on_disk):
        (ctx.pdf_dir / f"cvpr-2026-{n:02d}.pdf").write_bytes(b"%PDF-1.4")


def _complete_prefix(runner, ctx, store, start_stage: str) -> int:
    task_id = store.create_task(task_type=INGEST_PIPELINE, params={"venue": "CVPR", "year": 2026}, total=6)
    real_ctx = dataclasses.replace(ctx, task_id=task_id)
    runner._complete_external_stage_prefix(task_id, runner._build(real_ctx), start_stage, real_ctx)
    return task_id


def _gate_failed_events(store, task_id) -> list[tuple[str, str]]:
    with store._sf() as s:
        rows = s.query(TaskEvent).filter_by(task_id=task_id, event_type="stage.external_gate_failed")
        return [(row.stage, row.message) for row in rows]


def test_an_external_start_does_not_hide_a_download_below_the_gate(supplied_slice):
    ctx, store, data_factory, _papers = supplied_slice
    _slice_of_ten(ctx, pdfs_on_disk=5)  # 5/10, below download_gate's 90%
    runner = _runner(store, ctx, data_factory)

    task_id = _complete_prefix(runner, ctx, store, "extract")

    download = store.get_subtask(task_id, "download")
    assert download is None or download["status"] == "pending", download
    events = _gate_failed_events(store, task_id)
    assert [stage for stage, _ in events] == ["download"], events
    # The default download receives from the volume, so its gate also names where the rest must come from.
    assert ("externally supplied output fails the gate "
            "(only 5/10 PDFs downloaded (0 tried, 5 never tried); the missing PDFs") in events[0][1]
    assert events[0][1].endswith("the stage runs")
    assert store.get_subtask(task_id, "extract") is None  # the start stage itself is the orchestrator's


def test_an_external_start_keeps_a_download_that_passes_the_gate(supplied_slice):
    ctx, store, data_factory, _papers = supplied_slice
    _slice_of_ten(ctx, pdfs_on_disk=9)  # 9/10 = 90%
    runner = _runner(store, ctx, data_factory)

    task_id = _complete_prefix(runner, ctx, store, "extract")

    download = store.get_subtask(task_id, "download")
    assert (download["status"], download["progress"], download["total"]) == ("completed", 9, 10)
    assert download["message"] == "externally supplied before pipeline start_stage=extract (9/10)"
    assert _gate_failed_events(store, task_id) == []


def test_markdown_shipped_without_pdfs_still_completes_download_and_extract_is_judged_by_its_own_gate(supplied_slice):
    ctx, store, data_factory, _papers = supplied_slice
    assert ctx.pdf_count() == 0 and ctx.md_count_local() == 2
    runner = _runner(store, ctx, data_factory)

    task_id = _complete_prefix(runner, ctx, store, "preindex")

    download = store.get_subtask(task_id, "download")
    extract = store.get_subtask(task_id, "extract")
    assert (download["status"], download["progress"], download["total"]) == ("completed", 0, 2)
    assert (extract["status"], extract["progress"], extract["total"]) == ("completed", 2, 2)
    assert _gate_failed_events(store, task_id) == []

    (ctx.local_md_dir / "cvpr-2026-bb.md").unlink()  # 1/2 Markdown: extract's gate (95%) fails
    task_id = _complete_prefix(runner, ctx, store, "preindex")

    download = store.get_subtask(task_id, "download")
    extract = store.get_subtask(task_id, "extract")
    assert download["status"] == "completed"  # earlier stages may be empty, they are not judged
    assert extract["status"] == "pending", extract
    events = _gate_failed_events(store, task_id)
    assert [stage for stage, _ in events] == ["extract"], events
    assert "only 1/2 MD extracted" in events[0][1]


# ------------------------------------------------------------------ the route
@pytest.fixture
def ingest_client(supplied_slice, monkeypatch):
    """The real /api/pipeline/ingest route over a runner that has the temporary
    task table and data database; only the pipeline router is mounted."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.routes.pipeline import router
    from app.services.pipeline import service

    ctx, store, data_factory, papers = supplied_slice
    runner = _runner(store, ctx, data_factory, stage_keys={"download", "extract", "preindex"})
    monkeypatch.setattr(service, "_runner", runner)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app), ctx, store, data_factory, papers


def _wait_for_status(store, task_id, wanted, timeout=180.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = store.get_task(task_id)["status"]
        if status in wanted:
            return status
        time.sleep(0.2)
    raise AssertionError(f"task {task_id} stayed {store.get_task(task_id)['status']}")


def test_route_answers_422_for_an_external_start_with_no_output(ingest_client):
    client, ctx, store, _factory, _papers = ingest_client
    for md in ctx.local_md_dir.glob("*.md"):
        md.unlink()

    response = client.post("/api/pipeline/ingest", json={
        "venue": "CVPR", "year": 2026, "ssh_target": "unused", "start_stage": "preindex"})

    assert response.status_code == 422
    assert "extract" in response.json()["detail"] and "0/2" in response.json()["detail"]
    assert store.get_task(ctx.task_id + 1) is None


def test_route_passes_options_into_the_task_params(ingest_client):
    client, ctx, store, _factory, papers = ingest_client

    response = client.post("/api/pipeline/ingest", json={
        "venue": "CVPR", "year": 2026, "ssh_target": "unused", "start_stage": "preindex",
        "options": {"note": "from-the-route"}})

    assert response.status_code == 200, response.text
    task_id = response.json()["task_id"]
    assert _wait_for_status(store, task_id, {"completed", "failed", "cancelled"}) == "completed"
    assert store.get_task(task_id)["params"]["options"] == {"note": "from-the-route"}


def test_route_options_reach_the_venue_pipeline_validation(ingest_client):
    client, _ctx, store, _factory, _papers = ingest_client

    response = client.post("/api/pipeline/ingest", json={
        "venue": "Papers-Download", "year": 1, "ssh_target": "unused",
        "options": {"targets": ["not-a-slice"]}})

    assert response.status_code == 422  # the download-batch targets check saw the options
    assert store.list_running(INGEST_PIPELINE) == []


def test_a_local_download_is_refused_by_the_start_and_the_route_before_a_task_exists(ingest_client):
    client, ctx, store, data_factory, _papers = ingest_client
    message = "downloads run on cfscan into the 115 volume; this host only receives"
    runner = _runner(store, ctx, data_factory)

    with pytest.raises(ValueError, match=message):
        runner.start_ingest("CVPR", 2026, options={"download_from": "local"})
    response = client.post("/api/pipeline/ingest", json={
        "venue": "CVPR", "year": 2026, "ssh_target": "unused", "options": {"download_from": "local"}})

    assert response.status_code == 422
    assert message in response.json()["detail"]
    assert store.list_running(INGEST_PIPELINE) == []
    assert store.get_task(ctx.task_id + 1) is None  # no task row was written


def test_request_schema_options_is_an_optional_object():
    from pydantic import ValidationError

    from app.api.routes.pipeline import IngestPipelineRequest

    assert IngestPipelineRequest(venue="CVPR", year=2026, ssh_target="x").options is None
    assert IngestPipelineRequest(venue="CVPR", year=2026, ssh_target="x", options={"a": 1}).options == {"a": 1}
    with pytest.raises(ValidationError):
        IngestPipelineRequest(venue="CVPR", year=2026, ssh_target="x", options=["a"])
