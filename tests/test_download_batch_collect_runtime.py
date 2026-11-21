"""The optional collect stage of a download batch (options.collect, default off).

This host refuses to create a batch (downloads run on cfscan), so the tests that run one drive a
batch task row that already exists, the way a backend restart resumes it (``run_batch_task``).

Real manifest files, the real orchestrator over the production stage builders on a
real SQLite task table, the real download_subset.py subprocess behind the local
recording proxy of test_host_turn_download, and the real collect script as a
subprocess. What cannot run here is the collect script's SUCCESS path: it reads the
publishers' https pages through the proxy, which needs the live proxy and sites
(the parsers themselves are pinned to captured pages in test_tier1_sources, and the
manifest merge in test_collect_targets_and_slice_merge). The failure path below is
the real script failing for real, with no manifest change beyond its backup.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.pipeline.download_batch import (
    DownloadBatchContext,
    build_collect_cmd,
    collect_gate,
    collect_probe,
)
from app.services.pipeline.registry import build_ingest_pipeline
from app.services.pipeline.service import PipelineRunner
from app.services.pipeline.spec import Progress, StageSpec
from core.db.models import TaskEvent
from tests.test_download_batch_pipeline import (  # noqa: F401 - proxy is a fixture
    TARGETS,
    batch_rows,
    proxy,
    run_batch_task,
    runner_for,
    write_manifest,
)
from tests.test_host_turn_download import manifest_rows


def _ctx(corpus_root: Path, *, collect: bool, targets=TARGETS) -> DownloadBatchContext:
    options = {"targets": list(targets)}
    if collect:
        options["collect"] = True
    return DownloadBatchContext(venue="Papers-Download", year=1, store=None, task_id=0,
                                corpus_root=corpus_root, options=options)


def _rows_for(*slices: tuple[str, str]) -> list[dict]:
    return manifest_rows({(conference, host): 2 for conference, host in slices})


def test_collect_is_off_by_default_and_the_batch_keeps_its_single_stage(tmp_path):
    write_manifest(tmp_path, batch_rows())
    specs = build_ingest_pipeline(_ctx(tmp_path, collect=False))
    assert [(s.key, s.parent_progress_from_probe) for s in specs] == [("download", True)]
    # Falsy spellings of the option stay off, too.
    for off in (False, 0, None):
        ctx = dataclasses.replace(_ctx(tmp_path, collect=False), options={"targets": TARGETS, "collect": off})
        assert [s.key for s in build_ingest_pipeline(ctx)] == ["download"]


def test_collect_stage_is_first_takes_the_net_lock_and_leaves_parent_progress_to_the_stage_count(tmp_path):
    write_manifest(tmp_path, batch_rows())
    specs = build_ingest_pipeline(_ctx(tmp_path, collect=True))
    assert [(s.key, s.resource_key) for s in specs] == [("collect", "net"), ("download", "net")]
    # The orchestrator only allows a probe-driven parent for a single-stage pipeline.
    assert [s.parent_progress_from_probe for s in specs] == [False, False]
    assert specs[0].skip_when_satisfied and specs[0].poll_interval_s == 0


def test_collect_probe_gate_and_command_cover_only_the_slices_without_rows(tmp_path):
    # IEEE S&P 2024 has rows; JMLR 2024 and VLDB 2024 were never collected here.
    write_manifest(tmp_path, _rows_for(("IEEE S&P", "csdl.test"), ("ICLR", "openreview.test")))
    ctx = _ctx(tmp_path, collect=True)

    assert ctx.target_row_counts() == [("IEEE S&P", 2024, 2), ("JMLR", 2024, 0), ("VLDB", 2024, 0)]
    progress = collect_probe(ctx)
    assert (progress.done, progress.total) == (1, 3) and not progress.is_complete
    assert "JMLR 2024 0 rows" in progress.message
    gate = collect_gate(ctx)
    assert not gate.ok and "JMLR 2024, VLDB 2024" in gate.reason and "IEEE" not in gate.reason

    cmd = build_collect_cmd(ctx)
    assert cmd[1] == "scripts/collect/collect_2026_to_manifest.py"
    assert [cmd[i + 1] for i, a in enumerate(cmd) if a == "--target"] == ["JMLR:2024", "VLDB:2024"]
    assert cmd[cmd.index("--corpus-root") + 1] == str(tmp_path)  # this context's manifest, not the default

    (tmp_path / "manifests" / "paper_manifest.json").write_text(json.dumps(batch_rows()), encoding="utf-8")
    assert collect_probe(ctx).is_complete and collect_gate(ctx).ok


def test_batch_with_collect_skips_it_when_every_slice_has_rows_and_downloads(tmp_path, proxy):
    corpus_root = tmp_path / "corpus"
    write_manifest(corpus_root, batch_rows())
    runner = runner_for(tmp_path, corpus_root)

    task_id = run_batch_task(
        runner, TARGETS, min_interval_seconds=0.0, host_cooldown_seconds=1.0, collect=True)

    store = runner.store
    task = store.get_task(task_id)
    assert task["status"] == "completed", task
    assert task["params"]["options"]["collect"] is True
    collect = store.get_subtask(task_id, "collect")
    assert (collect["status"], collect["message"]) == ("completed", "already satisfied")
    assert (collect["progress"], collect["total"]) == (3, 3)
    assert store.get_subtask(task_id, "download")["status"] == "completed"
    assert (task["progress"], task["total"]) == (2, 2)  # stage count, since download is not alone
    for venue_dir in ("ieee s&p", "jmlr", "vldb"):
        assert len(list((corpus_root / "pdf" / venue_dir / "2024").glob("*.pdf"))) == 2


def test_a_slice_that_cannot_be_collected_stops_the_batch_before_the_download(tmp_path, monkeypatch):
    # No proxy is listening: the real collect script cannot reach any publisher.
    monkeypatch.setenv("PROXY__HTTP_PROXY", "http://127.0.0.1:9")
    corpus_root = tmp_path / "corpus"
    rows = _rows_for(("IEEE S&P", "csdl.test"), ("VLDB", "vldb.test"))  # JMLR 2024 is missing
    write_manifest(corpus_root, rows)
    manifest = corpus_root / "manifests" / "paper_manifest.json"
    before = manifest.read_bytes()
    runner = runner_for(tmp_path, corpus_root)

    task_id = run_batch_task(runner, TARGETS, min_interval_seconds=0.0, collect=True)

    store = runner.store
    task = store.get_task(task_id)
    assert task["status"] == "failed", task
    collect = store.get_subtask(task_id, "collect")
    assert collect["status"] == "failed" and "command failed" in collect["message"]
    download = store.get_subtask(task_id, "download")
    assert download is None or download["status"] != "completed"
    assert not (corpus_root / "pdf").exists()  # nothing was downloaded
    assert manifest.read_bytes() == before  # the failed collect changed no row
    with store._sf() as s:
        log = "\n".join(e.message for e in s.query(TaskEvent).filter_by(task_id=task_id))
    assert "1 of 3 slices have no manifest rows yet: JMLR 2024" in log
    assert "--target JMLR:2024 --corpus-root" in log  # only the missing slice went to the collector


# ----------------------------------------------------------------- the surface
def test_route_and_service_refuse_a_batch_before_any_task_exists(tmp_path, monkeypatch):
    from app.api.routes.pipeline import DownloadBatchRequest, router
    from app.services.pipeline import service
    from app.services.pipeline.service import INGEST_PIPELINE

    assert DownloadBatchRequest(targets=["ECCV:2024"]).collect is False

    runner = runner_for(tmp_path, tmp_path / "corpus")
    monkeypatch.setattr(service, "_runner", runner)
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    for body in ({"targets": ["ECCV:2024"]}, {"targets": ["ECCV:2024"], "collect": True}):
        response = client.post("/api/pipeline/download-batch", json=body)
        assert response.status_code == 422, response.text
        assert "downloads run on cfscan into the 115 volume; this host only receives" in response.json()["detail"]
    assert runner.store.list_running(INGEST_PIPELINE) == []
    assert runner.store.get_task(1) is None  # no task row was written
