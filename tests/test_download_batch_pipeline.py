"""A multi-venue download batch runs as an ingest_pipeline task with one download stage.

This host never downloads (downloads run on cfscan into the 115 volume), so
PipelineRunner refuses to create a batch. The batch machinery stays for a host
that does download: the end-to-end test drives a batch task that already exists
(``run_batch_task``, the restart path) on a real SQLite task table; the
orchestrator launches the real scripts/download_subset.py subprocess, which
downloads through a real local forward proxy (the recording proxy of
test_host_turn_download) into a temporary corpus. "IEEE S&P" is one of
the venues, so the '&' crosses the subprocess argv and the filesystem.
"""
from __future__ import annotations

import dataclasses
import json
import threading
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from app.services.pipeline.download_batch import (
    DOWNLOAD_BATCH_VENUE,
    DownloadBatchContext,
    build_download_batch_cmd,
    parse_download_targets,
)
from app.services.pipeline.registry import STAGE_KEYS, build_ingest_pipeline
from app.services.pipeline.service import INGEST_PIPELINE, PipelineRunner
from app.services.pipeline.store import TaskStore
from pipeline.paper_crawler.ccf_a_main_corpus import sanitize_file_component
from pipeline.paper_crawler.runtime_files import is_valid_pdf_artifact
from tests.sqlite_task_db import make_session_factory
from tests.test_host_turn_download import RecordingProxy, assert_no_avoidable_repeat, manifest_rows, tiny_pdf

TARGETS = ["IEEE S&P:2024", "JMLR:2024", "VLDB:2024"]


def write_manifest(corpus_root: Path, rows: list[dict]) -> None:
    path = corpus_root / "manifests" / "paper_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def batch_rows() -> list[dict]:
    rows = manifest_rows({("IEEE S&P", "csdl.test"): 2, ("JMLR", "jmlr.test"): 2, ("VLDB", "vldb.test"): 2})
    rows += manifest_rows({("ICLR", "openreview.test"): 2})  # another slice in the manifest, not a target
    return rows


def runner_for(tmp_path: Path, corpus_root: Path) -> PipelineRunner:
    store = TaskStore(make_session_factory(tmp_path / "tasks.db"))

    def ctx_factory(**kw):
        return dataclasses.replace(PipelineRunner._default_ctx(**kw), corpus_root=corpus_root)

    return PipelineRunner(store=store, context_factory=ctx_factory)


def run_batch_task(runner: PipelineRunner, targets: list[str], *, year: int = 1, **options) -> int:
    """Run a batch task that already exists, as a backend restart does (resume_running).

    start_download_batch refuses to create one on this host; the task row is written
    the way start_ingest wrote it, then the real orchestrator drives it to its end."""
    task_id = runner.store.create_task(
        task_type=INGEST_PIPELINE, total=len(STAGE_KEYS),
        params={"venue": DOWNLOAD_BATCH_VENUE, "year": year, "options": {"targets": list(targets), **options}})
    runner.store.set_task_status(task_id, "running")
    assert runner.resume_running(wait=True) == [task_id]
    return task_id


@pytest.fixture()
def proxy(monkeypatch):
    server = RecordingProxy(refuse_once={"jmlr.test"})
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # The stage subprocess inherits this; config.Settings reads PROXY__HTTP_PROXY.
    monkeypatch.setenv("PROXY__HTTP_PROXY", f"http://127.0.0.1:{server.server_address[1]}")
    yield server
    server.shutdown()
    server.server_close()


def test_batch_task_downloads_every_target_through_one_host_alternating_subprocess(tmp_path, proxy):
    corpus_root = tmp_path / "corpus"
    write_manifest(corpus_root, batch_rows())
    runner = runner_for(tmp_path, corpus_root)

    task_id = run_batch_task(runner, TARGETS, min_interval_seconds=0.0, host_cooldown_seconds=1.0)

    task = runner.store.get_task(task_id)
    assert task["status"] == "completed", task
    assert task["params"]["venue"] == DOWNLOAD_BATCH_VENUE and task["params"]["year"] == 1
    assert (task["progress"], task["total"]) == (6, 6)  # parent mirrors the PDF probe
    assert runner.store.get_subtask(task_id, "download")["status"] == "completed"
    for venue_dir in ("ieee s&p", "jmlr", "vldb"):
        pdfs = sorted((corpus_root / "pdf" / venue_dir / "2024").glob("*.pdf"))
        assert len(pdfs) == 2 and all(is_valid_pdf_artifact(p) for p in pdfs)
    assert not (corpus_root / "pdf" / "iclr").exists()
    hosts = [urlsplit(url).hostname for url in proxy.requests]
    assert sorted(set(hosts)) == ["csdl.test", "jmlr.test", "vldb.test"]
    assert hosts.count("jmlr.test") == 3  # one refusal, retried after its cooldown
    # The refused host waits out its cooldown, so its last paper may follow its
    # retry once no other host has work left; never before.
    assert_no_avoidable_repeat(hosts, Counter(hosts))

    # A second batch over the same slices is batch 2 and finds every PDF on disk.
    again = run_batch_task(runner, TARGETS, year=2)
    assert runner.store.get_task(again)["params"]["year"] == 2
    assert runner.store.get_task(again)["status"] == "completed"
    assert len(proxy.requests) == len(hosts)


def test_batch_rejects_bad_targets_before_a_task_exists(tmp_path):
    runner = runner_for(tmp_path, tmp_path / "corpus")
    for bad in ([], ["ECCV"], ["ECCV:20x4"], [":2024"]):
        with pytest.raises(ValueError):
            runner.start_download_batch(bad)
    with pytest.raises(ValueError):
        runner.start_ingest(DOWNLOAD_BATCH_VENUE, 1, options={"targets": ["ECCV:2024"]}, start_stage="extract")
    assert runner.store.list_running(INGEST_PIPELINE) == []


def test_this_host_refuses_to_create_a_download_batch_before_a_task_exists(tmp_path):
    runner = runner_for(tmp_path, tmp_path / "corpus")
    message = "downloads run on cfscan into the 115 volume; this host only receives"
    with pytest.raises(ValueError, match=message):
        runner.start_download_batch(["ECCV:2024"])
    with pytest.raises(ValueError, match=message):
        runner.start_download_batch(["ECCV:2024"], collect=True, wait=True)
    with pytest.raises(ValueError, match=message):
        runner.start_ingest(DOWNLOAD_BATCH_VENUE, 1, options={"targets": ["ECCV:2024"]})
    assert runner.store.list_running(INGEST_PIPELINE) == []
    assert runner.store.get_task(1) is None  # no task row was written


def test_batch_context_builds_a_single_download_stage_and_command(tmp_path):
    ctx = PipelineRunner._default_ctx(
        venue="Papers-Download", year=3, store=None, task_id=0, concurrency=8, world_size=8,
        options={"targets": TARGETS + ["JMLR:2024"]},
    )
    ctx = dataclasses.replace(ctx, corpus_root=tmp_path)
    assert isinstance(ctx, DownloadBatchContext)
    assert parse_download_targets(ctx.options["targets"]) == [("IEEE S&P", 2024), ("JMLR", 2024), ("VLDB", 2024)]
    specs = build_ingest_pipeline(ctx)
    assert [(s.key, s.resource_key, s.parent_progress_from_probe) for s in specs] == [("download", "net", True)]
    cmd = build_download_batch_cmd(ctx)
    assert cmd[1] == "scripts/download_subset.py"
    assert cmd[cmd.index("--target") + 1] == "IEEE S&P:2024"  # one argv item, never a shell string
    assert "--host-turns" in cmd and cmd[cmd.index("--min-interval-seconds") + 1] == "3"
    assert cmd[cmd.index("--corpus-root") + 1] == str(tmp_path)


def test_batch_probe_counts_only_this_slices_files(tmp_path):
    corpus_root = tmp_path / "corpus"
    write_manifest(corpus_root, batch_rows())
    stray = corpus_root / "pdf" / "vldb" / "2024"
    stray.mkdir(parents=True)
    (stray / "an-older-vldb-paper.pdf").write_bytes(tiny_pdf())  # not in the manifest slice
    (stray / "vldb-2024-vldb-0.pdf").write_bytes(tiny_pdf())
    ctx = DownloadBatchContext(
        venue=DOWNLOAD_BATCH_VENUE, year=1, store=None, task_id=0, corpus_root=corpus_root,
        options={"targets": TARGETS},
    )
    assert ctx.target_progress() == [
        ("IEEE S&P", 2024, 0, 0, 2), ("JMLR", 2024, 0, 0, 2), ("VLDB", 2024, 1, 0, 2),
    ]
    specs = build_ingest_pipeline(ctx)
    assert specs[0].probe(ctx).done == 1 and specs[0].probe(ctx).total == 6
    gate = specs[0].gate(ctx)
    assert not gate.ok and "IEEE S&P 2024 0+0 rejected/2" in gate.reason


def test_batch_gate_counts_identity_rejected_papers_as_settled(tmp_path):
    corpus_root = tmp_path / "corpus"
    rows = batch_rows()
    write_manifest(corpus_root, rows)
    settled = {"IEEE S&P": ("pdf", "rejected_identity"), "JMLR": ("pdf", "pdf"),
               "VLDB": ("rejected_identity", "rejected_identity")}
    for row in rows:
        if row["conference"] not in settled:
            continue
        index = int(row["paper_id"].rsplit("-", 1)[1])
        directory = corpus_root / settled[row["conference"]][index] / row["conference"].lower() / "2024"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{sanitize_file_component(row['paper_id'])}.pdf").write_bytes(tiny_pdf())
    ctx = DownloadBatchContext(
        venue=DOWNLOAD_BATCH_VENUE, year=1, store=None, task_id=0, corpus_root=corpus_root,
        options={"targets": TARGETS},
    )
    assert ctx.target_progress() == [
        ("IEEE S&P", 2024, 1, 1, 2), ("JMLR", 2024, 2, 0, 2), ("VLDB", 2024, 0, 2, 2),
    ]
    spec = build_ingest_pipeline(ctx)[0]
    assert spec.probe(ctx).done == 3 and "VLDB 2024 0/2 (2 rejected)" in spec.probe(ctx).message
    assert spec.gate(ctx).ok
