"""The arXiv delta run's stages: selection, context, download stage, listing, schedule kind.

Real SQLite files (the application's tables and the harvest database written by
HarvestStore), real subprocesses where a stage runs a CLI, real filesystem trees.
Nothing is faked; the one swapped command is `enumerate` (OAI-PMH over the network),
replaced by a harmless real process, and the download CLI runs against a harvest
database with nothing pending so it does not reach arXiv.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import func

from app.api.routes import pipeline as pipeline_routes
from app.services.pipeline import arxiv_schedule, arxiv_stages
from app.services.pipeline.arxiv_stages import (
    ArxivStageContext,
    NotADeltaRun,
    RunSources,
    arxiv_delta_download_probe,
    arxiv_extract_probe,
    build_arxiv_extract_cmd,
    build_delta_download_cmd,
    default_run_sources,
    delta_run_page,
    freeze_delta,
    make_arxiv_extract_stage,
    make_local_arxiv_delta_download_stage,
    validate_slice_key,
)
from app.services.pipeline.service import INGEST_PIPELINE, PipelineRunner
from app.services.pipeline.stages import (
    make_ingest_stage,
    make_embed_stage_for,
    nothing_to_embed,
    preindex_report_path,
)
from app.services.pipeline.store import TaskStore
from core.db.models import Document as DBDocument, Paper as DBPaper, SourceType
from pipeline.paper_crawler.arxiv_corpus import DELTA_KEY, paper_id_for
from pipeline.paper_crawler.arxiv_html import ArxivRawRecord, HarvestStore, html_path
from tests.sqlite_task_db import make_session_factory

REPO = Path(__file__).resolve().parents[1]


def raw(arxiv_id: str, *, version="v1", withdrawn=False, date="2026-04-02T00:00:00+00:00") -> ArxivRawRecord:
    return ArxivRawRecord(
        arxiv_id=arxiv_id, datestamp="2026-04-03", categories="cs.LG", title=f"title {arxiv_id}",
        authors="A B and C D", abstract="abstract", comments=None, license=None,
        latest_version=version, latest_version_date=date, first_version_date="2026-04-01T00:00:00+00:00",
        latest_size="0kb" if withdrawn else "10kb", content_version=version,
        content_version_date=date, withdrawn=withdrawn,
    )


def seed_sql(db, arxiv_id: str, version: str | None, withdrawn: bool = False, title: str | None = None) -> None:
    doc = DBDocument(title=arxiv_id, source="s", source_type=SourceType.paper)
    db.add(doc)
    db.flush()
    db.add(DBPaper(paper_id=paper_id_for(arxiv_id), doc_id=doc.id, title=title or f"sql {arxiv_id}",
                   arxiv_id=arxiv_id, source_version=version, withdrawn=withdrawn))


def new_ctx(tmp_path: Path, *, store=None, task_id: int = 7, **kw) -> ArxivStageContext:
    return ArxivStageContext(venue="arXiv", year=DELTA_KEY, store=store, task_id=task_id,
                             project_root=REPO, corpus_root=tmp_path / "corpus",
                             harvest_root=tmp_path / "harvest", **kw)


# ----------------------------------------------------------------- freeze_delta
def test_freeze_splits_harvest_rows_against_sql(tmp_path):
    factory = make_session_factory(tmp_path / "app.db")
    store = HarvestStore(tmp_path / "harvest" / "arxiv_cs.sqlite3")
    # Everything first arrives at v1 with its HTML fetched; some then move on.
    store.upsert([raw(f"2601.0000{i}") for i in range(1, 10)] + [raw("2601.00010"), raw("2512.00001"),
                                                               raw("cs/0101001")])
    for arxiv_id in [f"2601.0000{i}" for i in range(1, 10)] + ["2512.00001"]:
        store.mark(arxiv_id, "ok", html_version="v1")
    store.mark("2601.00010", "no_html")  # judged before this run: not this run's html_missing
    store.upsert([
        raw("2601.00002", version="v2"),                # new text, HTML not fetched again yet
        raw("2601.00003", withdrawn=True),              # withdrawn without a new text
        raw("2601.00009", withdrawn=False),             # revived: flag off in harvest, on in SQL
    ])
    store.mark("2601.00002", "ok", html_version="v2")   # ... and now it is: replace
    store.upsert([raw("2601.00004", version="v2", withdrawn=True)])  # new text, HTML missing, flag on
    store.mark("2601.00008", "ok", html_version="v1")
    store.upsert([raw("2601.00008", version="v2")])     # requeued: pending, stale HTML on disk
    started = datetime.now(timezone.utc).isoformat()
    time.sleep(0.02)
    store.upsert([raw("2601.00001")])                   # untouched row, but not in SQL: new
    store.mark("2601.00001", "ok", html_version="v1")
    store.mark("2601.00006", "no_html")                 # this run: arXiv has no HTML
    store.mark("2601.00007", "error", error="boom")     # this run: a failed download
    store.close()

    with factory() as db:
        seed_sql(db, "2601.00002", "v1")                # v1 indexed, v2 downloaded -> replaced
        seed_sql(db, "2601.00003", "v1")                # flag only
        seed_sql(db, "2601.00004", "v1")                # new text not downloaded: flag only
        seed_sql(db, "2601.00005", "v1")                # nothing changed
        seed_sql(db, "2601.00009", "v1", withdrawn=True)
        seed_sql(db, "2512.00001", None)                # outside the delta scope: ignored, not a blocker
        seed_sql(db, "cs/0101001", None)                # legacy id: ignored too

    frozen = freeze_delta(tmp_path / "harvest" / "arxiv_cs.sqlite3", factory, started=started)

    assert frozen["ids"] == ["2601.00001", "2601.00002"]
    assert frozen["withdrawn_only"] == {"arxiv-2601.00003": True, "arxiv-2601.00004": True,
                                        "arxiv-2601.00009": False}
    assert frozen["html_missing"] == ["2601.00006"]
    assert frozen["convert_rejected"] == []
    assert frozen["counts"] == {"selected": 2, "withdrawn_only": 3, "html_missing": 1, "convert_rejected": 0,
                                "download_errors": 1}
    assert frozen["frozen_at"]


def test_freeze_refuses_while_arxiv_papers_lack_a_version(tmp_path):
    factory = make_session_factory(tmp_path / "app.db")
    store = HarvestStore(tmp_path / "harvest" / "arxiv_cs.sqlite3")
    store.upsert([raw("2601.00001")])
    store.mark("2601.00001", "ok", html_version="v1")
    store.close()
    with factory() as db:
        seed_sql(db, "2601.00001", None)
        seed_sql(db, "2601.00002", "v1")
    with pytest.raises(RuntimeError, match=r"1 arXiv papers in SQL have no source_version.*backfill_arxiv"):
        freeze_delta(tmp_path / "harvest" / "arxiv_cs.sqlite3", factory, started="2026-01-01T00:00:00+00:00")


# ---------------------------------------------------------------------- context
def test_delta_context_paths_hooks_and_frozen_file(tmp_path):
    validate_slice_key(DELTA_KEY)
    ctx = new_ctx(tmp_path)
    assert ctx.is_delta and ctx.slice_key == 0
    assert ctx.delta_file == tmp_path / "corpus" / "deltas" / "7.json"
    assert "delta_7" in ctx.paths.manifest.name
    assert ctx.local_md_dir == tmp_path / "corpus" / "markdown" / "arxiv"
    assert (ctx.ok_ids(), ctx.withdrawn_only(), ctx.result_extra(), ctx.harvest_counts()) == ([], {}, {}, {})
    assert ctx.preindex_selection() == ["--manifest-path", str(ctx.manifest_path), "--conf", "arXiv"]

    ctx.delta_file.parent.mkdir(parents=True)
    ctx.delta_file.write_text(json.dumps({
        "ids": ["2601.00001", "2602.00002"], "withdrawn_only": {"arxiv-2601.00009": True, "arxiv-2601.00008": False},
        "html_missing": ["2601.00006"],
        "counts": {"selected": 2, "withdrawn_only": 2, "html_missing": 1, "download_errors": 0},
    }), encoding="utf-8")
    assert ctx.ok_ids() == ["2601.00001", "2602.00002"]
    assert ctx.withdrawn_only() == {"arxiv-2601.00009": True, "arxiv-2601.00008": False}
    assert ctx.result_extra() == {"selected": 2, "withdrawn_only": 2, "html_missing": 1, "download_errors": 0,
                                  "convert_rejected": 0, "convert_errors": 0}
    assert ctx.harvest_counts() == {"ok": 2, "no_html": 1}
    assert ctx.md_count_local() == 0
    md = ctx.paths.markdown("2601.00001")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("# t", encoding="utf-8")
    assert ctx.md_count_local() == 1


def test_delta_paper_clause_selects_exactly_the_frozen_ids_beyond_the_variable_limit(tmp_path):
    factory = make_session_factory(tmp_path / "app.db")
    ctx = new_ctx(tmp_path)
    ids = [f"2601.{n:05d}" for n in range(1, 40001)]  # above SQLite's 32766 bound variables
    ctx.delta_file.parent.mkdir(parents=True)
    ctx.delta_file.write_text(json.dumps({"ids": ids}), encoding="utf-8")
    with factory() as db:
        seed_sql(db, "2601.00001", "v1")
        seed_sql(db, "2601.39999", "v1")
        seed_sql(db, "2601.40001", "v1")  # not in the run
        seed_sql(db, "2512.00001", "v1")
        assert db.query(func.count(DBPaper.paper_id)).filter(ctx.paper_count_clause(DBPaper)).scalar() == 2


def test_delta_commands(tmp_path):
    ctx = new_ctx(tmp_path, options={"html_concurrency": 2, "no_html_retry_days": 5, "limit": 40})
    extract = build_arxiv_extract_cmd(ctx)
    assert extract[1:4] == ["scripts/collect/arxiv_cs_corpus.py", "convert", "--ids-file"]
    assert extract[4] == str(ctx.delta_file) and extract[5:7] == ["--run-id", "7"] and "--slice" not in extract

    download = build_delta_download_cmd(ctx)
    assert download[1:3] == ["scripts/collect/arxiv_cs_html.py", "download"]
    pairs = dict(zip(download[3::2], download[4::2]))
    assert pairs["--scope"] == "delta" and pairs["--concurrency"] == "2"
    assert pairs["--no-html-retry-days"] == "5" and pairs["--max-exits"] == "3" and pairs["--limit"] == "40"
    assert "--proxy-pool-file" not in download and "--push-to" not in download
    # The harvest CLI chains through a SOCKS5 first hop; the configured proxy is an http:// URL.
    for cmd in (download, arxiv_stages.build_delta_enumerate_cmd(ctx)):
        hop = cmd[cmd.index("--first-hop-proxy") + 1]
        assert hop.startswith("socks5://127.0.0.1:") and hop.rsplit(":", 1)[1].isdigit()

    plain = build_delta_download_cmd(new_ctx(tmp_path))
    assert "--limit" not in plain and plain[plain.index("--concurrency") + 1] == "3"


def test_max_exits_is_validated_by_the_download_cli(tmp_path):
    def download(value: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "scripts/collect/arxiv_cs_html.py", "download", "--root", str(tmp_path / "harvest"),
             "--first-hop-proxy", "socks5://127.0.0.1:1", "--scope", "delta", "--max-exits", value],
            cwd=REPO, capture_output=True, text=True, timeout=300)

    rejected = download("0")
    assert rejected.returncode == 2 and "--max-exits must be at least 1" in rejected.stderr
    accepted = download("2")  # an empty queue: nothing to fetch, so no network
    assert accepted.returncode == 0, accepted.stderr[-800:]


def touch(path: Path, when: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    os.utime(path, (when, when))


def test_extract_probe_counts_only_outputs_newer_than_their_html(tmp_path):
    ctx = new_ctx(tmp_path)
    HarvestStore(ctx.harvest_db).close()  # the harvest database the probe reads convert outcomes from
    ids = ["2601.00001", "2601.00002", "2601.00003"]
    ctx.delta_file.parent.mkdir(parents=True)
    ctx.delta_file.write_text(json.dumps({"ids": ids}), encoding="utf-8")
    now = time.time()
    # 1: converted after its HTML; 2: an older version's files, HTML newer; 3: no HTML on disk.
    touch(html_path(ctx.harvest_root, ids[0]), now - 100)
    touch(ctx.paths.markdown(ids[0]), now - 50)
    touch(ctx.paths.doc(ids[0]), now - 50)
    touch(html_path(ctx.harvest_root, ids[1]), now - 10)
    touch(ctx.paths.markdown(ids[1]), now - 500)
    touch(ctx.paths.doc(ids[1]), now - 500)
    touch(ctx.paths.markdown(ids[2]), now - 50)
    touch(ctx.paths.doc(ids[2]), now - 50)

    probe = arxiv_extract_probe(ctx)
    assert (probe.done, probe.total) == (1, 3)

    # Fully converted but without the manifest preindex reads: one short of done.
    touch(html_path(ctx.harvest_root, ids[2]), now - 100)
    touch(ctx.paths.markdown(ids[1]), now)
    touch(ctx.paths.doc(ids[1]), now)
    probe = arxiv_extract_probe(ctx)
    assert (probe.done, probe.total) == (2, 3)
    ctx.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.manifest_path.write_text("[]", encoding="utf-8")
    assert arxiv_extract_probe(ctx).done == 3


def test_extract_with_nothing_frozen_converts_nothing(tmp_path):
    ctx = new_ctx(tmp_path)
    spec = make_arxiv_extract_stage(ctx)
    lines: list[str] = []
    spec.run(ctx, lambda m, level="info": lines.append(m))  # would raise if it ran the converter
    assert lines == ["no new HTML in this run: nothing to convert"]
    assert spec.gate(ctx).ok


# ---------------------------------------------------------------- download stage
@pytest.fixture()
def store(tmp_path):
    return TaskStore(make_session_factory(tmp_path / "tasks.db"))


def test_local_download_stage_enumerates_downloads_freezes_and_resumes(tmp_path, store, monkeypatch):
    # make_arxiv_delta_download_stage never builds this stage on this host (source is volume); the
    # builder is called directly to keep the enumerate / freeze / resume behaviour pinned.
    factory = make_session_factory(tmp_path / "app.db")
    # Rows with nothing to fetch (HTML already held, no_html too old to retry), so the
    # real download CLI has an empty queue and never reaches arXiv.
    harvest = HarvestStore(tmp_path / "harvest" / "arxiv_cs.sqlite3")
    old = "2026-01-02T00:00:00+00:00"
    harvest.upsert([raw("2601.00001", date=old), raw("2601.00002", version="v2", date=old),
                    raw("2601.00003", withdrawn=True, date=old), raw("2601.00004", date=old)])
    for arxiv_id in ("2601.00001", "2601.00003"):
        harvest.mark(arxiv_id, "ok", html_version="v1")
    harvest.mark("2601.00002", "ok", html_version="v2")
    harvest.mark("2601.00004", "no_html")
    harvest.close()
    with factory() as db:
        seed_sql(db, "2601.00001", "v1")        # unchanged
        seed_sql(db, "2601.00002", "v1")        # replaced by v2
        seed_sql(db, "2601.00003", "v1")        # withdrawn flag only

    task_id = store.create_task(task_type=INGEST_PIPELINE, params={"venue": "arXiv", "year": 0})
    ctx = new_ctx(tmp_path, store=store, task_id=task_id, options={"no_html_retry_days": 0})
    marker = tmp_path / "enumerated.txt"
    monkeypatch.setattr(arxiv_stages, "build_delta_enumerate_cmd", lambda c: [
        sys.executable, "-c", f"open({str(marker)!r}, 'a').write('x'); print('enumerated')"])
    spec = make_local_arxiv_delta_download_stage(ctx, session_factory=factory)
    assert spec.resource_key == "net" and not spec.gate(ctx).ok
    assert (arxiv_delta_download_probe(ctx).done, arxiv_delta_download_probe(ctx).total) == (0, 0)

    lines: list[str] = []
    spec.run(ctx, lambda m, level="info": lines.append(m))

    assert marker.read_text() == "x" and "enumerated" in " ".join(lines)
    frozen = json.loads(ctx.delta_file.read_text(encoding="utf-8"))
    assert frozen["ids"] == ["2601.00002"]
    assert frozen["withdrawn_only"] == {"arxiv-2601.00003": True}
    assert frozen["counts"] == {"selected": 1, "withdrawn_only": 1, "html_missing": 0, "convert_rejected": 0,
                                "download_errors": 0}
    assert frozen["task_id"] == task_id
    assert spec.gate(ctx).ok
    first_start = ctx.state_get("delta_started")
    assert first_start and ctx.state_get("delta_queued") == 0

    spec.run(ctx, lambda *a, **k: None)  # a resumed run keeps the original start
    assert ctx.state_get("delta_started") == first_start


def test_download_probe_is_one_short_until_the_run_is_frozen(tmp_path, store):
    harvest = HarvestStore(tmp_path / "harvest" / "arxiv_cs.sqlite3")
    harvest.upsert([raw("2601.00001"), raw("2601.00002"), raw("2601.00003")])
    started = datetime.now(timezone.utc).isoformat()
    time.sleep(0.02)
    harvest.mark("2601.00001", "ok", html_version="v1")
    harvest.mark("2601.00002", "no_html")
    harvest.close()
    task_id = store.create_task(task_type=INGEST_PIPELINE, params={})
    ctx = new_ctx(tmp_path, store=store, task_id=task_id)
    ctx.state_set(delta_started=started, delta_queued=3)
    probe = arxiv_delta_download_probe(ctx)
    assert (probe.done, probe.total) == (2, 3)

    harvest = HarvestStore(tmp_path / "harvest" / "arxiv_cs.sqlite3")
    harvest.mark("2601.00003", "error", error="x")
    harvest.close()
    assert arxiv_delta_download_probe(ctx).done == 2  # all attempted, but not frozen yet
    ctx.delta_file.parent.mkdir(parents=True)
    ctx.delta_file.write_text("{}", encoding="utf-8")
    assert arxiv_delta_download_probe(ctx).done == 3


# ----------------------------------------------- nothing to embed after preindex
def test_embed_and_ingest_do_nothing_when_preindex_wrote_no_chunk(tmp_path, store):
    task_id = store.create_task(task_type=INGEST_PIPELINE, params={})
    ctx = new_ctx(tmp_path, store=store, task_id=task_id)
    assert not nothing_to_embed(ctx)  # preindex has not run
    ctx.state_set(chunk_id_before=100, chunk_id_after=100)
    assert nothing_to_embed(ctx)
    for spec in (make_embed_stage_for(ctx), make_ingest_stage(ctx)):
        lines: list[str] = []
        spec.run(ctx, lambda m, level="info": lines.append(m))
        assert lines == [f"{spec.key}: preindex wrote no chunks, nothing to do"]
        assert spec.gate(ctx).ok
        if spec.preflight is not None:
            assert spec.preflight(ctx).ok
    ctx.state_set(chunk_id_after=130)
    assert not nothing_to_embed(ctx)
    assert not make_ingest_stage(ctx).gate(ctx).ok  # the real gate is back: no new parquet


# ------------------------------------------------------------------ run listing
def make_sources(tmp_path: Path, store: TaskStore, factory) -> RunSources:
    return RunSources(store=store, session_factory=factory, corpus_root=tmp_path / "corpus",
                      harvest_root=tmp_path / "harvest")


def write_run_files(sources: RunSources, task_id: int) -> None:
    ctx = ArxivStageContext(venue="arXiv", year=DELTA_KEY, store=sources.store, task_id=task_id,
                            corpus_root=sources.corpus_root, harvest_root=sources.harvest_root)
    ctx.delta_file.parent.mkdir(parents=True, exist_ok=True)
    ctx.delta_file.write_text(json.dumps({
        "ids": ["2601.00001", "2601.00002"], "withdrawn_only": {"arxiv-2601.00003": True, "arxiv-2601.00004": False},
        "html_missing": ["2601.00006"], "convert_rejected": ["2601.00007"], "counts": {}}), encoding="utf-8")
    ctx.paths.report.parent.mkdir(parents=True, exist_ok=True)
    ctx.paths.report.write_text(json.dumps({"rejected": ["2601.00007", "2601.00001"], "errors": []}),
                                encoding="utf-8")
    report = preindex_report_path(ctx)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "new": ["arxiv-2601.00001"], "replaced": ["arxiv-2601.00002"],
        "withdrawn_changed": [{"paper_id": "arxiv-2601.00002", "withdrawn": True}]}), encoding="utf-8")


def test_run_listing_reads_the_files_the_run_wrote(tmp_path, store):
    factory = make_session_factory(tmp_path / "app.db")
    harvest = HarvestStore(tmp_path / "harvest" / "arxiv_cs.sqlite3")
    harvest.upsert([raw("2601.00006"), raw("2601.00007")])
    harvest.close()
    with factory() as db:
        seed_sql(db, "2601.00002", "v2", withdrawn=True, title="indexed title")
    sources = make_sources(tmp_path, store, factory)
    task_id = store.create_task(task_type=INGEST_PIPELINE, params={"venue": "arXiv", "year": 0})
    write_run_files(sources, task_id)

    page = delta_run_page(sources, task_id, "replaced", 0, 50)
    assert page["counts"] == {"new": 1, "replaced": 1, "withdrawn": 2, "revived": 1, "html_missing": 1,
                              "convert_rejected": 2}
    assert page["total"] == 1 and (page["offset"], page["limit"], page["kind"]) == (0, 50, "replaced")
    assert page["papers"] == [{"paper_id": "arxiv-2601.00002", "arxiv_id": "2601.00002",
                               "title": "indexed title", "source_version": "v2", "withdrawn": True}]

    missing = delta_run_page(sources, task_id, "html_missing", 0, 50)["papers"]
    assert missing == [{"paper_id": "arxiv-2601.00006", "arxiv_id": "2601.00006",
                        "title": "title 2601.00006", "source_version": None, "withdrawn": None}]
    assert [p["paper_id"] for p in delta_run_page(sources, task_id, "withdrawn", 1, 1)["papers"]] == [
        "arxiv-2601.00003"]

    # Rejected: one the freeze left out, one this run's convert found (listed once); titled from the harvest.
    rejected = delta_run_page(sources, task_id, "convert_rejected", 0, 50)["papers"]
    assert [(p["arxiv_id"], p["title"]) for p in rejected] == [
        ("2601.00007", "title 2601.00007"), ("2601.00001", None)]

    other = store.create_task(task_type=INGEST_PIPELINE, params={"venue": "arXiv", "year": 202601})
    for bad in (other, 999999):
        with pytest.raises(NotADeltaRun):
            delta_run_page(sources, bad, "new", 0, 10)


LAN_PEER = ("10.100.100.9", 51000)
INTERNAL = "http://10.100.100.1:8000"


@pytest.mark.asyncio
async def test_route_serves_a_page_and_rejects_other_tasks(tmp_path, store):
    factory = make_session_factory(tmp_path / "app.db")
    sources = make_sources(tmp_path, store, factory)
    task_id = store.create_task(task_type=INGEST_PIPELINE, params={"venue": "arXiv", "year": 0})
    write_run_files(sources, task_id)
    other = store.create_task(task_type=INGEST_PIPELINE, params={"venue": "ACL", "year": 2026})

    app = FastAPI()
    app.include_router(pipeline_routes.router)
    app.dependency_overrides[default_run_sources] = lambda: sources
    url = f"/api/pipeline/arxiv/runs/{task_id}/papers"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=LAN_PEER),
                                 base_url=INTERNAL) as http:
        ok = await http.get(url, params={"kind": "new"})
        assert ok.status_code == 200, ok.text
        body = ok.json()
        assert set(body) == {"task_id", "kind", "offset", "limit", "total", "counts", "papers"}
        assert body["task_id"] == task_id and body["total"] == 1 and body["limit"] == 50
        assert body["papers"][0]["paper_id"] == "arxiv-2601.00001"
        assert set(body["papers"][0]) == {"paper_id", "arxiv_id", "title", "source_version", "withdrawn"}

        assert (await http.get(f"/api/pipeline/arxiv/runs/{other}/papers", params={"kind": "new"})).status_code == 404
        assert (await http.get(url, params={"kind": "bogus"})).status_code == 422
        assert (await http.get(url, params={"kind": "new", "limit": 501})).status_code == 422

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=("8.8.8.8", 1)),
                                 base_url="http://example.com") as outsider:
        assert (await outsider.get(url, params={"kind": "new"})).status_code in (401, 403)


# --------------------------------------------------------------- schedule kind
def test_kind_is_registered_with_the_daily_seed_and_bounded_params():
    from app.services.scheduler.kinds import all_kinds

    kind = all_kinds()[arxiv_schedule.KIND]
    assert kind.name == "arxiv_delta"
    assert kind.retry.max_retries == 2 and kind.retry.base_delay_s == 1800
    assert kind.default_misfire_grace_s == 24 * 3600
    [seed] = kind.defaults
    assert seed.key == "arxiv_delta_daily"
    assert seed.trigger == {"type": "daily", "time": "05:00", "timezone": "Asia/Shanghai"}
    assert seed.params == {"html_concurrency": 3, "no_html_retry_days": 3, "limit": 0,
                           "source": "volume", "receive_deadline_hours": 6.0, "volume": "/cfs-arxiv"}
    for bad in ({"html_concurrency": 0}, {"html_concurrency": 7}, {"no_html_retry_days": 15}, {"limit": -1}):
        with pytest.raises(ValueError):
            arxiv_schedule.ArxivDeltaParams(**bad)


def test_fire_while_a_run_is_going_is_a_skipped_tick(tmp_path, store, monkeypatch):
    from app.services.pipeline import service as pipeline_service
    from app.services.scheduler.kinds import FireResult

    runner = PipelineRunner(store=store)
    monkeypatch.setattr(pipeline_service, "get_pipeline_runner", lambda: runner)
    running = store.create_task(task_type=INGEST_PIPELINE, params={"venue": "arXiv", "year": 0})
    store.set_task_status(running, "running")

    result = arxiv_schedule.fire_arxiv_delta(arxiv_schedule.ArxivDeltaParams(), {"schedule_id": 1})

    assert result == FireResult("skipped", running, f"already running #{running}")


def test_status_metrics_read_the_harvest_and_sql(tmp_path, monkeypatch):
    import core.db

    factory = make_session_factory(tmp_path / "app.db")
    harvest = HarvestStore(tmp_path / "harvest" / "arxiv_cs.sqlite3")
    harvest.upsert([raw("2601.00001"), raw("2601.00002")])
    harvest.mark("2601.00002", "ok", html_version="v1")
    harvest.set_state("oai_from", "2026-04-03")
    harvest.close()
    with factory() as db:
        seed_sql(db, "2601.00002", "v1", withdrawn=True)
        seed_sql(db, "2601.00003", "v1")
    monkeypatch.setenv("ARXIV_HARVEST_ROOT", str(tmp_path / "harvest"))
    monkeypatch.setattr(core.db, "get_db_session", factory)

    metrics = {m.name: m for m in arxiv_schedule.arxiv_delta_metrics()}

    assert metrics["OAI 水位"].value == "2026-04-03"
    assert (metrics["待下载"].value, metrics["待下载"].unit) == (1, "篇")
    assert metrics["语料"].value == 2 and metrics["撤稿"].value == 1
