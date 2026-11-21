"""Markdown supplied from outside the pipeline (another machine, a worker) is
enough for a slice to be preindexed: no manual manifest sync, no start_stage.

Real everything: a real manifest file, real Markdown files, the real orchestrator
over the production extract/preindex stage builders, the real preindex script
writing a real SQLite database in a temporary DATA_DIR, the real sync script as a
subprocess. Nothing is faked and no production path is touched.
"""
from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from app.services.pipeline import stages
from app.services.pipeline.arxiv_stages import ArxivStageContext
from app.services.pipeline.context import StageContext
from app.services.pipeline.orchestrator import PipelineOrchestrator
from app.services.pipeline.store import TaskStore
from core.db import models  # noqa: F401 - register every ORM table
from core.db.database import Base
from core.db.models import Chunk as DBChunk, Paper as DBPaper

REPO = Path(__file__).resolve().parents[1]
CORPUS_NAME = "ccf_a_main_conference_corpus"

MARKDOWN = """# {title}

## Abstract

{title} studies how a scheduler places jobs on accelerators under memory pressure.
We describe the design, an implementation and an evaluation on production traces.

## 1 Introduction

Resource schedulers decide which job runs where. Placement quality is decided by
how well the scheduler predicts memory growth during a run, and by what it does
when the prediction is wrong. This section motivates the problem with traces from
a shared cluster and states the contributions of the paper in detail.

## 2 Design

The scheduler keeps a profile per job class. When a job arrives it is matched to
the closest profile, placed on the device with the most headroom, and monitored.
If measured memory exceeds the profile the job is migrated rather than killed.

## 3 Evaluation

We replay two weeks of traces. Median completion time drops by eleven percent and
the number of out-of-memory kills drops to zero in every replayed week.
"""


def _task_store(db_path: Path) -> TaskStore:
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    local_session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_factory():
        session = local_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return TaskStore(session_factory)


def _record(paper_id: str, title: str) -> dict:
    return {
        "paper_id": paper_id, "title": title, "authors": ["A. Author"],
        "conference": "CVPR", "year": 2026, "pdf_url": "https://example.org/p.pdf",
        "paper_page_url": "https://example.org/p", "source_name": "test",
        "markdown_path": None, "pdf_path": None,
    }


@pytest.fixture
def supplied_slice(tmp_path, monkeypatch):
    """A (CVPR, 2026) slice whose Markdown arrived from outside: files exist under
    the corpus markdown tree, the manifest does not know about them, no PDF exists."""
    data_dir = tmp_path / "data"
    (data_dir / "db").mkdir(parents=True)
    monkeypatch.setenv("DATA_DIR", str(data_dir))  # inherited by every stage subprocess
    corpus = data_dir / "papers" / CORPUS_NAME
    (corpus / "manifests").mkdir(parents=True)
    papers = {"cvpr-2026-aa": "Predictive Placement", "cvpr-2026-bb": "Migrating Instead Of Killing"}
    (corpus / "manifests" / "paper_manifest.json").write_text(
        json.dumps([_record(pid, title) for pid, title in papers.items()]), encoding="utf-8")
    md_dir = corpus / "markdown" / "cvpr" / "2026"
    md_dir.mkdir(parents=True)
    for pid, title in papers.items():
        (md_dir / f"{pid}.md").write_text(MARKDOWN.format(title=title), encoding="utf-8")

    env_py = [sys.executable, "-c",
              "from core.db.database import engine, Base; import core.db.models; "
              "Base.metadata.create_all(engine)"]
    done = subprocess.run(env_py, cwd=REPO, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]

    store = _task_store(tmp_path / "tasks.db")
    task_id = store.create_task(task_type="ingest_pipeline", params={"venue": "CVPR", "year": 2026})
    ctx = StageContext(venue="CVPR", year=2026, store=store, task_id=task_id, corpus_root=corpus,
                       execution_mode="local")
    data_engine = create_engine(f"sqlite:///{data_dir / 'db' / 'retrieval.db'}",
                                connect_args={"check_same_thread": False})
    data_session = sessionmaker(bind=data_engine, expire_on_commit=False)

    @contextmanager
    def data_factory():
        session = data_session()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    yield ctx, store, data_factory, papers
    data_engine.dispose()


def test_extract_probe_falls_back_to_the_manifest_slice_when_no_pdf_is_present(supplied_slice):
    ctx, _store, _factory, papers = supplied_slice
    assert ctx.pdf_count() == 0
    progress = stages.extract_local_probe(ctx)
    assert (progress.done, progress.total) == (len(papers), len(papers))
    assert progress.is_complete  # externally supplied Markdown satisfies the stage

    (ctx.local_md_dir / "cvpr-2026-bb.md").unlink()
    partial = stages.extract_local_probe(ctx)
    assert (partial.done, partial.total) == (1, 2)
    assert not partial.is_complete  # still missing one: no PDFs to convert, so run() fails loudly


def test_extract_probe_still_counts_pdfs_when_they_exist(supplied_slice):
    ctx, _store, _factory, _papers = supplied_slice
    ctx.pdf_dir.mkdir(parents=True)
    for name in ("one", "two", "three"):
        (ctx.pdf_dir / f"{name}.pdf").write_bytes(b"%PDF-1.4")
    progress = stages.extract_local_probe(ctx)
    assert (progress.done, progress.total) == (2, 3)  # the PDFs on disk are the denominator


def test_preindex_syncs_manifest_paths_only_where_the_local_tree_is_the_source(supplied_slice, tmp_path):
    ctx, store, _factory, _papers = supplied_slice
    assert stages.preindex_syncs_manifest_paths(ctx) is True
    ctx.execution_mode = "host"
    assert stages.preindex_syncs_manifest_paths(ctx) is True
    ctx.execution_mode = "remote"
    assert stages.preindex_syncs_manifest_paths(ctx) is False  # md lives on the SMB share
    arxiv = ArxivStageContext(venue="arXiv", year=202609, store=store, task_id=ctx.task_id,
                              corpus_root=tmp_path / "arxiv-corpus", harvest_root=tmp_path / "harvest")
    assert stages.preindex_syncs_manifest_paths(arxiv) is False  # its convert step writes the manifest


def test_externally_supplied_markdown_is_preindexed_without_a_manual_sync(supplied_slice):
    ctx, store, data_factory, papers = supplied_slice
    manifest = json.loads(ctx.manifest_path.read_text(encoding="utf-8"))
    assert all(r["markdown_path"] is None for r in manifest)

    specs = [stages.make_extract_stage_local(ctx),
             stages.make_preindex_stage(ctx, session_factory=data_factory)]
    final = PipelineOrchestrator(ctx.task_id, specs, ctx, store).run()

    assert final == "completed"
    extract = store.get_subtask(ctx.task_id, "extract")
    assert extract["status"] == "completed" and extract["message"] == "already satisfied"
    preindex = store.get_subtask(ctx.task_id, "preindex")
    assert preindex["status"] == "completed"

    with data_factory() as db:
        assert db.query(func.count(DBPaper.paper_id)).scalar() == len(papers)
        assert {r[0] for r in db.query(DBPaper.paper_id)} == set(papers)
        assert db.query(func.count(DBChunk.id)).scalar() > len(papers)
    after = json.loads(ctx.manifest_path.read_text(encoding="utf-8"))
    assert all(r["markdown_path"] and r["markdown_path"].endswith(f"{r['paper_id']}.md") for r in after)
    # No chunk-id range is kept; embed dumps the documents of the papers the report names.
    assert ctx.state_get("chunk_id_before") is None and ctx.state_get("chunk_id_after") is None
    assert stages.task_doc_ids(ctx, session_factory=data_factory) is not None
    assert len(stages.task_doc_ids(ctx, session_factory=data_factory)) == len(papers)
    assert not stages.nothing_to_embed(ctx)
