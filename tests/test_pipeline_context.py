"""TDD: StageContext — venue/year + resolved paths + manifest helpers + state proxy.

The orchestrator forwards `ctx` to stage callbacks OPAQUELY, so StageContext is
just a plain carrier object. It must be fully GENERIC for ANY (venue, year): no
conference is special-cased. These tests build a temp corpus dir + temp manifest
and assert slice counts / path strings / pdf+md counts / state round-trips.

State proxy uses a real TaskStore on a private in-memory SQLite (StaticPool) just
like the orchestrator test, so the persisted-state path is exercised end-to-end.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.database import Base
from core.db import models  # noqa: F401  — register all tables on Base.metadata
from app.services.pipeline.store import TaskStore
from app.services.pipeline.context import StageContext


# --------------------------------------------------------------------------- db
@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def factory():
        s = Local()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    return factory


@pytest.fixture
def store(session_factory):
    return TaskStore(session_factory)


# ---------------------------------------------------------------------- corpus
def _manifest_record(conf, year, *, paper_id, markdown_path=None):
    return {
        "paper_id": paper_id,
        "title": f"{conf} {year} paper {paper_id}",
        "authors": [],
        "conference": conf,
        "year": year,
        "pdf_url": f"https://x/{paper_id}.pdf",
        "paper_page_url": f"https://x/{paper_id}",
        "source_name": "test",
        "markdown_path": markdown_path,
    }


@pytest.fixture
def corpus(tmp_path):
    """A temp corpus root with manifest holding 3 CVPR-2026 + 2 AAAI-2026 records."""
    corpus_root = tmp_path / "corpus"
    manifests = corpus_root / "manifests"
    manifests.mkdir(parents=True)
    records = [
        _manifest_record("CVPR", 2026, paper_id="cvpr-2026-a", markdown_path="/some/a.md"),
        _manifest_record("CVPR", 2026, paper_id="cvpr-2026-b", markdown_path=None),
        _manifest_record("CVPR", 2026, paper_id="cvpr-2026-c", markdown_path="/some/c.md"),
        _manifest_record("AAAI", 2026, paper_id="aaai-2026-x", markdown_path=None),
        _manifest_record("AAAI", 2026, paper_id="aaai-2026-y", markdown_path="/some/y.md"),
    ]
    (manifests / "paper_manifest.json").write_text(json.dumps(records), encoding="utf-8")
    return corpus_root


def _ctx(corpus, store, *, venue="CVPR", year=2026, task_id=1):
    return StageContext(
        venue=venue,
        year=year,
        store=store,
        task_id=task_id,
        corpus_root=corpus,
    )


# ------------------------------------------------------------------ properties
def test_computed_properties_are_generic(corpus, store):
    ctx = _ctx(corpus, store, venue="CVPR", year=2026)
    assert ctx.conf_lower == "cvpr"
    assert ctx.manifest_path == corpus / "manifests" / "paper_manifest.json"
    assert ctx.pdf_dir == corpus / "pdf" / "cvpr" / "2026"
    assert ctx.kss_pdf_dir == "/mnt/lsb/retrival/papers/pdf/cvpr/2026"
    assert ctx.kss_md_dir == "/mnt/lsb/retrival/papers/markdown/cvpr/2026"
    assert ctx.smb_md_dir == "//10.100.100.9/lsb/retrival/papers/markdown/cvpr/2026"


def test_properties_work_for_any_venue(corpus, store):
    ctx = _ctx(corpus, store, venue="AAAI", year=2026)
    assert ctx.conf_lower == "aaai"
    assert ctx.pdf_dir == corpus / "pdf" / "aaai" / "2026"
    assert ctx.kss_md_dir == "/mnt/lsb/retrival/papers/markdown/aaai/2026"


def test_defaults_for_concurrency_and_targets(corpus, store):
    ctx = _ctx(corpus, store)
    assert ctx.concurrency == 8
    assert ctx.world_size == 8
    assert ctx.ssh_target == "kss"


# -------------------------------------------------------------------- manifest
def test_manifest_slice_filters_by_venue_and_year(corpus, store):
    ctx = _ctx(corpus, store, venue="CVPR", year=2026)
    slice_ = ctx.manifest_slice()
    assert len(slice_) == 3
    assert {r["paper_id"] for r in slice_} == {"cvpr-2026-a", "cvpr-2026-b", "cvpr-2026-c"}

    ctx2 = _ctx(corpus, store, venue="AAAI", year=2026)
    assert ctx2.manifest_slice_total() == 2


def test_manifest_slice_extracted_counts_markdown_path(corpus, store):
    ctx = _ctx(corpus, store, venue="CVPR", year=2026)
    # 2 of 3 CVPR records have markdown_path set
    assert ctx.manifest_slice_extracted() == 2

    ctx2 = _ctx(corpus, store, venue="AAAI", year=2026)
    assert ctx2.manifest_slice_extracted() == 1


def test_manifest_path_override(corpus, store, tmp_path):
    other = tmp_path / "other_manifest.json"
    other.write_text(
        json.dumps([_manifest_record("CVPR", 2026, paper_id="only-one")]),
        encoding="utf-8",
    )
    ctx = StageContext(
        venue="CVPR", year=2026, store=store, task_id=1,
        corpus_root=corpus, manifest_path_override=other,
    )
    assert ctx.manifest_slice_total() == 1
    assert ctx.manifest_slice()[0]["paper_id"] == "only-one"


# ------------------------------------------------------------------ fs counts
def test_pdf_count_globs_pdf_dir(corpus, store):
    ctx = _ctx(corpus, store, venue="CVPR", year=2026)
    ctx.pdf_dir.mkdir(parents=True)
    (ctx.pdf_dir / "a.pdf").write_bytes(b"%PDF-1.4")
    (ctx.pdf_dir / "b.pdf").write_bytes(b"%PDF-1.4")
    (ctx.pdf_dir / "notes.txt").write_text("ignore me")
    assert ctx.pdf_count() == 2


def test_pdf_count_missing_dir_is_zero(corpus, store):
    ctx = _ctx(corpus, store, venue="CVPR", year=2026)
    assert ctx.pdf_count() == 0


def test_md_count_globs_smb_md_dir(corpus, store, tmp_path, monkeypatch):
    # Point the SMB md dir at a real temp dir by overriding arc_smb_unc to a
    # local path (the property only string-joins, md_count globs Path(smb_md_dir)).
    smb_root = tmp_path / "smb"
    md_dir = smb_root / "papers" / "markdown" / "cvpr" / "2026"
    md_dir.mkdir(parents=True)
    (md_dir / "one.md").write_text("# one")
    (md_dir / "two.md").write_text("# two")
    (md_dir / "skip.json").write_text("{}")
    ctx = StageContext(
        venue="CVPR", year=2026, store=store, task_id=1,
        corpus_root=corpus, arc_smb_unc=str(smb_root),
    )
    assert ctx.md_count() == 2


def test_md_count_missing_dir_tolerated(corpus, store, tmp_path):
    ctx = StageContext(
        venue="CVPR", year=2026, store=store, task_id=1,
        corpus_root=corpus, arc_smb_unc=str(tmp_path / "nonexistent"),
    )
    assert ctx.md_count() == 0


# ----------------------------------------------- W106 local (self-hosted) paths
def test_execution_mode_defaults_remote(corpus, store):
    ctx = _ctx(corpus, store)
    assert ctx.execution_mode == "remote"


def test_local_md_dir_mirrors_corpus_layout(corpus, store):
    ctx = _ctx(corpus, store, venue="CVPR", year=2026)
    assert ctx.local_md_dir == corpus / "markdown" / "cvpr" / "2026"


def test_local_vectors_dir_is_remote_embed_vectors(corpus, store, tmp_path):
    ctx = StageContext(
        venue="CVPR", year=2026, store=store, task_id=1,
        corpus_root=corpus, project_root=tmp_path / "proj",
    )
    assert ctx.local_vectors_dir == tmp_path / "proj" / "data" / "remote_embed" / "vectors"


def test_md_count_local_globs_local_md_dir(corpus, store):
    ctx = _ctx(corpus, store, venue="CVPR", year=2026)
    ctx.local_md_dir.mkdir(parents=True)
    (ctx.local_md_dir / "one.md").write_text("# one")
    (ctx.local_md_dir / "two.md").write_text("# two")
    (ctx.local_md_dir / "skip.json").write_text("{}")
    assert ctx.md_count_local() == 2


def test_md_count_local_missing_dir_is_zero(corpus, store):
    ctx = _ctx(corpus, store, venue="CVPR", year=2026)
    assert ctx.md_count_local() == 0


# ---------------------------------------------------------------- state proxy
def test_state_proxy_round_trips(corpus, store):
    task_id = store.create_task(task_type="ingest_pipeline")
    ctx = _ctx(corpus, store, task_id=task_id)
    assert ctx.state_get("chunk_id_before") is None
    assert ctx.state_get("chunk_id_before", 0) == 0

    ctx.state_set(chunk_id_before=12345, gpu_ids="0,1,2")
    assert ctx.state_get("chunk_id_before") == 12345
    assert ctx.state_get("gpu_ids") == "0,1,2"

    # second write merges, does not clobber
    ctx.state_set(shard_start=7)
    assert ctx.state_get("chunk_id_before") == 12345
    assert ctx.state_get("shard_start") == 7
