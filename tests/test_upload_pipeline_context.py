"""TDD: UploadStageContext — doc-keyed carrier for the upload ingest pipeline.

Mirrors StageContext (test_pipeline_context.py) but for a single uploaded
document instead of a (venue, year) batch. The orchestrator forwards `ctx`
opaquely, so this is a plain data carrier: doc_id + file_path + resolved md
path + artifact probes (md exists / SQL chunk count) + the same state proxy.

Local-only by design (W62: the HTTP upload path has no ssh_target), so there
are NO kss_/smb_ remote paths here — extract/preindex/index all run locally.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.database import Base
from core.db import models  # noqa: F401 — register all tables on Base.metadata
from app.services.pipeline.store import TaskStore
from app.services.pipeline.upload_context import UploadStageContext


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


def _ctx(tmp_path, store, *, doc_id=42, name="abc123.pdf", task_id=1):
    upload_root = tmp_path / "uploads"
    upload_root.mkdir(parents=True, exist_ok=True)
    return UploadStageContext(
        doc_id=doc_id,
        file_path=str(upload_root / name),
        store=store,
        task_id=task_id,
        project_root=tmp_path,
        upload_root=upload_root,
    )


# ------------------------------------------------------------------ properties
def test_computed_paths(tmp_path, store):
    ctx = _ctx(tmp_path, store, name="abc123.pdf")
    assert ctx.ext == ".pdf"
    assert ctx.file_stem == "abc123"
    assert ctx.md_dir == tmp_path / "uploads" / "md"
    assert ctx.md_path == tmp_path / "uploads" / "md" / "abc123.md"


def test_ext_is_lowercased(tmp_path, store):
    ctx = _ctx(tmp_path, store, name="Paper.PDF")
    assert ctx.ext == ".pdf"


def test_md_exists_reflects_artifact(tmp_path, store):
    ctx = _ctx(tmp_path, store)
    assert ctx.md_exists() is False
    ctx.md_dir.mkdir(parents=True, exist_ok=True)
    ctx.md_path.write_text("# extracted\n\nbody", encoding="utf-8")
    assert ctx.md_exists() is True


# -------------------------------------------------------------------- sql chunks
def test_chunk_count_filters_by_doc_id(tmp_path, store, session_factory):
    from core.db.models import Chunk

    # two chunks for our doc (42), one for a different doc (99)
    with session_factory() as s:
        s.add(Chunk(doc_id=42, chunk_index=0, level=1, content="a"))
        s.add(Chunk(doc_id=42, chunk_index=1, level=1, content="b"))
        s.add(Chunk(doc_id=99, chunk_index=0, level=1, content="c"))

    ctx = _ctx(tmp_path, store, doc_id=42)
    assert ctx.chunk_count(session_factory=session_factory) == 2


# ---------------------------------------------------------------- state proxy
def test_state_proxy_round_trips(tmp_path, store):
    task_id = store.create_task(task_type="upload_pipeline")
    ctx = _ctx(tmp_path, store, task_id=task_id)
    assert ctx.state_get("chunk_id_before") is None
    assert ctx.state_get("chunk_id_before", 0) == 0

    ctx.state_set(chunk_id_before=555, content_chars=1200)
    assert ctx.state_get("chunk_id_before") == 555
    assert ctx.state_get("content_chars") == 1200

    # merge, not clobber
    ctx.state_set(extracted=True)
    assert ctx.state_get("chunk_id_before") == 555
    assert ctx.state_get("extracted") is True
