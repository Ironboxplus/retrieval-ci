"""TDD: the 2 upload stage builders (extract / index).

Mirrors test_pipeline_stages.py but for a single uploaded document. Each stage is
decomposed into PURE, offline-testable pieces:
  - build_*_cmd(ctx)   → exact local-worker command, asserted arg-by-arg
  - *_probe(ctx)       → Progress from md file / SQL chunks / injected qdrant count
  - *_gate(ctx)        → GateResult verdict
  - make_upload_*_stage(ctx) → StageSpec assembling them

NO real extraction / embedding / Qdrant runs here: the thin run() wiring is
checked by monkeypatching run_local and asserting it is called with the BUILT
command. LOCAL only (W62) — every stage shells to a local worker via run_local.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.database import Base
from core.db import models  # noqa: F401
from core.db.models import Chunk as DBChunk
from app.services.pipeline.spec import StageSpec, Progress
from app.services.pipeline.store import TaskStore
from app.services.pipeline.upload_context import UploadStageContext
from app.services.pipeline import upload_stages as ust


# --------------------------------------------------------------------------- db
@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
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


@pytest.fixture
def ctx(tmp_path, store):
    up = tmp_path / "uploads"
    up.mkdir(parents=True, exist_ok=True)
    (up / "u7.pdf").write_bytes(b"%PDF-1.4 fake")
    task_id = store.create_task(task_type="upload_pipeline")
    return UploadStageContext(
        doc_id=7, file_path=str(up / "u7.pdf"), store=store, task_id=task_id,
        project_root=tmp_path / "proj", upload_root=up,
    )


def _add_chunks(session_factory, doc_id, n):
    with session_factory() as s:
        for i in range(n):
            s.add(DBChunk(doc_id=doc_id, chunk_index=i, level=1, content=f"c{i}"))


# ===================================================================== extract
def test_extract_cmd_is_local_worker(ctx):
    assert ust.build_extract_cmd(ctx) == [
        sys.executable, "scripts/upload/extract_one.py",
        "--doc-id", "7", "--input", str(ctx.file_path), "--output", str(ctx.md_path),
    ]


def test_extract_probe_tracks_md(ctx):
    assert ust.extract_probe(ctx) == Progress(done=0, total=1)
    ctx.md_dir.mkdir(parents=True, exist_ok=True)
    ctx.md_path.write_text("# title\n\nbody text", encoding="utf-8")
    pr = ust.extract_probe(ctx)
    assert pr.done == 1 and pr.total == 1 and pr.is_complete


def test_extract_gate_requires_nonempty_md(ctx):
    assert not ust.extract_gate(ctx).ok  # no md yet
    ctx.md_dir.mkdir(parents=True, exist_ok=True)
    ctx.md_path.write_text("# title\n\nreal extracted body", encoding="utf-8")
    assert ust.extract_gate(ctx).ok


def test_extract_run_invokes_local_worker(ctx, monkeypatch):
    seen = {}
    def fake_run_local(cmd, cwd, sink, **kw):
        seen["cmd"], seen["cwd"] = cmd, cwd
        return 0
    monkeypatch.setattr(ust, "run_local", fake_run_local)
    spec = ust.make_upload_extract_stage(ctx)
    spec.run(ctx, lambda *a, **k: None)
    assert seen["cmd"] == ust.build_extract_cmd(ctx)
    assert seen["cwd"] == ctx.project_root


# ======================================================================= index
def test_index_cmd_is_local_worker(ctx):
    assert ust.build_index_cmd(ctx) == [
        sys.executable, "scripts/upload/index_one.py", "--doc-id", "7",
    ]


def test_index_probe_compares_points_to_chunks(ctx, session_factory):
    _add_chunks(session_factory, 7, 5)
    # 3 of 5 chunks upserted to Qdrant so far
    pr = ust.index_probe(ctx, point_counter=lambda doc_id: 3, session_factory=session_factory)
    assert pr.done == 3 and pr.total == 5 and not pr.is_complete
    pr2 = ust.index_probe(ctx, point_counter=lambda doc_id: 5, session_factory=session_factory)
    assert pr2.is_complete


def test_index_gate(ctx, session_factory):
    _add_chunks(session_factory, 7, 2)
    assert not ust.index_gate(ctx, point_counter=lambda d: 0, session_factory=session_factory).ok
    assert ust.index_gate(ctx, point_counter=lambda d: 2, session_factory=session_factory).ok


# ================================================================ stage specs
def test_stage_specs_keys_and_titles(ctx):
    extract = ust.make_upload_extract_stage(ctx)
    index = ust.make_upload_index_stage(ctx)
    assert isinstance(extract, StageSpec)
    assert (extract.key, index.key) == ("extract", "index")
    # doc-scoped artifacts → satisfied-skip is SAFE (unlike venue's shared dirs)
    assert extract.skip_when_satisfied and index.skip_when_satisfied
