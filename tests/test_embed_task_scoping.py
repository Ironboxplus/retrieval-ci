"""Dump, embed and ingest files are task-scoped: data/remote_embed/tasks/<task_id>/{chunks,vectors}.

Real processes and files: the real dump_chunks.py reads a real SQLite database, the
real local embed stage runs it and then embeds in-process through the real shard loop
(``embed_shards``), and the real local_ingest.py (--dry-run, so Qdrant is only read) selects
the shards. The one stand-in is the model: the stage takes its ``encode`` as a parameter
(production passes the backend's shared embedder) and the tests pass a recording function
that returns zero vectors. A shard that is not valid gzip crashes the loop, exactly as it
crashes the real one.
"""
from __future__ import annotations

import gzip
import shutil
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager

from app.services.pipeline import stages as st
from app.services.pipeline.context import StageContext
from app.services.pipeline.store import TaskStore
from core.db import models  # noqa: F401
from core.db.database import Base

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def store() -> TaskStore:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
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

    return TaskStore(factory)


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """A project root: a SQLite database with 30 chunks of one paper, the real dump script,
    and leftovers of earlier runs in the OLD shared directories."""
    scripts = tmp_path / "scripts" / "remote_embed"
    scripts.mkdir(parents=True)
    shutil.copy2(REPO / "scripts" / "remote_embed" / "dump_chunks.py", scripts)
    (tmp_path / "data" / "db").mkdir(parents=True)
    db = sqlite3.connect(tmp_path / "data" / "db" / "retrieval.db")
    db.executescript("""
        CREATE TABLE documents (id INTEGER PRIMARY KEY, title TEXT, source TEXT, source_type TEXT,
                                metadata_json TEXT);
        CREATE TABLE papers (paper_id TEXT, doc_id INTEGER, venue TEXT, year INTEGER, venue_tier TEXT,
                             track_type TEXT, arxiv_id TEXT, doi TEXT, citation_count INTEGER,
                             withdrawn INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE chunks (id INTEGER PRIMARY KEY, doc_id INTEGER, content TEXT, level INTEGER,
                             start_idx INTEGER, end_idx INTEGER, metadata_json TEXT);
    """)
    db.execute("INSERT INTO documents VALUES (1, 't', 's', 'paper', '{}')")
    db.execute("INSERT INTO papers VALUES ('arxiv-2601.00001', 1, 'arXiv', 2026, 'preprint', 'preprint', "
               "'2601.00001', NULL, 0, 0)")
    db.executemany("INSERT INTO chunks VALUES (?, 1, ?, 2, ?, ?, '{}')",
                   [(i, f"sparse mixture of experts, chunk {i}", i * 10, i * 10 + 9) for i in range(1, 31)])
    db.commit()
    db.close()

    shared = tmp_path / "data" / "remote_embed"
    (shared / "chunks").mkdir(parents=True)
    (shared / "vectors").mkdir(parents=True)
    whole = gzip.compress(b'{"chunk_id_str": "x"}\n' * 5000)
    (shared / "chunks" / "chunks_0060.jsonl.gz").write_bytes(whole[: len(whole) // 2])  # truncated gzip
    (shared / "chunks" / "chunks_0059.jsonl.gz").write_bytes(whole)
    (shared / "vectors" / "vectors_0059.parquet").write_bytes(b"stale vectors of a replaced paper")
    return tmp_path


def leftovers(project: Path) -> dict[str, bytes]:
    shared = project / "data" / "remote_embed"
    return {str(p.relative_to(shared)): p.read_bytes()
            for sub in ("chunks", "vectors") for p in (shared / sub).iterdir()}


def make_ctx(store: TaskStore, project: Path) -> StageContext:
    task_id = store.create_task(task_type="ingest_pipeline")
    ctx = StageContext(venue="arXiv", year=20260900, store=store, task_id=task_id, project_root=project,
                       corpus_root=project / "corpus", execution_mode="local")
    ctx.state_set(chunk_id_before=10, chunk_id_after=30)  # this task's chunks: 11..30
    return ctx


class Recorder:
    """The stage's ``encode``: zero vectors, and a record of every text it was asked for."""

    def __init__(self):
        self.texts: list[str] = []

    def __call__(self, texts):
        self.texts.extend(texts)
        return np.zeros((len(texts), 1024), dtype=np.float32), [{1: 0.5} for _ in texts]


def run_embed(ctx: StageContext) -> tuple[list[str], Recorder]:
    """The local embed stage over this task's shards: its log lines and what it encoded."""
    lines: list[str] = []
    recorder = Recorder()
    st.make_embed_stage_local(ctx, encode=recorder).run(ctx, lambda m, level="info": lines.append(m))
    return lines, recorder


def chunk_contents(first: int, last: int) -> list[str]:
    return [f"sparse mixture of experts, chunk {i}" for i in range(first, last + 1)]


def test_layout_and_the_old_shared_directories_are_never_read(store, project):
    ctx = make_ctx(store, project)
    before = leftovers(project)
    task_dir = project / "data" / "remote_embed" / "tasks" / str(ctx.task_id)
    assert st._embed_task_dir(ctx) == task_dir
    assert st._embed_chunks_dir(ctx) == task_dir / "chunks" and st._embed_vectors_dir(ctx) == task_dir / "vectors"

    lines, recorder = run_embed(ctx)

    # The loop saw this task's shard and nothing else; the truncated leftover would have crashed it.
    assert "rank=0/1  shards = 1" in lines and sorted(recorder.texts) == sorted(chunk_contents(11, 30))
    assert sorted(p.name for p in (task_dir / "chunks").iterdir()) == ["chunks_0000.jsonl.gz", "done.txt"]
    assert sorted(p.name for p in (task_dir / "vectors").iterdir()) == ["done.rank0.txt", "vectors_0000.parquet"]
    progress = st.embed_local_probe(ctx)
    assert (progress.done, progress.total) == (1, 1) and st.embed_gate(ctx).ok
    assert (ctx.state_get("shard_start"), ctx.state_get("shard_end")) == (0, 1)
    assert leftovers(project) == before  # untouched: nothing wrote to, or deleted from, the shared dirs


def test_ingest_selects_only_the_task_vectors(store, project):
    ctx = make_ctx(store, project)
    run_embed(ctx)
    cmd = st.build_ingest_cmd(ctx)
    assert cmd[cmd.index("--vectors") + 1] == str(st._embed_vectors_dir(ctx))

    # The real local_ingest.py reads the stale shared parquet only if pointed at it; --dry-run
    # validates the task's own parquet and writes nothing to Qdrant.
    done = subprocess.run([*cmd, "--dry-run"], cwd=REPO, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout[-800:] + done.stderr[-800:]
    assert "INGEST vectors_0000.parquet" in done.stdout and "vectors_0059" not in done.stdout
    assert "shards = 1" in done.stdout


def test_a_redriven_task_resumes_from_its_own_directory(store, project):
    ctx = make_ctx(store, project)
    run_embed(ctx)
    first = (st._embed_vectors_dir(ctx) / "vectors_0000.parquet").read_bytes()

    # A new run of the same task (resume): the durable dump is reused and the loop skips the
    # shard whose parquet already exists.
    lines, recorder = run_embed(ctx)

    assert any("reusing dumped shards [0, 1)" in line for line in lines)
    assert not any("dump_chunks.py" in line for line in lines)
    assert "SKIP chunks_0000.jsonl.gz (done)" in lines and recorder.texts == []
    assert (st._embed_vectors_dir(ctx) / "vectors_0000.parquet").read_bytes() == first

    # Another task starts empty: its anchor is 0 and it sees none of the first task's shards.
    other = make_ctx(store, project)
    other_lines, other_recorder = run_embed(other)
    assert other.state_get("shard_start") == 0 and "rank=0/1  shards = 1" in other_lines
    assert len(other_recorder.texts) == 20
    assert st._embed_task_dir(other) != st._embed_task_dir(ctx)


def test_cleanup_removes_the_task_directory_only_after_ingest_finished(store, project):
    ctx = make_ctx(store, project)
    run_embed(ctx)
    before = leftovers(project)
    task_dir = st._embed_task_dir(ctx)
    lines: list[str] = []
    sink = lambda m, level="info": lines.append(m)  # noqa: E731

    # Ingest has not finished: the files stay for a re-drive.
    assert st.remove_embed_task_dir(ctx, sink) is False
    assert (task_dir / "vectors" / "vectors_0000.parquet").exists() and "keeping" in lines[-1]

    # local_ingest.py records each finished shard by file name; now the task is complete.
    (task_dir / "vectors" / "ingest_done.txt").write_text("vectors_0000.parquet\n", encoding="utf-8")
    assert st.ingest_gate(ctx).ok
    assert st.remove_embed_task_dir(ctx, sink) is True
    assert not task_dir.exists() and (task_dir.parent).exists()
    assert leftovers(project) == before                # the shared directories are not this task's to delete
    assert st.remove_embed_task_dir(ctx, sink) is False  # nothing left: a second cleanup is a no-op


def test_a_run_with_nothing_to_embed_has_no_directory_and_cleanup_has_nothing_to_remove(store, project):
    ctx = make_ctx(store, project)
    ctx.state_set(chunk_id_before=30, chunk_id_after=30)
    run_embed(ctx)  # the stage short-circuits
    assert not st._embed_task_dir(ctx).exists()
    assert st.remove_embed_task_dir(ctx, lambda *a, **k: None) is False


def test_the_retired_remote_mode_keeps_the_shared_directories_ship_sh_hard_codes(store, project):
    task_id = store.create_task(task_type="ingest_pipeline")
    ctx = StageContext(venue="CVPR", year=2026, store=store, task_id=task_id, project_root=project,
                       execution_mode="remote")
    assert st._embed_chunks_dir(ctx) == project / "data" / "remote_embed" / "chunks"
    assert st._embed_vectors_dir(ctx) == project / "data" / "remote_embed" / "vectors"
    assert "data/remote_embed/chunks" in (REPO / "scripts" / "remote_embed" / "ship.sh").read_text(encoding="utf-8", errors="replace")
