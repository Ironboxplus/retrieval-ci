"""Embed dumps the task's own documents, not a chunk-id range (PLAN-2026-11-15).

The chunks table is shared by every writer. A task that preindexed days ago and is
retried used to dump (chunk_id_before, chunk_id_after], which holds whatever other tasks
wrote in between (task 1305: a 8873-token chunk of another task's document). The task's
own documents are named by its preindex report.

Real files, real SQLite, real subprocesses (the real dump_chunks.py, a stand-in preindex
script run through the real run_local, the real preindex script's batch loop). The one
injected fault is a BaseException at a chosen paper, which is how the test simulates a
killed process.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import shutil
import sqlite3
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.services.pipeline import stages as st
from app.services.pipeline.context import StageContext
from app.services.pipeline.store import TaskStore
from core.db import models  # noqa: F401
from core.db.database import Base
from tests.sqlite_task_db import make_session_factory

REPO = Path(__file__).resolve().parents[1]
DUMP_SCRIPT = REPO / "scripts" / "remote_embed" / "dump_chunks.py"


# ------------------------------------------------------------------- fixtures
@pytest.fixture()
def store() -> TaskStore:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def factory():
        s = local()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    return TaskStore(factory)


def build_chunks_db(path: Path) -> None:
    """Interleaved chunk ids: doc 1 (paper pA) 1-3, doc 2 (another task's paper pX) 4-6,
    doc 3 (paper pB) 7-8. Paper pNull has no document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE documents (id INTEGER PRIMARY KEY, title TEXT, source TEXT, source_type TEXT,
                                metadata_json TEXT);
        CREATE TABLE papers (paper_id TEXT, doc_id INTEGER, venue TEXT, year INTEGER, venue_tier TEXT,
                             track_type TEXT, arxiv_id TEXT, doi TEXT, citation_count INTEGER,
                             withdrawn INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE chunks (id INTEGER PRIMARY KEY, doc_id INTEGER, content TEXT, level INTEGER,
                             start_idx INTEGER, end_idx INTEGER, metadata_json TEXT);
    """)
    layout = [(1, "pA", [1, 2, 3]), (2, "pX", [4, 5, 6]), (3, "pB", [7, 8])]
    for doc_id, paper_id, chunk_ids in layout:
        db.execute("INSERT INTO documents VALUES (?, 't', 's', 'paper', '{}')", (doc_id,))
        db.execute("INSERT INTO papers VALUES (?, ?, 'VLDB', 2023, 'A', 'main', NULL, NULL, 0, 0)",
                   (paper_id, doc_id))
        db.executemany("INSERT INTO chunks VALUES (?, ?, ?, 2, ?, ?, '{}')",
                       [(i, doc_id, f"chunk {i} of doc {doc_id}", i * 10, i * 10 + 9) for i in chunk_ids])
    db.execute("INSERT INTO papers VALUES ('pNull', NULL, 'VLDB', 2023, 'A', 'main', NULL, NULL, 0, 0)")
    db.commit()
    db.close()


def contents(chunks_dir: Path) -> list[str]:
    """Every dumped chunk of the directory, shard by shard, line by line."""
    out: list[str] = []
    for shard in sorted(chunks_dir.glob("chunks_*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            out.extend(json.loads(line)["content"] for line in fh)
    return out


def run_dump(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=300)


def text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """A project root: the real dump script and a chunks database with interleaved documents."""
    scripts = tmp_path / "scripts" / "remote_embed"
    scripts.mkdir(parents=True)
    shutil.copy2(DUMP_SCRIPT, scripts)
    build_chunks_db(tmp_path / "data" / "db" / "retrieval.db")
    return tmp_path


@pytest.fixture()
def data_factory(project: Path):
    """A session factory over the project's own database (what get_db_session is in production)."""
    engine = create_engine(f"sqlite:///{project / 'data' / 'db' / 'retrieval.db'}",
                           connect_args={"check_same_thread": False})
    local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def factory():
        s = local()
        try:
            yield s
        finally:
            s.close()

    yield factory
    engine.dispose()


def make_ctx(store: TaskStore, project: Path, *, execution_mode: str = "local",
             cls: type = StageContext) -> StageContext:
    task_id = store.create_task(task_type="ingest_pipeline")
    return cls(venue="VLDB", year=2023, store=store, task_id=task_id, project_root=project,
               corpus_root=project / "corpus", execution_mode=execution_mode)


def write_report(ctx: StageContext, *, new: list[str], replaced: list[str]) -> Path:
    path = st.preindex_report_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"new": new, "replaced": replaced, "unchanged": []}), encoding="utf-8")
    return path


# ------------------------------------------------- 1. dump_chunks --doc-ids-file
def test_dump_chunks_doc_ids_file_dumps_exactly_the_listed_documents(project, tmp_path):
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("1\n\n3\n3\n", encoding="utf-8")  # a blank line and a repeat are harmless
    out = tmp_path / "out"

    done = run_dump([sys.executable, str(DUMP_SCRIPT), "--db", str(project / "data" / "db" / "retrieval.db"),
                     "--out", str(out), "--doc-ids-file", str(ids_file), "--start-shard", "0",
                     "--shard-size", "4"], REPO)

    assert done.returncode == 0, text(done.stderr)[-800:]
    # Doc A (1-3) and doc B (7-8), in chunk-id order, never doc X (4-6); 5 rows over shards of 4.
    assert contents(out) == [f"chunk {i} of doc {d}" for i, d in [(1, 1), (2, 1), (3, 1), (7, 3), (8, 3)]]
    assert sorted(p.name for p in out.glob("chunks_*.jsonl.gz")) == ["chunks_0000.jsonl.gz", "chunks_0001.jsonl.gz"]
    assert "task dump: chunks of 2 documents" in text(done.stdout) and "5 rows" in text(done.stdout)


def test_dump_chunks_doc_ids_file_excludes_the_chunk_id_range_options(project, tmp_path):
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("1\n", encoding="utf-8")
    for option in ("--min-chunk-id", "--max-chunk-id"):
        done = run_dump([sys.executable, str(DUMP_SCRIPT), "--db", str(project / "data" / "db" / "retrieval.db"),
                         "--out", str(tmp_path / "out"), "--doc-ids-file", str(ids_file), option, "2"], REPO)
        assert done.returncode == 2
        assert "cannot be combined" in text(done.stderr)
    assert not (tmp_path / "out").exists()  # refused before anything was written


# ------------------------------------------ 2. the #1305 scenario, stage functions
def test_a_retried_task_dumps_its_own_documents_not_the_chunk_id_range_between(store, project, data_factory):
    ctx = make_ctx(store, project)
    # What the old stage left behind: before = low, after = the max chunk id of the retry,
    # so (before, after] covers doc X, which another task wrote in between.
    ctx.state_set(chunk_id_before=0, chunk_id_after=8)
    write_report(ctx, new=["pA", "pB", "pNull"], replaced=["pA"])

    assert st.task_doc_ids(ctx, session_factory=data_factory) == [1, 3]  # sorted, distinct, NULL skipped
    cmd = st.build_embed_dump_cmd(ctx, start_shard=0, session_factory=data_factory)
    assert "--min-chunk-id" not in cmd and "--max-chunk-id" not in cmd
    ids_file = Path(cmd[cmd.index("--doc-ids-file") + 1])
    assert ids_file == st._embed_task_dir(ctx) / "doc_ids.txt" and ids_file.read_text().split() == ["1", "3"]
    assert st.dump_selection_label(cmd) == "documents of this task: 2"

    done = run_dump(cmd, project)

    assert done.returncode == 0, text(done.stderr)[-800:]
    assert contents(st._embed_chunks_dir(ctx)) == [
        "chunk 1 of doc 1", "chunk 2 of doc 1", "chunk 3 of doc 1", "chunk 7 of doc 3", "chunk 8 of doc 3"]


def test_a_task_with_no_preindex_report_keeps_the_chunk_id_range(store, project, data_factory):
    ctx = make_ctx(store, project)
    ctx.state_set(chunk_id_before=3, chunk_id_after=6)  # an external start at embed
    assert st.task_doc_ids(ctx, session_factory=data_factory) is None

    cmd = st.build_embed_dump_cmd(ctx, start_shard=0, session_factory=data_factory)

    assert cmd[cmd.index("--min-chunk-id") + 1] == "4" and cmd[cmd.index("--max-chunk-id") + 1] == "6"
    assert "--doc-ids-file" not in cmd and st.dump_selection_label(cmd) == "chunk ids > 3"
    done = run_dump(cmd, project)
    assert done.returncode == 0, text(done.stderr)[-800:]
    assert contents(st._embed_chunks_dir(ctx)) == [f"chunk {i} of doc 2" for i in (4, 5, 6)]
    assert not (st._embed_task_dir(ctx) / "doc_ids.txt").exists()


# --------------------------------------- 3. a preindex re-run forgets the old dump
@dataclass
class SelectedContext(StageContext):
    """A context that names its own preindex selection (as the arXiv contexts do), so the
    stage runs the preindex command without the Markdown path sync."""

    def preindex_selection(self) -> list[str]:
        return ["--selection", "stand-in"]


STAND_IN_PREINDEX = textwrap.dedent('''
    import argparse, json, pathlib
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-path")
    args, _ = parser.parse_known_args()
    path = pathlib.Path(args.report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"new": ["pA"], "replaced": []}), encoding="utf-8")
    print("stand-in preindex wrote", path)
''')


def stale_dump_state(ctx: StageContext) -> None:
    ctx.state_set(chunk_id_before=5, chunk_id_after=30, dump_start_shard=0, shard_start=0, shard_end=31)


def write_stale_task_dir(ctx: StageContext) -> Path:
    task_dir = st._embed_task_dir(ctx)
    (task_dir / "chunks").mkdir(parents=True)
    (task_dir / "vectors").mkdir(parents=True)
    (task_dir / "chunks" / "chunks_0000.jsonl.gz").write_bytes(gzip.compress(b'{"content": "stale"}\n'))
    (task_dir / "chunks" / "chunks_0030.jsonl.gz").write_bytes(gzip.compress(b'{"content": "stale"}\n'))
    (task_dir / "chunks" / "done.txt").write_text("chunks_0000.jsonl.gz\n", encoding="utf-8")
    (task_dir / "vectors" / "vectors_0000.parquet").write_bytes(b"stale vectors")
    (task_dir / "doc_ids.txt").write_text("9\n", encoding="utf-8")
    return task_dir


def snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture()
def preindex_project(project: Path) -> Path:
    (project / "scripts" / "preindex_md_to_sql_only.py").write_text(STAND_IN_PREINDEX, encoding="utf-8")
    return project


def test_a_preindex_rerun_clears_the_dump_state_and_removes_the_old_dump(store, preindex_project, data_factory):
    ctx = make_ctx(store, preindex_project, cls=SelectedContext)
    stale_dump_state(ctx)
    task_dir = write_stale_task_dir(ctx)
    lines: list[str] = []

    st.make_preindex_stage(ctx, session_factory=data_factory).run(
        ctx, lambda m, level="info": lines.append(m))

    assert [ctx.state_get(k) for k in ("chunk_id_before", "chunk_id_after", "dump_start_shard",
                                      "shard_start", "shard_end")] == [None] * 5
    assert not task_dir.exists()
    assert any("removed" in line and "2 chunk shards" in line for line in lines)
    assert any("stand-in preindex wrote" in line for line in lines)  # the preindex command still ran
    assert json.loads(st.preindex_report_path(ctx).read_text(encoding="utf-8"))["new"] == ["pA"]
    # The next embed starts from the report: nothing in the state says otherwise.
    assert st.nothing_to_embed(ctx) is False


def test_a_remote_mode_preindex_rerun_clears_the_state_but_deletes_no_files(store, preindex_project, data_factory):
    ctx = make_ctx(store, preindex_project, execution_mode="remote", cls=SelectedContext)
    stale_dump_state(ctx)
    task_dir = write_stale_task_dir(ctx)
    shared = preindex_project / "data" / "remote_embed"
    (shared / "chunks").mkdir(parents=True)
    (shared / "vectors").mkdir(parents=True)
    (shared / "chunks" / "chunks_0059.jsonl.gz").write_bytes(b"another run's shard")
    (shared / "vectors" / "vectors_0059.parquet").write_bytes(b"another run's vectors")
    before = snapshot(preindex_project / "data")

    st.make_preindex_stage(ctx, session_factory=data_factory).run(ctx, lambda *a, **k: None)

    assert [ctx.state_get(k) for k in ("chunk_id_before", "chunk_id_after", "dump_start_shard",
                                      "shard_start", "shard_end")] == [None] * 5
    assert task_dir.exists() and snapshot(preindex_project / "data") == before


# --------------------------------------------------------------- 4. nothing_to_embed
def test_nothing_to_embed_follows_the_report_and_falls_back_to_the_chunk_range(store, tmp_path):
    ctx = make_ctx(store, tmp_path)
    ctx.state_set(chunk_id_before=100, chunk_id_after=130)  # a stale range must not matter once a report exists

    write_report(ctx, new=[], replaced=[])
    assert st.nothing_to_embed(ctx) is True
    write_report(ctx, new=["pA"], replaced=[])
    assert st.nothing_to_embed(ctx) is False
    write_report(ctx, new=[], replaced=["pB"])
    assert st.nothing_to_embed(ctx) is False

    # No report: the task never ran preindex, the chunk range decides.
    only_before = make_ctx(store, tmp_path)
    only_before.state_set(chunk_id_before=100)
    assert st.nothing_to_embed(only_before) is False
    empty_range = make_ctx(store, tmp_path)
    empty_range.state_set(chunk_id_before=100, chunk_id_after=100)
    assert st.nothing_to_embed(empty_range) is True
    wide_range = make_ctx(store, tmp_path)
    wide_range.state_set(chunk_id_before=100, chunk_id_after=130)
    assert st.nothing_to_embed(wide_range) is False


def test_an_empty_report_with_a_stale_wide_range_does_not_embed_anything(store, tmp_path):
    ctx = make_ctx(store, tmp_path)
    ctx.state_set(chunk_id_before=100, chunk_id_after=130)
    write_report(ctx, new=[], replaced=[])
    lines: list[str] = []

    spec = st.make_embed_stage_local(ctx, encode=lambda texts: pytest.fail("embedded something"))
    spec.run(ctx, lambda m, level="info": lines.append(m))

    assert any("nothing to do" in line for line in lines)
    assert spec.gate(ctx).ok
    assert not st._embed_task_dir(ctx).exists()


# ------------------------------------------------- 5. preindex write-ahead of new papers
@pytest.fixture(scope="module")
def preindex_module():
    spec = importlib.util.spec_from_file_location(
        "preindex_write_ahead_under_test", REPO / "scripts" / "preindex_md_to_sql_only.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def test_new_papers_of_a_committed_batch_are_in_the_report_when_the_process_dies(
        tmp_path, preindex_module, monkeypatch):
    factory = make_session_factory(tmp_path / "app.db")
    monkeypatch.setattr(preindex_module, "get_db_session", factory)
    monkeypatch.setattr(preindex_module, "_set_sqlite_pragma_fast", lambda: None)
    records = []
    for n in ("00001", "00002"):
        md = tmp_path / f"{n}.md"
        md.write_text(f"# Paper {n}\n\n## Abstract\n\nabstract {n}\n\n## 1 Introduction\n\n"
                      + "A sentence that fills the section. " * 40, encoding="utf-8")
        records.append({
            "paper_id": f"arxiv-2601.{n}", "title": f"Paper {n}", "authors": ["A B"], "conference": "arXiv",
            "year": 2026, "pdf_url": f"https://arxiv.org/pdf/2601.{n}",
            "paper_page_url": f"https://arxiv.org/abs/2601.{n}", "source_name": "arxiv_html",
            "markdown_path": str(md), "arxiv_id": f"2601.{n}", "venue_tier": "preprint",
            "track_type": "preprint", "source_version": "v1", "withdrawn": False})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(records), encoding="utf-8")
    report_path = tmp_path / "report.json"

    real_process = preindex_module._process_record
    calls = []

    def process_then_die(**kwargs):
        calls.append(kwargs["record"].paper_id)
        if len(calls) == 2:
            raise KeyboardInterrupt("killed while processing the second paper")  # not an Exception
        return real_process(**kwargs)

    monkeypatch.setattr(preindex_module, "_process_record", process_then_die)
    args = preindex_module.build_argparser().parse_args(
        ["--manifest-path", str(manifest), "--report-path", str(report_path),
         "--commit-batch", "1", "--checkpoint-interval", "1000"])

    with pytest.raises(KeyboardInterrupt):
        preindex_module.run(args)

    assert calls == [records[0]["paper_id"], records[1]["paper_id"]]
    with factory.engine.connect() as conn:
        rows = conn.exec_driver_sql("SELECT paper_id, doc_id FROM papers").fetchall()
    assert [r[0] for r in rows] == [records[0]["paper_id"]]  # the first batch was committed
    # ...and the report names it, although no checkpoint was reached.
    assert report_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8"))["new"] == [records[0]["paper_id"]]
