"""Embed on an ssh embed host, end to end on real hardware.

The host stage dumps a real SQLite chunk range, ships each shard to the embed
host over ssh, runs the real BGE-M3 embed_worker on its GPU and pulls the
parquet back. Nothing is faked; the round trip is skipped when the registered
host is unreachable from the machine running the test (the orchestrator is
.88, whose ~/.ssh/config carries the host alias).
"""
from __future__ import annotations

import shutil
import sqlite3
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.services.pipeline.context import StageContext
from app.services.pipeline.stages import (
    embed_host_preflight,
    embed_local_probe,
    load_embed_host,
    make_embed_stage_for,
)
from app.services.pipeline.store import TaskStore
from core.db import models  # noqa: F401  (register tables)
from core.db.database import Base

REPO = Path(__file__).resolve().parents[1]
HOST = "arc99"


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


def test_registry():
    host = load_embed_host(HOST)
    assert host.ssh_target == HOST and host.shard_size == 25_000
    with pytest.raises(ValueError, match="unknown embed host"):
        load_embed_host("no-such-host")


def _host_up() -> bool:
    try:
        return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", HOST, "hostname"],
                              capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False


def test_host_stage_round_trip(store: TaskStore, tmp_path: Path):
    if not _host_up():
        pytest.skip(f"ssh {HOST} unavailable")
    # A project root of its own: the dump reads data/db/retrieval.db and the
    # probe counts data/remote_embed/tasks/<task_id>/vectors under it.
    (tmp_path / "scripts" / "remote_embed").mkdir(parents=True)
    shutil.copy2(REPO / "scripts" / "remote_embed" / "dump_chunks.py", tmp_path / "scripts" / "remote_embed")
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
    db.executemany("INSERT INTO chunks VALUES (?, 1, ?, 2, ?, ?, '{}')",
                   [(i, f"sparse mixture of experts, chunk {i}", i * 10, i * 10 + 9) for i in range(1, 41)])
    db.commit()
    db.close()

    task_id = store.create_task(task_type="ingest_pipeline")
    ctx = StageContext(venue="arXiv", year=20260900, store=store, task_id=task_id, project_root=tmp_path,
                       execution_mode="host", ssh_target=HOST)
    # This task's own range (preindex records both ends): chunks 11..30.
    ctx.state_set(chunk_id_before=10, chunk_id_after=30)
    spec = make_embed_stage_for(ctx)
    assert spec.resource_key == f"gpu-{HOST}"
    assert embed_host_preflight(ctx).ok

    lines: list[str] = []
    spec.run(ctx, lambda m, level="info": lines.append(m))
    progress = embed_local_probe(ctx)
    assert (progress.done, progress.total) == (1, 1), lines
    vectors = sorted((tmp_path / "data" / "remote_embed" / "tasks" / str(task_id) / "vectors").glob("vectors_*.parquet"))
    assert [p.name for p in vectors] == ["vectors_0000.parquet"]
    table = pq.read_table(vectors[0])
    assert table.num_rows == 20
    assert len(table.column("dense")[0].as_py()) == 1024

    # The host keeps nothing for this task.
    host = load_embed_host(HOST)
    left = subprocess.run(["ssh", HOST, f'{host.python} -c "import os; print(os.path.exists(\'{host.workdir}/task_{task_id}\'))"'],
                          capture_output=True, text=True).stdout.strip()
    assert left == "False"


def test_preflight_gates_on_absolute_free_memory(store: TaskStore, tmp_path: Path, monkeypatch):
    """The real host's card, gated by a registry entry: a floor no card meets
    fails after the configured wait; the deployed floor passes."""
    if not _host_up():
        pytest.skip(f"ssh {HOST} unavailable")
    import json

    real = json.loads((REPO / "deploy" / "compute88" / "embed_hosts.json").read_text(encoding="utf-8"))[HOST]
    registry = tmp_path / "embed_hosts.json"
    registry.write_text(json.dumps({
        HOST: real,
        "impossible": {**real, "min_free_mb": 10 ** 7, "preflight_wait_s": 0},
    }), encoding="utf-8")
    monkeypatch.setenv("PIPELINE_EMBED_HOSTS", str(registry))
    task_id = store.create_task(task_type="ingest_pipeline")
    for name, ok in (("impossible", False), (HOST, True)):
        ctx = StageContext(venue="arXiv", year=20260900, store=store, task_id=task_id,
                           execution_mode="host", ssh_target=name)
        gate = embed_host_preflight(ctx, sleep_fn=lambda s: None)
        assert gate.ok is ok, gate
    assert "10000000 MB free" in embed_host_preflight(
        StageContext(venue="arXiv", year=20260900, store=store, task_id=task_id,
                     execution_mode="host", ssh_target="impossible"), sleep_fn=lambda s: None).reason
