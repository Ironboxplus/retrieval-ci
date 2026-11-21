"""The local embed stage encodes in the backend process with the backend's shared embedder.

Real dump_chunks.py over a real SQLite database, the real shard loop, and the shared embedder
(a client of a real BGE-M3 vLLM service; needs RETRIEVAL_TEST_VLLM_EMBED_URL, tests/vllm_support.py).
The reference is the same service asked for the same chunk texts directly: the stage's parquet rows
must equal it. No second model process is started.
"""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from app.services.pipeline import stages as st
from app.services.pipeline.context import StageContext
from tests.embedding_model_support import bge_m3_snapshot, repo_text
from tests.reembed_support import new_store, read_chunk_shard
from tests.vllm_support import EMBED_ENV, service_url

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def shared_embedder():
    """The backend's process-wide embedder, a client of the real service: the stage must find it
    through ``get_embedder``."""
    from config import settings
    from core.indexing import embedders

    url = service_url(EMBED_ENV)
    mp = pytest.MonkeyPatch()
    mp.setenv("BGE_M3_PATH", bge_m3_snapshot())
    mp.setenv("HF_HUB_OFFLINE", "1")
    mp.setattr(settings.embedding, "service_url", url)
    embedders.reset_embedder_cache()
    yield embedders.get_embedder()
    embedders.reset_embedder_cache()
    mp.undo()


def _texts(count: int) -> list[str]:
    """Slices of tracked repository files (the fixed concatenation `real_text` reads, present in
    the public CI snapshot too): Chinese and English text, from a few chars to ~7k."""
    source = repo_text()
    lengths = [30, 120, 400, 900, 2500, 7000]
    return [source[i * 211: i * 211 + lengths[i % len(lengths)]] for i in range(count)]


def _project(tmp_path: Path, texts: list[str]) -> Path:
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
                   [(i + 1, text, 0, len(text)) for i, text in enumerate(texts)])
    db.commit()
    db.close()
    return tmp_path


def _ctx(project: Path, count: int) -> StageContext:
    store, _ = new_store()
    task_id = store.create_task(task_type="ingest_pipeline")
    ctx = StageContext(venue="arXiv", year=20260900, store=store, task_id=task_id, project_root=project,
                       corpus_root=project / "corpus", execution_mode="local")
    ctx.state_set(chunk_id_before=0, chunk_id_after=count)
    return ctx


def _dense(path: Path) -> np.ndarray:
    return np.asarray(pq.read_table(path)["dense"].to_pylist(), dtype=np.float32)


def test_the_stage_runs_in_process_and_matches_the_service(shared_embedder, tmp_path, monkeypatch):
    texts = _texts(24)
    ctx = _ctx(_project(tmp_path, texts), len(texts))
    commands: list[str] = []
    real_run_local = st.run_local

    def recording_run_local(cmd, cwd, sink, **kw):
        commands.append(" ".join(map(str, cmd)))
        return real_run_local(cmd, cwd, sink, **kw)

    monkeypatch.setattr(st, "run_local", recording_run_local)
    lines: list[str] = []

    st.make_embed_stage_local(ctx).run(ctx, lambda m, level="info": lines.append(m))

    assert any("dump_chunks.py" in c for c in commands)
    assert not any("embed_worker.py" in c for c in commands), commands       # no worker process
    stage = st._embed_vectors_dir(ctx) / "vectors_0000.parquet"
    assert stage.exists(), lines

    records = read_chunk_shard(st._embed_chunks_dir(ctx) / "chunks_0000.jsonl.gz")
    reference = shared_embedder.embed_texts([r["content"] for r in records], priority="interactive")
    table = pq.read_table(stage)
    assert table["point_id"].to_pylist() == [r["point_id"] for r in records]
    ours = _dense(stage)
    assert ours.shape == reference.dense.shape == (24, 1024)
    cosines = np.sum(ours * reference.dense, axis=1) / (
        np.linalg.norm(ours, axis=1) * np.linalg.norm(reference.dense, axis=1))
    assert cosines.min() >= 0.999
    for row, weights in enumerate(reference.sparse):       # the 20 heaviest lexical tokens agree (the tail is fp16 noise)
        stored = dict(zip(table["sparse_indices"][row].as_py(), table["sparse_values"][row].as_py()))
        top = [set(sorted(w, key=lambda k: -w[k])[:20]) for w in ({str(k): v for k, v in stored.items()}, weights)]
        assert len(top[0] & top[1]) >= 0.9 * min(20, len(top[1]))


def test_an_abort_in_the_middle_is_a_cancellation_and_leaves_no_partial_parquet(shared_embedder, tmp_path):
    texts = _texts(70)                                      # two encode calls: 64 and 6
    ctx = _ctx(_project(tmp_path, texts), len(texts))
    calls = []

    def encode_then_abort(batch):
        calls.append(len(batch))
        ctx.abort.set()                                    # the Tasks API cancelled the run
        from core.indexing.embedders import embed_texts
        result = embed_texts(list(batch), priority="background")
        return result.dense, result.sparse

    with pytest.raises(RuntimeError, match="cancelled"):
        st.make_embed_stage_local(ctx, encode=encode_then_abort).run(ctx, lambda *a, **k: None)

    assert len(calls) == 1                                  # stopped before the second encode call
    assert not list(st._embed_vectors_dir(ctx).glob("*.parquet"))
