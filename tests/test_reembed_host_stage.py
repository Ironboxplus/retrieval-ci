"""The embed host stage with an injected dump, on real hardware.

The stage ships real chunk shards to the registered embed host, runs the real embed_worker
on its GPU and pulls the parquet back. The registered host (``arc99``) is this machine, whose
ssh alias exists only in other machines' configs, so the ssh/scp transport is routed to this
machine (``install_local_host``); the worker, the model and the files are real.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from app.services.pipeline import stages as st
from app.services.pipeline.context import StageContext
from core.indexing.embedders import vllm_client
from tests.embedding_model_support import bge_m3_snapshot, require_gpu
from tests.reembed_support import install_local_host, new_store, write_chunk_shard, write_host_registry

REPO = Path(__file__).resolve().parents[1]
WORKER = REPO / "scripts" / "remote_embed" / "embed_worker.py"
TEXTS = [
    "def add(a, b):\n    return a + b\n",
    "Retrieval of long journal entries must not silently truncate the embedded text.",
    "稀疏混合专家模型的路由策略",
    "SELECT id FROM chunks WHERE embed_fp IS NULL",
    "tokenizer budget",
]


@pytest.fixture()
def host(tmp_path, monkeypatch):
    snapshot = bge_m3_snapshot()
    require_gpu()
    try:        # the worker is a client of the host's BGE-M3 vLLM service; never stood in for
        urllib.request.build_opener(urllib.request.ProxyHandler({})).open(vllm_client.service_url() + "/health",
                                                                          timeout=3)
    except OSError:
        pytest.skip(f"BGE-M3 vLLM service is not running at {vllm_client.service_url()}")
    monkeypatch.setenv("BGE_M3_PATH", snapshot)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    registry = tmp_path / "embed_hosts.json"
    write_host_registry(registry, workdir=tmp_path / "hostwork", python=sys.executable, worker=WORKER,
                        model_dir=snapshot, shard_size=3)
    monkeypatch.setenv("PIPELINE_EMBED_HOSTS", str(registry))
    log: list = []
    install_local_host(monkeypatch, st, log)
    return log


def _ctx(tmp_path: Path) -> StageContext:
    store, _ = new_store()
    task_id = store.create_task(task_type="reembed_pipeline")
    return StageContext(venue="papers:uploads", year=0, store=store, task_id=task_id, project_root=tmp_path,
                        execution_mode="host", ssh_target="shim-host")


def test_injected_dump_replaces_the_sql_dump_and_the_selected_shards_are_embedded(host, tmp_path):
    ctx = _ctx(tmp_path)
    chunks = st._embed_chunks_dir(ctx)
    write_chunk_shard(chunks, 0, [{"point_id": 100 + i, "content": TEXTS[i]} for i in range(3)])
    write_chunk_shard(chunks, 1, [{"point_id": 200 + i, "content": TEXTS[3 + i]} for i in range(2)])
    dumps: list[str] = []

    def selected_shards(c, sink, host_):
        dumps.append(host_.name)
        c.state_set(dump_start_shard=0, shard_start=0, shard_end=2)

    lines: list[str] = []
    spec = st.make_embed_stage_host(ctx, dump=selected_shards, title="Embed papers:uploads on shim-host")
    assert spec.title == "Embed papers:uploads on shim-host" and spec.resource_key == "gpu-shim-host"
    spec.run(ctx, lambda m, level="info": lines.append(m))

    assert dumps == ["shim-host"]
    assert not [entry for entry in host if "dump_chunks.py" in " ".join(entry)]      # no SQL dump ran
    progress = st.embed_local_probe(ctx)
    assert (progress.done, progress.total) == (2, 2), lines
    vectors = st._embed_vectors_dir(ctx)
    first, second = pq.read_table(vectors / "vectors_0000.parquet"), pq.read_table(vectors / "vectors_0001.parquet")
    assert first["point_id"].to_pylist() == [100, 101, 102] and second["point_id"].to_pylist() == [200, 201]
    assert first["payload_json"].to_pylist() == ["{}"] * 3
    assert first["chunk_id_str"].to_pylist() == [None] * 3 and first["doc_id"].to_pylist() == [None] * 3
    dense = np.asarray(first["dense"].to_pylist() + second["dense"].to_pylist(), dtype=np.float32)
    np.testing.assert_allclose(np.linalg.norm(dense, axis=1), 1.0, atol=1e-2)
    assert not (tmp_path / "hostwork" / f"task_{ctx.task_id}").exists()                 # the host keeps nothing


def test_the_default_dump_is_still_the_sql_dump(host, tmp_path):
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
                   [(i, f"sparse mixture of experts, chunk {i}", i * 10, i * 10 + 9) for i in range(1, 6)])
    db.commit()
    db.close()
    ctx = _ctx(tmp_path)
    ctx.state_set(chunk_id_before=0, chunk_id_after=5)

    st.make_embed_stage_host(ctx).run(ctx, lambda *a, **k: None)

    assert [entry for entry in host if "dump_chunks.py" in " ".join(entry)]
    assert (ctx.state_get("dump_start_shard"), ctx.state_get("shard_start"), ctx.state_get("shard_end")) == (0, 0, 2)
    rows = sum(pq.read_table(p).num_rows for p in st._embed_vectors_dir(ctx).glob("vectors_*.parquet"))
    assert rows == 5


def _set_concurrent_shards(tmp_path: Path, value: int) -> None:
    registry = tmp_path / "embed_hosts.json"
    hosts = json.loads(registry.read_text(encoding="utf-8"))
    hosts["shim-host"]["concurrent_shards"] = value
    registry.write_text(json.dumps(hosts), encoding="utf-8")


def _record_worker_runs(monkeypatch) -> list:
    """Real (start, end) wall-clock times of every worker command the stage runs on the host."""
    runs: list = []
    lock = threading.Lock()
    shim = st.run_remote

    def timed(target, cmd, sink, **kw):
        start = time.monotonic()
        try:
            return shim(target, cmd, sink, **kw)
        finally:
            if "embed_worker.py" in cmd:
                with lock:
                    runs.append((start, time.monotonic(), cmd))

    monkeypatch.setattr(st, "run_remote", timed)
    return runs


def _journal_texts(count: int) -> list[str]:
    """Real prose of mixed length (400..3,000 characters) from the repository's journal."""
    source = (REPO / "journal.md").read_text(encoding="utf-8")
    texts, at = [], 0
    for i in range(count):
        size = 400 + (i * 617) % 2600
        texts.append(source[at:at + size])
        at += size
    return texts


def _selected(ctx: StageContext, shards: list[list[str]]):
    chunks = st._embed_chunks_dir(ctx)
    for index, texts in enumerate(shards):
        write_chunk_shard(chunks, index, [{"point_id": 1000 * (index + 1) + j, "content": t}
                                          for j, t in enumerate(texts)])

    def dump(c, sink, host_):
        c.state_set(dump_start_shard=0, shard_start=0, shard_end=len(shards))
    return dump


def test_two_concurrent_shards_equal_the_serial_result(host, tmp_path):
    texts = _journal_texts(12)
    shards = [texts[i:i + 3] for i in range(0, 12, 3)]
    serial_ctx = _ctx(tmp_path / "serial")
    st.make_embed_stage_host(serial_ctx, dump=_selected(serial_ctx, shards)).run(serial_ctx, lambda *a, **k: None)

    _set_concurrent_shards(tmp_path, 2)
    ctx = _ctx(tmp_path / "concurrent")
    lines: list[str] = []
    st.make_embed_stage_host(ctx, dump=_selected(ctx, shards)).run(ctx, lambda m, level="info": lines.append(m))

    assert sum(line.startswith("embed.host shard ") and " done (" in line for line in lines) == 4, lines
    for index in range(4):
        name = f"vectors_{index:04d}.parquet"
        got = pq.read_table(st._embed_vectors_dir(ctx) / name)
        want = pq.read_table(st._embed_vectors_dir(serial_ctx) / name)
        for column in ("point_id", "chunk_id_str", "doc_id", "payload_json"):
            assert got[column].to_pylist() == want[column].to_pylist()
        # With fp16 accumulation a row's result depends on which requests share its engine step: the other
        # worker's, or production's on the same service. Two serial runs differ by up to 1.4e-3 per element
        # when other traffic is present (bit-identical only on an idle service), and a gate run measured
        # 2.66e-3. The bound is therefore the numeric gate's per-row cosine, plus the heaviest sparse terms.
        a = np.asarray(got["dense"].to_pylist(), np.float32)
        b = np.asarray(want["dense"].to_pylist(), np.float32)
        cos = (a * b).sum(axis=1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
        assert cos.min() >= 0.998 and cos.mean() >= 0.9999, (name, cos)
        for gi, gv, wi, wv in zip(got["sparse_indices"].to_pylist(), got["sparse_values"].to_pylist(),
                                  want["sparse_indices"].to_pylist(), want["sparse_values"].to_pylist()):
            top = lambda i, v: {k for k, _ in sorted(zip(i, v), key=lambda kv: -kv[1])[:20]}
            assert len(top(gi, gv) & top(wi, wv)) >= 0.85 * min(20, len(wi))


def test_two_concurrent_shards_run_their_workers_at_the_same_time(host, tmp_path, monkeypatch):
    _set_concurrent_shards(tmp_path, 2)
    runs = _record_worker_runs(monkeypatch)
    ctx = _ctx(tmp_path)
    texts = _journal_texts(8)

    st.make_embed_stage_host(ctx, dump=_selected(ctx, [texts[i:i + 2] for i in range(0, 8, 2)])).run(
        ctx, lambda *a, **k: None)

    assert len(runs) == 4
    spans = sorted((start, end) for start, end, _ in runs)
    # Real wall-clock intervals of the worker processes: serial runs never overlap, two at a time must.
    overlaps = [min(spans[i][1], spans[i + 1][1]) - spans[i + 1][0] for i in range(len(spans) - 1)]
    assert max(overlaps) > 1.0, spans

def test_a_failing_shard_keeps_the_finished_one_and_the_stage_raises_after_it(host, tmp_path, monkeypatch):
    _set_concurrent_shards(tmp_path, 2)
    runs = _record_worker_runs(monkeypatch)
    ctx = _ctx(tmp_path)
    over_limit = "token " * 9000                      # InputTooLong in the worker: every attempt fails
    dump = _selected(ctx, [_journal_texts(3), [TEXTS[0], over_limit]])

    with pytest.raises(RuntimeError) as raised:
        st.make_embed_stage_host(ctx, dump=dump, sleep_fn=lambda s: None).run(ctx, lambda *a, **k: None)
    failed_at = time.monotonic()

    assert "vectors_0001" not in str(raised.value) and "exit" in str(raised.value)
    vectors = st._embed_vectors_dir(ctx)
    assert pq.read_table(vectors / "vectors_0000.parquet")["point_id"].to_pylist() == [1000, 1001, 1002]
    assert not (vectors / "vectors_0001.parquet").exists()
    assert sum("chunks_0001" in cmd or "_0001/in" in cmd for _, _, cmd in runs) == 4      # every attempt ran
    assert all(end <= failed_at for _, end, _ in runs)                     # nothing left running behind it
