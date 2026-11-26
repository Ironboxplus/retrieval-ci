"""The re-embed embed stage in its two modes: an ssh embed host, and this backend's own GPU.

Both modes take the same chunk shards through the same dump function; only where they are
encoded differs. Against a real BGE-M3 vLLM service (the host is this machine behind the ssh/scp
shim, its worker a client of the service; the local mode uses the backend's shared embedder, also a
client of it) the two parquet sets must agree, and the two stages must be able to run at once.
That test needs RETRIEVAL_TEST_VLLM_EMBED_URL (tests/vllm_support.py) and is skipped without it.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from app.services.pipeline import stages as st
from app.services.pipeline.reembed_pipeline import ReembedContext, make_reembed_embed_stage
from core.indexing.embedding_profile import get_profile, load_tokenizer
from tests.embedding_model_support import bge_m3_snapshot, real_text
from tests.reembed_support import install_local_host, new_store, write_chunk_shard, write_host_registry
from tests.vllm_support import EMBED_ENV, service_url

WORKER = Path(__file__).resolve().parents[1] / "scripts" / "remote_embed" / "embed_worker.py"
LENGTHS = (30, 200, 600, 900, 1500, 2600, 4000, 7000, 90, 700, 1200, 3000)      # tokens; > 512 on purpose


def _context(tmp_path: Path, mode: str, **kw) -> ReembedContext:
    store, _ = new_store()
    task_id = store.create_task(task_type="reembed_pipeline")
    return ReembedContext(venue="code_chunks:truncated", year=0, store=store, task_id=task_id, project_root=tmp_path,
                          execution_mode=mode, ssh_target="shim-host" if mode == "host" else "local",
                          scope_name="code_chunks:truncated", collection="unused", pause_s=0.0, **kw)


def _selected(ctx: ReembedContext, texts: list[str], per_shard: int = 6) -> None:
    """What the select stage leaves behind: chunk shards and their range in the task state."""
    shards = [texts[i:i + per_shard] for i in range(0, len(texts), per_shard)]
    for index, group in enumerate(shards):
        base = 1000 * (index + 1)
        write_chunk_shard(ctx.chunks_dir, index, [{"point_id": base + j, "content": t} for j, t in enumerate(group)])
    ctx.state_set(select_shard_start=0, select_shard_end=len(shards), select_complete=True)


def _lines(ctx):
    return lambda message, level="info": None


def test_the_local_mode_hands_the_encoder_length_ordered_groups_of_the_profile_batch(tmp_path):
    calls: list[list[str]] = []

    def recording(batch):
        calls.append(list(batch))
        return np.ones((len(batch), 1024), np.float32) / 32.0, [{1: 0.5} for _ in batch]

    ctx = _context(tmp_path, "local", embed_fn=recording)
    texts = ["x" * n for n in (900, 5, 4000, 60, 12, 700, 3, 250, 2200, 31, 8000, 77)]
    _selected(ctx, texts, per_shard=6)

    make_reembed_embed_stage(ctx).run(ctx, _lines(ctx))

    batch = get_profile().encode_batch_size
    assert sorted(t for call in calls for t in call) == sorted(texts)                 # every text once
    for call in calls:
        assert 0 < len(call) <= batch
        assert [len(t) for t in call] == sorted(len(t) for t in call)                 # shortest first: little padding
    assert calls[0][0] == "x" * 5 and len(calls) == 2                                 # one group per shard


@pytest.fixture()
def service(monkeypatch):
    """The real BGE-M3 service, reached by the backend's shared embedder and by the host worker alike."""
    from config import settings
    from core.indexing import embedders

    url = service_url(EMBED_ENV)
    snapshot = bge_m3_snapshot()
    monkeypatch.setenv("BGE_M3_PATH", snapshot)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("EMBEDDING__SERVICE_URL", url)               # the worker subprocess reads it
    monkeypatch.setattr(settings.embedding, "service_url", url)
    embedders.reset_embedder_cache()
    yield snapshot
    embedders.reset_embedder_cache()


def test_both_modes_produce_equal_parquet_for_the_same_shards_running_at_once(service, tmp_path, monkeypatch):
    snapshot = service
    registry = tmp_path / "embed_hosts.json"
    write_host_registry(registry, workdir=tmp_path / "hostwork", python=sys.executable, worker=WORKER,
                        model_dir=snapshot, shard_size=6)
    monkeypatch.setenv("PIPELINE_EMBED_HOSTS", str(registry))
    install_local_host(monkeypatch, st, [])

    tokenizer = load_tokenizer(get_profile())
    longs = [real_text(tokenizer, n)[(i % 5) * 13:] for i, n in enumerate(LENGTHS)]
    shorts = [real_text(tokenizer, 20 + (i * 4) % 100)[i % 3:] + f" s{i}" for i in range(28)]
    texts = longs[:6] + shorts[:14] + longs[6:] + shorts[14:]               # two shards of 20
    host_ctx, local_ctx = _context(tmp_path / "h", "host"), _context(tmp_path / "l", "local")
    for ctx in (host_ctx, local_ctx):
        ctx.project_root.mkdir()
        _selected(ctx, texts, per_shard=20)
    stages = {"host": make_reembed_embed_stage(host_ctx), "local": make_reembed_embed_stage(local_ctx)}
    assert (stages["host"].resource_key, stages["local"].resource_key) == ("gpu-shim-host", "embed-service")

    errors: list = []
    started = threading.Barrier(2)

    def drive(name: str, ctx: ReembedContext) -> None:
        try:
            started.wait(timeout=60)
            stages[name].run(ctx, _lines(ctx))
        except BaseException as exc:            # noqa: BLE001 - reported below with the thread's name
            errors.append((name, repr(exc)))

    threads = [threading.Thread(target=drive, args=(n, c)) for n, c in (("host", host_ctx), ("local", local_ctx))]
    [t.start() for t in threads]
    [t.join(timeout=900) for t in threads]
    assert not errors, errors

    for index in range(2):
        host_table = pq.read_table(host_ctx.vectors_dir / f"vectors_{index:04d}.parquet")
        local_table = pq.read_table(local_ctx.vectors_dir / f"vectors_{index:04d}.parquet")
        assert host_table["point_id"].to_pylist() == local_table["point_id"].to_pylist()
        assert host_table["payload_json"].to_pylist() == local_table["payload_json"].to_pylist()
        a = np.asarray(host_table["dense"].to_pylist(), dtype=np.float32)
        b = np.asarray(local_table["dense"].to_pylist(), dtype=np.float32)
        assert a.shape == b.shape == (20, 1024)
        cosine = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        assert cosine.min() >= 0.999, cosine
        for row in range(20):                  # the 10 heaviest lexical tokens agree; below them weights tie
            top = [set(dict(sorted(zip(t["sparse_indices"][row].as_py(), t["sparse_values"][row].as_py()),
                                   key=lambda kv: -kv[1])[:10])) for t in (host_table, local_table)]
            assert len(top[0] & top[1]) >= min(9, len(top[0]), len(top[1])), (index, row)
