"""The embed_worker CLI as a client of the real BGE-M3 vLLM service.

Real process, real gzip shards, real parquet, the real service (skipped when it is not running)
and the real tokenizer. The worker loads no model: its rows must equal what vllm_client.encode
returns for the same texts, and a missing service must stop it before any shard is touched.
"""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from core.indexing.embedders import vllm_client
from core.indexing.embedding_profile import get_profile, load_tokenizer
from tests.embedding_model_support import bge_m3_snapshot, real_text

REPO = Path(__file__).resolve().parents[1]
WORKER = REPO / "scripts" / "remote_embed" / "embed_worker.py"


@pytest.fixture(scope="module")
def snapshot():
    path = bge_m3_snapshot()
    mp = pytest.MonkeyPatch()
    mp.setenv("BGE_M3_PATH", path)
    yield path
    mp.undo()


def write_shard(directory: Path, index: int, texts: list[str], first_id: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / f"chunks_{index:04d}.jsonl.gz", "wt", encoding="utf-8") as fh:
        for j, text in enumerate(texts):
            fh.write(json.dumps({"point_id": first_id + j, "content": text}, ensure_ascii=False) + "\n")


def run_worker(chunks: Path, vectors: Path, snapshot: str, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(WORKER), "--input", str(chunks), "--output", str(vectors),
                           "--model-dir", snapshot, *extra], capture_output=True, text=True, timeout=600)


def test_the_worker_writes_the_service_vectors_without_loading_a_model(tmp_path, snapshot):
    url = vllm_client.service_url()
    try:
        urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url + "/health", timeout=3)
    except OSError:
        pytest.skip(f"BGE-M3 vLLM service is not running at {url}")
    tokenizer = load_tokenizer(get_profile())
    texts = [real_text(tokenizer, 20 + (i * 211) % 900) + ("\n" if i % 3 == 0 else "") for i in range(40)]
    chunks, vectors = tmp_path / "chunks", tmp_path / "vectors"
    write_shard(chunks, 0, texts[:25], 100)
    write_shard(chunks, 1, texts[25:], 200)

    done = run_worker(chunks, vectors, snapshot, "--batch-size", "7")

    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert "FlagEmbedding" not in done.stderr
    dense, sparse = vllm_client.encode(texts, get_profile(), tokenizer=tokenizer)
    tables = [pq.read_table(vectors / f"vectors_{i:04d}.parquet") for i in (0, 1)]
    assert [t["point_id"].to_pylist() for t in tables] == [list(range(100, 125)), list(range(200, 215))]
    stored = np.concatenate([np.asarray(t["dense"].to_pylist(), np.float32) for t in tables])
    assert np.abs(stored - dense.astype(np.float16).astype(np.float32)).max() <= 2e-3     # batch-invariant to fp16
    # Under fp16 accumulation the same text in a differently packed engine step moves the low bits, and on long
    # chunks many near-zero weights flip across the relu (key Jaccard down to 0.85 measured), so the check is the
    # one that matters for retrieval: the 20 heaviest terms.
    rows = [dict(zip(i, v)) for t in tables for i, v in zip(t["sparse_indices"].to_pylist(), t["sparse_values"].to_pylist())]

    def top20(d):
        return {k for k, _ in sorted(d.items(), key=lambda kv: -kv[1])[:20]}

    overlap = [len(top20(a) & top20(b)) / min(20, len(b)) for a, b in zip(rows, sparse)]
    assert np.mean(overlap) >= 0.95, overlap
    assert tables[0]["payload_json"].to_pylist() == ["{}"] * 25


def test_without_a_service_the_worker_stops_before_any_shard(tmp_path, snapshot):
    chunks, vectors = tmp_path / "chunks", tmp_path / "vectors"
    write_shard(chunks, 0, ["alpha", "beta"], 1)

    done = run_worker(chunks, vectors, snapshot, "--service-url", "http://127.0.0.1:9")

    assert done.returncode != 0
    assert "not answering" in done.stderr and "bge_m3_service.sh start" in done.stderr
    assert not vectors.exists() or not list(vectors.iterdir())
