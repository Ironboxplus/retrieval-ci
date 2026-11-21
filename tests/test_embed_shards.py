"""embed_shards: the one shard-processing loop shared by the embed_worker CLI and the backend.

Real files only: gzipped jsonl shards in tmp_path are read, real parquet is written and read
back. The encode function is a deterministic real function (text -> vector by hashing), the
boundary where the CLI plugs in the GPU model and the backend plugs in the shared embedder.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.indexing.embedding_profile import get_profile

REPO = Path(__file__).resolve().parents[1]
WORKER_PATH = REPO / "scripts" / "remote_embed" / "embed_worker.py"


def _load_worker():
    spec = importlib.util.spec_from_file_location("embed_worker_under_test", WORKER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ew = _load_worker()


def det_vector(text: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
    vec = np.random.default_rng(seed).standard_normal(1024).astype(np.float32)
    return vec / np.linalg.norm(vec)


def det_sparse(text: str) -> dict[str, float]:
    # str keys: the shared embedder returns token ids as strings, the FlagEmbedding model as ints.
    return {str(int.from_bytes(hashlib.sha256(w.encode()).digest()[:3], "little") % 250000):
            round(0.1 + (len(w) % 7) / 10, 3) for w in text.split()}


class Encoder:
    """Deterministic encode that records every call it receives."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        dense = np.stack([det_vector(t) for t in texts]) if texts else np.zeros((0, 1024), np.float32)
        return dense, [det_sparse(t) for t in texts]

    @property
    def seen(self) -> list[str]:
        return [t for call in self.calls for t in call]


def make_record(shard: int, i: int) -> dict:
    # Lengths deliberately out of order inside a shard, a non-ascii text, a metadata key
    # that overrides nothing and one record without metadata.
    pad = "word " * ((i * 7 + shard * 3) % 11)
    content = f"s{shard}r{i} {pad}é中文" if i % 4 == 0 else f"s{shard}r{i} {pad}".strip()
    rec = {
        "point_id": 1000 * shard + i, "chunk_id_str": f"c{shard}-{i}", "doc_id": 50 + shard,
        "content": content, "level": 2, "start_idx": i * 10, "end_idx": i * 10 + 9,
    }
    if i % 5:
        rec["metadata"] = {"venue": "VLDB", "year": 2023, "paper_id": f"p{shard}"}
    return rec


def write_shard(chunks_dir: Path, index: int, records: list[dict]) -> None:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(chunks_dir / f"chunks_{index:04d}.jsonl.gz", "wt", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


@pytest.fixture()
def shards(tmp_path):
    chunks = tmp_path / "chunks"
    records = {0: [make_record(0, i) for i in range(5)],
               1: [make_record(1, i) for i in range(7)],
               2: [make_record(2, i) for i in range(4)]}
    for index, recs in records.items():
        write_shard(chunks, index, recs)
    return chunks, tmp_path / "vectors", records


def expected_payload(rec: dict) -> str:
    payload = {"chunk_id": rec["chunk_id_str"], "doc_id": rec["doc_id"], "content": rec["content"],
               "level": rec["level"], "start_idx": rec["start_idx"], "end_idx": rec["end_idx"]}
    payload.update(rec.get("metadata") or {})
    payload["embed_fp"] = get_profile().fingerprint
    return json.dumps(payload, ensure_ascii=False)


def test_embed_shards_writes_every_row_with_schema_payload_and_input_order(shards):
    chunks, vectors, records = shards
    encode = Encoder()

    finished = ew.embed_shards(chunks, vectors, encode, group_size=3, log=lambda m: None)

    assert finished is True
    assert sorted(p.name for p in vectors.iterdir()) == [
        "done.rank0.txt", "vectors_0000.parquet", "vectors_0001.parquet", "vectors_0002.parquet"]
    assert (vectors / "done.rank0.txt").read_text().splitlines() == [
        "vectors_0000.parquet", "vectors_0001.parquet", "vectors_0002.parquet"]
    for index, recs in records.items():
        table = pq.read_table(vectors / f"vectors_{index:04d}.parquet")
        assert table.schema == ew.parquet_schema()
        assert table["point_id"].to_pylist() == [r["point_id"] for r in recs]
        assert table["chunk_id_str"].to_pylist() == [r["chunk_id_str"] for r in recs]
        assert table["doc_id"].to_pylist() == [r["doc_id"] for r in recs]
        assert table["payload_json"].to_pylist() == [expected_payload(r) for r in recs]
        assert table["dense"].to_pylist() == [
            np.asarray(det_vector(r["content"]), dtype=np.float16).tolist() for r in recs]
        sparse = [det_sparse(r["content"]) for r in recs]
        assert table["sparse_indices"].to_pylist() == [[int(k) for k in s] for s in sparse]
        assert table["sparse_values"].to_pylist() == [
            np.asarray(list(s.values()), dtype=np.float32).tolist() for s in sparse]


def test_each_group_reaches_encode_sorted_by_length_and_rows_keep_input_order(shards):
    chunks, vectors, records = shards
    encode = Encoder()

    ew.embed_shards(chunks, vectors, encode, group_size=6, encode_batch_size=2, log=lambda m: None)

    # Shard 1 has 7 records: groups of 6 and 1. Every call is at most 2 texts and the texts of a
    # group, concatenated over its calls, never get shorter.
    assert max(len(call) for call in encode.calls) <= 2
    flat = encode.seen
    sizes = [5, 6, 1, 4]  # groups: shard0 | shard1 first 6 | shard1 last 1 | shard2
    start = 0
    for size, shard_records in zip(sizes, [records[0], records[1][:6], records[1][6:], records[2]]):
        group = flat[start:start + size]
        start += size
        lengths = [len(t) for t in group]
        assert lengths == sorted(lengths)
        assert sorted(group) == sorted(r["content"] for r in shard_records)
    assert start == len(flat)
    # Input order of the output is untouched although encode saw a different order.
    table = pq.read_table(vectors / "vectors_0001.parquet")
    assert table["chunk_id_str"].to_pylist() == [r["chunk_id_str"] for r in records[1]]
    assert table["dense"].to_pylist()[0] == np.asarray(det_vector(records[1][0]["content"]), np.float16).tolist()


def test_whole_group_goes_to_one_encode_call_without_a_batch_size(shards):
    chunks, vectors, records = shards
    encode = Encoder()
    ew.embed_shards(chunks, vectors, encode, group_size=2048, log=lambda m: None)
    assert [len(call) for call in encode.calls] == [5, 7, 4]


def test_group_size_zero_hands_each_whole_shard_to_one_encode_call(shards):
    """The CLI's default: the service client keeps its requests flowing across the whole shard,
    so no group boundary leaves the GPU waiting for the next group to be read."""
    chunks, vectors, records = shards
    encode = Encoder()
    ew.embed_shards(chunks, vectors, encode, group_size=0, log=lambda m: None)
    assert [len(call) for call in encode.calls] == [5, 7, 4]
    table = pq.read_table(vectors / "vectors_0001.parquet")
    assert table["point_id"].to_pylist() == [r["point_id"] for r in records[1]]


def test_rows_without_sparse_weights_get_empty_lists(tmp_path):
    chunks, vectors = tmp_path / "chunks", tmp_path / "vectors"
    write_shard(chunks, 0, [{"point_id": i, "content": f"text {i}"} for i in range(3)])

    def encode(texts):
        dense, _ = Encoder()(texts)
        return dense, [{} if t == "text 1" else {"5": 0.25, 7: 0.5} for t in texts]

    assert ew.embed_shards(chunks, vectors, encode, group_size=0, log=lambda m: None) is True
    table = pq.read_table(vectors / "vectors_0000.parquet")
    assert table["sparse_indices"].to_pylist() == [[5, 7], [], [5, 7]]
    assert table["sparse_values"].to_pylist() == [[0.25, 0.5], [], [0.25, 0.5]]

def test_a_finished_parquet_and_a_done_entry_are_skipped(shards):
    chunks, vectors, records = shards
    vectors.mkdir()
    # A whole parquet from an earlier run (another rank or world size, so not in this done file).
    pq.write_table(pa.table({"finished": ["earlier, must not be rewritten"]}), vectors / "vectors_0000.parquet")
    finished = (vectors / "vectors_0000.parquet").read_bytes()
    (vectors / "done.rank0.txt").write_text("vectors_0001.parquet\n")
    encode = Encoder()
    lines: list[str] = []

    ew.embed_shards(chunks, vectors, encode, log=lines.append)

    assert (vectors / "vectors_0000.parquet").read_bytes() == finished
    assert not (vectors / "vectors_0001.parquet").exists()      # listed done: skipped, not recreated
    assert sorted(encode.seen) == sorted(r["content"] for r in records[2])
    assert (vectors / "vectors_0002.parquet").exists()
    assert sum(line.startswith("SKIP") for line in lines) == 2


def test_a_parquet_cut_off_by_a_kill_is_embedded_again(shards, tmp_path):
    """ECCV 2026 (2026-11-21): oomd killed the backend while it wrote vectors_0000.parquet; the
    resume skipped the half file because it existed, and the ingest stage failed on it."""
    chunks, vectors, records = shards
    reference = tmp_path / "reference"
    ew.embed_shards(chunks, reference, Encoder(), log=lambda m: None)
    whole = (reference / "vectors_0000.parquet").read_bytes()
    vectors.mkdir()
    (vectors / "vectors_0000.parquet").write_bytes(whole[: len(whole) // 2])
    encode = Encoder()
    lines: list[str] = []

    assert ew.embed_shards(chunks, vectors, encode, log=lines.append) is True

    assert sorted(encode.seen) == sorted(r["content"] for recs in records.values() for r in recs)
    for index in (0, 1, 2):
        name = f"vectors_{index:04d}.parquet"
        assert pq.read_table(vectors / name).equals(pq.read_table(reference / name))
    assert [line for line in lines if line.startswith("REDO")] == [
        "REDO chunks_0000.jsonl.gz: vectors_0000.parquet is not a complete parquet (cut off while being written)"]
    assert (vectors / "done.rank0.txt").read_text().splitlines() == [
        "vectors_0000.parquet", "vectors_0001.parquet", "vectors_0002.parquet"]


class _Killed(BaseException):
    """Stands for the process dying: nothing after the raise runs."""


def test_a_write_killed_midway_leaves_nothing_under_the_final_name(shards, tmp_path, monkeypatch):
    chunks, vectors, records = shards
    reference = tmp_path / "reference"
    ew.embed_shards(chunks, reference, Encoder(), log=lambda m: None)
    real_write = pq.write_table

    def killed_while_writing(table, where, **kwargs):
        real_write(table, where, **kwargs)
        with open(where, "r+b") as fh:          # the bytes that reached the disk before the kill
            fh.truncate(Path(where).stat().st_size // 2)
        raise _Killed()

    monkeypatch.setattr(pq, "write_table", killed_while_writing)
    with pytest.raises(_Killed):
        ew.embed_shards(chunks, vectors, Encoder(), log=lambda m: None)
    monkeypatch.setattr(pq, "write_table", real_write)

    assert not list(vectors.glob("vectors_*.parquet"))
    assert not (vectors / "done.rank0.txt").exists()

    assert ew.embed_shards(chunks, vectors, Encoder(), log=lambda m: None) is True
    for index in (0, 1, 2):
        name = f"vectors_{index:04d}.parquet"
        assert pq.read_table(vectors / name).equals(pq.read_table(reference / name))


def test_each_rank_takes_its_own_shards_and_records_them_in_its_own_done_file(shards):
    chunks, vectors, records = shards
    encode = Encoder()
    ew.embed_shards(chunks, vectors, encode, rank=1, world_size=2, log=lambda m: None)
    assert sorted(p.name for p in vectors.iterdir()) == ["done.rank1.txt", "vectors_0001.parquet"]


def test_stopping_mid_shard_leaves_no_partial_parquet_and_the_next_run_completes(shards, tmp_path):
    chunks, vectors, records = shards
    reference = tmp_path / "reference"
    ew.embed_shards(chunks, reference, Encoder(), log=lambda m: None)

    # Shard 0 takes 3 encode calls (groups of 4 and 1, two texts per call); the stop arrives
    # after the 4th call, in the middle of shard 1's first group.
    first = Encoder()
    stopped = ew.embed_shards(chunks, vectors, first, group_size=4, encode_batch_size=2,
                              should_stop=lambda: len(first.calls) >= 4, log=lambda m: None)

    assert stopped is False
    assert (vectors / "vectors_0000.parquet").exists()
    assert not (vectors / "vectors_0001.parquet").exists() and not (vectors / "vectors_0002.parquet").exists()
    assert not [p for p in vectors.iterdir() if p.name not in ("vectors_0000.parquet", "done.rank0.txt")]
    assert (vectors / "done.rank0.txt").read_text().splitlines() == ["vectors_0000.parquet"]

    second = Encoder()
    assert ew.embed_shards(chunks, vectors, second, group_size=4, log=lambda m: None) is True

    # Shard 0 was not embedded again; the final files equal an uninterrupted run's.
    assert not set(second.seen) & {r["content"] for r in records[0]}
    for index in (0, 1, 2):
        name = f"vectors_{index:04d}.parquet"
        assert pq.read_table(vectors / name).equals(pq.read_table(reference / name))
    assert (vectors / "done.rank0.txt").read_text().splitlines() == [
        "vectors_0000.parquet", "vectors_0001.parquet", "vectors_0002.parquet"]


def test_stop_is_checked_before_every_encode_call_not_only_between_shards(shards):
    chunks, vectors, records = shards
    calls = 0

    def encode(texts):
        nonlocal calls
        calls += 1
        return Encoder()(texts)

    ew.embed_shards(chunks, vectors, encode, encode_batch_size=2,
                    should_stop=lambda: calls >= 1, log=lambda m: None)

    assert calls == 1
    assert not list(vectors.glob("*.parquet"))


def test_generic_records_without_paper_fields_keep_their_own_payload(tmp_path):
    """The re-embed pipeline ships {point_id, content[, payload_json]} records: no chunk id,
    document id or level. Such a row has null paper columns and the record's own payload
    (or "{}"); a paper record in the same shard is built exactly as before."""
    chunks, vectors = tmp_path / "chunks", tmp_path / "vectors"
    own = json.dumps({"entry_id": "e1", "embed_fp": "feedfacefeedface", "content": "gamma"}, ensure_ascii=False)
    paper = make_record(0, 1)
    write_shard(chunks, 0, [
        {"point_id": 7, "content": "alpha beta"},
        {"point_id": 8, "content": "gamma", "payload_json": own},
        paper,
    ])

    assert ew.embed_shards(chunks, vectors, Encoder(), log=lambda m: None) is True

    table = pq.read_table(vectors / "vectors_0000.parquet")
    assert table.schema == ew.parquet_schema()
    assert table["point_id"].to_pylist() == [7, 8, paper["point_id"]]
    assert table["chunk_id_str"].to_pylist() == [None, None, paper["chunk_id_str"]]
    assert table["doc_id"].to_pylist() == [None, None, paper["doc_id"]]
    assert table["payload_json"].to_pylist() == ["{}", own, expected_payload(paper)]
    assert table["dense"].to_pylist()[0] == np.asarray(det_vector("alpha beta"), np.float16).tolist()


def test_importing_the_worker_does_not_load_the_model_library_or_touch_the_environment():
    env = {k: v for k, v in os.environ.items() if k != "CUDA_DEVICE_ORDER"}
    code = (
        "import importlib.util, os, sys;"
        f"spec = importlib.util.spec_from_file_location('w', {str(WORKER_PATH)!r});"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
        "print('FlagEmbedding' in sys.modules, 'torch' in sys.modules, 'CUDA_DEVICE_ORDER' in os.environ)"
    )
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-800:]
    assert done.stdout.split() == ["False", "False", "False"]
