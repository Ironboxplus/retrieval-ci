"""BGE-M3 embed worker (single GPU).

Standalone — does NOT import the project's `config` layer (which pulls in
pydantic_settings etc.). Uses FlagEmbedding.BGEM3FlagModel directly so the
remote worker can run inside the kss `order` venv without extra deps.

Sparse output format matches `core/indexing/embedders/bge_m3.py:embed_texts`
(token_id_int -> float weight); local_ingest.py uses the same format when
building Qdrant SparseVector.

Run via run_embed_kss.sh (binds one process per GPU via CUDA_VISIBLE_DEVICES).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

# Force CUDA to enumerate by PCI bus id so CUDA_VISIBLE_DEVICES=N selects
# PHYSICAL GPU N. On a mixed-arch box (the 4090 host: 5090 + 4090) the default
# FASTEST_FIRST ordering scrambles indices and a worker can land on GPU 0
# (another user's card). Must be set before any torch / FlagEmbedding import.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from FlagEmbedding import BGEM3FlagModel  # noqa: E402


def iter_lines(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def parquet_schema():
    return pa.schema([
        ("point_id", pa.int64()),
        ("chunk_id_str", pa.string()),
        ("doc_id", pa.int64()),
        ("payload_json", pa.string()),
        ("dense", pa.list_(pa.float16(), 1024)),
        ("sparse_indices", pa.list_(pa.int32())),
        ("sparse_values", pa.list_(pa.float32())),
    ])


def build_payload(rec: dict) -> dict:
    payload = {
        "chunk_id": rec["chunk_id_str"],
        "doc_id": rec["doc_id"],
        "content": rec["content"],
        "level": rec["level"],
        "start_idx": rec["start_idx"],
        "end_idx": rec["end_idx"],
    }
    payload.update(rec.get("metadata") or {})
    return payload


def process_shard(embedder: BGEM3FlagModel, in_path: Path, out_path: Path,
                  batch_size: int):
    point_ids, chunk_ids, doc_ids, payloads = [], [], [], []
    dense_rows: list[list[float]] = []
    sp_idx: list[list[int]] = []
    sp_val: list[list[float]] = []

    batch_texts: list[str] = []
    batch_meta: list[dict] = []
    total = 0

    def flush():
        nonlocal batch_texts, batch_meta, total
        if not batch_texts:
            return
        output = embedder.encode(
            batch_texts,
            batch_size=len(batch_texts),
            max_length=2048,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = np.asarray(output["dense_vecs"], dtype=np.float16)
        sparses = output["lexical_weights"]
        for j, rec in enumerate(batch_meta):
            point_ids.append(int(rec["point_id"]))
            chunk_ids.append(rec["chunk_id_str"])
            doc_ids.append(int(rec["doc_id"]))
            payloads.append(json.dumps(build_payload(rec), ensure_ascii=False))
            dense_rows.append(dense[j].tolist())
            sp = sparses[j] or {}
            sp_idx.append([int(k) for k in sp.keys()])
            sp_val.append([float(v) for v in sp.values()])
        total += len(batch_texts)
        print(f"    [{in_path.name}] {total:,} done", flush=True)
        batch_texts, batch_meta = [], []

    for rec in iter_lines(in_path):
        batch_texts.append(rec["content"])
        batch_meta.append(rec)
        if len(batch_texts) >= batch_size:
            flush()
    flush()

    table = pa.Table.from_pydict({
        "point_id": point_ids,
        "chunk_id_str": chunk_ids,
        "doc_id": doc_ids,
        "payload_json": payloads,
        "dense": dense_rows,
        "sparse_indices": sp_idx,
        "sparse_values": sp_val,
    }, schema=parquet_schema())
    pq.write_table(table, out_path, compression="zstd")
    print(f"  wrote {out_path.name}  rows={total:,}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="dir of chunks_*.jsonl.gz")
    parser.add_argument("--output", required=True, help="dir for vectors_*.parquet")
    parser.add_argument("--model-dir", required=True,
                        help="local path to BGE-M3 (e.g. /mnt/nvme-disk/.cache/modelscope/BAAI/bge-m3)")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / f"done.rank{args.rank}.txt"
    done = set(done_path.read_text().splitlines()) if done_path.exists() else set()

    shards = sorted(in_dir.glob("chunks_*.jsonl.gz"))
    my_shards = [s for i, s in enumerate(shards) if i % args.world_size == args.rank]
    print(f"rank={args.rank}/{args.world_size}  shards = {len(my_shards)}", flush=True)

    embedder = BGEM3FlagModel(args.model_dir, use_fp16=True, device="cuda:0")

    for shard in my_shards:
        idx = shard.name.replace("chunks_", "").replace(".jsonl.gz", "")
        out_path = out_dir / f"vectors_{idx}.parquet"
        # W34 fix: skip if EITHER the per-rank done file lists it OR the parquet
        # already exists. The old `and` form broke across a changed world_size —
        # done.rank{N}.txt sharding no longer matches the new rank, so every shard
        # whose old-rank != new-rank got re-embedded. `or exists()` is world-size-
        # agnostic: any rank that sees a finished parquet skips it. (Lets the
        # CVPR run land high shard indices on top of 100+ old S.9 staging shards
        # without re-embedding them.)
        if out_path.name in done or out_path.exists():
            print(f"SKIP {shard.name} (done)", flush=True)
            continue
        print(f"PROCESS {shard.name} -> {out_path.name}", flush=True)
        process_shard(embedder, shard, out_path, args.batch_size)
        with done_path.open("a") as f:
            f.write(out_path.name + "\n")


if __name__ == "__main__":
    main()
