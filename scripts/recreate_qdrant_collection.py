"""Recreate Qdrant `papers` collection with the full production config in one shot.

One-shot creation avoids the W10 disaster path:
  - PATCH-driven config drift (W5)
  - existing segments missing payload index → loader补建串行 (W10)
  - reactive PUT payload index after ingest → 168 task storm

By declaring everything at collection creation:
  - on_disk dense vector + on_disk HNSW graph (RAM 4-6 GB)
  - int8 scalar quantization (4x compression, 2x speed via SIMD)
  - 8 payload indexes pre-created (filter pushdown ready before first upsert)
  - `--indexes-only` adds the indexes a collection lacks (the BOOL `withdrawn` index
    on an existing `papers`/`arxiv`) without touching its points; safe to repeat
  - optimizer config baked in (W6 safe threads, W7 on_disk segment is born correct)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Force NO_PROXY so qdrant_client doesn't get poisoned by system proxy (W4)
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PayloadSchemaType,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from core.storage.papers_payload import (
    COMMON_OPTIONAL_STRING_KEYS,
    OPTIONAL_PAPER_KEYS,
    PAPER_PROFILE_KEYS,
)


COLLECTION = "papers"

# Derived from PAPER_PROFILE_KEYS (core/storage/papers_payload.py) — the
# projector's single source of truth for what a `papers` payload contains —
# instead of a second hand-typed list. A hardcoded second copy is exactly
# how this script grew a stray 9th `data_source` index in production: it
# duplicated local_ingest.PAYLOAD_KEEP_KEYS rather than importing it, and
# the two drifted (papers_payload_contract_plan.md 1.3/1.4). No index for
# `content` (never a filter predicate, only rerank input) or `data_source`
# (D3: legitimate upload/webpage payload key, but no Qdrant search path
# filters by it — an index nothing queries is dead weight, not a safety net).
_INTEGER_FIELDS = {"year", "level"}

def _schema(field: str) -> PayloadSchemaType:
    if field in _INTEGER_FIELDS:
        return PayloadSchemaType.INTEGER
    # Optional keys are booleans (`withdrawn`): Qdrant indexes them as `bool`, and a
    # keyword index would not see a JSON true.
    return PayloadSchemaType.BOOL if field in OPTIONAL_PAPER_KEYS else PayloadSchemaType.KEYWORD


PAYLOAD_INDEXES = [
    (field, _schema(field))
    for field in sorted((PAPER_PROFILE_KEYS - {"content"}) | OPTIONAL_PAPER_KEYS
                        | COMMON_OPTIONAL_STRING_KEYS)
]


def ensure_payload_indexes(client: QdrantClient, collection: str) -> list[str]:
    """Create the payload indexes `collection` lacks and return their field names.

    A field already indexed is left alone (whatever its schema), so this is
    idempotent and also what a fresh collection goes through.
    """
    have = set(client.get_collection(collection).payload_schema or {})
    created = []
    for field, schema in PAYLOAD_INDEXES:
        if field in have:
            continue
        print(f"[index] PUT {field} ({schema.value})")
        client.create_payload_index(
            collection_name=collection, field_name=field, field_schema=schema, wait=True,
        )
        created.append(field)
    return created


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop existing collection (if any) before recreating",
    )
    parser.add_argument(
        "--indexes-only",
        action="store_true",
        help="Only add the payload indexes the existing collection lacks; no data is touched",
    )
    args = parser.parse_args()

    client = QdrantClient(url=args.url, timeout=60)
    if args.indexes_only:
        created = ensure_payload_indexes(client, args.collection)
        print(f"[done] collection='{args.collection}' created indexes: {created or 'none (already complete)'}")
        return

    # Safety: check existing collection
    existing = [c.name for c in client.get_collections().collections]
    if args.collection in existing:
        if not args.force:
            print(f"[!] collection '{args.collection}' already exists. "
                  f"Pass --force to drop and recreate.")
            sys.exit(1)
        print(f"[force] deleting existing collection '{args.collection}'")
        client.delete_collection(args.collection)

    print(f"[create] collection='{args.collection}' "
          f"dense=1024d on_disk hnsw_on_disk int8_quantization "
          f"+ {len(PAYLOAD_INDEXES)} payload indexes")

    client.create_collection(
        collection_name=args.collection,
        # ━━━━━━ Vector ━━━━━━
        vectors_config={
            "dense": VectorParams(
                size=1024,
                distance=Distance.COSINE,
                on_disk=True,                       # raw 1024d float32 → mmap disk
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    on_disk=True,                   # HNSW graph itself → mmap disk
                    max_indexing_threads=6,         # per-segment HNSW build threads (W11: 单 segment 内, 官方推荐 8-16, 留 2 核给 Qdrant 其他线程)
                ),
            ),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=True),
            ),
        },
        # ━━━━━━ int8 quantization ━━━━━━
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=0.99,
                always_ram=False,                   # quantized vectors 走 mmap；2026-05-17 09:45 改 true→false 释放 9.5GB RAM（W15）。代价：search 第一次 cold disk read，稳态 OS page cache 兜底
            ),
        ),
        # ━━━━━━ Optimizer ━━━━━━
        optimizers_config=OptimizersConfigDiff(
            max_optimization_threads=3,             # W6: 设 8 锁全表; 3 安全
            indexing_threshold=20000,
            memmap_threshold=50000,                 # cold segment 自动 mmap 出 RAM
            default_segment_number=5,               # 明确控制 segment 数, 不让自动膨胀
            flush_interval_sec=5,
        ),
        # ━━━━━━ Single-machine ━━━━━━
        shard_number=1,
        replication_factor=1,
        write_consistency_factor=1,
    )

    print(f"[create] collection created")

    # PUT the payload indexes BEFORE first upsert
    # This way, every segment born during ingest already has these indexes,
    # avoiding the W9/W10 retroactive build storm.
    ensure_payload_indexes(client, args.collection)

    # Final state report
    info = client.get_collection(args.collection)
    print(f"\n[done] collection='{args.collection}'")
    print(f"       status={info.status}")
    print(f"       points_count={info.points_count}")
    print(f"       payload_indexes={list(info.payload_schema.keys())}")


if __name__ == "__main__":
    main()
