"""Inspect or explicitly convert one Qdrant collection to low-memory storage.

Dry-run is the default. Execution is intentionally limited to one known Retrieval
collection at a time because Qdrant rebuilds existing segments after these settings
change; running several rebuilds concurrently can exhaust RAM and temporary disk.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from qdrant_client import QdrantClient, models


DEFAULT_COLLECTIONS = ("journal_chunks", "code_chunks", "public_code_chunks")
EXECUTION_ACK = "I_ACCEPT_QDRANT_SEGMENT_REBUILD"


def _dump(info: Any) -> dict[str, Any]:
    if hasattr(info, "model_dump"):
        return info.model_dump(mode="json")
    if isinstance(info, dict):
        return info
    raise TypeError(f"unsupported Qdrant collection info type: {type(info).__name__}")


def _get(mapping: dict[str, Any], *path: str) -> Any:
    current: Any = mapping
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def collection_storage_state(info: Any) -> dict[str, Any]:
    payload = _dump(info)
    dense_hnsw = _get(
        payload, "config", "params", "vectors", "dense", "hnsw_config", "on_disk"
    )
    if dense_hnsw is None:
        dense_hnsw = _get(payload, "config", "hnsw_config", "on_disk")
    return {
        "dense_on_disk": bool(
            _get(payload, "config", "params", "vectors", "dense", "on_disk")
        ),
        "dense_hnsw_on_disk": bool(dense_hnsw),
        "sparse_on_disk": bool(
            _get(
                payload,
                "config",
                "params",
                "sparse_vectors",
                "sparse",
                "index",
                "on_disk",
            )
        ),
        "payload_on_disk": bool(
            _get(payload, "config", "params", "on_disk_payload")
        ),
        "memmap_threshold": _get(
            payload, "config", "optimizer_config", "memmap_threshold"
        ),
        "max_optimization_threads": _get(
            payload, "config", "optimizer_config", "max_optimization_threads"
        ),
    }


def _is_compliant(
    state: dict[str, Any], *, memmap_threshold: int, max_optimization_threads: int
) -> bool:
    return (
        state["dense_on_disk"]
        and state["dense_hnsw_on_disk"]
        and state["sparse_on_disk"]
        and state["payload_on_disk"]
        and state["memmap_threshold"] == memmap_threshold
        and state["max_optimization_threads"] == max_optimization_threads
    )


def configure_collection(
    client: Any,
    collection_name: str,
    *,
    execute: bool,
    memmap_threshold: int = 5000,
    max_optimization_threads: int = 1,
) -> dict[str, Any]:
    if collection_name not in DEFAULT_COLLECTIONS:
        raise ValueError(
            f"unsupported collection {collection_name!r}; choose one of "
            f"{', '.join(DEFAULT_COLLECTIONS)}"
        )
    before = collection_storage_state(client.get_collection(collection_name))
    if _is_compliant(
        before,
        memmap_threshold=memmap_threshold,
        max_optimization_threads=max_optimization_threads,
    ):
        return {
            "collection": collection_name,
            "action": "already_compliant",
            "before": before,
        }
    if not execute:
        return {
            "collection": collection_name,
            "action": "would_update",
            "before": before,
        }

    client.update_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParamsDiff(
                on_disk=True,
                hnsw_config=models.HnswConfigDiff(on_disk=True),
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=True)
            )
        },
        collection_params=models.CollectionParamsDiff(on_disk_payload=True),
        hnsw_config=models.HnswConfigDiff(on_disk=True),
        optimizers_config=models.OptimizersConfigDiff(
            memmap_threshold=memmap_threshold,
            max_optimization_threads=max_optimization_threads,
        ),
    )
    return {
        "collection": collection_name,
        "action": "update_submitted",
        "before": before,
    }


def wait_until_ready(
    client: Any,
    collection_name: str,
    *,
    memmap_threshold: int,
    max_optimization_threads: int,
    timeout_seconds: int,
    poll_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        info = client.get_collection(collection_name)
        payload = _dump(info)
        state = collection_storage_state(payload)
        status = payload.get("status")
        optimizer_status = payload.get("optimizer_status")
        queue_length = _get(payload, "update_queue", "length") or 0
        if (
            status == "green"
            and optimizer_status == "ok"
            and queue_length == 0
            and _is_compliant(
                state,
                memmap_threshold=memmap_threshold,
                max_optimization_threads=max_optimization_threads,
            )
        ):
            return {
                "collection": collection_name,
                "status": status,
                "optimizer_status": optimizer_status,
                "update_queue_length": queue_length,
                "storage": state,
            }
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"{collection_name} did not become green and low-memory compliant "
                f"within {timeout_seconds}s; status={status!r}, "
                f"optimizer_status={optimizer_status!r}, queue={queue_length}, "
                f"storage={state!r}"
            )
        time.sleep(poll_seconds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
    )
    parser.add_argument(
        "--collection",
        action="append",
        choices=DEFAULT_COLLECTIONS,
        help="repeat for dry-run; execute accepts exactly one collection",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-rebuild-overhead", default="")
    parser.add_argument("--memmap-threshold", type=int, default=5000)
    parser.add_argument("--max-optimization-threads", type=int, default=1)
    parser.add_argument("--wait-timeout-seconds", type=int, default=21600)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    collections = tuple(args.collection or DEFAULT_COLLECTIONS)
    if args.execute:
        if len(collections) != 1:
            parser.error("--execute requires exactly one --collection")
        if args.acknowledge_rebuild_overhead != EXECUTION_ACK:
            parser.error(
                "--execute requires --acknowledge-rebuild-overhead " + EXECUTION_ACK
            )
    if args.memmap_threshold < 1:
        parser.error("--memmap-threshold must be at least 1")
    if args.max_optimization_threads < 1:
        parser.error("--max-optimization-threads must be at least 1")
    if args.wait_timeout_seconds < 1 or args.poll_seconds <= 0:
        parser.error("wait timeout and poll interval must be positive")

    client = QdrantClient(url=args.url, timeout=60)
    try:
        for collection_name in collections:
            result = configure_collection(
                client,
                collection_name,
                execute=args.execute,
                memmap_threshold=args.memmap_threshold,
                max_optimization_threads=args.max_optimization_threads,
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            if args.execute and result["action"] == "update_submitted":
                ready = wait_until_ready(
                    client,
                    collection_name,
                    memmap_threshold=args.memmap_threshold,
                    max_optimization_threads=args.max_optimization_threads,
                    timeout_seconds=args.wait_timeout_seconds,
                    poll_seconds=args.poll_seconds,
                )
                print(json.dumps(ready, ensure_ascii=False, sort_keys=True))
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
