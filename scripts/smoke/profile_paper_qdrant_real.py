"""One-shot read-only phase profile for the production paper Qdrant query."""
from __future__ import annotations

import argparse
import json
import time

import httpx
from qdrant_client import QdrantClient, models


def _timed(name, call):
    started = time.perf_counter()
    response = call()
    elapsed = time.perf_counter() - started
    points = response.points
    return {
        "phase": name,
        "seconds": elapsed,
        "count": len(points),
        "paper_ids": [
            (point.payload or {}).get("paper_id")
            for point in points[:4]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--qdrant-grpc-port", type=int, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--phases",
        default="dense_only,sparse_only,hybrid_no_filter,hybrid_exclude_review",
        help="Comma-separated ordered phase names",
    )
    args = parser.parse_args()

    with httpx.Client(base_url=args.backend.rstrip("/"), timeout=args.timeout) as client:
        started = time.perf_counter()
        response = client.post(
            "/api/embed",
            json={"texts": [args.query], "include_sparse": True},
        )
        response.raise_for_status()
        embedded = response.json()
        embed_seconds = time.perf_counter() - started
    dense = embedded["dense"][0]
    sparse_payload = embedded["sparse"][0]
    if "indices" in sparse_payload and "values" in sparse_payload:
        sparse = models.SparseVector(
            indices=[int(index) for index in sparse_payload["indices"]],
            values=[float(value) for value in sparse_payload["values"]],
        )
    else:
        items = sorted((int(index), float(value)) for index, value in sparse_payload.items())
        sparse = models.SparseVector(
            indices=[index for index, _ in items],
            values=[value for _, value in items],
        )
    client = QdrantClient(
        url=args.qdrant_url,
        grpc_port=args.qdrant_grpc_port,
        prefer_grpc=True,
        timeout=args.timeout,
    )
    exclude_reviews = models.Filter(must_not=[
        models.FieldCondition(
            key="source_type",
            match=models.MatchAny(any=["review"]),
        )
    ])
    papers_only = models.Filter(must=[
        models.FieldCondition(
            key="source_type",
            match=models.MatchValue(value="paper"),
        )
    ])

    calls = {
        "dense_only": lambda: client.query_points(
            "papers", query=dense, using="dense", limit=20, with_payload=True,
        ),
        "sparse_only": lambda: client.query_points(
            "papers", query=sparse, using="sparse", limit=20, with_payload=True,
        ),
        "hybrid_no_filter": lambda: client.query_points(
            "papers",
            prefetch=[
                models.Prefetch(query=dense, using="dense", limit=100),
                models.Prefetch(query=sparse, using="sparse", limit=100),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=20,
            with_payload=True,
        ),
        "hybrid_exclude_review": lambda: client.query_points(
            "papers",
            prefetch=[
                models.Prefetch(query=dense, using="dense", limit=100, filter=exclude_reviews),
                models.Prefetch(query=sparse, using="sparse", limit=100, filter=exclude_reviews),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=20,
            with_payload=True,
        ),
        "hybrid_papers_only": lambda: client.query_points(
            "papers",
            prefetch=[
                models.Prefetch(query=dense, using="dense", limit=100, filter=papers_only),
                models.Prefetch(query=sparse, using="sparse", limit=100, filter=papers_only),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=20,
            with_payload=True,
        ),
        "hybrid_papers_only_no_payload": lambda: client.query_points(
            "papers",
            prefetch=[
                models.Prefetch(query=dense, using="dense", limit=100, filter=papers_only),
                models.Prefetch(query=sparse, using="sparse", limit=100, filter=papers_only),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=20,
            with_payload=False,
        ),
    }
    phase_names = [name.strip() for name in args.phases.split(",") if name.strip()]
    unknown = [name for name in phase_names if name not in calls]
    if unknown:
        raise SystemExit(f"unknown phases: {unknown}; choose from {sorted(calls)}")
    phases = [_timed(name, calls[name]) for name in phase_names]
    print(json.dumps({
        "query": args.query,
        "embed_seconds": embed_seconds,
        "phases": phases,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
