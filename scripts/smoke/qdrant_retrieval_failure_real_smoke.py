"""Real transport-failure smoke; it never contacts or mutates production Qdrant."""
from __future__ import annotations

import json

from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from core.storage.qdrant_store import QdrantRetrievalUpstreamError, QdrantStore


def store_for_closed_loopback() -> QdrantStore:
    # Deliberately bypass __init__: its collection bootstrap is a write-oriented
    # setup concern.  The client itself is real and points at an unused port.
    store = object.__new__(QdrantStore)
    store.collection_name = "never-contacted-smoke"
    store.qdrant_dir = ""
    store.client = QdrantClient(url="http://127.0.0.1:1", timeout=1, prefer_grpc=False)
    return store


def main() -> None:
    store = store_for_closed_loopback(); dense = [0.0] * 1024
    sparse = SparseVector(indices=[1], values=[1.0])
    calls = {
        "dense": lambda: store.search(dense, top_k=1),
        "sparse": lambda: store.sparse_search(sparse, top_k=1),
        "hybrid": lambda: store.hybrid_search(dense, sparse, top_k=1),
        "batch": lambda: store.hybrid_search_many([(dense, sparse)], top_k=1),
    }
    outcome = {}
    for name, call in calls.items():
        try:
            call()
        except QdrantRetrievalUpstreamError as exc:
            outcome[name] = {"type": type(exc).__name__, "cause": type(exc.__cause__).__name__}
        else:
            raise SystemExit(f"{name} incorrectly became a successful empty result")
    print(json.dumps(outcome));


if __name__ == "__main__":
    main()
