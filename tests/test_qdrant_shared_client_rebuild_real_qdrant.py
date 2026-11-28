"""A shared-client rebuild must reach every QdrantStore built before it, on a real Qdrant.

2026-11-28 on .88: Qdrant was oomd-killed at 05:26 and briefly unreachable at 06:00:13.
The periodic validation in ``_get_shared_client`` failed, so ``_validate_or_rebuild``
closed the shared client and built a new one. The long-lived stores behind /api/search
had kept the old client in ``self.client`` since construction, so for seven hours every
call they made raised ``ValueError: Cannot invoke RPC on closed channel!``, surfaced as
HTTP 503 "Search retrieval upstream unavailable". A Qdrant that was healthy again could
not end it; only a backend restart did.

The rebuild here is the production path: the shared gRPC client really fails its
validation (it was closed, the same state the rebuild leaves the old client in), and
``_validate_or_rebuild`` closes it and builds a fresh one. Skipped unless a Qdrant
answers on loopback (REST 6333, gRPC 6334). Never touches production collection names.
"""
from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402

from config import settings  # noqa: E402
from core.storage import qdrant_store  # noqa: E402
from core.storage.qdrant_store import QdrantStore  # noqa: E402

LOCAL_URL = "http://127.0.0.1:6333"
COLLECTION = "shared_client_rebuild_test"


@pytest.fixture()
def shared(monkeypatch):
    try:
        probe = QdrantClient(url=LOCAL_URL, timeout=10)
        probe.get_collections()
    except Exception as exc:
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    if getattr(settings, "qdrant_url", None):
        monkeypatch.setattr(settings, "qdrant_url", LOCAL_URL)
    monkeypatch.setattr(qdrant_store, "QDRANT_URL", LOCAL_URL)
    # A private shared-client slot for this test; the module's own is restored after.
    monkeypatch.setattr(qdrant_store, "_shared_client", None)
    monkeypatch.setattr(qdrant_store, "_shared_client_key", None)
    monkeypatch.setattr(qdrant_store, "_shared_client_validated_at", 0.0)
    if probe.collection_exists(COLLECTION):
        probe.delete_collection(COLLECTION)
    yield probe
    if qdrant_store._shared_client is not None:
        qdrant_store._shared_client.close()
    if probe.collection_exists(COLLECTION):
        probe.delete_collection(COLLECTION)
    probe.close()


def _rebuild_after_validation_fails() -> None:
    """The shared client fails validation, so the rebuild closes it and builds another."""
    old = qdrant_store._shared_client
    old.close()
    assert qdrant_store._validate_or_rebuild(reason="test") is True
    assert qdrant_store._shared_client is not old


def test_a_store_built_before_a_rebuild_uses_the_rebuilt_client(shared):
    store = QdrantStore(COLLECTION)
    vector = [1.0] + [0.0] * (settings.embedding.dimension - 1)
    assert store.search(vector, top_k=3) == []

    _rebuild_after_validation_fails()

    # Old code: store.client is still the closed client -> "Cannot invoke RPC on closed
    # channel!" on every call, and search raises QdrantRetrievalUpstreamError (HTTP 503).
    # (count() is no witness: it swallows the error and returns 0.)
    assert store.search(vector, top_k=3) == []
    assert store.client.collection_exists(COLLECTION)


def test_a_store_built_after_a_rebuild_uses_the_same_live_client(shared):
    before = QdrantStore(COLLECTION)
    _rebuild_after_validation_fails()
    after = QdrantStore(COLLECTION, create_if_missing=False)
    assert before.client is after.client is qdrant_store._shared_client


def test_a_client_assigned_to_a_store_stays_with_that_store(shared):
    store = QdrantStore(COLLECTION)
    pinned = QdrantClient(url=LOCAL_URL, timeout=10)
    try:
        store.client = pinned
        _rebuild_after_validation_fails()
        assert store.client is pinned
        assert store.client.collection_exists(COLLECTION)
    finally:
        pinned.close()
