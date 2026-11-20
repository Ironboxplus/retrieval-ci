"""Payload indexes created at runtime follow the deployment's storage mode.

.88 runs Qdrant without LOW_MEMORY_MODE, so an index created without on_disk loads
into RAM at the next restart. With on_disk requested, the collection schema must say
so and an optimized segment must store it as an on-disk mmap index; without it the
default (in-RAM) schema is unchanged. Existing indexes are never re-created.

Skipped unless a local Qdrant is reachable on loopback.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models  # noqa: E402

from core.storage.qdrant_operations import ensure_keyword_payload_indexes  # noqa: E402

LOCAL_URL = "http://127.0.0.1:6333"
COLLECTION = "payload_index_on_disk_test"
LOCAL_STORAGE = Path(__file__).resolve().parents[1] / "tools" / "qdrant" / "storage" / "collections"


def _retry(_operation, run):
    return run()


@pytest.fixture()
def client():
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=60)
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(size=8, distance=models.Distance.COSINE),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1),
    )
    rng = random.Random(3)
    client.upsert(COLLECTION, wait=True, points=[
        models.PointStruct(id=i, vector=[rng.gauss(0, 1) for _ in range(8)],
                           payload={"path": f"p{i % 7}", "language": "py"})
        for i in range(2000)
    ])
    yield client
    client.delete_collection(COLLECTION)
    client.close()


def _schema(client):
    return client.get_collection(COLLECTION).payload_schema


def test_on_disk_creates_on_disk_keyword_indexes(client):
    failed = ensure_keyword_payload_indexes(
        client=client, collection_name=COLLECTION, field_names=("path", "language"),
        retry=_retry, on_disk=True)

    assert failed == set()
    schema = _schema(client)
    for field in ("path", "language"):
        assert schema[field].data_type == models.PayloadSchemaType.KEYWORD
        assert schema[field].params.on_disk is True
    count = client.count(COLLECTION, exact=True, count_filter=models.Filter(must=[
        models.FieldCondition(key="path", match=models.MatchValue(value="p3"))])).count
    assert count == len([i for i in range(2000) if i % 7 == 3])


def test_default_keeps_in_ram_schema_and_skips_existing(client):
    assert ensure_keyword_payload_indexes(
        client=client, collection_name=COLLECTION, field_names=("path",),
        retry=_retry) == set()
    assert _schema(client)["path"].params is None

    # An existing index is left alone even when on_disk is now requested.
    ensure_keyword_payload_indexes(
        client=client, collection_name=COLLECTION, field_names=("path", "language"),
        retry=_retry, on_disk=True)
    schema = _schema(client)
    assert schema["path"].params is None
    assert schema["language"].params.on_disk is True


def test_optimized_segments_store_the_index_on_disk(client):
    ensure_keyword_payload_indexes(
        client=client, collection_name=COLLECTION, field_names=("path",),
        retry=_retry, on_disk=True)
    deadline = time.monotonic() + 60
    while client.get_collection(COLLECTION).status != models.CollectionStatus.GREEN:
        assert time.monotonic() < deadline, "collection never finished optimizing"
        time.sleep(0.5)
    count = client.count(COLLECTION, exact=True, count_filter=models.Filter(must=[
        models.FieldCondition(key="path", match=models.MatchValue(value="p0"))])).count
    assert count == len(range(0, 2000, 7))

    configs = list((LOCAL_STORAGE / COLLECTION).glob("0/segments/*/payload_index/config.json"))
    if not configs:
        pytest.skip(f"local Qdrant storage not at {LOCAL_STORAGE}")
    def indexed(path):
        try:  # a segment the optimizer just replaced is removed after the glob
            return json.loads(path.read_text())["indexed_types"]["path"]
        except FileNotFoundError:
            return []

    immutable = [t for path in configs for t in indexed(path)
                 if t["mutability"] == "immutable" and t["index_type"] == "keyword_index"]
    assert immutable, "no optimized segment carries the path index"
    assert all(t["storage_type"] == {"mmap": {"is_on_disk": True}} for t in immutable)
