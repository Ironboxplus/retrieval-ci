"""Real-Qdrant checks of the compact papers copy (scripts/compact_papers_index.py).

The copy is what a later production switch would serve from, so the properties that
matter are about stored data, not bookkeeping: every point arrives with identical
payload, dense and sparse vectors; an interrupted copy resumes from its checkpoint
without loss; the collection is born with the pinned compact quantization and
on-disk payload indexes the compact instance depends on; and indexing stays off
until the copy is complete.

Skipped unless a local Qdrant is reachable on loopback.
"""
from __future__ import annotations

import random
import shutil
import sys
import time

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models  # noqa: E402

from scripts import compact_papers_index as compact  # noqa: E402

# Loopback only, hardcoded: never resolved from settings/QDRANT_URL.
LOCAL_URL = "http://127.0.0.1:6333"
SOURCE = "compact_copy_source_test"
TARGET = "compact_copy_target_test"
POINTS = 600


def _drop(client: QdrantClient, name: str, attempts: int = 4) -> None:
    # Windows Qdrant can leave the data directory behind; deleting an unregistered
    # name clears it, so this is safe to call unconditionally.
    for attempt in range(attempts):
        try:
            client.delete_collection(name)
            return
        except Exception:  # noqa: BLE001
            if attempt == attempts - 1:
                raise
            time.sleep(0.5 * (attempt + 1))


def _payload(i: int) -> dict:
    return {"doc_id": f"doc-{i // 3}", "paper_id": f"paper-{i // 3}", "source_type": "paper",
            "level": i % 3, "venue": "ICLR", "year": 2020 + i % 5, "venue_tier": "A",
            "content": f"chunk {i} — 中文 text"}


@pytest.fixture()
def client():
    try:
        c = QdrantClient(url=LOCAL_URL, timeout=60)
        c.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    for name in (SOURCE, TARGET):
        _drop(c, name)
    c.create_collection(
        SOURCE,
        vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    rng = random.Random(7)
    points = []
    for i in range(POINTS):
        ids = sorted(rng.sample(range(1, 250000), 5))
        points.append(models.PointStruct(
            id=rng.randrange(1, 2**53),
            vector={"dense": [rng.gauss(0, 1) for _ in range(1024)],
                    "sparse": models.SparseVector(indices=ids,
                                                  values=[rng.random() for _ in ids])},
            payload=_payload(i)))
    for start in range(0, POINTS, 200):
        c.upsert(SOURCE, points=points[start:start + 200], wait=True)
    yield c
    for name in (SOURCE, TARGET):
        _drop(c, name)


def _all(client: QdrantClient, name: str) -> dict:
    out, offset = {}, None
    while True:
        page, offset = client.scroll(name, limit=256, offset=offset, with_payload=True,
                                     with_vectors=True)
        out.update({p.id: p for p in page})
        if offset is None:
            return out


@pytest.mark.parametrize("grpc_port", [None, 6334], ids=["rest", "grpc"])
def test_interrupted_copy_resumes_to_an_identical_compact_collection(client, tmp_path,
                                                                     grpc_port):
    state = tmp_path / "copy_state.json"
    compact.create_compact_papers(client, TARGET, compact._variants()["turbo_bits2"],
                                  indexing_threads=1)
    # Production runs the copy over gRPC, whose point/vector types differ from REST.
    transport = compact._client(LOCAL_URL, grpc_port)

    first = compact.copy_points(transport, SOURCE, transport, TARGET, state_path=state,
                                batch=128, limit=256)
    assert first["done"] is False and first["copied"] == 256 and first["offset"] is not None
    assert client.count(TARGET, exact=True).count == 256

    final = compact.copy_points(transport, SOURCE, transport, TARGET, state_path=state,
                                batch=128)
    assert final["done"] is True and final["copied"] == POINTS

    source, target = _all(client, SOURCE), _all(client, TARGET)
    assert set(target) == set(source)
    for pid, src in source.items():
        dst = target[pid]
        assert dst.payload == src.payload
        assert dst.vector["dense"] == pytest.approx(src.vector["dense"], abs=1e-6)
        assert dst.vector["sparse"].indices == src.vector["sparse"].indices
        assert dst.vector["sparse"].values == pytest.approx(src.vector["sparse"].values)


def test_disjoint_id_ranges_together_copy_every_point_exactly_once(client, tmp_path):
    compact.create_compact_papers(client, TARGET, compact._variants()["turbo_bits2"],
                                  indexing_threads=1)
    ids = sorted(_all(client, SOURCE))
    # Split points that are not themselves point ids, plus one that is, so both the
    # "bound falls between ids" and "bound equals an id" cases are exercised.
    bounds = [None, ids[200] + 1, ids[400], None]
    states = []
    for i in range(3):
        states.append(compact.copy_points(
            client, SOURCE, client, TARGET, state_path=tmp_path / f"range{i}.json", batch=64,
            start_id=bounds[i], end_id=bounds[i + 1]))
    assert all(s["done"] for s in states)
    assert [s["copied"] for s in states] == [201, 199, 200]
    assert sorted(_all(client, TARGET)) == ids


def test_copy_stops_resumably_when_free_disk_falls_below_the_floor(client, tmp_path):
    compact.create_compact_papers(client, TARGET, compact._variants()["turbo_bits2"],
                                  indexing_threads=1)
    state = tmp_path / "copy_state.json"
    impossible_floor = shutil.disk_usage(tmp_path).total + 1
    stopped = compact.copy_points(client, SOURCE, client, TARGET, state_path=state, batch=64,
                                  disk_path=tmp_path, min_free_bytes=impossible_floor)
    assert stopped["stopped"] == "low disk" and stopped["done"] is False
    assert stopped["copied"] == 0 and client.count(TARGET, exact=True).count == 0

    resumed = compact.copy_points(client, SOURCE, client, TARGET, state_path=state, batch=64,
                                  disk_path=tmp_path, min_free_bytes=0)
    assert resumed["done"] is True and "stopped" not in resumed
    assert client.count(TARGET, exact=True).count == POINTS


def test_compact_collection_is_born_pinned_disk_safe_and_indexing(client):
    compact.create_compact_papers(client, TARGET, compact._variants()["turbo_bits2"],
                                  indexing_threads=1)
    info = client.get_collection(TARGET)
    turbo = info.config.quantization_config.turbo
    assert turbo.bits == models.TurboQuantBitSize.BITS2 and turbo.always_ram is True
    dense = info.config.params.vectors["dense"]
    assert dense.on_disk is True and dense.hnsw_config.on_disk is False
    assert info.config.params.sparse_vectors["sparse"].index.on_disk is True
    assert info.config.params.on_disk_payload is True
    # Indexing on from birth: unoptimized segments hold their sparse/payload indexes
    # on the heap, which is what got the copy oomd-killed with indexing off.
    assert info.config.optimizer_config.indexing_threshold == \
        compact.PRODUCTION_INDEXING_THRESHOLD_KB
    assert set(info.payload_schema) == set(compact.PAYLOAD_INDEX_FIELDS)
    assert "content" not in info.payload_schema
    for field, schema in info.payload_schema.items():
        assert schema.params.on_disk is True, field


def test_existing_target_is_never_recreated(client):
    compact.create_compact_papers(client, TARGET, compact._variants()["turbo_bits2"],
                                  indexing_threads=1)
    client.upsert(TARGET, points=[models.PointStruct(
        id=1, vector={"dense": [0.1] * 1024}, payload=_payload(1))], wait=True)
    with pytest.raises(SystemExit):
        compact.create_compact_papers(client, TARGET, compact._variants()["turbo_bits2"],
                                      indexing_threads=1)
    assert client.count(TARGET, exact=True).count == 1


@pytest.mark.parametrize("argv", [
    ["copy", "--quantization", "turbo_bits2", "--state", "s.json",
     "--target-url", "http://127.0.0.1:16333"],
    ["copy", "--quantization", "turbo_bits2", "--state", "s.json", "--target-grpc-port", "16334"],
    ["trial", "--report", "r.json", "--target-url", "http://127.0.0.1:16333/"],
])
def test_cli_refuses_production_as_target(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["compact_papers_index.py", *argv])
    with pytest.raises(SystemExit) as exc:
        compact.main()
    assert exc.value.code == 2
