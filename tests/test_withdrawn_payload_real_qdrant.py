"""The optional `withdrawn` payload key, from SQL to the papers collections.

Written only as True (core/storage/papers_payload.OPTIONAL_PAPER_KEYS), through the
one projector every write path uses; set and cleared on existing points by
qdrant_operations.set_paper_withdrawn; covered by a BOOL payload index that
scripts/recreate_qdrant_collection.py adds idempotently. Real SQLite, real local
Qdrant (scratch collections), no fakes.
"""
from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from app.models.schemas import Chunk  # noqa: E402
from core.db.models import Chunk as DBChunk, Document as DBDocument, Paper as DBPaper, SourceType  # noqa: E402
from core.storage import corpus_layout  # noqa: E402
from core.storage.corpus_store import paper_store  # noqa: E402
from core.storage.papers_payload import OPTIONAL_PAPER_KEYS, project_papers_payload  # noqa: E402
from core.storage.qdrant_operations import set_paper_withdrawn  # noqa: E402
from tests.qdrant_scratch import drop, scratch_names  # noqa: E402
from tests.sqlite_task_db import make_session_factory  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LOCAL_URL = "http://127.0.0.1:6333"
DIM = 8
BASE = {"doc_id": 1, "source_type": "paper", "level": 2, "content": "x", "paper_id": "arxiv-2601.1",
        "venue": "arXiv", "year": 2026, "venue_tier": "preprint"}


def script_module(directory: str, name: str):
    sys.path.insert(0, str(REPO / "scripts" / directory))
    try:
        return __import__(name)
    finally:
        sys.path.pop(0)


# ------------------------------------------------------------------ projector
def test_withdrawn_survives_the_projector_only_as_true():
    assert OPTIONAL_PAPER_KEYS == {"withdrawn", "preprint_full_text"}
    assert project_papers_payload({**BASE, "withdrawn": True})["withdrawn"] is True
    assert project_papers_payload({**BASE, "withdrawn": True, "source_type": "review"})["withdrawn"] is True
    for off in (False, None):
        assert "withdrawn" not in project_papers_payload({**BASE, "withdrawn": off})
    assert "withdrawn" not in project_papers_payload(BASE)


def test_false_withdrawn_is_dropped_quietly_but_other_profiles_still_reject_the_key(caplog):
    with caplog.at_level(logging.WARNING, logger="core.storage.papers_payload"):
        project_papers_payload({**BASE, "withdrawn": False})
    assert not caplog.records  # a legitimate optional key is not an out-of-profile key
    upload = {"doc_id": 5, "source_type": "upload", "content": "x", "level": 1, "withdrawn": True}
    with caplog.at_level(logging.WARNING, logger="core.storage.papers_payload"):
        projected = project_papers_payload(upload)
    assert "withdrawn" not in projected
    assert any("withdrawn" in r.getMessage() for r in caplog.records)


def test_local_ingest_keeps_the_flag_through_the_shared_projection():
    local_ingest = script_module("remote_embed", "local_ingest")
    assert local_ingest.canonicalize_payload({**BASE, "withdrawn": True})["withdrawn"] is True
    assert "withdrawn" not in local_ingest.canonicalize_payload({**BASE, "withdrawn": False})


def test_chunk_dump_reads_the_flag_from_the_papers_row(tmp_path):
    dump_chunks = script_module("remote_embed", "dump_chunks")
    path = tmp_path / "dump.db"
    factory = make_session_factory(path)
    with factory() as db:
        for pid, flag in (("arxiv-2601.1", True), ("arxiv-2601.2", False)):
            doc = DBDocument(title=pid, source="s", source_type=SourceType.paper)
            db.add(doc)
            db.flush()
            db.add(DBPaper(paper_id=pid, doc_id=doc.id, title=pid, venue="arXiv", year=2026,
                           venue_tier="preprint", withdrawn=flag))
            db.add(DBChunk(doc_id=doc.id, chunk_index=0, content="c", level=1, start_idx=0, end_idx=1,
                           metadata_json={"paper_id": pid}))
    conn = sqlite3.connect(path)
    try:
        payloads = {p["paper_id"]: p for p in map(dump_chunks.build_payload, dump_chunks.iter_chunks(conn))}
    finally:
        conn.close()
    assert payloads["arxiv-2601.1"]["withdrawn"] is True
    assert "withdrawn" not in payloads["arxiv-2601.2"]
    # ... and what ingest would write from them keeps exactly that.
    assert project_papers_payload({**payloads["arxiv-2601.1"], "content": "c", "doc_id": 1, "level": 1}
                                  )["withdrawn"] is True


# --------------------------------------------------------------- real Qdrant
NAMES: dict[str, str] = {}


@pytest.fixture()
def client(monkeypatch, tmp_path):
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=10)
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    NAMES.update(scratch_names("withdrawn"))
    monkeypatch.setattr(corpus_layout, "QDRANT_PAPERS_COLLECTION", NAMES["papers"])
    monkeypatch.setattr(corpus_layout, "QDRANT_ARXIV_COLLECTION", NAMES["arxiv"])
    monkeypatch.setattr(corpus_layout, "QDRANT_REVIEWS_COLLECTION", NAMES["reviews"])
    monkeypatch.setenv("QDRANT_CORPUS_LAYOUT_FILE", str(tmp_path / "corpus_layout.json"))
    for name in NAMES.values():
        client.create_collection(
            name,
            vectors_config={"dense": qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)},
            sparse_vectors_config={"sparse": qmodels.SparseVectorParams()},
        )
    yield client
    drop(client, NAMES.values())


def chunk(doc_id: str, paper_id: str, **meta) -> Chunk:
    return Chunk(id=f"{doc_id}-c", doc_id=doc_id, content=f"text {doc_id}", level=2, start_idx=0, end_idx=6,
                 metadata={"paper_id": paper_id, "venue": "arXiv", "year": 2026, "venue_tier": "preprint", **meta},
                 embedding=[0.5] * DIM)


def payloads(client, name) -> dict[str, dict]:
    points, _ = client.scroll(name, limit=100, with_payload=True)
    return {p.payload["paper_id"]: p.payload for p in points}


def count(client, name, *, must=None, must_not=None) -> int:
    flt = qmodels.Filter(
        must=[qmodels.FieldCondition(key="withdrawn", match=qmodels.MatchValue(value=True))] if must else None,
        must_not=[qmodels.FieldCondition(key="withdrawn", match=qmodels.MatchValue(value=True))] if must_not else None)
    return client.count(name, count_filter=flt, exact=True).count


def test_add_chunks_writes_the_flag_only_when_true(client):
    paper_store(NAMES["papers"]).add_chunks(
        [chunk("1", "p1", withdrawn=True), chunk("2", "p2", withdrawn=False), chunk("3", "p3")],
        source_type="paper")
    stored = payloads(client, NAMES["papers"])
    assert stored["p1"]["withdrawn"] is True
    assert "withdrawn" not in stored["p2"] and "withdrawn" not in stored["p3"]


def test_set_and_clear_round_trip_on_existing_points(client):
    name = NAMES["papers"]
    paper_store(name).add_chunks([chunk(str(i), f"p{i}") for i in range(1, 5)], source_type="paper")
    client.create_payload_index(name, "withdrawn", qmodels.PayloadSchemaType.BOOL, wait=True)
    client.create_payload_index(name, "paper_id", qmodels.PayloadSchemaType.KEYWORD, wait=True)

    # A point that lacks the key is not "withdrawn=true", so the default exclusion keeps it.
    assert (count(client, name, must=True), count(client, name, must_not=True)) == (0, 4)

    # More ids than one request carries, most of them unknown: only the real ones are touched.
    ids = ["p2", "p3"] + [f"ghost-{i}" for i in range(1100)]
    set_paper_withdrawn(client, name, ids, True)
    set_paper_withdrawn(client, name, ids, True)  # repeating is harmless
    assert (count(client, name, must=True), count(client, name, must_not=True)) == (2, 2)
    stored = payloads(client, name)
    assert stored["p2"]["withdrawn"] is True and "withdrawn" not in stored["p1"]
    assert stored["p2"]["content"] == "text 2"  # the rest of the payload is untouched

    set_paper_withdrawn(client, name, ["p3"], False)
    set_paper_withdrawn(client, name, ["p3"], False)
    assert count(client, name, must=True) == 1
    assert "withdrawn" not in payloads(client, name)["p3"]
    assert payloads(client, name)["p3"]["content"] == "text 3"


def test_recreate_script_adds_the_bool_index_once(client):
    recreate = script_module("", "recreate_qdrant_collection")
    name = NAMES["arxiv"]
    created = recreate.ensure_payload_indexes(client, name)
    assert "withdrawn" in created and set(created) == {f for f, _ in recreate.PAYLOAD_INDEXES}
    schema = client.get_collection(name).payload_schema
    assert schema["withdrawn"].data_type == qmodels.PayloadSchemaType.BOOL
    assert schema["year"].data_type == qmodels.PayloadSchemaType.INTEGER
    assert recreate.ensure_payload_indexes(client, name) == []  # nothing left to add

    # An existing collection that already has the old indexes only gains the new one.
    other = NAMES["reviews"]
    for field, schema_type in recreate.PAYLOAD_INDEXES:
        if field != "withdrawn":
            client.create_payload_index(other, field, schema_type, wait=True)
    assert recreate.ensure_payload_indexes(client, other) == ["withdrawn"]
