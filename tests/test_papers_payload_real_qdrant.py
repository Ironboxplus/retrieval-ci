"""Real-Qdrant round-trip for the `papers` write contract.

tests/test_papers_write_paths_contract.py asserts what the write paths BUILD,
using a recording client. That cannot prove what Qdrant actually STORES, and
the defect this whole change exists to fix lives precisely there: a KEYWORD
payload index only indexes string values, so the ~6.9M production points whose
`doc_id` was written as an int are absent from the index and every doc_id
filter over them returns 0 matches WITHOUT raising
(docs/papers_payload_contract_plan.md 1.2).

So these tests write through the real code path into a real Qdrant, read the
points back, and run a real filter. `test_int_doc_id_is_invisible_...` is the
control: it reproduces the production failure by upserting a bypassing point,
which is what makes the passing assertions meaningful rather than tautological.

Skipped unless a local Qdrant is reachable on loopback.
"""
from __future__ import annotations

import pytest

qdrant_client = pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from app.models.schemas import Chunk  # noqa: E402
from pipeline.paper_crawler.integration.api_routes import _build_paper_point_payload  # noqa: E402
from config import settings  # noqa: E402
from core.storage.papers_payload import project_papers_payload  # noqa: E402
from core.storage.qdrant_store import QdrantStore  # noqa: E402

# Loopback only, hardcoded: never resolved from settings/QDRANT_URL, so a
# deploy host's config can't redirect this suite at the 13.5M-point collection.
LOCAL_URL = "http://127.0.0.1:6333"

# The projector is keyed on the literal collection name in
# QdrantStore.add_chunks, so the scratch collection has to be called `papers`
# too. _local_client() therefore refuses to run against anything that looks
# like a real deployment.
COLLECTION = "papers"
DEPLOYMENT_MARKERS = {"journal_chunks", "code_chunks", "public_code_chunks"}
DIM = 4


@pytest.fixture()
def store():
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=10)
        existing = {c.name for c in client.get_collections().collections}
    except Exception as exc:
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")

    # A marker only means "real deployment" when it actually HOLDS data.
    # QdrantStore._ensure_collection creates a collection on first construction, so
    # an ordinary test run against a dev box leaves an EMPTY `public_code_chunks`
    # behind — and on 2026-09-01 that empty artifact silently skipped all six of
    # these tests, which is exactly the invisible loss of coverage they exist to
    # prevent. Emptiness is the discriminator; the `papers`-exists check below
    # remains the real protection for someone else's data.
    populated = sorted(
        name for name in existing & DEPLOYMENT_MARKERS
        if client.count(name).count > 0
    )
    if populated:
        pytest.skip(
            f"{LOCAL_URL} hosts populated {populated} — that is a real "
            f"deployment, not a scratch instance; refusing to touch its `{COLLECTION}`")
    if COLLECTION in existing:
        # Never mutate a `papers` someone else populated: an earlier version
        # skipped merely on non-empty and recreated an empty one, whose teardown
        # then deleted a collection set up on this instance by someone else.
        #
        # But an EMPTY `papers` is this fixture's own litter — a teardown that
        # raced, or QdrantStore._ensure_collection running in another test — and
        # skipping on it means the suite silently disables six tests because of
        # an artifact it created itself. Emptiness is the discriminator, exactly
        # as for DEPLOYMENT_MARKERS above.
        if client.count(COLLECTION).count > 0:
            pytest.skip(
                f"`{COLLECTION}` on {LOCAL_URL} holds data this fixture did not write; "
                f"drop it manually if you want these tests to run")
        client.delete_collection(COLLECTION)

    # add_chunks emits a NAMED "dense" vector when the configured embedder
    # supports sparse and a bare list otherwise (QdrantStore.supports_sparse
    # reads settings). Build whichever the real config implies, so this
    # exercises the deployed shape instead of one hardcoded here.
    params = qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)
    dense_is_named = settings.embedding.supports_sparse
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": params} if dense_is_named else params,
    )
    # Same index type production uses for doc_id — the whole point of the test.
    client.create_payload_index(
        collection_name=COLLECTION, field_name="doc_id",
        field_schema=qmodels.PayloadSchemaType.KEYWORD, wait=True,
    )

    s = QdrantStore.__new__(QdrantStore)
    s.collection_name = COLLECTION
    s.client = client
    s.dense_is_named = dense_is_named
    try:
        yield s
    finally:
        client.delete_collection(COLLECTION)


def _raw_upsert(store, point_id, payload):
    """Write a payload straight to Qdrant, bypassing the projector — used to
    reproduce what the pre-fix write paths put in production."""
    vector = [0.1] * DIM
    store.client.upsert(collection_name=COLLECTION, wait=True, points=[
        qmodels.PointStruct(
            id=point_id,
            vector={"dense": vector} if store.dense_is_named else vector,
            payload=payload,
        )])


def _stored_payloads(store):
    points, _ = store.client.scroll(
        collection_name=COLLECTION, limit=100, with_payload=True, with_vectors=False)
    return [p.payload for p in points]


def _doc_id_filter(store, value):
    return store.client.count(
        collection_name=COLLECTION,
        count_filter=qmodels.Filter(must=[qmodels.FieldCondition(
            key="doc_id", match=qmodels.MatchValue(value=value))]),
        exact=True,
    ).count


def test_stored_payload_is_exactly_the_projector_output(store):
    """Round-trip: what Qdrant returns must equal the projector's dict, not
    merely what the write path assembled in memory."""
    chunk = Chunk(id="c1", doc_id="555", content="text", level=2,
                  start_idx=0, end_idx=4, embedding=[0.1] * DIM,
                  metadata={"venue": "ICML", "year": 2026, "venue_tier": "CCF-A"})

    store.add_chunks([chunk], source_type="paper", doc_title="T", doc_source="S")

    stored = _stored_payloads(store)
    assert len(stored) == 1
    expected = project_papers_payload({
        "doc_id": "555", "content": "text", "level": 2,
        "title": "T", "source": "S", "source_type": "paper",
        "venue": "ICML", "year": 2026, "venue_tier": "CCF-A",
    })
    assert stored[0] == expected
    # title/source are out of the paper profile: prove they did not survive the
    # round trip, rather than trusting the in-memory dict.
    assert "title" not in stored[0]
    assert "source" not in stored[0]


def test_metadata_backdoor_keys_do_not_survive_the_round_trip(store):
    chunk = Chunk(id="c1", doc_id="1", content="text", level=1,
                  start_idx=0, end_idx=4, embedding=[0.1] * DIM,
                  metadata={"totally_made_up_field": "should not survive",
                            "another_leak": 123})

    store.add_chunks([chunk], source_type="paper")

    stored = _stored_payloads(store)[0]
    assert "totally_made_up_field" not in stored
    assert "another_leak" not in stored


def test_int_doc_id_is_unindexed_and_missed_by_the_str_filter(store):
    """CONTROL — reproduces the production defect in this very Qdrant.

    Two separate facts, measured on Qdrant 1.18, both needed to state the
    defect correctly:

    1. The int-valued point is genuinely absent from the KEYWORD index —
       `payload_schema.doc_id.points` counts one fewer than the collection.
       This is what makes production's 6,689,513 / 13,556,378 = 49.3% index
       coverage line up with the int/str split.
    2. An int-VALUED filter still matches it anyway, via a payload scan over
       the unindexed values. Verified separately at 12,002 points, i.e. past
       the default full_scan_threshold of 10,000, so this is not a
       small-collection artifact.

    So the data is not unreachable in principle. It is unreachable through the
    query shape production actually uses: every caller (api_routes.py,
    GET /index-status, scroll-by-doc_id) filters on `str(doc_id)`, and that
    returns 0 over int-stored points, silently and without raising.

    Without this control the assertions below would not prove anything.
    """
    _raw_upsert(store, 1, {"doc_id": "111", "source_type": "paper"})
    _raw_upsert(store, 2, {"doc_id": 42605, "source_type": "paper"})

    info = store.client.get_collection(COLLECTION)
    assert info.points_count == 2
    assert info.payload_schema["doc_id"].points == 1, (
        "the int-valued doc_id must be absent from the KEYWORD index")

    assert _doc_id_filter(store, "42605") == 0   # what production asks: 0 hits
    assert _doc_id_filter(store, 42605) == 1     # reachable only by an int filter
    assert _doc_id_filter(store, "111") == 1     # the str-stored point is fine


def test_projector_makes_int_doc_id_filterable(store):
    """D2: the int doc_id comes from SQL through the single-paper reindex
    endpoint (QdrantStore.add_chunks cannot be the source — Chunk.doc_id is
    typed str). _build_paper_point_payload must normalize it, so the filter
    that returns 0 in the control above finds this point."""
    payload = _build_paper_point_payload(
        {"doc_id": 42605, "content": "text", "level": 1, "start_idx": 0, "end_idx": 4},
        {"paper_id": "aaai-2022-42605", "venue": "AAAI", "year": 2026,
         "venue_tier": "CCF-A"},
    )
    assert payload["doc_id"] == "42605", "SQL int must be normalized before the write"

    _raw_upsert(store, 2, payload)

    assert _doc_id_filter(store, "42605") == 1


def test_upload_profile_keeps_its_business_fields_through_qdrant(store):
    """D1: upload is not a paper — original_filename/data_source must survive."""
    chunk = Chunk(id="c1", doc_id="7", content="text", level=1,
                  start_idx=0, end_idx=4, embedding=[0.1] * DIM,
                  metadata={"original_filename": "report.pdf", "file_size": 1024,
                            "chunk_size": 4, "leak_me": "no"})

    store.add_chunks([chunk], source_type="upload", data_source="user",
                     doc_title="T", doc_source="S")

    stored = _stored_payloads(store)[0]
    assert stored["original_filename"] == "report.pdf"
    assert stored["data_source"] == "user"
    assert stored["title"] == "T"          # in the upload profile, unlike paper
    assert "leak_me" not in stored


def test_out_of_profile_warning_is_reported_once_per_key_set(store, caplog):
    """QdrantStore.add_chunks structurally adds chunk_id/start_idx/end_idx/
    title/source to every point, and none are in the paper profile — so an
    unconditional warning fires on EVERY chunk, 13.5M lines for one full
    reindex. (Dropping the `data_source` kwarg at index_integration.py's call
    site shortens that list but does not stop the flood.) The signal must
    survive without the volume: same key set reports once, a genuinely new
    out-of-bound key still reports.
    """
    import core.storage.papers_payload as pp
    pp._warned_drops.clear()

    def write(point_suffix, metadata):
        store.add_chunks(
            [Chunk(id=f"c{point_suffix}", doc_id=str(point_suffix), content="text",
                   level=1, start_idx=0, end_idx=4, embedding=[0.1] * DIM,
                   metadata=metadata)],
            source_type="paper")

    with caplog.at_level("WARNING", logger="core.storage.papers_payload"):
        write(1, {})
        write(2, {})
        write(3, {})
    assert len(caplog.records) == 1, (
        f"three identical writes must warn once, got {len(caplog.records)}: "
        f"{[r.getMessage() for r in caplog.records]}")
    assert "chunk_id" in caplog.records[0].getMessage()

    caplog.clear()
    with caplog.at_level("WARNING", logger="core.storage.papers_payload"):
        write(4, {"a_brand_new_leak": 1})
    assert len(caplog.records) == 1, "a new key set must still be reported"
    assert "a_brand_new_leak" in caplog.records[0].getMessage()

    caplog.clear()
    with caplog.at_level("WARNING", logger="core.storage.papers_payload"):
        write(5, {"a_brand_new_leak": 2})
    assert caplog.records == [], "...but only once"
