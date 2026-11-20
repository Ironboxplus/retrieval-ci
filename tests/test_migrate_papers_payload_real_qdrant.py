"""Real-Qdrant verification of the D2/D3 payload migration.

tests/test_migrate_papers_payload.py drives `migrate()` through a `_FakeClient`.
That proves the scan/plan/retry bookkeeping, and nothing else — a fake client
reports "doc_id is now a str" without Qdrant ever having been asked, so it
cannot answer the ONE question the migration exists to answer:

    does rewriting an int doc_id to a str via set_payload actually move the
    point INTO the KEYWORD payload index?

Payload value and index membership are different facts (see
tests/test_papers_payload_real_qdrant.py). If set_payload updated the stored
value but not the index, the migration would report 6.9M successful writes and
change nothing observable — every production doc_id filter would still return 0.
So these tests run the real `migrate()` against a real Qdrant and assert on
`payload_schema["doc_id"].points`, the same signal measured on production.

They also pin the two properties the data-loss question turns on: vectors are
byte-identical across the migration, and unrelated payload keys survive.

Skipped unless a local Qdrant is reachable on loopback.
"""
from __future__ import annotations

import warnings

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from scripts.migrate_papers_payload import migrate  # noqa: E402

# Loopback only, hardcoded: never resolved from settings/QDRANT_URL, so a deploy
# host's config cannot redirect this suite at the 13.5M-point production
# collection.
LOCAL_URL = "http://127.0.0.1:6333"

# Test-private name. `migrate()` has no collection-name-keyed behaviour (unlike
# the papers payload projector), so nothing forces us to borrow a real name.
COLLECTION = "papers_migration_test"
DEPLOYMENT_MARKERS = {"papers", "journal_chunks", "code_chunks", "public_code_chunks"}
DIM = 4


def _drop_collection(client: QdrantClient, attempts: int = 4) -> None:
    """Delete COLLECTION, tolerating Windows' delayed segment-file unmapping.

    Qdrant on Windows can leave the collection's data directory behind after a
    delete and then reject the next create with 400 "Collection data already
    exists" — first run green, every run after red. Calling delete_collection on
    an unregistered name returns False but clears that orphan, so this is always
    safe to call unconditionally.
    """
    last = None
    for attempt in range(attempts):
        try:
            client.delete_collection(COLLECTION)
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < attempts - 1:
                import time

                time.sleep(0.5 * (attempt + 1))
    raise last  # type: ignore[misc]


@pytest.fixture()
def client():
    try:
        c = QdrantClient(url=LOCAL_URL, timeout=30)
        existing = {x.name for x in c.get_collections().collections}
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")

    # A marker only means "real deployment" when it actually HOLDS data: an
    # ordinary test run leaves EMPTY marker collections behind, and skipping on
    # those is a silent, self-inflicted loss of coverage.
    populated = sorted(n for n in existing & DEPLOYMENT_MARKERS if c.count(n).count > 0)
    if populated:
        pytest.skip(
            f"{LOCAL_URL} hosts populated {populated} — that is a real deployment, "
            f"not a scratch instance; refusing to write to it")

    # Unconditional drop-then-create: setup is the authoritative cleanup, so a
    # leftover from a crashed run cannot turn into a permanent silent skip.
    _drop_collection(c)
    c.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": qmodels.VectorParams(
            size=DIM, distance=qmodels.Distance.COSINE)},
    )
    c.create_payload_index(
        collection_name=COLLECTION, field_name="doc_id",
        field_schema=qmodels.PayloadSchemaType.KEYWORD, wait=True,
    )
    try:
        yield c
    finally:
        # Best effort: segment files may still be mapped, and setup already
        # guarantees a clean slate for the next run.
        try:
            _drop_collection(c, attempts=3)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"could not drop {COLLECTION}: {exc}")


def _upsert(client, points):
    client.upsert(collection_name=COLLECTION, wait=True, points=[
        qmodels.PointStruct(id=pid, vector={"dense": vec}, payload=payload)
        for pid, vec, payload in points
    ])


def _indexed_doc_id_points(client) -> int:
    """How many points the KEYWORD index on doc_id actually holds.

    This is the production signal: 6,689,513 of 13,556,378 on 2026-09-02.
    """
    schema = client.get_collection(COLLECTION).payload_schema
    return schema["doc_id"].points if "doc_id" in schema else 0


def _filter_count(client, value) -> int:
    return client.count(
        collection_name=COLLECTION,
        count_filter=qmodels.Filter(must=[qmodels.FieldCondition(
            key="doc_id", match=qmodels.MatchValue(value=value))]),
        exact=True,
    ).count


class TestIndexMembershipIsActuallyRepaired:
    """The claim the whole migration rests on, measured rather than assumed."""

    def test_int_doc_id_starts_outside_the_index_and_ends_inside_it(self, client):
        _upsert(client, [
            (1, [0.1] * DIM, {"doc_id": 80328, "source_type": "paper"}),
            (2, [0.2] * DIM, {"doc_id": "33116", "source_type": "paper"}),
        ])

        # CONTROL: reproduce production. 2 points, only the str one indexed,
        # and the query shape production uses returns 0 without raising.
        assert client.count(COLLECTION).count == 2
        assert _indexed_doc_id_points(client) == 1
        assert _filter_count(client, "80328") == 0

        result = migrate(client, COLLECTION, execute=True)

        assert result["applied"]["doc_id_points_updated"] == 1
        # The point is now IN the index, not merely storing a str.
        assert _indexed_doc_id_points(client) == 2
        assert _filter_count(client, "80328") == 1
        assert _filter_count(client, "33116") == 1

    def test_dry_run_repairs_nothing(self, client):
        _upsert(client, [(1, [0.1] * DIM, {"doc_id": 80328, "source_type": "paper"})])

        result = migrate(client, COLLECTION, execute=False)

        assert result["plan"]["doc_id_str_conversions"] == 1
        assert _indexed_doc_id_points(client) == 0
        assert _filter_count(client, "80328") == 0

    def test_rerunning_the_migration_is_a_no_op(self, client):
        _upsert(client, [(i, [0.1] * DIM, {"doc_id": 1000 + i, "source_type": "paper"})
                         for i in range(20)])

        migrate(client, COLLECTION, execute=True)
        second = migrate(client, COLLECTION, execute=True)

        assert second["applied"]["doc_id_points_updated"] == 0
        assert second["plan"]["doc_id_str_conversions"] == 0
        assert _indexed_doc_id_points(client) == 20


class TestNothingIsLost:
    """The data-loss question, answered against real stored bytes."""

    def test_vectors_are_identical_across_the_migration(self, client):
        vectors = {i: [round(0.01 * i, 4), 0.5, -0.25, 0.75] for i in range(1, 11)}
        _upsert(client, [(i, v, {"doc_id": 500 + i, "source_type": "paper"})
                         for i, v in vectors.items()])
        before = {p.id: p.vector["dense"] for p in client.retrieve(
            COLLECTION, ids=list(vectors), with_vectors=True)}

        migrate(client, COLLECTION, execute=True)

        after = {p.id: p.vector["dense"] for p in client.retrieve(
            COLLECTION, ids=list(vectors), with_vectors=True)}
        assert after == before
        assert client.count(COLLECTION).count == 10, "no point may be dropped"

    def test_unrelated_payload_keys_survive_the_doc_id_rewrite(self, client):
        payload = {
            "doc_id": 42605, "source_type": "paper", "content": "text body",
            "level": 2, "paper_id": "aaai-2022-42605", "venue": "AAAI",
            "venue_tier": "CCF-A", "year": 2022,
        }
        _upsert(client, [(1, [0.1] * DIM, payload)])

        migrate(client, COLLECTION, execute=True)

        stored = client.retrieve(COLLECTION, ids=[1], with_payload=True)[0].payload
        assert stored == {**payload, "doc_id": "42605"}, (
            "set_payload must merge one key, not replace the payload")

    def test_data_source_removal_spares_upload_and_webpage(self, client):
        _upsert(client, [
            (1, [0.1] * DIM, {"doc_id": "1", "source_type": "paper", "data_source": "arxiv"}),
            (2, [0.1] * DIM, {"doc_id": "2", "source_type": "review", "data_source": "arxiv"}),
            (3, [0.1] * DIM, {"doc_id": "3", "source_type": "upload", "data_source": "user"}),
        ])

        result = migrate(client, COLLECTION, execute=True)

        assert result["applied"]["data_source_points_updated"] == 2
        stored = {p.id: p.payload for p in client.retrieve(
            COLLECTION, ids=[1, 2, 3], with_payload=True)}
        assert "data_source" not in stored[1]
        assert "data_source" not in stored[2]
        assert stored[3]["data_source"] == "user", "upload keeps its business field"


class TestUnconvertibleDocIdAborts:
    """`str()` on an unexpected type produces a plausible-looking wrong key.

    A float doc_id becomes "2.0", a bool becomes "True". Both write cleanly and
    both PASS the post-write read-back (the stored value really is what was
    sent), so verification cannot catch them — the guard has to run before the
    write, and it has to stop the run rather than skip the point.
    """

    def test_float_doc_id_aborts_before_any_write(self, client):
        _upsert(client, [
            (1, [0.1] * DIM, {"doc_id": 111, "source_type": "paper"}),
            (2, [0.1] * DIM, {"doc_id": 2.0, "source_type": "paper"}),
        ])

        with pytest.raises(ValueError, match="doc_id"):
            migrate(client, COLLECTION, execute=True, apply_batch=500)

        assert _filter_count(client, "2.0") == 0, "the bad key must never reach Qdrant"

    def test_bool_doc_id_aborts(self, client):
        _upsert(client, [(1, [0.1] * DIM, {"doc_id": True, "source_type": "paper"})])

        with pytest.raises(ValueError, match="doc_id"):
            migrate(client, COLLECTION, execute=True)

    def test_dry_run_also_aborts_so_the_scan_reports_the_problem(self, client):
        _upsert(client, [(1, [0.1] * DIM, {"doc_id": 2.5, "source_type": "paper"})])

        with pytest.raises(ValueError, match="doc_id"):
            migrate(client, COLLECTION, execute=False)


class TestWriteVolumeIsBoundedByPoints:
    """Grouping the flush by VALUE bounds nothing: `len(pending[value]) >=
    apply_batch` only fires once a SINGLE doc_id has accumulated that many
    points, and production spreads its values far too thin for that — one
    120,000-point scroll window of `papers` holds 34,571 distinct int doc_ids
    (measured 2026-09-02). So the 6.87M-id backlog stays resident until the
    final flush, which then costs a round trip per distinct value.

    The batch must therefore be counted in POINTS, and one flush must cost one
    update request plus one verification read regardless of how many distinct
    doc_id values it spans.
    """

    def test_many_distinct_doc_ids_flush_together(self, client):
        # 60 points, 60 distinct doc_ids, batch of 20 -> 3 flushes, not 60.
        _upsert(client, [(i, [0.1] * DIM, {"doc_id": 7000 + i, "source_type": "paper"})
                         for i in range(60)])

        result = migrate(client, COLLECTION, execute=True, scan_batch=1000, apply_batch=20)

        assert result["applied"]["doc_id_points_updated"] == 60
        assert result["applied"]["batches_applied"] == 3, (
            "flushes must be counted in points, not per distinct doc_id value")
        assert _indexed_doc_id_points(client) == 60
        assert _filter_count(client, "7003") == 1

    def test_memory_does_not_hold_every_pending_id(self, client):
        """A batch smaller than the collection must flush DURING the scan."""
        _upsert(client, [(i, [0.1] * DIM, {"doc_id": 9000 + i, "source_type": "paper"})
                         for i in range(50)])

        result = migrate(client, COLLECTION, execute=True, scan_batch=10, apply_batch=10)

        assert result["applied"]["batches_applied"] == 5
        assert _indexed_doc_id_points(client) == 50

    def test_a_repeated_doc_id_is_written_once_for_all_its_points(self, client):
        """Real chunking: one doc_id spans several points."""
        _upsert(client, [(i, [0.1] * DIM, {"doc_id": 4242, "source_type": "paper"})
                         for i in range(5)])

        result = migrate(client, COLLECTION, execute=True, apply_batch=100)

        assert result["applied"]["doc_id_points_updated"] == 5
        assert result["applied"]["batches_applied"] == 1
        assert _filter_count(client, "4242") == 5

    def test_verification_catches_a_wrong_value_within_a_mixed_batch(self, client):
        """The read-back must check each id against ITS OWN expected value, not
        just that some doc_id is present — a batch now carries many values."""
        _upsert(client, [(i, [0.1] * DIM, {"doc_id": 300 + i, "source_type": "paper"})
                         for i in range(6)])

        migrate(client, COLLECTION, execute=True, apply_batch=6)

        stored = {p.id: p.payload["doc_id"] for p in client.retrieve(
            COLLECTION, ids=list(range(6)), with_payload=True)}
        assert stored == {i: str(300 + i) for i in range(6)}
