"""Real-Qdrant coverage for federated-search decision 10's `must_not` exclusion.

docs/federated_code_search_plan.md decision 10: an owner-wide federated search must
exclude individually-withheld scopes via a Qdrant `must_not` on `project`, not a
positive allow-list, because a scope indexed before its catalog marker existed has no
durable row and would be silently dropped by an allow-list. `CodeVectorStore.search`'s
`excluded_projects` param (app/services/code_kb/vector_store.py) is the mechanism; this
reuses `QdrantStore._build_filter`'s pre-existing reserved `"__exclude__"` key (the same
one `papers` already uses to drop review points), so what needs verifying here is that
the exclusion actually reaches Qdrant and does not silently no-op.

Skipped unless a Qdrant is reachable on loopback. Never touches the production
`code_chunks` collection.
"""
from __future__ import annotations

import time
import warnings

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from app.services.code_kb.schemas import CodeSearchRequest  # noqa: E402
from app.services.code_kb.service import CodeKBService  # noqa: E402
from app.services.code_kb.vector_store import CodeVectorStore  # noqa: E402
from core.code_kb.vector_records import CodeChunkVectorRecord  # noqa: E402
from core.storage.qdrant_store import QdrantStore  # noqa: E402

# Loopback only and hardcoded, never read from settings/QDRANT_URL -- see
# tests/test_code_duplicates_real_qdrant.py for why.
LOCAL_URL = "http://127.0.0.1:6333"

# A scratch name of our own, NOT `code_chunks`.
COLLECTION = "code_chunks_mustnot_test"
DIM = 8
OWNER = "mustnot-owner"

E = [[1.0 if i == j else 0.0 for i in range(DIM)] for j in range(DIM)]


def _drop_collection(client: QdrantClient, *, attempts: int) -> Exception | None:
    """Drop COLLECTION, tolerating the transient Windows rename failure. Mirrors
    tests/test_code_duplicates_real_qdrant.py's fixture (see its comment for why this
    is unconditional-drop-in-setup rather than skip-if-exists)."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            client.delete_collection(COLLECTION)
            return None
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt + 1 < attempts:
                time.sleep(0.5 * (attempt + 1))
    return last


@pytest.fixture()
def store():
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=10)
        client.get_collections()  # fail fast when Qdrant is not reachable
    except Exception as exc:
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")

    _drop_collection(client, attempts=4)

    from config import settings
    dense_is_named = settings.embedding.supports_sparse
    params = qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": params} if dense_is_named else params,
    )

    inner = QdrantStore.__new__(QdrantStore)
    inner.collection_name = COLLECTION
    inner.client = client

    cvs = CodeVectorStore(collection_name=COLLECTION, store=inner)
    cvs.ensure_search_payload_indexes()
    try:
        yield cvs
    finally:
        failure = _drop_collection(client, attempts=3)
        if failure is not None:
            warnings.warn(
                f"could not drop `{COLLECTION}` in teardown ({failure}); the next "
                "run's setup will clear it", stacklevel=1)


def _seed(cvs: CodeVectorStore, *, project: str, path: str, owner: str = OWNER) -> None:
    record = CodeChunkVectorRecord(
        rel_path=path, language="python", start_line=1, end_line=9,
        label="fn", content="x", code="x", chunk_ordinal=0)
    cvs.upsert(owner_user_id=owner, project=project, host="testhost",
               items=[record], dense_vectors=[E[0]])


def test_without_exclusion_both_scopes_are_reachable(store):
    """Positive control: nothing here silently narrows the result on its own."""
    _seed(store, project="/repo-a", path="a.py")
    _seed(store, project="/repo-b", path="b.py")

    hits = store.search(dense_vector=E[0], top_k=10, owner_user_id=OWNER)
    assert {h["project"] for h in hits} == {"/repo-a", "/repo-b"}


def test_excluded_projects_removes_only_the_named_scope(store):
    _seed(store, project="/repo-a", path="a.py")
    _seed(store, project="/repo-b", path="b.py")

    hits = store.search(
        dense_vector=E[0], top_k=10, owner_user_id=OWNER, excluded_projects=["/repo-b"])
    assert {h["project"] for h in hits} == {"/repo-a"}


def test_excluded_projects_with_no_positive_project_filter_is_owner_wide_must_not(store):
    """Decision 10's owner-wide branch: no positive `project` filter at all, only the
    `must_not`. Three scopes; two of them are excluded via one call."""
    _seed(store, project="/repo-a", path="a.py")
    _seed(store, project="/repo-b", path="b.py")
    _seed(store, project="/repo-c", path="c.py")

    hits = store.search(
        dense_vector=E[0], top_k=10, owner_user_id=OWNER,
        excluded_projects=["/repo-b", "/repo-c"])
    assert {h["project"] for h in hits} == {"/repo-a"}


def test_excluded_projects_does_not_leak_across_owners(store):
    """`must_not` on `project` must still respect the owner_user_id filter -- excluding
    one owner's scope name must not touch another owner's points of the same name."""
    _seed(store, project="/repo-a", path="a.py", owner=OWNER)
    _seed(store, project="/repo-a", path="a.py", owner="other-owner")

    hits = store.search(
        dense_vector=E[0], top_k=10, owner_user_id=OWNER, excluded_projects=["/repo-a"])
    assert hits == []

    other = store.search(dense_vector=E[0], top_k=10, owner_user_id="other-owner")
    assert {h["project"] for h in other} == {"/repo-a"}


def test_excluded_project_absent_from_the_catalog_is_still_excluded(store):
    """The whole point of `must_not` over an allow-list (decision 10): a legacy scope
    that predates any durable catalog row is excluded by NAME alone, with nothing else
    needed to make it invisible."""
    _seed(store, project="/legacy-pre-catalog", path="old.py")
    _seed(store, project="/repo-a", path="a.py")

    hits = store.search(
        dense_vector=E[0], top_k=10, owner_user_id=OWNER,
        excluded_projects=["/legacy-pre-catalog"])
    assert {h["project"] for h in hits} == {"/repo-a"}


def test_service_search_threads_excluded_projects_through_to_qdrant(store):
    """One level up: CodeKBService.search's excluded_projects param (used by the
    federated-search route) actually reaches Qdrant, not just the store's own API."""
    _seed(store, project="/repo-a", path="a.py")
    _seed(store, project="/repo-b", path="b.py")

    def embed(texts):
        return [E[0] for _ in texts]

    service = CodeKBService(embed_fn=embed, vector_store=store)
    response = service.search(
        CodeSearchRequest(query="q", top_k=10), owner_user_id=OWNER,
        excluded_projects=["/repo-b"])

    assert {hit.project for hit in response.hits} == {"/repo-a"}
