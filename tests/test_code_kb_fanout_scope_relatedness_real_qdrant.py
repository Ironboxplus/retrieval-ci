"""Cross-scope fan-out must only look at scopes that can hold the SAME physical file.

2026-11 production measurement: every index job began with `list_owner_files`, which
facets ALL of the owner's projects and then facets `path` for each of them -- 236
projects / 347,887 files, ~50 s -- so even a 1-chunk job sat at 0% for ~55 s. The inventory is now read from
the SQL catalog; these tests record which scopes' catalog paths were read. The
fan-out identity is `file_abspath(scope, rel)` and stored rel paths never contain `..`,
so two (scope, rel) pairs can resolve to one absolute path only when one scope equals
or is an ancestor of the other. Every other scope is irrelevant and must not be
path-faceted.

Real Qdrant on loopback, test-private scratch collection (unconditionally dropped and
recreated in setup). The store subclass only RECORDS which projects' catalog paths were read;
fan-out inventory now comes from the SQL catalog (no Qdrant facet).
"""
from __future__ import annotations

import time
import warnings

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from app.services.code_kb.schemas import CodeChunkIn, CodeFileIn, CodeIndexRequest  # noqa: E402
from app.services.code_kb.service import CodeKBService  # noqa: E402
from app.services.code_kb.vector_store import CodeVectorStore  # noqa: E402
from app.services.index_catalog.repository import IndexCatalogRepository  # noqa: E402
from core.storage.qdrant_store import QdrantStore  # noqa: E402
from tests.test_code_kb_fanout_catalog_inventory_real_qdrant import (  # noqa: E402
    _repository, _retarget_catalog, _write_catalog,
)

# Loopback only and hardcoded so a deploy host's config can never aim this at production.
LOCAL_URL = "http://127.0.0.1:6333"
COLLECTION = "code_chunks_fanout_relatedness_test"
OWNER = "fanout-owner"
DIM = 1024

def _drop_collection(client: QdrantClient, attempts: int = 4):
    failure = None
    for attempt in range(attempts):
        try:
            client.delete_collection(COLLECTION)
            return None
        except Exception as exc:  # noqa: BLE001
            failure = exc
            if attempt + 1 < attempts:
                time.sleep(0.4 * (attempt + 1))
    return failure


class _RecordingRepository(IndexCatalogRepository):
    """Real SQLite catalog that remembers which projects' paths were read."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reads: list[str] = []

    def list_code_paths(self, *, owner_user_id, project):
        self.reads.append(project)
        return super().list_code_paths(owner_user_id=owner_user_id, project=project)


@pytest.fixture()
def store():
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=30)
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")

    _drop_collection(client)
    from config import settings
    params = qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": params} if settings.embedding.supports_sparse else params,
    )
    inner = QdrantStore.__new__(QdrantStore)
    inner.collection_name = COLLECTION
    inner.client = client
    cvs = CodeVectorStore(collection_name=COLLECTION, store=inner,
                          catalog=_repository(_RecordingRepository))
    cvs.ensure_search_payload_indexes()
    try:
        yield cvs
    finally:
        failure = _drop_collection(client, attempts=3)
        if failure is not None:
            warnings.warn(f"could not drop `{COLLECTION}` in teardown ({failure}); "
                          "the next run's setup will clear it", stacklevel=1)


def _embed(texts):
    out = []
    for text in texts:
        row = [0.0] * DIM
        row[hash(text) % DIM] = 1.0
        out.append(row)
    return out


def _service(store: CodeVectorStore) -> CodeKBService:
    return CodeKBService(embed_fn=_embed, vector_store=store)


def _index(svc: CodeKBService, project: str, rel: str, body: str) -> None:
    chunk = CodeChunkIn(start_line=0, end_line=1, label="f", content=body, code=body)
    request = CodeIndexRequest(
        project=project, host="testhost", root_dir=project,
        files=[CodeFileIn(rel_path=rel, language="python", chunks=[chunk])])
    svc.index(request, owner_user_id=OWNER)
    _write_catalog(svc.vector_store.catalog, project, [rel], owner=OWNER)


def _stored_contents(store: CodeVectorStore, project: str, rel: str) -> set[str]:
    rows, _ = store.store.client.scroll(
        COLLECTION,
        scroll_filter=store.store._build_filter(
            {"owner_user_id": OWNER, "project": project, "path": rel}),
        limit=100, with_payload=True, with_vectors=False)
    return {str((row.payload or {}).get("content")) for row in rows}


def _retarget_project(store: CodeVectorStore, old: str, new: str) -> None:
    """Rewrite stored points to a legacy spelling of their scope.

    The request schema canonicalises Windows scopes, so a differently-cased stored
    scope can only come from older data; this reproduces that data directly.
    """
    store.store.client.set_payload(
        COLLECTION, payload={"project": new},
        points=qmodels.FilterSelector(filter=store.store._build_filter(
            {"owner_user_id": OWNER, "project": old})),
        wait=True)
    _retarget_catalog(store.catalog, old, new, owner=OWNER)


class TestRelatedScopesStillFanOut:

    def test_indexing_the_child_refreshes_the_parent_copy(self, store):
        svc = _service(store)
        _index(svc, "C:/w/erp", "sub/a.py", "def a(): return 'OLD'")
        assert _stored_contents(store, "C:/w/erp", "sub/a.py") == {"def a(): return 'OLD'"}

        _index(svc, "C:/w/erp/sub", "a.py", "def a(): return 'NEW'")

        assert _stored_contents(store, "C:/w/erp", "sub/a.py") == {"def a(): return 'NEW'"}
        assert _stored_contents(store, "C:/w/erp/sub", "a.py") == {"def a(): return 'NEW'"}

    def test_indexing_the_parent_refreshes_the_child_copy(self, store):
        svc = _service(store)
        _index(svc, "C:/w/erp/sub", "a.py", "def a(): return 'OLD'")

        _index(svc, "C:/w/erp", "sub/a.py", "def a(): return 'NEW'")

        assert _stored_contents(store, "C:/w/erp/sub", "a.py") == {"def a(): return 'NEW'"}
        assert _stored_contents(store, "C:/w/erp", "sub/a.py") == {"def a(): return 'NEW'"}

    def test_a_windows_case_variant_of_the_same_scope_is_related(self, store):
        # Indexing the canonical spelling itself merges legacy variants (its own
        # project filter unions them), so the variant's relatedness is asserted at the
        # discovery boundary fan-out consumes.
        svc = _service(store)
        _index(svc, "C:/w/erp/sub", "a.py", "def a(): return 'OLD'")
        _index(svc, "C:/w/erp-backup", "a.py", "def a(): return 'BACKUP'")
        _retarget_project(store, "C:/w/erp/sub", "c:/W/ERP/Sub")
        assert _stored_contents(store, "c:/W/ERP/Sub", "a.py") == {"def a(): return 'OLD'"}
        store.catalog.reads.clear()

        got = store.list_owner_files(owner_user_id=OWNER, related_to="C:/w/erp/sub")

        assert got == {("c:/W/ERP/Sub", "a.py")}
        assert store.catalog.reads == ["c:/W/ERP/Sub"]

    def test_a_case_variant_ancestor_is_related(self, store):
        svc = _service(store)
        _index(svc, "C:/w/erp", "sub/a.py", "def a(): return 'OLD'")
        _retarget_project(store, "C:/w/erp", "c:/W/ERP")

        _index(svc, "C:/w/erp/sub", "a.py", "def a(): return 'NEW'")

        assert _stored_contents(store, "c:/W/ERP", "sub/a.py") == {"def a(): return 'NEW'"}


class TestUnrelatedScopesAreNotEnumerated:

    def test_siblings_and_unrelated_scopes_are_not_path_faceted(self, store):
        svc = _service(store)
        _index(svc, "C:/w/erp", "sub/a.py", "def a(): return 'OLD'")
        _index(svc, "C:/w/erp/sub", "a.py", "def a(): return 'OLD'")
        # Shares a string prefix with the target but is a different directory.
        _index(svc, "C:/w/erp-backup", "sub/a.py", "def a(): return 'BACKUP'")
        _index(svc, "D:/other", "sub/a.py", "def a(): return 'OTHER'")
        _index(svc, "/srv/unrelated", "sub/a.py", "def a(): return 'UNIX'")
        store.catalog.reads.clear()

        _index(svc, "C:/w/erp/sub", "a.py", "def a(): return 'NEW'")

        faceted = set(store.catalog.reads)
        # The job's own scope is related to itself but its files are discarded, so it is
        # not faceted either (tests/test_code_kb_fanout_own_scope_not_enumerated_real_qdrant.py).
        assert faceted == {"C:/w/erp"}, faceted
        # The unrelated copies are untouched, and the related parent was refreshed.
        assert _stored_contents(store, "C:/w/erp", "sub/a.py") == {"def a(): return 'NEW'"}
        assert _stored_contents(store, "C:/w/erp-backup", "sub/a.py") == {
            "def a(): return 'BACKUP'"}
        assert _stored_contents(store, "D:/other", "sub/a.py") == {"def a(): return 'OTHER'"}
        assert _stored_contents(store, "/srv/unrelated", "sub/a.py") == {
            "def a(): return 'UNIX'"}

    def test_path_facet_count_does_not_grow_with_unrelated_scope_count(self, store):
        svc = _service(store)
        _index(svc, "C:/w/erp", "sub/a.py", "def a(): return 'OLD'")
        for n in range(12):
            _index(svc, f"C:/noise/proj{n}", "x.py", f"def x{n}(): return {n}")
        store.catalog.reads.clear()

        _index(svc, "C:/w/erp/sub", "a.py", "def a(): return 'NEW'")

        # The indexed child has no points yet when candidates are computed (full
        # index discovers before it writes), so only the parent is enumerated.
        assert store.catalog.reads == ["C:/w/erp"]
        assert _stored_contents(store, "C:/w/erp", "sub/a.py") == {"def a(): return 'NEW'"}


class TestListOwnerFilesContract:

    def test_without_related_to_every_project_is_returned(self, store):
        svc = _service(store)
        _index(svc, "C:/w/erp", "sub/a.py", "def a(): return 1")
        _index(svc, "C:/w/erp-backup", "b.py", "def b(): return 2")
        _index(svc, "/srv/unrelated", "c.py", "def c(): return 3")
        store.catalog.reads.clear()

        assert store.list_owner_files(owner_user_id=OWNER) == {
            ("C:/w/erp", "sub/a.py"),
            ("C:/w/erp-backup", "b.py"),
            ("/srv/unrelated", "c.py"),
        }
        assert sorted(store.catalog.reads) == [
            "/srv/unrelated", "C:/w/erp", "C:/w/erp-backup"]

    def test_related_to_returns_only_the_ancestor_chain(self, store):
        svc = _service(store)
        _index(svc, "C:/w/erp", "sub/a.py", "def a(): return 1")
        _index(svc, "C:/w/erp/sub", "a.py", "def a(): return 1")
        _index(svc, "C:/w/erp/sub/deeper", "z.py", "def z(): return 1")
        _index(svc, "C:/w/erp-backup", "b.py", "def b(): return 2")
        _index(svc, "C:/w/erp/other", "o.py", "def o(): return 4")

        got = store.list_owner_files(owner_user_id=OWNER, related_to="C:/w/erp/sub")

        assert got == {
            ("C:/w/erp", "sub/a.py"),
            ("C:/w/erp/sub", "a.py"),
            ("C:/w/erp/sub/deeper", "z.py"),
        }

    def test_a_malformed_stored_scope_is_skipped_rather_than_failing_the_listing(self, store):
        svc = _service(store)
        _index(svc, "C:/w/erp", "a.py", "def a(): return 1")
        _index(svc, "C:/w/erp-x", "tmp.py", "def t(): return 1")
        _retarget_project(store, "C:/w/erp-x", "E:drive-relative")

        got = store.list_owner_files(owner_user_id=OWNER, related_to="C:/w/erp")

        assert got == {("C:/w/erp", "a.py")}
