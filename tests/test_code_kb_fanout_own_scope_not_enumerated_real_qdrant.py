"""A code job must not read its OWN scope's file list just to discard the result.

``CodeKBService._duplicate_candidates`` builds the cross-scope fan-out targets for every
code job, including 1-file watcher jobs that run inside the owner's search fence. The
job's own scope always passes ``scopes_share_files`` (it equals itself), so its complete
path set was faceted from Qdrant and then dropped (``if proj2 == project: continue``) --
for an already-indexed project, the largest facet of the job and the only one with no
possible use.

Real Qdrant on loopback, test-private scratch collection (dropped and recreated in
setup). The recording catalog repository forwards every read to real SQLite.
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

LOCAL_URL = "http://127.0.0.1:6333"
COLLECTION = "code_chunks_own_scope_facet_test"
OWNER = "own-scope-owner"
DIM = 1024


class _RecordingRepository(IndexCatalogRepository):
    """Real SQLite catalog that remembers which projects' paths were read."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reads: list[str] = []

    def list_code_paths(self, *, owner_user_id, project):
        self.reads.append(project)
        return super().list_code_paths(owner_user_id=owner_user_id, project=project)


def _drop(client: QdrantClient, attempts: int = 4):
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


@pytest.fixture()
def store():
    try:
        real = QdrantClient(url=LOCAL_URL, timeout=30)
        real.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    _drop(real)
    from config import settings
    params = qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)
    real.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": params} if settings.embedding.supports_sparse else params,
    )
    inner = QdrantStore.__new__(QdrantStore)
    inner.collection_name = COLLECTION
    inner.client = real
    cvs = CodeVectorStore(collection_name=COLLECTION, store=inner,
                          catalog=_repository(_RecordingRepository))
    cvs.ensure_search_payload_indexes()
    try:
        yield cvs
    finally:
        failure = _drop(real, attempts=3)
        if failure is not None:
            warnings.warn(f"could not drop `{COLLECTION}` in teardown ({failure})", stacklevel=1)


def _embed(texts):
    out = []
    for text in texts:
        row = [0.0] * DIM
        row[hash(text) % DIM] = 1.0
        out.append(row)
    return out


def _index(svc, project, files, *, incremental=False, removed=()):
    request = CodeIndexRequest(
        project=project, host="testhost", root_dir=project, incremental=incremental,
        removed_paths=list(removed),
        files=[CodeFileIn(rel_path=rel, language="python", chunks=[
            CodeChunkIn(start_line=0, end_line=1, label="f", content=body, code=body)])
            for rel, body in files.items()])
    response = svc.index(request, owner_user_id=OWNER)
    _write_catalog(svc.vector_store.catalog, project, list(files), owner=OWNER)
    return response


def _contents(store, project, rel):
    rows, _ = store.store.client.scroll(
        COLLECTION,
        scroll_filter=store.store._build_filter(
            {"owner_user_id": OWNER, "project": project, "path": rel}),
        limit=100, with_payload=True, with_vectors=False)
    return {str((row.payload or {}).get("content")) for row in rows}


def test_list_owner_files_can_skip_one_project_without_changing_the_rest(store):
    svc = CodeKBService(embed_fn=_embed, vector_store=store)
    _index(svc, "C:/w/erp", {"sub/a.py": "A", "sub/b.py": "B"})
    _index(svc, "C:/w/erp/sub", {"a.py": "A"})
    full = store.list_owner_files(owner_user_id=OWNER, related_to="C:/w/erp/sub")
    store.catalog.reads.clear()

    skipped = store.list_owner_files(
        owner_user_id=OWNER, related_to="C:/w/erp/sub", exclude_project="C:/w/erp/sub")

    assert skipped == {pair for pair in full if pair[0] != "C:/w/erp/sub"}
    assert store.catalog.reads == ["C:/w/erp"]


def test_watcher_sized_job_on_an_indexed_project_never_facets_its_own_paths(store):
    svc = CodeKBService(embed_fn=_embed, vector_store=store)
    own = "C:/w/erp/sub"
    _index(svc, "C:/w/erp", {"sub/a.py": "def a(): return 'OLD'"})
    _index(svc, own, {f"mod{n}.py": f"def m{n}(): return {n}" for n in range(30)}
           | {"a.py": "def a(): return 'OLD'"})
    for n in range(5):
        _index(svc, f"C:/noise/proj{n}", {"x.py": f"def x{n}(): return {n}"})
    store.catalog.reads.clear()

    _index(svc, own, {"a.py": "def a(): return 'NEW'"}, incremental=True)

    # The own scope (30+ files) is never enumerated; the related ancestor still is, and
    # fan-out refreshed its copy exactly as before.
    assert store.catalog.reads == ["C:/w/erp"]
    assert _contents(store, "C:/w/erp", "sub/a.py") == {"def a(): return 'NEW'"}
    assert _contents(store, own, "a.py") == {"def a(): return 'NEW'"}


def test_removed_file_still_fans_out_to_the_related_scope(store):
    svc = CodeKBService(embed_fn=_embed, vector_store=store)
    own = "C:/w/erp/sub"
    _index(svc, "C:/w/erp", {"sub/a.py": "A", "sub/keep.py": "K"})
    _index(svc, own, {"a.py": "A", "keep.py": "K"})

    _index(svc, own, {"keep.py": "K"}, incremental=True, removed=["a.py"])

    assert _contents(store, "C:/w/erp", "sub/a.py") == set()
    assert _contents(store, "C:/w/erp", "sub/keep.py") == {"K"}


def test_case_variant_of_the_own_scope_is_still_enumerated(store):
    svc = CodeKBService(embed_fn=_embed, vector_store=store)
    own = "C:/w/erp/sub"
    _index(svc, own, {"a.py": "OLD"})
    store.store.client.set_payload(
        COLLECTION, payload={"project": "c:/W/ERP/Sub"},
        points=qmodels.FilterSelector(filter=store.store._build_filter(
            {"owner_user_id": OWNER, "project": own})),
        wait=True)
    _retarget_catalog(store.catalog, own, "c:/W/ERP/Sub", owner=OWNER)
    store.catalog.reads.clear()

    got = store.list_owner_files(
        owner_user_id=OWNER, related_to=own, exclude_project=own)

    assert got == {("c:/W/ERP/Sub", "a.py")}
    assert store.catalog.reads == ["c:/W/ERP/Sub"]
