"""Cross-scope fan-out takes its file inventory from the SQL catalog, never a Qdrant facet.

Production telemetry (.88, since 2026-10-16): the fan-out/full-rebuild path facets
(owner-wide project facet ~97 s, per-project path facet 4-33 s) dominated Qdrant time.
The catalog already lists the same projects and paths; these tests pin that fan-out and
full rebuild use it, that catalog rows without points are never recreated, and that a
failed enumeration skips fan-out once per job instead of once per batch.

Real Qdrant on loopback (test-private scratch collection) + real in-memory SQLite
catalog. The recording client forwards every call to the real client and counts facets.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.services.code_kb.schemas import CodeChunkIn, CodeFileIn, CodeIndexRequest  # noqa: E402
from app.services.code_kb.service import EMBED_BATCH, CodeKBService  # noqa: E402
from app.services.code_kb.vector_store import CodeVectorStore  # noqa: E402
from app.services.index_catalog.repository import IndexCatalogRepository  # noqa: E402
from app.services.index_catalog.schemas import IndexFileRecord, IndexReportRequest  # noqa: E402
from app.services.index_catalog.service import IndexCatalogService  # noqa: E402
from core.db.database import Base  # noqa: E402
from core.storage.qdrant_store import QdrantStore  # noqa: E402
from tests.qdrant_scratch import drop, scratch_names  # noqa: E402

LOCAL_URL = "http://127.0.0.1:6333"
OWNER = "catalog-inventory-owner"
HOST = "testhost"
DIM = 1024


class _FacetCountingClient:
    def __init__(self, inner: QdrantClient):
        self._inner = inner
        self.facets = 0

    def facet(self, *args, **kwargs):
        self.facets += 1
        return self._inner.facet(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _repository(cls=IndexCatalogRepository) -> IndexCatalogRepository:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_factory():
        session = session_local()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return cls(session_factory=session_factory)


def _write_catalog(repository, project, rels, collection="code_chunks", owner=OWNER):
    """What a successful job's ``_write_catalog`` does: one live row per file."""
    IndexCatalogService(repository=repository).report(IndexReportRequest(
        host=HOST, project=project, root_dir=project, replace=False,
        files=[IndexFileRecord(rel_path=r, kind="code", collection=collection,
                               language="python", chunk_count=1) for r in rels]),
        owner_user_id=owner)


def _retarget_catalog(repository, old, new, owner=OWNER):
    """Rename a scope's catalog rows, mirroring a payload `project` rewrite."""
    from core.db.models import IndexCatalog
    with repository.session() as session:
        session.query(IndexCatalog).filter(
            IndexCatalog.owner_user_id == owner, IndexCatalog.project == old,
        ).update({"project": new})


class _Env:
    def __init__(self, real: QdrantClient, collection: str, store_cls=CodeVectorStore):
        self.real = real
        self.collection = collection
        self.repository = _repository()
        self.catalog = IndexCatalogService(repository=self.repository)
        params = qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)
        from config import settings
        real.create_collection(
            collection_name=collection,
            vectors_config={"dense": params} if settings.embedding.supports_sparse else params)
        inner = QdrantStore.__new__(QdrantStore)
        inner.collection_name = collection
        self.client = _FacetCountingClient(real)
        inner.client = self.client
        self.store = store_cls(collection_name=collection, store=inner,
                               catalog=self.repository)
        self.store.ensure_search_payload_indexes()
        self.service = CodeKBService(embed_fn=_embed, vector_store=self.store)

    def index(self, project, files, *, incremental=False):
        """Index like a job does: vectors first, catalog row after success."""
        request = CodeIndexRequest(
            project=project, host=HOST, root_dir=project, incremental=incremental,
            files=[CodeFileIn(rel_path=rel, language="python", chunks=[
                CodeChunkIn(start_line=n, end_line=n + 1, label=f"f{n}", content=body,
                            code=body)
                for n, body in enumerate(bodies if isinstance(bodies, list) else [bodies])])
                for rel, bodies in files.items()])
        response = self.service.index(request, owner_user_id=OWNER)
        self.write_catalog(project, list(files))
        return response

    def write_catalog(self, project, rels):
        _write_catalog(self.repository, project, rels, self.collection)

    def contents(self, project, rel) -> set[str]:
        rows, _ = self.real.scroll(
            self.collection,
            scroll_filter=self.store.store._build_filter(
                {"owner_user_id": OWNER, "project": project, "path": rel}),
            limit=1000, with_payload=True, with_vectors=False)
        return {str((row.payload or {}).get("content")) for row in rows}


def _embed(texts):
    out = []
    for text in texts:
        row = [0.0] * DIM
        row[hash(text) % DIM] = 1.0
        out.append(row)
    return out


@pytest.fixture()
def real():
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=30)
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    names = scratch_names("catinv", roles=("c",))
    try:
        yield client, names["c"]
    finally:
        drop(client, names.values())


def test_fan_out_finds_the_parent_copy_from_the_catalog_without_a_facet(real):
    env = _Env(*real)
    env.index("C:/w/erp", {"sub/a.py": "OLD"})
    env.index("C:/w/erp/sub", {"a.py": "OLD"})
    env.client.facets = 0

    env.index("C:/w/erp/sub", {"a.py": "NEW"}, incremental=True)

    assert env.contents("C:/w/erp", "sub/a.py") == {"NEW"}
    assert env.contents("C:/w/erp/sub", "a.py") == {"NEW"}
    assert env.client.facets == 0


def test_catalog_row_without_points_is_not_recreated(real):
    env = _Env(*real)
    env.write_catalog("C:/w/erp", ["sub/a.py"])  # catalog row, no points

    env.index("C:/w/erp/sub", {"a.py": "NEW"})

    assert env.contents("C:/w/erp", "sub/a.py") == set()
    assert env.contents("C:/w/erp/sub", "a.py") == {"NEW"}


def test_full_reindex_fans_out_a_removed_file_without_a_facet(real):
    env = _Env(*real)
    env.index("C:/w/erp", {"sub/a.py": "A", "sub/b.py": "B"})
    env.index("C:/w/erp/sub", {"a.py": "A", "b.py": "B"})
    env.client.facets = 0

    env.index("C:/w/erp/sub", {"a.py": "A"})  # full rebuild, b.py gone

    assert env.contents("C:/w/erp", "sub/b.py") == set()
    assert env.contents("C:/w/erp", "sub/a.py") == {"A"}
    assert env.contents("C:/w/erp/sub", "b.py") == set()
    assert env.client.facets == 0


def test_enumeration_failure_skips_fan_out_once_not_per_batch(real):
    calls = {"n": 0}

    class _FailingEnumeration(CodeVectorStore):
        def list_owner_files(self, **kwargs):
            calls["n"] += 1
            raise RuntimeError("injected enumeration failure")

    env = _Env(*real, store_cls=_FailingEnumeration)
    per_file = EMBED_BATCH // 2 + 6  # 70: two files exceed one batch, so >= 2 batches
    files = {f"m{k}.py": [f"def f{k}_{n}(): return {n}" for n in range(per_file)]
             for k in range(3)}

    response = env.index("C:/w/erp/sub", files)

    assert response.indexed == 3 * per_file
    assert env.contents("C:/w/erp/sub", "m0.py")
    assert calls["n"] == 1


class _NoQdrant:
    def __getattr__(self, name):
        raise AssertionError(f"Qdrant accessed: {name}")


def test_list_owner_files_reads_only_the_catalog():
    repository = _repository()
    for project, rels in (("/repo", ["a.py"]), ("/repo/child", ["b.py"]),
                          ("/elsewhere", ["c.py"])):
        _write_catalog(repository, project, rels)
    store = CodeVectorStore(store=_NoQdrant(), catalog=repository)

    got = store.list_owner_files(owner_user_id=OWNER, related_to="/repo/child")
    assert got == {("/repo", "a.py"), ("/repo/child", "b.py")}

    got = store.list_owner_files(owner_user_id=OWNER, related_to="/repo/child",
                                 exclude_project="/repo/child")
    assert got == {("/repo", "a.py")}
