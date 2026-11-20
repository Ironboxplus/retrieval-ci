"""Incremental code jobs must not enumerate the project's paths in Qdrant for removals.

``CodeKBService._index_incremental`` ran ``list_paths`` (a path facet over every point of
the project) on every manifest-carrying job to compute ``existing - manifest``, a set the
client had just computed itself from the catalog inventory (``known - manifest``). With
``removed_paths_complete`` the client sends that set and the server checks it against the
catalog instead. Real Qdrant (scratch collections) + real SQLite catalog + the real job
queue; the recording client forwards every facet to the real server.

The legacy request (manifest only) is kept as the oracle: both protocols must leave the
same vectors and catalog behind.
"""
from __future__ import annotations

import time
import warnings
from contextlib import contextmanager

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.services.code_kb.jobs import CodeIndexJobQueue, JobStatus  # noqa: E402
from app.services.code_kb.schemas import (  # noqa: E402
    CodeChunkIn, CodeFileIn, CodeIndexReport, CodeIndexRequest,
)
from app.services.code_kb.service import CodeKBService  # noqa: E402
from app.services.code_kb.vector_store import CodeVectorStore  # noqa: E402
from app.services.index_catalog.repository import IndexCatalogRepository  # noqa: E402
from app.services.index_catalog.schemas import IndexFileRecord  # noqa: E402
from app.services.index_catalog.service import IndexCatalogService  # noqa: E402
from core.db.database import Base  # noqa: E402
from core.storage.qdrant_store import QdrantStore  # noqa: E402
from tests.qdrant_scratch import drop, scratch_names  # noqa: E402

LOCAL_URL = "http://127.0.0.1:6333"
OWNER = "removals-owner"
PROJECT = "C:/w/removals"
HOST = "testhost"
DIM = 1024


class _RecordingClient:
    def __init__(self, inner: QdrantClient):
        self._inner = inner
        self.path_facet_projects: list[str] = []

    def facet(self, *args, **kwargs):
        if kwargs["key"] == "path":
            for cond in (kwargs["facet_filter"].must or []):
                if cond.key == "project":
                    self.path_facet_projects.append(cond.match.value)
        return self._inner.facet(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Env:
    """One isolated (collection, catalog, queue) triple."""

    def __init__(self, real: QdrantClient, collection: str):
        self.real = real
        self.collection = collection
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

        repository = IndexCatalogRepository(session_factory=session_factory)
        self.catalog = IndexCatalogService(repository=repository)
        params = qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)
        from config import settings
        real.create_collection(
            collection_name=collection,
            vectors_config={"dense": params} if settings.embedding.supports_sparse else params)
        inner = QdrantStore.__new__(QdrantStore)
        inner.collection_name = collection
        self.client = _RecordingClient(real)
        inner.client = self.client

        class CatalogBackedStore(CodeVectorStore):
            def catalog_paths(self_inner, *, owner_user_id, project):
                return repository.list_code_paths(owner_user_id=owner_user_id, project=project)

        self.store = CatalogBackedStore(collection_name=collection, store=inner)
        self.store.ensure_search_payload_indexes()
        self.service = CodeKBService(embed_fn=_embed, vector_store=self.store)
        self.queue = CodeIndexJobQueue(service=self.service, catalog_service=self.catalog)

    def run(self, request: CodeIndexRequest) -> JobStatus:
        job = self.queue.submit(request, owner_user_id=OWNER)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            current = self.queue.get(job.job_id, owner_user_id=OWNER)
            if current.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                self.last_error = current.error
                return current.status
            time.sleep(0.1)
        raise AssertionError("index job did not finish")

    def stored(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        offset = None
        while True:
            rows, offset = self.real.scroll(
                self.collection, limit=500, offset=offset, with_payload=True,
                with_vectors=False)
            for row in rows:
                payload = row.payload or {}
                out.setdefault(payload["path"], set()).add(payload["content"])
            if offset is None:
                return out

    def catalog_paths(self) -> set[str]:
        return self.store.catalog_paths(owner_user_id=OWNER, project=PROJECT)


def _embed(texts):
    out = []
    for text in texts:
        row = [0.0] * DIM
        row[hash(text) % DIM] = 1.0
        out.append(row)
    return out


def _file(rel: str, body: str) -> CodeFileIn:
    return CodeFileIn(rel_path=rel, language="python", content_hash=f"h-{body}", chunks=[
        CodeChunkIn(start_line=0, end_line=1, label="f", content=body, code=body)])


def _record(rel: str, body: str) -> IndexFileRecord:
    return IndexFileRecord(rel_path=rel, kind="code", collection="code_chunks",
                           language="python", chunk_count=1, content_hash=f"h-{body}")


def _full(files: dict[str, str]) -> CodeIndexRequest:
    return CodeIndexRequest(
        project=PROJECT, host=HOST, root_dir=PROJECT,
        files=[_file(rel, body) for rel, body in files.items()],
        report=CodeIndexReport(files=[_record(r, b) for r, b in files.items()], replace=True))


def _incremental(changed: dict[str, str], tree: dict[str, str], *, known: set[str],
                 explicit: bool) -> CodeIndexRequest:
    removed = sorted(known - set(tree))
    return CodeIndexRequest(
        project=PROJECT, host=HOST, root_dir=PROJECT, incremental=True,
        files=[_file(rel, body) for rel, body in changed.items()],
        manifest=sorted(tree),
        removed_paths=removed if explicit else [],
        removed_paths_complete=explicit,
        report=CodeIndexReport(files=[_record(r, b) for r, b in tree.items()], replace=True))


@pytest.fixture()
def real():
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=30)
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    names = scratch_names("removals", roles=("a", "b"))
    envs: list[_Env] = []
    try:
        yield client, names, envs
    finally:
        for env in envs:
            env.queue.shutdown()
        drop(client, names.values())


def _new_env(real, which: str) -> _Env:
    client, names, envs = real
    env = _Env(client, names[which])
    envs.append(env)
    return env


BASE = {"a.py": "def a(): 1", "b.py": "def b(): 1", "c.py": "def c(): 1",
        "d.py": "def d(): 1", "e.py": "def e(): 1"}


def test_explicit_removals_leave_the_same_index_without_a_project_path_facet(real):
    legacy, explicit = _new_env(real, "a"), _new_env(real, "b")
    after = {"a.py": BASE["a.py"], "c.py": "def c(): 2", "d.py": BASE["d.py"],
             "e.py": BASE["e.py"], "f.py": "def f(): 1"}  # b removed, c changed, f new
    for env, flagged in ((legacy, False), (explicit, True)):
        assert env.run(_full(BASE)) is JobStatus.COMPLETED, env.last_error
        env.client.path_facet_projects.clear()
        request = _incremental(
            {"c.py": after["c.py"], "f.py": after["f.py"]}, after,
            known=env.catalog_paths(), explicit=flagged)
        assert env.run(request) is JobStatus.COMPLETED, env.last_error

    assert legacy.stored() == explicit.stored()
    assert set(legacy.stored()) == set(after)
    assert legacy.catalog_paths() == explicit.catalog_paths() == set(after)
    # legacy enumerated the project's own paths in Qdrant; explicit removals never do.
    assert PROJECT in legacy.client.path_facet_projects
    assert PROJECT not in explicit.client.path_facet_projects


def test_a_removal_only_job_with_explicit_removals_deletes_exactly_those(real):
    env = _new_env(real, "a")
    assert env.run(_full(BASE)) is JobStatus.COMPLETED, env.last_error
    tree = {k: v for k, v in BASE.items() if k not in {"b.py", "e.py"}}
    assert env.run(_incremental({}, tree, known=env.catalog_paths(), explicit=True)) \
        is JobStatus.COMPLETED, env.last_error
    assert set(env.stored()) == set(tree)
    assert env.catalog_paths() == set(tree)


def test_safety_valve_refuses_to_delete_every_cataloged_path(real):
    env = _new_env(real, "a")
    assert env.run(_full(BASE)) is JobStatus.COMPLETED, env.last_error
    wrong_tree = {"unrelated.py": "def u(): 1"}  # a broken walk / wrong scope
    request = _incremental(wrong_tree, wrong_tree, known=env.catalog_paths(), explicit=True)
    assert env.run(request) is JobStatus.FAILED
    assert "refusing" in (env.last_error or "")
    assert set(env.stored()) == set(BASE)  # nothing was deleted
    assert env.catalog_paths() == set(BASE)


def test_an_unsettled_scope_falls_back_to_reconciling_against_qdrant(real):
    env = _new_env(real, "a")
    assert env.run(_full(BASE)) is JobStatus.COMPLETED, env.last_error
    # A previous job died after upserting vectors its catalog report never recorded.
    orphan = CodeIndexRequest(
        project=PROJECT, host=HOST, root_dir=PROJECT, incremental=True,
        files=[_file("orphan.py", "def orphan(): 1")])
    env.service.index(orphan, owner_user_id=OWNER)
    assert "orphan.py" in env.stored() and "orphan.py" not in env.catalog_paths()
    env.catalog.mark_code_scope_recovery_required(
        owner_user_id=OWNER, host=HOST, project=PROJECT, job_id="dead-job")

    env.client.path_facet_projects.clear()
    request = _incremental({}, BASE, known=env.catalog_paths(), explicit=True)
    assert env.run(request) is JobStatus.COMPLETED, env.last_error

    assert "orphan.py" not in env.stored()  # only the Qdrant enumeration can find it
    assert PROJECT in env.client.path_facet_projects


def test_a_cleanly_completed_scope_trusts_the_catalog_baseline(real):
    env = _new_env(real, "a")
    assert env.run(_full(BASE)) is JobStatus.COMPLETED, env.last_error
    orphan = CodeIndexRequest(
        project=PROJECT, host=HOST, root_dir=PROJECT, incremental=True,
        files=[_file("orphan.py", "def orphan(): 1")])
    env.service.index(orphan, owner_user_id=OWNER)

    env.client.path_facet_projects.clear()
    assert env.run(_incremental({}, BASE, known=env.catalog_paths(), explicit=True)) \
        is JobStatus.COMPLETED, env.last_error

    assert PROJECT not in env.client.path_facet_projects
    assert "orphan.py" in env.stored()  # documented trade-off: the catalog is the baseline
