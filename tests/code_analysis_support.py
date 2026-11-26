"""Real-store harness for the duplicate-code analysis tests.

A real Qdrant on loopback (private, uniquely named scratch collection per test, so
xdist workers never share one), a real SQLite file holding the application's tables, the
real CodeKBService / CodeIndexJobQueue / IndexCatalogService on top. The only thing
injected is the embedding function, which maps a chunk's text to a hand-built unit
vector so "these two chunks are duplicates" is an exact property of the fixture, not of
an embedding model.

Chunk text is ``basis N``: every chunk whose text ends in the same N has the same
vector (cosine 1.0); different N are orthogonal (0.0).
"""
from __future__ import annotations

import hashlib
import time
import uuid
import warnings
from dataclasses import dataclass

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from app.services.code_kb.jobs import CodeIndexJobQueue, JobStatus  # noqa: E402
from app.services.code_kb.schemas import (  # noqa: E402
    CodeChunkIn, CodeFileIn, CodeIndexReport, CodeIndexRequest)
from app.services.code_kb.service import CodeKBService  # noqa: E402
from app.services.code_kb.vector_store import CodeVectorStore  # noqa: E402
from app.services.index_catalog.repository import IndexCatalogRepository  # noqa: E402
from app.services.index_catalog.schemas import IndexFileRecord  # noqa: E402
from app.services.index_catalog.service import IndexCatalogService  # noqa: E402
from app.services.pipeline.queue_tasks import QueueTasks  # noqa: E402
from app.services.pipeline.store import TaskStore  # noqa: E402
from core.storage.qdrant_store import QdrantStore  # noqa: E402
from tests.sqlite_task_db import make_session_factory  # noqa: E402

# Loopback only and hardcoded, never read from settings, so a deploy host's config cannot
# aim these tests at a production collection.
LOCAL_URL = "http://127.0.0.1:6333"
DIM = 8
OWNER = "analysis-owner"
PROJECT = "/repo"

E = [[1.0 if i == j else 0.0 for i in range(DIM)] for j in range(DIM)]


def embed(texts):
    return [E[int(text.split()[-1])] for text in texts]


def _drop_collection(client: QdrantClient, name: str, *, attempts: int):
    failure = None
    for attempt in range(attempts):
        try:
            client.delete_collection(name)
            return None
        except Exception as exc:  # noqa: BLE001 - Windows 500s while segments are mapped
            failure = exc
            if attempt + 1 < attempts:
                time.sleep(0.4 * (attempt + 1))
    return failure


@pytest.fixture()
def vector_store():
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=10)
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")

    name = f"code_chunks_analysis_{uuid.uuid4().hex[:12]}"
    from config import settings
    params = qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": params} if settings.embedding.supports_sparse else params)
    inner = QdrantStore.__new__(QdrantStore)
    inner.collection_name = name
    inner.client = client
    cvs = CodeVectorStore(collection_name=name, store=inner)
    cvs.ensure_search_payload_indexes()
    try:
        yield cvs
    finally:
        failure = _drop_collection(client, name, attempts=3)
        if failure is not None:
            warnings.warn(f"could not drop `{name}` in teardown ({failure})", stacklevel=1)


@dataclass
class Env:
    vector_store: CodeVectorStore
    service: CodeKBService
    catalog: IndexCatalogService
    task_store: TaskStore
    tasks: QueueTasks
    session_factory: object
    queue: CodeIndexJobQueue

    def index(self, project: str, files: dict[str, list[int]] | None = None, *,
              removed: list[str] | None = None, owner: str = OWNER, timeout: float = 20.0):
        """Run one incremental index job to completion, exactly as the MCP client does:
        changed files' chunks plus a catalog report whose hash follows the content."""
        request = make_request(project, files or {}, removed or [])
        job = self.queue.submit(request, owner_user_id=owner)
        deadline = time.time() + timeout
        while time.time() < deadline:
            current = self.queue.get(job.job_id)
            if current.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                assert current.status is JobStatus.COMPLETED, current.error
                return current
            time.sleep(0.01)
        raise AssertionError("index job did not finish")


def make_request(project: str, files: dict[str, list[int]], removed: list[str]) -> CodeIndexRequest:
    file_models, records = [], []
    for path, basis in files.items():
        chunks = [
            CodeChunkIn(start_line=ordinal * 10 + 1, end_line=ordinal * 10 + 10,
                        label=f"chunk{ordinal}", content=f"basis {b}", code=f"basis {b}")
            for ordinal, b in enumerate(basis)]
        digest = hashlib.sha256(",".join(map(str, basis)).encode()).hexdigest()
        file_models.append(CodeFileIn(
            rel_path=path, language="python", content_hash=digest, chunks=chunks))
        records.append(IndexFileRecord(
            rel_path=path, kind="code", collection="code_chunks", language="python",
            chunk_count=len(chunks), content_hash=digest))
    return CodeIndexRequest(
        project=project, host="host", root_dir=project, files=file_models,
        incremental=True, removed_paths=removed,
        report=CodeIndexReport(files=records, replace=False))


@pytest.fixture()
def env(vector_store, tmp_path):
    factory = make_session_factory(tmp_path / "analysis.db")
    catalog = IndexCatalogService(repository=IndexCatalogRepository(session_factory=factory))
    service = CodeKBService(embed_fn=embed, vector_store=vector_store)
    task_store = TaskStore(factory)
    tasks = QueueTasks(task_store, min_interval=0.0)
    queue = CodeIndexJobQueue(service=service, catalog_service=catalog, tasks=tasks)
    yield Env(vector_store=vector_store, service=service, catalog=catalog,
              task_store=task_store, tasks=tasks, session_factory=factory, queue=queue)
    queue.shutdown(timeout=5)
    factory.engine.dispose()


class SlowQueryClient:
    """Wraps the REAL Qdrant client and delays/observes each batched neighbour query.

    Every query still runs in Qdrant; the delay makes a scan last long enough for a
    test to measure what other threads can do meanwhile, and ``on_query`` lets a test
    perform a real write at an exact point of the scan."""

    def __init__(self, inner, *, delay: float = 0.0, on_query=None):
        self._inner = inner
        self._delay = delay
        self._on_query = on_query
        self.queries = 0

    def query_batch_points(self, *args, **kwargs):
        self.queries += 1
        if self._on_query is not None:
            self._on_query(self.queries)
        if self._delay:
            time.sleep(self._delay)
        return self._inner.query_batch_points(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)
