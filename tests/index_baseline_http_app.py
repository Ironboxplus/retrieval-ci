"""Small real FastAPI process for tests/test_index_baseline_http_real.py.

It registers the production Index Catalog and Code KB routes on the production SQLite
schema (``DATA_DIR`` points at a temp directory) and writes every request it receives
to ``REQUEST_LOG`` so a test can count the real HTTP traffic of an MCP client.

Two things differ from ``main`` and both are about isolation, not behaviour:

* the Code KB job queue is built over a scratch Qdrant collection (``E2E_CODE_COLLECTION``)
  with a deterministic hashing embedder instead of the shared BGE-M3 model, the same
  arrangement the ``*_real_qdrant.py`` tests use;
* ``CodeVectorStore.list_paths`` -- the whole-project path facet -- is wrapped to log
  each call; the wrapper forwards to the real method.

``/api/journal/docs/reconcile`` answers what the Journal answers for a scope without
stale documents; the Journal store itself is not part of what these tests measure.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from core.db.database import init_db

REQUEST_LOG = os.environ["REQUEST_LOG"]
DIM = 1024


def _log(entry: dict) -> None:
    with open(REQUEST_LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def _embed(texts):
    out = []
    for text in texts:
        row = [0.0] * DIM
        row[hash(text) % DIM] = 1.0
        out.append(row)
    return out


def _install_isolated_code_queue(collection: str) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

    from app.services.code_kb import jobs
    from app.services.code_kb.service import CodeKBService
    from app.services.code_kb.vector_store import CodeVectorStore
    from app.services.index_catalog.service import get_index_catalog_service
    from config import settings
    from core.storage.qdrant_store import QdrantStore

    client = QdrantClient(url="http://127.0.0.1:6333", timeout=30)
    params = qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)
    client.create_collection(
        collection_name=collection,
        vectors_config={"dense": params} if settings.embedding.supports_sparse else params)
    inner = QdrantStore.__new__(QdrantStore)
    inner.collection_name = collection
    inner.client = client
    store = CodeVectorStore(collection_name=collection, store=inner)
    store.ensure_search_payload_indexes()

    original = CodeVectorStore.list_paths

    def logged_list_paths(self, *args, **kwargs):
        _log({"event": "qdrant_list_paths", "project": kwargs.get("project")})
        return original(self, *args, **kwargs)

    CodeVectorStore.list_paths = logged_list_paths
    service = CodeKBService(embed_fn=_embed, vector_store=store)
    jobs._QUEUE = jobs.CodeIndexJobQueue(
        service=service, catalog_service=get_index_catalog_service())


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    collection = os.environ.get("E2E_CODE_COLLECTION")
    if collection:
        _install_isolated_code_queue(collection)
    yield


app = FastAPI(title="index-baseline-http", lifespan=lifespan)


@app.middleware("http")
async def record_requests(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    if not request.url.path.startswith("/__"):
        _log({"event": "request", "method": request.method, "path": request.url.path,
              "query": request.url.query, "status": response.status_code,
              "length": int(response.headers.get("content-length") or 0),
              "ms": round((time.perf_counter() - started) * 1000, 1)})
    return response


@app.get("/__ready")
async def ready():
    return {"ok": True}


@app.post("/api/journal/docs/reconcile")
async def reconcile_docs(request: Request):
    body = await request.json()
    return {"project_slug": body.get("project_slug"), "removed_entries": 0, "skipped": False}


from app.api.routes.code_kb import router as code_kb_router  # noqa: E402
from app.api.routes.index_catalog import router as index_catalog_router  # noqa: E402

app.include_router(index_catalog_router, prefix="/api/index")
app.include_router(code_kb_router, prefix="/api/code")
