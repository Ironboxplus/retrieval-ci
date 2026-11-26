"""Deliver-once scope notices: the server decides, the client keeps no state.

The background analysis (app/services/code_kb/analysis.py) stores a summary with a
version. `/api/code/search`, `/api/code/jobs/{id}` and `/api/journal/search` attach it to
the first response that reaches the scope after a new version appears, and to no other:
a conditional UPDATE of `delivered_version` decides, so two concurrent requests, or two
clients on two machines, can never both get it. Real SQLite throughout; code search runs
on a real Qdrant scratch collection through the same route functions the handlers call.
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
import warnings
from datetime import datetime

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import event, text

from app.api.routes import code_kb as code_routes
from app.api.routes import journal as journal_routes
from app.services.code_kb.analysis_store import CodeAnalysisStore, claim_scope_notice
from app.services.code_kb.jobs import CodeIndexJobQueue, JobStatus
from app.services.code_kb.schemas import CodeSearchRequest, FederatedCodeSearchRequest
from app.services.journal.repository import JournalRepository
from app.services.journal.schemas import JournalRecordRequest
from app.services.journal.service import JournalService
from app.services.journal.vector_store import JournalVectorStore
from tests.catalog_sqlite import CatalogDb
from tests.code_analysis_support import (
    OWNER, embed, make_request, vector_store)  # noqa: F401

NOW = datetime(2026, 11, 9, 12, 0, 0)


def _publish(store: CodeAnalysisStore, project: str, message: str, *, owner: str = OWNER) -> int:
    return store.apply_run(owner, project, touched_paths=[], pairs=[], watermark=NOW,
                           summary={"text": message}, now=NOW)


def _delivered(db: CatalogDb, project: str) -> int:
    with db.engine.connect() as conn:
        return conn.execute(text(
            "SELECT delivered_version FROM code_scope_analyses WHERE project = :p"),
            {"p": project}).scalar_one()


# ------------------------------------------------------------------ the claim itself

def test_the_first_claim_gets_the_summary_and_the_second_gets_nothing(tmp_path):
    db = CatalogDb(tmp_path / "notice.db")
    store = CodeAnalysisStore(db.session)
    _publish(store, "/repo", "Duplicate code found: a <-> b")

    assert claim_scope_notice(OWNER, "/repo", store=store) == "Duplicate code found: a <-> b"
    assert claim_scope_notice(OWNER, "/repo", store=store) is None
    assert _delivered(db, "/repo") == 1


def test_a_newer_version_is_delivered_again_and_only_once(tmp_path):
    db = CatalogDb(tmp_path / "notice.db")
    store = CodeAnalysisStore(db.session)
    _publish(store, "/repo", "first")
    assert claim_scope_notice(OWNER, "/repo", store=store) == "first"

    _publish(store, "/repo", "second")
    assert claim_scope_notice(OWNER, "/repo", store=store) == "second"
    assert claim_scope_notice(OWNER, "/repo", store=store) is None


def test_a_run_with_nothing_to_say_does_not_re_deliver_the_old_summary(tmp_path):
    db = CatalogDb(tmp_path / "notice.db")
    store = CodeAnalysisStore(db.session)
    _publish(store, "/repo", "first")
    claim_scope_notice(OWNER, "/repo", store=store)
    store.apply_run(OWNER, "/repo", touched_paths=[], pairs=[], watermark=NOW,
                    summary=None, now=NOW)
    assert claim_scope_notice(OWNER, "/repo", store=store) is None


def test_concurrent_requests_for_one_result_exactly_one_gets_it(tmp_path):
    db = CatalogDb(tmp_path / "notice.db")  # a file: every thread gets its own connection
    store = CodeAnalysisStore(db.session)
    _publish(store, "/repo", "only once")
    workers = 12
    barrier = threading.Barrier(workers)
    got: list[str | None] = []
    lock = threading.Lock()

    def request() -> None:
        barrier.wait()
        notice = claim_scope_notice(OWNER, "/repo", store=store)
        with lock:
            got.append(notice)

    threads = [threading.Thread(target=request) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    assert len(got) == workers
    assert got.count("only once") == 1 and got.count(None) == workers - 1


def test_a_scope_with_nothing_undelivered_costs_one_indexed_read(tmp_path):
    db = CatalogDb(tmp_path / "notice.db")
    store = CodeAnalysisStore(db.session)
    _publish(store, "/repo", "x")
    claim_scope_notice(OWNER, "/repo", store=store)

    statements: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(db.engine, "before_cursor_execute", record)
    try:
        assert claim_scope_notice(OWNER, "/repo", store=store) is None
        assert claim_scope_notice(OWNER, "/never-analysed", store=store) is None
    finally:
        event.remove(db.engine, "before_cursor_execute", record)

    assert len(statements) == 2, statements
    assert all(s.lstrip().upper().startswith("SELECT") for s in statements)
    with db.engine.connect() as conn:
        plan = conn.exec_driver_sql(
            "EXPLAIN QUERY PLAN SELECT version FROM code_scope_analyses WHERE "
            "owner_user_id = ? AND project = ? AND kind = ?", ("o", "p", "duplicates")
        ).fetchall()
    assert not any(row[3].startswith("SCAN") for row in plan), plan


def test_an_ambiguous_call_gets_nothing_and_touches_no_table(tmp_path):
    db = CatalogDb(tmp_path / "notice.db")
    store = CodeAnalysisStore(db.session)
    _publish(store, "/repo", "x")
    statements: list[str] = []
    event.listen(db.engine, "before_cursor_execute",
                 lambda *args: statements.append(args[2]))
    assert claim_scope_notice(None, "/repo", store=store) is None
    assert claim_scope_notice(OWNER, None, store=store) is None
    assert claim_scope_notice(OWNER, "", store=store) is None
    assert statements == []


def test_a_windows_scope_is_found_under_its_canonical_spelling(tmp_path):
    db = CatalogDb(tmp_path / "notice.db")
    store = CodeAnalysisStore(db.session)
    _publish(store, "E:/work/repo", "found")
    assert claim_scope_notice(OWNER, "E:\\Work\\Repo", store=store) == "found"


def test_another_owner_never_receives_it(tmp_path):
    db = CatalogDb(tmp_path / "notice.db")
    store = CodeAnalysisStore(db.session)
    _publish(store, "/repo", "mine")
    assert claim_scope_notice("someone-else", "/repo", store=store) is None
    assert claim_scope_notice(OWNER, "/repo", store=store) == "mine"


def test_a_broken_lookup_reads_as_nothing_to_deliver(tmp_path):
    db = CatalogDb(tmp_path / "notice.db")
    store = CodeAnalysisStore(db.session)
    with db.engine.begin() as conn:
        conn.execute(text("DROP TABLE code_scope_analyses"))
    assert claim_scope_notice(OWNER, "/repo", store=store) is None


# ------------------------------------------------------------------- /api/code/search

@pytest.fixture()
def code_env(vector_store, monkeypatch, tmp_path):
    from app.services.code_kb.service import CodeKBService

    catalog = CatalogDb(tmp_path / "catalog.db")  # a file: threads must not share one connection
    service = CodeKBService(embed_fn=embed, vector_store=vector_store)
    # The default analysis store opens ``core.db.database.SessionLocal`` at call time.
    monkeypatch.setattr("core.db.database.SessionLocal", catalog._session_local)
    monkeypatch.setattr(code_routes, "get_code_kb_service", lambda: service)
    monkeypatch.setattr(code_routes, "get_existing_code_index_queue", lambda: None)
    monkeypatch.setattr(code_routes, "get_index_catalog_service", lambda: catalog.service)
    monkeypatch.setattr(code_routes, "resolve_owner", lambda *args: OWNER)

    class Env:
        pass

    state = Env()
    state.service, state.catalog, state.store = service, catalog, CodeAnalysisStore(
        catalog.session)
    return state


def _index(env, project, files):
    env.service.index(make_request(project, files, []), owner_user_id=OWNER)
    env.catalog.add_files(project, list(files), owner=OWNER)


def _search(project, **extra):
    request = CodeSearchRequest(
        query="basis 0", project=project, mode="dense", top_k=20, **extra)
    return code_routes._run_private_code_search(request, OWNER)


def test_the_first_code_search_carries_the_notice_and_the_second_does_not(code_env):
    _index(code_env, "/repo", {"app/a.py": [0, 1]})
    _publish(code_env.store, "/repo", "Duplicate code found: a <-> b")

    first = _search("/repo")
    assert first.scope_notice == "Duplicate code found: a <-> b"
    assert first.hits, "the notice rides along with real results"
    assert _search("/repo").scope_notice is None

    _publish(code_env.store, "/repo", "New duplicate code: c <-> d")
    assert _search("/repo").scope_notice == "New duplicate code: c <-> d"
    assert _search("/repo").scope_notice is None


def test_a_subdirectory_search_receives_its_indexed_parents_notice(code_env):
    _index(code_env, "/repo", {"app/a.py": [0, 1], "docs/b.py": [0]})
    _publish(code_env.store, "/repo", "parent notice")

    response = _search("/repo/app")
    assert response.resolved_scope == "/repo"
    assert response.scope_notice == "parent notice"
    assert _search("/repo").scope_notice is None, "it was delivered once, to the subdirectory"


def test_a_scope_with_its_own_index_is_judged_by_its_own_notice(code_env):
    _index(code_env, "/repo", {"app/a.py": [0]})
    _index(code_env, "/repo/app", {"x.py": [0]})
    _publish(code_env.store, "/repo", "parent notice")
    _publish(code_env.store, "/repo/app", "child notice")

    assert _search("/repo/app").scope_notice == "child notice"
    assert _search("/repo").scope_notice == "parent notice"


def test_concurrent_code_searches_deliver_the_notice_once(code_env):
    _index(code_env, "/repo", {"app/a.py": [0, 1]})
    _publish(code_env.store, "/repo", "once")
    workers = 8
    barrier = threading.Barrier(workers)
    notices: list[str | None] = []
    lock = threading.Lock()

    def search() -> None:
        barrier.wait()
        notice = _search("/repo").scope_notice
        with lock:
            notices.append(notice)

    threads = [threading.Thread(target=search) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert len(notices) == workers
    assert notices.count("once") == 1


def test_a_withheld_scope_does_not_spend_the_delivery(code_env):
    _index(code_env, "/repo", {"app/a.py": [0, 1]})
    _publish(code_env.store, "/repo", "kept for later")
    code_env.catalog.service.mark_code_scope_preparing(
        owner_user_id=OWNER, host="host", project="/repo", job_id="writer")

    withheld = _search("/repo")
    assert withheld.scope_snapshot_state == "preparing" and withheld.scope_notice is None

    code_env.catalog.service.mark_code_scope_completed(
        owner_user_id=OWNER, host="host", project="/repo", job_id="writer")
    assert _search("/repo").scope_notice == "kept for later"


def test_a_failing_notice_lookup_never_fails_the_search(code_env):
    _index(code_env, "/repo", {"app/a.py": [0, 1]})
    with code_env.catalog.engine.begin() as conn:
        conn.execute(text("DROP TABLE code_scope_analyses"))
    response = _search("/repo")
    assert response.hits and response.scope_notice is None


def test_a_federated_search_neither_carries_nor_consumes_the_notice(code_env):
    _index(code_env, "/repo", {"app/a.py": [0, 1]})
    _publish(code_env.store, "/repo", "not for federated")

    request = FederatedCodeSearchRequest(
        query="basis 0", projects=["/repo"], mode="dense", top_k=5)
    response, _excluded, _searched = code_routes._run_federated_private_leg(
        request, OWNER, pool=20)
    assert response.scope_notice is None
    assert _search("/repo").scope_notice == "not for federated"


def test_an_owner_wide_search_has_no_single_scope_so_no_notice(code_env):
    _index(code_env, "/repo", {"app/a.py": [0, 1]})
    _publish(code_env.store, "/repo", "scoped only")
    request = CodeSearchRequest(query="basis 0", mode="dense", top_k=5)
    assert code_routes._run_private_code_search(request, OWNER).scope_notice is None
    assert _search("/repo").scope_notice == "scoped only"


# --------------------------------------------------------------- /api/code/jobs/{id}

def _http(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def test_the_job_status_carries_the_notice_only_when_asked(code_env, monkeypatch):
    queue = CodeIndexJobQueue(service=code_env.service, catalog_service=code_env.catalog.service)
    try:
        monkeypatch.setattr(code_routes, "get_existing_code_index_queue", lambda: queue)
        job = queue.submit(make_request("/repo", {"app/a.py": [0, 1]}, []), owner_user_id=OWNER)
        deadline = time.time() + 20
        while queue.get(job.job_id).status is not JobStatus.COMPLETED and time.time() < deadline:
            time.sleep(0.02)
        assert queue.get(job.job_id).status is JobStatus.COMPLETED
        _publish(code_env.store, "/repo", "after the index")

        app = FastAPI()
        app.include_router(code_routes.router, prefix="/api/code")

        async def poll(query: str = "") -> dict:
            async with _http(app) as client:
                response = await client.get(f"/api/code/jobs/{job.job_id}{query}")
            assert response.status_code == 200, response.text
            return response.json()

        # A client's own progress polling must not consume a message meant for a person.
        assert (asyncio.run(poll()))["scope_notice"] is None
        assert (asyncio.run(poll()))["scope_notice"] is None
        assert (asyncio.run(poll("?scope_notice=true")))["scope_notice"] == "after the index"
        assert (asyncio.run(poll("?scope_notice=true")))["scope_notice"] is None
    finally:
        queue.shutdown(timeout=5)


# ------------------------------------------------------------------ /api/journal/search

@pytest.fixture()
def journal_env(monkeypatch, tmp_path):
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    from config import settings
    from core.storage.qdrant_store import QdrantStore

    try:
        client = QdrantClient(url="http://127.0.0.1:6333", timeout=30)
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no local Qdrant: {exc}")
    name = f"journal_notice_{uuid.uuid4().hex[:8]}_test"
    params = qmodels.VectorParams(size=8, distance=qmodels.Distance.COSINE)
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": params} if settings.embedding.supports_sparse else params)
    inner = QdrantStore.__new__(QdrantStore)
    inner.collection_name = name
    inner.client = client

    catalog = CatalogDb(tmp_path / "catalog.db")  # a file: threads must not share one connection
    service = JournalService(
        repository=JournalRepository(session_factory=catalog.session),
        vector_store=JournalVectorStore(collection_name=name, store=inner),
        embed_fn=lambda texts, include_sparse: ([[1.0] + [0.0] * 7 for _ in texts], None),
        catalog=catalog.service)
    monkeypatch.setattr("core.db.database.SessionLocal", catalog._session_local)
    monkeypatch.setattr(journal_routes, "get_journal_service", lambda: service)
    monkeypatch.setattr(journal_routes, "resolve_owner", lambda *args: OWNER)

    class Env:
        pass

    state = Env()
    state.service, state.catalog = service, catalog
    state.store = CodeAnalysisStore(catalog.session)
    try:
        yield state
    finally:
        for attempt in range(3):
            try:
                client.delete_collection(name)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    warnings.warn(f"could not drop scratch collection {name}: {exc}")
                time.sleep(0.5 * (attempt + 1))


def _journal_search(scope: str) -> dict:
    app = FastAPI()
    app.include_router(journal_routes.router, prefix="/api/journal")

    async def call() -> dict:
        async with _http(app) as client:
            response = await client.post("/api/journal/search", json={
                "project_slug": scope, "query": "marker", "mode": "keyword", "top_k": 5})
        assert response.status_code == 200, response.text
        return response.json()

    return asyncio.run(call())


def _journal_doc(env, scope, title):
    env.service.record(JournalRecordRequest(
        project_slug=scope, client="claude", source_kind="doc", title=title,
        body="shared marker text", owner_user_id=OWNER, dedupe_policy="return_existing"))
    env.catalog.add_files(scope, [title], owner=OWNER, kind="doc")


def test_the_first_journal_search_carries_the_notice_and_the_second_does_not(journal_env):
    _journal_doc(journal_env, "/proj", "docs/a.md")
    _publish(journal_env.store, "/proj", "journal-side notice")

    first = _journal_search("/proj")
    assert first["hits"] and first["scope_notice"] == "journal-side notice"
    assert _journal_search("/proj")["scope_notice"] is None

    _publish(journal_env.store, "/proj", "newer notice")
    assert _journal_search("/proj")["scope_notice"] == "newer notice"


def test_a_journal_subdirectory_search_receives_its_parents_notice(journal_env):
    _journal_doc(journal_env, "/proj", "docs/a.md")
    _publish(journal_env.store, "/proj", "parent notice")

    response = _journal_search("/proj/docs")
    assert response["resolved_scope"] == "/proj"
    assert response["scope_notice"] == "parent notice"
    assert _journal_search("/proj")["scope_notice"] is None


def test_a_failing_notice_lookup_never_fails_the_journal_search(journal_env):
    _journal_doc(journal_env, "/proj", "docs/a.md")
    with journal_env.catalog.engine.begin() as conn:
        conn.execute(text("DROP TABLE code_scope_analyses"))
    response = _journal_search("/proj")
    assert response["hits"] and response["scope_notice"] is None
