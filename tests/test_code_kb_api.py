"""TDD for code KB routes (/api/code/*): owner resolved server-side, owner isolation."""
from __future__ import annotations

import asyncio
import math
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.code_kb.service import CodeKBService
from app.services.code_kb.schemas import CodeIndexRequest, CodeSearchRequest, CodeSearchResponse
from app.services.scope_paths import scopes_share_files

_VOCAB = ["qdrant", "upsert", "wait", "react", "button", "render", "vector"]


def _bow(text):
    t = text.lower()
    v = [float(t.count(w)) for w in _VOCAB]
    return v if any(v) else [1.0] * len(_VOCAB)


def _embed(texts):
    return [_bow(t) for t in texts]


async def _assert_route_yields_to_event_loop(coro):
    task = asyncio.create_task(coro)
    await asyncio.sleep(0.02)
    event_loop_remained_responsive = not task.done()
    await task
    return event_loop_remained_responsive


def test_search_route_does_not_block_the_event_loop(monkeypatch):
    from app.api.routes import code_kb

    class SlowService:
        def search(self, body, *, owner_user_id):
            time.sleep(0.1)
            return CodeSearchResponse(hits=[], total=0)

    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "get_code_kb_service", lambda: SlowService())

    assert asyncio.run(_assert_route_yields_to_event_loop(code_kb.search_code(
        CodeSearchRequest(query="qdrant"), request=object(), user_id="u1")))


def test_search_route_initializes_service_off_the_event_loop(monkeypatch):
    from app.api.routes import code_kb

    class FastService:
        def search(self, body, *, owner_user_id):
            return CodeSearchResponse(hits=[], total=0)

    def slow_factory():
        time.sleep(0.1)
        return FastService()

    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "get_code_kb_service", slow_factory)

    assert asyncio.run(_assert_route_yields_to_event_loop(code_kb.search_code(
        CodeSearchRequest(query="qdrant"), request=object(), user_id="u1")))


def test_windows_search_scope_alias_lookup_runs_off_the_event_loop(monkeypatch):
    from app.api.routes import code_kb

    class SlowCatalog:
        def equivalent_project_scopes(self, **_kwargs):
            time.sleep(0.1)
            return ["E:/repo", "E:/Repo"]

    class FastService:
        def search(self, body, *, owner_user_id, project_aliases):
            assert project_aliases == ["E:/repo", "E:/Repo"]
            return CodeSearchResponse(hits=[], total=0)

    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "get_index_catalog_service", lambda: SlowCatalog())
    monkeypatch.setattr(code_kb, "get_code_kb_service", lambda: FastService())

    assert asyncio.run(_assert_route_yields_to_event_loop(code_kb.search_code(
        CodeSearchRequest(query="qdrant", project=r"E:\Repo"),
        request=object(), user_id="u1")))


def test_list_jobs_reads_legacy_windows_scope_aliases_and_merges_newest(monkeypatch):
    from app.api.routes import code_kb
    from app.services.code_kb.jobs import CodeIndexJob

    canonical = "E:/repo"
    legacy = "E:/Repo"
    old_job = CodeIndexJob(
        job_id="old", owner_user_id="u1", project=legacy, host="old-host",
        files_total=1, chunks_total=1, created_at=10,
    )
    new_job = CodeIndexJob(
        job_id="new", owner_user_id="u1", project=canonical, host="new-host",
        files_total=1, chunks_total=1, created_at=20,
    )

    class Catalog:
        def equivalent_project_scopes(self, **_kwargs):
            return [canonical, legacy]

    class Queue:
        def list_jobs(self, *, project, **_kwargs):
            return {canonical: [new_job], legacy: [old_job]}[project]

    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "get_index_catalog_service", lambda: Catalog())
    monkeypatch.setattr(code_kb, "get_code_index_queue", lambda: Queue())

    result = asyncio.run(code_kb.list_index_jobs(
        request=object(), user_id="u1", project=r"e:\Repo", limit=20,
        active_only=False,
    ))

    assert [job.job_id for job in result.jobs] == ["new", "old"]


def test_delete_route_runs_catalog_and_vector_work_off_the_event_loop(monkeypatch):
    from app.api.routes import code_kb
    from app.services.code_kb.schemas import CodeDeleteResponse

    class SlowCatalog:
        def equivalent_project_scopes(self, **_kwargs):
            time.sleep(0.1)
            return ["E:/repo"]

        def delete_project(self, **_kwargs):
            return 1

    class Queue:
        def cancel_project(self, **_kwargs):
            return 0

    class Service:
        def delete_project(self, project, **_kwargs):
            return CodeDeleteResponse(project=project, deleted=1)

    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "get_index_catalog_service", lambda: SlowCatalog())
    monkeypatch.setattr(code_kb, "get_code_index_queue", lambda: Queue())
    monkeypatch.setattr(code_kb, "get_code_kb_service", lambda: Service())

    assert asyncio.run(_assert_route_yields_to_event_loop(code_kb.delete_project(
        request=object(), project=r"e:\Repo", user_id="u1")))


def test_delete_route_marks_visibility_before_vector_delete_and_clears_after_success(monkeypatch):
    """A successful delete exposes neither a mid-delete vector state nor a stale marker."""
    from app.api.routes import code_kb
    from app.services.code_kb.schemas import CodeDeleteResponse

    events: list[tuple[str, str]] = []

    class Catalog:
        markers: dict[tuple[str, str], dict[str, object]] = {}

        def mark_code_scope_preparing(self, *, owner_user_id, host, project, job_id):
            events.append(("preparing", project))
            assert job_id is None
            self.markers[(owner_user_id, project)] = {"state": "preparing"}

        def mark_code_scope_recovery_required(self, *, owner_user_id, host, project, job_id):
            events.append(("recovery_required", project))
            self.markers[(owner_user_id, project)] = {"state": "recovery_required"}

        def delete_project(self, *, owner_user_id, project, kind):
            events.append(("catalog_delete", project))
            assert self.markers[(owner_user_id, project)]["state"] == "preparing"
            self.markers.pop((owner_user_id, project), None)
            return 1

    class Queue:
        def cancel_project(self, **_kwargs):
            return 0

    class Service:
        def delete_project(self, project, *, owner_user_id, project_aliases):
            events.append(("vector_delete", project))
            assert catalog.markers[(owner_user_id, project)]["state"] == "preparing"
            return CodeDeleteResponse(project=project, deleted=1)

    catalog = Catalog()
    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "get_index_catalog_service", lambda: catalog)
    monkeypatch.setattr(code_kb, "get_code_index_queue", lambda: Queue())
    monkeypatch.setattr(code_kb, "get_code_kb_service", lambda: Service())

    response = asyncio.run(code_kb.delete_project(
        request=object(), project="/repo/delete-success", user_id="u1"))

    assert response.deleted == 1
    assert events == [
        ("preparing", "/repo/delete-success"),
        ("vector_delete", "/repo/delete-success"),
        ("catalog_delete", "/repo/delete-success"),
    ]
    assert catalog.markers == {}


def test_delete_route_failure_marks_recovery_and_search_remains_fail_closed(monkeypatch):
    """A partial vector delete becomes recovery_required instead of a misleading zero hit."""
    from app.api.routes import code_kb

    class Catalog:
        def __init__(self):
            self.markers: dict[tuple[str, str], dict[str, object]] = {}

        def mark_code_scope_preparing(self, *, owner_user_id, host, project, job_id):
            assert job_id is None
            self.markers[(owner_user_id, project)] = {
                "state": "preparing", "job_id": job_id,
                "previous_completed_snapshot_available": True,
            }

        def mark_code_scope_recovery_required(self, *, owner_user_id, host, project, job_id):
            assert job_id is None
            self.markers[(owner_user_id, project)] = {
                "state": "recovery_required", "job_id": job_id,
                "previous_completed_snapshot_available": True,
            }

        def code_scope_visibility(self, *, owner_user_id, projects):
            return self.markers.get((owner_user_id, projects[0]))

        def delete_project(self, **_kwargs):
            raise AssertionError("catalog must not run after vector delete failure")

    class Queue:
        def cancel_project(self, **_kwargs):
            return 0

    class Service:
        search_calls = 0

        @contextmanager
        def scope_snapshot(self, *, owner_user_id):
            assert owner_user_id == "u1"
            yield

        def delete_project(self, project, *, owner_user_id, project_aliases):
            assert catalog.markers[(owner_user_id, project)]["state"] == "preparing"
            raise RuntimeError("simulated partial vector delete")

        def search(self, *_args, **_kwargs):
            self.search_calls += 1
            return CodeSearchResponse(hits=[], total=1)

    catalog = Catalog()
    service = Service()
    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "get_index_catalog_service", lambda: catalog)
    monkeypatch.setattr(code_kb, "get_code_index_queue", lambda: Queue())
    monkeypatch.setattr(code_kb, "get_existing_code_index_queue", lambda: None)
    monkeypatch.setattr(code_kb, "get_code_kb_service", lambda: service)

    with pytest.raises(RuntimeError, match="partial vector delete"):
        asyncio.run(code_kb.delete_project(
            request=object(), project="/repo/delete-failure", user_id="u1"))

    assert catalog.markers[("u1", "/repo/delete-failure")]["state"] == "recovery_required"
    response = asyncio.run(code_kb.search_code(
        CodeSearchRequest(query="qdrant", project="/repo/delete-failure"),
        request=object(), user_id="u1"))
    assert response.scope_snapshot_state == "recovery_required"
    assert response.recovery_required is True
    assert service.search_calls == 0


def test_index_route_initializes_queue_off_the_event_loop(monkeypatch):
    from app.api.routes import code_kb
    from app.services.code_kb.jobs import CodeIndexJob

    class FastQueue:
        def submit(self, body, *, owner_user_id):
            return CodeIndexJob(
                job_id="queued-job", owner_user_id=owner_user_id,
                project=body.project, host=body.host,
                files_total=len(body.files), chunks_total=0,
            )

    def slow_factory():
        time.sleep(0.1)
        return FastQueue()

    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "get_code_index_queue", slow_factory)
    monkeypatch.setattr(code_kb, "emit_index_audit", lambda record: record)
    monkeypatch.setattr(code_kb, "push_index_audit_bg", lambda _record: None)

    assert asyncio.run(_assert_route_yields_to_event_loop(code_kb.index_code(
        CodeIndexRequest.model_validate(_index_body()),
        request=object(), user_id="u1")))


def test_index_route_fingerprints_request_off_the_event_loop(monkeypatch):
    from app.api.routes import code_kb
    from app.services.code_kb.jobs import CodeIndexJob

    class SlowQueue:
        def submit(self, body, *, owner_user_id):
            time.sleep(0.1)
            return CodeIndexJob(
                job_id="queued-job", owner_user_id=owner_user_id,
                project=body.project, host=body.host,
                files_total=len(body.files), chunks_total=0,
            )

    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "get_code_index_queue", lambda: SlowQueue())
    monkeypatch.setattr(code_kb, "emit_index_audit", lambda record: record)
    monkeypatch.setattr(code_kb, "push_index_audit_bg", lambda _record: None)

    assert asyncio.run(_assert_route_yields_to_event_loop(code_kb.index_code(
        CodeIndexRequest.model_validate(_index_body()),
        request=object(), user_id="u1")))


def test_get_index_job_returns_404_without_initializing_a_missing_queue(monkeypatch):
    """A read-only poll cannot create a Qdrant-backed worker just to return a miss."""
    from fastapi import HTTPException
    from app.api.routes import code_kb

    def unexpected_lazy_init():
        raise AssertionError("GET /api/code/jobs must not initialize the queue")

    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "get_existing_code_index_queue", lambda: None)
    monkeypatch.setattr(code_kb, "get_code_index_queue", unexpected_lazy_init)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(code_kb.get_index_job(
            "missing-job", request=object(), user_id="u1"))

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "job not found"


class FakeStore:
    supports_sparse = True

    def reusable_vectors(self, *, items, **_scope):
        """In-memory stores keep no vectors worth reusing; real Qdrant is tested separately."""
        return [None] * len(items)

    def reusable_vectors_by_content(self, **_scope):
        return {}

    def __init__(self):
        self.points = []

    @staticmethod
    def _project_matches(candidate, project):
        if project is None:
            return True
        if isinstance(project, str):
            return candidate == project
        return candidate in project

    def upsert(self, *, owner_user_id, project, host, items, dense_vectors, sparse_vectors=None):
        for i, (it, vec) in enumerate(zip(items, dense_vectors)):
            self.points.append({"owner_user_id": owner_user_id, "project": project, "host": host,
                                "path": it.rel_path, "language": it.language,
                                "start_line": it.start_line, "end_line": it.end_line,
                                "label": it.label, "code": it.code, "vector": vec,
                                "sparse": sparse_vectors[i] if sparse_vectors else None})
        return len(items)

    def delete_project(self, *, owner_user_id, project):
        before = len(self.points)
        self.points = [p for p in self.points if not (p["owner_user_id"] == owner_user_id
                       and p["project"] == project)]
        return before - len(self.points)

    def list_paths(self, *, owner_user_id, project):
        return {p["path"] for p in self.points
                if p["owner_user_id"] == owner_user_id
                and self._project_matches(p["project"], project)}

    def list_point_ids_and_paths(self, *, owner_user_id, project):
        return (
            self.list_point_ids_for_paths(owner_user_id=owner_user_id, project=project),
            self.list_paths(owner_user_id=owner_user_id, project=project),
        )

    def list_point_ids_for_paths(self, *, owner_user_id, project, paths=None):
        return {
            id(point) for point in self.points
            if point["owner_user_id"] == owner_user_id
            and point["project"] == project
            and (paths is None or point["path"] in paths)
        }

    def list_point_ids(self, *, owner_user_id, project, path):
        return self.list_point_ids_for_paths(
            owner_user_id=owner_user_id,
            project=project,
            paths={path},
        )

    def delete_point_ids(self, point_ids):
        point_ids = set(point_ids)
        before = len(self.points)
        self.points = [point for point in self.points if id(point) not in point_ids]
        return before - len(self.points)

    def list_owner_files(self, *, owner_user_id, related_to=None, exclude_project=None):
        return {(p["project"], p["path"]) for p in self.points
                if p["owner_user_id"] == owner_user_id
                and p["project"] != exclude_project
                and (related_to is None or scopes_share_files(p["project"], related_to))}

    def list_owner_paths(self, *, owner_user_id):
        return {p["path"] for p in self.points if p["owner_user_id"] == owner_user_id}

    def matching_paths(self, *, owner_user_id, project, pattern, case_sensitive):
        from app.services.scope_paths import path_matches
        paths = (self.list_paths(owner_user_id=owner_user_id, project=project)
                 if project else self.list_owner_paths(owner_user_id=owner_user_id))
        return [p for p in paths if path_matches(p, pattern, case_sensitive=case_sensitive)], len(paths)

    def delete_paths(self, *, owner_user_id, project, paths):
        if not paths:
            return 0
        before = len(self.points)
        self.points = [p for p in self.points if not (p["owner_user_id"] == owner_user_id
                       and p["project"] == project and p["path"] in paths)]
        return before - len(self.points)

    def search(self, *, dense_vector, top_k, owner_user_id, project=None, language=None,
               sparse_vector=None, mode="dense", paths=None, excluded_projects=None):
        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
            return dot / (na * nb) if na and nb else 0.0
        allowed = set(paths) if paths is not None else None
        excluded = set(excluded_projects) if excluded_projects else None
        cands = [p for p in self.points if p["owner_user_id"] == owner_user_id
                 and self._project_matches(p["project"], project)
                 and (language is None or p["language"] == language)
                 and (allowed is None or p["path"] in allowed)
                 and (excluded is None or p["project"] not in excluded)]
        scored = sorted(((cos(dense_vector, p["vector"]), p) for p in cands),
                        key=lambda x: x[0], reverse=True)
        return [{"score": s, **p} for s, p in scored[:top_k]]

    def count(self, *, owner_user_id=None, project=None):
        return len(self.points)


class _CatalogSpy:
    """Records delete_project calls so the route test can assert the code route also
    clears the inventory catalog (not just the vectors)."""

    def __init__(self):
        self.deleted = []
        self.scope_aliases = {}

    def equivalent_project_scopes(self, *, owner_user_id, project, kind="code"):
        return [project, *self.scope_aliases.get((owner_user_id, project, kind), [])]

    def delete_project(self, *, owner_user_id, project, host=None, kind=None):
        self.deleted.append({"owner_user_id": owner_user_id, "project": project,
                             "host": host, "kind": kind})
        return 1


@pytest.fixture
def client():
    from app.api.routes import code_kb
    from app.services.code_kb.jobs import CodeIndexJobQueue
    svc = CodeKBService(embed_fn=_embed, vector_store=FakeStore())
    catalog = _CatalogSpy()
    queue = CodeIndexJobQueue(service=svc, catalog_service=catalog)
    code_kb.get_code_kb_service = lambda: svc
    code_kb.get_code_index_queue = lambda: queue
    code_kb.get_existing_code_index_queue = lambda: queue
    code_kb.get_index_catalog_service = lambda: catalog
    app = FastAPI()
    app.include_router(code_kb.router, prefix="/api/code")
    tc = TestClient(app)
    tc.catalog = catalog  # exposed for delete assertions
    tc.service = svc      # exposed for legacy-scope compatibility assertions
    yield tc
    queue.shutdown(timeout=5)


def _u(uid="u1"):
    return {"X-Retrival-User-Id": uid}


def _index_body(project="proj"):
    return {"project": project, "host": "h1", "root_dir": "/repo",
            "files": [{"rel_path": "core/store.py", "language": "python",
                       "chunks": [{"start_line": 1, "end_line": 9, "label": "upsert_batch",
                                   "content": "qdrant upsert wait vector", "code": "def upsert(): ..."}]}]}


def _multi_body(project="proj"):
    """Two languages + several files/chunks, for exercising the search filters."""
    def chunk(sl, label, content):
        return {"start_line": sl, "end_line": sl + 5, "label": label,
                "content": content, "code": f"// {label}\n{content}"}
    return {"project": project, "host": "h1", "root_dir": "/repo", "files": [
        {"rel_path": "conf/app.yml", "language": "yaml",
         "chunks": [chunk(1, "cfg", "qdrant upsert vector wait")]},
        {"rel_path": "src/store.rs", "language": "rust",
         "chunks": [chunk(1, "up1", "qdrant upsert vector"),
                    chunk(20, "up2", "qdrant upsert wait"),
                    chunk(40, "up3", "qdrant vector render"),
                    chunk(60, "up4", "qdrant upsert button")]},
        {"rel_path": "ui/App.tsx", "language": "typescript",
         "chunks": [chunk(1, "app", "react render button")]},
    ]}


def _index_and_wait(client, headers, body):
    """POST /index (202 + job_id since W156) and poll the job to completion."""
    import time
    r = client.post("/api/code/index", headers=headers, json=body)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    deadline = time.time() + 10
    while time.time() < deadline:
        j = client.get(f"/api/code/jobs/{job_id}", headers=headers).json()
        if j["status"] in ("completed", "failed"):
            return j
        time.sleep(0.02)
    raise AssertionError("index job did not finish")


def test_index_code_emits_audit_at_receipt(client):
    """W172: submitting an index job must emit a faithful audit record at RECEIPT
    (owner/kind/project/files/job_id) — independent of whether the job later
    completes — so no index request is ever dropped from the audit log."""
    import json
    import logging
    from app.services.index_audit import INDEX_AUDIT_LOGGER_NAME

    lines: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda r: lines.append(r.getMessage())  # type: ignore[assignment]
    logging.getLogger(INDEX_AUDIT_LOGGER_NAME).addHandler(handler)
    try:
        r = client.post("/api/code/index", headers=_u("auditu"), json=_index_body())
        assert r.status_code == 202, r.text
        job_id = r.json()["job_id"]
    finally:
        logging.getLogger(INDEX_AUDIT_LOGGER_NAME).removeHandler(handler)

    recs = [json.loads(x) for x in lines if '"index_request"' in x]
    assert len(recs) == 1, recs
    rec = recs[0]
    assert rec["evt"] == "index_request" and rec["kind"] == "code"
    assert rec["owner_user_id"] == "auditu"      # owner, never from body
    assert rec["project"] == "proj" and rec["files"] == 1
    assert rec["job_id"] == job_id               # ties the audit to the job


def test_index_then_search_scoped_to_owner(client):
    job = _index_and_wait(client, _u("u1"), _index_body())
    assert job["status"] == "completed" and job["indexed"] == 1

    hits = client.post("/api/code/search", headers=_u("u1"),
                       json={"query": "qdrant upsert vector", "top_k": 5}).json()
    assert hits["total"] == 1 and hits["hits"][0]["path"] == "core/store.py"

    # another owner sees nothing
    assert client.post("/api/code/search", headers=_u("u2"),
                       json={"query": "qdrant upsert vector"}).json()["total"] == 0


def test_owner_not_from_body(client):
    _index_and_wait(client, _u("real"), {**_index_body(), "owner_user_id": "SPOOF"})
    assert client.post("/api/code/search", headers=_u("real"),
                       json={"query": "qdrant"}).json()["total"] == 1
    assert client.post("/api/code/search", headers=_u("SPOOF"),
                       json={"query": "qdrant"}).json()["total"] == 0


def test_search_language_filter_via_api(client):
    _index_and_wait(client, _u("u1"), _multi_body())
    r = client.post("/api/code/search", headers=_u("u1"),
                    json={"query": "qdrant upsert vector", "language": "rust", "top_k": 10})
    hits = r.json()["hits"]
    assert hits and all(h["language"] == "rust" for h in hits)


def test_search_path_glob_filter_via_api(client):
    _index_and_wait(client, _u("u1"), _multi_body())
    prefix = client.post("/api/code/search", headers=_u("u1"),
                         json={"query": "qdrant upsert vector", "path_glob": "src/", "top_k": 10}).json()["hits"]
    assert prefix and all(h["path"].startswith("src/") for h in prefix)
    globbed = client.post("/api/code/search", headers=_u("u1"),
                          json={"query": "react render button", "path_glob": "*.tsx", "top_k": 10}).json()["hits"]
    assert globbed and all(h["path"].endswith(".tsx") for h in globbed)


def test_search_hides_partial_first_scope_snapshot_while_index_writer_is_active(
    client, monkeypatch,
):
    """A first index must never make its first uploaded chunks look like a completed scope.

    The reported sing-box zero was a search racing the fan-out/index write window.  A
    partial vector response is worse than an empty result because the client receives no
    reason to wait for the accepted job.  The route therefore has to gate this scope
    before it delegates to the vector store.
    """
    from app.api.routes import code_kb

    scope = "E:/jetb/pythons/playground/et/upstream/sing-box"
    client.service.vector_store.points.append({
        "owner_user_id": "u1", "project": scope, "host": "DESKTOP",
        "path": "experimental/Hysteria-CC-RX/partial.go", "language": "go",
        "start_line": 1, "end_line": 4, "label": "SendBPS",
        "code": "func SendBPS() {}", "vector": _bow("Hysteria-CC-RX SendBPS"),
        "sparse": None,
    })

    class Queue:
        def scope_search_state(self, **_kwargs):
            return SimpleNamespace(
                active_job_id="initial-sing-box", mutation_started=True,
                previous_completed_snapshot_available=False,
            )

    monkeypatch.setattr(code_kb, "get_existing_code_index_queue", lambda: Queue())
    response = client.post("/api/code/search", headers=_u("u1"), json={
        "query": "Hysteria-CC-RX SendBPS", "project": scope, "mode": "dense",
    })

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["hits"] == [] and body["total"] == 0
    assert body["scope_snapshot_state"] == "preparing"
    assert body["active_index_job_id"] == "initial-sing-box"
    assert body["previous_completed_snapshot_available"] is False
    assert "partial" in body["notice"].lower()
    assert "no matching" not in body["notice"].lower()


def test_search_can_serve_the_previous_completed_snapshot_before_refresh_writes(
    client, monkeypatch,
):
    """Embedding a refresh is safe to read; only its write phase needs the hard gate."""
    from app.api.routes import code_kb

    scope = "E:/repo/sing-box"
    client.service.vector_store.points.append({
        "owner_user_id": "u1", "project": scope, "host": "DESKTOP",
        "path": "experimental/Hysteria-CC-RX/config.go", "language": "go",
        "start_line": 10, "end_line": 16, "label": "HysteriaCCRX",
        "code": "func SendBPS() {}", "vector": _bow("Hysteria-CC-RX SendBPS"),
        "sparse": None,
    })

    class Queue:
        def scope_search_state(self, **_kwargs):
            return SimpleNamespace(
                active_job_id="refresh-sing-box", mutation_started=False,
                previous_completed_snapshot_available=True,
            )

    monkeypatch.setattr(code_kb, "get_existing_code_index_queue", lambda: Queue())
    response = client.post("/api/code/search", headers=_u("u1"), json={
        "query": "Hysteria-CC-RX SendBPS", "project": scope, "mode": "dense",
    })

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 1
    assert body["scope_snapshot_state"] == "completed"
    assert body["active_index_job_id"] == "refresh-sing-box"
    assert body["previous_completed_snapshot_available"] is True
    assert "previous completed snapshot" in body["notice"].lower()


def test_scoped_search_rechecks_visibility_only_after_taking_owner_fence(monkeypatch):
    """A pre-fence ``None`` is not a permission to search.

    The writer can acquire the owner fence and flip ``mutation_started`` immediately
    after an optimistic state read.  Search must instead acquire that same fence first
    and then inspect the queue, otherwise it can return a partially-upserted scope.
    """
    from app.api.routes import code_kb

    class Service:
        in_owner_fence = False
        search_calls = 0

        @contextmanager
        def scope_snapshot(self, *, owner_user_id):
            assert owner_user_id == "u1"
            self.in_owner_fence = True
            try:
                yield
            finally:
                self.in_owner_fence = False

        def search(self, _body, *, owner_user_id):
            assert owner_user_id == "u1"
            self.search_calls += 1
            return CodeSearchResponse(hits=[], total=1, notice="partial vector result")

    service = Service()

    class Queue:
        def scope_search_state(self, **_kwargs):
            if not service.in_owner_fence:
                return None
            return SimpleNamespace(
                active_job_id="writer-won-the-race", mutation_started=True,
                previous_completed_snapshot_available=False,
            )

    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "get_code_kb_service", lambda: service)
    monkeypatch.setattr(code_kb, "get_existing_code_index_queue", lambda: Queue())

    response = asyncio.run(code_kb.search_code(
        CodeSearchRequest(query="SendBPS", project="/repo/sing-box"),
        request=object(), user_id="u1"))

    assert response.scope_snapshot_state == "preparing"
    assert response.total == 0
    assert service.search_calls == 0


def test_unscoped_search_takes_owner_fence_before_returning_any_scope(monkeypatch):
    """An owner-wide search can include a half-written scoped result too.

    It must use the same fence rather than treating the absence of a project filter as
    proof that all of the owner's scopes are stable.
    """
    from app.api.routes import code_kb

    class Service:
        in_owner_fence = False
        search_calls = 0

        @contextmanager
        def scope_snapshot(self, *, owner_user_id):
            assert owner_user_id == "u1"
            self.in_owner_fence = True
            try:
                yield
            finally:
                self.in_owner_fence = False

        def search(self, _body, *, owner_user_id):
            assert owner_user_id == "u1"
            self.search_calls += 1
            return CodeSearchResponse(hits=[], total=1, notice="partial owner result")

    service = Service()

    class Queue:
        def owner_search_state(self, **_kwargs):
            if not service.in_owner_fence:
                return None
            return SimpleNamespace(
                active_job_id="owner-writer", mutation_started=True,
                previous_completed_snapshot_available=False,
            )

    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "get_code_kb_service", lambda: service)
    monkeypatch.setattr(code_kb, "get_existing_code_index_queue", lambda: Queue())

    response = asyncio.run(code_kb.search_code(
        CodeSearchRequest(query="SendBPS"), request=object(), user_id="u1"))

    assert response.scope_snapshot_state == "preparing"
    assert response.total == 0
    assert service.search_calls == 0


def _sqlite_catalog():
    """A real IndexCatalogService on an in-memory SQLite database."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from core.db.database import Base
    from app.services.index_catalog.repository import IndexCatalogRepository
    from app.services.index_catalog.service import IndexCatalogService

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
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

    return IndexCatalogService(repository=IndexCatalogRepository(session_factory=session_factory))


def test_unscoped_search_after_restart_serves_durably_completed_scope_during_refresh(monkeypatch):
    """After a backend restart the in-memory queue has no completed jobs, so the queue
    alone cannot tell that a refreshing scope already had a completed snapshot. The
    owner-wide route must consult the durable catalog like the single-scope branch does,
    instead of withholding every scope the owner has until the refresh finishes."""
    import threading

    from app.api.routes import code_kb
    from app.services.code_kb.jobs import CodeIndexJobQueue, JobStatus

    catalog = _sqlite_catalog()
    store = FakeStore()
    first = _index_body("/repo/a")
    first["root_dir"] = "/repo/a"

    # Process 1: a real job completes and records the durable "completed" marker.
    before_restart = CodeIndexJobQueue(
        service=CodeKBService(embed_fn=_embed, vector_store=store), catalog_service=catalog)
    done = before_restart.submit(CodeIndexRequest(**first), owner_user_id="u1")
    deadline = time.time() + 10
    while done.status not in (JobStatus.COMPLETED, JobStatus.FAILED) and time.time() < deadline:
        time.sleep(0.02)
    before_restart.shutdown(timeout=5)
    assert done.status is JobStatus.COMPLETED, done.error

    # Process 2 ("restarted"): empty in-memory history. A refresh of the same scope is
    # held in embedding, i.e. running with no vector mutation started yet.
    release = threading.Event()
    embedding_refresh = threading.Event()

    def gated_embed(texts):
        if any("refreshed" in t for t in texts):
            embedding_refresh.set()
            release.wait(timeout=10)
        return _embed(texts)

    svc = CodeKBService(embed_fn=gated_embed, vector_store=store)
    after_restart = CodeIndexJobQueue(service=svc, catalog_service=catalog)
    refresh = _index_body("/repo/a")
    refresh["root_dir"] = "/repo/a"
    refresh["files"][0]["chunks"][0]["content"] = "qdrant upsert wait vector refreshed"
    job = after_restart.submit(CodeIndexRequest(**refresh), owner_user_id="u1")
    try:
        assert embedding_refresh.wait(timeout=10), "refresh job never reached embedding"
        assert job.status is JobStatus.RUNNING and not job.mutation_started

        monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
        monkeypatch.setattr(code_kb, "get_code_kb_service", lambda: svc)
        monkeypatch.setattr(code_kb, "get_existing_code_index_queue", lambda: after_restart)
        monkeypatch.setattr(code_kb, "get_index_catalog_service", lambda: catalog)

        response = asyncio.run(code_kb.search_code(
            CodeSearchRequest(query="qdrant upsert"), request=object(), user_id="u1"))

        assert response.scope_snapshot_state == "completed"
        assert response.total >= 1
        assert {h.project for h in response.hits} == {"/repo/a"}
        assert response.active_index_job_id == job.job_id
        assert response.previous_completed_snapshot_available is True
    finally:
        release.set()
        after_restart.shutdown(timeout=5)


def test_unscoped_search_after_restart_still_withholds_a_never_completed_scope(monkeypatch):
    """The durable fallback must not open the door for a first-ever index: with no
    completed marker in the catalog, owner-wide search stays withheld."""
    import threading

    from app.api.routes import code_kb
    from app.services.code_kb.jobs import CodeIndexJobQueue, JobStatus

    catalog = _sqlite_catalog()
    release = threading.Event()
    embedding = threading.Event()

    def gated_embed(texts):
        if any("wait vector" in t for t in texts):
            embedding.set()
            release.wait(timeout=10)
        return _embed(texts)

    svc = CodeKBService(embed_fn=gated_embed, vector_store=FakeStore())
    queue = CodeIndexJobQueue(service=svc, catalog_service=catalog)
    body = _index_body("/repo/new")
    body["root_dir"] = "/repo/new"
    job = queue.submit(CodeIndexRequest(**body), owner_user_id="u1")
    try:
        assert embedding.wait(timeout=10), "first index never reached embedding"
        assert job.status is JobStatus.RUNNING and not job.mutation_started

        monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
        monkeypatch.setattr(code_kb, "get_code_kb_service", lambda: svc)
        monkeypatch.setattr(code_kb, "get_existing_code_index_queue", lambda: queue)
        monkeypatch.setattr(code_kb, "get_index_catalog_service", lambda: catalog)

        response = asyncio.run(code_kb.search_code(
            CodeSearchRequest(query="qdrant upsert"), request=object(), user_id="u1"))

        assert response.scope_snapshot_state == "preparing"
        assert response.total == 0
        assert response.previous_completed_snapshot_available is False
    finally:
        release.set()
        queue.shutdown(timeout=5)


def test_unscoped_first_active_index_stays_preparing_before_mutation(monkeypatch):
    """An owner-wide first index has no completed snapshot to serve while embedding.

    ``mutation_started=False`` only means no partial Qdrant batch exists yet; it does
    not authorize an owner-wide response to claim a completed snapshot when the first
    active job has no prior completed scope.
    """
    from app.api.routes import code_kb

    class Service:
        search_calls = 0

        @contextmanager
        def scope_snapshot(self, *, owner_user_id):
            assert owner_user_id == "u1"
            yield

        def search(self, _body, *, owner_user_id):
            assert owner_user_id == "u1"
            self.search_calls += 1
            return CodeSearchResponse(hits=[], total=1, notice="would-be stale owner result")

    class Queue:
        def owner_search_state(self, **_kwargs):
            return SimpleNamespace(
                active_job_id="owner-first-index", mutation_started=False,
                previous_completed_snapshot_available=False,
            )

    class Catalog:
        def owner_code_scope_visibility(self, **_kwargs):
            return None

    service = Service()
    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "get_code_kb_service", lambda: service)
    monkeypatch.setattr(code_kb, "get_existing_code_index_queue", lambda: Queue())
    monkeypatch.setattr(code_kb, "get_index_catalog_service", lambda: Catalog())

    response = asyncio.run(code_kb.search_code(
        CodeSearchRequest(query="SendBPS"), request=object(), user_id="u1"))

    assert response.scope_snapshot_state == "preparing"
    assert response.active_index_job_id == "owner-first-index"
    assert response.previous_completed_snapshot_available is False
    assert response.total == 0
    assert service.search_calls == 0


def test_scoped_search_reads_durable_preparing_marker_when_queue_was_lost(monkeypatch):
    """A queue/service restart cannot turn a partially-written scope into a match.

    The persistent catalog marker is the source of truth once the in-memory queue is
    gone.  Search must fail closed before calling the vector store for that owner/scope.
    """
    from app.api.routes import code_kb

    class Service:
        search_calls = 0

        @contextmanager
        def scope_snapshot(self, *, owner_user_id):
            assert owner_user_id == "u1"
            yield

        def search(self, _body, *, owner_user_id):
            assert owner_user_id == "u1"
            self.search_calls += 1
            return CodeSearchResponse(hits=[], total=1, notice="partial after worker crash")

    class Catalog:
        def code_scope_visibility(self, *, owner_user_id, projects):
            assert projects == ["/repo/crashed"]
            if owner_user_id == "u1":
                return {
                    "job_id": "crashed-before-catalog-publish",
                    "state": "preparing",
                    "previous_completed_snapshot_available": False,
                }
            return None

    service = Service()
    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "get_code_kb_service", lambda: service)
    monkeypatch.setattr(code_kb, "get_existing_code_index_queue", lambda: None)
    monkeypatch.setattr(code_kb, "get_index_catalog_service", lambda: Catalog())

    response = asyncio.run(code_kb.search_code(
        CodeSearchRequest(query="SendBPS", project="/repo/crashed"),
        request=object(), user_id="u1"))

    assert response.scope_snapshot_state == "preparing"
    assert response.active_index_job_id == "crashed-before-catalog-publish"
    assert response.total == 0
    assert service.search_calls == 0


def test_recovery_required_search_response_omits_failed_job_poll(monkeypatch):
    """A known failed in-place write is a recovery action, not an active job poll."""
    from app.api.routes import code_kb

    class Service:
        search_calls = 0

        @contextmanager
        def scope_snapshot(self, *, owner_user_id):
            assert owner_user_id == "u1"
            yield

        def search(self, _body, *, owner_user_id):
            self.search_calls += 1
            return CodeSearchResponse(hits=[], total=1)

    class Catalog:
        def code_scope_visibility(self, **_kwargs):
            return {
                "state": "recovery_required",
                "job_id": "failed-write-job",
                "previous_completed_snapshot_available": False,
            }

    service = Service()
    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "get_code_kb_service", lambda: service)
    monkeypatch.setattr(code_kb, "get_existing_code_index_queue", lambda: None)
    monkeypatch.setattr(code_kb, "get_index_catalog_service", lambda: Catalog())

    response = asyncio.run(code_kb.search_code(
        CodeSearchRequest(query="SendBPS", project="/repo/recovery"),
        request=object(), user_id="u1"))

    assert response.scope_snapshot_state == "recovery_required"
    assert response.recovery_required is True
    assert response.active_index_job_id is None
    assert "full recovery index" in (response.notice or "").lower()
    assert "poll" not in (response.notice or "").lower()
    assert "failed-write-job" not in (response.notice or "")
    assert service.search_calls == 0


def test_catalog_visibility_unavailable_response_has_no_fake_poll_job(monkeypatch):
    """A catalog outage must be explicit rather than a fake preparing job receipt."""
    from app.api.routes import code_kb

    class Service:
        search_calls = 0

        @contextmanager
        def scope_snapshot(self, *, owner_user_id):
            assert owner_user_id == "u1"
            yield

        def search(self, _body, *, owner_user_id):
            self.search_calls += 1
            return CodeSearchResponse(hits=[], total=1)

    class Catalog:
        def code_scope_visibility(self, **_kwargs):
            raise RuntimeError("catalog is unavailable")

    service = Service()
    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "get_code_kb_service", lambda: service)
    monkeypatch.setattr(code_kb, "get_existing_code_index_queue", lambda: None)
    monkeypatch.setattr(code_kb, "get_index_catalog_service", lambda: Catalog())

    response = asyncio.run(code_kb.search_code(
        CodeSearchRequest(query="SendBPS", project="/repo/catalog-unavailable"),
        request=object(), user_id="u1"))

    assert response.scope_snapshot_state == "unavailable"
    assert response.recovery_required is False
    assert response.active_index_job_id is None
    assert "catalog" in (response.notice or "").lower()
    assert "retry" in (response.notice or "").lower()
    assert "poll" not in (response.notice or "").lower()
    assert "visibility-unavailable" not in (response.notice or "")
    assert service.search_calls == 0


def test_search_windows_scope_includes_legacy_catalog_case_variant(client):
    """The REST route must pass catalog aliases through to the Qdrant MatchAny filter."""
    canonical = "E:/jetb/pythons/playground/octopus/axonhub/llm"
    legacy = "E:/JETB/Pythons/Playground/Octopus/AxonHub/LLM"
    client.catalog.scope_aliases[("u1", canonical, "code")] = [legacy]
    client.service.vector_store.points.append({
        "owner_user_id": "u1", "project": legacy, "host": "DESKTOP-OTHER",
        "path": "Conversion/Planner/Compact.go", "language": "go",
        "start_line": 1, "end_line": 8, "label": "RequestTypeCompact",
        "code": "func convertCompact() {}", "vector": _bow("RequestTypeCompact"),
        "sparse": {"1": 1.0},
    })

    response = client.post("/api/code/search", headers=_u("u1"), json={
        "query": "RequestTypeCompact",
        "project": r"e:\JETB\Pythons\Playground\Octopus\AxonHub\LLM",
        "mode": "dense",
        "path_glob": "conversion\\planner\\",
        "top_k": 5,
    })

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 1
    assert body["hits"][0]["project"] == legacy
    assert body["scope_aliases"] == [legacy]


def test_search_max_per_file_via_api(client):
    _index_and_wait(client, _u("u1"), _multi_body())
    hits = client.post("/api/code/search", headers=_u("u1"),
                       json={"query": "qdrant upsert vector", "max_per_file": 2, "top_k": 10}).json()["hits"]
    from collections import Counter
    counts = Counter(h["path"] for h in hits)
    assert all(n <= 2 for n in counts.values()), counts


def test_search_bad_max_per_file_rejected(client):
    # ge=1/le=20 validation on the request schema.
    assert client.post("/api/code/search", headers=_u("u1"),
                       json={"query": "x", "max_per_file": 0}).status_code == 422
    assert client.post("/api/code/search", headers=_u("u1"),
                       json={"query": "x", "max_per_file": 999}).status_code == 422


def test_search_accepts_mode_and_rerank(client):
    _index_and_wait(client, _u("u1"), _multi_body())
    for mode in ("hybrid", "dense", "keyword"):
        r = client.post("/api/code/search", headers=_u("u1"),
                        json={"query": "qdrant upsert vector", "mode": mode, "top_k": 5})
        assert r.status_code == 200, (mode, r.text)
    # rerank flag accepted (FakeStore dense-only → mode degrades to dense, no reranker call)
    assert client.post("/api/code/search", headers=_u("u1"),
                       json={"query": "qdrant", "rerank": False}).status_code == 200


def test_search_invalid_mode_rejected(client):
    assert client.post("/api/code/search", headers=_u("u1"),
                       json={"query": "x", "mode": "fuzzy"}).status_code == 422


def test_incremental_index_via_api(client):
    # Full index of a.py + b.py, then incremental: change a.py, drop b.py (manifest omits it).
    body = {"project": "proj", "host": "h1", "root_dir": "/repo", "files": [
        {"rel_path": "a.py", "language": "python",
         "chunks": [{"start_line": 1, "end_line": 5, "label": "a",
                     "content": "qdrant upsert", "code": "def a(): ..."}]},
        {"rel_path": "b.py", "language": "python",
         "chunks": [{"start_line": 1, "end_line": 5, "label": "b",
                     "content": "vector wait", "code": "def b(): ..."}]},
    ]}
    _index_and_wait(client, _u("u1"), body)

    inc = {"project": "proj", "host": "h1", "root_dir": "/repo", "incremental": True,
           "manifest": ["a.py"], "files": [
               {"rel_path": "a.py", "language": "python",
                "chunks": [{"start_line": 1, "end_line": 5, "label": "a",
                            "content": "qdrant upsert changed", "code": "def a(): return 1"}]}]}
    _index_and_wait(client, _u("u1"), inc)

    # b.py gone (manifest reconciliation), a.py present
    hits = client.post("/api/code/search", headers=_u("u1"),
                       json={"query": "qdrant upsert vector wait", "top_k": 10}).json()["hits"]
    paths = {h["path"] for h in hits}
    assert paths == {"a.py"}, paths


def test_incremental_api_accepts_precise_removed_paths_without_manifest(client):
    body = {"project": "proj", "host": "h1", "root_dir": "/repo", "files": [
        {"rel_path": "a.py", "language": "python",
         "chunks": [{"start_line": 1, "end_line": 2, "label": "a",
                     "content": "qdrant upsert", "code": "def a(): ..."}]},
        {"rel_path": "b.py", "language": "python",
         "chunks": [{"start_line": 1, "end_line": 2, "label": "b",
                     "content": "vector wait", "code": "def b(): ..."}]},
    ]}
    _index_and_wait(client, _u("u1"), body)

    _index_and_wait(client, _u("u1"), {
        "project": "proj", "host": "h1", "root_dir": "/repo",
        "incremental": True, "files": [], "removed_paths": ["b.py"],
    })

    hits = client.post("/api/code/search", headers=_u("u1"),
                       json={"query": "qdrant upsert vector wait", "top_k": 10}).json()["hits"]
    assert {hit["path"] for hit in hits} == {"a.py"}


def test_anonymous_non_lan_rejected(client):
    assert client.post("/api/code/search", json={"query": "x"}).status_code == 401
    assert client.post("/api/code/index", json=_index_body()).status_code == 401


def test_delete_project(client):
    _index_and_wait(client, _u("u1"), _index_body("p1"))
    # project is now passed as a query param (the scope is an abspath — can't be a path
    # segment); the route clears BOTH the vectors and the inventory catalog.
    r = client.request("DELETE", "/api/code", params={"project": "p1"}, headers=_u("u1"))
    assert r.status_code == 200 and r.json()["deleted"] >= 1
    # vectors gone
    assert client.post("/api/code/search", headers=_u("u1"),
                       json={"query": "qdrant upsert vector"}).json()["total"] == 0
    # inventory catalog cleared for (owner, project), kind=code, all hosts
    assert client.catalog.deleted == [{"owner_user_id": "u1", "project": "p1",
                                       "host": None, "kind": "code"}]


def test_delete_project_accepts_abspath_scope(client):
    # The scope is the indexed dir's abspath (slashes + a Windows drive colon) — it must
    # survive as a query param where a path segment would break routing. Different slash
    # styles and casing identify the same NTFS scope, including the explicit delete path.
    indexed_scope = r"c:\Users\Arc\AppData\Local\Temp\probe_x"
    queried_scope = "C:/USERS/arc/appdata/local/temp/PROBE_X"
    canonical_scope = "C:/users/arc/appdata/local/temp/probe_x"
    _index_and_wait(client, _u("u1"), _index_body(indexed_scope))
    r = client.request("DELETE", "/api/code", params={"project": queried_scope}, headers=_u("u1"))
    assert r.status_code == 200 and r.json()["deleted"] >= 1
    assert r.json()["project"] == canonical_scope
    assert client.catalog.deleted[-1]["project"] == canonical_scope


def test_delete_windows_scope_clears_legacy_case_variant_too(client):
    """Explicit deletion has the same case-insensitive scope identity as search.

    Otherwise an old mixed-case Qdrant project would stay searchable even after the
    user deleted the directory through its canonical Windows spelling.
    """
    canonical = "E:/jetb/pythons/playground/octopus/axonhub/llm"
    legacy = "E:/JETB/Pythons/Playground/Octopus/AxonHub/LLM"
    client.catalog.scope_aliases[("u1", canonical, "code")] = [legacy]
    client.service.vector_store.points.append({
        "owner_user_id": "u1", "project": legacy, "host": "DESKTOP-OTHER",
        "path": "conversion/compact.go", "language": "go",
        "start_line": 1, "end_line": 3, "label": "compact",
        "code": "func compact() {}", "vector": _bow("compact"), "sparse": None,
    })

    response = client.request(
        "DELETE",
        "/api/code",
        params={"project": r"e:\JETB\Pythons\Playground\Octopus\AxonHub\LLM"},
        headers=_u("u1"),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["project"] == canonical
    assert body["deleted"] == 1
    assert body["scope_aliases"] == [legacy]
    assert client.service.vector_store.points == []
    assert [call["project"] for call in client.catalog.deleted] == [canonical, legacy]


def test_list_jobs_canonicalizes_windows_scope_query(client):
    indexed_scope = r"e:\JETB\Pythons\Playground\Octopus\AxonHub\LLM"
    canonical_scope = "E:/jetb/pythons/playground/octopus/axonhub/llm"
    job = _index_and_wait(client, _u("u1"), _index_body(indexed_scope))

    response = client.get(
        "/api/code/jobs",
        params={"project": "e:/jetb/pythons/playground/octopus/axonhub/LLM"},
        headers=_u("u1"),
    )

    assert response.status_code == 200, response.text
    assert [item["job_id"] for item in response.json()["jobs"]] == [job["job_id"]]
    assert response.json()["jobs"][0]["project"] == canonical_scope


def test_delete_project_requires_project(client):
    assert client.request("DELETE", "/api/code", headers=_u("u1")).status_code == 422
    assert client.post("/api/code/search", headers=_u("u1"),
                       json={"query": "qdrant"}).json()["total"] == 0


def test_code_api_rejects_drive_relative_windows_scope(client):
    response = client.post(
        "/api/code/search", headers=_u("u1"),
        json={"query": "qdrant", "project": "E:repo"},
    )
    assert response.status_code == 422
    assert "Windows scope must be absolute" in response.text


# ---- federated-search (docs/federated_code_search_plan.md A3, revision R1) ----
# These test dispatch/assembly (auth fail-closed, concurrency, diagnostics passthrough,
# decision-10 per-scope exclusion) with monkeypatched service/catalog/queue stand-ins,
# following this file's existing style; there is no data-store semantics here so this
# is the allowed monkeypatch exception.

def _federated_public_response(*, hits=(), selected_source_ids=(), notice=None,
                               effective_mode="hybrid"):
    from app.services.public_code.search_service import PublicCodeSearchResponse
    return PublicCodeSearchResponse(
        query="q", selected_source_ids=tuple(selected_source_ids), hits=tuple(hits),
        requested_mode="hybrid", effective_mode=effective_mode, notice=notice,
        rerank_applied=False, candidate_count=len(hits), rerank_candidate_count=0,
    )


class _NoFenceService:
    """A private service with no ``scope_snapshot`` at all -- the federated leg then
    skips fencing entirely and searches directly, same fallback ``_run_private_code_search``
    already has for `/api/code/search`."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def search(self, request, *, owner_user_id, project_aliases=None, excluded_projects=None):
        self.calls.append({
            "owner_user_id": owner_user_id, "project": request.project,
            "project_aliases": project_aliases, "excluded_projects": excluded_projects,
        })
        return self.response


def test_federated_search_non_admin_include_public_gets_403_without_calling_public_service(
    monkeypatch,
):
    from fastapi import HTTPException
    from app.api.routes import code_kb
    from app.services.code_kb.schemas import CodeSearchResponse, FederatedCodeSearchRequest

    public_calls = []

    class PublicService:
        def search(self, request):
            public_calls.append(request)
            raise AssertionError("public service must not be called for a non-admin caller")

    def deny_admin(*_args):
        raise HTTPException(status_code=403, detail="public code requires admin tier")

    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "resolve_admin", deny_admin)
    monkeypatch.setattr(
        code_kb, "get_code_kb_service",
        lambda: _NoFenceService(CodeSearchResponse(hits=[], total=0)))
    monkeypatch.setattr(code_kb, "get_public_code_search_service", lambda: PublicService())

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(code_kb.federated_code_search(
            FederatedCodeSearchRequest(query="q", include_public=True),
            request=object(), user_id="u1", tier="member"))

    assert exc_info.value.status_code == 403
    assert public_calls == []


def test_federated_search_default_params_returns_200_private_only(monkeypatch):
    """No include_public passed at all -- R1's default is False, not True."""
    from app.api.routes import code_kb
    from app.services.code_kb.schemas import CodeHit, CodeSearchResponse, FederatedCodeSearchRequest

    private = _NoFenceService(CodeSearchResponse(
        hits=[CodeHit(path="a.py", start_line=1, end_line=2, score=0.9, code="x")], total=1))

    class PublicService:
        def search(self, request):
            raise AssertionError("public service must not be called by default")

    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "get_code_kb_service", lambda: private)
    monkeypatch.setattr(code_kb, "get_public_code_search_service", lambda: PublicService())

    response = asyncio.run(code_kb.federated_code_search(
        FederatedCodeSearchRequest(query="q"),
        request=object(), user_id="u1", tier=None))

    assert response.total == 1
    assert [hit.origin for hit in response.hits] == ["private"]
    assert response.private.included is True
    assert response.public.included is False
    assert response.public.candidate_count == 0
    assert response.private.searched_projects == []
    assert response.private.excluded_scopes == []


def test_federated_search_reports_timings_ms(monkeypatch):
    """F2-b: embed/retrieve come from the (always-on) private leg's own timings; fuse
    always runs (always non-null); rerank is null when rerank=false; total covers all
    of them."""
    from app.api.routes import code_kb
    from app.services.code_kb.schemas import (
        CodeHit, CodeSearchResponse, CodeSearchTimingsMs, FederatedCodeSearchRequest,
    )

    private = _NoFenceService(CodeSearchResponse(
        hits=[CodeHit(path="a.py", start_line=1, end_line=2, score=0.9, code="x")], total=1,
        timings_ms=CodeSearchTimingsMs(embed=1.5, retrieve=2.5, total=4.0)))

    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "get_code_kb_service", lambda: private)

    response = asyncio.run(code_kb.federated_code_search(
        FederatedCodeSearchRequest(query="q"),
        request=object(), user_id="u1", tier=None))

    # embed/retrieve pass through from the private leg's own measurement (the
    # NoFenceService stub returns instantly, so — unlike the real-timing service-level
    # tests — these are fixed stand-in values, not wall-clock proof; that invariant is
    # covered by test_search_reports_per_phase_timings_ms against real elapsed time).
    t = response.timings_ms
    assert t.embed == 1.5
    assert t.retrieve == 2.5
    assert t.fuse is not None and t.fuse >= 0, "fuse always runs in federated-search"
    assert t.rerank is None, "rerank=false must leave rerank timing null"
    assert t.total is not None and t.total >= 0


def test_federated_search_admin_include_public_true_gets_both_origins_with_diagnostics(
    monkeypatch,
):
    from app.api.routes import code_kb
    from app.services.code_kb.schemas import CodeHit, CodeSearchResponse, FederatedCodeSearchRequest
    from app.services.public_code.search_service import PublicCodeSearchHit

    private = _NoFenceService(CodeSearchResponse(
        hits=[CodeHit(path="a.py", start_line=1, end_line=2, score=0.9, code="priv")],
        total=1, scope_snapshot_state="completed",
    ))

    class PublicService:
        def search(self, request):
            return _federated_public_response(
                hits=[PublicCodeSearchHit(
                    score=0.8, retrieval_score=0.8, reranker_score=None,
                    source_id="pion-webrtc", revision="deadbeef00", domain="github.com",
                    path="x.go", language="go", start_line=1, end_line=2, label="",
                    content="pub", code="pub",
                )],
                selected_source_ids=("pion-webrtc",),
            )

    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "resolve_admin", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "get_code_kb_service", lambda: private)
    monkeypatch.setattr(code_kb, "get_public_code_search_service", lambda: PublicService())

    response = asyncio.run(code_kb.federated_code_search(
        FederatedCodeSearchRequest(query="q", include_public=True),
        request=object(), user_id="u1", tier="admin"))

    assert sorted({hit.origin for hit in response.hits}) == ["private", "public"]
    assert response.private.included is True
    assert response.private.candidate_count == 1
    assert response.private.scope_snapshot_state == "completed"
    assert response.public.included is True
    assert response.public.candidate_count == 1
    assert response.public.selected_source_ids == ["pion-webrtc"]


def test_federated_search_include_public_false_with_source_ids_is_422(client):
    response = client.post(
        "/api/code/federated-search", headers=_u("u1"),
        json={"query": "q", "include_public": False, "source_ids": ["pion-webrtc"]},
    )
    assert response.status_code == 422


def test_federated_search_include_public_false_with_domains_is_422(client):
    response = client.post(
        "/api/code/federated-search", headers=_u("u1"),
        json={"query": "q", "include_public": False, "domains": ["github.com"]},
    )
    assert response.status_code == 422


def test_federated_search_explicit_projects_issue_one_query_with_aliases(monkeypatch):
    """projects=[a, b] must search in ONE Qdrant query (single service.search call),
    with each Windows-case alias expanded into project_aliases."""
    from app.api.routes import code_kb
    from app.services.code_kb.schemas import CodeSearchResponse, FederatedCodeSearchRequest
    from contextlib import contextmanager

    class Catalog:
        def equivalent_project_scopes(self, *, owner_user_id, project, kind="code"):
            # The schema validator already canonicalised the request's "C:/Repo" to
            # "C:/repo" (case-folded); this stand-in returns the legacy mixed-case
            # spellings a real catalog would have on file for that canonical scope.
            if project == "C:/repo":
                return ["C:/repo", "C:/Repo", "C:/REPO"]
            return [project]

        def code_scope_visibility(self, *, owner_user_id, projects):
            return None

    class Service:
        def __init__(self):
            self.calls = []

        @contextmanager
        def scope_snapshot(self, *, owner_user_id):
            yield

        def search(self, request, *, owner_user_id, project_aliases=None, excluded_projects=None):
            self.calls.append({
                "project": request.project, "project_aliases": project_aliases,
                "excluded_projects": excluded_projects,
            })
            return CodeSearchResponse(hits=[], total=0)

    service = Service()
    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "get_code_kb_service", lambda: service)
    monkeypatch.setattr(code_kb, "get_index_catalog_service", lambda: Catalog())
    monkeypatch.setattr(code_kb, "get_existing_code_index_queue", lambda: None)

    response = asyncio.run(code_kb.federated_code_search(
        FederatedCodeSearchRequest(query="q", projects=["C:/Repo", "/other"]),
        request=object(), user_id="u1", tier=None))

    assert len(service.calls) == 1
    call = service.calls[0]
    assert call["project"] == "C:/repo"
    assert sorted(call["project_aliases"] or []) == sorted(["C:/REPO", "C:/Repo", "/other"])
    assert call["excluded_projects"] is None
    assert response.private.searched_projects == ["C:/repo", "/other"]
    assert response.private.excluded_scopes == []


class _DecisionTenQueue:
    """A stand-in whose per-scope durable/in-memory evidence differs by project name,
    so tests can drive `_private_scope_decision` down each branch precisely."""

    def __init__(self, *, active_states=None):
        # project -> "recovery_required" | "preparing" | "allowed_with_newer_job" | None
        self.active_states = active_states or {}

    def scope_search_state(self, *, owner_user_id, projects):
        for project in projects:
            state = self.active_states.get(project)
            if state == "recovery_required":
                return SimpleNamespace(
                    active_job_id="job-recover", mutation_started=True,
                    previous_completed_snapshot_available=True, recovery_required=True)
            if state == "preparing":
                return SimpleNamespace(
                    active_job_id="job-prep", mutation_started=True,
                    previous_completed_snapshot_available=True, recovery_required=False)
            if state == "allowed_with_newer_job":
                return SimpleNamespace(
                    active_job_id="job-newer", mutation_started=False,
                    previous_completed_snapshot_available=True, recovery_required=False)
        return None

    def inventory_scope_states(self, *, owner_user_id):
        return {}


class _DecisionTenCatalog:
    def __init__(self, *, incomplete_projects=()):
        self.incomplete_projects = list(incomplete_projects)

    def equivalent_project_scopes(self, *, owner_user_id, project, kind="code"):
        return [project]

    def code_scope_visibility(self, *, owner_user_id, projects):
        return None  # no durable marker; the in-memory queue state carries the case

    def list_incomplete_code_scopes(self, *, owner_user_id):
        return [{"project": p} for p in self.incomplete_projects]


def _decision_ten_service(monkeypatch, *, catalog, queue):
    from app.api.routes import code_kb
    from contextlib import contextmanager

    class Service:
        def __init__(self):
            self.calls = []

        @contextmanager
        def scope_snapshot(self, *, owner_user_id):
            yield

        def search(self, request, *, owner_user_id, project_aliases=None, excluded_projects=None):
            self.calls.append({
                "project": request.project, "project_aliases": project_aliases,
                "excluded_projects": excluded_projects,
            })
            return CodeSearchResponse(hits=[], total=0)

    service = Service()
    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "get_code_kb_service", lambda: service)
    monkeypatch.setattr(code_kb, "get_index_catalog_service", lambda: catalog)
    monkeypatch.setattr(code_kb, "get_existing_code_index_queue", lambda: queue)
    return service


def test_federated_search_single_scope_route_is_unaffected_by_decision_ten(monkeypatch):
    """Control: the OLD /api/code/search (project=None, whole-scope fail-closed) must
    behave exactly as before -- decision 10's per-scope exclusion is federated-search
    ONLY, never applied to the single-project route."""
    from app.api.routes import code_kb
    from app.services.code_kb.schemas import CodeSearchRequest

    catalog = _DecisionTenCatalog(incomplete_projects=["/withheld"])
    queue = _DecisionTenQueue(active_states={"/withheld": "preparing"})
    service = _decision_ten_service(monkeypatch, catalog=catalog, queue=queue)

    response = asyncio.run(code_kb.search_code(
        CodeSearchRequest(query="q"), request=object(), user_id="u1"))

    # Owner-wide search has no `/withheld`-specific marker of its own, so this must
    # take the unmodified owner-wide branch and simply search (no exclusion machinery).
    assert response.total == 0
    assert service.calls == [{
        "project": None, "project_aliases": None, "excluded_projects": None,
    }]


def test_federated_search_owner_wide_excludes_withheld_scopes_via_must_not(monkeypatch):
    catalog = _DecisionTenCatalog(incomplete_projects=["/withheld"])
    queue = _DecisionTenQueue(active_states={"/withheld": "preparing"})
    service = _decision_ten_service(monkeypatch, catalog=catalog, queue=queue)
    from app.api.routes import code_kb
    from app.services.code_kb.schemas import FederatedCodeSearchRequest

    response = asyncio.run(code_kb.federated_code_search(
        FederatedCodeSearchRequest(query="q"), request=object(), user_id="u1", tier=None))

    assert len(service.calls) == 1
    assert service.calls[0]["project"] is None
    assert service.calls[0]["excluded_projects"] == ["/withheld"]
    assert response.private.excluded_scopes == [
        {"project": "/withheld", "state": "preparing", "active_index_job_id": "job-prep"}]


def test_federated_search_scope_with_newer_job_but_prior_snapshot_is_not_excluded(monkeypatch):
    """B is preparing a NEWER job but has a previous completed snapshot and no mutation
    started -- same rule as the single-scope route's "serve stale" branch: included."""
    catalog = _DecisionTenCatalog(incomplete_projects=["/b"])
    queue = _DecisionTenQueue(active_states={"/b": "allowed_with_newer_job"})
    service = _decision_ten_service(monkeypatch, catalog=catalog, queue=queue)
    from app.api.routes import code_kb
    from app.services.code_kb.schemas import FederatedCodeSearchRequest

    response = asyncio.run(code_kb.federated_code_search(
        FederatedCodeSearchRequest(query="q"), request=object(), user_id="u1", tier=None))

    assert service.calls[0]["excluded_projects"] is None
    assert response.private.excluded_scopes == []


def test_federated_search_explicit_projects_with_one_withheld_excludes_only_that_one(
    monkeypatch,
):
    catalog = _DecisionTenCatalog()
    queue = _DecisionTenQueue(active_states={"/b": "preparing"})
    service = _decision_ten_service(monkeypatch, catalog=catalog, queue=queue)
    from app.api.routes import code_kb
    from app.services.code_kb.schemas import FederatedCodeSearchRequest

    response = asyncio.run(code_kb.federated_code_search(
        FederatedCodeSearchRequest(query="q", projects=["/a", "/b"]),
        request=object(), user_id="u1", tier=None))

    assert len(service.calls) == 1
    assert service.calls[0]["project"] == "/a"
    assert service.calls[0]["project_aliases"] is None
    assert service.calls[0]["excluded_projects"] is None
    assert response.private.excluded_scopes == [
        {"project": "/b", "state": "preparing", "active_index_job_id": "job-prep"}]
    assert response.private.searched_projects == ["/a", "/b"]


def test_federated_search_every_explicit_project_withheld_is_not_reported_as_completed(
    monkeypatch,
):
    # Zero hits with scope_snapshot_state="completed" reads as "no match"; when every
    # requested scope is withheld the private leg must say so, like the single-scope route.
    catalog = _DecisionTenCatalog()
    queue = _DecisionTenQueue(active_states={"/b": "preparing"})
    service = _decision_ten_service(monkeypatch, catalog=catalog, queue=queue)
    from app.api.routes import code_kb
    from app.services.code_kb.schemas import FederatedCodeSearchRequest

    response = asyncio.run(code_kb.federated_code_search(
        FederatedCodeSearchRequest(query="q", projects=["/b"]),
        request=object(), user_id="u1", tier=None))

    assert service.calls == []
    assert response.hits == []
    assert response.private.scope_snapshot_state == "preparing"
    assert response.private.notice and "excluded_scopes" in response.private.notice
    assert response.private.excluded_scopes == [
        {"project": "/b", "state": "preparing", "active_index_job_id": "job-prep"}]


def test_federated_search_catalog_failure_withholds_whole_owner_wide_private_leg(
    monkeypatch,
):
    from contextlib import contextmanager
    from app.api.routes import code_kb
    from app.services.code_kb.schemas import FederatedCodeSearchRequest

    class BrokenCatalog:
        def list_incomplete_code_scopes(self, *, owner_user_id):
            raise RuntimeError("sqlite unavailable")

    class Service:
        def __init__(self):
            self.search_calls = 0

        @contextmanager
        def scope_snapshot(self, *, owner_user_id):
            yield

        def search(self, *_args, **_kwargs):
            self.search_calls += 1
            raise AssertionError("must not search when the catalog read failed")

    service = Service()
    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "get_code_kb_service", lambda: service)
    monkeypatch.setattr(code_kb, "get_index_catalog_service", lambda: BrokenCatalog())
    monkeypatch.setattr(code_kb, "get_existing_code_index_queue", lambda: None)

    response = asyncio.run(code_kb.federated_code_search(
        FederatedCodeSearchRequest(query="q"), request=object(), user_id="u1", tier=None))

    assert service.search_calls == 0
    assert response.private.scope_snapshot_state == "unavailable"
    assert response.private.candidate_count == 0


def test_federated_search_runs_private_and_public_legs_concurrently(monkeypatch):
    from app.api.routes import code_kb
    from app.services.code_kb.schemas import CodeSearchResponse, FederatedCodeSearchRequest

    class SlowPrivateService(_NoFenceService):
        def search(self, request, *, owner_user_id, project_aliases=None, excluded_projects=None):
            time.sleep(0.5)
            return super().search(
                request, owner_user_id=owner_user_id, project_aliases=project_aliases,
                excluded_projects=excluded_projects)

    class SlowPublicService:
        def search(self, request):
            time.sleep(0.5)
            return _federated_public_response()

    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "resolve_admin", lambda *_args: "u1")
    monkeypatch.setattr(
        code_kb, "get_code_kb_service",
        lambda: SlowPrivateService(CodeSearchResponse(hits=[], total=0)))
    monkeypatch.setattr(code_kb, "get_public_code_search_service", lambda: SlowPublicService())

    async def run():
        return await code_kb.federated_code_search(
            FederatedCodeSearchRequest(query="q", include_public=True),
            request=object(), user_id="u1", tier="admin")

    started = time.perf_counter()
    asyncio.run(run())
    elapsed = time.perf_counter() - started
    assert elapsed < 0.9, elapsed


def test_federated_search_anonymous_non_lan_rejected(client):
    response = client.post("/api/code/federated-search", json={"query": "q"})
    assert response.status_code == 401


def test_federated_search_public_leg_unknown_source_id_is_a_client_error_not_500(
    monkeypatch,
):
    """The public leg's errors (e.g. an unknown source_id) must map to the SAME status
    /api/public-code/search gives for the identical failure -- reusing that route's
    `_run_service` error mapping -- never an unhandled 500."""
    from fastapi import HTTPException
    from app.api.routes import code_kb
    from app.services.code_kb.schemas import CodeSearchResponse, FederatedCodeSearchRequest
    from app.services.public_code.source_catalog import PublicSourceSelectionError

    class FailingPublicService:
        def search(self, _request):
            raise PublicSourceSelectionError("Unknown public-code source IDs: missing")

    monkeypatch.setattr(code_kb, "resolve_owner", lambda *_args: "u1")
    monkeypatch.setattr(code_kb, "resolve_admin", lambda *_args: "u1")
    monkeypatch.setattr(
        code_kb, "get_code_kb_service",
        lambda: _NoFenceService(CodeSearchResponse(hits=[], total=0)))
    monkeypatch.setattr(
        code_kb, "get_public_code_search_service", lambda: FailingPublicService())

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(code_kb.federated_code_search(
            FederatedCodeSearchRequest(
                query="q", include_public=True, source_ids=["missing"]),
            request=object(), user_id="u1", tier="admin"))

    assert exc_info.value.status_code == 422
    assert "Unknown public-code source IDs" in str(exc_info.value.detail)
