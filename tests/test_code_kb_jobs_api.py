"""TDD for the async code-KB index HTTP surface.

POST /api/code/index → 202 {job_id, status} immediately (no long-held request);
GET  /api/code/jobs/{job_id} → per-owner job progress;
GET  /api/code/jobs → the caller's recent jobs (newest first).
Owner scoping mirrors the sync endpoints: header identity, never the body.
"""
from __future__ import annotations

import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.code_kb.jobs import CodeIndexJobQueue
from app.services.code_kb.service import CodeKBService

from tests.test_code_kb_api import FakeStore, _CatalogSpy, _embed, _index_body, _u


@pytest.fixture
def client():
    svc = CodeKBService(embed_fn=_embed, vector_store=FakeStore())
    queue = CodeIndexJobQueue(service=svc, catalog_service=_CatalogSpy())
    from app.api.routes import code_kb
    code_kb.get_code_kb_service = lambda: svc
    code_kb.get_code_index_queue = lambda: queue
    code_kb.get_existing_code_index_queue = lambda: queue
    code_kb.get_index_catalog_service = lambda: queue.catalog_service
    app = FastAPI()
    app.include_router(code_kb.router, prefix="/api/code")
    with TestClient(app) as c:
        yield c
    queue.shutdown(timeout=5)


def _wait_status(client, job_id, want, headers, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/code/jobs/{job_id}", headers=headers)
        if r.status_code == 200 and r.json()["status"] == want:
            return r.json()
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached {want}")


def test_index_returns_202_with_job_id_immediately(client):
    r = client.post("/api/code/index", headers=_u("u1"), json=_index_body())
    assert r.status_code == 202
    body = r.json()
    assert body["job_id"] and body["status"] in ("queued", "running")
    assert body["chunks_total"] == 1


def test_client_job_id_is_the_durable_status_receipt(client):
    body = _index_body("durable-receipt")
    body["client_job_id"] = "c" * 32

    receipt = client.post("/api/code/index", headers=_u("u1"), json=body)

    assert receipt.status_code == 202
    assert receipt.json()["job_id"] == "c" * 32
    done = _wait_status(client, "c" * 32, "completed", _u("u1"))
    assert done["job_id"] == "c" * 32


def test_client_job_id_cannot_be_rebound_to_another_request(client):
    first = _index_body("first-project")
    first["client_job_id"] = "d" * 32
    assert client.post("/api/code/index", headers=_u("u1"), json=first).status_code == 202

    conflicting = _index_body("second-project")
    conflicting["client_job_id"] = "d" * 32
    response = client.post("/api/code/index", headers=_u("u1"), json=conflicting)

    assert response.status_code == 409
    assert "already bound" in response.json()["detail"]


def test_delete_only_job_reports_removed_path_count(client):
    body = {
        "project": "delete-only", "host": "h1", "root_dir": "/repo",
        "incremental": True, "files": [], "removed_paths": ["gone.py"],
    }
    receipt = client.post("/api/code/index", headers=_u("u1"), json=body)

    assert receipt.status_code == 202
    assert receipt.json()["removed_total"] == 1
    done = _wait_status(client, receipt.json()["job_id"], "completed", _u("u1"))
    assert done["removed_total"] == 1


def test_job_progress_pollable_until_completed(client):
    job_id = client.post("/api/code/index", headers=_u("u1"), json=_index_body()).json()["job_id"]
    done = _wait_status(client, job_id, "completed", _u("u1"))
    assert done["indexed"] == 1 and done["progress"] == 1.0
    # and the data actually landed — search finds it
    hits = client.post("/api/code/search", headers=_u("u1"),
                       json={"query": "qdrant upsert vector"}).json()
    assert hits["total"] == 1


def test_job_visibility_scoped_to_owner(client):
    job_id = client.post("/api/code/index", headers=_u("alice"), json=_index_body()).json()["job_id"]
    assert client.get(f"/api/code/jobs/{job_id}", headers=_u("bob")).status_code == 404
    assert client.get(f"/api/code/jobs/{job_id}", headers=_u("alice")).status_code == 200


def test_jobs_list_returns_own_jobs_newest_first(client):
    id1 = client.post("/api/code/index", headers=_u("u1"),
                      json=_index_body("p1")).json()["job_id"]
    id2 = client.post("/api/code/index", headers=_u("u1"),
                      json=_index_body("p2")).json()["job_id"]
    client.post("/api/code/index", headers=_u("other"), json=_index_body("px"))
    _wait_status(client, id2, "completed", _u("u1"))
    jobs = client.get("/api/code/jobs", headers=_u("u1")).json()["jobs"]
    assert [j["job_id"] for j in jobs] == [id2, id1]
    assert all(j["project"] in ("p1", "p2") for j in jobs)


def test_anonymous_rejected(client):
    assert client.post("/api/code/index", json=_index_body()).status_code == 401
    assert client.get("/api/code/jobs").status_code == 401


def test_cancel_completed_job_returns_409_already_ended(client):
    job_id = client.post("/api/code/index", headers=_u("u1"), json=_index_body()).json()["job_id"]
    _wait_status(client, job_id, "completed", _u("u1"))

    r = client.post(f"/api/code/jobs/{job_id}/cancel", headers=_u("u1"))

    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "code_index_already_ended"


def test_cancel_unknown_job_returns_404(client):
    r = client.post("/api/code/jobs/does-not-exist/cancel", headers=_u("u1"))
    assert r.status_code == 404


def test_cancel_another_owners_job_returns_404_not_403(client):
    job_id = client.post("/api/code/index", headers=_u("alice"), json=_index_body()).json()["job_id"]
    r = client.post(f"/api/code/jobs/{job_id}/cancel", headers=_u("bob"))
    assert r.status_code == 404


def test_cancel_anonymous_rejected(client):
    assert client.post("/api/code/jobs/anything/cancel").status_code == 401


def test_cancel_queued_job_then_next_job_for_scope_completes_and_is_searchable(client):
    """Regression (F2-a): a cancelled job must not poison the scope for the next job."""
    from app.api.routes import code_kb

    gate = threading.Event()
    queue = code_kb.get_code_index_queue()
    original_index = queue.service.index

    def slow(request, *, owner_user_id, progress_cb=None, **callbacks):
        gate.wait(timeout=5)
        return original_index(
            request, owner_user_id=owner_user_id, progress_cb=progress_cb, **callbacks)

    queue.service.index = slow
    try:
        blocker_id = client.post(
            "/api/code/index", headers=_u("u1"), json=_index_body("blocker")).json()["job_id"]
        _wait_status(client, blocker_id, "running", _u("u1"))

        queued_id = client.post(
            "/api/code/index", headers=_u("u1"), json=_index_body("cancel-then-reindex"),
        ).json()["job_id"]
        assert client.get(f"/api/code/jobs/{queued_id}", headers=_u("u1")).json()["status"] == "queued"

        cancel_resp = client.post(f"/api/code/jobs/{queued_id}/cancel", headers=_u("u1"))
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["status"] == "cancelled"

        gate.set()
        _wait_status(client, blocker_id, "completed", _u("u1"))

        next_id = client.post(
            "/api/code/index", headers=_u("u1"), json=_index_body("cancel-then-reindex"),
        ).json()["job_id"]
        done = _wait_status(client, next_id, "completed", _u("u1"))
        assert done["indexed"] == 1

        hits = client.post(
            "/api/code/search", headers=_u("u1"),
            json={"query": "qdrant upsert vector", "project": "cancel-then-reindex"},
        ).json()
        assert hits["total"] == 1
        assert hits["scope_snapshot_state"] == "completed"
    finally:
        gate.set()
