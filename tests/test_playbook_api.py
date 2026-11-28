"""TDD for per-user + admin ACE playbook routes (compute side)."""
from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.database import Base
from app.services.playbook.service import PlaybookService

_DIM = 32


def _embed(texts):
    out = []
    for t in texts:
        v = [0.0] * _DIM
        for w in t.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % _DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v))
        out.append([x / n for x in v] if n else v)
    return out


@pytest.fixture
def client():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def sf():
        s = session_local()
        try:
            yield s
        finally:
            s.close()

    svc = PlaybookService(embed_fn=_embed, session_factory=sf)
    from app.api.routes import playbook, admin_console
    playbook.get_playbook_service = lambda: svc
    admin_console.get_playbook_service = lambda: svc

    app = FastAPI()
    app.include_router(playbook.router, prefix="/api/playbook")
    app.include_router(admin_console.router, prefix="/api/admin/console")
    return TestClient(app)


def _u(uid="u1"):
    return {"X-Retrival-User-Id": uid}


def _admin():
    return {"X-Retrival-User-Id": "9", "X-Retrival-User-Tier": "admin"}


def test_record_list_search_scoped_to_owner(client):
    r = client.post("/api/playbook/bullets", headers=_u("u1"),
                    json={"project_slug": "retrival", "content": "wait=True on qdrant upsert"})
    assert r.status_code == 200
    assert client.get("/api/playbook/bullets?project_slug=retrival", headers=_u("u1")).json()["total"] == 1
    # another user sees nothing
    assert client.get("/api/playbook/bullets", headers=_u("u2")).json()["total"] == 0
    hits = client.post("/api/playbook/search", headers=_u("u1"),
                       json={"project_slug": "retrival", "query": "qdrant upsert", "top_k": 5}).json()
    assert hits["total"] == 1


def test_owner_taken_from_header_not_body(client):
    r = client.post("/api/playbook/bullets", headers=_u("real-user"),
                    json={"project_slug": "p", "content": "x", "owner_user_id": "SPOOFED"})
    assert r.status_code == 200
    # visible to real-user, not to the spoofed id
    assert client.get("/api/playbook/bullets", headers=_u("real-user")).json()["total"] == 1
    assert client.get("/api/playbook/bullets", headers=_u("SPOOFED")).json()["total"] == 0


def test_anonymous_non_lan_rejected(client):
    # TestClient default host is "testclient" (not LAN) -> resolve_owner 401
    assert client.get("/api/playbook/bullets").status_code == 401
    assert client.post("/api/playbook/bullets",
                       json={"project_slug": "p", "content": "x"}).status_code == 401


def test_mark_and_delete(client):
    b = client.post("/api/playbook/bullets", headers=_u("u1"),
                    json={"project_slug": "p", "content": "a lesson"}).json()
    marked = client.post(f"/api/playbook/bullets/{b['id']}/mark", headers=_u("u1"),
                         json={"helpful": True}).json()
    assert marked["helpful"] == 1
    # cross-user mark -> 404
    assert client.post(f"/api/playbook/bullets/{b['id']}/mark", headers=_u("u2"),
                       json={"helpful": True}).status_code == 404
    assert client.request("DELETE", f"/api/playbook/bullets/{b['id']}", headers=_u("u1")).status_code == 200
    assert client.get("/api/playbook/bullets", headers=_u("u1")).json()["total"] == 0


def test_admin_playbook_is_cross_user(client):
    client.post("/api/playbook/bullets", headers=_u("u1"), json={"project_slug": "p", "content": "l1"})
    client.post("/api/playbook/bullets", headers=_u("u2"), json={"project_slug": "p", "content": "l2"})
    # non-admin blocked
    assert client.get("/api/admin/console/playbook/bullets",
                      headers={"X-Retrival-User-Id": "3", "X-Retrival-User-Tier": "paying"}).status_code == 403
    full = client.get("/api/admin/console/playbook/bullets", headers=_admin()).json()
    assert full["total"] == 2
    scoped = client.get("/api/admin/console/playbook/bullets?owner_user_id=u1", headers=_admin()).json()
    assert scoped["total"] == 1
