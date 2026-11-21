"""TDD for admin console + per-user index-report routes (compute side)."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.database import Base
from app.api.identity import resolve_admin, resolve_owner, INTERNAL_ADMIN_ID
from app.services.index_catalog.repository import IndexCatalogRepository
from app.services.index_catalog.schemas import IndexFileRecord, IndexReportRequest
from app.services.index_catalog.service import IndexCatalogService
from app.services.journal.repository import JournalRepository
from app.services.journal.schemas import JournalRecordRequest
from app.services.journal.service import JournalService


def _fake_embed(texts, include_sparse):
    return [[float(len(t)), 1.0] for t in texts], None


def _factory():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
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

    return session_factory


@pytest.fixture
def env():
    sf = _factory()
    journal = JournalService(
        repository=JournalRepository(session_factory=sf),
        vector_store=SimpleNamespace(
            upsert_chunks=lambda *a, **k: 0, delete_by_entry_id=lambda e: 0,
        ),
        embed_fn=_fake_embed,
    )
    index = IndexCatalogService(repository=IndexCatalogRepository(session_factory=sf))
    # Seed two users' journal + index data.
    for owner in ("user-1", "user-2"):
        journal.record(JournalRecordRequest(
            project_slug="retrival", client="codex", source_kind="memory",
            title=f"note {owner}", body=f"observability work for {owner}", owner_user_id=owner,
        ))
    index.report(IndexReportRequest(
        host="4090", project="retrival", root_dir="/repo",
        files=[IndexFileRecord(rel_path="core/db/models.py", chunk_count=4)],
    ), owner_user_id="user-1")
    index.report(IndexReportRequest(
        host="kss", project="other", root_dir="/o",
        files=[IndexFileRecord(rel_path="src/a.ts", chunk_count=2)],
    ), owner_user_id="user-2")

    from app.api.routes import admin_console, index_catalog
    admin_console.get_journal_service = lambda: journal
    admin_console.get_index_catalog_service = lambda: index
    index_catalog.get_index_catalog_service = lambda: index

    app = FastAPI()
    app.include_router(admin_console.router, prefix="/api/admin/console")
    app.include_router(index_catalog.router, prefix="/api/index")
    return SimpleNamespace(client=TestClient(app), journal=journal, index=index)


def _admin():
    return {"X-Retrival-User-Id": "9", "X-Retrival-User-Tier": "admin"}


def test_admin_routes_reject_anonymous_and_non_admin(env):
    c = env.client
    assert c.get("/api/admin/console/overview").status_code == 401
    assert c.get(
        "/api/admin/console/overview",
        headers={"X-Retrival-User-Id": "3", "X-Retrival-User-Tier": "paying"},
    ).status_code == 403
    assert c.get("/api/admin/console/overview", headers=_admin()).status_code == 200


def test_overview_and_users_are_cross_user(env):
    c = env.client
    ov = c.get("/api/admin/console/overview", headers=_admin()).json()
    assert ov["journal"]["entries"] == 2
    assert ov["index"]["files"] == 2

    users = c.get("/api/admin/console/users", headers=_admin()).json()
    owners = {u["owner_user_id"] for u in users["users"]}
    assert {"user-1", "user-2"} <= owners


def test_admin_journal_search_keyword_scopes_by_owner(env):
    c = env.client
    allr = c.post("/api/admin/console/journal/search",
                  headers=_admin(), json={"query": "observability", "mode": "keyword"}).json()
    scoped = c.post("/api/admin/console/journal/search",
                    headers=_admin(),
                    json={"query": "observability", "mode": "keyword", "owner_user_id": "user-1"}).json()
    assert allr["total"] == 2
    assert scoped["total"] == 1


def test_admin_inventory_tree_and_owner_filter(env):
    c = env.client
    full = c.get("/api/admin/console/index/inventory", headers=_admin()).json()
    assert full["totals"]["files"] == 2
    assert full["totals"]["hosts"] == 2
    assert all(
        project["dirs"] == []
        for user in full["users"]
        for host in user["hosts"]
        for project in host["projects"]
    )

    scoped = c.get(
        "/api/admin/console/index/inventory"
        "?owner_user_id=user-1&project=retrival&detail=true",
        headers=_admin(),
    ).json()
    assert scoped["totals"]["files"] == 1
    assert scoped["users"][0]["hosts"][0]["projects"][0]["dirs"][0]["files"]


def test_index_report_takes_owner_from_gateway_not_body(env):
    c = env.client
    resp = c.post("/api/index/report", headers={"X-Retrival-User-Id": "user-9"}, json={
        "host": "laptop", "project": "p", "root_dir": "/p",
        "owner_user_id": "SPOOFED",
        "files": [{"rel_path": "x.py", "chunk_count": 1}],
    })
    assert resp.status_code == 200
    assert resp.json()["owner_user_id"] == "user-9"
    # Anonymous non-LAN report is rejected.
    assert c.post("/api/index/report", json={"host": "h", "project": "p", "files": []}).status_code == 401


def test_identity_resolvers_units():
    from app.api import identity

    lan = SimpleNamespace(
        client=SimpleNamespace(host="10.100.100.9"),
        scope={"server": ("0.0.0.0", identity.INTERNAL_API_PORT)},
    )
    wan = SimpleNamespace(client=SimpleNamespace(host="8.8.8.8"))

    assert resolve_owner(lan, None) == INTERNAL_ADMIN_ID
    assert resolve_owner(wan, "u5") == "u5"
    with pytest.raises(HTTPException):
        resolve_owner(wan, None)

    assert resolve_admin(lan, None, None) == INTERNAL_ADMIN_ID
    assert resolve_admin(wan, "u5", "admin") == "u5"
    with pytest.raises(HTTPException):
        resolve_admin(wan, "u5", "paying")


def test_public_door_never_resolves_internal_admin():
    """Internal vs public is separated by LISTENING PORT (source IP is unreliable on the
    SNAT'd lab overlay). The public door (PUBLIC_API_PORT, where the xray tunnel lands)
    must NEVER grant internal-admin — even for a private-looking source; it requires a
    gateway identity. The internal door keeps the LAN-admin shortcut."""
    from app.api import identity

    def req(host, port):
        return SimpleNamespace(
            client=SimpleNamespace(host=host),
            scope={"server": ("0.0.0.0", port)},
        )

    internal, public = identity.INTERNAL_API_PORT, identity.PUBLIC_API_PORT
    assert internal == 8000
    assert public == 8001

    # Internal door: an overlay peer is the trusted internal-admin.
    assert identity.resolve_owner(req("10.100.100.9", internal), None) == identity.INTERNAL_ADMIN_ID
    assert identity.resolve_admin(req("10.100.100.9", internal), None, None) == identity.INTERNAL_ADMIN_ID

    # Public door: NEVER internal-admin, even from a private-looking source → 401.
    for host in ("10.100.100.9", "127.0.0.1", "8.8.8.8"):
        with pytest.raises(HTTPException) as ei:
            identity.resolve_owner(req(host, public), None)
        assert ei.value.status_code == 401
        with pytest.raises(HTTPException):
            identity.resolve_admin(req(host, public), None, None)

    # Public door WITH a gateway-injected identity → scoped to that owner (never admin).
    assert identity.resolve_owner(req("127.0.0.1", public), "u5") == "u5"
    assert identity.resolve_admin(req("127.0.0.1", public), "u5", "admin") == "u5"

    # A private peer on any unrelated listener must not inherit internal-admin merely
    # because that port is not the configured public door.
    unrelated = 19000
    with pytest.raises(HTTPException) as ei:
        identity.resolve_owner(req("10.100.100.9", unrelated), None)
    assert ei.value.status_code == 401


def _identity_import_with_ports(internal: str, public: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({"INTERNAL_API_PORT": internal, "PUBLIC_API_PORT": public})
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from app.api.identity import INTERNAL_API_PORT, PUBLIC_API_PORT, "
                "LISTEN_ADDRESSES; "
                "print(INTERNAL_API_PORT, PUBLIC_API_PORT, LISTEN_ADDRESSES)"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_api_listener_ports_support_isolated_host_overrides():
    result = _identity_import_with_ports("18080", "18081")
    assert result.returncode == 0, result.stderr
    assert "18080 18081" in result.stdout
    assert "('0.0.0.0', 18080)" in result.stdout
    assert "('127.0.0.1', 18081)" in result.stdout


@pytest.mark.parametrize(
    ("internal", "public", "expected"),
    [
        ("18080", "18080", "must use different ports"),
        ("not-a-port", "18081", "INTERNAL_API_PORT must be an integer"),
        ("18080", "65536", "PUBLIC_API_PORT must be between 1 and 65535"),
    ],
)
def test_api_listener_ports_fail_fast_with_actionable_errors(internal, public, expected):
    result = _identity_import_with_ports(internal, public)
    assert result.returncode != 0
    assert expected in result.stderr
