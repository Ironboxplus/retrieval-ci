"""Admin console is admin-gated at the gateway; index report is login-gated."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.auth import accounts
from gateway.db.models import Tier
from gateway.settings import GatewaySettings

PEPPER = "test-pepper"


class RecordingProxy:
    def __init__(self):
        self.calls = []

    async def request(self, method, path, *, headers=None, params=None, content=b""):
        self.calls.append({
            "method": method, "path": path, "headers": dict(headers or {}),
            "params": dict(params or {}),
            "body": json.loads(content or b"{}") if content else None,
        })
        return httpx.Response(200, json={"ok": True, "path": path})

    async def stream(self, method, path, *, headers=None, params=None, content=b""):
        async def gen():
            yield b""
        return 200, {"content-type": "text/event-stream"}, gen()


@pytest.fixture
def app():
    settings = GatewaySettings(
        db_url="sqlite://", key_pepper=PEPPER, session_secret="s",
        admin_root_key="sk-root", search_cost=1,
    )
    return create_app(settings, proxy=RecordingProxy())


@pytest.fixture
def client(app):
    return TestClient(app)


def _mk_key(app, *, tier=Tier.paying, balance=0):
    db = app.state.session_factory()
    try:
        user = accounts.register(db, oidc_sub=f"cdk:{tier.value}", username=f"u_{tier.value}")
        user.tier = tier
        user.balance = balance
        key, _ = accounts.issue_key(db, user, PEPPER)
        db.commit()
        return user.id, key
    finally:
        db.close()


def _auth(key):
    return {"Authorization": f"Bearer {key}"}


def test_admin_console_requires_admin_tier(app, client):
    assert client.get("/api/admin/console/overview").status_code == 401

    _, paying = _mk_key(app, tier=Tier.paying)
    assert client.get("/api/admin/console/overview", headers=_auth(paying)).status_code == 403

    uid, admin = _mk_key(app, tier=Tier.admin)
    resp = client.get("/api/admin/console/overview", headers=_auth(admin))
    assert resp.status_code == 200
    call = app.state.proxy.calls[-1]
    assert call["path"] == "/api/admin/console/overview"
    assert call["headers"]["X-Retrival-User-Tier"] == "admin"
    assert call["headers"]["X-Retrival-User-Id"] == str(uid)


def test_admin_console_subpaths_forward(app, client):
    _, admin = _mk_key(app, tier=Tier.admin)
    client.get("/api/admin/console/index/inventory?owner_user_id=user-1", headers=_auth(admin))
    call = app.state.proxy.calls[-1]
    assert call["path"] == "/api/admin/console/index/inventory"
    assert call["params"] == {"owner_user_id": "user-1"}


def test_index_report_open_to_paying_free_and_forwards_identity(app, client):
    assert client.post("/api/index/report", json={"host": "h", "project": "p", "files": []}).status_code == 401

    # Index bookkeeping is now open to any logged-in user (paying+) and is free —
    # it's a catalog write, not compute. Identity is forwarded so it scopes per-user.
    uid, paying = _mk_key(app, tier=Tier.paying, balance=2)
    resp = client.post("/api/index/report", headers=_auth(paying),
                       json={"host": "laptop", "project": "p", "files": []})
    assert resp.status_code == 200
    call = app.state.proxy.calls[-1]
    assert call["path"] == "/api/index/report"
    assert call["headers"]["X-Retrival-User-Id"] == str(uid)
    assert client.get("/auth/me", headers=_auth(paying)).json()["balance"] == 2  # free
