"""Aggregate endpoints (/api/conferences, /api/stats, /api/dashboard/distribution)
are LOGIN-GATED (no credit charge) — W127b reverses the earlier "keep aggregates
anonymous" rule (W109). Anonymous → 401; logged-in → 200 passthrough.
"""
import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.auth import accounts
from gateway.settings import GatewaySettings

PEPPER = "test-pepper"
GATED = ["/api/conferences", "/api/stats", "/api/dashboard/distribution"]


class AggProxy:
    async def request(self, method, path, *, headers=None, params=None, content=b""):
        return httpx.Response(200, json={"ok": True, "path": path})

    async def stream(self, method, path, *, headers=None, params=None, content=b""):
        async def gen():
            yield b""
        return 200, {"content-type": "text/event-stream"}, gen()


@pytest.fixture
def app():
    settings = GatewaySettings(db_url="sqlite://", key_pepper=PEPPER,
                               session_secret="s", admin_root_key="sk-root")
    return create_app(settings, proxy=AggProxy())


@pytest.fixture
def client(app):
    return TestClient(app)


def _mk_key(app):
    db = app.state.session_factory()
    try:
        u = accounts.register(db, oidc_sub="cdk:agg", username="agg")
        key, _ = accounts.issue_key(db, u, PEPPER)
        db.commit()
        return key
    finally:
        db.close()


def test_aggregates_require_login(client):
    for path in GATED:
        assert client.get(path).status_code == 401, path


def test_aggregates_ok_when_authed(app, client):
    key = _mk_key(app)
    for path in GATED:
        r = client.get(path, headers={"Authorization": f"Bearer {key}"})
        assert r.status_code == 200, path
        assert r.json()["ok"] is True


def test_aggregates_not_charged(app, client):
    """Gated but FREE — a paying user's balance is untouched."""
    from gateway.db.models import Tier
    db = app.state.session_factory()
    u = accounts.register(db, oidc_sub="cdk:aggpay", username="aggpay")
    u.tier = Tier.paying
    u.balance = 7
    key, _ = accounts.issue_key(db, u, PEPPER)
    db.commit()
    db.close()
    for path in GATED:
        client.get(path, headers={"Authorization": f"Bearer {key}"})
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {key}"}).json()
    assert me["balance"] == 7
