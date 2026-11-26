"""Journal / code KB / index inventory are open to any logged-in user (paying+),
scoped per-user by forwarded identity headers. Paying users are metered per the
per-op costs (journal write 0.1, code index 1, code search 0.1); journal search +
index report/inventory are free reads. internal/admin are billing-exempt. ACE
playbook stays invite-only (internal+admin). Anonymous → 401.
"""
import json
from decimal import Decimal

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.auth import accounts
from gateway.db.models import Tier
from gateway.settings import GatewaySettings

PEPPER = "test-pepper"


class JournalProxy:
    def __init__(self):
        self.calls = []

    async def request(self, method, path, *, headers=None, params=None, content=b""):
        self.calls.append({
            "method": method,
            "path": path,
            "headers": dict(headers or {}),
            "params": dict(params or {}),
            "body": json.loads(content or b"{}") if content else None,
        })
        if path == "/api/journal/entries" and method == "GET":
            return httpx.Response(200, json={"entries": [], "total": 0})
        return httpx.Response(200, json={"ok": True, "path": path})

    async def stream(self, method, path, *, headers=None, params=None, content=b""):
        async def gen():
            yield b""
        return 200, {"content-type": "text/event-stream"}, gen()


@pytest.fixture
def app():
    settings = GatewaySettings(
        db_url="sqlite://",
        key_pepper=PEPPER,
        session_secret="s",
        admin_root_key="sk-root",
    )
    return create_app(settings, proxy=JournalProxy())


@pytest.fixture
def client(app):
    return TestClient(app)


def _mk_key(app, *, tier=Tier.paying, balance=5):
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


def _balance(client, key):
    return Decimal(str(client.get("/auth/me", headers=_auth(key)).json()["balance"]))


def test_journal_endpoints_require_login(client):
    assert client.post("/api/journal/entries", json={"project_slug": "p", "body": "x"}).status_code == 401
    assert client.post("/api/journal/search", json={"project_slug": "p", "query": "x"}).status_code == 401
    assert client.get("/api/journal/entries?project_slug=p").status_code == 401


def test_code_jobs_endpoints_require_login(client):
    assert client.get("/api/code/jobs").status_code == 401
    assert client.get("/api/code/jobs/abc123").status_code == 401


def test_paying_user_can_use_journal_and_code(app, client):
    """New policy: journal / code / index are open to paying+ (was internal-only)."""
    uid, key = _mk_key(app, tier=Tier.paying, balance=10)
    for method, path, body in [
        ("POST", "/api/journal/entries", {"project_slug": "p", "body": "x"}),
        ("POST", "/api/journal/search", {"project_slug": "p", "query": "x"}),
        ("GET", "/api/journal/entries?project_slug=p", None),
        ("GET", "/api/code/jobs", None),
        ("POST", "/api/code/index", {"project": "p", "host": "h", "root_dir": "/r", "files": []}),
        ("POST", "/api/code/search", {"query": "x"}),
        ("GET", "/api/index/inventory", None),
        ("POST", "/api/index/report", {"host": "h", "project": "p", "files": []}),
    ]:
        r = client.request(method, path, headers=_auth(key), json=body)
        assert r.status_code == 200, f"paying should pass {method} {path}, got {r.status_code}"


def test_playbook_stays_internal_only(app, client):
    """ACE playbook is NOT opened to paying users — still internal+admin."""
    _, paying = _mk_key(app, tier=Tier.paying, balance=5)
    assert client.get("/api/playbook/bullets", headers=_auth(paying)).status_code == 403
    assert client.post("/api/playbook/search", headers=_auth(paying),
                       json={"project_slug": "p", "query": "x"}).status_code == 403
    _, internal = _mk_key(app, tier=Tier.internal, balance=0)
    assert client.get("/api/playbook/bullets", headers=_auth(internal)).status_code == 200


def test_paying_billing_per_op(app, client):
    """Paying user: journal write 0.1, code index 1, code search 0.1 are charged;
    journal search + index report/inventory + GETs are free."""
    _, key = _mk_key(app, tier=Tier.paying, balance=10)
    start = _balance(client, key)

    client.post("/api/journal/entries", headers=_auth(key), json={"project_slug": "p", "body": "x"})
    assert start - _balance(client, key) == Decimal("0.1")

    b = _balance(client, key)
    client.post("/api/journal/search", headers=_auth(key), json={"project_slug": "p", "query": "x"})
    assert _balance(client, key) == b  # search is free

    b = _balance(client, key)
    client.get("/api/index/inventory", headers=_auth(key))
    client.post("/api/index/report", headers=_auth(key), json={"host": "h", "project": "p", "files": []})
    assert _balance(client, key) == b  # index bookkeeping is free

    b = _balance(client, key)
    client.post("/api/code/index", headers=_auth(key),
                json={"project": "p", "host": "h", "root_dir": "/r", "files": []})
    assert b - _balance(client, key) == Decimal("1")  # code index = 1

    b = _balance(client, key)
    client.post("/api/code/search", headers=_auth(key), json={"query": "x"})
    assert b - _balance(client, key) == Decimal("0.1")  # code search = 0.1


def test_paying_federated_search_billed_and_forwards_identity_and_tier(app, client):
    """/api/code/federated-search: same billing (0.1) and identity-forward pattern as
    /api/code/search (docs/federated_code_search_plan.md A4)."""
    uid, key = _mk_key(app, tier=Tier.paying, balance=10)
    start = _balance(client, key)

    r = client.post("/api/code/federated-search", headers=_auth(key), json={"query": "x"})
    assert r.status_code == 200, r.text
    assert start - _balance(client, key) == Decimal("0.1")

    call = app.state.proxy.calls[-1]
    assert call["path"] == "/api/code/federated-search"
    assert call["headers"]["X-Retrival-User-Id"] == str(uid)
    assert call["headers"]["X-Retrival-User-Tier"] == "paying"


def test_internal_not_charged(app, client):
    """member (internal) tier uses everything free."""
    uid, key = _mk_key(app, tier=Tier.internal, balance=5)
    for method, path, body in [
        ("POST", "/api/journal/entries", {"project_slug": "p", "body": "x"}),
        ("POST", "/api/code/index", {"project": "p", "host": "h", "root_dir": "/r", "files": []}),
        ("POST", "/api/code/search", {"query": "x"}),
    ]:
        client.request(method, path, headers=_auth(key), json=body)
    assert _balance(client, key) == Decimal("5")  # untouched
    # identity still forwarded so compute scopes to the owner (privacy).
    call = app.state.proxy.calls[-1]
    assert call["headers"]["X-Retrival-User-Id"] == str(uid)
    assert call["headers"]["X-Retrival-User-Tier"] == "internal"


def test_paying_insufficient_credits_402(app, client):
    _, key = _mk_key(app, tier=Tier.paying, balance=0)
    r = client.post("/api/code/index", headers=_auth(key),
                    json={"project": "p", "host": "h", "root_dir": "/r", "files": []})
    assert r.status_code == 402


def test_identity_headers_forwarded_and_client_spoofing_stripped(app, client):
    uid, key = _mk_key(app, tier=Tier.paying, balance=5)
    client.post("/api/journal/entries", headers={**_auth(key), "X-Retrival-User-Id": "999"},
                json={"project_slug": "retrival", "body": "work log"})
    call = app.state.proxy.calls[-1]
    assert call["headers"]["X-Retrival-User-Id"] == str(uid)  # authentic id, not the spoofed 999
    assert call["headers"]["X-Retrival-Username"] == "u_paying"
