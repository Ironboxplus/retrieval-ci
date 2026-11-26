"""Structured per-request access log (W128/W131).

EVERY request emits one JSON line on the `gateway.access` logger (→ stdout →
shipped to Loki on the 4090; BWH stays stateless) — authenticated AND anonymous
("所有的请求都要记录"). Authenticated lines carry username + tier so Grafana can
count calls per user for ANY tier — including billing-exempt internal/admin,
which never hit the transactions ledger. Anonymous lines carry null username +
the client `ip` so 401s / public hits are still a complete edge access record.
"""
import asyncio
import json
import logging
import types

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway import access_log
from gateway.app import create_app
from gateway.auth import accounts
from gateway.db.models import Tier
from gateway.settings import GatewaySettings
from gateway.proxy import ComputeError

PEPPER = "test-pepper"


class StubProxy:
    async def request(self, method, path, *, headers=None, params=None, content=b""):
        return httpx.Response(200, json={"ok": True, "path": path})

    async def stream(self, method, path, *, headers=None, params=None, content=b""):
        async def gen():
            yield b""
        return 200, {"content-type": "text/event-stream"}, gen()


@pytest.fixture
def app():
    s = GatewaySettings(db_url="sqlite://", key_pepper=PEPPER,
                        session_secret="s", admin_root_key="sk-root")
    return create_app(s, proxy=StubProxy())


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def access_lines():
    captured = []

    class _H(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    h = _H()
    lg = logging.getLogger("gateway.access")
    lg.addHandler(h)
    yield captured
    lg.removeHandler(h)


def _key(app, *, username, tier=Tier.paying, balance=0):
    db = app.state.session_factory()
    try:
        u = accounts.register(db, oidc_sub=f"cdk:{username}", username=username)
        u.tier = tier
        u.balance = balance
        key, _ = accounts.issue_key(db, u, PEPPER)
        db.commit()
        return key
    finally:
        db.close()


def _access(lines):
    return [json.loads(m) for m in lines if '"evt": "access"' in m]


def test_authenticated_request_emits_access_log(app, client, access_lines):
    key = _key(app, username="m1110", tier=Tier.internal)
    r = client.get("/api/conferences", headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200
    evs = _access(access_lines)
    hit = [e for e in evs if e["path"] == "/api/conferences"]
    assert hit, evs
    e = hit[0]
    assert e["username"] == "m1110"
    assert e["tier"] == "internal"
    assert e["method"] == "GET"
    assert e["status"] == 200


def test_anonymous_request_is_logged_with_null_user(app, client, access_lines):
    # no Authorization → 401 before any user is resolved, but it MUST still be
    # logged (W131: 所有的请求都要记录) with null username + the client ip.
    r = client.get("/api/conferences")
    assert r.status_code == 401
    evs = _access(access_lines)
    hit = [e for e in evs if e["path"] == "/api/conferences"]
    assert hit, evs
    e = hit[0]
    assert e["username"] is None
    assert e["user_id"] is None
    assert e["tier"] is None
    assert e["status"] == 401
    assert e["method"] == "GET"
    assert "ip" in e  # client identifier captured even when anonymous


def test_internal_search_logged_even_though_not_charged(app, client, access_lines):
    key = _key(app, username="m1110", tier=Tier.internal)
    client.post("/api/search", headers={"Authorization": f"Bearer {key}"})
    evs = _access(access_lines)
    search = [e for e in evs if e["path"] == "/api/search"]
    assert search and search[0]["username"] == "m1110"
    assert search[0]["cost"] == 0  # internal exempt → recorded but cost 0


def test_paying_search_records_cost(app, client, access_lines):
    key = _key(app, username="payer", tier=Tier.paying, balance=5)
    client.post("/api/search", headers={"Authorization": f"Bearer {key}"})
    evs = _access(access_lines)
    search = [e for e in evs if e["path"] == "/api/search"]
    assert search and search[0]["cost"] == 1


def test_insufficient_credit_response_records_zero_cost(app, client, access_lines):
    key = _key(app, username="empty-payer", tier=Tier.paying, balance=0)

    response = client.post(
        "/api/search", headers={"Authorization": f"Bearer {key}"})

    assert response.status_code == 402
    search = [e for e in _access(access_lines) if e["path"] == "/api/search"]
    assert search and search[-1]["cost"] == 0


def test_refunded_compute_error_records_zero_cost(access_lines):
    class FailingProxy(StubProxy):
        async def request(self, method, path, *, headers=None, params=None, content=b""):
            raise ComputeError(503, "compute unavailable")

    settings = GatewaySettings(db_url="sqlite://", key_pepper=PEPPER,
                               session_secret="s", admin_root_key="sk-root")
    failing_app = create_app(settings, proxy=FailingProxy())
    key = _key(failing_app, username="refunded-payer", tier=Tier.paying, balance=5)

    with TestClient(failing_app) as failing_client:
        response = failing_client.post(
            "/api/search", headers={"Authorization": f"Bearer {key}"})

    assert response.status_code == 502
    search = [e for e in _access(access_lines) if e["path"] == "/api/search"]
    assert search and search[-1]["cost"] == 0


# --- W128 Phase 2: ship the access record to the 4090 Loki relay over the tunnel ---

class _User:
    def __init__(self, uid, username, tier):
        self.id, self.username, self.tier = uid, username, tier


def test_emit_returns_the_logged_record(access_lines):
    """emit() returns the exact dict it logs, so the middleware can ship it without
    rebuilding (single source of truth for the stdout line and the Loki line)."""
    u = _User(7, "m1110", Tier.internal)
    rec = access_log.emit(u, method="POST", path="/api/search",
                          status=200, duration_ms=12.3, cost=0)
    logged = _access(access_lines)
    assert logged and logged[-1] == rec
    assert rec["username"] == "m1110" and rec["tier"] == "internal"


def test_ship_posts_record_to_relay_with_internal_token():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["token"] = request.headers.get("X-Internal-Token")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    rec = {"evt": "access", "username": "m1110", "path": "/api/search"}
    ok = asyncio.run(access_log.ship(rec, base_url="http://retrieval-xray:18000",
                                     token="tok-123", client=client))
    asyncio.run(client.aclose())
    assert ok is True
    assert captured["url"] == "http://retrieval-xray:18000/internal/ingest-log"
    assert captured["token"] == "tok-123"
    assert captured["body"] == rec


def test_ship_is_best_effort_and_never_raises():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("relay down")

    client = httpx.AsyncClient(transport=httpx.MockTransport(boom))
    ok = asyncio.run(access_log.ship({"evt": "access"}, base_url="http://x:18000",
                                     token="t", client=client))
    asyncio.run(client.aclose())
    assert ok is False  # swallowed, no exception


def test_ship_noop_without_base_url():
    assert asyncio.run(access_log.ship({"evt": "access"}, base_url="", token="t")) is False
