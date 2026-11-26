"""End-to-end social-login + account-binding via the gateway's HTTP routes.

Drives /api/linuxdooid/* and /api/google/* with a FAKE oauth_fetch injected on
app.state (no network). Covers: first-login → /register, returning login →
session, binding a 2nd provider to a logged-in account (shared balance),
trust-level gate, unlink, and the last-identity guard.
"""
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.auth import accounts, oauth2
from gateway.settings import GatewaySettings

PEPPER = "test-pepper"
ROOT_KEY = "sk-root-admin-key"


class FakeProxy:
    async def request(self, method, path, *, headers=None, params=None, content=b""):
        return httpx.Response(200, json={"echo": path})

    async def stream(self, method, path, *, headers=None, params=None, content=b""):
        async def gen():
            yield b"data: x\n\n"
        return 200, {"content-type": "text/event-stream"}, gen()


@pytest.fixture
def app():
    settings = GatewaySettings(
        db_url="sqlite://", key_pepper=PEPPER, session_secret="sess-secret",
        admin_root_key=ROOT_KEY,
        public_base_url="https://retrieval.rnarket.com",
        linuxdo_client_id="ld-client", linuxdo_client_secret="ld-secret",
        linuxdo_min_trust_level=0,
        google_client_id="g-client", google_client_secret="g-secret")
    return create_app(settings, proxy=FakeProxy())


@pytest.fixture
def client(app):
    return TestClient(app)


def _set_fetch(app, identity: oauth2.OAuthIdentity):
    async def fake(provider, code, redirect_uri):
        return identity
    app.state.oauth_fetch = fake


def _start(client, kind):
    """Hit the login initiator, return the OAuth `state` it stashed in the session."""
    r = client.get(f"/api/{kind}/login", follow_redirects=False)
    assert r.status_code in (302, 307)
    return parse_qs(urlparse(r.headers["location"]).query)["state"][0]


def _ident(provider, sub, username="user", trust=5):
    return oauth2.OAuthIdentity(provider=provider, sub=sub, username=username,
                                trust_level=trust, raw={})


# --- providers gated by config ----------------------------------------------

def test_login_redirects_to_provider_authorize(client):
    r = client.get("/api/linuxdooid/login", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"].startswith("https://connect.linux.do/oauth2/authorize?")

    r = client.get("/api/google/login", follow_redirects=False)
    assert r.headers["location"].startswith("https://accounts.google.com/o/oauth2/v2/auth?")


def test_unknown_provider_404(client):
    assert client.get("/api/github/login", follow_redirects=False).status_code == 404


# --- first login → register, then returning login ---------------------------

def test_first_login_routes_to_register_then_persists(app, client):
    _set_fetch(app, _ident("linuxdo", "12345", "ld_user"))
    state = _start(client, "linuxdooid")
    r = client.get(f"/api/linuxdooid/callback?code=c&state={state}", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].endswith("/register")

    # claim username → real account
    reg = client.post("/auth/register", json={"username": "ld_user"})
    assert reg.status_code == 200
    me = client.get("/auth/me").json()
    assert me["username"] == "ld_user"
    assert any(i["provider"] == "linuxdo" for i in me["identities"])


def test_returning_login_sets_session_directly(app, client):
    _set_fetch(app, _ident("linuxdo", "777", "rep"))
    state = _start(client, "linuxdooid")
    client.get(f"/api/linuxdooid/callback?code=c&state={state}", follow_redirects=False)
    client.post("/auth/register", json={"username": "rep"})
    client.post("/auth/logout")

    # second time: identity already exists → straight to "/"
    state = _start(client, "linuxdooid")
    r = client.get(f"/api/linuxdooid/callback?code=c&state={state}", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].rstrip("/").endswith("rnarket.com")
    assert client.get("/auth/me").json()["username"] == "rep"


def test_callback_bad_state_rejected(app, client):
    _set_fetch(app, _ident("linuxdo", "1", "x"))
    _start(client, "linuxdooid")
    assert client.get("/api/linuxdooid/callback?code=c&state=WRONG",
                      follow_redirects=False).status_code == 400


# --- account binding: one account, two providers, shared balance ------------

def test_bind_second_provider_to_logged_in_account(app, client):
    # register via linux.do
    _set_fetch(app, _ident("linuxdo", "12345", "owner"))
    state = _start(client, "linuxdooid")
    client.get(f"/api/linuxdooid/callback?code=c&state={state}", follow_redirects=False)
    client.post("/auth/register", json={"username": "owner"})
    uid = client.get("/auth/me").json()["id"]

    # now bind Google while logged in (link mode)
    _set_fetch(app, _ident("google", "g-1", "owner-google"))
    r = client.get("/api/google/link", follow_redirects=False)
    state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
    cb = client.get(f"/api/google/callback?code=c&state={state}", follow_redirects=False)
    assert cb.status_code == 303 and "bind=ok" in cb.headers["location"]

    me = client.get("/auth/me").json()
    assert me["id"] == uid  # same account
    provs = {i["provider"] for i in me["identities"]}
    assert provs == {"linuxdo", "google"}


def test_bind_identity_owned_by_other_account_conflicts(app, client):
    # account A owns google g-shared
    _set_fetch(app, _ident("google", "g-shared", "a"))
    state = _start(client, "google")
    client.get(f"/api/google/callback?code=c&state={state}", follow_redirects=False)
    client.post("/auth/register", json={"username": "a"})
    client.post("/auth/logout")

    # account B (linux.do) logs in, then tries to bind the SAME google identity
    _set_fetch(app, _ident("linuxdo", "b-ld", "b"))
    state = _start(client, "linuxdooid")
    client.get(f"/api/linuxdooid/callback?code=c&state={state}", follow_redirects=False)
    client.post("/auth/register", json={"username": "b"})

    _set_fetch(app, _ident("google", "g-shared", "a"))
    r = client.get("/api/google/link", follow_redirects=False)
    state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
    cb = client.get(f"/api/google/callback?code=c&state={state}", follow_redirects=False)
    assert cb.status_code == 303 and "bind=conflict" in cb.headers["location"]
    # B still has only its linux.do identity
    provs = {i["provider"] for i in client.get("/auth/me").json()["identities"]}
    assert provs == {"linuxdo"}


def test_link_requires_login(client):
    assert client.get("/api/google/link", follow_redirects=False).status_code == 401


# --- trust-level gate (linux.do 最低等级) -----------------------------------

def test_low_trust_level_blocked(app):
    settings = GatewaySettings(
        db_url="sqlite://", key_pepper=PEPPER, session_secret="s",
        public_base_url="https://retrieval.rnarket.com",
        linuxdo_client_id="ld", linuxdo_client_secret="s", linuxdo_min_trust_level=2)
    app2 = create_app(settings, proxy=FakeProxy())
    c = TestClient(app2)
    _set_fetch(app2, _ident("linuxdo", "low", "noob", trust=0))
    r = c.get("/api/linuxdooid/login", follow_redirects=False)
    state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
    cb = c.get(f"/api/linuxdooid/callback?code=c&state={state}", follow_redirects=False)
    assert cb.status_code == 303 and "trust" in cb.headers["location"]
    assert c.get("/auth/me").status_code == 401  # not logged in


# --- unlink + last-identity guard -------------------------------------------

def test_unlink_identity_and_last_guard(app, client):
    _set_fetch(app, _ident("linuxdo", "12345", "owner"))
    state = _start(client, "linuxdooid")
    client.get(f"/api/linuxdooid/callback?code=c&state={state}", follow_redirects=False)
    client.post("/auth/register", json={"username": "owner"})

    _set_fetch(app, _ident("google", "g-1", "owner"))
    r = client.get("/api/google/link", follow_redirects=False)
    state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
    client.get(f"/api/google/callback?code=c&state={state}", follow_redirects=False)

    idents = client.get("/auth/identities").json()
    assert len(idents) == 2
    google_id = next(i["id"] for i in idents if i["provider"] == "google")
    linuxdo_id = next(i["id"] for i in idents if i["provider"] == "linuxdo")

    # unlink google → ok
    assert client.delete(f"/auth/identities/{google_id}").status_code == 200
    # unlink the remaining (last) identity → refused
    assert client.delete(f"/auth/identities/{linuxdo_id}").status_code == 400
    assert len(client.get("/auth/identities").json()) == 1
