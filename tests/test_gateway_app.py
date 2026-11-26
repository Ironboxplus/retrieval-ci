"""End-to-end gateway tests via TestClient — auth (Bearer sk-), metered search,
402 on empty balance, tier exemption, admin adjust/guard, key mgmt, topup+notify.
A fake proxy stands in for the 4090 so nothing hits the network."""
import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.auth import accounts
from gateway.db.models import Tier
from gateway.payment import sign
from gateway.settings import GatewaySettings

PEPPER = "test-pepper"
PAY_SECRET = "mock-secret"
ROOT_KEY = "sk-root-admin-key"


class FakeProxy:
    def __init__(self):
        self.calls = []

    async def request(self, method, path, *, headers=None, params=None, content=b""):
        self.calls.append((method, path))
        return httpx.Response(200, json={"echo": path})

    async def stream(self, method, path, *, headers=None, params=None, content=b""):
        async def gen():
            yield b"data: x\n\n"
        return 200, {"content-type": "text/event-stream"}, gen()


@pytest.fixture
def app():
    settings = GatewaySettings(
        db_url="sqlite://", key_pepper=PEPPER, session_secret="sess-secret",
        admin_root_key=ROOT_KEY, payment_secret=PAY_SECRET, search_cost=1,
        payment_notify_url="http://wb/api/payment/notify")
    return create_app(settings, proxy=FakeProxy())


@pytest.fixture
def client(app):
    return TestClient(app)


def _mk_user(app, sub, username, tier=Tier.paying, balance=0):
    db = app.state.session_factory()
    try:
        u = accounts.register(db, oidc_sub=sub, username=username)
        u.tier = tier
        u.balance = balance
        key, _ = accounts.issue_key(db, u, PEPPER)
        db.commit()
        return u.id, key
    finally:
        db.close()


def _auth(key):
    return {"Authorization": f"Bearer {key}"}


# --- auth --------------------------------------------------------------------

def test_me_requires_auth(client):
    assert client.get("/auth/me").status_code == 401


def test_bearer_auth_me(app, client):
    uid, key = _mk_user(app, "cdk:a", "alice", balance=3)
    r = client.get("/auth/me", headers=_auth(key))
    assert r.status_code == 200
    assert r.json()["username"] == "alice" and r.json()["balance"] == 3


# --- metered search ----------------------------------------------------------

def test_search_charges_one_credit(app, client):
    uid, key = _mk_user(app, "cdk:pay", "payer", tier=Tier.paying, balance=5)
    r = client.post("/api/search", headers=_auth(key), json={"query": "x"})
    assert r.status_code == 200 and r.json()["echo"] == "/api/search"
    assert client.get("/auth/me", headers=_auth(key)).json()["balance"] == 4


def test_search_insufficient_returns_402(app, client):
    uid, key = _mk_user(app, "cdk:broke", "broke", tier=Tier.paying, balance=0)
    assert client.post("/api/search", headers=_auth(key), json={"query": "x"}).status_code == 402


def test_internal_tier_not_charged(app, client):
    uid, key = _mk_user(app, "cdk:int", "intern", tier=Tier.internal, balance=2)
    assert client.post("/api/search", headers=_auth(key), json={"query": "x"}).status_code == 200
    assert client.get("/auth/me", headers=_auth(key)).json()["balance"] == 2


# --- admin -------------------------------------------------------------------

def test_admin_adjust_with_root_key(app, client):
    uid, _ = _mk_user(app, "cdk:t", "target", balance=0)
    r = client.post(f"/admin/users/{uid}/adjust", headers=_auth(ROOT_KEY),
                    json={"amount": 10, "reason": "grant"})
    assert r.status_code == 200 and r.json()["balance"] == 10


def test_require_admin_forbids_normal_user(app, client):
    uid, key = _mk_user(app, "cdk:n", "normal", balance=1)
    assert client.get("/admin/users", headers=_auth(key)).status_code == 403


def test_admin_set_tier(app, client):
    uid, _ = _mk_user(app, "cdk:p", "promote", tier=Tier.paying)
    r = client.post(f"/admin/users/{uid}/tier", headers=_auth(ROOT_KEY), json={"tier": "internal"})
    assert r.status_code == 200 and r.json()["tier"] == "internal"


# --- key management ----------------------------------------------------------

def test_create_and_list_keys(app, client):
    uid, key = _mk_user(app, "cdk:k", "keyer", balance=0)
    created = client.post("/auth/keys", headers=_auth(key))
    assert created.status_code == 200 and created.json()["key"].startswith("sk-")
    listed = client.get("/auth/keys", headers=_auth(key)).json()
    assert len(listed) == 2  # the seed key + the new one


# --- payment (mock) ----------------------------------------------------------

def test_topup_then_notify_credits_balance(app, client):
    uid, key = _mk_user(app, "cdk:buy", "buyer", balance=0)
    started = client.post("/api/topup", headers=_auth(key), json={"credits": 100, "price": "5.00"})
    assert started.status_code == 200
    otn = started.json()["out_trade_no"]

    params = {"out_trade_no": otn, "money": "5.00", "trade_status": "TRADE_SUCCESS"}
    params["sign"] = sign.md5_sign(params, PAY_SECRET)
    notified = client.post("/api/payment/notify", data=params)
    assert notified.status_code == 200 and notified.text == "success"

    assert client.get("/auth/me", headers=_auth(key)).json()["balance"] == 100


def test_notify_forged_signature_rejected(app, client):
    uid, key = _mk_user(app, "cdk:fraud", "fraud", balance=0)
    otn = client.post("/api/topup", headers=_auth(key),
                      json={"credits": 100, "price": "5.00"}).json()["out_trade_no"]
    bad = {"out_trade_no": otn, "money": "5.00", "trade_status": "TRADE_SUCCESS", "sign": "forged"}
    assert client.post("/api/payment/notify", data=bad).text == "fail"
    assert client.get("/auth/me", headers=_auth(key)).json()["balance"] == 0
