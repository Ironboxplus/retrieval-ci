"""System / integration test for the gateway as a DEPLOYED stack.

A real OIDC-provider-plus-compute upstream is served on a live TCP port (uvicorn
in a thread). The gateway runs in-process via Starlette TestClient (the full real
ASGI stack: SessionMiddleware, cookies, DB, dependencies). The test drives the
genuine flows over real HTTP:

  * OIDC: /auth/login → (user approves at the real upstream /authorize) →
    /auth/callback (gateway does a REAL httpx token exchange + REAL JWKS verify) →
    /auth/register → session cookie → /auth/me
  * sk-key issuance, then a metered /api/search REAL-proxied to the upstream
  * refund when the upstream compute returns 5xx
  * admin funding via the fixed root key

This exercises default_oidc_exchange, PyJWKClient JWKS fetch, InternalProxy, the
session middleware, and the ledger together — i.e. the deployable artifact.
"""
import base64
import hashlib
import json
import socket
import threading
import time
from urllib.parse import urlencode, urlparse

import httpx
import jwt
import pytest
import uvicorn
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from jwt.algorithms import RSAAlgorithm
from starlette.testclient import TestClient

from gateway.app import create_app
from gateway.settings import GatewaySettings

ROOT_KEY = "sk-root-system-key"
PEPPER = "sys-pepper"
CLIENT_ID = "retrival-web"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_keypair():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv_pem = priv.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption())
    pub_jwk = json.loads(RSAAlgorithm.to_jwk(priv.public_key()))
    pub_jwk.update({"kid": "k1", "use": "sig", "alg": "RS256"})
    return priv_pem, pub_jwk


def _make_upstream(issuer: str, priv_pem: bytes, pub_jwk: dict) -> FastAPI:
    """Fake CF Worker (OIDC) + fake 4090 (compute) in one test double."""
    app = FastAPI()
    codes: dict[str, dict] = {}

    @app.get("/authorize")
    def authorize(client_id: str, redirect_uri: str, state: str = "", nonce: str = "",
                  code_challenge: str = "", code_challenge_method: str = "S256",
                  cdk: str = "alice", is_admin: int = 0):
        # Simulate an approved CDK: mint a code bound to the PKCE challenge + nonce.
        import secrets as _s
        code = _s.token_hex(8)
        codes[code] = {"nonce": nonce, "challenge": code_challenge,
                       "method": code_challenge_method, "cdk": cdk,
                       "is_admin": bool(is_admin), "client_id": client_id}
        return RedirectResponse(redirect_uri + "?" + urlencode({"code": code, "state": state}),
                                status_code=302)

    @app.post("/token")
    async def token(request: Request):
        form = await request.form()
        rec = codes.pop(form.get("code"), None)
        if rec is None:
            return JSONResponse({"error": "invalid_grant"}, status_code=400)
        verifier = form.get("code_verifier", "")
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        if rec["method"] == "S256" and challenge != rec["challenge"]:
            return JSONResponse({"error": "invalid_grant", "error_description": "pkce"},
                                status_code=400)
        now = int(time.time())
        payload = {"iss": issuer, "sub": f"cdk:{rec['cdk']}", "aud": rec["client_id"],
                   "iat": now, "nbf": now, "exp": now + 3600, "jti": "j1",
                   "is_admin": rec["is_admin"],
                   "roles": ["admin", "user"] if rec["is_admin"] else ["user"],
                   "groups": ["admin", "user"] if rec["is_admin"] else ["user"]}
        if rec["nonce"]:
            payload["nonce"] = rec["nonce"]
        id_token = jwt.encode(payload, priv_pem, algorithm="RS256", headers={"kid": "k1"})
        return {"id_token": id_token, "access_token": "a", "token_type": "Bearer",
                "expires_in": 3600}

    @app.get("/keys")
    def keys():
        return {"keys": [pub_jwk]}

    @app.post("/api/search")
    async def search(request: Request):
        raw = await request.body()
        if b'"boom"' in raw:
            return JSONResponse({"detail": "compute exploded"}, status_code=500)
        return {"papers": [{"paper_id": "p1", "doc_title": "Sys Paper",
                            "max_score": 0.91, "chunks": [{"content": "hit"}]}]}

    @app.get("/api/conferences")
    def conferences():
        return {"conferences": [{"venue": "NeurIPS"}]}

    return app


class _Server:
    def __init__(self, app, port):
        self.port = port
        cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                             lifespan="off")
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self):
        self._thread.start()
        deadline = time.time() + 15
        while not self._server.started and time.time() < deadline:
            time.sleep(0.05)
        if not self._server.started:
            raise RuntimeError("upstream server did not start")

    def stop(self):
        self._server.should_exit = True
        self._thread.join(timeout=10)


@pytest.fixture(scope="module")
def upstream():
    port = _free_port()
    issuer = f"http://127.0.0.1:{port}"
    priv_pem, pub_jwk = _make_keypair()
    server = _Server(_make_upstream(issuer, priv_pem, pub_jwk), port)
    server.start()
    try:
        yield issuer
    finally:
        server.stop()


@pytest.fixture
def gateway(upstream):
    settings = GatewaySettings(
        db_url="sqlite://", key_pepper=PEPPER, session_secret="sys-sess",
        admin_root_key=ROOT_KEY, payment_secret="mock-secret", search_cost=1,
        oidc_issuer=upstream, oidc_client_id=CLIENT_ID,
        oidc_jwks_uri=f"{upstream}/keys",
        oidc_redirect_uri="http://testserver/auth/callback",
        internal_base_url=upstream,
        payment_notify_url="http://testserver/api/payment/notify",
        public_base_url="http://testserver")
    return create_app(settings)


def _do_login(gc: TestClient, *, cdk="alice", is_admin=0):
    """Run the real OIDC handshake; return the gateway callback response (303)."""
    r = gc.get("/auth/login", follow_redirects=False)
    assert r.status_code in (302, 307)
    authorize_url = r.headers["location"]
    # user approves at the real upstream /authorize (separate real HTTP client)
    sep = "&" if "?" in authorize_url else "?"
    approve_url = f"{authorize_url}{sep}{urlencode({'cdk': cdk, 'is_admin': is_admin})}"
    wr = httpx.get(approve_url, follow_redirects=False)
    assert wr.status_code == 302, wr.text
    cb = urlparse(wr.headers["location"])
    return gc.get(f"/auth/callback?{cb.query}", follow_redirects=False)


# ---------------------------------------------------------------------------

def test_full_oidc_register_then_session(gateway):
    gc = TestClient(gateway)
    cb = _do_login(gc, cdk="alice")
    assert cb.status_code == 303 and cb.headers["location"].endswith("/register")

    reg = gc.post("/auth/register", json={"username": "alice"})
    assert reg.status_code == 200 and reg.json()["username"] == "alice"
    assert reg.json()["tier"] == "paying"

    me = gc.get("/auth/me")
    assert me.status_code == 200 and me.json()["username"] == "alice"


def test_returning_user_skips_register(gateway):
    gc = TestClient(gateway)
    cb = _do_login(gc, cdk="bob")
    gc.post("/auth/register", json={"username": "bob"})
    # second login for the same subject → straight to "/" (no register)
    gc2 = TestClient(gateway)
    cb2 = _do_login(gc2, cdk="bob")
    assert cb2.status_code == 303 and cb2.headers["location"].endswith("/")
    assert gc2.get("/auth/me").json()["username"] == "bob"


def test_admin_via_oidc_is_admin_claim(gateway):
    gc = TestClient(gateway)
    _do_login(gc, cdk="boss", is_admin=1)
    gc.post("/auth/register", json={"username": "boss"})
    assert gc.get("/auth/me").json()["tier"] == "admin"
    # admin-claimed user can hit an admin route via its session
    assert gc.get("/admin/users").status_code == 200


def test_metered_search_decrements_over_real_proxy(gateway):
    gc = TestClient(gateway)
    _do_login(gc, cdk="carol")
    gc.post("/auth/register", json={"username": "carol"})
    uid = gc.get("/auth/me").json()["id"]

    # admin (fixed root key) funds the user
    fund = gc.post(f"/admin/users/{uid}/adjust",
                   headers={"Authorization": f"Bearer {ROOT_KEY}"}, json={"amount": 3})
    assert fund.status_code == 200 and fund.json()["balance"] == 3

    # issue an sk-key (session-authed), then search with Bearer → real-proxied to upstream
    key = gc.post("/auth/keys").json()["key"]
    r = gc.post("/api/search", headers={"Authorization": f"Bearer {key}"}, json={"query": "x"})
    assert r.status_code == 200 and r.json()["papers"][0]["paper_id"] == "p1"
    assert gc.get("/auth/me").json()["balance"] == 2


def test_refund_when_compute_500(gateway):
    gc = TestClient(gateway)
    _do_login(gc, cdk="dave")
    gc.post("/auth/register", json={"username": "dave"})
    uid = gc.get("/auth/me").json()["id"]
    gc.post(f"/admin/users/{uid}/adjust",
            headers={"Authorization": f"Bearer {ROOT_KEY}"}, json={"amount": 5})
    key = gc.post("/auth/keys").json()["key"]

    r = gc.post("/api/search", headers={"Authorization": f"Bearer {key}"}, json={"boom": True})
    assert r.status_code == 502                     # compute 5xx surfaced
    assert gc.get("/auth/me").json()["balance"] == 5  # refunded, made whole


def test_conferences_requires_login(gateway):
    # W127b: aggregates (conferences/stats/distribution) are now LOGIN-GATED — the
    # operator chose to gate ALL data (reverses the earlier anonymous-aggregate rule).
    gc = TestClient(gateway)
    assert gc.get("/api/conferences").status_code == 401
    assert gc.get("/api/stats").status_code == 401
    assert gc.get("/api/dashboard/distribution").status_code == 401
