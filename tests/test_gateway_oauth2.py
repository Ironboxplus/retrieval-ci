"""Generic OAuth2 authorization-code client used for social login
(linux.do Connect + Google). Locks the authorize-URL build and the
exchange→userinfo→normalized-identity path. Network is monkeypatched and
coroutines are driven with asyncio.run (no pytest-asyncio dependency).
"""
import asyncio
from urllib.parse import parse_qs, urlparse

import pytest

from gateway.auth import oauth2

LINUXDO = oauth2.OAuth2Provider(
    name="linuxdo",
    client_id="ld-client",
    client_secret="ld-secret",
    authorize_endpoint="https://connect.linux.do/oauth2/authorize",
    token_endpoint="https://connect.linux.do/oauth2/token",
    userinfo_endpoint="https://connect.linux.do/api/user",
    redirect_uri="https://retrieval.rnarket.com/api/linuxdooid/callback",
    scope="",
    sub_field="id",
    username_fields=("username", "name"),
    trust_field="trust_level",
    min_trust_level=1,
)

GOOGLE = oauth2.OAuth2Provider(
    name="google",
    client_id="g-client",
    client_secret="g-secret",
    authorize_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
    token_endpoint="https://oauth2.googleapis.com/token",
    userinfo_endpoint="https://openidconnect.googleapis.com/v1/userinfo",
    redirect_uri="https://retrieval.rnarket.com/api/google/callback",
    scope="openid email profile",
    sub_field="sub",
    username_fields=("name", "email"),
)


def test_build_authorize_url_linuxdo():
    url = oauth2.build_authorize_url(LINUXDO, redirect_uri=LINUXDO.redirect_uri, state="S1")
    assert url.startswith("https://connect.linux.do/oauth2/authorize?")
    q = parse_qs(urlparse(url).query)
    assert q["client_id"][0] == "ld-client"
    assert q["redirect_uri"][0] == LINUXDO.redirect_uri
    assert q["response_type"][0] == "code"
    assert q["state"][0] == "S1"
    assert "scope" not in q  # empty scope omitted


def test_build_authorize_url_google_has_scope():
    url = oauth2.build_authorize_url(GOOGLE, redirect_uri=GOOGLE.redirect_uri, state="S2")
    q = parse_qs(urlparse(url).query)
    assert q["scope"][0] == "openid email profile"


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._p = payload

    def json(self):
        return self._p


def test_fetch_identity_linuxdo(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, data=None, headers=None):
            captured["token_url"] = url
            captured["token_data"] = data
            return _Resp(200, {"access_token": "AT-1", "token_type": "Bearer"})

        async def get(self, url, headers=None):
            captured["userinfo_url"] = url
            captured["userinfo_headers"] = headers
            return _Resp(200, {"id": 12345, "username": "ld_user", "name": "LD",
                               "trust_level": 3})

    monkeypatch.setattr(oauth2.httpx, "AsyncClient", FakeClient)
    ident = asyncio.run(oauth2.fetch_identity(LINUXDO, code="abc", redirect_uri=LINUXDO.redirect_uri))
    assert ident.provider == "linuxdo"
    assert ident.sub == "12345"            # coerced to str
    assert ident.username == "ld_user"
    assert ident.trust_level == 3
    # exchange sent the right client creds + code + redirect
    assert captured["token_url"] == LINUXDO.token_endpoint
    assert captured["token_data"]["code"] == "abc"
    assert captured["token_data"]["client_id"] == "ld-client"
    assert captured["token_data"]["client_secret"] == "ld-secret"
    assert captured["token_data"]["redirect_uri"] == LINUXDO.redirect_uri
    assert captured["userinfo_headers"]["Authorization"] == "Bearer AT-1"


def test_fetch_identity_google_username_fallback(monkeypatch):
    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, data=None, headers=None):
            return _Resp(200, {"access_token": "AT"})

        async def get(self, url, headers=None):
            # no "name" → falls back to email
            return _Resp(200, {"sub": "108", "email": "u@example.com"})

    monkeypatch.setattr(oauth2.httpx, "AsyncClient", FakeClient)
    ident = asyncio.run(oauth2.fetch_identity(GOOGLE, code="c", redirect_uri=GOOGLE.redirect_uri))
    assert ident.sub == "108"
    assert ident.username == "u@example.com"
    assert ident.trust_level is None  # google has no trust_field


def test_fetch_identity_token_failure_raises(monkeypatch):
    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, data=None, headers=None):
            return _Resp(401, {"error": "invalid_grant"})

        async def get(self, url, headers=None):
            return _Resp(200, {})

    monkeypatch.setattr(oauth2.httpx, "AsyncClient", FakeClient)
    with pytest.raises(oauth2.OAuthError):
        asyncio.run(oauth2.fetch_identity(GOOGLE, code="c", redirect_uri=GOOGLE.redirect_uri))
