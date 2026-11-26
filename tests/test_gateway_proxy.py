"""Tests for the internal reverse proxy — header sanitization, token injection,
5xx → ComputeError, 4xx pass-through, streaming. Uses httpx.MockTransport; async
coroutines are driven with asyncio.run (no pytest-asyncio dependency)."""
import asyncio

import httpx
import pytest

from gateway.proxy import (
    INTERNAL_TOKEN_HEADER, ComputeError, InternalProxy,
    attach_user_headers, build_forward_headers,
)

TOKEN = "internal-secret"


class _FakeUser:
    def __init__(self, uid, username, tier):
        self.id = uid
        self.username = username
        self.tier = type("T", (), {"value": tier})()


def test_attach_user_headers_strips_spoofed_client_identity():
    # W147: a client smuggles identity headers (any case). attach must drop every
    # case-variant before injecting the authentic values, so the compute tier's
    # case-insensitive lookup can only ever see OUR value (no second header line).
    incoming = {
        "content-type": "application/json",
        "x-retrival-user-id": "999",        # spoofed (lowercase, as dict(request.headers) yields)
        "X-Retrival-Username": "victim",     # spoofed (mixed case)
        "X-RETRIVAL-USER-TIER": "admin",     # spoofed (upper)
    }
    out = attach_user_headers(incoming, _FakeUser(42, "real", "pro"))
    # exactly one key per identity header, holding the authentic value
    assert {k: v for k, v in out.items() if k.lower() == "x-retrival-user-id"} == {"X-Retrival-User-Id": "42"}
    assert {k: v for k, v in out.items() if k.lower() == "x-retrival-username"} == {"X-Retrival-Username": "real"}
    assert {k: v for k, v in out.items() if k.lower() == "x-retrival-user-tier"} == {"X-Retrival-User-Tier": "pro"}
    assert out["content-type"] == "application/json"  # non-identity headers untouched


def test_build_forward_headers_sanitizes_and_injects():
    incoming = {
        "Host": "gateway.example",
        "Content-Length": "10",
        "Authorization": "Bearer sk-leak",
        "Cookie": "session=leak",
        "Connection": "keep-alive",
        "Content-Type": "application/json",
        "X-Request-Id": "abc",
        "X-Retrival-User-Id": "7",
    }
    out = build_forward_headers(incoming, TOKEN)
    assert out[INTERNAL_TOKEN_HEADER] == TOKEN
    assert out["Content-Type"] == "application/json"
    assert out["X-Request-Id"] == "abc"
    assert out["X-Retrival-User-Id"] == "7"
    for dropped in ("Host", "Content-Length", "Authorization", "Cookie", "Connection"):
        assert dropped not in out          # identity + hop-by-hop stripped


def _proxy(handler):
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    return InternalProxy("http://compute:8000", TOKEN, client=client)


def test_request_injects_token_and_returns_body():
    seen = {}

    def handler(request: httpx.Request):
        seen["token"] = request.headers.get(INTERNAL_TOKEN_HEADER)
        seen["auth"] = request.headers.get("Authorization")
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"ok": True})

    proxy = _proxy(handler)
    resp = asyncio.run(proxy.request(
        "POST", "/api/search",
        headers={"Authorization": "Bearer sk-user", "Content-Type": "application/json"},
        content=b'{"q":"x"}'))
    assert resp.status_code == 200 and resp.json() == {"ok": True}
    assert seen["token"] == TOKEN
    assert seen["auth"] is None                 # user sk-key never reaches compute
    assert seen["url"] == "http://compute:8000/api/search"


def test_request_raises_on_5xx():
    proxy = _proxy(lambda req: httpx.Response(502, text="upstream boom"))
    with pytest.raises(ComputeError) as ei:
        asyncio.run(proxy.request("GET", "/api/search"))
    assert ei.value.status_code == 502


def test_request_passes_through_4xx():
    proxy = _proxy(lambda req: httpx.Response(422, json={"detail": "bad"}))
    resp = asyncio.run(proxy.request("POST", "/api/search"))
    assert resp.status_code == 422            # client error returned, not raised


def test_stream_yields_chunks():
    def handler(request: httpx.Request):
        assert request.headers.get(INTERNAL_TOKEN_HEADER) == TOKEN

        async def body():            # async-gen body => genuinely streamable mock
            yield b"data: 1\n\n"
            yield b"data: 2\n\n"

        return httpx.Response(200, content=body(),
                              headers={"Content-Type": "text/event-stream"})

    proxy = _proxy(handler)

    async def run():
        status, headers, aiter = await proxy.stream("GET", "/api/search/stream")
        body = b"".join([chunk async for chunk in aiter])
        return status, headers, body

    status, headers, body = asyncio.run(run())
    assert status == 200
    assert headers.get("content-type") == "text/event-stream"
    assert body == b"data: 1\n\ndata: 2\n\n"


def test_stream_raises_on_5xx_init():
    proxy = _proxy(lambda req: httpx.Response(500, text="down"))
    with pytest.raises(ComputeError):
        asyncio.run(proxy.stream("GET", "/api/search/stream"))
