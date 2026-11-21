"""Reverse proxy from the web backend (BWH) to the internal compute box (4090).

The gateway authenticates + meters the user, then forwards the bare compute
request inward over httpx, attaching the shared `X-Internal-Token`. The 4090 sees
NO user identity, sk-key, cookie, or billing state — only the compute payload and
the internal token (see the env-gated gate in the compute app's main.py).

`ComputeError` is raised on a 5xx from the 4090 so the metering wrapper refunds the
credit (a server fault is not the user's fault). 4xx pass through unchanged.
"""
from __future__ import annotations

import logging

import httpx

logger = logging.getLogger("gateway.proxy")

INTERNAL_TOKEN_HEADER = "X-Internal-Token"
USER_ID_HEADER = "X-Retrival-User-Id"
USERNAME_HEADER = "X-Retrival-Username"
USER_TIER_HEADER = "X-Retrival-User-Tier"

# Hop-by-hop headers (RFC 7230 §6.1) must not be forwarded by a proxy.
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}
# Plus: never leak the caller's identity/host into the compute tier.
_DROP = _HOP_BY_HOP | {"host", "content-length", "authorization", "cookie"}


class ComputeError(Exception):
    """The internal compute tier returned a server error (5xx)."""

    def __init__(self, status_code: int, detail: str = ""):
        super().__init__(f"compute returned {status_code}: {detail}".rstrip(": "))
        self.status_code = status_code
        self.detail = detail


def build_forward_headers(incoming, internal_token: str) -> dict:
    """Outbound headers for the inward request: drop hop-by-hop + identity headers,
    inject the internal token. `incoming` is anything with `.items()`."""
    out: dict[str, str] = {}
    for k, v in incoming.items():
        if k.lower() in _DROP:
            continue
        out[k] = v
    out[INTERNAL_TOKEN_HEADER] = internal_token
    return out


_IDENTITY_HEADERS = {USER_ID_HEADER.lower(), USERNAME_HEADER.lower(), USER_TIER_HEADER.lower()}


def attach_user_headers(headers: dict, user) -> dict:
    """Attach the authenticated gateway identity for private compute APIs.

    Strips ANY client-supplied identity header (case-insensitively) BEFORE injecting
    our authentic values. Without this, a caller could smuggle a spoofed identity
    header — e.g. lowercase ``x-retrival-user-id`` — which ``dict(request.headers)``
    preserves; our mixed-case ``X-Retrival-User-Id`` would then ride alongside it as
    a second header line, and the compute tier's case-insensitive lookup reads the
    FIRST (attacker) value → cross-user read/write. Dropping the client's copy first
    guarantees only our value survives (W147)."""
    out = {k: v for k, v in headers.items() if k.lower() not in _IDENTITY_HEADERS}
    out[USER_ID_HEADER] = str(user.id)
    out[USERNAME_HEADER] = user.username
    out[USER_TIER_HEADER] = user.tier.value
    return out


class InternalProxy:
    def __init__(self, base_url: str, token: str, *,
                 client: httpx.AsyncClient | None = None, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def request(self, method: str, path: str, *, headers=None,
                      params=None, content: bytes = b"") -> httpx.Response:
        """Forward a non-streaming request. Raises ComputeError on 5xx; returns the
        httpx.Response (incl. 4xx) otherwise."""
        url = self.base_url + path
        out_headers = build_forward_headers(headers or {}, self.token)
        resp = await self._client.request(method, url, headers=out_headers,
                                          params=params, content=content)
        if resp.status_code >= 500:
            raise ComputeError(resp.status_code, resp.text[:200])
        return resp

    async def stream(self, method: str, path: str, *, headers=None,
                     params=None, content: bytes = b""):
        """Open a streaming (SSE) forward. Returns (status_code, headers, aiter).
        Raises ComputeError if the connection opens with a 5xx. Caller is
        responsible for consuming the iterator (it closes the upstream on exit)."""
        url = self.base_url + path
        out_headers = build_forward_headers(headers or {}, self.token)
        req = self._client.build_request(method, url, headers=out_headers,
                                         params=params, content=content)
        resp = await self._client.send(req, stream=True)
        if resp.status_code >= 500:
            body = await resp.aread()
            await resp.aclose()
            raise ComputeError(resp.status_code, body[:200].decode("utf-8", "replace"))

        async def _iter():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()

        return resp.status_code, dict(resp.headers), _iter()

    async def aclose(self):
        await self._client.aclose()
