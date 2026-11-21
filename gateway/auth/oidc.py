"""OIDC id_token verification against the external CF Worker provider.

The Worker issues RS256 id_tokens (Authorization Code + PKCE, exp=1h, `sub=cdk:<key>`,
custom claims `is_admin`/`roles`/`groups`), publishing its public keys as a JWKS. We
verify signature + `iss` + `aud` + `exp` (+ optional `nonce`) and hand back normalized
`Claims`. `decode_id_token` is the pure core (key already resolved); `verify_id_token`
adds JWKS resolution via a cached `PyJWKClient`. We never call /userinfo and there is
no refresh token — the durable credential is the sk-key, not this token.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import jwt
from jwt import PyJWKClient

logger = logging.getLogger("gateway.auth.oidc")

_ALGORITHMS = ["RS256"]
_REQUIRE = ["exp", "iat", "sub"]

# Cloudflare's managed bot rules 403 the default urllib User-Agent
# ("Python-urllib/x") at the Worker's edge — and PyJWKClient fetches the JWKS with
# urllib. With the default UA the /keys fetch 403s, so EVERY id_token verification
# fails ("could not resolve signing key" → 401). Any non-urllib UA is allowed
# (httpx/curl/browser all return 200), so we set an explicit one. (W120)
_JWKS_HEADERS = {"User-Agent": "retrival-gateway/1.0"}


def make_jwks_client(jwks_uri: str) -> PyJWKClient:
    """PyJWKClient with a non-urllib User-Agent so Cloudflare doesn't 403 the JWKS fetch."""
    return PyJWKClient(jwks_uri, headers=_JWKS_HEADERS)


class OIDCError(Exception):
    """id_token failed verification (bad sig / iss / aud / exp / nonce)."""


@dataclass
class OIDCConfig:
    issuer: str          # token `iss`, e.g. https://116.ironbox.workers.dev
    client_id: str       # token `aud`
    jwks_uri: str        # where the Worker publishes its public keys
    leeway: int = 30     # clock-skew tolerance (seconds)


@dataclass
class Claims:
    sub: str
    is_admin: bool = False
    roles: list = field(default_factory=list)
    groups: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)


def decode_id_token(token: str, key, *, issuer: str, audience: str,
                    nonce: str | None = None, leeway: int = 30) -> Claims:
    """Verify a token against an already-resolved public `key`. Raises OIDCError on
    any failure. `nonce`, when given, must equal the token's `nonce` claim."""
    try:
        decoded = jwt.decode(
            token, key, algorithms=_ALGORITHMS, audience=audience, issuer=issuer,
            leeway=leeway, options={"require": _REQUIRE},
        )
    except jwt.PyJWTError as exc:
        raise OIDCError(f"id_token rejected: {exc}") from exc
    if nonce is not None and decoded.get("nonce") != nonce:
        raise OIDCError("nonce mismatch")
    return Claims(
        sub=decoded["sub"],
        is_admin=bool(decoded.get("is_admin", False)),
        roles=list(decoded.get("roles", []) or []),
        groups=list(decoded.get("groups", []) or []),
        raw=decoded,
    )


def verify_id_token(token: str, config: OIDCConfig, *,
                    jwks_client: PyJWKClient | None = None,
                    nonce: str | None = None) -> Claims:
    """Resolve the signing key from the Worker's JWKS (cached), then verify.
    Pass a shared `jwks_client` to reuse its key cache across requests."""
    client = jwks_client or make_jwks_client(config.jwks_uri)
    try:
        key = client.get_signing_key_from_jwt(token).key
    except jwt.PyJWTError as exc:
        raise OIDCError(f"could not resolve signing key: {exc}") from exc
    return decode_id_token(token, key, issuer=config.issuer,
                           audience=config.client_id, nonce=nonce,
                           leeway=config.leeway)
