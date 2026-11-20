"""Generic OAuth2 authorization-code client for social login.

Two providers ride on this: **linux.do Connect** (connect.linux.do — OAuth2 +
`/api/user`, carries a `trust_level` we gate on) and **Google** (OIDC userinfo).
Both are plain authorization-code: redirect to `authorize` → exchange `code` for
an `access_token` at `token` → read the profile from `userinfo`. We normalize the
profile to an `OAuthIdentity` (provider, sub, username, trust_level).

The cf-oidc CDK login is NOT here — it returns a signed id_token verified in
`oidc.py`. This module is for opaque-token providers where we call userinfo.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from urllib.parse import urlencode

import httpx

logger = logging.getLogger("gateway.auth.oauth2")

_TIMEOUT = 15.0


class OAuthError(Exception):
    """A social-login exchange failed (token endpoint / userinfo / missing sub)."""


@dataclass
class OAuth2Provider:
    name: str                 # identity provider key: "linuxdo" | "google"
    client_id: str
    client_secret: str
    authorize_endpoint: str
    token_endpoint: str
    userinfo_endpoint: str
    redirect_uri: str
    scope: str = ""
    sub_field: str = "sub"                       # userinfo field holding the stable id
    username_fields: tuple = ("username", "name", "email")  # first present wins
    trust_field: str | None = None              # linux.do: "trust_level"
    min_trust_level: int = 0                    # reject logins below this (0 = no gate)


@dataclass
class OAuthIdentity:
    provider: str
    sub: str
    username: str | None = None
    trust_level: int | None = None
    raw: dict = field(default_factory=dict)


def build_authorize_url(provider: OAuth2Provider, *, redirect_uri: str, state: str) -> str:
    """The provider authorize URL to redirect the browser to (PKCE not used —
    these are confidential clients with a server-side secret)."""
    params = {
        "client_id": provider.client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "state": state,
    }
    if provider.scope:
        params["scope"] = provider.scope
    return f"{provider.authorize_endpoint}?{urlencode(params)}"


async def fetch_identity(provider: OAuth2Provider, *, code: str,
                         redirect_uri: str) -> OAuthIdentity:
    """Exchange `code` for an access token, then read + normalize the profile.
    Raises OAuthError on any non-200 or a missing subject."""
    token_data = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": provider.client_id,
        "client_secret": provider.client_secret,
        "redirect_uri": redirect_uri,
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        tr = await client.post(provider.token_endpoint, data=token_data,
                               headers={"Accept": "application/json"})
    if tr.status_code != 200:
        logger.warning("oauth token exchange failed provider=%s status=%s",
                       provider.name, tr.status_code)
        raise OAuthError(f"token exchange failed ({tr.status_code})")
    access_token = (tr.json() or {}).get("access_token")
    if not access_token:
        raise OAuthError("no access_token in token response")

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        ur = await client.get(provider.userinfo_endpoint,
                              headers={"Authorization": f"Bearer {access_token}",
                                       "Accept": "application/json"})
    if ur.status_code != 200:
        logger.warning("oauth userinfo failed provider=%s status=%s",
                       provider.name, ur.status_code)
        raise OAuthError(f"userinfo failed ({ur.status_code})")
    info = ur.json() or {}

    sub = info.get(provider.sub_field)
    if sub in (None, ""):
        raise OAuthError(f"no '{provider.sub_field}' in userinfo")
    username = next((str(info[f]) for f in provider.username_fields
                    if info.get(f)), None)
    trust = info.get(provider.trust_field) if provider.trust_field else None
    logger.info("oauth identity provider=%s sub=%s trust=%s", provider.name, sub, trust)
    return OAuthIdentity(provider=provider.name, sub=str(sub), username=username,
                         trust_level=trust, raw=info)
