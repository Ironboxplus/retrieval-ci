"""Gateway (app-tier) settings — its OWN config, separate from the 4090 `config.py`.

Runs on the public web backend (BWH). Holds the references to all sensitive
material (pepper, session secret, admin root key, payment secret, internal token);
in production these MUST come from the environment (prefix `GATEWAY_`), never the
insecure dev defaults below. `get_settings()` is cached so the app shares one copy.
"""
from __future__ import annotations

from decimal import Decimal
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GATEWAY_", env_file=".env.gateway", extra="ignore")

    # app-tier DB (users / keys / ledger / orders)
    db_url: str = "sqlite:///./gateway.db"

    # secrets — OVERRIDE IN PROD
    key_pepper: str = "dev-insecure-pepper"        # HMAC pepper for sk-key hashing
    session_secret: str = "dev-insecure-session"   # signed-cookie session key
    admin_root_key: str = ""                       # fixed admin sk-key; "" => no bootstrap

    # external OIDC (CF Worker) — the CDK / access-key login
    oidc_issuer: str = "https://116.ironbox.workers.dev"
    oidc_client_id: str = "retrival-web"
    oidc_jwks_uri: str = ""                         # default: issuer + /keys
    oidc_redirect_uri: str = "http://localhost:8001/auth/callback"

    # social login: linux.do Connect (OAuth2 + /api/user, has trust_level)
    # Enabled iff linuxdo_client_id is set. Secrets come from env only.
    linuxdo_client_id: str = ""
    linuxdo_client_secret: str = ""
    linuxdo_authorize_endpoint: str = "https://connect.linux.do/oauth2/authorize"
    linuxdo_token_endpoint: str = "https://connect.linux.do/oauth2/token"
    linuxdo_userinfo_endpoint: str = "https://connect.linux.do/api/user"
    linuxdo_scope: str = ""
    linuxdo_redirect_uri: str = "https://retrieval.rnarket.com/api/linuxdooid/callback"
    linuxdo_min_trust_level: int = 0                # reject logins below this (0 = no gate)

    # social login: Google (OIDC userinfo). Enabled iff google_client_id is set.
    google_client_id: str = ""
    google_client_secret: str = ""
    google_authorize_endpoint: str = "https://accounts.google.com/o/oauth2/v2/auth"
    google_token_endpoint: str = "https://oauth2.googleapis.com/token"
    google_userinfo_endpoint: str = "https://openidconnect.googleapis.com/v1/userinfo"
    google_scope: str = "openid email profile"
    google_redirect_uri: str = "https://retrieval.rnarket.com/api/google/callback"

    # internal compute (4090) — reverse-proxy target + shared token
    internal_base_url: str = "http://10.100.100.111:8000"
    internal_token: str = ""                        # must match 4090 INTERNAL_TOKEN

    # billing / payment — per-op credit costs (Decimal; only `paying` tier is charged,
    # internal/admin are exempt). member+ (internal) get everything free.
    search_cost: Decimal = Decimal("1")          # /api/search + /api/match-doc
    paper_view_cost: Decimal = Decimal("0.1")    # detail/fulltext/Markdown; PDF is member-only
    code_index_cost: Decimal = Decimal("1")      # POST /api/code/index (embed a repo)
    code_search_cost: Decimal = Decimal("0.1")   # POST /api/code/search
    journal_write_cost: Decimal = Decimal("0.1") # POST /api/journal/entries (record / batch index)
    # journal search + index report/inventory are free reads.
    payment_provider: str = "mock"                  # "mock" | "epay" (linux.do Credit)
    payment_secret: str = "mock-secret"             # epay `key` (Client Secret)
    payment_pid: str = ""                           # epay `pid` (linux.do Client ID)
    payment_gateway_base: str = "https://credit.linux.do/epay"
    payment_return_url: str = ""                    # browser return after pay (untrusted)
    payment_notify_url: str = "http://localhost:8001/api/payment/notify"
    public_base_url: str = "http://localhost:8001"

    @property
    def jwks_uri(self) -> str:
        return self.oidc_jwks_uri or f"{self.oidc_issuer.rstrip('/')}/keys"


@lru_cache
def get_settings() -> GatewaySettings:
    return GatewaySettings()
