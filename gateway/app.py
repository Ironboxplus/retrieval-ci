"""Gateway app factory (web backend / BWH).

Wires: signed-cookie sessions, the app-tier DB (users/keys/ledger/orders), the
payment provider, the reverse proxy to the 4090, an OIDC JWKS client, and the
fixed-root-key admin bootstrap. Sensitive state lives only here.

    uvicorn gateway.app:app          # uses get_settings() (env GATEWAY_*)
"""
from __future__ import annotations

import asyncio
import logging
import time

from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware

from gateway.api import routes
from gateway.auth import accounts
from gateway.auth.deps import bootstrap_admin
from gateway.auth.oauth2 import OAuth2Provider
from gateway.db.session import make_engine, make_session_factory
from gateway.payment.base import PaymentProvider
from gateway.payment.epay import EpayProvider
from gateway.payment.mock import MockProvider
from gateway.proxy import InternalProxy
from gateway.settings import GatewaySettings, get_settings

logger = logging.getLogger("gateway.app")


def _make_payment_provider(settings: GatewaySettings) -> PaymentProvider:
    """Select the payment backend from config: real linux.do Credit epay, else mock."""
    if settings.payment_provider == "epay":
        logger.info("payment provider: epay (linux.do Credit) pid=%s base=%s",
                    settings.payment_pid, settings.payment_gateway_base)
        return EpayProvider(pid=settings.payment_pid, key=settings.payment_secret,
                            gateway_base=settings.payment_gateway_base)
    logger.info("payment provider: mock")
    return MockProvider(secret=settings.payment_secret)


def _make_social_providers(settings: GatewaySettings) -> dict[str, OAuth2Provider]:
    """Build the enabled social-login providers (those with a client_id set).
    linux.do carries a trust_level we gate on; Google is plain OIDC userinfo."""
    providers: dict[str, OAuth2Provider] = {}
    if settings.linuxdo_client_id:
        providers["linuxdo"] = OAuth2Provider(
            name="linuxdo", client_id=settings.linuxdo_client_id,
            client_secret=settings.linuxdo_client_secret,
            authorize_endpoint=settings.linuxdo_authorize_endpoint,
            token_endpoint=settings.linuxdo_token_endpoint,
            userinfo_endpoint=settings.linuxdo_userinfo_endpoint,
            redirect_uri=settings.linuxdo_redirect_uri, scope=settings.linuxdo_scope,
            sub_field="id", username_fields=("username", "name"),
            trust_field="trust_level", min_trust_level=settings.linuxdo_min_trust_level)
    if settings.google_client_id:
        providers["google"] = OAuth2Provider(
            name="google", client_id=settings.google_client_id,
            client_secret=settings.google_client_secret,
            authorize_endpoint=settings.google_authorize_endpoint,
            token_endpoint=settings.google_token_endpoint,
            userinfo_endpoint=settings.google_userinfo_endpoint,
            redirect_uri=settings.google_redirect_uri, scope=settings.google_scope,
            sub_field="sub", username_fields=("name", "email"))
    logger.info("social login providers enabled: %s", sorted(providers) or "none")
    return providers


def create_app(settings: GatewaySettings | None = None, *,
               payment_provider=None, proxy=None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="retrival gateway", version="0.1.0")
    app.add_middleware(SessionMiddleware, secret_key=settings.session_secret, same_site="lax")

    # Per-request access log (W128/W131): one JSON line per request — EVERY
    # request, authenticated or not ("所有的请求都要记录") → stdout → Loki on the
    # 4090. `get_current_user` stashes request.state.user when auth succeeds;
    # anonymous hits (401s, public endpoints, pre-auth failures) log with
    # user=None → null username and the client `ip` as the only identifier.
    # Metered routes stash request.state.billed_cost. BWH stays stateless.
    @app.middleware("http")
    async def _access_log_mw(request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        from gateway.access_log import emit, ship
        # Behind Traefik the socket peer is the proxy, so the real client lives in
        # X-Forwarded-For (first hop); fall back to the direct peer if absent.
        xff = request.headers.get("x-forwarded-for")
        ip = (xff.split(",")[0].strip() if xff
              else (request.client.host if request.client else None))
        record = emit(getattr(request.state, "user", None),
                      method=request.method, path=request.url.path,
                      status=response.status_code,
                      duration_ms=round((time.perf_counter() - start) * 1000, 1),
                      cost=getattr(request.state, "billed_cost", 0), ip=ip)
        # Fire-and-forget ship to the 4090 Loki relay over the existing tunnel;
        # never awaited so a logging hop can't add latency or fail the request.
        # Only when the authenticated internal tunnel is configured (prod) — keeps
        # the test suite (no token) from doing real network I/O.
        s = request.app.state.settings
        if s.internal_base_url and s.internal_token:
            asyncio.create_task(ship(record, base_url=s.internal_base_url,
                                     token=s.internal_token))
        return response

    engine = make_engine(settings.db_url)
    app.state.settings = settings
    app.state.session_factory = make_session_factory(engine)
    app.state.payment_provider = payment_provider or _make_payment_provider(settings)
    app.state.proxy = proxy or InternalProxy(settings.internal_base_url, settings.internal_token)
    app.state.oidc_exchange = None  # default real exchange; tests may inject a fake
    app.state.social_providers = _make_social_providers(settings)
    app.state.oauth_fetch = None    # default real fetch_identity; tests inject a fake

    try:
        from gateway.auth.oidc import make_jwks_client
        # Custom UA — Cloudflare 403s the default urllib UA on the Worker /keys (W120).
        app.state.jwks_client = make_jwks_client(settings.jwks_uri)
    except Exception as exc:  # pragma: no cover - PyJWKClient ctor is offline-safe
        logger.warning("jwks client init deferred: %s", exc)
        app.state.jwks_client = None

    # Fixed-root-key admin (env GATEWAY_ADMIN_ROOT_KEY) + identity backfill.
    # Both idempotent; backfill migrates pre-existing users.oidc_sub → identity rows
    # so the new account-binding model resolves legacy accounts on first boot.
    db = app.state.session_factory()
    try:
        if settings.admin_root_key:
            bootstrap_admin(db, root_key=settings.admin_root_key, pepper=settings.key_pepper)
            logger.info("admin bootstrap complete")
        accounts.backfill_identities(db)
        db.commit()
    finally:
        db.close()

    app.include_router(routes.router)
    return app


app = create_app()
