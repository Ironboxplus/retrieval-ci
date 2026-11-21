"""FastAPI dependencies — pull shared state off `app.state`, resolve the caller.

Kept separate from the framework-light `gateway.auth.deps` core (which holds the
testable resolution logic); this module is the thin FastAPI adapter.
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request

from gateway.auth.deps import extract_bearer, is_admin, is_internal, resolve_user
from gateway.db.models import Tier, User


@dataclass(frozen=True)
class AccessLogIdentity:
    """Detached-safe identity retained until access-log middleware completes."""

    id: int
    username: str
    tier: Tier


def get_settings(request: Request):
    return request.app.state.settings


def get_db(request: Request):
    db = request.app.state.session_factory()
    try:
        yield db
    finally:
        db.close()


def get_proxy(request: Request):
    return request.app.state.proxy


def get_provider(request: Request):
    return request.app.state.payment_provider


def get_current_user(request: Request, db=Depends(get_db), settings=Depends(get_settings)) -> User:
    """Resolve via `Authorization: Bearer sk-...` or the session cookie; 401 if neither."""
    bearer = extract_bearer(request.headers.get("Authorization"))
    session_uid = request.session.get("uid")
    user = resolve_user(db, bearer=bearer, session_uid=session_uid, pepper=settings.key_pepper)
    if user is None:
        raise HTTPException(status_code=401, detail="authentication required")
    # The route may commit or roll back its ORM session before middleware logs.
    request.state.user = AccessLogIdentity(
        id=user.id,
        username=user.username,
        tier=user.tier,
    )
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if not is_admin(user):
        raise HTTPException(status_code=403, detail="admin only")
    return user


def require_internal(user: User = Depends(get_current_user)) -> User:
    """Gate the invite-only private APIs (code KB / journal / index inventory / ACE
    playbook) to internal + admin tiers. Logged-in paying users get 403 for now."""
    if not is_internal(user):
        raise HTTPException(status_code=403, detail="internal access only")
    return user
