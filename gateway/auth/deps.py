"""Auth dependency core: turn a request's credentials into a User, plus tier
predicates and admin bootstrap.

Two credential paths, both resolving to the same `users` row:
  * `Authorization: Bearer sk-...`  — the durable API credential (sk-key).
  * web session cookie               — carries the user id after OIDC login.

Admin is a fixed root key (env `ADMIN_ROOT_KEY`): `bootstrap_admin` ensures a
`root` admin user exists and registers that key as a normal sk-key, so it
resolves through the same code path — no special-casing on every request.

The FastAPI `Depends(...)` wrappers live with the app entrypoint; this module
stays framework-light so it's unit-testable without spinning up the app.
"""
from __future__ import annotations

import logging

from gateway.auth import accounts
from gateway.auth.keys import hash_key, key_display
from gateway.db.models import ApiKey, Tier, User

logger = logging.getLogger("gateway.auth.deps")

ROOT_OIDC_SUB = "root"
ROOT_USERNAME = "root"


def extract_bearer(authorization: str | None) -> str | None:
    """Pull the token out of an `Authorization: Bearer <token>` header.
    Scheme match is case-insensitive; returns None if absent/blank/other-scheme."""
    if not authorization:
        return None
    scheme, _, rest = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = rest.strip()
    return token or None


def resolve_user(db, *, bearer: str | None = None, session_uid: int | None = None,
                 pepper: str) -> User | None:
    """Resolve the caller to a User. Bearer sk-key takes precedence over the
    session cookie. Returns None when no credential resolves."""
    if bearer:
        return accounts.resolve_key(db, bearer, pepper)
    if session_uid is not None:
        return db.get(User, session_uid)
    return None


def bootstrap_admin(db, *, root_key: str, pepper: str,
                    username: str = ROOT_USERNAME) -> User:
    """Idempotently ensure the fixed-root-key admin exists. Creates the `root`
    admin user if missing (or promotes an existing `root` subject), and registers
    `root_key` as an sk-key so it authenticates via the normal path."""
    user = accounts.get_user_by_sub(db, ROOT_OIDC_SUB)
    if user is None:
        user = User(oidc_sub=ROOT_OIDC_SUB, username=username, tier=Tier.admin, balance=0)
        db.add(user)
        db.flush()
        logger.info("bootstrap_admin created root user id=%s", user.id)
    elif user.tier != Tier.admin:
        user.tier = Tier.admin
        logger.info("bootstrap_admin promoted root user id=%s to admin", user.id)

    h = hash_key(root_key, pepper)
    if db.query(ApiKey).filter(ApiKey.key_hash == h).first() is None:
        prefix, last4 = key_display(root_key)
        db.add(ApiKey(user_id=user.id, key_hash=h, prefix=prefix, last4=last4))
        db.flush()
        logger.info("bootstrap_admin registered root sk-key for user id=%s", user.id)
    return user


def is_admin(user: User) -> bool:
    return user.tier == Tier.admin


def is_internal(user: User) -> bool:
    """Internal + admin tiers may use the invite-only private APIs (code KB, journal,
    index inventory, ACE playbook). Paying users cannot yet (operator's call: these
    stay internal-only for now — flip a route to `get_current_user` to open it up)."""
    return user.tier in (Tier.internal, Tier.admin)


def is_billable(user: User) -> bool:
    """Only `paying` users are metered; internal and admin are billing-exempt."""
    return user.tier == Tier.paying
