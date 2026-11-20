"""Account service: first-OIDC registration + sk-key lifecycle.

The OIDC Worker authenticates a person and hands us verified claims (`sub`,
`is_admin`, chosen `username`). On first login we `register()` them; thereafter
the durable credential is the sk-key, so we `issue_key()` once and `resolve_key()`
it on every API request. Sensitive material (only the key HASH) lives here on the
web backend — never on the shared 4090.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from gateway.auth.keys import generate_key, hash_key, key_display
from gateway.db.models import ApiKey, Tier, User, UserIdentity

logger = logging.getLogger("gateway.auth.accounts")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UsernameTaken(Exception):
    """Chosen username already belongs to another account."""


class AlreadyRegistered(Exception):
    """This identity already has an account (login, don't re-register)."""


class IdentityTaken(Exception):
    """This (provider, sub) is already bound to a DIFFERENT account."""


def split_sub(oidc_sub: str) -> tuple[str, str]:
    """Decompose a primary identity tag into (provider, raw_sub).

    "cdk:admin_arc" → ("cdk", "admin_arc"); "linuxdo:12345" → ("linuxdo", "12345").
    A bare value (legacy bootstrap "root", no colon) is treated as a cdk subject.
    """
    provider, sep, raw = oidc_sub.partition(":")
    if not sep:
        return "cdk", oidc_sub
    return provider, raw


def get_user_by_sub(session, oidc_sub: str) -> User | None:
    return session.query(User).filter(User.oidc_sub == oidc_sub).first()


# --- identities (account binding) -------------------------------------------

def get_identity(session, provider: str, sub: str) -> UserIdentity | None:
    return (session.query(UserIdentity)
            .filter(UserIdentity.provider == provider, UserIdentity.sub == sub)
            .first())


def resolve_identity(session, provider: str, sub: str) -> User | None:
    """An external (provider, sub) → its owning User, or None if unbound."""
    ident = get_identity(session, provider, sub)
    return session.get(User, ident.user_id) if ident else None


def list_identities(session, user: User) -> list[UserIdentity]:
    return (session.query(UserIdentity)
            .filter(UserIdentity.user_id == user.id)
            .order_by(UserIdentity.id).all())


def add_identity(session, user: User, provider: str, sub: str) -> UserIdentity:
    """Bind (provider, sub) to `user`. No-op if already bound to this user;
    raises IdentityTaken if it belongs to a different account."""
    existing = get_identity(session, provider, sub)
    if existing is not None:
        if existing.user_id == user.id:
            return existing
        raise IdentityTaken(f"{provider}:{sub}")
    ident = UserIdentity(user_id=user.id, provider=provider, sub=sub)
    session.add(ident)
    session.flush()
    logger.info("bound identity user=%s provider=%s sub=%s", user.id, provider, sub)
    return ident


def remove_identity(session, user: User, identity_id: int) -> bool:
    """Unlink one identity from `user`. Refuses to strip the user's LAST login
    method, or one they don't own. Returns False (no change) in those cases."""
    ident = session.get(UserIdentity, identity_id)
    if ident is None or ident.user_id != user.id:
        return False
    if len(list_identities(session, user)) <= 1:
        return False
    session.delete(ident)
    session.flush()
    logger.info("unbound identity id=%s user=%s", identity_id, user.id)
    return True


def register_with_identity(session, *, provider: str, sub: str, username: str,
                           is_admin: bool = False,
                           oidc_sub: str | None = None) -> User:
    """Create an account for a first-time identity with their chosen username,
    and bind that identity. Tier: admin claim → admin; everyone else `paying`.

    `oidc_sub` (the primary tag) defaults to "{provider}:{sub}".
    """
    if resolve_identity(session, provider, sub) is not None:
        raise AlreadyRegistered(f"{provider}:{sub}")
    if session.query(User).filter(User.username == username).first() is not None:
        raise UsernameTaken(username)
    tag = oidc_sub or f"{provider}:{sub}"
    tier = Tier.admin if is_admin else Tier.paying
    user = User(oidc_sub=tag, username=username, tier=tier, balance=0)
    session.add(user)
    session.flush()
    add_identity(session, user, provider, sub)
    logger.info("registered user=%s tag=%s provider=%s tier=%s",
                username, tag, provider, tier.value)
    return user


def register(session, oidc_sub: str, username: str, is_admin: bool = False) -> User:
    """Back-compat shim: register by a primary tag (e.g. "cdk:abc"). Splits the
    tag into a (provider, sub) identity and preserves the original tag verbatim."""
    provider, sub = split_sub(oidc_sub)
    return register_with_identity(session, provider=provider, sub=sub,
                                  username=username, is_admin=is_admin,
                                  oidc_sub=oidc_sub)


def backfill_identities(session) -> int:
    """Idempotent migration: ensure every existing user has at least one identity
    row, derived from its `oidc_sub`. Safe to run on every boot. Returns the
    number of identity rows created."""
    created = 0
    for user in session.query(User).all():
        if session.query(UserIdentity).filter(UserIdentity.user_id == user.id).first():
            continue
        provider, sub = split_sub(user.oidc_sub)
        if get_identity(session, provider, sub) is None:
            session.add(UserIdentity(user_id=user.id, provider=provider, sub=sub))
            created += 1
    if created:
        session.flush()
        logger.info("backfilled %s identity row(s) from oidc_sub", created)
    return created


def issue_key(session, user: User, pepper: str) -> tuple[str, ApiKey]:
    """Mint a fresh sk-key for `user`. Returns (plaintext, row); the plaintext is
    shown to the user ONCE — only its HMAC hash is persisted."""
    key = generate_key()
    prefix, last4 = key_display(key)
    ak = ApiKey(user_id=user.id, key_hash=hash_key(key, pepper),
                prefix=prefix, last4=last4)
    session.add(ak)
    session.flush()
    logger.info("issued sk-key id=%s user=%s prefix=%s", ak.id, user.id, prefix)
    return key, ak


def resolve_key(session, presented: str, pepper: str) -> User | None:
    """sk-key → owning User (O(1) hash lookup), or None if unknown/revoked.
    Stamps last_used_at as a side effect."""
    h = hash_key(presented, pepper)
    ak = (session.query(ApiKey)
          .filter(ApiKey.key_hash == h, ApiKey.revoked_at.is_(None)).first())
    if ak is None:
        return None
    ak.last_used_at = _utcnow()
    session.flush()
    return session.get(User, ak.user_id)


def revoke_key(session, key_id: int) -> None:
    ak = session.get(ApiKey, key_id)
    if ak is None or ak.revoked_at is not None:
        return
    ak.revoked_at = _utcnow()
    session.flush()
    logger.info("revoked sk-key id=%s user=%s", key_id, ak.user_id)
