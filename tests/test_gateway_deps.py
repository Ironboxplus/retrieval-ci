"""Tests for auth dependency core — bearer extraction, user resolution,
admin bootstrap (fixed root key), tier predicates."""
import pytest

from gateway.auth import accounts, deps
from gateway.db.models import ApiKey, Tier, User
from gateway.db.session import make_engine, make_session_factory

PEPPER = "test-pepper"
ROOT_KEY = "sk-root-fixed-key-xyz"


@pytest.fixture
def session():
    engine = make_engine("sqlite://")
    s = make_session_factory(engine)()
    try:
        yield s
    finally:
        s.close()


# --- extract_bearer ----------------------------------------------------------

def test_extract_bearer():
    assert deps.extract_bearer(None) is None
    assert deps.extract_bearer("Bearer sk-abc") == "sk-abc"
    assert deps.extract_bearer("bearer sk-abc") == "sk-abc"   # case-insensitive scheme
    assert deps.extract_bearer("Basic xyz") is None
    assert deps.extract_bearer("Bearer    ") is None
    assert deps.extract_bearer("") is None


# --- resolve_user ------------------------------------------------------------

def test_resolve_user_by_bearer(session):
    u = accounts.register(session, oidc_sub="cdk:a", username="alice")
    session.commit()
    key, _ = accounts.issue_key(session, u, PEPPER)
    session.commit()
    got = deps.resolve_user(session, bearer=key, pepper=PEPPER)
    assert got is not None and got.id == u.id


def test_resolve_user_by_session_uid(session):
    u = accounts.register(session, oidc_sub="cdk:a", username="alice")
    session.commit()
    got = deps.resolve_user(session, session_uid=u.id, pepper=PEPPER)
    assert got is not None and got.id == u.id


def test_resolve_user_none_when_no_credentials(session):
    assert deps.resolve_user(session, pepper=PEPPER) is None
    assert deps.resolve_user(session, bearer="sk-nope", pepper=PEPPER) is None


# --- bootstrap_admin (fixed root key) ----------------------------------------

def test_bootstrap_admin_creates_admin_and_root_key_resolves(session):
    admin = deps.bootstrap_admin(session, root_key=ROOT_KEY, pepper=PEPPER)
    session.commit()
    assert admin.tier == Tier.admin
    resolved = deps.resolve_user(session, bearer=ROOT_KEY, pepper=PEPPER)
    assert resolved is not None and resolved.id == admin.id


def test_bootstrap_admin_is_idempotent(session):
    deps.bootstrap_admin(session, root_key=ROOT_KEY, pepper=PEPPER)
    session.commit()
    deps.bootstrap_admin(session, root_key=ROOT_KEY, pepper=PEPPER)
    session.commit()
    assert session.query(User).filter(User.oidc_sub == "root").count() == 1
    assert session.query(ApiKey).count() == 1


def test_bootstrap_admin_promotes_existing_subject(session):
    # an existing non-admin "root" subject gets promoted, not duplicated
    u = User(oidc_sub="root", username="root", tier=Tier.paying, balance=0)
    session.add(u)
    session.commit()
    admin = deps.bootstrap_admin(session, root_key=ROOT_KEY, pepper=PEPPER)
    session.commit()
    assert admin.id == u.id and admin.tier == Tier.admin


# --- tier predicates ---------------------------------------------------------

def test_tier_predicates():
    paying = User(tier=Tier.paying)
    internal = User(tier=Tier.internal)
    admin = User(tier=Tier.admin)
    assert deps.is_billable(paying) is True
    assert deps.is_billable(internal) is False   # billing-exempt
    assert deps.is_billable(admin) is False
    assert deps.is_admin(admin) is True
    assert deps.is_admin(paying) is False
