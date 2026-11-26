"""Tests for the account service — first-OIDC registration, sk-key issue/resolve/revoke."""
import pytest

from gateway.auth import accounts
from gateway.auth.keys import SK_PREFIX
from gateway.db.models import ApiKey, Tier, User
from gateway.db.session import make_engine, make_session_factory

PEPPER = "test-pepper"


@pytest.fixture
def session():
    engine = make_engine("sqlite://")
    s = make_session_factory(engine)()
    try:
        yield s
    finally:
        s.close()


# --- registration -----------------------------------------------------------

def test_register_creates_paying_user(session):
    u = accounts.register(session, oidc_sub="cdk:abc", username="alice")
    session.commit()
    assert u.id is not None
    assert u.oidc_sub == "cdk:abc"
    assert u.username == "alice"
    assert u.tier == Tier.paying
    assert u.balance == 0


def test_register_admin_claim_gets_admin_tier(session):
    u = accounts.register(session, oidc_sub="cdk:root", username="root", is_admin=True)
    session.commit()
    assert u.tier == Tier.admin


def test_register_duplicate_sub_raises(session):
    accounts.register(session, oidc_sub="cdk:abc", username="alice")
    session.commit()
    with pytest.raises(accounts.AlreadyRegistered):
        accounts.register(session, oidc_sub="cdk:abc", username="other")


def test_register_duplicate_username_raises(session):
    accounts.register(session, oidc_sub="cdk:abc", username="alice")
    session.commit()
    with pytest.raises(accounts.UsernameTaken):
        accounts.register(session, oidc_sub="cdk:def", username="alice")


def test_get_user_by_sub(session):
    accounts.register(session, oidc_sub="cdk:abc", username="alice")
    session.commit()
    found = accounts.get_user_by_sub(session, "cdk:abc")
    assert found is not None and found.username == "alice"
    assert accounts.get_user_by_sub(session, "cdk:nope") is None


# --- sk-key issue / resolve / revoke ----------------------------------------

def test_issue_key_returns_plaintext_and_stores_hash_only(session):
    u = accounts.register(session, oidc_sub="cdk:abc", username="alice")
    session.commit()
    key, ak = accounts.issue_key(session, u, PEPPER)
    session.commit()
    assert key.startswith(SK_PREFIX)
    assert ak.user_id == u.id
    assert ak.prefix == key[:6] and ak.last4 == key[-4:]
    # plaintext is NOT stored anywhere on the row
    assert key not in ak.key_hash
    assert len(ak.key_hash) == 64


def test_resolve_key_finds_user(session):
    u = accounts.register(session, oidc_sub="cdk:abc", username="alice")
    session.commit()
    key, _ = accounts.issue_key(session, u, PEPPER)
    session.commit()
    resolved = accounts.resolve_key(session, key, PEPPER)
    assert resolved is not None and resolved.id == u.id


def test_resolve_key_wrong_key_returns_none(session):
    u = accounts.register(session, oidc_sub="cdk:abc", username="alice")
    session.commit()
    accounts.issue_key(session, u, PEPPER)
    session.commit()
    assert accounts.resolve_key(session, "sk-totally-wrong", PEPPER) is None


def test_resolve_key_updates_last_used(session):
    u = accounts.register(session, oidc_sub="cdk:abc", username="alice")
    session.commit()
    key, ak = accounts.issue_key(session, u, PEPPER)
    session.commit()
    assert ak.last_used_at is None
    accounts.resolve_key(session, key, PEPPER)
    session.commit()
    assert session.get(ApiKey, ak.id).last_used_at is not None


def test_revoked_key_does_not_resolve(session):
    u = accounts.register(session, oidc_sub="cdk:abc", username="alice")
    session.commit()
    key, ak = accounts.issue_key(session, u, PEPPER)
    session.commit()
    accounts.revoke_key(session, ak.id)
    session.commit()
    assert accounts.resolve_key(session, key, PEPPER) is None
