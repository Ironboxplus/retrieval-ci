"""Account-binding model: one User, many UserIdentity rows (cdk / linuxdo / google)
all sharing one balance. Covers identity resolve/add/register, the last-identity
unlink guard, conflict detection, and the idempotent backfill migration that turns
existing `users.oidc_sub` into identity rows.
"""
import pytest

from gateway.auth import accounts
from gateway.db.models import Tier, User, UserIdentity
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


# --- split_sub (oidc_sub → provider, raw) -----------------------------------

def test_split_sub_prefixed():
    assert accounts.split_sub("cdk:admin_arc") == ("cdk", "admin_arc")
    assert accounts.split_sub("linuxdo:12345") == ("linuxdo", "12345")
    assert accounts.split_sub("google:108") == ("google", "108")


def test_split_sub_bare_is_cdk_legacy():
    # bootstrap root user has a bare oidc_sub "root"
    assert accounts.split_sub("root") == ("cdk", "root")


# --- register_with_identity --------------------------------------------------

def test_register_with_identity_creates_user_and_identity(session):
    u = accounts.register_with_identity(session, provider="linuxdo", sub="12345",
                                        username="ld_user")
    session.commit()
    assert u.id is not None
    assert u.oidc_sub == "linuxdo:12345"
    assert u.tier == Tier.paying
    idents = accounts.list_identities(session, u)
    assert len(idents) == 1
    assert idents[0].provider == "linuxdo" and idents[0].sub == "12345"


def test_register_with_identity_admin(session):
    u = accounts.register_with_identity(session, provider="cdk", sub="root",
                                        username="root2", is_admin=True)
    session.commit()
    assert u.tier == Tier.admin


def test_register_with_identity_duplicate_identity_raises(session):
    accounts.register_with_identity(session, provider="google", sub="g1", username="a")
    session.commit()
    with pytest.raises(accounts.AlreadyRegistered):
        accounts.register_with_identity(session, provider="google", sub="g1", username="b")


def test_register_with_identity_duplicate_username_raises(session):
    accounts.register_with_identity(session, provider="google", sub="g1", username="a")
    session.commit()
    with pytest.raises(accounts.UsernameTaken):
        accounts.register_with_identity(session, provider="linuxdo", sub="l1", username="a")


# --- resolve_identity --------------------------------------------------------

def test_resolve_identity_finds_user(session):
    u = accounts.register_with_identity(session, provider="linuxdo", sub="12345", username="x")
    session.commit()
    found = accounts.resolve_identity(session, "linuxdo", "12345")
    assert found is not None and found.id == u.id
    assert accounts.resolve_identity(session, "linuxdo", "999") is None
    assert accounts.resolve_identity(session, "google", "12345") is None  # provider-scoped


# --- add_identity (binding) --------------------------------------------------

def test_add_identity_links_second_provider_to_same_user(session):
    u = accounts.register_with_identity(session, provider="linuxdo", sub="12345", username="x")
    session.commit()
    accounts.add_identity(session, u, "google", "g-9")
    session.commit()
    # both identities resolve to the SAME user (shared balance)
    assert accounts.resolve_identity(session, "linuxdo", "12345").id == u.id
    assert accounts.resolve_identity(session, "google", "g-9").id == u.id
    assert len(accounts.list_identities(session, u)) == 2


def test_add_identity_idempotent_for_same_user(session):
    u = accounts.register_with_identity(session, provider="linuxdo", sub="12345", username="x")
    session.commit()
    accounts.add_identity(session, u, "google", "g-9")
    accounts.add_identity(session, u, "google", "g-9")  # no-op, no raise
    session.commit()
    assert len(accounts.list_identities(session, u)) == 2


def test_add_identity_owned_by_other_user_raises(session):
    a = accounts.register_with_identity(session, provider="linuxdo", sub="a1", username="a")
    b = accounts.register_with_identity(session, provider="linuxdo", sub="b1", username="b")
    session.commit()
    # g-shared already belongs to a
    accounts.add_identity(session, a, "google", "g-shared")
    session.commit()
    with pytest.raises(accounts.IdentityTaken):
        accounts.add_identity(session, b, "google", "g-shared")


# --- remove_identity (unlink) ------------------------------------------------

def test_remove_identity_drops_one_binding(session):
    u = accounts.register_with_identity(session, provider="linuxdo", sub="12345", username="x")
    session.commit()
    g = accounts.add_identity(session, u, "google", "g-9")
    session.commit()
    assert accounts.remove_identity(session, u, g.id) is True
    session.commit()
    assert accounts.resolve_identity(session, "google", "g-9") is None
    assert len(accounts.list_identities(session, u)) == 1


def test_remove_last_identity_refused(session):
    u = accounts.register_with_identity(session, provider="linuxdo", sub="12345", username="x")
    session.commit()
    only = accounts.list_identities(session, u)[0]
    assert accounts.remove_identity(session, u, only.id) is False  # can't strip last login method
    assert len(accounts.list_identities(session, u)) == 1


def test_remove_identity_not_owned_refused(session):
    a = accounts.register_with_identity(session, provider="linuxdo", sub="a1", username="a")
    b = accounts.register_with_identity(session, provider="google", sub="b1", username="b")
    session.commit()
    b_ident = accounts.list_identities(session, b)[0]
    assert accounts.remove_identity(session, a, b_ident.id) is False


# --- backfill migration ------------------------------------------------------

def test_backfill_creates_identities_from_oidc_sub(session):
    # Simulate the pre-migration state: users with oidc_sub but no identity rows.
    session.add(User(oidc_sub="cdk:admin_arc", username="arc", tier=Tier.admin))
    session.add(User(oidc_sub="root", username="root", tier=Tier.admin))
    session.commit()
    assert session.query(UserIdentity).count() == 0

    n = accounts.backfill_identities(session)
    session.commit()
    assert n == 2
    # the cf-oidc CDK login (claims.sub="cdk:admin_arc") now resolves
    assert accounts.resolve_identity(session, "cdk", "admin_arc").username == "arc"
    assert accounts.resolve_identity(session, "cdk", "root").username == "root"


def test_backfill_is_idempotent(session):
    session.add(User(oidc_sub="cdk:admin_arc", username="arc", tier=Tier.admin))
    session.commit()
    accounts.backfill_identities(session)
    session.commit()
    n2 = accounts.backfill_identities(session)  # second run does nothing
    session.commit()
    assert n2 == 0
    assert session.query(UserIdentity).count() == 1
