"""Tests for request metering — pre-debit a paying user, refund if compute fails,
no-op for billing-exempt tiers, 402 surfaces as InsufficientCredits."""
import threading
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, inspect, text

from gateway.billing import ledger, metering
from gateway.db.models import Tier, Transaction, User
from gateway.db.session import make_engine, make_session_factory


@pytest.fixture
def session():
    engine = make_engine("sqlite://")
    s = make_session_factory(engine)()
    try:
        yield s
    finally:
        s.close()


def _user(s, tier=Tier.paying, balance=5):
    u = User(oidc_sub=f"cdk:{tier.value}", username=tier.value, tier=tier, balance=balance)
    s.add(u)
    s.commit()
    return u


def test_gateway_schema_migrates_legacy_idempotency_requests_table(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'legacy-idempotency.db'}"
    legacy_engine = create_engine(database_url)
    with legacy_engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE idempotency_requests ("
            "idempotency_key VARCHAR(128) PRIMARY KEY, "
            "user_id INTEGER NOT NULL, amount NUMERIC(12, 2) NOT NULL, "
            "ref VARCHAR(128) NOT NULL, status VARCHAR(16) NOT NULL, "
            "attempt_number INTEGER NOT NULL, created_at DATETIME NOT NULL, "
            "updated_at DATETIME NOT NULL)"
        ))
        connection.execute(text(
            "INSERT INTO idempotency_requests VALUES ("
            "'legacy-request', 1, 1, 'GET /artifact', 'pending', 1, "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        ))
    legacy_engine.dispose()

    migrated_engine = make_engine(database_url)
    columns = {
        column["name"]
        for column in inspect(migrated_engine).get_columns("idempotency_requests")
    }
    with migrated_engine.connect() as connection:
        lease_token = connection.execute(text(
            "SELECT lease_token FROM idempotency_requests "
            "WHERE idempotency_key='legacy-request'"
        )).scalar_one()

    assert "lease_token" in columns
    assert len(lease_token) == 32


def test_paying_success_debits_once(session):
    u = _user(session, Tier.paying, balance=5)
    with metering.charge(session, u, cost=1, ref="search"):
        pass  # compute succeeds
    assert session.get(User, u.id).balance == 4
    kinds = [t.kind for t in session.query(Transaction).all()]
    assert kinds == ["debit"]


def test_paying_failure_refunds(session):
    u = _user(session, Tier.paying, balance=5)
    with pytest.raises(RuntimeError):
        with metering.charge(session, u, cost=1, ref="search", idempotency_key="req-1"):
            raise RuntimeError("compute 5xx")
    assert session.get(User, u.id).balance == 5  # made whole
    kinds = sorted(t.kind for t in session.query(Transaction).all())
    assert kinds == ["debit", "refund"]


def test_refunded_idempotent_request_can_retry_and_charge_once(session):
    u = _user(session, Tier.paying, balance=5)

    with pytest.raises(RuntimeError):
        with metering.charge(
            session, u, cost=1, ref="GET /artifact", idempotency_key="req-retry"
        ):
            raise RuntimeError("first transfer failed")

    with metering.charge(
        session, u, cost=1, ref="GET /artifact", idempotency_key="req-retry"
    ):
        pass

    assert session.get(User, u.id).balance == 4
    assert [t.kind for t in session.query(Transaction).order_by(Transaction.id)] == [
        "debit", "refund", "debit",
    ]


def test_failed_duplicate_does_not_refund_a_prior_success(session):
    u = _user(session, Tier.paying, balance=5)

    with metering.charge(
        session, u, cost=1, ref="GET /artifact", idempotency_key="req-success"
    ):
        pass

    with pytest.raises(RuntimeError):
        with metering.charge(
            session, u, cost=1, ref="GET /artifact", idempotency_key="req-success"
        ):
            raise RuntimeError("duplicate transport failed")

    assert session.get(User, u.id).balance == 4
    assert [t.kind for t in session.query(Transaction).order_by(Transaction.id)] == [
        "debit",
    ]


def test_idempotency_key_rejects_different_request_semantics(session):
    u = _user(session, Tier.paying, balance=5)
    with metering.charge(
        session, u, cost=1, ref="GET /paper-a", idempotency_key="req-conflict"
    ):
        pass

    with pytest.raises(ledger.IdempotencyConflict):
        with metering.charge(
            session, u, cost=1, ref="GET /paper-b", idempotency_key="req-conflict"
        ):
            pass

    assert session.get(User, u.id).balance == 4


def test_credit_cannot_reuse_a_debit_idempotency_key(session):
    u = _user(session, Tier.paying, balance=5)
    ledger.debit(
        session, u.id, 1, ref="GET /artifact", idempotency_key="shared-key"
    )
    session.commit()

    with pytest.raises(ledger.IdempotencyConflict):
        ledger.credit(
            session,
            u.id,
            10,
            kind="topup",
            ref="payment-order",
            idempotency_key="shared-key",
        )

    assert session.get(User, u.id).balance == 4


def test_concurrent_duplicate_is_rejected_until_the_owner_finishes(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'concurrent-idempotency.db'}")
    session_factory = make_session_factory(engine)
    setup = session_factory()
    user = _user(setup, Tier.paying, balance=5)
    user_id = user.id
    setup.close()

    owner_entered = threading.Event()
    release_owner = threading.Event()
    owner_errors: list[str] = []

    def fail_owner() -> None:
        owner_session = session_factory()
        owner_user = owner_session.get(User, user_id)
        try:
            with metering.charge(
                owner_session,
                owner_user,
                cost=1,
                ref="GET /artifact",
                idempotency_key="concurrent-request",
            ):
                owner_entered.set()
                assert release_owner.wait(timeout=5)
                raise RuntimeError("owner failed")
        except RuntimeError as exc:
            owner_errors.append(str(exc))
        finally:
            owner_session.close()

    owner_thread = threading.Thread(target=fail_owner)
    owner_thread.start()
    assert owner_entered.wait(timeout=5)

    duplicate_session = session_factory()
    duplicate_user = duplicate_session.get(User, user_id)
    with pytest.raises(ledger.IdempotencyInProgress):
        with metering.charge(
            duplicate_session,
            duplicate_user,
            cost=1,
            ref="GET /artifact",
            idempotency_key="concurrent-request",
        ):
            pass
    duplicate_session.close()

    release_owner.set()
    owner_thread.join(timeout=5)
    assert owner_errors == ["owner failed"]

    retry_session = session_factory()
    retry_user = retry_session.get(User, user_id)
    with metering.charge(
        retry_session,
        retry_user,
        cost=1,
        ref="GET /artifact",
        idempotency_key="concurrent-request",
    ):
        pass
    assert retry_session.get(User, user_id).balance == 4
    retry_session.close()


def test_stale_pending_request_is_refunded_and_reclaimed_after_a_crash(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'stale-idempotency.db'}")
    session_factory = make_session_factory(engine)
    crashed_session = session_factory()
    user = _user(crashed_session, Tier.paying, balance=5)
    user_id = user.id

    reservation = ledger.reserve_debit_attempt(
        crashed_session,
        user_id,
        1,
        "GET /artifact",
        "crashed-request",
    )
    ledger.debit(
        crashed_session,
        user_id,
        1,
        ref="GET /artifact",
        idempotency_key=reservation.attempt_key,
    )
    crashed_session.commit()

    pending = crashed_session.get(ledger.IdempotencyRequest, "crashed-request")
    pending.updated_at = datetime.now(timezone.utc) - timedelta(hours=1)
    crashed_session.commit()
    crashed_session.close()

    retry_session = session_factory()
    retry_user = retry_session.get(User, user_id)
    with metering.charge(
        retry_session,
        retry_user,
        cost=1,
        ref="GET /artifact",
        idempotency_key="crashed-request",
    ):
        pass

    assert retry_session.get(User, user_id).balance == 4
    assert [
        transaction.kind
        for transaction in retry_session.query(Transaction).order_by(Transaction.id)
    ] == ["debit", "refund", "debit"]
    request = retry_session.get(ledger.IdempotencyRequest, "crashed-request")
    assert request.status == "succeeded"
    assert request.attempt_number == 2
    retry_session.close()


def test_insufficient_raises_before_compute(session):
    u = _user(session, Tier.paying, balance=0)
    ran = {"compute": False}
    with pytest.raises(ledger.InsufficientCredits):
        with metering.charge(session, u, cost=1):
            ran["compute"] = True
    assert ran["compute"] is False           # never entered the body
    assert session.get(User, u.id).balance == 0
    assert session.query(Transaction).count() == 0


@pytest.mark.parametrize("tier", [Tier.internal, Tier.admin])
def test_exempt_tiers_not_charged(session, tier):
    u = _user(session, tier, balance=5)
    with metering.charge(session, u, cost=1):
        pass
    assert session.get(User, u.id).balance == 5
    assert session.query(Transaction).count() == 0


def test_fractional_cost_charges_and_refunds(session):
    from decimal import Decimal
    u = _user(session, Tier.paying, balance=1)
    with metering.charge(session, u, cost=Decimal("0.1"), ref="code-search"):
        pass
    assert Decimal(str(session.get(User, u.id).balance)) == Decimal("0.9")
    with pytest.raises(RuntimeError):
        with metering.charge(session, u, cost=Decimal("0.1"), ref="code", idempotency_key="r1"):
            raise RuntimeError("boom")
    assert Decimal(str(session.get(User, u.id).balance)) == Decimal("0.9")  # refunded


def test_exempt_tier_failure_no_refund_noise(session):
    u = _user(session, Tier.admin, balance=5)
    with pytest.raises(RuntimeError):
        with metering.charge(session, u, cost=1):
            raise RuntimeError("boom")
    assert session.query(Transaction).count() == 0
