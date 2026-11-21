"""Tests for the credit ledger — atomic debit, idempotent credit, refund."""
import pytest

from gateway.billing import ledger
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


def _user(s, balance=5, tier=Tier.paying):
    u = User(oidc_sub="cdk:x", username="u", tier=tier, balance=balance)
    s.add(u)
    s.commit()
    return u


def test_debit_decrements_and_records(session):
    u = _user(session, balance=5)
    bal = ledger.debit(session, u.id, 1, ref="search")
    session.commit()
    assert bal == 4
    txns = session.query(Transaction).all()
    assert len(txns) == 1
    assert txns[0].kind == "debit" and txns[0].amount == -1 and txns[0].balance_after == 4


def test_debit_insufficient_raises_and_no_change(session):
    u = _user(session, balance=0)
    with pytest.raises(ledger.InsufficientCredits):
        ledger.debit(session, u.id, 1)
    session.rollback()
    assert session.get(User, u.id).balance == 0
    assert session.query(Transaction).count() == 0


def test_credit_topup_increments(session):
    u = _user(session, balance=0)
    bal = ledger.credit(session, u.id, 10, kind="topup", ref="order-1")
    session.commit()
    assert bal == 10
    assert session.get(User, u.id).balance == 10


def test_credit_is_idempotent(session):
    u = _user(session, balance=0)
    ledger.credit(session, u.id, 10, kind="topup", idempotency_key="K")
    session.commit()
    ledger.credit(session, u.id, 10, kind="topup", idempotency_key="K")
    session.commit()
    assert session.get(User, u.id).balance == 10        # applied once
    assert session.query(Transaction).count() == 1


def test_refund_restores_balance(session):
    u = _user(session, balance=5)
    ledger.debit(session, u.id, 1, ref="s")
    session.commit()
    ledger.refund(session, u.id, 1, ref="s")
    session.commit()
    assert session.get(User, u.id).balance == 5
    kinds = [t.kind for t in session.query(Transaction).order_by(Transaction.id).all()]
    assert kinds == ["debit", "refund"]


def test_fractional_debit_exact(session):
    """Sub-credit costs (0.1 for a code/paper read) debit exactly — the ledger holds
    Decimal, not truncated integers."""
    from decimal import Decimal
    u = _user(session, balance=1)
    bal = ledger.debit(session, u.id, Decimal("0.1"), ref="code-search")
    session.commit()
    assert Decimal(str(bal)) == Decimal("0.9")
    txn = session.query(Transaction).one()
    assert Decimal(str(txn.amount)) == Decimal("-0.1")
    assert Decimal(str(txn.balance_after)) == Decimal("0.9")


def test_fractional_debit_insufficient(session):
    from decimal import Decimal
    u = _user(session, balance=0)  # 0 < 0.1
    with pytest.raises(ledger.InsufficientCredits):
        ledger.debit(session, u.id, Decimal("0.1"))
    session.rollback()
    assert session.query(Transaction).count() == 0


def test_ten_fractional_debits_sum_to_one(session):
    """Ten 0.1 debits from 1.00 leave exactly 0 (no float drift past 2dp)."""
    from decimal import Decimal
    u = _user(session, balance=1)
    for _ in range(10):
        ledger.debit(session, u.id, Decimal("0.1"))
        session.commit()
    assert Decimal(str(session.get(User, u.id).balance)).quantize(Decimal("0.01")) == Decimal("0.00")


@pytest.mark.parametrize("start", [2, 3, 5, 10])
def test_many_fractional_debits_no_false_insufficient(session, start):
    """N×0.1 debits from an integer balance must ALL succeed and land exactly on 0 —
    without per-write rounding, accumulated float error trips a false InsufficientCredits
    on the last debit (reproduced at balance=2 → fails on the 20th 0.1 debit)."""
    from decimal import Decimal
    u = _user(session, balance=start)
    for _ in range(start * 10):
        ledger.debit(session, u.id, Decimal("0.1"))
        session.commit()
    assert Decimal(str(session.get(User, u.id).balance)).quantize(Decimal("0.01")) == Decimal("0.00")
