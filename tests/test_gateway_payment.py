"""Tests for the payment layer — epay-style signing, MockProvider, and the
topup service (create order → notify → idempotent credit). Payment is MOCKED
this phase; these lock the Strategy contract + the notify/credit idempotency."""
from decimal import Decimal

import pytest

from gateway.auth import accounts
from gateway.db.models import Order, Tier, Transaction, User
from gateway.db.session import make_engine, make_session_factory
from gateway.payment import sign
from gateway.payment.mock import MockProvider
from gateway.payment.service import handle_notify, start_topup

SECRET = "mock-secret"


@pytest.fixture
def session():
    engine = make_engine("sqlite://")
    s = make_session_factory(engine)()
    try:
        yield s
    finally:
        s.close()


def _user(s, balance=0):
    u = accounts.register(s, oidc_sub="cdk:a", username="alice")
    u.balance = balance
    s.commit()
    return u


# --- sign helper (epay MD5) --------------------------------------------------

def test_sign_string_sorts_and_skips_sign_and_empty():
    params = {"money": "1.00", "name": "credits", "sign": "x", "sign_type": "MD5", "blank": ""}
    s = sign.build_sign_string(params)
    assert s == "money=1.00&name=credits"      # sorted ASCII, sign/sign_type/empty dropped


def test_md5_sign_roundtrip_and_tamper_detection():
    params = {"out_trade_no": "T1", "money": "5.00"}
    signed = dict(params, sign=sign.md5_sign(params, SECRET))
    assert sign.verify_md5(signed, SECRET) is True
    signed["money"] = "9999.00"                 # tampered
    assert sign.verify_md5(signed, SECRET) is False
    assert sign.verify_md5(dict(params), SECRET) is False  # missing sign


# --- MockProvider ------------------------------------------------------------

def test_mock_create_order_returns_pay_url_and_signed_params():
    p = MockProvider(secret=SECRET)
    created = p.create_order(out_trade_no="T1", amount=Decimal("5.00"),
                             subject="100 credits", notify_url="https://wb/notify")
    assert created.out_trade_no == "T1"
    assert created.pay_url
    assert sign.verify_md5(created.params, SECRET)


def test_mock_verify_notify_valid_and_invalid():
    p = MockProvider(secret=SECRET)
    good = {"out_trade_no": "T1", "money": "5.00", "trade_status": "TRADE_SUCCESS"}
    good["sign"] = sign.md5_sign(good, SECRET)
    res = p.verify_notify(good)
    assert res.valid and res.paid and res.out_trade_no == "T1"

    bad = dict(good, sign="deadbeef")
    assert p.verify_notify(bad).valid is False

    unpaid = {"out_trade_no": "T2", "money": "5.00", "trade_status": "WAIT_PAY"}
    unpaid["sign"] = sign.md5_sign(unpaid, SECRET)
    r2 = p.verify_notify(unpaid)
    assert r2.valid is True and r2.paid is False


# --- topup service: create → notify → credit --------------------------------

def _notify_params(provider, out_trade_no, money="5.00"):
    params = {"out_trade_no": out_trade_no, "money": money, "trade_status": "TRADE_SUCCESS"}
    params["sign"] = sign.md5_sign(params, SECRET)
    return params


def test_start_topup_creates_pending_order(session):
    u = _user(session)
    p = MockProvider(secret=SECRET)
    created = start_topup(session, u, credits=100, price=Decimal("5.00"),
                          provider=p, notify_url="https://wb/notify")
    session.commit()
    order = session.query(Order).filter(Order.out_trade_no == created.out_trade_no).first()
    assert order is not None and order.status == "pending" and order.credits == 100


def test_handle_notify_credits_user_and_marks_paid(session):
    u = _user(session, balance=0)
    p = MockProvider(secret=SECRET)
    created = start_topup(session, u, credits=100, price=Decimal("5.00"),
                          provider=p, notify_url="https://wb/notify")
    session.commit()

    result = handle_notify(session, p, _notify_params(p, created.out_trade_no))
    session.commit()
    assert result == "success"
    assert session.get(User, u.id).balance == 100
    order = session.query(Order).filter(Order.out_trade_no == created.out_trade_no).first()
    assert order.status == "paid"


def test_handle_notify_is_idempotent(session):
    u = _user(session, balance=0)
    p = MockProvider(secret=SECRET)
    created = start_topup(session, u, credits=100, price=Decimal("5.00"),
                          provider=p, notify_url="https://wb/notify")
    session.commit()
    params = _notify_params(p, created.out_trade_no)

    assert handle_notify(session, p, params) == "success"
    session.commit()
    assert handle_notify(session, p, params) == "success"   # duplicate callback
    session.commit()
    assert session.get(User, u.id).balance == 100           # credited once
    assert session.query(Transaction).filter(Transaction.kind == "topup").count() == 1


def test_handle_notify_bad_signature_rejected(session):
    u = _user(session, balance=0)
    p = MockProvider(secret=SECRET)
    created = start_topup(session, u, credits=100, price=Decimal("5.00"),
                          provider=p, notify_url="https://wb/notify")
    session.commit()
    params = _notify_params(p, created.out_trade_no)
    params["sign"] = "forged"

    assert handle_notify(session, p, params) == "fail"
    session.commit()
    assert session.get(User, u.id).balance == 0


def test_handle_notify_unknown_order_rejected(session):
    p = MockProvider(secret=SECRET)
    assert handle_notify(session, p, _notify_params(p, "does-not-exist")) == "fail"


def test_handle_notify_amount_mismatch_rejected(session):
    """A validly-signed notify whose money != the order's price must NOT credit
    (defense against tampered/replayed amount — linux.do 铁律: money 要核对)."""
    u = _user(session, balance=0)
    p = MockProvider(secret=SECRET)
    created = start_topup(session, u, credits=100, price=Decimal("5.00"),
                          provider=p, notify_url="https://wb/notify")
    session.commit()
    # validly-signed but for a different amount than the order (5.00)
    tampered = _notify_params(p, created.out_trade_no, money="0.01")
    assert handle_notify(session, p, tampered) == "fail"
    session.commit()
    assert session.get(User, u.id).balance == 0
