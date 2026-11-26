"""Tests for the real linux.do Credit (彩虹易支付-compatible) EpayProvider.

Locks: page-jump submit URL + epay-MD5 signing (pid/type/sign_type), notify
verification, and api.php re-query parsing. Network (query_order) is monkeypatched
— no live call to credit.linux.do here.
"""
from decimal import Decimal
from urllib.parse import parse_qs, urlparse

import pytest

from gateway.payment import epay, sign
from gateway.payment.base import PaymentProvider

PID = "test-pid-0000"     # epay Client ID (fully fake test stand-in — never real creds)
KEY = "test-key-0000"     # epay Client Secret (fake; real key lives only in BWH env)


@pytest.fixture
def provider():
    return epay.EpayProvider(pid=PID, key=KEY, gateway_base="https://credit.linux.do/epay")


def test_is_a_payment_provider(provider):
    assert isinstance(provider, PaymentProvider)
    assert provider.name == "epay"


def test_create_order_builds_signed_submit_url(provider):
    created = provider.create_order(
        out_trade_no="T1", amount=Decimal("5"), subject="100 credits",
        notify_url="https://retrieval.rnarket.com/api/payment/notify",
        return_url="https://retrieval.rnarket.com/?pay=done")
    # page-jump to /epay/pay/submit.php
    assert created.pay_url.startswith("https://credit.linux.do/epay/pay/submit.php?")
    assert created.out_trade_no == "T1"
    # required epay fields present + correctly signed
    p = created.params
    assert p["pid"] == PID and p["type"] == "epay" and p["sign_type"] == "MD5"
    assert p["money"] == "5.00"                       # 2-dp money
    assert p["out_trade_no"] == "T1"
    assert sign.verify_md5(p, KEY)
    # the query string carries the same signed params
    q = parse_qs(urlparse(created.pay_url).query)
    assert q["sign"][0] == p["sign"]
    assert q["pid"][0] == PID


def test_create_order_without_return_url(provider):
    created = provider.create_order(
        out_trade_no="T2", amount=Decimal("20"), subject="500 credits",
        notify_url="https://wb/notify")
    assert "return_url" not in created.params
    assert sign.verify_md5(created.params, KEY)


def test_verify_notify_valid_paid(provider):
    good = {"pid": PID, "out_trade_no": "T1", "money": "5.00", "trade_status": "TRADE_SUCCESS"}
    good["sign"] = sign.md5_sign(good, KEY)
    res = provider.verify_notify(good)
    assert res.valid and res.paid
    assert res.out_trade_no == "T1" and res.amount == Decimal("5.00")


def test_verify_notify_bad_sign_and_unpaid(provider):
    forged = {"out_trade_no": "T1", "money": "5.00", "trade_status": "TRADE_SUCCESS", "sign": "deadbeef"}
    assert provider.verify_notify(forged).valid is False

    unpaid = {"out_trade_no": "T3", "money": "5.00", "trade_status": "WAIT_BUYER_PAY"}
    unpaid["sign"] = sign.md5_sign(unpaid, KEY)
    r = provider.verify_notify(unpaid)
    assert r.valid is True and r.paid is False


def test_query_order_parses_status(provider, monkeypatch):
    calls = {}

    class _Resp:
        def __init__(self, payload): self._p = payload
        def json(self): return self._p

    def fake_get(url, params=None, timeout=None):
        calls["url"] = url
        calls["params"] = params
        return _Resp({"code": 1, "status": 1, "out_trade_no": params["out_trade_no"]})

    monkeypatch.setattr(epay.httpx, "get", fake_get)
    assert provider.query_order("T1") is True
    assert calls["url"] == "https://credit.linux.do/epay/api.php"
    assert calls["params"]["pid"] == PID and calls["params"]["out_trade_no"] == "T1"


def test_query_order_unpaid_and_network_error_are_false(provider, monkeypatch):
    class _Resp:
        def __init__(self, payload): self._p = payload
        def json(self): return self._p

    monkeypatch.setattr(epay.httpx, "get", lambda *a, **k: _Resp({"code": 1, "status": 0}))
    assert provider.query_order("T1") is False  # not paid

    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(epay.httpx, "get", boom)
    assert provider.query_order("T1") is False  # fail-safe: no credit on query failure
