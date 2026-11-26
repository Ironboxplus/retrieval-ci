"""MockProvider — first-phase stand-in for a real payment gateway.

Signs/verifies with the same epay MD5 scheme as the real provider, so the topup →
notify → credit flow is exercised end-to-end without touching linux.do. `query_order`
trusts any known order as paid (there's no real gateway to re-query).
"""
from __future__ import annotations

from decimal import Decimal
from urllib.parse import urlencode

from gateway.payment import sign
from gateway.payment.base import CreatedOrder, NotifyResult, PaymentProvider

_SUCCESS = "TRADE_SUCCESS"


class MockProvider(PaymentProvider):
    name = "mock"

    def __init__(self, secret: str, pay_base: str = "https://mock-pay.local/pay"):
        self.secret = secret
        self.pay_base = pay_base

    def create_order(self, *, out_trade_no, amount, subject, notify_url, return_url=None):
        params = {
            "out_trade_no": out_trade_no,
            "money": f"{Decimal(amount):.2f}",
            "name": subject,
            "notify_url": notify_url,
        }
        if return_url:
            params["return_url"] = return_url
        params["sign"] = sign.md5_sign(params, self.secret)
        return CreatedOrder(
            out_trade_no=out_trade_no,
            pay_url=f"{self.pay_base}?{urlencode(params)}",
            params=params,
        )

    def verify_notify(self, params: dict) -> NotifyResult:
        if not sign.verify_md5(params, self.secret):
            return NotifyResult(valid=False, paid=False, raw=params)
        amount = params.get("money")
        return NotifyResult(
            valid=True,
            paid=params.get("trade_status") == _SUCCESS,
            out_trade_no=params.get("out_trade_no"),
            amount=Decimal(amount) if amount else None,
            raw=params,
        )

    def query_order(self, out_trade_no: str) -> bool:
        return True  # no real gateway to re-query; trust known orders
