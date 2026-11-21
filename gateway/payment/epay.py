"""EpayProvider — the real linux.do Credit payment gateway (彩虹易支付-compatible).

linux.do Credit moves community points (LDC), not fiat. Same epay-MD5 scheme as
[[reference_linuxdo_credit_epay]]: Client ID = `pid`, Client Secret = `key`, sign =
sorted non-empty params (excl. sign/sign_type) + key appended RAW, MD5 lowercase.

Flow:
  1. create_order → a signed GET URL to `/epay/pay/submit.php` (page-jump). The
     browser opens it (AccountModal does `window.open`) and credit.linux.do renders
     the cashier / 302s to its paying page. We do NOT POST server-side here.
  2. The user pays; LDC is deducted.
  3. credit.linux.do GETs our notify_url with `trade_status=TRADE_SUCCESS` + sign.
     verify_notify checks the signature; the service layer re-queries + credits
     idempotently and returns plain "success".
  4. return_url is browser-only and untrusted — never credit on it.
  5. query_order re-confirms via `/epay/api.php` (defense against a forged notify).
"""
from __future__ import annotations

import logging
from decimal import Decimal
from urllib.parse import urlencode

import httpx

from gateway.payment import sign
from gateway.payment.base import CreatedOrder, NotifyResult, PaymentProvider

logger = logging.getLogger("gateway.payment.epay")

_SUCCESS = "TRADE_SUCCESS"
_DEFAULT_BASE = "https://credit.linux.do/epay"


class EpayProvider(PaymentProvider):
    name = "epay"

    def __init__(self, *, pid: str, key: str, gateway_base: str = _DEFAULT_BASE,
                 pay_type: str = "epay", timeout: float = 15.0):
        self.pid = pid
        self.key = key
        self.base = gateway_base.rstrip("/")
        self.pay_type = pay_type
        self.timeout = timeout

    def create_order(self, *, out_trade_no, amount, subject, notify_url, return_url=None):
        params = {
            "pid": self.pid,
            "type": self.pay_type,
            "out_trade_no": out_trade_no,
            "notify_url": notify_url,
            "name": subject,
            "money": f"{Decimal(amount):.2f}",   # ≤2 dp
            "sign_type": "MD5",                   # sent but excluded from the sign string
        }
        if return_url:
            params["return_url"] = return_url
        params["sign"] = sign.md5_sign(params, self.key)
        pay_url = f"{self.base}/pay/submit.php?{urlencode(params)}"
        logger.info("epay order created out_trade_no=%s money=%s", out_trade_no, params["money"])
        return CreatedOrder(out_trade_no=out_trade_no, pay_url=pay_url, params=params)

    def verify_notify(self, params: dict) -> NotifyResult:
        if not sign.verify_md5(params, self.key):
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
        """Re-query /epay/api.php; True only if the gateway reports the order paid
        (status == 1). Any network/parse error is fail-safe False (no credit)."""
        q = {"act": "order", "pid": self.pid, "key": self.key, "out_trade_no": out_trade_no}
        try:
            r = httpx.get(f"{self.base}/api.php", params=q, timeout=self.timeout)
            data = r.json()
        except Exception as exc:  # network down / non-JSON → don't grant credit
            logger.warning("epay query_order failed out_trade_no=%s: %s", out_trade_no, exc)
            return False
        return str(data.get("status")) == "1"
