"""Topup orchestration: create an order, and turn a verified notify into an
idempotent credit.

`handle_notify` hardening (mirrors what the real epay/ldcpay flow needs):
  1. verify signature (provider.verify_notify),
  2. look up our order by out_trade_no,
  3. re-query the gateway (provider.query_order) — defense against a forged-but-
     somehow-valid notify,
  4. credit via the ledger keyed on out_trade_no (so a duplicate callback can't
     double-credit), mark the order paid,
  5. return exactly "success" so the gateway stops retrying; "fail" otherwise.
"""
from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone
from decimal import Decimal

from gateway.billing import ledger
from gateway.db.models import Order
from gateway.payment.base import CreatedOrder, PaymentProvider

logger = logging.getLogger("gateway.payment.service")

SUCCESS = "success"
FAIL = "fail"


def _new_out_trade_no() -> str:
    return secrets.token_hex(16)


def start_topup(db, user, *, credits: int, price: Decimal, provider: PaymentProvider,
                notify_url: str, return_url: str | None = None,
                subject: str | None = None) -> CreatedOrder:
    """Persist a pending order and ask the provider to create the gateway order."""
    out_trade_no = _new_out_trade_no()
    db.add(Order(out_trade_no=out_trade_no, user_id=user.id, credits=credits,
                 price=f"{Decimal(price):.2f}", provider=provider.name, status="pending"))
    db.flush()
    created = provider.create_order(
        out_trade_no=out_trade_no, amount=Decimal(price),
        subject=subject or f"{credits} credits", notify_url=notify_url, return_url=return_url)
    logger.info("topup started user=%s out_trade_no=%s credits=%s price=%s",
                user.id, out_trade_no, credits, price)
    return created


def handle_notify(db, provider: PaymentProvider, params: dict) -> str:
    """Process an async payment notify. Idempotent; returns "success"/"fail"."""
    res = provider.verify_notify(params)
    if not res.valid or not res.paid or not res.out_trade_no:
        logger.warning("notify rejected valid=%s paid=%s otn=%s",
                       res.valid, res.paid, res.out_trade_no)
        return FAIL

    order = db.query(Order).filter(Order.out_trade_no == res.out_trade_no).first()
    if order is None:
        logger.warning("notify for unknown order otn=%s", res.out_trade_no)
        return FAIL

    # Paid amount must match what we ordered — defends against a tampered/replayed
    # notify carrying a different money (linux.do 铁律: money 要和下单核对).
    if res.amount is not None and Decimal(str(order.price)) != res.amount:
        logger.warning("notify amount mismatch otn=%s notify=%s order=%s",
                       res.out_trade_no, res.amount, order.price)
        return FAIL

    if order.status == "paid":
        return SUCCESS  # duplicate callback — already credited

    # Re-query the gateway to confirm before granting anything.
    if not provider.query_order(order.out_trade_no):
        logger.warning("notify re-query says unpaid otn=%s", order.out_trade_no)
        return FAIL

    # Ledger credit keyed on out_trade_no => duplicate callbacks can't double-credit.
    ledger.credit(db, order.user_id, order.credits, kind="topup",
                  ref=order.out_trade_no, idempotency_key=order.out_trade_no)
    order.status = "paid"
    order.paid_at = datetime.now(timezone.utc)
    db.flush()
    logger.info("topup credited user=%s otn=%s credits=%s",
                order.user_id, order.out_trade_no, order.credits)
    return SUCCESS
