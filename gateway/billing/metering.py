"""Request metering: charge a paying user up-front, refund if the work fails.

Used as a context manager around the actual compute call (proxy to the 4090):

    with metering.charge(db, user, cost=1, ref=req_id, idempotency_key=req_id):
        result = call_compute(...)   # raises on 5xx

Semantics:
  * Billing-exempt tiers (internal/admin) → no-op.
  * Paying user → atomic debit BEFORE the body (so an over-budget caller is
    rejected before any compute). The debit is committed immediately so it is
    durable and concurrency-safe; if the body raises, we refund + commit.
  * Insufficient balance → `ledger.InsufficientCredits` propagates (the route maps
    it to HTTP 402); the body never runs.

`idempotency_key` makes a client retry safe: the same key won't double-charge, and
the matching refund key won't double-refund.
"""
from __future__ import annotations

import logging
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal

from gateway.auth.deps import is_billable
from gateway.billing import ledger

logger = logging.getLogger("gateway.billing.metering")


@dataclass(frozen=True)
class ChargeReceipt:
    """Whether this invocation created a new debit transaction."""

    charged: bool


def api_idempotency_key(user_id: int, client_key: str | None) -> str | None:
    """Namespace an untrusted client key away from payments and other users."""
    if not client_key:
        return None
    digest = hashlib.sha256(client_key.encode("utf-8")).hexdigest()
    return f"api:{user_id}:{digest}"


def request_ref(method: str, path: str, body: bytes | None = None) -> str:
    """Build the semantic identity checked when an API idempotency key is reused."""
    route = f"{method.upper()} {path}"
    if len(route) > 88:
        route_digest = hashlib.sha256(route.encode("utf-8")).hexdigest()[:16]
        route = f"{route[:68]}...#{route_digest}"
    if body is None:
        return route
    body_digest = hashlib.sha256(body).hexdigest()[:24]
    return f"{route} body={body_digest}"


@contextmanager
def charge(db, user, *, cost: Decimal | int | float = 1, ref: str | None = None,
           idempotency_key: str | None = None):
    billable = is_billable(user) and Decimal(str(cost)) > 0
    charged_this_attempt = False
    reservation = ledger.DebitReservation(None, None, 1, None, False, True)
    if billable:
        try:
            reservation = ledger.reserve_debit_attempt(
                db, user.id, cost, ref, idempotency_key)
            if not reservation.already_charged:
                # raises InsufficientCredits (→ 402) before the body runs
                ledger.debit(
                    db,
                    user.id,
                    cost,
                    ref=ref,
                    idempotency_key=reservation.attempt_key,
                )
                db.commit()
                charged_this_attempt = True
        except Exception:
            db.rollback()
            raise
    try:
        yield ChargeReceipt(charged=charged_this_attempt)
    except Exception:
        if charged_this_attempt:
            refund_key = (
                ledger.related_idempotency_key(reservation.attempt_key, "refund")
                if reservation.attempt_key else None
            )
            ledger.refund(db, user.id, cost, ref=ref, idempotency_key=refund_key)
            ledger.finish_debit_attempt(db, reservation, succeeded=False)
            db.commit()
            logger.info("metering refund user=%s cost=%s ref=%s", user.id, cost, ref)
        raise
    else:
        if charged_this_attempt:
            ledger.finish_debit_attempt(db, reservation, succeeded=True)
            db.commit()
