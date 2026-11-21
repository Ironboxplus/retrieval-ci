"""Credit ledger: atomic debit / credit / refund / adjust.

`users.balance` is a cache; `transactions` is the append-only source of truth. The
debit uses a SINGLE conditional UPDATE (`... WHERE balance >= cost`) so concurrent
requests can't double-spend (no read-modify-write window). `idempotency_key` makes
client retries / duplicate payment callbacks safe.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import text

from gateway.db.models import IdempotencyRequest, Transaction, User

logger = logging.getLogger("gateway.billing.ledger")

# Amounts may be sub-credit Decimals (e.g. 0.1). sqlite3 can't bind a Decimal to raw
# SQL, so we bind the numeric params as float. 0.1 has no exact float representation,
# so every write ROUNDs the balance back to 2dp — otherwise repeated 0.1 debits drift
# (e.g. 2.0 after 20×0.1 lands at 0.0999999… < 0.1 and falsely trips InsufficientCredits).
# The comparison also rounds so a drifted stored value can't wrongly block a valid spend.
# The ledger records keep exact Decimal.
Number = Decimal | int | float
PENDING_REQUEST_LEASE = timedelta(minutes=15)


def _f(x: Number) -> float:
    return float(x)


class InsufficientCredits(Exception):
    def __init__(self, user_id: int, cost: Number):
        super().__init__(f"user {user_id} lacks {cost} credit(s)")
        self.user_id = user_id
        self.cost = cost


class IdempotencyConflict(Exception):
    """A client reused one key for a different user, operation, or request."""

    def __init__(self, idempotency_key: str):
        super().__init__(f"idempotency key conflicts with an earlier transaction: {idempotency_key}")
        self.idempotency_key = idempotency_key


class IdempotencyInProgress(Exception):
    def __init__(self, idempotency_key: str):
        super().__init__(f"idempotent request is already in progress: {idempotency_key}")
        self.idempotency_key = idempotency_key


@dataclass(frozen=True)
class DebitReservation:
    logical_key: str | None
    attempt_key: str | None
    attempt_number: int
    lease_token: str | None
    already_charged: bool
    owns_attempt: bool


def _existing(session, idempotency_key):
    if not idempotency_key:
        return None
    return (session.query(Transaction)
            .filter(Transaction.idempotency_key == idempotency_key).first())


def _amount(value: Number) -> Decimal:
    return Decimal(str(value))


def _assert_existing_matches(
    transaction: Transaction,
    *,
    user_id: int,
    kind: str,
    amount: Decimal,
    ref: str | None,
) -> None:
    if (
        transaction.user_id != user_id
        or transaction.kind != kind
        or Decimal(transaction.amount) != amount
        or transaction.ref != ref
    ):
        raise IdempotencyConflict(str(transaction.idempotency_key))


def related_idempotency_key(idempotency_key: str, suffix: str) -> str:
    """Derive a bounded key for a retry/refund transaction."""
    candidate = f"{idempotency_key}:{suffix}"
    if len(candidate) <= 128:
        return candidate
    digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
    return f"related:{digest}:{suffix}"[:128]


def _assert_request_matches(
    request: IdempotencyRequest,
    *,
    user_id: int,
    cost: Number,
    ref: str | None,
) -> None:
    if (
        request.user_id != user_id
        or Decimal(request.amount) != _amount(cost)
        or request.ref != ref
    ):
        raise IdempotencyConflict(request.idempotency_key)


def _attempt_idempotency_key(logical_key: str, attempt_number: int) -> str:
    if attempt_number == 1:
        return logical_key
    return related_idempotency_key(logical_key, f"attempt-{attempt_number}")


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _new_lease_token() -> str:
    return uuid.uuid4().hex


def reserve_debit_attempt(
    session,
    user_id: int,
    cost: Number,
    ref: str | None,
    idempotency_key: str | None,
) -> DebitReservation:
    """Atomically reserve one logical request attempt across threads/processes."""
    if not idempotency_key:
        return DebitReservation(None, None, 1, None, False, True)
    if ref is None:
        raise ValueError("idempotent debit reservations require a request ref")

    while True:
        now = datetime.now(timezone.utc)
        lease_token = _new_lease_token()
        inserted = session.execute(
            text(
                "INSERT OR IGNORE INTO idempotency_requests "
                "(idempotency_key, user_id, amount, ref, status, attempt_number, "
                "lease_token, created_at, updated_at) "
                "VALUES (:key, :user_id, :amount, :ref, 'pending', 1, "
                ":lease_token, :now, :now)"
            ),
            {
                "key": idempotency_key,
                "user_id": user_id,
                "amount": _f(cost),
                "ref": ref,
                "lease_token": lease_token,
                "now": now,
            },
        )
        if inserted.rowcount == 1:
            return DebitReservation(
                idempotency_key, idempotency_key, 1, lease_token, False, True)

        session.expire_all()
        request = session.get(IdempotencyRequest, idempotency_key)
        if request is None:
            continue
        _assert_request_matches(
            request, user_id=user_id, cost=cost, ref=ref)
        if request.status == "pending":
            if _as_utc(request.updated_at) > now - PENDING_REQUEST_LEASE:
                raise IdempotencyInProgress(idempotency_key)

            next_attempt = request.attempt_number + 1
            claimed = session.execute(
                text(
                    "UPDATE idempotency_requests SET attempt_number=:next_attempt, "
                    "lease_token=:next_lease_token, updated_at=:now "
                    "WHERE idempotency_key=:key AND status='pending' "
                    "AND attempt_number=:previous_attempt "
                    "AND lease_token=:previous_lease_token "
                    "AND updated_at=:previous_updated_at"
                ),
                {
                    "key": idempotency_key,
                    "next_attempt": next_attempt,
                    "next_lease_token": lease_token,
                    "previous_attempt": request.attempt_number,
                    "previous_lease_token": request.lease_token,
                    "previous_updated_at": request.updated_at,
                    "now": now,
                },
            )
            if claimed.rowcount != 1:
                continue

            previous_attempt_key = _attempt_idempotency_key(
                idempotency_key, request.attempt_number)
            refund(
                session,
                user_id,
                cost,
                ref=ref,
                idempotency_key=related_idempotency_key(
                    previous_attempt_key, "refund"),
            )
            logger.warning(
                "recovered stale idempotency lease key=%s attempt=%s",
                idempotency_key,
                request.attempt_number,
            )
            return DebitReservation(
                idempotency_key,
                _attempt_idempotency_key(idempotency_key, next_attempt),
                next_attempt,
                lease_token,
                False,
                True,
            )
        if request.status == "succeeded":
            return DebitReservation(
                idempotency_key,
                _attempt_idempotency_key(
                    idempotency_key, request.attempt_number),
                request.attempt_number,
                None,
                True,
                False,
            )
        if request.status != "failed":
            raise ValueError(f"unknown idempotency request status: {request.status}")

        next_attempt = request.attempt_number + 1
        claimed = session.execute(
            text(
                "UPDATE idempotency_requests SET status='pending', "
                "attempt_number=:next_attempt, lease_token=:lease_token, "
                "updated_at=:now "
                "WHERE idempotency_key=:key AND status='failed' "
                "AND attempt_number=:previous_attempt"
            ),
            {
                "key": idempotency_key,
                "next_attempt": next_attempt,
                "previous_attempt": request.attempt_number,
                "lease_token": lease_token,
                "now": now,
            },
        )
        if claimed.rowcount == 1:
            return DebitReservation(
                idempotency_key,
                _attempt_idempotency_key(idempotency_key, next_attempt),
                next_attempt,
                lease_token,
                False,
                True,
            )


def finish_debit_attempt(
    session,
    reservation: DebitReservation,
    *,
    succeeded: bool,
) -> None:
    if not reservation.logical_key or not reservation.owns_attempt:
        return
    target_status = "succeeded" if succeeded else "failed"
    updated = session.execute(
        text(
            "UPDATE idempotency_requests SET status=:status, updated_at=:now "
            "WHERE idempotency_key=:key AND status='pending' "
            "AND attempt_number=:attempt_number AND lease_token=:lease_token"
        ),
        {
            "status": target_status,
            "now": datetime.now(timezone.utc),
            "key": reservation.logical_key,
            "attempt_number": reservation.attempt_number,
            "lease_token": reservation.lease_token,
        },
    )
    if updated.rowcount != 1:
        raise RuntimeError(
            f"lost idempotency reservation {reservation.logical_key} "
            f"attempt {reservation.attempt_number}"
        )


def _fresh_balance(session, user_id: int) -> Decimal:
    """Read the post-UPDATE balance. The raw SQL above bypassed the ORM identity
    map, so expire the cached object to force a reload."""
    u = session.get(User, user_id)
    if u is None:
        raise ValueError(f"no such user {user_id}")
    session.expire(u)
    return u.balance


def _record(session, user_id, kind, amount, balance_after, ref, idempotency_key):
    txn = Transaction(user_id=user_id, kind=kind, amount=amount,
                      balance_after=balance_after, ref=ref,
                      idempotency_key=idempotency_key)
    session.add(txn)
    session.flush()
    return txn


def debit(session, user_id: int, cost: Number = 1, ref: str | None = None,
          idempotency_key: str | None = None) -> Decimal:
    """Atomically remove `cost` credits; raise InsufficientCredits if short.
    Returns the new balance. Idempotent on idempotency_key."""
    dup = _existing(session, idempotency_key)
    if dup is not None:
        _assert_existing_matches(
            dup,
            user_id=user_id,
            kind="debit",
            amount=-_amount(cost),
            ref=ref,
        )
        logger.info("debit idempotent-hit key=%s user=%s", idempotency_key, user_id)
        return dup.balance_after
    res = session.execute(
        text("UPDATE users SET balance = ROUND(balance - :c, 2) "
             "WHERE id = :u AND ROUND(balance, 2) >= :c"),
        {"c": _f(cost), "u": user_id},
    )
    if res.rowcount == 0:
        logger.info("debit insufficient user=%s cost=%s", user_id, cost)
        raise InsufficientCredits(user_id, cost)
    bal = _fresh_balance(session, user_id)
    _record(session, user_id, "debit", -_amount(cost), bal, ref, idempotency_key)
    logger.info("debit user=%s cost=%s balance=%s", user_id, cost, bal)
    return bal


def credit(session, user_id: int, amount: Number, kind: str = "topup",
           ref: str | None = None, idempotency_key: str | None = None) -> Decimal:
    """Add `amount` credits (topup/refund/adjust; may be negative for adjust).
    Idempotent on idempotency_key. Returns the new balance."""
    dup = _existing(session, idempotency_key)
    if dup is not None:
        _assert_existing_matches(
            dup,
            user_id=user_id,
            kind=kind,
            amount=_amount(amount),
            ref=ref,
        )
        logger.info("credit idempotent-hit key=%s user=%s", idempotency_key, user_id)
        return dup.balance_after
    session.execute(
        text("UPDATE users SET balance = ROUND(balance + :a, 2) WHERE id = :u"),
        {"a": _f(amount), "u": user_id},
    )
    bal = _fresh_balance(session, user_id)
    _record(session, user_id, kind, _amount(amount), bal, ref, idempotency_key)
    logger.info("credit user=%s amount=%s kind=%s balance=%s", user_id, amount, kind, bal)
    return bal


def refund(session, user_id: int, amount: Number, ref: str | None = None,
           idempotency_key: str | None = None) -> Decimal:
    return credit(session, user_id, amount, kind="refund", ref=ref,
                  idempotency_key=idempotency_key)
