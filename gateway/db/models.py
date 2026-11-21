"""Gateway (app-tier) data models — users, API keys, credit ledger.

This is the SEPARATE app-tier DB (runs on the public web backend / BWH), NOT the
4090 compute DB. It owns all sensitive state: user identities (from external OIDC),
sk-key hashes, balances, and the transaction ledger. The 4090 never sees this.
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (Column, DateTime, Enum, ForeignKey, Integer, Numeric,
                        String, UniqueConstraint)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Tier(str, enum.Enum):
    paying = "paying"      # metered: 1 credit per search, must top up
    internal = "internal"  # invited, billing-exempt
    admin = "admin"        # full access, billing-exempt, management


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    # Primary identity tag, format "{provider}:{sub}" (e.g. "cdk:admin_arc",
    # "linuxdo:12345", "google:108..."). Kept for backward-compat; ALL login
    # resolution goes through `user_identities` so one account can hold several.
    oidc_sub = Column(String(191), unique=True, nullable=False, index=True)
    username = Column(String(64), unique=True, nullable=False)
    tier = Column(Enum(Tier), default=Tier.paying, nullable=False)
    # Credits as fixed-point Decimal (2dp) so sub-credit costs work (code/paper read
    # = 0.1). SQLite has no native decimal → stored as REAL; the 2dp scale keeps the
    # tiny float error far below display precision. Topups grant whole credits.
    balance = Column(Numeric(12, 2), default=0, nullable=False)
    created_at = Column(DateTime, default=_utcnow, nullable=False)

    api_keys = relationship("ApiKey", back_populates="user")
    identities = relationship("UserIdentity", back_populates="user",
                              cascade="all, delete-orphan")


class UserIdentity(Base):
    """A single external login bound to a User. Account binding = several rows
    (different providers) pointing at the SAME user_id → one shared balance.
    `sub` is the provider-unique id: cf-oidc CDK subject, linux.do numeric id,
    or Google `sub`. (provider, sub) is globally unique."""
    __tablename__ = "user_identities"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    provider = Column(String(32), nullable=False)   # cdk | linuxdo | google
    sub = Column(String(191), nullable=False)
    created_at = Column(DateTime, default=_utcnow, nullable=False)

    user = relationship("User", back_populates="identities")

    __table_args__ = (UniqueConstraint("provider", "sub", name="uq_identity_provider_sub"),)


class ApiKey(Base):
    __tablename__ = "api_keys"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    key_hash = Column(String(64), unique=True, nullable=False, index=True)  # HMAC-SHA256 hex
    prefix = Column(String(12), nullable=False)   # display: leading chars
    last4 = Column(String(4), nullable=False)     # display: trailing 4
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="api_keys")


class Transaction(Base):
    """Append-only credit ledger. `amount` is a SIGNED delta (debit<0, topup/refund>0);
    sum(amount) per user == users.balance (balance is a cache of this ledger)."""
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    kind = Column(String(16), nullable=False)        # topup | debit | refund | adjust
    amount = Column(Numeric(12, 2), nullable=False)  # signed delta (Decimal, 2dp)
    balance_after = Column(Numeric(12, 2), nullable=False)
    ref = Column(String(128), nullable=True)         # e.g. out_trade_no / request id
    idempotency_key = Column(String(128), unique=True, nullable=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)


class IdempotencyRequest(Base):
    """Persistent ownership/state for one retry-safe metered API request."""

    __tablename__ = "idempotency_requests"
    idempotency_key = Column(String(128), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    amount = Column(Numeric(12, 2), nullable=False)
    ref = Column(String(128), nullable=False)
    status = Column(String(16), nullable=False)  # pending | succeeded | failed
    attempt_number = Column(Integer, nullable=False, default=1)
    lease_token = Column(String(32), nullable=False)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, nullable=False)


class Order(Base):
    """A top-up order. `out_trade_no` is our id sent to the payment gateway and is
    the idempotency anchor for crediting (a duplicate notify can't double-credit).
    `credits` is granted on payment; `price` is the fiat/LDC amount (Decimal as str)."""
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True)
    out_trade_no = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    credits = Column(Integer, nullable=False)        # credits to grant on success
    price = Column(String(32), nullable=False)       # fiat/LDC amount, Decimal as str
    provider = Column(String(32), nullable=False)    # e.g. "mock" | "epay" | "ldcpay"
    status = Column(String(16), nullable=False, default="pending")  # pending|paid|failed
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    paid_at = Column(DateTime, nullable=True)
