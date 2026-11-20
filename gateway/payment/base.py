"""PaymentProvider Strategy interface + result types.

A provider only talks to the payment gateway; it never touches the ledger or the
DB. The `service` module orchestrates persistence + crediting. Money is `Decimal`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class CreatedOrder:
    out_trade_no: str
    pay_url: str                       # where to send the user to pay
    params: dict = field(default_factory=dict)  # signed params (for forms/debug)


@dataclass
class NotifyResult:
    valid: bool                        # signature verified
    paid: bool                         # trade_status indicates success
    out_trade_no: str | None = None
    amount: Decimal | None = None
    raw: dict = field(default_factory=dict)


class PaymentProvider(ABC):
    name: str = "base"

    @abstractmethod
    def create_order(self, *, out_trade_no: str, amount: Decimal, subject: str,
                     notify_url: str, return_url: str | None = None) -> CreatedOrder:
        """Create a gateway order; return where to send the user + signed params."""

    @abstractmethod
    def verify_notify(self, params: dict) -> NotifyResult:
        """Verify an async notify callback's signature + payment status."""

    @abstractmethod
    def query_order(self, out_trade_no: str) -> bool:
        """Re-query the gateway to confirm payment (defense against forged notify).
        Returns True if the gateway reports the order paid."""
