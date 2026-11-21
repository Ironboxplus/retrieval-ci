"""Payment layer (web backend / BWH only).

A `PaymentProvider` Strategy abstracts the gateway; `MockProvider` is the
first-phase stand-in (payment is mocked). The `service` module turns a verified
notify into an idempotent credit via the ledger. Real providers (linux.do epay /
ldcpay) slot in behind the same interface — see `sign.py` for the epay MD5 scheme.
"""
