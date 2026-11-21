"""epay-style signing (linux.do Credit / 彩虹易支付 family).

Sign string = all params EXCEPT `sign`/`sign_type` and empty values, sorted by key
in ASCII order, joined as `k1=v1&k2=v2`, with the merchant secret appended RAW
(no separator), then hashed. MD5 here (epay); the ldcpay variant signs the same
string with Ed25519 — both reuse `build_sign_string`.
"""
from __future__ import annotations

import hashlib
import hmac

_SKIP_KEYS = {"sign", "sign_type"}


def build_sign_string(params: dict) -> str:
    """Canonical pre-sign string: sorted, non-empty, sign/sign_type excluded."""
    items = [(k, v) for k, v in params.items()
             if k not in _SKIP_KEYS and v not in (None, "")]
    items.sort(key=lambda kv: kv[0])
    return "&".join(f"{k}={v}" for k, v in items)


def md5_sign(params: dict, key: str) -> str:
    """MD5(sign_string + key) — the epay signature."""
    return hashlib.md5((build_sign_string(params) + key).encode("utf-8")).hexdigest()


def verify_md5(params: dict, key: str) -> bool:
    """Constant-time check of `params['sign']` against a recomputed MD5."""
    presented = params.get("sign")
    if not presented:
        return False
    return hmac.compare_digest(str(presented).lower(), md5_sign(params, key))
