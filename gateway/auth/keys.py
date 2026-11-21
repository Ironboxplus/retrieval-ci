"""sk-key generation, hashing, verification (pure, no DB).

Standard API-key handling (OpenAI/Stripe style): generate with `secrets`, store a
FAST keyed hash (HMAC-SHA256 with a server-side pepper) in an indexed column for
O(1) lookup + constant-time compare. NEVER store plaintext; NEVER bcrypt per request
— bcrypt is for low-entropy passwords; sk-keys are already high-entropy random tokens,
so a keyed hash is both correct and fast enough to run on every request.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets

SK_PREFIX = "sk-"
_TOKEN_BYTES = 32


def generate_key() -> str:
    """A fresh `sk-...` key. Shown to the user ONCE; only its hash is stored."""
    return SK_PREFIX + secrets.token_urlsafe(_TOKEN_BYTES)


def hash_key(key: str, pepper: str) -> str:
    """HMAC-SHA256(pepper, key) hex — the value stored & looked up (indexed column)."""
    return hmac.new(pepper.encode("utf-8"), key.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_key(key: str, stored_hash: str, pepper: str) -> bool:
    return hmac.compare_digest(hash_key(key, pepper), stored_hash)


def key_display(key: str) -> tuple[str, str]:
    """(prefix, last4) for non-secret display, e.g. show 'sk-AbC…WXYZ'."""
    return key[:6], key[-4:]
