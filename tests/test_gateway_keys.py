"""Tests for sk-key generation / hashing (pure, no DB)."""
from gateway.auth.keys import SK_PREFIX, generate_key, hash_key, key_display, verify_key


def test_generate_has_prefix_and_is_unique():
    a, b = generate_key(), generate_key()
    assert a.startswith(SK_PREFIX) and b.startswith(SK_PREFIX)
    assert a != b


def test_hash_roundtrip_and_pepper_sensitivity():
    k = generate_key()
    h = hash_key(k, "pepper-1")
    assert verify_key(k, h, "pepper-1")
    assert not verify_key(k + "x", h, "pepper-1")   # wrong key
    assert not verify_key(k, h, "pepper-2")         # wrong pepper


def test_hash_is_not_plaintext():
    k = generate_key()
    h = hash_key(k, "p")
    assert k not in h
    assert len(h) == 64  # sha256 hex


def test_display_prefix_and_last4():
    pfx, last4 = key_display("sk-AbCdEf1234WXYZ")
    assert pfx == "sk-AbC"
    assert last4 == "WXYZ"
