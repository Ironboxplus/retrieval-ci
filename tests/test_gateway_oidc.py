"""Tests for OIDC id_token verification.

The Worker signs RS256 id_tokens; we verify sig + iss + aud + exp (+ optional nonce).
These tests generate a local RSA keypair and exercise `decode_id_token` (the pure
core) offline — no network, no JWKS fetch.
"""
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from gateway.auth import oidc

ISS = "https://116.ironbox.workers.dev"
AUD = "retrival-web"


@pytest.fixture(scope="module")
def keypair():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv_pem = priv.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    pub = priv.public_key()
    return priv_pem, pub


@pytest.fixture(scope="module")
def other_pubkey():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048).public_key()


def _token(priv_pem, **overrides):
    now = datetime.now(timezone.utc)
    payload = {
        "iss": ISS,
        "aud": AUD,
        "sub": "cdk:abc123",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=1)).timestamp()),
    }
    payload.update(overrides)
    return jwt.encode(payload, priv_pem, algorithm="RS256", headers={"kid": "k1"})


def test_valid_token_yields_claims(keypair):
    priv, pub = keypair
    tok = _token(priv, is_admin=True, roles=["a"], groups=["g"])
    claims = oidc.decode_id_token(tok, pub, issuer=ISS, audience=AUD)
    assert claims.sub == "cdk:abc123"
    assert claims.is_admin is True
    assert claims.roles == ["a"] and claims.groups == ["g"]
    assert claims.raw["aud"] == AUD


def test_is_admin_defaults_false(keypair):
    priv, pub = keypair
    claims = oidc.decode_id_token(_token(priv), pub, issuer=ISS, audience=AUD)
    assert claims.is_admin is False
    assert claims.roles == [] and claims.groups == []


def test_wrong_audience_rejected(keypair):
    priv, pub = keypair
    tok = _token(priv, aud="someone-else")
    with pytest.raises(oidc.OIDCError):
        oidc.decode_id_token(tok, pub, issuer=ISS, audience=AUD)


def test_wrong_issuer_rejected(keypair):
    priv, pub = keypair
    tok = _token(priv, iss="https://evil.example")
    with pytest.raises(oidc.OIDCError):
        oidc.decode_id_token(tok, pub, issuer=ISS, audience=AUD)


def test_expired_rejected(keypair):
    priv, pub = keypair
    past = datetime.now(timezone.utc) - timedelta(hours=2)
    tok = _token(priv, exp=int(past.timestamp()))
    with pytest.raises(oidc.OIDCError):
        oidc.decode_id_token(tok, pub, issuer=ISS, audience=AUD)


def test_bad_signature_rejected(keypair, other_pubkey):
    priv, _ = keypair
    tok = _token(priv)
    with pytest.raises(oidc.OIDCError):
        oidc.decode_id_token(tok, other_pubkey, issuer=ISS, audience=AUD)


def test_nonce_must_match_when_required(keypair):
    priv, pub = keypair
    tok = _token(priv, nonce="N1")
    assert oidc.decode_id_token(tok, pub, issuer=ISS, audience=AUD, nonce="N1").sub
    with pytest.raises(oidc.OIDCError):
        oidc.decode_id_token(tok, pub, issuer=ISS, audience=AUD, nonce="N2")


def test_jwks_client_uses_non_urllib_user_agent():
    """Cloudflare 403s the default urllib User-Agent on the Worker /keys edge (W120),
    so PyJWKClient must carry an explicit non-urllib UA or the JWKS fetch fails and
    every id_token verification 401s. Regression guard."""
    client = oidc.make_jwks_client("https://116.ironbox.workers.dev/keys")
    ua = client.headers.get("User-Agent", "")
    assert ua and "urllib" not in ua.lower()
