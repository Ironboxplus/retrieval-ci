"""Tests for the 4090 internal-access gate decision (core/internal_gate.py).

Key requirement: LAN/intranet clients stay UNAUTHENTICATED (frictionless internal
use); the public path (loopback peer via tunnel) must present the token."""
from core.internal_gate import gate_allows, is_private_lan

TOKEN = "internal-secret"
EXEMPT = frozenset({"/health"})


def _allows(host, token=None, method="POST", path="/api/search", trust_lan=True):
    return gate_allows(method=method, path=path, client_host=host, token_header=token,
                       expected_token=TOKEN, exempt=EXEMPT, trust_lan=trust_lan)


# --- is_private_lan ----------------------------------------------------------

def test_is_private_lan_classification():
    assert is_private_lan("10.100.100.5") is True       # lab overlay
    assert is_private_lan("192.168.1.20") is True
    assert is_private_lan("172.16.0.1") is True
    assert is_private_lan("127.0.0.1") is False         # loopback = tunnel/public, NOT LAN
    assert is_private_lan("169.254.1.1") is False       # link-local
    assert is_private_lan("8.8.8.8") is False           # public
    assert is_private_lan(None) is False
    assert is_private_lan("garbage") is False


# --- LAN stays open (the new requirement) -----------------------------------

def test_lan_client_allowed_without_token():
    assert _allows("10.100.100.7", token=None) is True
    assert _allows("192.168.0.50", token=None) is True


def test_lan_trust_can_be_disabled():
    assert _allows("10.100.100.7", token=None, trust_lan=False) is False
    assert _allows("10.100.100.7", token=TOKEN, trust_lan=False) is True  # token still works


# --- public path (loopback via tunnel) requires the token -------------------

def test_loopback_requires_token():
    assert _allows("127.0.0.1", token=None) is False           # public w/o token → blocked
    assert _allows("127.0.0.1", token=TOKEN) is True           # gateway-injected token → ok
    assert _allows("127.0.0.1", token="wrong") is False


def test_public_ip_requires_token():
    # a real public peer (8.8.8.8) is non-private; in practice the 4090 only ever
    # sees LAN peers or the loopback tunnel, never a raw public IP.
    assert _allows("8.8.8.8", token=None) is False
    assert _allows("8.8.8.8", token=TOKEN) is True


# --- exemptions / preflight --------------------------------------------------

def test_exempt_path_and_options_always_pass():
    assert _allows("8.8.8.8", token=None, path="/health") is True
    assert _allows("127.0.0.1", token=None, method="OPTIONS") is True
