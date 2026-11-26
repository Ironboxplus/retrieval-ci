from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_compute111_forwarding_unit_uses_a_restricted_persistent_local_tunnel():
    unit = (
        ROOT / "deploy" / "compute111" / "retrieval-forward-88.service"
    ).read_text(encoding="utf-8")

    assert "Description=Retrieval internal forward from compute .111 to compute .88" in unit
    # SSH's remote forwarding endpoint creates the TCP connection seen by FastAPI.
    # Use .88's LAN address rather than loopback so identity.py keeps the legacy
    # :8000 caller on the trusted internal door rather than classifying it as the
    # public tunnel peer.
    assert "-L 0.0.0.0:8000:10.100.100.88:18080" in unit
    assert "arc@10.100.100.88" in unit
    assert "ExitOnForwardFailure=yes" in unit
    assert "StrictHostKeyChecking=yes" in unit
    assert "UserKnownHostsFile=%h/.ssh/known_hosts" in unit
    assert "IdentitiesOnly=yes" in unit
    assert "IdentityFile=%h/.ssh/retrieval-forward-88_ed25519" in unit
    assert "Restart=always" in unit
    assert "RestartSec=5" in unit
    assert "CUDA_VISIBLE_DEVICES" not in unit
    assert "API_KEY=" not in unit
