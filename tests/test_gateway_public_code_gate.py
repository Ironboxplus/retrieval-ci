"""Public Code is reachable through the gateway only for admin users."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.auth import accounts
from gateway.db.models import Tier
from gateway.settings import GatewaySettings


PEPPER = "test-pepper"
ROOT = Path(__file__).resolve().parents[1]


class RecordingProxy:
    def __init__(self):
        self.calls = []

    async def request(self, method, path, *, headers=None, params=None, content=b""):
        self.calls.append({
            "method": method,
            "path": path,
            "headers": dict(headers or {}),
            "params": dict(params or {}),
            "body": json.loads(content or b"{}") if content else None,
        })
        return httpx.Response(200, json={"ok": True})


@pytest.fixture
def app():
    settings = GatewaySettings(
        db_url="sqlite://",
        key_pepper=PEPPER,
        session_secret="s",
        admin_root_key="sk-root",
    )
    return create_app(settings, proxy=RecordingProxy())


def _key(app, tier: Tier):
    db = app.state.session_factory()
    try:
        user = accounts.register(db, oidc_sub=f"public-code:{tier.value}", username=tier.value)
        user.tier = tier
        key, _ = accounts.issue_key(db, user, PEPPER)
        db.commit()
        return user.id, key
    finally:
        db.close()


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_gateway_public_code_requires_admin_and_forwards_identity(app):
    client = TestClient(app)
    assert client.get("/api/public-code/sources").status_code == 401
    _, paying = _key(app, Tier.paying)
    assert client.get("/api/public-code/sources", headers=_auth(paying)).status_code == 403

    user_id, admin = _key(app, Tier.admin)
    response = client.post(
        "/api/public-code/search",
        headers=_auth(admin),
        json={"query": "packet decoder", "source_ids": ["ffmpeg"]},
    )

    assert response.status_code == 200
    call = app.state.proxy.calls[-1]
    assert call["path"] == "/api/public-code/search"
    assert call["headers"]["X-Retrival-User-Id"] == str(user_id)
    assert call["headers"]["X-Retrival-User-Tier"] == "admin"


def test_bwh_traefik_routes_public_code_to_gateway():
    for path in (
        ROOT / "deploy" / "bwh" / "retrieval.yml",
        ROOT / "deploy" / "bwh" / "traefik-retrieval.patch.yml",
    ):
        assert "PathPrefix(`/api/public-code`)" in path.read_text(encoding="utf-8")
