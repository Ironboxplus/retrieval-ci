"""The document library (/api/documents/*) is the READ-ONLY public browse surface.

Policy at the gateway edge:
  - GET is login-gated + FREE (browsing the shared corpus: papers/webpages/uploads).
  - Only GET is routed → POST/PUT/DELETE (upload/edit/delete/crawl) get 405: the public
    site cannot mutate the corpus.
  - Raw source-file bytes (/{id}/download) are 403 — metadata + chunk previews only.
  - JSON is scrubbed of internal filesystem paths + internal download URLs.
Anonymous → 401.
"""
import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.auth import accounts
from gateway.db.models import Tier
from gateway.settings import GatewaySettings

PEPPER = "test-pepper"


class DocProxy:
    """Stands in for the 4090: returns a document list/detail carrying internal paths."""
    async def request(self, method, path, *, headers=None, params=None, content=b""):
        if path.rstrip("/") == "/api/documents":
            return httpx.Response(200, json={
                "total": 1, "page": 1, "page_size": 20,
                "documents": [{
                    "id": 7, "title": "Doc", "source_type": "webpage", "status": "indexed",
                    "chunk_count": 3,
                    "file_path": "/srv/uploads/x.pdf",
                    "download_url": "/api/documents/7/download",
                    "source_file_path": "\\\\10.100.100.9\\lsb\\x.md",
                }]})
        return httpx.Response(200, json={
            "id": 7, "title": "Doc", "source_type": "webpage", "status": "indexed",
            "chunk_count": 3, "file_path": "/srv/uploads/x.pdf",
            "download_url": "/api/documents/7/download",
            "chunks": [{"level": 1, "chunk_index": 0, "content_preview": "hi"}]})

    async def stream(self, method, path, *, headers=None, params=None, content=b""):
        async def gen():
            yield b""
        return 200, {"content-type": "text/event-stream"}, gen()


@pytest.fixture
def app():
    settings = GatewaySettings(db_url="sqlite://", key_pepper=PEPPER,
                               session_secret="s", admin_root_key="sk-root")
    return create_app(settings, proxy=DocProxy())


@pytest.fixture
def client(app):
    return TestClient(app)


def _mk_paying(app, balance=3):
    db = app.state.session_factory()
    u = accounts.register(db, oidc_sub="cdk:pay", username="pay")
    u.tier = Tier.paying
    u.balance = balance
    key, _ = accounts.issue_key(db, u, PEPPER)
    db.commit()
    db.close()
    return key


def _bal(client, key):
    from decimal import Decimal
    return Decimal(str(client.get("/auth/me", headers={"Authorization": f"Bearer {key}"}).json()["balance"]))


def test_documents_requires_login(client):
    assert client.get("/api/documents").status_code == 401
    assert client.get("/api/documents/7").status_code == 401


def test_documents_list_authed_ok_and_scrubbed(app, client):
    key = _mk_paying(app)
    r = client.get("/api/documents", headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200
    dumped = r.text
    for leak in ("file_path", "source_file_path", "download_url"):
        assert leak not in dumped
    body = r.json()
    assert body["documents"][0]["title"] == "Doc"  # non-path fields intact


def test_documents_detail_scrubbed(app, client):
    key = _mk_paying(app)
    r = client.get("/api/documents/7", headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200
    assert "file_path" not in r.text and "download_url" not in r.text
    assert r.json()["chunks"][0]["content_preview"] == "hi"


def test_documents_browsing_is_free(app, client):
    from decimal import Decimal
    key = _mk_paying(app, balance=3)
    start = _bal(client, key)
    client.get("/api/documents", headers={"Authorization": f"Bearer {key}"})
    client.get("/api/documents/7", headers={"Authorization": f"Bearer {key}"})
    assert _bal(client, key) == start


@pytest.mark.parametrize("method", ["post", "put", "delete"])
def test_documents_mutations_blocked(app, client, method):
    key = _mk_paying(app)
    r = getattr(client, method)("/api/documents/7",
                                headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 405  # only GET is routed — corpus is read-only publicly


@pytest.mark.parametrize("sub", ["download", "source-download"])
def test_documents_raw_file_routes_forbidden(app, client, sub):
    """Both raw-file routes are blocked — the whitelist can't be out-run by a route
    whose name merely ends in '-download' (source-download would slip an endswith check)."""
    key = _mk_paying(app)
    r = client.get(f"/api/documents/7/{sub}", headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 403


def test_documents_index_status_allowed(app, client):
    """index-status is a whitelisted metadata sub-route → forwards (200)."""
    key = _mk_paying(app)
    r = client.get("/api/documents/7/index-status",
                   headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200


def test_documents_unknown_subroute_forbidden(app, client):
    """Anything outside the metadata whitelist (list/detail/index-status) is 403."""
    key = _mk_paying(app)
    r = client.get("/api/documents/7/fulltext",
                   headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 403
