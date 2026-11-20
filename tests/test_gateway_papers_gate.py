"""Paper content resources (/api/papers/*) are LOGIN-GATED (no credit charge) and
the gateway scrubs internal filesystem paths (txt_path/pdf_path = SMB/UNC infra
paths) from the public JSON before returning. Anonymous → 401; logged-in → 200
with internal paths stripped, other fields intact. Binary (pdf) passes through.
"""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.auth import accounts
from gateway.settings import GatewaySettings

PEPPER = "test-pepper"


class PaperProxy:
    """Stands in for the 4090: returns a paper detail carrying internal paths."""
    async def request(self, method, path, *, headers=None, params=None, content=b""):
        if path.endswith("/pdf"):
            return httpx.Response(200, content=b"%PDF-1.7 binary",
                                  headers={"content-type": "application/pdf"})
        return httpx.Response(200, json={
            "paper_id": "p1", "title": "T",
            "metadata": {"venue": "AAAI", "year": 2026,
                         "txt_path": "\\\\10.100.100.9\\lsb\\x.md",
                         "pdf_path": "/srv/papers/x.pdf"},
            "chunks": [{"content": "hi", "txt_path": "leak"}]})

    async def stream(self, method, path, *, headers=None, params=None, content=b""):
        async def gen():
            yield b"%PDF-1.7 binary" if path.endswith("/pdf") else b""
        content_type = "application/pdf" if path.endswith("/pdf") else "text/event-stream"
        return 200, {"content-type": content_type}, gen()


@pytest.fixture
def app():
    settings = GatewaySettings(db_url="sqlite://", key_pepper=PEPPER,
                               session_secret="s", admin_root_key="sk-root")
    return create_app(settings, proxy=PaperProxy())


@pytest.fixture
def client(app):
    return TestClient(app)


def _mk_user(app):
    db = app.state.session_factory()
    try:
        u = accounts.register(db, oidc_sub="cdk:u", username="u")
        u.balance = 5  # enough to open a paper (metered 0.1); these tests check scrubbing
        key, _ = accounts.issue_key(db, u, PEPPER)
        db.commit()
        return key
    finally:
        db.close()


def _mk_member(app):
    from gateway.db.models import Tier
    db = app.state.session_factory()
    try:
        u = accounts.register(db, oidc_sub="cdk:member", username="member")
        u.tier = Tier.internal
        key, _ = accounts.issue_key(db, u, PEPPER)
        db.commit()
        return key
    finally:
        db.close()


def test_papers_requires_login(client):
    assert client.get("/api/papers/p1").status_code == 401
    assert client.get("/api/papers/p1/fulltext").status_code == 401
    assert client.get("/api/papers/p1/markdown").status_code == 401
    assert client.post("/api/papers/download-batch", json={
        "paper_ids": ["p1"], "artifacts": "pdf",
    }).status_code == 401


def test_invalid_batch_body_is_not_parsed_before_authentication(client):
    response = client.post(
        "/api/papers/download-batch",
        content=b"[not valid json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 401


def test_papers_authed_ok_and_scrubs_internal_paths(app, client):
    key = _mk_user(app)
    r = client.get("/api/papers/p1", headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200
    dumped = r.text
    assert "txt_path" not in dumped and "pdf_path" not in dumped  # scrubbed everywhere
    body = r.json()
    assert body["metadata"]["venue"] == "AAAI" and body["metadata"]["year"] == 2026  # intact
    assert body["chunks"][0]["content"] == "hi"


def test_papers_pdf_binary_passes_through_for_member(app, client):
    key = _mk_member(app)
    r = client.get("/api/papers/p1/pdf", headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200
    assert r.content.startswith(b"%PDF")
    assert r.headers["content-type"].startswith("application/pdf")


def test_papers_pdf_streams_without_buffering_the_whole_file():
    class StreamingPaperProxy(PaperProxy):
        def __init__(self):
            self.stream_paths = []

        async def request(self, method, path, *, headers=None, params=None, content=b""):
            if path.endswith("/pdf"):
                raise AssertionError("PDF route must use proxy.stream, not buffered request")
            return await super().request(
                method, path, headers=headers, params=params, content=content)

        async def stream(self, method, path, *, headers=None, params=None, content=b""):
            self.stream_paths.append(path)

            async def gen():
                yield b"%PDF-1.7\n"
                yield b"streamed body"

            return 200, {
                "content-type": "application/pdf",
                "content-length": "22",
                "content-disposition": "inline; filename=paper.pdf",
            }, gen()

    proxy = StreamingPaperProxy()
    settings = GatewaySettings(db_url="sqlite://", key_pepper=PEPPER,
                               session_secret="s", admin_root_key="sk-root")
    streaming_app = create_app(settings, proxy=proxy)
    key = _mk_member(streaming_app)

    with TestClient(streaming_app) as streaming_client:
        response = streaming_client.get(
            "/api/papers/p1/pdf", headers={"Authorization": f"Bearer {key}"})

    assert response.content == b"%PDF-1.7\nstreamed body"
    assert response.headers["content-length"] == "22"
    assert response.headers["content-disposition"] == "inline; filename=paper.pdf"
    assert proxy.stream_paths == ["/api/papers/p1/pdf"]


def test_papers_markdown_streams_without_buffering_the_whole_file():
    class StreamingMarkdownProxy(PaperProxy):
        async def request(self, method, path, *, headers=None, params=None, content=b""):
            if path.endswith("/markdown"):
                raise AssertionError("Markdown file route must use proxy.stream")
            return await super().request(
                method, path, headers=headers, params=params, content=content)

        async def stream(self, method, path, *, headers=None, params=None, content=b""):
            async def gen():
                yield b"# Paper\n"
                yield b"\nComplete body.\n"

            return 200, {
                "content-type": "text/markdown; charset=utf-8",
                "content-length": "24",
                "content-disposition": "attachment; filename=paper.md",
            }, gen()

    proxy = StreamingMarkdownProxy()
    settings = GatewaySettings(db_url="sqlite://", key_pepper=PEPPER,
                               session_secret="s", admin_root_key="sk-root")
    markdown_app = create_app(settings, proxy=proxy)
    key = _mk_user(markdown_app)

    with TestClient(markdown_app) as markdown_client:
        response = markdown_client.get(
            "/api/papers/p1/markdown",
            headers={"Authorization": f"Bearer {key}"},
        )

    assert response.status_code == 200
    assert response.content == b"# Paper\n\nComplete body.\n"
    assert response.headers["content-type"].startswith("text/markdown")


def test_papers_pdf_preserves_partial_content_range_headers():
    class RangePaperProxy(PaperProxy):
        async def stream(self, method, path, *, headers=None, params=None, content=b""):
            assert headers.get("range") == "bytes=0-3"

            async def gen():
                yield b"%PDF"

            return 206, {
                "content-type": "application/pdf",
                "content-length": "4",
                "content-range": "bytes 0-3/100",
                "accept-ranges": "bytes",
            }, gen()

    proxy = RangePaperProxy()
    settings = GatewaySettings(db_url="sqlite://", key_pepper=PEPPER,
                               session_secret="s", admin_root_key="sk-root")
    range_app = create_app(settings, proxy=proxy)
    key = _mk_member(range_app)

    with TestClient(range_app) as range_client:
        response = range_client.get(
            "/api/papers/p1/pdf",
            headers={"Authorization": f"Bearer {key}", "Range": "bytes=0-3"},
        )

    assert response.status_code == 206
    assert response.content == b"%PDF"
    assert response.headers["content-range"] == "bytes 0-3/100"
    assert response.headers["accept-ranges"] == "bytes"


def test_papers_pdf_preserves_unsatisfied_range_size():
    class UnsatisfiedRangePaperProxy(PaperProxy):
        async def stream(self, method, path, *, headers=None, params=None, content=b""):
            async def gen():
                yield b""

            return 416, {
                "content-type": "application/pdf",
                "content-length": "0",
                "content-range": "bytes */100",
                "accept-ranges": "bytes",
            }, gen()

    proxy = UnsatisfiedRangePaperProxy()
    settings = GatewaySettings(db_url="sqlite://", key_pepper=PEPPER,
                               session_secret="s", admin_root_key="sk-root")
    range_app = create_app(settings, proxy=proxy)
    key = _mk_member(range_app)

    with TestClient(range_app) as range_client:
        response = range_client.get(
            "/api/papers/p1/pdf",
            headers={"Authorization": f"Bearer {key}", "Range": "bytes=1000-"},
        )

    assert response.status_code == 416
    assert response.headers["content-range"] == "bytes */100"


def test_papers_batch_download_streams_authenticated_zip():
    class BatchPaperProxy(PaperProxy):
        def __init__(self):
            self.body = None

        async def stream(self, method, path, *, headers=None, params=None, content=b""):
            assert method == "POST"
            assert path == "/api/papers/download-batch"
            self.body = json.loads(content)

            async def gen():
                yield b"PK\x03\x04batch"

            return 200, {
                "content-type": "application/zip",
                "content-length": "9",
                "content-disposition": "attachment; filename=papers.zip",
                "x-retrieval-artifact-count": "4",
                "x-retrieval-error-count": "0",
            }, gen()

    proxy = BatchPaperProxy()
    settings = GatewaySettings(db_url="sqlite://", key_pepper=PEPPER,
                               session_secret="s", admin_root_key="sk-root")
    batch_app = create_app(settings, proxy=proxy)
    key = _mk_member(batch_app)

    with TestClient(batch_app) as batch_client:
        response = batch_client.post(
            "/api/papers/download-batch",
            headers={"Authorization": f"Bearer {key}"},
            json={"paper_ids": ["p1", "p2"], "artifacts": "both"},
        )

    assert response.status_code == 200
    assert response.content == b"PK\x03\x04batch"
    assert response.headers["content-disposition"] == "attachment; filename=papers.zip"
    assert response.headers["x-retrieval-artifact-count"] == "4"
    assert response.headers["x-retrieval-error-count"] == "0"
    assert proxy.body == {"paper_ids": ["p1", "p2"], "artifacts": "both"}


def test_papers_batch_download_preserves_retry_after_on_capacity_limit():
    class BusyBatchProxy(PaperProxy):
        async def stream(self, method, path, *, headers=None, params=None, content=b""):
            async def gen():
                yield b'{"detail":"busy"}'

            return 429, {
                "content-type": "application/json",
                "content-length": "17",
                "retry-after": "2",
            }, gen()

    settings = GatewaySettings(db_url="sqlite://", key_pepper=PEPPER,
                               session_secret="s", admin_root_key="sk-root")
    busy_app = create_app(settings, proxy=BusyBatchProxy())
    key = _mk_member(busy_app)

    with TestClient(busy_app) as busy_client:
        response = busy_client.post(
            "/api/papers/download-batch",
            headers={"Authorization": f"Bearer {key}"},
            json={"paper_ids": ["p1"], "artifacts": "pdf"},
        )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "2"


def _mk_paying(app, balance=3, identity="pay"):
    from gateway.db.models import Tier
    db = app.state.session_factory()
    u = accounts.register(db, oidc_sub=f"cdk:{identity}", username=identity)
    u.tier = Tier.paying
    u.balance = balance
    key, _ = accounts.issue_key(db, u, PEPPER)
    db.commit()
    db.close()
    return key


def _bal(client, key):
    from decimal import Decimal
    return Decimal(str(client.get("/auth/me", headers={"Authorization": f"Bearer {key}"}).json()["balance"]))


def test_paying_charged_for_opening_a_paper(app, client):
    """Opening a paper detail costs 0.1 for a paying user (member+ is free)."""
    from decimal import Decimal
    key = _mk_paying(app, balance=3)
    start = _bal(client, key)
    client.get("/api/papers/p1", headers={"Authorization": f"Bearer {key}"})
    assert start - _bal(client, key) == Decimal("0.1")


@pytest.mark.parametrize("endpoint", ["browse", "tags", "stats"])
def test_catalog_endpoints_are_free(app, client, endpoint):
    """Catalog/metadata endpoints (list, filter facets, stats) are free reads — not
    per-paper viewing. A metered `tags`/`stats` would 402 low-balance users and empty
    the 论文库 filter dropdowns."""
    key = _mk_paying(app, balance=3)
    start = _bal(client, key)
    r = client.get(f"/api/papers/{endpoint}", headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200
    assert _bal(client, key) == start


def test_paper_content_subresources_charge_paying_user(app, client):
    """Markdown is metered; PDF remains a member-only artifact."""
    from decimal import Decimal
    key = _mk_paying(app, balance=3)
    start = _bal(client, key)
    fulltext = client.get(
        "/api/papers/p1/fulltext", headers={"Authorization": f"Bearer {key}"})
    markdown = client.get(
        "/api/papers/p1/markdown", headers={"Authorization": f"Bearer {key}"})
    pdf = client.get(
        "/api/papers/p1/pdf", headers={"Authorization": f"Bearer {key}"})

    assert fulltext.status_code == 200
    assert markdown.status_code == 200
    assert pdf.status_code == 403
    assert start - _bal(client, key) == Decimal("0.2")


def test_batch_download_charges_unique_papers_once_with_idempotency(app):
    key = _mk_paying(app, balance=3)
    headers = {
        "Authorization": f"Bearer {key}",
        "X-Idempotency-Key": "paper-batch-download-1",
    }
    payload = {"paper_ids": ["p1", "p1", "p2"], "artifacts": "markdown"}
    start = _bal(TestClient(app), key)

    with TestClient(app) as batch_client:
        first = batch_client.post("/api/papers/download-batch", headers=headers, json=payload)
        second = batch_client.post("/api/papers/download-batch", headers=headers, json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    with TestClient(app) as balance_client:
        from decimal import Decimal
        assert start - _bal(balance_client, key) == Decimal("0.2")


def test_markdown_ranges_share_one_logical_charge(app):
    key = _mk_paying(app, balance=3, identity="range-pay")
    headers = {
        "Authorization": f"Bearer {key}",
        "X-Idempotency-Key": "one-markdown-transfer",
    }
    start = _bal(TestClient(app), key)

    with TestClient(app) as range_client:
        first = range_client.get(
            "/api/papers/p1/markdown",
            headers={**headers, "Range": "bytes=0-3"},
        )
        second = range_client.get(
            "/api/papers/p1/markdown",
            headers={**headers, "Range": "bytes=4-7"},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    with TestClient(app) as balance_client:
        from decimal import Decimal
        assert start - _bal(balance_client, key) == Decimal("0.1")


def test_reusing_idempotency_key_for_another_paper_is_rejected(app):
    key = _mk_paying(app, balance=3, identity="conflict-pay")
    headers = {
        "Authorization": f"Bearer {key}",
        "X-Idempotency-Key": "paper-transfer-key",
    }
    start = _bal(TestClient(app), key)

    with TestClient(app) as conflict_client:
        first = conflict_client.get("/api/papers/p1/markdown", headers=headers)
        second = conflict_client.get("/api/papers/p2/markdown", headers=headers)

    assert first.status_code == 200
    assert second.status_code == 409
    with TestClient(app) as balance_client:
        from decimal import Decimal
        assert start - _bal(balance_client, key) == Decimal("0.1")


def test_same_client_idempotency_key_is_isolated_per_user(app):
    first_key = _mk_paying(app, balance=3, identity="first-pay")
    second_key = _mk_paying(app, balance=3, identity="second-pay")
    shared = "same-client-key"

    with TestClient(app) as isolated_client:
        first = isolated_client.get(
            "/api/papers/p1/markdown",
            headers={
                "Authorization": f"Bearer {first_key}",
                "X-Idempotency-Key": shared,
            },
        )
        second = isolated_client.get(
            "/api/papers/p1/markdown",
            headers={
                "Authorization": f"Bearer {second_key}",
                "X-Idempotency-Key": shared,
            },
        )

    assert first.status_code == second.status_code == 200
    with TestClient(app) as balance_client:
        from decimal import Decimal
        assert _bal(balance_client, first_key) == Decimal("2.9")
        assert _bal(balance_client, second_key) == Decimal("2.9")


@pytest.mark.parametrize("payload", [None, [], "markdown", 7])
def test_invalid_batch_json_is_rejected_before_billing(app, payload):
    key = _mk_paying(app, balance=3, identity=f"invalid-{type(payload).__name__}")
    start = _bal(TestClient(app), key)

    with TestClient(app) as invalid_client:
        response = invalid_client.post(
            "/api/papers/download-batch",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
        )

    assert response.status_code == 422
    with TestClient(app) as balance_client:
        assert _bal(balance_client, key) == start


def test_oversized_batch_is_rejected_before_billing(app):
    key = _mk_paying(app, balance=3, identity="oversized-pay")
    start = _bal(TestClient(app), key)

    with TestClient(app) as invalid_client:
        response = invalid_client.post(
            "/api/papers/download-batch",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "paper_ids": [f"paper-{index}" for index in range(11)],
                "artifacts": "markdown",
            },
        )

    assert response.status_code == 422
    with TestClient(app) as balance_client:
        assert _bal(balance_client, key) == start


def test_batch_request_body_has_a_hard_size_limit(app):
    key = _mk_paying(app, balance=3, identity="large-body-pay")
    start = _bal(TestClient(app), key)
    oversized_body = b'{' + b'"padding":"' + (b"x" * 20_000) + b'"}'

    with TestClient(app) as invalid_client:
        response = invalid_client.post(
            "/api/papers/download-batch",
            content=oversized_body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )

    assert response.status_code == 413
    with TestClient(app) as balance_client:
        assert _bal(balance_client, key) == start


@pytest.mark.parametrize("status_code", [404, 413, 416, 422, 429])
def test_failed_markdown_stream_is_refunded(status_code):
    class FailedMarkdownProxy(PaperProxy):
        async def stream(self, method, path, *, headers=None, params=None, content=b""):
            async def gen():
                yield b'{"detail":"artifact unavailable"}'

            return status_code, {"content-type": "application/json"}, gen()

    settings = GatewaySettings(db_url="sqlite://", key_pepper=PEPPER,
                               session_secret="s", admin_root_key="sk-root")
    failed_app = create_app(settings, proxy=FailedMarkdownProxy())
    key = _mk_paying(failed_app, balance=3, identity=f"failed-md-{status_code}")
    start = _bal(TestClient(failed_app), key)

    with TestClient(failed_app) as failed_client:
        response = failed_client.get(
            "/api/papers/missing/markdown",
            headers={"Authorization": f"Bearer {key}"},
        )

    assert response.status_code == status_code
    with TestClient(failed_app) as balance_client:
        assert _bal(balance_client, key) == start


def test_failed_fulltext_is_refunded_and_internal_path_is_scrubbed():
    class FulltextProxy(PaperProxy):
        def __init__(self):
            self.status_code = 404

        async def request(self, method, path, *, headers=None, params=None, content=b""):
            return httpx.Response(
                self.status_code,
                json={
                    "paper_id": "p1",
                    "content": "# Paper",
                    "path": "/mnt/lsb/private/p1.md",
                },
            )

    proxy = FulltextProxy()
    settings = GatewaySettings(db_url="sqlite://", key_pepper=PEPPER,
                               session_secret="s", admin_root_key="sk-root")
    fulltext_app = create_app(settings, proxy=proxy)
    key = _mk_paying(fulltext_app, balance=3, identity="fulltext-pay")
    start = _bal(TestClient(fulltext_app), key)

    with TestClient(fulltext_app) as fulltext_client:
        failed = fulltext_client.get(
            "/api/papers/p1/fulltext",
            headers={"Authorization": f"Bearer {key}"},
        )
        proxy.status_code = 200
        succeeded = fulltext_client.get(
            "/api/papers/p1/fulltext",
            headers={"Authorization": f"Bearer {key}"},
        )

    assert failed.status_code == 404
    assert succeeded.status_code == 200
    assert "path" not in succeeded.json()
    assert "/mnt/lsb" not in succeeded.text
    with TestClient(fulltext_app) as balance_client:
        from decimal import Decimal
        assert start - _bal(balance_client, key) == Decimal("0.1")


@pytest.mark.parametrize("artifacts", ["pdf", "both"])
def test_batch_pdf_download_requires_member(app, artifacts):
    key = _mk_paying(app, balance=3)
    start = _bal(TestClient(app), key)

    with TestClient(app) as batch_client:
        response = batch_client.post(
            "/api/papers/download-batch",
            headers={"Authorization": f"Bearer {key}"},
            json={"paper_ids": ["p1"], "artifacts": artifacts},
        )

    assert response.status_code == 403
    with TestClient(app) as balance_client:
        assert _bal(balance_client, key) == start


def test_member_not_charged_for_papers(app, client):
    from decimal import Decimal
    from gateway.db.models import Tier
    db = app.state.session_factory()
    u = accounts.register(db, oidc_sub="cdk:m", username="m")
    u.tier = Tier.internal
    u.balance = 3
    key, _ = accounts.issue_key(db, u, PEPPER)
    db.commit()
    db.close()
    client.get("/api/papers/p1", headers={"Authorization": f"Bearer {key}"})
    assert _bal(client, key) == Decimal("3")
