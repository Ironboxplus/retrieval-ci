"""Run a real HTTP smoke test for ACE Journal through the gateway.

This boots two local uvicorn servers:
- compute app with the real journal router and JournalService
- gateway app with real sk-key auth and proxy forwarding

The journal vector layer is an in-memory test double so this smoke never writes
to the production paper/Qdrant collections. The purpose is to verify the
machine-level API/auth/owner-isolation/MCP chain, not vector recall quality.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes import journal as journal_routes
from app.services.journal.repository import JournalRepository
from app.services.journal.service import JournalService
from core.db.database import Base
from gateway.app import create_app
from gateway.auth import accounts
from gateway.settings import GatewaySettings
from mcp_server.backend import RetrievalClient, format_journal_recent, format_journal_search

INTERNAL_TOKEN = "journal-smoke-internal-token"


@contextmanager
def make_journal_session_factory() -> Iterator[Any]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_factory():
        session = session_local()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    try:
        yield session_factory
    finally:
        engine.dispose()


def fake_embed(texts, include_sparse):
    dense_vectors = []
    sparse_vectors = []
    for text in texts:
        words = text.lower().split()
        dense_vectors.append([float(len(text)), float(len(words)), 1.0])
        sparse_vectors.append({word: float(words.count(word)) for word in set(words)})
    return dense_vectors, sparse_vectors if include_sparse else None


class InMemoryJournalVectorStore:
    """Small vector-store double that honors the same filter contract."""

    def __init__(self):
        self.points: list[dict[str, Any]] = []

    def upsert_chunks(self, chunks, **kwargs) -> int:
        for chunk in chunks:
            self.points.append({
                "id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": {
                    "project_id": kwargs["project_id"],
                    "project_slug": kwargs["project_slug"],
                    "owner_user_id": kwargs.get("owner_user_id") or "",
                    "entry_id": kwargs["entry_id"],
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "source_kind": kwargs["source_kind"],
                    "client": kwargs["client"],
                    "level": chunk.level,
                    "heading": chunk.heading or "",
                    "token_count": chunk.token_count,
                    "tags": kwargs.get("tags") or [],
                },
            })
        return len(chunks)

    def search(self, *, dense_vector, sparse_vector, mode, top_k, filters) -> list[dict[str, Any]]:
        query_terms = set((sparse_vector or {}).keys())
        matches = []
        for point in self.points:
            metadata = point["metadata"]
            if any(metadata.get(key) != expected for key, expected in filters.items() if key != "tags"):
                continue
            if "tags" in filters and not set(filters["tags"]).issubset(set(metadata.get("tags") or [])):
                continue

            text_terms = set(point["content"].lower().split())
            overlap = len(query_terms & text_terms)
            matches.append({
                "id": point["id"],
                "content": point["content"],
                "score": 1.0 + overlap if overlap else 0.25,
                "metadata": metadata,
            })
        return sorted(matches, key=lambda item: item["score"], reverse=True)[:top_k]

    def delete_by_entry_id(self, entry_id: str) -> int:
        before = len(self.points)
        self.points = [
            point for point in self.points
            if point["metadata"].get("entry_id") != entry_id
        ]
        return before - len(self.points)


def build_compute_app(service: JournalService) -> FastAPI:
    app = FastAPI(title="journal-smoke-compute")

    @app.middleware("http")
    async def internal_token_gate(request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        if request.headers.get("X-Internal-Token") != INTERNAL_TOKEN:
            return JSONResponse({"detail": "internal token required"}, status_code=401)
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/internal/ingest-log")
    async def ingest_log():
        return {"ok": True}

    journal_routes.get_journal_service = lambda: service
    app.include_router(journal_routes.router, prefix="/api/journal")
    return app


def run_server(app: FastAPI, port: int):
    server = uvicorn.Server(uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        lifespan="on",
    ))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread


def wait_http(url: str, headers: dict[str, str] | None = None) -> httpx.Response:
    deadline = time.time() + 15
    last_error = None
    while time.time() < deadline:
        try:
            response = httpx.get(url, headers=headers, timeout=1.0)
            if response.status_code < 500:
                return response
        except Exception as exc:  # noqa: BLE001 - keep retrying until timeout.
            last_error = exc
        time.sleep(0.2)
    raise RuntimeError(f"server did not become ready: {url}; last_error={last_error!r}")


def create_gateway_users(gateway_app: FastAPI, settings: GatewaySettings) -> tuple[str, str]:
    db = gateway_app.state.session_factory()
    try:
        user_one = accounts.register_with_identity(
            db, provider="cdk", sub="smoke:user-one", username="smoke_user_one")
        user_two = accounts.register_with_identity(
            db, provider="cdk", sub="smoke:user-two", username="smoke_user_two")
        key_one, _ = accounts.issue_key(db, user_one, settings.key_pepper)
        key_two, _ = accounts.issue_key(db, user_two, settings.key_pepper)
        db.commit()
        return key_one, key_two
    finally:
        db.close()


async def run_mcp_checks(gateway_url: str, api_key: str) -> tuple[dict, dict, dict, str, str]:
    client = RetrievalClient(base_url=gateway_url, api_key=api_key)
    try:
        recorded = await client.record_journal(
            project_slug="retrival-smoke",
            client="mcp-smoke",
            source_kind="session_summary",
            title="MCP smoke",
            body="MCP journal_record over gateway bearer auth gamma memory",
            tags=["smoke", "mcp"],
        )
        found = await client.search_journal(
            project_slug="retrival-smoke",
            query="gamma gateway bearer",
            top_k=3,
            include_chunks=True,
        )
        recent = await client.recent_journal_entries(project_slug="retrival-smoke", limit=5)
        return recorded, found, recent, format_journal_search(found), format_journal_recent(recent)
    finally:
        await client.aclose()


def run_smoke(compute_port: int, gateway_port: int) -> dict[str, Any]:
    with (
        make_journal_session_factory() as journal_session_factory,
        tempfile.TemporaryDirectory(prefix="retrival_gateway_smoke_", ignore_cleanup_errors=True) as tmp,
    ):
        service = JournalService(
            repository=JournalRepository(session_factory=journal_session_factory),
            vector_store=InMemoryJournalVectorStore(),
            embed_fn=fake_embed,
        )
        compute_server, compute_thread = run_server(build_compute_app(service), compute_port)

        settings = GatewaySettings(
            db_url=f"sqlite:///{Path(tmp) / 'gateway.db'}",
            internal_base_url=f"http://127.0.0.1:{compute_port}",
            internal_token=INTERNAL_TOKEN,
            session_secret="journal-smoke-session",
            key_pepper="journal-smoke-pepper",
            payment_provider="mock",
        )
        gateway_app = create_app(settings=settings)
        gateway_server, gateway_thread = run_server(gateway_app, gateway_port)

        try:
            compute_url = f"http://127.0.0.1:{compute_port}"
            gateway_url = f"http://127.0.0.1:{gateway_port}"
            wait_http(f"{compute_url}/health")
            wait_http(f"{gateway_url}/auth/providers")

            key_one, key_two = create_gateway_users(gateway_app, settings)
            headers_one = {"Authorization": f"Bearer {key_one}"}
            headers_two = {"Authorization": f"Bearer {key_two}"}

            direct_no_token = httpx.post(
                f"{compute_url}/api/journal/entries",
                json={"project_slug": "retrival-smoke", "body": "no token"},
                timeout=5,
            )
            direct_no_user = httpx.post(
                f"{compute_url}/api/journal/entries",
                headers={"X-Internal-Token": INTERNAL_TOKEN},
                json={"project_slug": "retrival-smoke", "body": "no user"},
                timeout=5,
            )
            gateway_anonymous = httpx.post(
                f"{gateway_url}/api/journal/entries",
                json={"project_slug": "retrival-smoke", "body": "anonymous"},
                timeout=5,
            )

            record_one = httpx.post(
                f"{gateway_url}/api/journal/entries",
                headers=headers_one,
                json={
                    "project_slug": "retrival-smoke",
                    "client": "codex-smoke",
                    "source_kind": "session_summary",
                    "title": "ACE Journal smoke user one",
                    "body": "ACE journal real HTTP smoke memory alpha owner isolation gateway auth",
                    "tags": ["smoke", "journal"],
                    "owner_user_id": "malicious-user-id-must-be-overwritten",
                },
                timeout=10,
            )
            record_one.raise_for_status()
            entry_one = record_one.json()["entry"]

            record_two = httpx.post(
                f"{gateway_url}/api/journal/entries",
                headers=headers_two,
                json={
                    "project_slug": "retrival-smoke",
                    "client": "codex-smoke",
                    "source_kind": "session_summary",
                    "title": "ACE Journal smoke user two",
                    "body": "ACE journal real HTTP smoke memory beta second owner",
                    "tags": ["smoke", "journal"],
                },
                timeout=10,
            )
            record_two.raise_for_status()
            entry_two = record_two.json()["entry"]

            search_one = httpx.post(
                f"{gateway_url}/api/journal/search",
                headers=headers_one,
                json={
                    "project_slug": "retrival-smoke",
                    "query": "alpha owner isolation gateway auth",
                    "top_k": 5,
                    "dedup_by_entry": True,
                    "include_chunks": True,
                },
                timeout=10,
            )
            search_one.raise_for_status()
            search_one_json = search_one.json()

            search_two_for_alpha = httpx.post(
                f"{gateway_url}/api/journal/search",
                headers=headers_two,
                json={
                    "project_slug": "retrival-smoke",
                    "query": "alpha owner isolation gateway auth",
                    "top_k": 5,
                    "dedup_by_entry": True,
                    "include_chunks": True,
                },
                timeout=10,
            )
            search_two_for_alpha.raise_for_status()
            search_two_json = search_two_for_alpha.json()

            list_one = httpx.get(
                f"{gateway_url}/api/journal/entries",
                headers=headers_one,
                params={"project_slug": "retrival-smoke", "limit": 10},
                timeout=10,
            )
            list_one.raise_for_status()
            list_two = httpx.get(
                f"{gateway_url}/api/journal/entries",
                headers=headers_two,
                params={"project_slug": "retrival-smoke", "limit": 10},
                timeout=10,
            )
            list_two.raise_for_status()
            get_one_as_two = httpx.get(
                f"{gateway_url}/api/journal/entries/{entry_one['entry_id']}",
                headers=headers_two,
                timeout=10,
            )

            mcp_recorded, mcp_found, mcp_recent, mcp_search_text, mcp_recent_text = asyncio.run(
                run_mcp_checks(gateway_url, key_one))

            user_one_hit_ids = [hit["entry"]["entry_id"] for hit in search_one_json["hits"]]
            user_two_alpha_ids = [hit["entry"]["entry_id"] for hit in search_two_json["hits"]]
            list_one_ids = [entry["entry_id"] for entry in list_one.json()["entries"]]
            list_two_ids = [entry["entry_id"] for entry in list_two.json()["entries"]]
            mcp_recorded_id = mcp_recorded.get("entry", {}).get("entry_id")

            assertions = {
                "direct_compute_requires_internal_token": direct_no_token.status_code == 401,
                "direct_compute_requires_gateway_user_header": direct_no_user.status_code == 401,
                "gateway_anonymous_rejected": gateway_anonymous.status_code == 401,
                "gateway_record_user_one_ok": record_one.status_code == 200 and record_one.json()["indexed"] is True,
                "gateway_record_user_two_ok": record_two.status_code == 200 and record_two.json()["indexed"] is True,
                "same_slug_separate_projects": entry_one["project_id"] != entry_two["project_id"],
                "user_one_search_finds_own_entry": entry_one["entry_id"] in user_one_hit_ids,
                "user_two_search_does_not_see_user_one_entry": entry_one["entry_id"] not in user_two_alpha_ids,
                "user_one_list_isolated": entry_one["entry_id"] in list_one_ids and entry_two["entry_id"] not in list_one_ids,
                "user_two_list_isolated": entry_two["entry_id"] in list_two_ids and entry_one["entry_id"] not in list_two_ids,
                "user_two_cannot_get_user_one_entry": get_one_as_two.status_code == 404,
                "mcp_record_journal_over_gateway_ok": bool(mcp_recorded_id and mcp_recorded_id.startswith("journal_entry_")),
                "mcp_search_over_gateway_ok": mcp_recorded_id in [hit["entry"]["entry_id"] for hit in mcp_found.get("hits", [])],
                "mcp_recent_over_gateway_ok": mcp_recorded_id in [entry["entry_id"] for entry in mcp_recent.get("entries", [])],
            }
            failed = [name for name, ok in assertions.items() if not ok]
            return {
                "compute_url": compute_url,
                "gateway_url": gateway_url,
                "assertions": assertions,
                "failed": failed,
                "status_codes": {
                    "direct_no_token": direct_no_token.status_code,
                    "direct_no_user": direct_no_user.status_code,
                    "gateway_anonymous": gateway_anonymous.status_code,
                    "record_one": record_one.status_code,
                    "record_two": record_two.status_code,
                    "get_one_as_two": get_one_as_two.status_code,
                },
                "entry_one": entry_one["entry_id"],
                "entry_two": entry_two["entry_id"],
                "project_ids": [entry_one["project_id"], entry_two["project_id"]],
                "user_one_search_hits": user_one_hit_ids,
                "user_two_alpha_hits": user_two_alpha_ids,
                "user_one_list": list_one_ids,
                "user_two_list": list_two_ids,
                "mcp_recorded": mcp_recorded_id,
                "mcp_search_preview": mcp_search_text.splitlines()[:4],
                "mcp_recent_preview": mcp_recent_text.splitlines()[:4],
            }
        finally:
            gateway_server.should_exit = True
            compute_server.should_exit = True
            gateway_thread.join(timeout=5)
            compute_thread.join(timeout=5)
            try:
                asyncio.run(gateway_app.state.proxy.aclose())
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-port", type=int, default=18190)
    parser.add_argument("--gateway-port", type=int, default=18191)
    args = parser.parse_args()

    summary = run_smoke(args.compute_port, args.gateway_port)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
