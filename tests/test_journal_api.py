"""TDD for ACE journal FastAPI routes."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.journal.schemas import (
    JournalDeleteResponse,
    JournalDocScopeDeleteResponse,
    JournalEntryListResponse,
    JournalEntryResponse,
    JournalRecordResponse,
    JournalSearchResponse,
)


class FakeJournalService:
    def __init__(self):
        self.record_request = None
        self.search_request = None
        self.list_kwargs = None
        self.get_args = None
        self.delete_args = None
        self.reconcile_request = None

    def record(self, request):
        self.record_request = request
        entry = JournalEntryResponse(
            entry_id="journal_entry_test",
            project_id=1,
            project_slug=request.project_slug,
            client=request.client,
            source_kind=request.source_kind,
            title=request.title,
            content_hash="abc",
            chunk_count=1,
            indexed=True,
        )
        return JournalRecordResponse(entry=entry, chunk_count=1, indexed=True)

    def list_entries(self, **kwargs):
        self.list_kwargs = kwargs
        return JournalEntryListResponse(entries=[], total=0)

    def get_entry(self, entry_id, owner_user_id=None):
        self.get_args = (entry_id, owner_user_id)
        if entry_id == "missing":
            return None
        return JournalEntryResponse(
            entry_id=entry_id,
            project_id=1,
            project_slug="retrival",
            client="codex",
            source_kind="memory",
            content_hash="abc",
        )

    def delete_entry(self, entry_id, reason=None, owner_user_id=None):
        self.delete_args = (entry_id, reason, owner_user_id)
        return JournalDeleteResponse(entry_id=entry_id, deleted=entry_id != "missing", deleted_vectors=1)

    def search(self, request):
        self.search_request = request
        return JournalSearchResponse(
            query=request.query,
            project_slug=request.project_slug,
            mode=request.mode,
            total=0,
            hits=[],
        )

    def delete_doc_scope(self, *, project_slug, owner_user_id=None):
        self.delete_scope_args = (project_slug, owner_user_id)
        return JournalDocScopeDeleteResponse(
            project_slug=project_slug, removed_entries=3,
            removed_entry_ids=["a", "b", "c"], deleted_vectors=9, catalog_removed=0)

    def reconcile_doc_scope(self, *, project_slug, live_titles, owner_user_id=None,
                            max_depth=None, patterns=None):
        self.reconcile_request = {
            "project_slug": project_slug,
            "live_titles": live_titles,
            "owner_user_id": owner_user_id,
            "max_depth": max_depth,
            "patterns": patterns,
        }
        return {
            "project_slug": project_slug,
            "live_titles": len(live_titles),
            "removed_entries": 2,
            "removed_entry_ids": ["journal_entry_stale_a", "journal_entry_stale_b"],
            "deleted_vectors": 6,
            "skipped": False,
            "reason": None,
        }


def _client(fake):
    from app.api.routes import journal as journal_routes

    app = FastAPI()
    app.include_router(journal_routes.router, prefix="/api/journal")
    app.dependency_overrides = {}
    journal_routes.get_journal_service = lambda: fake
    return TestClient(app)


def test_post_entries_delegates_to_service():
    fake = FakeJournalService()
    client = _client(fake)

    response = client.post("/api/journal/entries", json={
        "project_slug": "retrival",
        "client": "codex",
        "source_kind": "memory",
        "title": "Work log",
        "body": "Detailed work memory.",
        "tags": ["ace"],
    }, headers={
        "X-Retrival-User-Id": "user-7",
        "X-Retrival-Username": "alice",
        "X-Retrival-User-Tier": "internal",
    })

    assert response.status_code == 200
    assert response.json()["entry"]["entry_id"] == "journal_entry_test"
    assert fake.record_request.project_slug == "retrival"
    assert fake.record_request.tags == ["ace"]
    assert fake.record_request.owner_user_id == "user-7"
    assert fake.record_request.metadata["gateway_username"] == "alice"
    assert fake.record_request.metadata["gateway_tier"] == "internal"


def test_search_delegates_query_contract_to_service():
    fake = FakeJournalService()
    client = _client(fake)

    response = client.post("/api/journal/search", json={
        "project_slug": "retrival",
        "query": "journal storage",
        "mode": "hybrid",
        "dedup_by_entry": False,
    }, headers={"X-Retrival-User-Id": "user-7"})

    assert response.status_code == 200
    assert fake.search_request.query == "journal storage"
    assert fake.search_request.dedup_by_entry is False
    assert fake.search_request.owner_user_id == "user-7"


def test_journal_api_rejects_drive_relative_windows_scope():
    client = _client(FakeJournalService())
    response = client.post("/api/journal/search", json={
        "project_slug": "E:repo",
        "query": "scope validation",
        "mode": "keyword",
    }, headers={"X-Retrival-User-Id": "user-7"})

    assert response.status_code == 422
    assert "Windows scope must be absolute" in response.text


def test_windows_journal_contracts_canonicalize_scope_and_relative_paths():
    fake = FakeJournalService()
    client = _client(fake)
    headers = {"X-Retrival-User-Id": "user-7"}

    record = client.post("/api/journal/entries", json={
        "project_slug": r"e:\JETB\Repo\.\Docs",
        "client": "mcp",
        "source_kind": "doc",
        "title": r"Guides\Compact.md",
        "body": "compact guide",
    }, headers=headers)
    search = client.post("/api/journal/search", json={
        "project_slug": "E:/JETB/REPO/DOCS",
        "query": "compact",
        "mode": "keyword",
        "path_glob": r"GUIDES\*.MD",
    }, headers=headers)
    listed = client.get("/api/journal/entries", params={
        "project_slug": r"e:\JETB\Repo\Docs",
    }, headers=headers)

    assert record.status_code == search.status_code == listed.status_code == 200
    assert fake.record_request.project_slug == "E:/jetb/repo/docs"
    assert fake.record_request.title == "Guides/Compact.md"
    assert fake.search_request.project_slug == "E:/jetb/repo/docs"
    assert fake.search_request.path_glob == "GUIDES/*.MD"
    assert fake.list_kwargs["project_slug"] == "E:/jetb/repo/docs"


def test_doc_reconcile_route_passes_gateway_identity_to_service():
    fake = FakeJournalService()
    client = _client(fake)

    response = client.post("/api/journal/docs/reconcile", json={
        "project_slug": "retrival",
        "live_titles": ["README.md", "docs/setup.md"],
    }, headers={"X-Retrival-User-Id": "user-7"})

    assert response.status_code == 200
    assert response.json()["removed_entries"] == 2
    assert fake.reconcile_request == {
        "project_slug": "retrival",
        "live_titles": ["README.md", "docs/setup.md"],
        "owner_user_id": "user-7",
        "max_depth": None,
        "patterns": None,
    }


def test_windows_doc_reconcile_normalizes_scope_and_live_titles():
    fake = FakeJournalService()
    client = _client(fake)

    response = client.post("/api/journal/docs/reconcile", json={
        "project_slug": r"e:\JETB\Repo",
        "live_titles": [r"Docs\A.md", ".\\README.md"],
    }, headers={"X-Retrival-User-Id": "user-7"})

    assert response.status_code == 200
    assert fake.reconcile_request["project_slug"] == "E:/jetb/repo"
    assert fake.reconcile_request["live_titles"] == ["Docs/A.md", "README.md"]


def test_delete_docs_route_clears_scope_and_catalog():
    from app.api.routes import journal as journal_routes

    fake = FakeJournalService()
    client = _client(fake)

    class _FakeCatalog:
        def __init__(self):
            self.args = None

        def delete_project(self, *, owner_user_id, project, kind=None):
            self.args = (owner_user_id, project, kind)
            return 3

    catalog = _FakeCatalog()
    journal_routes.get_index_catalog_service = lambda: catalog

    response = client.request("DELETE", "/api/journal/docs",
                              params={"project_slug": "/abs/scope"},
                              headers={"X-Retrival-User-Id": "user-7"})

    assert response.status_code == 200
    body = response.json()
    assert body["removed_entries"] == 3
    assert body["deleted_vectors"] == 9
    assert body["catalog_removed"] == 3          # filled from the catalog service
    assert fake.delete_scope_args == ("/abs/scope", "user-7")
    assert catalog.args == ("user-7", "/abs/scope", "doc")  # doc-only, keeps code + memory


def test_delete_docs_route_canonicalizes_windows_scope():
    from app.api.routes import journal as journal_routes

    fake = FakeJournalService()
    client = _client(fake)

    class _FakeCatalog:
        def equivalent_project_scopes(self, **_kwargs):
            return ["E:/jetb/repo"]

        def delete_project(self, *, owner_user_id, project, kind=None):
            assert (owner_user_id, project, kind) == (
                "user-7", "E:/jetb/repo", "doc")
            return 0

    journal_routes.get_index_catalog_service = lambda: _FakeCatalog()
    response = client.request(
        "DELETE", "/api/journal/docs",
        params={"project_slug": r"e:\JETB\Repo"},
        headers={"X-Retrival-User-Id": "user-7"},
    )

    assert response.status_code == 200
    assert fake.delete_scope_args == ("E:/jetb/repo", "user-7")


def test_delete_docs_route_removes_all_proven_legacy_catalog_scopes():
    from app.api.routes import journal as journal_routes

    fake = FakeJournalService()
    client = _client(fake)
    canonical = "E:/jetb/repo"
    legacy = "E:/JETB/Repo"

    class _FakeCatalog:
        def __init__(self):
            self.deleted = []

        def equivalent_project_scopes(self, **_kwargs):
            return [canonical, legacy]

        def delete_project(self, *, owner_user_id, project, kind=None):
            self.deleted.append((owner_user_id, project, kind))
            return 1

    catalog = _FakeCatalog()
    journal_routes.get_index_catalog_service = lambda: catalog
    response = client.request(
        "DELETE", "/api/journal/docs",
        params={"project_slug": r"e:\JETB\Repo"},
        headers={"X-Retrival-User-Id": "user-7"},
    )

    assert response.status_code == 200
    assert response.json()["catalog_removed"] == 2
    assert response.json()["scope_aliases"] == [legacy]
    assert catalog.deleted == [
        ("user-7", canonical, "doc"),
        ("user-7", legacy, "doc"),
    ]


def test_read_routes_pass_gateway_identity_to_service():
    fake = FakeJournalService()
    client = _client(fake)

    list_response = client.get(
        "/api/journal/entries?project_slug=retrival",
        headers={"X-Retrival-User-Id": "user-7"},
    )
    get_response = client.get(
        "/api/journal/entries/journal_entry_test",
        headers={"X-Retrival-User-Id": "user-7"},
    )
    delete_response = client.delete(
        "/api/journal/entries/journal_entry_test?reason=cleanup",
        headers={"X-Retrival-User-Id": "user-7"},
    )

    assert list_response.status_code == 200
    assert get_response.status_code == 200
    assert delete_response.status_code == 200
    assert fake.list_kwargs["owner_user_id"] == "user-7"
    assert fake.get_args == ("journal_entry_test", "user-7")
    assert fake.delete_args == ("journal_entry_test", "cleanup", "user-7")


def test_compute_journal_write_requires_gateway_identity_header():
    fake = FakeJournalService()
    client = _client(fake)

    response = client.post("/api/journal/entries", json={
        "project_slug": "retrival",
        "client": "codex",
        "source_kind": "memory",
        "body": "No gateway identity.",
    })

    assert response.status_code == 401


def test_missing_entry_returns_404():
    client = _client(FakeJournalService())

    response = client.get(
        "/api/journal/entries/missing",
        headers={"X-Retrival-User-Id": "user-7"},
    )

    assert response.status_code == 404
