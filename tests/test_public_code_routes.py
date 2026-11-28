from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@dataclass(frozen=True)
class _SourceState:
    source_id: str
    domain: str
    relative_path: str = "multimedia/ffmpeg"
    upstream_url: str = "https://example.invalid/ffmpeg.git"
    default_ref: str = "HEAD"
    proxy_profile: str | None = None
    credential_profile: str | None = None
    language_overrides: dict[str, str] | None = None
    source_origin: str = "bootstrap"
    created_by_user_id: str | None = None
    active_manifest_file_count: int = 0
    active_chunk_count: int = 0
    active_vector_count: int = 0
    active_promoted_at: object | None = None
    created_at: object | None = None
    updated_at: object | None = None
    update_policy: str = "manual"
    update_interval_minutes: int | None = None
    active_revision: str | None = None
    last_mirrored_revision: str | None = None
    last_remote_revision: str | None = None
    last_check_error: str | None = None
    last_checked_at: object | None = None

    def __post_init__(self):
        if self.language_overrides is None:
            object.__setattr__(self, "language_overrides", {})


@dataclass(frozen=True)
class _UpdateCheck:
    source_id: str
    domain: str = "multimedia"
    local_revision: str | None = "abc123"
    remote_revision: str | None = "abc123"
    status: str = "up_to_date"
    message: str | None = None


@dataclass(frozen=True)
class _SearchResult:
    query: str
    selected_source_ids: tuple[str, ...]
    hits: tuple[object, ...] = ()
    requested_mode: str = "hybrid"
    effective_mode: str = "hybrid"
    notice: str | None = None
    rerank_applied: bool = False
    candidate_count: int = 0
    rerank_candidate_count: int = 0


@dataclass(frozen=True)
class _Job:
    job_id: str = "job-public-code"
    operation: str = "check_updates"
    source_ids: tuple[str, ...] = ("ffmpeg",)
    total_sources: int = 1
    status: str = "queued"
    completed_sources: int = 0
    results: tuple[dict, ...] = ()
    errors: dict[str, str] | None = None
    phase: str = "queued"
    current_source_id: str | None = None
    current_path: str | None = None
    files_done: int = 0
    files_total: int = 0
    chunks_done: int = 0
    message: str | None = None
    source_progress: dict[str, dict] | None = None
    created_at: float = 1.0
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def progress(self):
        return self.completed_sources / self.total_sources

    def __post_init__(self):
        if self.source_progress is None:
            object.__setattr__(self, "source_progress", {})


class _ControlService:
    def list_sources(self):
        return [_SourceState(source_id="ffmpeg", domain="multimedia", active_revision="abc123")]

    def check_updates(self, *, source_ids, domains):
        return [_UpdateCheck(source_id="ffmpeg", status="up_to_date")]

    def set_update_policy(self, *, source_id, update_policy, update_interval_minutes):
        return _SourceState(
            source_id=source_id,
            domain="multimedia",
            update_policy=update_policy,
            update_interval_minutes=update_interval_minutes,
        )

    def register_source(
        self,
        source,
        *,
        update_policy,
        update_interval_minutes,
        created_by_user_id,
    ):
        return _SourceState(
            source_id=source.source_id,
            domain=source.domain,
            relative_path=source.relative_path,
            upstream_url=source.upstream_url,
            default_ref=source.default_ref,
            proxy_profile=source.proxy_profile,
            credential_profile=source.credential_profile,
            language_overrides=dict(source.language_overrides),
            source_origin="api",
            created_by_user_id=created_by_user_id,
        )


class _SearchService:
    def search(self, request):
        return _SearchResult(query=request.query, selected_source_ids=request.source_ids)


class _JobService:
    def submit_update_check(self, *, source_ids, domains):
        return _Job(operation="check_updates", source_ids=tuple(source_ids) or ("ffmpeg",))

    def submit_index(self, *, source_ids, domains):
        return _Job(operation="index", source_ids=tuple(source_ids) or ("ffmpeg",))

    def get(self, job_id):
        return _Job(job_id=job_id, status="completed", completed_sources=1)

    def list_recent(self, *, limit):
        return (_Job(job_id="recent-job", status="completed", completed_sources=1),)


@pytest.fixture
def route_module(monkeypatch):
    from app.api.routes import public_code

    monkeypatch.setattr(public_code, "get_public_code_control_service", lambda: _ControlService())
    monkeypatch.setattr(public_code, "get_public_code_search_service", lambda: _SearchService())
    monkeypatch.setattr(public_code, "get_public_code_job_service", lambda: _JobService())
    return public_code


def _client(route_module) -> TestClient:
    app = FastAPI()
    app.include_router(route_module.router, prefix="/api/public-code")
    return TestClient(app)


def test_public_code_allows_gateway_admin(route_module):
    headers = {"X-Retrival-User-Id": "gateway-admin", "X-Retrival-User-Tier": "admin"}
    client = _client(route_module)

    sources = client.get("/api/public-code/sources", headers=headers)
    search = client.post(
        "/api/public-code/search",
        headers=headers,
        json={"query": "packet decoder", "source_ids": ["ffmpeg"]},
    )

    assert sources.status_code == 200
    assert search.status_code == 200


def test_gateway_admin_can_register_a_source_and_start_indexing(route_module):
    headers = {"X-Retrival-User-Id": "gateway-admin", "X-Retrival-User-Tier": "admin"}

    response = _client(route_module).post(
        "/api/public-code/sources",
        headers=headers,
        json={
            "source_id": "pion-webrtc",
            "domain": "multimedia",
            "upstream_url": "https://github.com/pion/webrtc.git",
            "relative_path": "multimedia/pion-webrtc",
            "default_ref": "main",
            "proxy_profile": "local-proxy",
            "credential_profile": "github-read",
            "language_overrides": {".inc": "c"},
            "update_policy": "manual",
            "index_now": True,
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["source"]["source_id"] == "pion-webrtc"
    assert body["source"]["proxy_profile"] == "local-proxy"
    assert body["index_job"]["operation"] == "index"
    assert body["index_job"]["source_ids"] == ["pion-webrtc"]


def test_source_registration_rejects_non_admin_and_raw_credentials(route_module):
    client = _client(route_module)
    valid_payload = {
        "source_id": "pion-webrtc",
        "domain": "multimedia",
        "upstream_url": "https://github.com/pion/webrtc.git",
        "relative_path": "multimedia/pion-webrtc",
    }

    forbidden = client.post(
        "/api/public-code/sources",
        headers={"X-Retrival-User-Id": "gateway-member", "X-Retrival-User-Tier": "member"},
        json=valid_payload,
    )
    raw_secret = client.post(
        "/api/public-code/sources",
        headers={"X-Retrival-User-Id": "gateway-admin", "X-Retrival-User-Tier": "admin"},
        json={**valid_payload, "cookie": "must-not-be-accepted"},
    )

    assert forbidden.status_code == 403
    assert raw_secret.status_code == 422
    assert "cookie" in raw_secret.text


def test_source_registration_maps_domain_validation_errors_to_422(route_module):
    response = _client(route_module).post(
        "/api/public-code/sources",
        headers={"X-Retrival-User-Id": "gateway-admin", "X-Retrival-User-Tier": "admin"},
        json={
            "source_id": "invalid-url",
            "domain": "fixture",
            "upstream_url": "https://example.invalid/repository.git?forbidden=yes",
            "relative_path": "fixture/invalid-url",
            "index_now": False,
        },
    )

    assert response.status_code == 422
    assert "query parameters" in response.json()["detail"]


def test_admin_can_enable_scheduled_updates_with_a_bounded_interval(route_module):
    response = _client(route_module).put(
        "/api/public-code/sources/ffmpeg/update-policy",
        headers={"X-Retrival-User-Id": "gateway-admin", "X-Retrival-User-Tier": "admin"},
        json={"update_policy": "scheduled", "update_interval_minutes": 60},
    )

    assert response.status_code == 200
    assert response.json()["update_policy"] == "scheduled"
    assert response.json()["update_interval_minutes"] == 60


@pytest.mark.parametrize("tier", ["paying", "member", "internal"])
def test_public_code_rejects_non_admin_gateway_users(route_module, tier):
    headers = {"X-Retrival-User-Id": "gateway-user", "X-Retrival-User-Tier": tier}
    client = _client(route_module)

    sources = client.get("/api/public-code/sources", headers=headers)
    search = client.post(
        "/api/public-code/search",
        headers=headers,
        json={"query": "packet decoder", "source_ids": ["ffmpeg"]},
    )

    assert sources.status_code == 403
    assert search.status_code == 403


def test_internal_public_code_routes_do_not_require_gateway_headers(route_module, monkeypatch):
    monkeypatch.setattr(route_module, "resolve_admin", lambda *_args: "internal-admin")
    client = _client(route_module)

    sources = client.get("/api/public-code/sources")
    checks = client.post("/api/public-code/sources/check-updates", json={"source_ids": ["ffmpeg"]})
    indexing = client.post("/api/public-code/sources/index", json={"source_ids": ["ffmpeg"]})
    job_status = client.get("/api/public-code/jobs/job-public-code")
    policy = client.put(
        "/api/public-code/sources/ffmpeg/update-policy",
        json={"update_policy": "manual", "update_interval_minutes": None},
    )
    search = client.post(
        "/api/public-code/search",
        json={"query": "packet decoder", "source_ids": ["ffmpeg"]},
    )

    assert sources.status_code == 200
    assert sources.json()[0]["source_id"] == "ffmpeg"
    assert checks.status_code == 202
    assert checks.json()["operation"] == "check_updates"
    assert indexing.status_code == 202
    assert indexing.json()["operation"] == "index"
    assert job_status.status_code == 200
    assert job_status.json()["status"] == "completed"
    assert policy.status_code == 200
    assert policy.json()["source_id"] == "ffmpeg"
    assert search.status_code == 200
    assert search.json()["selected_source_ids"] == ["ffmpeg"]


def test_invalid_scheduled_policy_is_rejected_before_service_work(route_module, monkeypatch):
    monkeypatch.setattr(route_module, "resolve_admin", lambda *_args: "internal-admin")
    response = _client(route_module).put(
        "/api/public-code/sources/ffmpeg/update-policy",
        json={"update_policy": "scheduled", "update_interval_minutes": None},
    )

    assert response.status_code == 422


def test_unknown_search_scope_is_a_client_error(route_module, monkeypatch):
    from app.services.public_code.source_catalog import PublicSourceSelectionError

    class FailingSearchService:
        def search(self, _request):
            raise PublicSourceSelectionError("Unknown public-code source IDs: missing")

    monkeypatch.setattr(route_module, "resolve_admin", lambda *_args: "internal-admin")
    monkeypatch.setattr(route_module, "get_public_code_search_service", lambda: FailingSearchService())
    app = FastAPI()
    app.include_router(route_module.router, prefix="/api/public-code")

    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/public-code/search",
        json={"query": "x", "source_ids": ["missing"]},
    )

    assert response.status_code == 422
    assert "Unknown public-code source IDs" in response.json()["detail"]


def test_public_code_openapi_has_concrete_success_models(route_module):
    app = FastAPI()
    app.include_router(route_module.router, prefix="/api/public-code")
    paths = app.openapi()["paths"]

    for path, method in (
        ("/api/public-code/sources", "get"),
        ("/api/public-code/sources", "post"),
        ("/api/public-code/sources/check-updates", "post"),
        ("/api/public-code/sources/index", "post"),
        ("/api/public-code/jobs", "get"),
        ("/api/public-code/jobs/{job_id}", "get"),
        ("/api/public-code/sources/{source_id}/update-policy", "put"),
        ("/api/public-code/search", "post"),
    ):
        if method == "post" and path == "/api/public-code/sources":
            success_status = "201"
        elif method == "post" and path.endswith(("check-updates", "index")):
            success_status = "202"
        else:
            success_status = "200"
        schema = paths[path][method]["responses"][success_status]["content"]["application/json"]["schema"]
        assert schema not in ({}, None)


def test_public_code_search_openapi_exposes_retrieval_and_reranker_scores(route_module):
    app = FastAPI()
    app.include_router(route_module.router, prefix="/api/public-code")
    hit_schema = app.openapi()["components"]["schemas"]["PublicCodeSearchHitResponse"]

    assert {"score", "retrieval_score", "reranker_score"} <= set(hit_schema["properties"])
    assert "retrieval_score" in hit_schema["required"]

    response_schema = app.openapi()["components"]["schemas"]["PublicCodeSearchResponseBody"]
    assert {
        "requested_mode",
        "effective_mode",
        "notice",
        "rerank_applied",
        "candidate_count",
        "rerank_candidate_count",
    } <= set(response_schema["properties"])


def test_public_code_search_serializes_retrieval_and_reranker_scores(route_module, monkeypatch):
    @dataclass(frozen=True)
    class _ScoredHit:
        score: float = 0.91
        retrieval_score: float = 0.42
        reranker_score: float | None = 0.91
        source_id: str = "ffmpeg"
        revision: str = "abc123"
        domain: str = "multimedia"
        path: str = "libavcodec/encode.c"
        language: str = "c"
        start_line: int = 10
        end_line: int = 20
        label: str = "encode"
        content: str = "# path: libavcodec/encode.c\nint encode(void);"
        code: str = "int encode(void);"

    class _ScoredSearchService:
        def search(self, request):
            return _SearchResult(
                query=request.query,
                selected_source_ids=request.source_ids,
                hits=(_ScoredHit(),),
                requested_mode="hybrid",
                effective_mode="hybrid",
                rerank_applied=True,
                candidate_count=20,
                rerank_candidate_count=12,
            )

    monkeypatch.setattr(route_module, "resolve_admin", lambda *_args: "internal-admin")
    monkeypatch.setattr(route_module, "get_public_code_search_service", lambda: _ScoredSearchService())

    response = _client(route_module).post(
        "/api/public-code/search",
        json={"query": "encode", "source_ids": ["ffmpeg"], "rerank": True},
    )

    assert response.status_code == 200
    hit = response.json()["hits"][0]
    assert hit["score"] == 0.91
    assert hit["retrieval_score"] == 0.42
    assert hit["reranker_score"] == 0.91
    assert response.json()["requested_mode"] == "hybrid"
    assert response.json()["effective_mode"] == "hybrid"
    assert response.json()["notice"] is None
    assert response.json()["rerank_applied"] is True
    assert response.json()["candidate_count"] == 20
    assert response.json()["rerank_candidate_count"] == 12


def test_unknown_public_code_job_returns_404(route_module, monkeypatch):
    class MissingJobService(_JobService):
        def get(self, _job_id):
            return None

    monkeypatch.setattr(route_module, "resolve_admin", lambda *_args: "internal-admin")
    monkeypatch.setattr(route_module, "get_public_code_job_service", lambda: MissingJobService())

    response = _client(route_module).get("/api/public-code/jobs/missing")

    assert response.status_code == 404


def test_recent_public_code_jobs_have_a_pollable_list_endpoint(route_module, monkeypatch):
    monkeypatch.setattr(route_module, "resolve_admin", lambda *_args: "internal-admin")

    response = _client(route_module).get("/api/public-code/jobs")

    assert response.status_code == 200
    assert response.json()[0]["job_id"] == "recent-job"
