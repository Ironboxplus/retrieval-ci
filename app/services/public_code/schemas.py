"""HTTP contracts for public-code discovery and source maintenance."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.services.scope_paths import canonicalize_scope_path, is_windows_scope


PublicCodeSearchMode = Literal["hybrid", "dense", "keyword"]
PublicCodeSearchEffectiveMode = Literal["hybrid", "dense", "keyword", "unavailable"]


class PublicCodeSourceSelection(BaseModel):
    source_ids: list[str] = Field(default_factory=list, max_length=100)
    domains: list[str] = Field(default_factory=list, max_length=50)


class PublicCodeUpdatePolicyRequest(BaseModel):
    update_policy: Literal["manual", "scheduled"]
    update_interval_minutes: int | None = Field(default=None, ge=1, le=43_200)

    @model_validator(mode="after")
    def validate_interval(self):
        if self.update_policy == "manual":
            self.update_interval_minutes = None
        elif self.update_interval_minutes is None:
            raise ValueError("scheduled update policy requires update_interval_minutes")
        return self


class PublicCodeSourceCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(..., min_length=1, max_length=100)
    domain: str = Field(..., min_length=1, max_length=100)
    upstream_url: str = Field(..., min_length=1, max_length=2000)
    relative_path: str = Field(..., min_length=1, max_length=500)
    default_ref: str = Field(default="HEAD", min_length=1, max_length=255)
    proxy_profile: str | None = Field(default=None, max_length=100)
    credential_profile: str | None = Field(default=None, max_length=100)
    language_overrides: dict[str, str] = Field(default_factory=dict)
    update_policy: Literal["manual", "scheduled"] = "manual"
    update_interval_minutes: int | None = Field(default=None, ge=1, le=43_200)
    index_now: bool = True

    @model_validator(mode="after")
    def validate_interval(self):
        if self.update_policy == "manual":
            self.update_interval_minutes = None
        elif self.update_interval_minutes is None:
            raise ValueError("scheduled update policy requires update_interval_minutes")
        return self


class PublicCodeSearchRequestBody(PublicCodeSourceSelection):
    query: str = Field(..., min_length=1, max_length=10_000)
    top_k: int = Field(default=10, ge=1, le=50)
    mode: str = Field(default="hybrid", pattern="^(dense|hybrid|keyword)$")
    language: str | None = Field(default=None, max_length=100)
    rerank: bool = False


class PublicCodeSourceResponse(BaseModel):
    source_id: str
    domain: str
    relative_path: str
    upstream_url: str
    default_ref: str
    proxy_profile: str | None
    credential_profile: str | None
    language_overrides: dict[str, str]
    source_origin: str
    created_by_user_id: str | None
    active_manifest_file_count: int
    active_chunk_count: int
    active_vector_count: int
    active_promoted_at: datetime | None
    created_at: datetime | None
    updated_at: datetime | None
    update_policy: str
    update_interval_minutes: int | None
    active_revision: str | None
    last_mirrored_revision: str | None
    last_remote_revision: str | None
    last_check_error: str | None
    last_checked_at: datetime | None


class PublicCodeSourceCreateResponse(BaseModel):
    source: PublicCodeSourceResponse
    index_job: "PublicCodeJobResponse | None"


class PublicCodeUpdateCheckResponse(BaseModel):
    source_id: str
    domain: str
    local_revision: str | None
    remote_revision: str | None
    status: str
    message: str | None = None


class PublicCodeSearchHitResponse(BaseModel):
    score: float
    retrieval_score: float
    reranker_score: float | None = None
    source_id: str
    revision: str
    domain: str
    path: str
    language: str
    start_line: int
    end_line: int
    label: str
    content: str
    code: str


class PublicCodeSearchResponseBody(BaseModel):
    query: str
    selected_source_ids: list[str]
    hits: list[PublicCodeSearchHitResponse]
    requested_mode: PublicCodeSearchMode
    effective_mode: PublicCodeSearchEffectiveMode
    notice: str | None = Field(
        default=None,
        description="Search diagnostic, including strict keyword unavailability or a dense-only hybrid fallback.",
    )
    rerank_applied: bool
    candidate_count: int = Field(
        ge=0,
        description="Retrieved candidate count before rerank candidate diversification.",
    )
    rerank_candidate_count: int = Field(
        ge=0,
        description="Candidate count supplied to the reranker after duplicate and file diversity control.",
    )


class PublicCodeJobResponse(BaseModel):
    job_id: str
    operation: str
    source_ids: list[str]
    total_sources: int
    status: str
    completed_sources: int
    progress: float
    phase: str
    current_source_id: str | None
    current_path: str | None
    files_done: int
    files_total: int
    chunks_done: int
    message: str | None
    source_progress: dict[str, dict[str, Any]]
    results: list[dict[str, Any]]
    errors: dict[str, str]
    created_at: float
    started_at: float | None
    finished_at: float | None


class PublicCodeImportCreateRequest(BaseModel):
    """Declare a git bundle that will seed one source's bare mirror."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(..., min_length=1, max_length=100)
    domain: str = Field(..., min_length=1, max_length=100)
    upstream_url: str = Field(
        ..., min_length=1, max_length=2000,
        description="The public upstream the bundle was cloned from; becomes the mirror origin.",
    )
    relative_path: str = Field(..., min_length=1, max_length=500)
    default_ref: str = Field(default="HEAD", min_length=1, max_length=255)
    head_commit: str = Field(
        ..., min_length=40, max_length=40,
        description="Commit that default_ref must name in the bundle and, after import, in the mirror.",
    )
    size_bytes: int = Field(..., ge=1, description="Exact bundle size; uploads beyond it are rejected.")
    sha256: str = Field(..., min_length=64, max_length=64, description="SHA-256 of the whole bundle.")
    language_overrides: dict[str, str] = Field(default_factory=dict)
    update_policy: Literal["manual", "scheduled"] = "manual"
    update_interval_minutes: int | None = Field(default=None, ge=1, le=43_200)
    private_project: str | None = Field(
        default=None, min_length=1, max_length=2000,
        description="Absolute private Code KB scope of the importing admin; exact chunk vectors are reused.",
    )

    @field_validator("private_project")
    @classmethod
    def _canonical_private_project(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not is_windows_scope(value) and not value.startswith("/"):
            raise ValueError("private_project must be an absolute index scope")
        return canonicalize_scope_path(value)


class PublicCodeImportResponse(BaseModel):
    import_id: str
    source_id: str
    domain: str
    relative_path: str
    upstream_url: str
    default_ref: str
    head_commit: str
    size_bytes: int
    sha256: str
    received_bytes: int = Field(description="Bytes stored so far; the next upload offset.")
    status: Literal["uploading", "submitted", "published", "failed"]
    source_exists: bool = Field(
        description="True when the import updates an already registered source.")
    update_policy: str
    update_interval_minutes: int | None
    job_id: str | None
    error: str | None
    created_at: float
    updated_at: float
    private_project: str | None = None


class PublicCodeImportCompleteResponse(BaseModel):
    import_: PublicCodeImportResponse = Field(alias="import")
    job: PublicCodeJobResponse

    model_config = ConfigDict(populate_by_name=True)
