"""Pydantic contracts for the index-inventory catalog."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.services.scope_paths import canonicalize_scope_path, normalize_relative_path


class IndexFileRecord(BaseModel):
    """One indexed file, reported by the client that owns the code."""
    rel_path: str = Field(..., min_length=1, max_length=1000)
    kind: str = Field(default="code", min_length=1, max_length=50)
    collection: str | None = Field(default=None, max_length=100)
    language: str | None = Field(default=None, max_length=50)
    chunk_count: int = Field(default=0, ge=0)
    content_hash: str | None = Field(default=None, max_length=64)
    size_bytes: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("rel_path")
    @classmethod
    def _normalise_rel_path(cls, value: str) -> str:
        return normalize_relative_path(value)


class IndexReportRequest(BaseModel):
    host: str = Field(..., min_length=1, max_length=200)
    project: str = Field(..., min_length=1, max_length=2000,
                         description="Scope = indexed dir abspath; wide enough for a full path.")
    root_dir: str | None = Field(default=None, max_length=1000)
    files: list[IndexFileRecord] = Field(default_factory=list)
    # When true, files under (owner, host, project) that are absent from this report
    # are soft-deleted — a full-sync snapshot. When false, this is an incremental upsert.
    replace: bool = True
    owner_user_id: str | None = Field(default=None, max_length=200)
    # Replace-sweep envelope (W169b): restrict the soft-delete to catalog rows this run
    # could actually have (re)indexed, so a "*.txt" doc report doesn't wipe the "*.md"
    # rows, and a shallow re-run doesn't wipe deeper rows. None on either axis = unbounded
    # (code reports pass neither, so they full-sync the whole kind exactly as before).
    patterns: list[str] | None = Field(
        default=None,
        description="Basename globs this report covered; only matching absent rows are swept. "
                    "None or ['*'] = every basename.",
    )
    max_depth: int | None = Field(
        default=None, ge=0,
        description="Directory-depth (rel_path slash count) ceiling this report scanned; "
                    "deeper absent rows are kept. None = unbounded.",
    )

    @field_validator("project")
    @classmethod
    def _canonical_project(cls, value: str) -> str:
        return canonicalize_scope_path(value)

    @field_validator("root_dir")
    @classmethod
    def _canonical_root_dir(cls, value: str | None) -> str | None:
        return canonicalize_scope_path(value) if value else value


class IndexReportResponse(BaseModel):
    host: str
    project: str
    owner_user_id: str
    upserted: int
    removed: int
    total_files: int


class IndexScopePolicyUpsert(BaseModel):
    host: str = Field(..., min_length=1, max_length=200)
    project: str = Field(..., min_length=1, max_length=2000)
    kind: str = Field(default="code", min_length=1, max_length=50)
    root_dir: str = Field(..., min_length=1, max_length=2000)
    auto_refresh: bool = True
    respect_gitignore: bool = True
    exclude_globs: list[str] = Field(default_factory=list, max_length=200)
    min_check_interval_s: int = Field(default=10, ge=0, le=3600)

    @field_validator("project", "root_dir")
    @classmethod
    def _canonical_scope_fields(cls, value: str) -> str:
        return canonicalize_scope_path(value)


class InventoryFileNode(BaseModel):
    rel_path: str
    kind: str
    collection: str | None = None
    language: str | None = None
    chunk_count: int = 0
    content_hash: str | None = None
    size_bytes: int | None = None
    indexed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InventoryDirNode(BaseModel):
    dir: str
    file_count: int = 0
    chunk_count: int = 0
    files: list[InventoryFileNode] = Field(default_factory=list)


class InventoryProjectNode(BaseModel):
    project: str
    host: str
    root_dir: str | None = None
    file_count: int = 0
    chunk_count: int = 0
    # Server-computed freshness so the SUMMARY view (dirs omitted) can still show "last
    # indexed @commit" without shipping the whole per-file tree (which ballooned to ~10MB).
    last_indexed_at: datetime | None = None
    last_commit: str | None = None
    # Code policies are registered before client scan/upload; code inventory rows appear
    # only after a successful backend job.  Keep the two facts distinct so a first index
    # cannot be read as a completed snapshot merely because its scope is known.
    code_snapshot_state: Literal["registered", "preparing", "completed"] | None = Field(
        default=None,
        description="Code scope visibility: registered policy, active preparation, or completed snapshot.")
    code_active_index_job_id: str | None = Field(
        default=None,
        description="Code KB job whose visibility marker currently blocks a completed result.")
    code_previous_completed_snapshot_available: bool | None = Field(
        default=None,
        description="Whether a preparing Code KB scope has a previously completed snapshot.")
    code_recovery_required: bool | None = Field(
        default=None,
        description="True when an interrupted in-place Code KB mutation requires a full recovery index.")
    dirs: list[InventoryDirNode] = Field(default_factory=list)


class InventoryHostNode(BaseModel):
    host: str
    file_count: int = 0
    chunk_count: int = 0
    projects: list[InventoryProjectNode] = Field(default_factory=list)


class InventoryUserNode(BaseModel):
    owner_user_id: str
    file_count: int = 0
    chunk_count: int = 0
    hosts: list[InventoryHostNode] = Field(default_factory=list)


class InventoryResponse(BaseModel):
    users: list[InventoryUserNode] = Field(default_factory=list)
    totals: dict[str, int] = Field(default_factory=dict)


class ScopeConflict(BaseModel):
    """One existing indexed scope that overlaps a candidate scope (parent or child)."""
    project: str
    host: str


class ScopeOverlapResponse(BaseModel):
    """Result of the nested-scope overlap check for a candidate index scope.

    ``ancestor_blocks``: existing parent scopes that ALREADY indexed a file under the
    candidate → indexing the candidate child adds no coverage (the guard refuses it).
    ``child_warnings``: existing child scopes under the candidate → indexing the parent
    proceeds but warns (it can add files outside the child).
    ``ancestor_widenings``: parent scopes that overlap the candidate but were indexed
    with respect_gitignore ON while the candidate turns it OFF → the child reaches files
    the parent provably never indexed, so it proceeds with a warning instead of a block.
    All three are KB-scale conflict lists computed server-side, replacing a per-call
    multi-MB inventory download."""
    ancestor_blocks: list[ScopeConflict] = Field(default_factory=list)
    child_warnings: list[ScopeConflict] = Field(default_factory=list)
    ancestor_widenings: list[ScopeConflict] = Field(default_factory=list)


class IndexOwnerSummary(BaseModel):
    owner_user_id: str
    file_count: int = 0
    chunk_count: int = 0
    host_count: int = 0
    project_count: int = 0
