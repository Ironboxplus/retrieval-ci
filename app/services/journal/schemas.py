"""Pydantic contracts for ACE journal CRUD and search."""
from __future__ import annotations

from datetime import datetime
import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.services.scope_paths import canonicalize_scope_path, normalize_relative_path


JournalSearchMode = Literal["dense", "sparse", "hybrid", "keyword"]


class JournalRecordRequest(BaseModel):
    project_slug: str = Field(
        ..., min_length=1, max_length=2000,
        description="项目命名空间 = 当前工作目录绝对路径（MCP 客户端强制）；同一 slug 下的记忆"
                    "互相可检索，跨 slug 隔离。够长以容纳绝对路径。",
    )
    client: str = Field(default="http", min_length=1, max_length=100,
                        description="来源客户端标识，如 http/mcp/codex/claude。")
    source_kind: str = Field(default="manual_note", min_length=1, max_length=100,
                             description="条目类型，如 memory/session_summary/handoff/manual_note。")
    title: str | None = Field(default=None, max_length=500)
    summary: str | None = None
    body: str = Field(..., min_length=1, description="要长期记住的工作记忆正文（Markdown 可）。")
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    workspace_hint: str | None = Field(default=None, max_length=1000)
    owner_user_id: str | None = Field(default=None, max_length=200)
    dedupe_policy: Literal["append", "return_existing"] = "append"

    @field_validator("project_slug")
    @classmethod
    def _canonical_project_slug(cls, value: str) -> str:
        return canonicalize_scope_path(value)

    @model_validator(mode="after")
    def _canonical_doc_title(self) -> "JournalRecordRequest":
        if self.source_kind == "doc" and self.title:
            self.title = normalize_relative_path(self.title)
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "summary": "记录一段排障结论",
                    "value": {
                        "project_slug": "C:/Users/Arc/Prd/retrival",
                        "source_kind": "memory",
                        "title": "RRF 融合 off-by-one 修复",
                        "body": "pipeline.py 的 RRF 排名从 0 起算导致首位权重偏高；改为 1-based 后 test_pipeline_rrf 通过。",
                        "tags": ["bugfix", "retrieval"],
                    },
                }
            ]
        }
    )


class JournalChunkResponse(BaseModel):
    chunk_id: str
    entry_id: str
    project_id: int
    chunk_index: int
    level: int
    heading: str | None = None
    content: str
    token_count: int | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class JournalEntryResponse(BaseModel):
    entry_id: str
    project_id: int
    project_slug: str
    client: str
    source_kind: str
    title: str | None = None
    summary: str | None = None
    body: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_hash: str
    chunk_count: int = 0
    indexed: bool = False
    index_error: str | None = None
    deleted: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None
    chunks: list[JournalChunkResponse] = Field(default_factory=list)


class JournalRecordResponse(BaseModel):
    entry: JournalEntryResponse
    chunk_count: int
    indexed: bool
    deduped: bool = False
    index_error: str | None = None
    duplicates_updated: int = Field(
        default=0,
        description="方案2 cross-scope sync: how many copies of this doc that live under "
                    "ANOTHER scope (same owner, matched by absolute path) were refreshed to "
                    "this record's latest content. Only set for source_kind=\"doc\".")


class JournalBulkDocItem(BaseModel):
    """One changed local file, submitted by journal_index_dir after hash diffing."""
    title: str = Field(..., min_length=1, max_length=1000)
    body: str = Field(..., min_length=1)
    content_hash: str = Field(..., min_length=64, max_length=64)
    size_bytes: int = Field(..., ge=0)

    @field_validator("title")
    @classmethod
    def _canonical_title(cls, value: str) -> str:
        return normalize_relative_path(value)

    @model_validator(mode="after")
    def _verify_body_facts(self) -> "JournalBulkDocItem":
        encoded = self.body.encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != self.content_hash:
            raise ValueError("content_hash must equal SHA-256 of the UTF-8 body")
        if len(encoded) != self.size_bytes:
            raise ValueError("size_bytes must equal UTF-8 body byte length")
        return self


class JournalBulkIndexRequest(BaseModel):
    project_slug: str = Field(..., min_length=1, max_length=2000)
    host: str = Field(..., min_length=1, max_length=200)
    root_dir: str = Field(..., min_length=1, max_length=2000)
    files: list[JournalBulkDocItem] = Field(default_factory=list, max_length=2000)
    report_files: list[dict[str, Any]] = Field(default_factory=list, max_length=10000)
    replace: bool = True
    live_titles: list[str] = Field(default_factory=list, max_length=10000)
    patterns: list[str] | None = None
    max_depth: int | None = Field(default=None, ge=0)
    client_job_id: str | None = Field(default=None, min_length=1, max_length=100)
    client: str = Field(default="mcp", min_length=1, max_length=100)

    @field_validator("project_slug", "root_dir")
    @classmethod
    def _canonical_scope(cls, value: str) -> str:
        return canonicalize_scope_path(value)


class JournalBulkIndexJobOut(BaseModel):
    job_id: str
    project_slug: str
    host: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    files_total: int
    files_done: int = 0
    chunks_total: int = 0
    chunks_done: int = 0
    indexed: int = 0
    progress: float = 0.0
    queue_position: int = 0
    error: str | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    skipped_binary: list[str] = Field(default_factory=list)


class JournalBulkIndexJobList(BaseModel):
    jobs: list[JournalBulkIndexJobOut] = Field(default_factory=list)


class JournalEntryListResponse(BaseModel):
    entries: list[JournalEntryResponse]
    total: int


class JournalDeleteResponse(BaseModel):
    entry_id: str
    deleted: bool
    deleted_vectors: int = 0


class JournalDocReconcileRequest(BaseModel):
    project_slug: str = Field(..., min_length=1, max_length=2000)
    live_titles: list[str] = Field(default_factory=list)
    max_depth: int | None = Field(
        default=None,
        ge=0,
        description="When set, only sweep stale docs whose rel_path depth (slash count) "
                    "is within the scan envelope — a shallow re-run won't reconcile deeper "
                    "docs it never scanned.",
    )
    patterns: list[str] | None = Field(
        default=None,
        description="When set, only sweep stale docs whose basename matches one of these "
                    "globs — the scan's pattern envelope. A '*.txt' re-run won't reconcile "
                    "the '*.md' docs it never globbed for. None or ['*'] = full reconcile.",
    )

    @field_validator("project_slug")
    @classmethod
    def _canonical_project_slug(cls, value: str) -> str:
        return canonicalize_scope_path(value)

    @field_validator("live_titles")
    @classmethod
    def _canonical_live_titles(cls, values: list[str]) -> list[str]:
        return [normalize_relative_path(value) for value in values]


class JournalDocReconcileResponse(BaseModel):
    project_slug: str
    live_titles: int
    removed_entries: int
    removed_entry_ids: list[str] = Field(default_factory=list)
    deleted_vectors: int = 0
    skipped: bool = False
    reason: str | None = None


class JournalDocScopeDeleteResponse(BaseModel):
    project_slug: str
    removed_entries: int
    removed_entry_ids: list[str] = Field(default_factory=list)
    deleted_vectors: int = 0
    catalog_removed: int = 0
    scope_aliases: list[str] = Field(
        default_factory=list,
        description="Legacy case-variant Windows scopes deleted together with project_slug.",
    )


class JournalSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="检索内容。")
    project_slug: str = Field(..., min_length=1, max_length=2000,
                             description="项目命名空间（= 目录绝对路径），只在该项目的记忆内检索。")
    mode: JournalSearchMode = Field(
        default="hybrid",
        description="keyword=FTS5 关键词（免嵌入，embed 端不可用时也能工作）；hybrid/dense/sparse=语义（需 embed 端）。",
    )
    top_k: int = Field(default=10, ge=1, le=100)
    level: int | None = None
    source_kind: str | None = None
    client: str | None = None
    tags: list[str] = Field(default_factory=list)
    title_only: bool = False
    dedup_by_entry: bool = True
    include_chunks: bool = False
    chunk_limit: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional cap for chunks returned per hit when include_chunks is true.",
    )
    rerank: bool = False
    path_glob: str | None = Field(
        default=None,
        max_length=500,
        description="限定在已索引父目录范围内的子目录 / 子文件里检索——匹配文档的 scope 相对路径"
                    "（title）。可传子树前缀（如 'notes/2026/' 或单个文件 'notes/a.md'）或 glob"
                    "（如 '*.md'、'notes/**/*.md'）。命中集在检索前解析为过滤条件下推，top_k 只从"
                    "命中文档里填充，绝不后置裁剪丢失结果；对 dense/sparse/hybrid/keyword 全模式生效。"
                    "不传则不影响正常检索性能。",
    )
    owner_user_id: str | None = Field(default=None, max_length=200)

    @field_validator("project_slug")
    @classmethod
    def _canonical_project_slug(cls, value: str) -> str:
        return canonicalize_scope_path(value)

    @field_validator("path_glob")
    @classmethod
    def _canonical_path_glob(cls, value: str | None) -> str | None:
        return normalize_relative_path(value) if value else value

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "summary": "关键词检索（免嵌入，最稳）",
                    "value": {
                        "project_slug": "C:/Users/Arc/Prd/retrival",
                        "query": "qdrant upsert wait flag",
                        "mode": "keyword",
                        "top_k": 5,
                        "include_chunks": True,
                        "chunk_limit": 5,
                    },
                }
            ]
        }
    )


class JournalSearchHit(BaseModel):
    entry: JournalEntryResponse
    score: float
    best_chunk: JournalChunkResponse | None = None
    chunks: list[JournalChunkResponse] = Field(default_factory=list)


class JournalSearchResponse(BaseModel):
    query: str
    project_slug: str
    mode: JournalSearchMode
    total: int
    hits: list[JournalSearchHit]
    metrics: dict[str, Any] = Field(default_factory=dict)
    notice: str | None = Field(
        default=None,
        description="Set only on an empty result that needs explaining — e.g. a path_glob "
                    "that matched 0 docs while the scope DOES have docs (tells the user the "
                    "glob is scope-relative and their depth is wrong).",
    )
    resolved_scope: str | None = Field(
        default=None,
        description="Set when the requested directory had no index of its own and the search "
                    "ran in this indexed ancestor scope instead. Null when the requested "
                    "scope itself was searched.",
    )
    resolved_path_prefix: str | None = Field(
        default=None,
        description="With resolved_scope: the requested directory relative to that ancestor. "
                    "Only docs below it were searched (and path_glob is relative to it); the "
                    "ancestor's memory entries and .conversations/ segments still take part.",
    )
    scope_notice: str | None = Field(
        default=None,
        description="Server-decided, deliver-once message about this scope (e.g. duplicate "
                    "code found after its last index). Present on exactly one response per "
                    "result; clients show it verbatim above their output and keep no state.",
    )
