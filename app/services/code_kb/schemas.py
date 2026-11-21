"""Pydantic contracts for the server-side code knowledge base.

The client AST-chunks its repo locally and POSTs the chunks here; the backend
embeds + stores them in the 4090 Qdrant `code_chunks` collection. Owner is always
resolved server-side (gateway identity / LAN admin) — never trusted from the body.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.services.index_catalog.schemas import IndexFileRecord
from app.services.scope_paths import canonicalize_scope_path, normalize_relative_path


class CodeChunkIn(BaseModel):
    start_line: int = Field(..., ge=0)
    end_line: int = Field(..., ge=0)
    label: str = Field(default="", max_length=200, description="Symbol/scope, e.g. a function name.")
    content: str = Field(..., min_length=1, description="Embed text (chunk with a `# path:` header).")
    code: str = Field(..., min_length=1, description="Raw code shown in results.")


class CodeFileIn(BaseModel):
    rel_path: str = Field(..., min_length=1, max_length=1000)
    language: str | None = Field(default=None, max_length=50)
    content_hash: str | None = Field(
        default=None, max_length=64,
        description="Hash of the file's bytes; lets incremental indexing skip unchanged files.")
    chunks: list[CodeChunkIn] = Field(default_factory=list)

    @field_validator("rel_path")
    @classmethod
    def _normalise_rel_path(cls, value: str) -> str:
        return normalize_relative_path(value)


class CodeIndexReport(BaseModel):
    """Index-inventory records the backend writes to the catalog WHEN THE JOB COMPLETES.

    Bundling the report into the submit lets index_code fire-and-forget: the client
    returns the job_id immediately instead of blocking on the embed just to report
    afterwards. The write happens only on success (never optimistically — a failed job
    must not leave the catalog claiming files are indexed that never landed)."""
    files: list[IndexFileRecord] = Field(default_factory=list)
    replace: bool = Field(default=True,
                          description="True = full snapshot (removed files swept); False = upsert.")


class CodeIndexRequest(BaseModel):
    client_job_id: str | None = Field(
        default=None,
        min_length=32,
        max_length=32,
        pattern=r"^[0-9a-f]{32}$",
        description="Optional durable client receipt. When supplied, job status remains "
                    "queryable by this ID after the MCP client process restarts.",
    )
    project: str = Field(..., min_length=1, max_length=2000,
                         description="Scope namespace = the indexed directory's absolute "
                                     "path (the MCP client forces this); searches scope to "
                                     "(owner, project). Wide enough to hold a full abspath.")
    host: str = Field(default="", max_length=200, description="Machine the code lives on.")
    root_dir: str = Field(default="", max_length=2000)
    files: list[CodeFileIn] = Field(default_factory=list)
    incremental: bool = Field(
        default=False,
        description="Only (re)embed the files in `files` and reconcile deletions against "
                    "`manifest`, instead of wiping + rebuilding the whole project. The "
                    "client sends only changed/new files' chunks.")
    manifest: list[str] | None = Field(
        default=None, max_length=100000,
        description="Incremental only: the full set of rel_paths that currently exist in "
                    "the repo. Indexed files absent from it are deleted (removed-file "
                    "reconciliation). None = don't delete anything, only add/update.")
    removed_paths: list[str] = Field(
        default_factory=list, max_length=100000,
        description="Incremental only: exact repo-relative paths known to have been "
                    "deleted. Filesystem watchers use this without enumerating a full "
                    "manifest. Changed files in `files` take precedence over this list.")
    report: CodeIndexReport | None = Field(
        default=None,
        description="Catalog inventory records to write to the index catalog once the job "
                    "COMPLETES successfully. None = don't touch the catalog. This lets the "
                    "client submit and return without blocking on the embed to report.")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "project": "retrival", "host": "DESKTOP-1", "root_dir": "C:/repo",
                "files": [{
                    "rel_path": "core/store.py", "language": "python",
                    "chunks": [{"start_line": 10, "end_line": 40, "label": "upsert_batch",
                                "content": "# path: core/store.py\ndef upsert_batch(...): ...",
                                "code": "def upsert_batch(...): ..."}],
                }],
            }]
        }
    )

    @field_validator("project")
    @classmethod
    def _canonical_project(cls, value: str) -> str:
        return canonicalize_scope_path(value)

    @field_validator("root_dir")
    @classmethod
    def _canonical_root_dir(cls, value: str) -> str:
        return canonicalize_scope_path(value) if value else value

    @field_validator("manifest", "removed_paths")
    @classmethod
    def _normalise_relative_path_lists(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return [normalize_relative_path(path) for path in value]


CodeSearchMode = Literal["hybrid", "dense", "keyword"]
CodeSearchEffectiveMode = Literal["hybrid", "dense", "keyword", "unavailable"]
CodeScopeSnapshotState = Literal[
    "completed",
    "preparing",
    "recovery_required",
    "unavailable",
]


class CodeSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Describe behavior for hybrid/dense mode, or pass an exact symbol/token for keyword mode.",
    )
    project: str | None = Field(default=None, max_length=2000,
                                description="Restrict to one project (= indexed dir abspath).")
    top_k: int = Field(default=8, ge=1, le=50)
    mode: CodeSearchMode = Field(
        default="hybrid",
        description="hybrid = dense+sparse RRF (default, best general ranking); "
                    "dense = semantic only; keyword = sparse lexical only (exact "
                    "symbol names, BM25-like). keyword never substitutes dense results; "
                    "when sparse is unavailable the response explains how to re-index.")
    rerank: bool = Field(
        default=False,
        description="Re-score the fused candidates with the cross-encoder reranker for "
                    "higher precision (slower). Off by default.")
    language: str | None = Field(
        default=None, max_length=50,
        description="Restrict to one tree-sitter language, e.g. 'rust' or 'python'.")
    path_glob: str | None = Field(
        default=None, max_length=500,
        description="Scope-relative path filter: prefix ('src/peers/') or glob "
                    "('*.rs', 'src/**/*.go'). Backslashes are accepted; Windows scopes "
                    "match case-insensitively. Do not pass an absolute repo path here.")
    max_per_file: int | None = Field(
        default=None, ge=1, le=20,
        description="Cap hits from any single file so one file can't monopolise the top-k.")

    @field_validator("project")
    @classmethod
    def _canonical_project(cls, value: str | None) -> str | None:
        return canonicalize_scope_path(value) if value else value

    @field_validator("path_glob")
    @classmethod
    def _normalise_path_glob(cls, value: str | None) -> str | None:
        return normalize_relative_path(value) if value else value


class CodeSearchTimingsMs(BaseModel):
    """Per-phase wall time in milliseconds (F2-b). A phase this request never ran
    (e.g. ``rerank`` when ``rerank=false``, ``fuse`` on a single-leg search) is null,
    never 0 — 0 would misrepresent "ran instantly" as "did not run"."""
    embed: float | None = Field(default=None, description="Query embedding.")
    retrieve: float | None = Field(default=None, description="Qdrant vector search.")
    fuse: float | None = Field(
        default=None,
        description="RRF fusion across legs. Null for a single-leg (non-federated) search.")
    rerank: float | None = Field(default=None, description="Cross-encoder rescoring.")
    total: float | None = Field(default=None, description="Whole request, wall time.")


class CodeHit(BaseModel):
    path: str
    start_line: int
    end_line: int
    label: str = ""
    language: str | None = None
    project: str = ""
    host: str = ""
    score: float
    code: str = ""


class CodeSearchResponse(BaseModel):
    hits: list[CodeHit]
    total: int
    requested_mode: CodeSearchMode = "hybrid"
    effective_mode: CodeSearchEffectiveMode = "hybrid"
    notice: str | None = Field(
        default=None,
        description="Search diagnostic, e.g. strict keyword retrieval unavailable or a "
                    "scope-relative path_glob matching zero indexed files.")
    path_glob_matched_files: int | None = Field(
        default=None,
        description="Number of indexed files selected by path_glob; null when no path_glob was used.")
    indexed_file_count: int | None = Field(
        default=None,
        description="Indexed file count inspected for path_glob; null when no path_glob was used.")
    scope_aliases: list[str] = Field(
        default_factory=list,
        description="Legacy case-variant Windows scope spellings included for compatibility.")
    scope_snapshot_state: CodeScopeSnapshotState = Field(
        default="completed",
        description="completed = results came from a stable scope snapshot; preparing = an active "
                    "code-index mutation prevents exposing a partial in-place write; "
                    "recovery_required = an interrupted mutation needs a full recovery index; "
                    "unavailable = Index Catalog visibility could not be verified.")
    active_index_job_id: str | None = Field(
        default=None,
        description="Live code-index job responsible for a preparing or refresh diagnostic, if any. "
                    "Null for recovery_required and unavailable states so clients do not poll a failed "
                    "or synthetic job id.")
    recovery_required: bool = Field(
        default=False,
        description="True when an interrupted in-place mutation requires a full recovery index before "
                    "this scope can be searched. This is not an active job to poll.")
    previous_completed_snapshot_available: bool | None = Field(
        default=None,
        description="During preparation, whether a previous completed snapshot is known to exist.")
    timings_ms: CodeSearchTimingsMs = Field(default_factory=CodeSearchTimingsMs)


class CodeIndexResponse(BaseModel):
    project: str
    host: str
    files: int
    indexed: int
    duplicates_updated: int = Field(
        default=0,
        description="方案2 cross-scope sync: how many copies of the just-indexed files "
                    "that live under ANOTHER scope (same owner, matched by absolute path) "
                    "were refreshed to this run's latest content + vectors.")


class CodeDeleteResponse(BaseModel):
    project: str
    deleted: int                      # vectors removed from code_chunks
    catalog_removed: int = 0          # inventory rows soft-deleted for this project
    scope_aliases: list[str] = Field(
        default_factory=list,
        description="Legacy case-variant Windows scopes deleted together with project.")


class CodeIndexJobOut(BaseModel):
    """Public view of an async index job (POST /index → 202, poll GET /jobs/{id})."""
    job_id: str
    project: str
    host: str
    status: str = Field(description="queued | running | completed | failed | cancelled")
    progress: float = Field(ge=0.0, le=1.0)
    files_total: int
    chunks_total: int
    removed_total: int = Field(
        default=0,
        description="Repo-relative file paths explicitly removed by this incremental job.")
    chunks_done: int
    indexed: int
    duplicates_updated: int = Field(
        default=0,
        description="方案2 cross-scope sync count once the job completes (0 while running).")
    queue_position: int = Field(description="0 = running/finished; N = Nth in line.")
    error: str | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None


class CodeIndexJobList(BaseModel):
    jobs: list[CodeIndexJobOut]


class CodeDuplicateScanRequest(BaseModel):
    """Ask which files in one indexed scope duplicate each other.

    Defaults come from the 2026-09-01 measurement in
    core/code_kb/duplicate_rollup: threshold moves the result set very little,
    `min_shared_chunks` is what separates real duplicated design from the
    one-shared-boilerplate-chunk noise that is 93% of raw candidates.
    """
    project: str = Field(
        ..., min_length=1, max_length=1000,
        description="Indexed scope to scan. Required: an owner-wide scan would compare "
                    "every repository against every other and report cross-project "
                    "coincidences as duplication.")
    threshold: float = Field(
        default=0.92, ge=0.0, le=1.0,
        description="Cosine score at which two chunks count as near-duplicates.")
    min_shared_chunks: int = Field(
        default=2, ge=1, le=100,
        description="Chunk pairs a file pair must share to be reported. 1 is very noisy: "
                    "in the reference measurement 176 of 190 candidate pairs shared "
                    "exactly one chunk and none of those were real duplication.")
    max_pairs: int = Field(default=50, ge=1, le=500)
    max_examples: int = Field(
        default=3, ge=1, le=20,
        description="Highest-scoring shared chunks quoted per file pair.")
    max_chunks: int = Field(
        default=20000, ge=1, le=200000,
        description="Safety cap on chunks scanned. Truncation is reported, never silent.")
    exclude_globs: list[str] | None = Field(
        default=None, max_length=100,
        description="Gitignore-syntax exclusions. Null uses the defaults (vendored, "
                    "generated and archived trees); pass [] to scan everything.")

    @field_validator("project")
    @classmethod
    def _canonical_project(cls, value: str) -> str:
        return canonicalize_scope_path(value)


class CodeDuplicateChunk(BaseModel):
    path: str
    start_line: int
    end_line: int
    label: str = ""


class CodeDuplicateExample(BaseModel):
    a: CodeDuplicateChunk
    b: CodeDuplicateChunk
    score: float


class CodeDuplicatePair(BaseModel):
    path_a: str
    path_b: str
    shared_chunks: int = Field(
        description="Distinct chunk pairs these two files share above the threshold. "
                    "This is the primary ranking signal, not the score.")
    max_score: float
    mean_score: float
    examples: list[CodeDuplicateExample] = Field(default_factory=list)


class CodeDuplicateScanResponse(BaseModel):
    project: str
    pairs: list[CodeDuplicatePair]
    total: int
    scanned_chunks: int
    scanned_files: int
    truncated: bool = Field(
        default=False,
        description="True when max_chunks stopped the scan early. An empty `pairs` with "
                    "truncated=True does NOT mean the scope is duplicate-free.")
    notice: str | None = None
    scope_aliases: list[str] = Field(
        default_factory=list,
        description="Legacy case-variant Windows scope spellings included in the scan.")


# ---- federated private+public code search (docs/federated_code_search_plan.md) ----


class FederatedCodeSearchRequest(BaseModel):
    """`search_code`'s superset: the private leg always runs; the public leg is opt-in
    (R1, operator-confirmed 2026-09-23) so a non-admin caller's default behaviour is
    identical to an ordinary private search."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    mode: CodeSearchMode = "hybrid"
    rerank: bool = False
    language: str | None = Field(
        default=None, max_length=50, description="Applies to both legs.")
    include_public: bool = Field(
        default=False,
        description="Off by default. True additionally requires admin tier (403 for "
                    "any other tier) and never silently falls back to a private-only "
                    "result.")
    projects: list[str] = Field(
        default_factory=list, max_length=50,
        description="Private only. Empty = every scope the caller owns. Non-empty "
                    "scopes are searched in ONE Qdrant query (MatchAny); an "
                    "individually-withheld scope is excluded and reported in "
                    "private.excluded_scopes rather than withholding the whole leg.")
    path_glob: str | None = Field(default=None, max_length=500, description="Private only.")
    max_per_file: int | None = Field(
        default=None, ge=1, le=50, description="Private only; same semantics as "
                    "CodeSearchRequest.max_per_file.")
    source_ids: list[str] = Field(
        default_factory=list, max_length=100, description="Public only.")
    domains: list[str] = Field(
        default_factory=list, max_length=50, description="Public only.")

    @field_validator("projects")
    @classmethod
    def _canonical_projects(cls, value: list[str]) -> list[str]:
        canonical = (canonicalize_scope_path(project) for project in value if project)
        return list(dict.fromkeys(canonical))

    @field_validator("path_glob")
    @classmethod
    def _normalise_path_glob(cls, value: str | None) -> str | None:
        return normalize_relative_path(value) if value else value

    @model_validator(mode="after")
    def _public_only_filters_require_include_public(self) -> "FederatedCodeSearchRequest":
        if not self.include_public and (self.source_ids or self.domains):
            raise ValueError(
                "source_ids/domains require include_public=true; "
                "they are silently ignored otherwise, which this route refuses to do")
        return self


class FederatedCodeHit(BaseModel):
    origin: Literal["private", "public"]
    path: str
    start_line: int
    end_line: int
    label: str = ""
    language: str | None = None
    score: float = Field(description="Final sort score: reranker score, else RRF score.")
    retrieval_score: float = Field(description="That leg's own original retrieval score.")
    fusion_rank_score: float = Field(description="Reciprocal-rank-fusion score.")
    reranker_score: float | None = None
    code: str = ""
    project: str | None = None    # private
    host: str | None = None       # private
    source_id: str | None = None  # public
    revision: str | None = None   # public
    domain: str | None = None     # public


class FederatedLegDiagnostics(BaseModel):
    included: bool
    candidate_count: int = 0
    effective_mode: str | None = None
    notice: str | None = None
    scope_snapshot_state: str | None = None    # private only
    excluded_scopes: list[dict] = Field(
        default_factory=list,
        description="Private only (decision 10). Each entry: "
                    "{project, state, active_index_job_id} for a scope withheld from "
                    "this response instead of failing the whole private leg closed.")
    searched_projects: list[str] = Field(
        default_factory=list,
        description="Private only. Empty means every scope the caller owns.")
    selected_source_ids: list[str] = Field(default_factory=list)   # public only


class FederatedCodeSearchResponse(BaseModel):
    query: str
    hits: list[FederatedCodeHit]
    total: int
    fusion: Literal["rrf", "rerank"]
    rerank_applied: bool
    private: FederatedLegDiagnostics
    public: FederatedLegDiagnostics
    timings_ms: CodeSearchTimingsMs = Field(default_factory=CodeSearchTimingsMs)
