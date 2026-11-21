"""
Pipeline management API — trigger and monitor collection, extraction, indexing.
"""
from typing import Any, Dict, List, Literal, Optional

import asyncio

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field, model_validator

from app.services.pipeline_service import PipelineService
from app.services.pipeline.arxiv_stages import (
    NotADeltaRun,
    RunSources,
    default_run_sources,
    delta_run_page,
    is_arxiv_venue,
    validate_slice_key,
)

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])
_service = PipelineService()
_UserId = Header(default=None, alias="X-Retrival-User-Id")
_Tier = Header(default=None, alias="X-Retrival-User-Tier")


class BatchIndexRequest(BaseModel):
    venue: Optional[str] = Field(default=None, description="Filter by venue (e.g. NeurIPS, AAAI)")
    year: Optional[int] = Field(default=None, description="Filter by year (e.g. 2024)")
    paper_ids: Optional[list[str]] = Field(default=None, description="Explicit paper IDs to index")
    batch_size: int = Field(default=50, ge=1, le=200, description="Papers per GPU batch")
    force: bool = Field(default=False, description="Re-index already indexed papers")
    dry_run: bool = Field(default=False, description="Count eligible papers without indexing")


batch_index_router = APIRouter(tags=["papers"])


@batch_index_router.post("/api/papers/batch-index")
def start_batch_index(request: BatchIndexRequest = BatchIndexRequest()):
    """Batch index papers by filter (venue/year) or explicit IDs.

    Uses BGE-M3 embedder + Qdrant hybrid (dense+sparse). Runs in background,
    returns task_id for progress tracking via GET /api/pipeline/tasks/{id}.
    """
    return _service.start_batch_index(
        venue=request.venue,
        year=request.year,
        paper_ids=request.paper_ids,
        batch_size=request.batch_size,
        force=request.force,
        dry_run=request.dry_run,
    )


class IngestPipelineRequest(BaseModel):
    venue: str = Field(..., description="Conference venue, e.g. CVPR")
    year: int = Field(..., description="Year, e.g. 2026; for venue 'arXiv' a month slice YYYYMM (202311 = ids before 2023-12)")
    ssh_target: str = Field(
        ...,
        description=(
            "Remote ssh alias for extract/embed GPU stages. REQUIRED — never "
            "defaulted, since silently picking a busy shared box can collide "
            "with other jobs. e.g. '4090' or 'kss'."
        ),
    )
    remote_repo: Optional[str] = Field(
        default=None,
        description=(
            "Repo root on the remote box (where remote_marker/remote_embed "
            "launcher scripts live). Defaults to the KSS layout "
            "(/home/kss/arc/retrival) — override for any other ssh target."
        ),
    )
    concurrency: int = Field(default=8, ge=1, le=32, description="Download concurrency")
    world_size: int = Field(default=8, ge=1, le=16, description="Remote GPU ranks for extract/embed (fallback when ranks_per_gpu not set)")
    ranks_per_gpu: Optional[int] = Field(
        default=None, ge=1, le=4,
        description=(
            "Stacking factor: marker ranks to launch per GPU that passes "
            "preflight. Effective world_size = #idle_gpus * ranks_per_gpu. "
            "Use 2 on 4090/5090 (24GB+) to stack 2 marker procs per card. "
            "Leave None to use world_size verbatim (kss old behavior)."
        ),
    )
    execution_mode: Literal["remote", "local", "host"] = Field(
        default="local",
        description=(
            "Where the GPU stages (extract/embed) run. 'local' (default): "
            "GPU work runs ON this box via local subprocesses (no ssh/SMB/"
            "ship.sh, W106). 'remote': retired KSS-style ssh + SMB + ship.sh "
            "topology, kept only for explicit reuse; its SMB tree no longer "
            "exists. 'host': embed (and, for a host with a 'marker' block, "
            "extract) on the ssh host named by ssh_target "
            "(deploy/compute88/embed_hosts.json). download/preindex/"
            "ingest run local in every mode."
        ),
    )
    start_stage: Optional[
        Literal["download", "extract", "preindex", "embed", "ingest"]
    ] = Field(
        default=None,
        description=(
            "Begin orchestration at this existing stage. Earlier stages remain "
            "visible in the canonical five-step ingest task and are durably "
            "marked completed as externally supplied. Use only after an "
            "internal operator has verified those external artifacts."
        ),
    )
    chunk_id_before: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Required only when start_stage='embed': the exact chunks.id "
            "watermark captured immediately before the externally completed "
            "SQL preindex. Embedding exports chunks with id greater than this "
            "value, preventing an accidental full-corpus export."
        ),
    )
    shard_start: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Required only when start_stage='ingest': first externally supplied "
            "vector shard number, inclusive."
        ),
    )
    shard_end: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Required only when start_stage='ingest': final externally supplied "
            "vector shard number, exclusive. Must be greater than shard_start."
        ),
    )
    options: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Per-run options handed to the stage context and stored in the task "
            "params, so a retry or restart resumes with the same values. Keys are "
            "defined by the venue's pipeline (arXiv: batch/delta options; the "
            "delta's source is 'volume' (the default; 'local' is refused, this "
            "host never downloads) and imports the bundles the downloading host "
            "(cfscan) left in the 115 volume options.volume (default /cfs-arxiv) and "
            "waits up to receive_deadline_hours (default 6) for a run to "
            "finish; Papers-Download: refused on this host; conferences: the "
            "PDFs are copied from a 115 volume on this host's OpenList node "
            "(options.volume, default /cfs-tier1; options.volume_prefix, "
            "default '<conference lower>/<year>'), and the volume has to hold "
            "them first (cfscan downloads them there, and leaves a record for "
            "every paper it tried and could not get: a slice whose missing "
            "papers were all tried goes on, and they stay on the refill list); "
            "download_from = 'volume' (the default) or 'push' (waits for the "
            "PDFs that the producing host pushes with "
            "scripts/sync/push_slice_bundle.py), 'local' is refused: this host "
            "never downloads; extract_from = 'push' "
            "likewise for Markdown; receive_poll_seconds); other unknown keys "
            "are ignored."
        ),
    )

    @model_validator(mode="after")
    def validate_arxiv_slice(self):
        # venue "arXiv" is sliced by submission month: year carries YYYYMM.
        if is_arxiv_venue(self.venue):
            validate_slice_key(self.year)
        return self

    @model_validator(mode="after")
    def validate_external_stage_cursor(self):
        if self.start_stage == "embed":
            if self.chunk_id_before is None:
                raise ValueError("chunk_id_before is required when start_stage='embed'")
            if self.shard_start is not None or self.shard_end is not None:
                raise ValueError("shard_start/shard_end are only valid when start_stage='ingest'")
        elif self.start_stage == "ingest":
            if self.shard_start is None or self.shard_end is None:
                raise ValueError("shard_start and shard_end are required when start_stage='ingest'")
            if self.shard_end <= self.shard_start:
                raise ValueError("shard_end must be greater than shard_start")
            if self.chunk_id_before is not None:
                raise ValueError("chunk_id_before is only valid when start_stage='embed'")
        elif any(value is not None for value in (
            self.chunk_id_before,
            self.shard_start,
            self.shard_end,
        )):
            raise ValueError(
                "external cursors are only valid for start_stage='embed' or 'ingest'"
            )
        return self


class ArxivIngestRequest(BaseModel):
    """Start one ingest_pipeline task per arXiv batch (~2,000 papers) of the
    given months; they stream through the stages together."""

    months: List[int] = Field(..., min_length=1, description="Month slices YYYYMM, e.g. [202609, 202608].")
    execution_mode: Optional[Literal["remote", "local", "host"]] = Field(
        default=None,
        description="Where embed runs: this backend's vLLM embedding service (default, 'local'), 'host' = the ssh embed host named by ssh_target "
                    "(deploy/compute88/embed_hosts.json), or the retired kss 'remote' topology.")
    ssh_target: Optional[str] = Field(default=None, description="Embed host name for 'host'; kss target for 'remote'.")
    remote_repo: Optional[str] = Field(default=None, description="Same as /ingest.")
    concurrency: int = Field(default=8, ge=1, le=64)
    world_size: int = Field(default=8, ge=1, le=64)
    ranks_per_gpu: Optional[int] = Field(default=None, ge=1, le=8)


class DownloadBatchRequest(BaseModel):
    """Download the PDFs of several manifest slices as one ingest_pipeline task.

    One serial queue that alternates download hosts (and venues), cools down a
    host that refuses, and holds the shared ``net`` lock; ingest of each slice
    is a separate /ingest run later. Refused (422) on a host that never
    downloads: the ingest host only receives what cfscan put in the 115 volume.
    """

    targets: List[str] = Field(
        ..., min_length=1,
        description="Manifest slices CONF:YEAR as in the manifest, e.g. [\"ECCV:2024\", \"IEEE S&P:2023\"].")
    min_interval_seconds: Optional[float] = Field(
        default=None, ge=0, le=600, description="Seconds between two requests (default 3).")
    host_cooldown_seconds: Optional[float] = Field(
        default=None, ge=1, le=3600,
        description="First cooldown of a refusing host, doubling per refusal (default 300).")
    collect: bool = Field(
        default=False,
        description="Collect first: slices with no manifest rows on this host are collected from their "
                    "publishers into the manifest (backup first) before the download. Slices that already "
                    "have rows are not re-collected.")


class MedicalDownloadRequest(BaseModel):
    """Start the deployment-owned, resumable medical PDF download Task.

    The caller cannot choose a filesystem destination, proxy, or transfer rate:
    compute .88 always uses its local ``127.0.0.1:10808`` proxy and the shared
    decimal 2,000,000 B/s (= 2.00 MB/s) limiter.
    """

    download_limit: int = Field(
        default=0,
        ge=0,
        le=100_000,
        description="Maximum pending PDFs to process; 0 processes all pending manifest rows.",
    )
    allow_cf_bypass: bool = Field(
        default=True,
        description="Allow the existing proxy-routed Turnstile fallback after a challenge.",
    )


class MedicalDownloadStartResponse(BaseModel):
    status: Literal["started"]
    task_id: int


class MedicalDownloadConflictDetail(BaseModel):
    code: Literal["medical_download_already_running"]
    task_id: int
    message: str


class MedicalDownloadConflictResponse(BaseModel):
    detail: MedicalDownloadConflictDetail


class MedicalDownloadUnavailableResponse(BaseModel):
    detail: str


@router.post("/ingest")
async def start_ingest_pipeline(request: IngestPipelineRequest):
    """Start the full ingest pipeline (download -> extract -> preindex -> embed
    -> ingest) for one venue+year as a background `ingest_pipeline` task.

    Returns a task_id; observe per-stage status/progress/logs via the existing
    GET /api/tasks/{id}, /api/tasks/{id}/subtasks and /api/tasks/{id}/stream.
    """
    from app.services.pipeline.service import get_pipeline_runner
    try:
        return await asyncio.to_thread(
            get_pipeline_runner().start_ingest,
            request.venue, request.year,
            ssh_target=request.ssh_target,
            remote_repo=request.remote_repo,
            concurrency=request.concurrency, world_size=request.world_size,
            ranks_per_gpu=request.ranks_per_gpu,
            execution_mode=request.execution_mode,
            start_stage=request.start_stage,
            chunk_id_before=request.chunk_id_before,
            shard_start=request.shard_start,
            shard_end=request.shard_end,
            options=request.options,
        )
    except ValueError as exc:  # bad arXiv key, unknown embed host, external start with no output
        raise HTTPException(422, str(exc))


class ReembedRequest(BaseModel):
    """Start one named re-embed scope (``reembed_pipeline``); there are no free-form filters."""

    model_config = {"extra": "forbid"}

    collection: str = Field(..., description="The collection family the scope works on, e.g. 'papers'.")
    scope: str = Field(
        ...,
        description=(
            "A preset: code_chunks:truncated, public_code_chunks:truncated, code_chunks:align, "
            "public_code_chunks:align, journal_chunks:rebuild, papers:uploads, papers:pipeline, arxiv:all. "
            "Its prefix must equal `collection`."
        ),
    )
    ssh_target: str = Field(
        default="arc99",
        description="Embed host (deploy/compute88/embed_hosts.json; resource gpu-<host>), or 'local' to embed on "
                    "this backend's vLLM embedding service (resource embed-service; no GPU of its own).")


@router.post("/reembed")
async def start_reembed_pipeline(
    body: ReembedRequest,
    request: Request,
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
):
    """Start a `reembed_pipeline` task: re-embed the vectors written before the embedding profile.

    Admin only. Observe it like any task: GET /api/tasks/{id}, its subtasks and events."""
    from app.api.identity import resolve_admin
    from app.services.pipeline.reembed_pipeline import get_reembed_runner

    resolve_admin(request, user_id, tier)
    try:
        return await asyncio.to_thread(
            get_reembed_runner().start_reembed, body.collection, body.scope, body.ssh_target)
    except ValueError as exc:  # unknown scope, scope of another collection, unknown embed host
        raise HTTPException(422, str(exc))


@router.post("/arxiv")
async def start_arxiv_pipelines(request: ArxivIngestRequest):
    """One arXiv ingest task per batch of the given months (see ArxivIngestRequest).

    Each task appears in the Tasks panel with its five steps and live
    per-stage progress; the download step shows the downloading host's
    progress until the batch is pushed.
    """
    from app.services.pipeline.service import get_pipeline_runner
    try:
        return await asyncio.to_thread(
            get_pipeline_runner().start_arxiv_batches, request.months,
            ssh_target=request.ssh_target, remote_repo=request.remote_repo,
            concurrency=request.concurrency, world_size=request.world_size,
            ranks_per_gpu=request.ranks_per_gpu, execution_mode=request.execution_mode,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.post("/download-batch")
async def start_download_batch(request: DownloadBatchRequest):
    """One download-only ingest_pipeline task over the given slices (see DownloadBatchRequest);
    422 before a task exists on the ingest host, which never downloads."""
    from app.services.pipeline.service import get_pipeline_runner
    try:
        return await asyncio.to_thread(
            get_pipeline_runner().start_download_batch, request.targets,
            min_interval_seconds=request.min_interval_seconds,
            host_cooldown_seconds=request.host_cooldown_seconds,
            collect=request.collect,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))


class ArxivRunPaper(BaseModel):
    paper_id: str
    arxiv_id: Optional[str] = None
    title: Optional[str] = None
    source_version: Optional[str] = None
    withdrawn: Optional[bool] = None


class ArxivRunPapersResponse(BaseModel):
    task_id: int
    kind: Literal["new", "replaced", "withdrawn", "revived", "html_missing", "convert_rejected"]
    offset: int
    limit: int
    total: int = Field(description="Papers of this kind in the run")
    counts: dict[str, int] = Field(description="Papers per kind, for the tab labels")
    papers: List[ArxivRunPaper]


@router.get(
    "/arxiv/runs/{task_id}/papers",
    response_model=ArxivRunPapersResponse,
    responses={
        401: {"description": "Internal authentication is required."},
        403: {"description": "Internal admin tier is required."},
        404: {"description": "The task is not an arXiv delta run."},
    },
)
async def arxiv_run_papers(
    task_id: int,
    request: Request,
    kind: Literal["new", "replaced", "withdrawn", "revived", "html_missing", "convert_rejected"] = Query(
        ..., description="new: first indexed; replaced: a new version replaced the text; withdrawn / revived: "
                         "the withdrawn flag turned on / off; html_missing: arXiv has no HTML yet; "
                         "convert_rejected: the HTML has no usable article and is not retried"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
    sources: RunSources = Depends(default_run_sources),
):
    """The papers one arXiv daily-update run touched, by kind, from the files the run wrote
    (its frozen selection and its preindex report), titled from SQL."""
    from app.api.identity import resolve_admin

    resolve_admin(request, user_id, tier)
    try:
        return await asyncio.to_thread(delta_run_page, sources, task_id, kind, offset, limit)
    except NotADeltaRun as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/medical-download",
    status_code=202,
    response_model=MedicalDownloadStartResponse,
    responses={
        401: {"description": "Internal authentication is required."},
        403: {"description": "Internal admin tier is required."},
        409: {"model": MedicalDownloadConflictResponse, "description": "A medical download task is already active."},
        503: {"model": MedicalDownloadUnavailableResponse, "description": "The designated executor, proxy, or artifact mount failed preflight."},
    },
)
async def start_medical_download(
    body: MedicalDownloadRequest,
    request: Request,
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
):
    """Create one durable ``medical_corpus_download`` task.

    Track its ordinary task id via the existing ``GET /api/tasks``,
    ``GET /api/tasks/{id}``, subtasks, events and cancel/retry actions.  This
    intentionally adds no parallel job list or polling contract.
    """
    from app.api.identity import resolve_admin
    from app.services.medical_corpus import get_medical_download_runner

    resolve_admin(request, user_id, tier)
    try:
        result = await asyncio.to_thread(
            get_medical_download_runner().start_download,
            download_limit=body.download_limit,
            allow_cf_bypass=body.allow_cf_bypass,
        )
        if result.get("status") == "conflict":
            raise HTTPException(status_code=409, detail={
                "code": "medical_download_already_running",
                "task_id": result.get("task_id"),
                "message": result.get("error"),
            })
        return result
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/tasks")
def list_tasks():
    """List all pipeline tasks (running and completed)."""
    return _service.list_tasks()


@router.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    """Get status of a specific pipeline task."""
    status = _service.get_task_status(task_id)
    if status is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Task not found")
    return status
