"""
Pipeline management API — trigger and monitor collection, extraction, indexing.
"""
from typing import Literal, Optional

import asyncio

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field, model_validator

from app.services.pipeline_service import PipelineService

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
async def start_batch_index(request: BatchIndexRequest = BatchIndexRequest()):
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
    year: int = Field(..., description="Year, e.g. 2026")
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
    execution_mode: Literal["remote", "local"] = Field(
        default="remote",
        description=(
            "Where the GPU stages (extract/embed) run. 'remote' (default): "
            "KSS-style ssh + SMB + ship.sh — the original split topology. "
            "'local': self-hosted, GPU work runs ON this box via local "
            "subprocesses (no ssh/SMB/ship.sh) — use when the orchestrator "
            "runs on the GPU machine itself (4090, W106). download/preindex/"
            "ingest run local in both modes."
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
    return get_pipeline_runner().start_ingest(
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
    )


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
async def list_tasks():
    """List all pipeline tasks (running and completed)."""
    return _service.list_tasks()


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a specific pipeline task."""
    status = _service.get_task_status(task_id)
    if status is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Task not found")
    return status
