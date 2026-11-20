"""Internal and gateway-admin public-code endpoints."""

from __future__ import annotations

import asyncio
from dataclasses import asdict

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response

from app.api.identity import resolve_admin
from app.services.public_code import (
    get_public_code_access_profiles,
    get_public_code_control_service,
    get_public_code_import_service,
    get_public_code_job_service,
    get_public_code_search_service,
)
from app.services.public_code.bundle_import import (
    PublicCodeImportConflict,
    PublicCodeImportError,
    PublicCodeImportNotFound,
)
from app.services.public_code.schemas import (
    PublicCodeImportCompleteResponse,
    PublicCodeImportCreateRequest,
    PublicCodeImportResponse,
    PublicCodeJobResponse,
    PublicCodeSourceCreateRequest,
    PublicCodeSourceCreateResponse,
    PublicCodeSearchRequestBody,
    PublicCodeSearchResponseBody,
    PublicCodeSourceSelection,
    PublicCodeSourceResponse,
    PublicCodeUpdatePolicyRequest,
)
from app.services.public_code.search_service import PublicCodeSearchRequest
from app.services.public_code.source_catalog import PublicSourceSelectionError
from app.services.public_code.source_catalog import PublicCodeSource


router = APIRouter()
_UserId = Header(default=None, alias="X-Retrival-User-Id")
_Tier = Header(default=None, alias="X-Retrival-User-Tier")
_AUTH_RESPONSES = {
    401: {"description": "Authentication required"},
    403: {"description": "Admin access required"},
}


async def _run_service(operation):
    try:
        return await asyncio.to_thread(operation)
    except PublicCodeImportNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PublicCodeImportConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except PublicCodeImportError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PublicSourceSelectionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        message = str(exc)
        if message.startswith("unknown public-code source:"):
            status_code = 404
        elif "already exists" in message:
            status_code = 409
        else:
            status_code = 422
        raise HTTPException(status_code=status_code, detail=message) from exc


def _job_response(job):
    return {
        "job_id": job.job_id,
        "operation": job.operation,
        "source_ids": list(job.source_ids),
        "total_sources": job.total_sources,
        "status": job.status.value if hasattr(job.status, "value") else job.status,
        "completed_sources": job.completed_sources,
        "progress": job.progress,
        "phase": job.phase,
        "current_source_id": job.current_source_id,
        "current_path": job.current_path,
        "files_done": job.files_done,
        "files_total": job.files_total,
        "chunks_done": job.chunks_done,
        "message": job.message,
        "source_progress": job.source_progress,
        "results": list(job.results),
        "errors": job.errors or {},
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }


@router.get(
    "/sources",
    response_model=list[PublicCodeSourceResponse],
    responses=_AUTH_RESPONSES,
)
async def list_public_code_sources(
    request: Request,
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
):
    """List source state for internal callers and public gateway admins."""

    resolve_admin(request, user_id, tier)
    sources = await _run_service(get_public_code_control_service().list_sources)
    return [asdict(source) for source in sources]


@router.post(
    "/sources",
    status_code=201,
    response_model=PublicCodeSourceCreateResponse,
    responses={**_AUTH_RESPONSES, 409: {"description": "Source already exists"}},
)
async def register_public_code_source(
    body: PublicCodeSourceCreateRequest,
    request: Request,
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
):
    """Register one Git source and optionally queue its first mirror/index job."""

    admin_user_id = resolve_admin(request, user_id, tier)
    source = await _run_service(
        lambda: PublicCodeSource(
            source_id=body.source_id,
            domain=body.domain,
            relative_path=body.relative_path,
            upstream_url=body.upstream_url,
            default_ref=body.default_ref,
            proxy_profile=body.proxy_profile,
            credential_profile=body.credential_profile,
            language_overrides=body.language_overrides,
        )
    )
    state = await _run_service(
        lambda: get_public_code_control_service().register_source(
            source,
            update_policy=body.update_policy,
            update_interval_minutes=body.update_interval_minutes,
            created_by_user_id=admin_user_id,
        )
    )
    job = None
    if body.index_now:
        job = await _run_service(
            lambda: get_public_code_job_service().submit_index(
                source_ids=(source.source_id,),
                domains=(),
            )
        )
    return {
        "source": asdict(state),
        "index_job": _job_response(job) if job is not None else None,
    }


@router.post(
    "/sources/check-updates",
    status_code=202,
    response_model=PublicCodeJobResponse,
    responses=_AUTH_RESPONSES,
)
async def check_public_code_updates(
    body: PublicCodeSourceSelection,
    request: Request,
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
):
    """Admin-only HEAD probes for one, many, or all configured sources."""

    resolve_admin(request, user_id, tier)
    job = await _run_service(
        lambda: get_public_code_job_service().submit_update_check(
            source_ids=tuple(body.source_ids),
            domains=tuple(body.domains),
        )
    )
    return _job_response(job)


@router.post(
    "/sources/index",
    status_code=202,
    response_model=PublicCodeJobResponse,
    responses=_AUTH_RESPONSES,
)
async def index_public_code_sources(
    body: PublicCodeSourceSelection,
    request: Request,
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
):
    """Queue mirror refresh and candidate indexing for selected public sources."""

    resolve_admin(request, user_id, tier)
    job = await _run_service(
        lambda: get_public_code_job_service().submit_index(
            source_ids=tuple(body.source_ids),
            domains=tuple(body.domains),
        )
    )
    return _job_response(job)


@router.get(
    "/jobs",
    response_model=list[PublicCodeJobResponse],
    responses=_AUTH_RESPONSES,
)
async def list_public_code_jobs(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
):
    """Return recent in-process jobs for management-page polling."""

    resolve_admin(request, user_id, tier)
    jobs = await _run_service(lambda: get_public_code_job_service().list_recent(limit=limit))
    return [_job_response(job) for job in jobs]


@router.get(
    "/jobs/{job_id}",
    response_model=PublicCodeJobResponse,
    responses={**_AUTH_RESPONSES, 404: {"description": "Job not found"}},
)
async def get_public_code_job(
    job_id: str,
    request: Request,
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
):
    """Return a pollable source/file/chunk progress snapshot for one job."""

    resolve_admin(request, user_id, tier)
    job = await _run_service(lambda: get_public_code_job_service().get(job_id))
    if job is None:
        raise HTTPException(status_code=404, detail="public-code job not found")
    return _job_response(job)


@router.put(
    "/sources/{source_id}/update-policy",
    response_model=PublicCodeSourceResponse,
    responses=_AUTH_RESPONSES,
)
async def set_public_code_update_policy(
    source_id: str,
    body: PublicCodeUpdatePolicyRequest,
    request: Request,
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
):
    """Admin-only policy persistence for manual update jobs."""

    resolve_admin(request, user_id, tier)
    state = await _run_service(
        lambda: get_public_code_control_service().set_update_policy(
            source_id=source_id,
            update_policy=body.update_policy,
            update_interval_minutes=body.update_interval_minutes,
        )
    )
    return asdict(state)


@router.post(
    "/search",
    response_model=PublicCodeSearchResponseBody,
    responses=_AUTH_RESPONSES,
)
async def search_public_code(
    body: PublicCodeSearchRequestBody,
    request: Request,
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
):
    """Search one source, multiple sources, or domains as an internal/admin caller."""

    resolve_admin(request, user_id, tier)
    result = await _run_service(
        lambda: get_public_code_search_service().search(
            PublicCodeSearchRequest(
                query=body.query,
                top_k=body.top_k,
                mode=body.mode,
                source_ids=tuple(body.source_ids),
                domains=tuple(body.domains),
                language=body.language,
                rerank=body.rerank,
            )
        )
    )
    return {
        "query": result.query,
        "selected_source_ids": list(result.selected_source_ids),
        "hits": [asdict(hit) for hit in result.hits],
        "requested_mode": result.requested_mode,
        "effective_mode": result.effective_mode,
        "notice": result.notice,
        "rerank_applied": result.rerank_applied,
        "candidate_count": result.candidate_count,
        "rerank_candidate_count": result.rerank_candidate_count,
    }


# ---- bundle import: seed a mirror from an operator's clone instead of a network clone ----

_IMPORT_WRITE_BUFFER_BYTES = 8 * 1024 * 1024
_IMPORT_RESPONSES = {
    **_AUTH_RESPONSES,
    404: {"description": "Import not found"},
    409: {"description": "Conflicts with the upload or source state"},
    422: {"description": "Invalid bundle, commit, checksum or size"},
}


@router.post(
    "/imports",
    status_code=201,
    response_model=PublicCodeImportResponse,
    responses=_IMPORT_RESPONSES,
)
async def create_public_code_import(
    body: PublicCodeImportCreateRequest,
    request: Request,
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
):
    """Declare a `git bundle` that will become a source's bare mirror.

    Create the bundle from a clean clone with `git bundle create <file> HEAD <branch>`,
    upload it with `PUT /imports/{import_id}/data`, then finish with
    `POST /imports/{import_id}/complete`. Chunking, embedding and indexing run on the
    server. An existing source must match source_id, domain, upstream_url,
    relative_path and default_ref; a new one is registered when the import completes.
    With private_project, the server reuses exact chunk vectors from the authenticated
    creator's private Code KB scope; missing or changed chunks are embedded normally."""

    admin_user_id = resolve_admin(request, user_id, tier)

    def create():
        source = PublicCodeSource(
            source_id=body.source_id,
            domain=body.domain,
            relative_path=body.relative_path,
            upstream_url=body.upstream_url,
            default_ref=body.default_ref,
            language_overrides=body.language_overrides,
        )
        get_public_code_access_profiles().validate_source(source)
        return get_public_code_import_service().create(
            source,
            head_commit=body.head_commit,
            size_bytes=body.size_bytes,
            sha256=body.sha256,
            update_policy=body.update_policy,
            update_interval_minutes=body.update_interval_minutes,
            created_by_user_id=admin_user_id,
            private_project=body.private_project,
        )

    return asdict(await _run_service(create))


@router.get(
    "/imports/{import_id}",
    response_model=PublicCodeImportResponse,
    responses=_IMPORT_RESPONSES,
)
async def get_public_code_import(
    import_id: str,
    request: Request,
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
):
    """Upload state; `received_bytes` is the offset to resume from."""

    resolve_admin(request, user_id, tier)
    return asdict(await _run_service(lambda: get_public_code_import_service().get(import_id)))


@router.put(
    "/imports/{import_id}/data",
    response_model=PublicCodeImportResponse,
    responses=_IMPORT_RESPONSES,
    openapi_extra={"requestBody": {
        "required": True,
        "content": {"application/octet-stream": {
            "schema": {"type": "string", "format": "binary"}}},
    }},
)
async def upload_public_code_import_data(
    import_id: str,
    request: Request,
    offset: int = Query(
        ..., ge=0, description="Must equal received_bytes; after a failure resend from there."),
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
):
    """Append raw bundle bytes (`application/octet-stream`) at `offset`.

    Send the bundle in one or several requests; each must start at the current
    received_bytes, so an interrupted upload resumes without resending what arrived."""

    resolve_admin(request, user_id, tier)
    service = get_public_code_import_service()
    position = offset
    buffer = bytearray()
    state = None
    async for chunk in request.stream():
        buffer.extend(chunk)
        if len(buffer) >= _IMPORT_WRITE_BUFFER_BYTES:
            data, buffer = bytes(buffer), bytearray()
            state = await _run_service(lambda: service.write(import_id, position, data))
            position += len(data)
    if buffer or state is None:
        data = bytes(buffer)
        state = await _run_service(lambda: service.write(import_id, position, data))
    return asdict(state)


@router.post(
    "/imports/{import_id}/complete",
    status_code=202,
    response_model=PublicCodeImportCompleteResponse,
    response_model_by_alias=True,
    responses=_IMPORT_RESPONSES,
)
async def complete_public_code_import(
    import_id: str,
    request: Request,
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
):
    """Queue the import job and return its receipt; poll it with `GET /jobs/{job_id}`.

    The job verifies size and sha256, checks that default_ref is head_commit,
    publishes the bare mirror (origin = upstream_url), registers a new source,
    then indexes that revision without fetching upstream. If private_project was
    declared, the creator's matching private vectors are reused; job results report
    reused_chunk_count. Private vectors and owner metadata remain in their own scope."""

    resolve_admin(request, user_id, tier)
    imports = get_public_code_import_service()
    jobs = get_public_code_job_service()
    state = await _run_service(lambda: imports.get(import_id))
    if await _run_service(lambda: jobs.has_active_index(state.source_id)):
        raise HTTPException(
            status_code=409,
            detail=f"public-code source {state.source_id} is already being indexed")
    state = await _run_service(lambda: imports.mark_submitted(import_id))

    def prepare(progress):
        published = imports.publish_mirror(import_id, progress)
        if published.source_exists:
            return
        progress("registering the new source")
        try:
            get_public_code_control_service().register_source(
                published.source(),
                update_policy=published.update_policy,
                update_interval_minutes=published.update_interval_minutes,
                created_by_user_id=published.created_by_user_id,
            )
        except Exception as exc:
            imports.fail(import_id, f"mirror published but registration failed: {exc}")
            raise

    job = await _run_service(
        lambda: jobs.submit_import(
            source_id=state.source_id, prepare=prepare,
            private_owner_user_id=state.created_by_user_id,
            private_project=state.private_project))
    state = await _run_service(lambda: imports.attach_job(import_id, job.job_id))
    return {"import": asdict(state), "job": _job_response(job)}


@router.delete(
    "/imports/{import_id}",
    status_code=204,
    response_class=Response,
    responses=_IMPORT_RESPONSES,
)
async def delete_public_code_import(
    import_id: str,
    request: Request,
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
):
    """Drop an abandoned, failed or finished import and its uploaded bytes."""

    resolve_admin(request, user_id, tier)
    await _run_service(lambda: get_public_code_import_service().discard(import_id))
    return Response(status_code=204)
