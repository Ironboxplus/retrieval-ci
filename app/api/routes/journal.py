"""ACE journal API routes (per-user; owner from gateway or internal-admin on LAN)."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Header, HTTPException, Query, Request

from app.api.identity import resolve_owner
from app.services.index_catalog.service import get_index_catalog_service
from app.services.journal import get_journal_service
from app.services.journal.jobs import JournalBulkIndexJob, get_journal_index_queue
from app.services.journal.schemas import (
    JournalDeleteResponse,
    JournalBulkIndexJobOut,
    JournalBulkIndexRequest,
    JournalDocReconcileRequest,
    JournalDocReconcileResponse,
    JournalDocScopeDeleteResponse,
    JournalEntryListResponse,
    JournalEntryResponse,
    JournalRecordRequest,
    JournalRecordResponse,
    JournalSearchRequest,
    JournalSearchResponse,
)
from app.services.scope_paths import canonicalize_scope_path, is_windows_scope

router = APIRouter()


def _bulk_job_out(job: JournalBulkIndexJob) -> JournalBulkIndexJobOut:
    return JournalBulkIndexJobOut(
        job_id=job.job_id, project_slug=job.project_slug, host=job.host,
        status=job.status.value, files_total=job.files_total, files_done=job.files_done,
        chunks_total=job.chunks_total, chunks_done=job.chunks_done, indexed=job.indexed,
        progress=job.progress, queue_position=job.queue_position, error=job.error,
        created_at=job.created_at, started_at=job.started_at, finished_at=job.finished_at)


def _attach_gateway_identity(
    request: JournalRecordRequest,
    *,
    user_id: str,
    username: str | None,
    tier: str | None,
) -> JournalRecordRequest:
    metadata = dict(request.metadata or {})
    if username:
        metadata["gateway_username"] = username
    if tier:
        metadata["gateway_tier"] = tier
    return request.model_copy(update={"owner_user_id": user_id, "metadata": metadata})


@router.post("/entries", response_model=JournalRecordResponse)
async def record_journal_entry(
    body: JournalRecordRequest,
    request: Request,
    user_id: str | None = Header(default=None, alias="X-Retrival-User-Id"),
    username: str | None = Header(default=None, alias="X-Retrival-Username"),
    tier: str | None = Header(default=None, alias="X-Retrival-User-Tier"),
):
    user_id = resolve_owner(request, user_id)
    body = _attach_gateway_identity(body, user_id=user_id, username=username, tier=tier)
    try:
        return await asyncio.to_thread(get_journal_service().record, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/bulk-index", response_model=JournalBulkIndexJobOut, status_code=202)
async def bulk_index_journal_docs(
    body: JournalBulkIndexRequest, request: Request,
    user_id: str | None = Header(default=None, alias="X-Retrival-User-Id"),
):
    """Queue changed Journal files; polling keeps MCP stdio responsive during embeds."""
    owner = resolve_owner(request, user_id)
    try:
        job = await asyncio.to_thread(get_journal_index_queue().submit, body, owner_user_id=owner)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _bulk_job_out(job)


@router.get("/bulk-index/jobs/{job_id}", response_model=JournalBulkIndexJobOut)
async def get_bulk_journal_index_job(
    job_id: str, request: Request,
    user_id: str | None = Header(default=None, alias="X-Retrival-User-Id"),
):
    owner = resolve_owner(request, user_id)
    job = get_journal_index_queue().get(job_id, owner_user_id=owner)
    if job is None:
        raise HTTPException(status_code=404, detail="journal bulk index job not found")
    return _bulk_job_out(job)


@router.get("/entries", response_model=JournalEntryListResponse)
async def list_journal_entries(
    request: Request,
    project_slug: str = Query(..., min_length=1),
    source_kind: str | None = Query(default=None),
    client: str | None = Query(default=None),
    tag: str | None = Query(default=None),
    q: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user_id: str | None = Header(default=None, alias="X-Retrival-User-Id"),
):
    user_id = resolve_owner(request, user_id)
    project_slug = canonicalize_scope_path(project_slug)
    return await asyncio.to_thread(
        get_journal_service().list_entries,
        project_slug=project_slug,
        source_kind=source_kind,
        client=client,
        tag=tag,
        q=q,
        owner_user_id=user_id,
        limit=limit,
        offset=offset,
    )


@router.post("/docs/reconcile", response_model=JournalDocReconcileResponse)
async def reconcile_journal_docs(
    body: JournalDocReconcileRequest,
    request: Request,
    user_id: str | None = Header(default=None, alias="X-Retrival-User-Id"),
):
    user_id = resolve_owner(request, user_id)
    return await asyncio.to_thread(
        get_journal_service().reconcile_doc_scope,
        project_slug=body.project_slug,
        live_titles=body.live_titles,
        owner_user_id=user_id,
        max_depth=body.max_depth,
        patterns=body.patterns,
    )


@router.delete("/docs", response_model=JournalDocScopeDeleteResponse)
async def delete_journal_docs(
    request: Request,
    project_slug: str = Query(..., min_length=1, max_length=2000),
    user_id: str | None = Header(default=None, alias="X-Retrival-User-Id"),
):
    """Clear every batch-indexed doc (source_kind="doc") in a scope: drop ALL its journal
    entries + vectors AND its doc inventory rows — the symmetric counterpart to
    `DELETE /api/code`. `project_slug` is the indexed directory's absolute path (query param
    because a path segment can't hold slashes / a drive colon). ACE memory entries in the
    scope are left untouched. Idempotent: an empty scope removes nothing."""
    user_id = resolve_owner(request, user_id)
    project_slug = canonicalize_scope_path(project_slug)
    def delete_scope() -> JournalDocScopeDeleteResponse:
        catalog = get_index_catalog_service()
        scopes = [project_slug]
        if is_windows_scope(project_slug):
            # Resolve all catalog aliases before the first mutation so an alias lookup
            # failure cannot leave Journal and inventory in a half-deleted state.
            scopes = catalog.equivalent_project_scopes(
                owner_user_id=user_id, project=project_slug, kind="doc")
        scopes = list(dict.fromkeys(scope for scope in scopes if scope))
        resp = get_journal_service().delete_doc_scope(
            project_slug=project_slug, owner_user_id=user_id)
        resp.catalog_removed = sum(
            catalog.delete_project(
                owner_user_id=user_id, project=scope, kind="doc")
            for scope in scopes
        )
        resp.scope_aliases = list(dict.fromkeys([
            *resp.scope_aliases,
            *(scope for scope in scopes if scope != project_slug),
        ]))
        return resp

    return await asyncio.to_thread(delete_scope)


@router.get("/entries/{entry_id}", response_model=JournalEntryResponse)
async def get_journal_entry(
    entry_id: str,
    request: Request,
    user_id: str | None = Header(default=None, alias="X-Retrival-User-Id"),
):
    user_id = resolve_owner(request, user_id)
    entry = await asyncio.to_thread(
        get_journal_service().get_entry, entry_id, owner_user_id=user_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="journal entry not found")
    return entry


@router.delete("/entries/{entry_id}", response_model=JournalDeleteResponse)
async def delete_journal_entry(
    entry_id: str,
    request: Request,
    reason: str | None = Query(default=None, max_length=500),
    user_id: str | None = Header(default=None, alias="X-Retrival-User-Id"),
):
    user_id = resolve_owner(request, user_id)
    result = await asyncio.to_thread(
        get_journal_service().delete_entry,
        entry_id,
        reason=reason,
        owner_user_id=user_id,
    )
    if not result.deleted:
        raise HTTPException(status_code=404, detail="journal entry not found")
    return result


@router.post("/search", response_model=JournalSearchResponse)
async def search_journal(
    body: JournalSearchRequest,
    request: Request,
    user_id: str | None = Header(default=None, alias="X-Retrival-User-Id"),
):
    user_id = resolve_owner(request, user_id)
    body = body.model_copy(update={"owner_user_id": user_id})
    return await asyncio.to_thread(get_journal_service().search, body)
