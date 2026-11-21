"""Admin-only management console (cross-user journal + index inventory).

All routes are admin-gated: gateway must inject ``X-Retrival-User-Tier: admin``, or
the caller is a LAN-trusted internal-admin (direct 4090:8000 panel). Non-admins get
403. ``owner_user_id`` query params let an admin scope to one user; omit for all.
"""
from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.api.identity import resolve_admin
from app.services.code_kb.jobs import get_existing_code_index_queue
from app.services.index_catalog import get_index_catalog_service
from app.services.journal import get_journal_service
from app.services.journal.schemas import (
    JournalDeleteResponse,
    JournalEntryListResponse,
    JournalEntryResponse,
    JournalSearchResponse,
)
from app.services.playbook import get_playbook_service
from app.services.playbook.schemas import PlaybookDeleteResponse, PlaybookListResponse

router = APIRouter()

_UserId = Header(default=None, alias="X-Retrival-User-Id")
_Tier = Header(default=None, alias="X-Retrival-User-Tier")


def _guard(request: Request, user_id: str | None, tier: str | None) -> str:
    return resolve_admin(request, user_id, tier)


class AdminJournalSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: Literal["keyword", "hybrid", "dense", "sparse"] = "keyword"
    top_k: int = Field(default=20, ge=1, le=100)
    owner_user_id: str | None = None
    project_slug: str | None = None
    source_kind: str | None = None
    include_chunks: bool = False
    chunk_limit: int | None = Field(default=5, ge=1, le=20)
    rerank: bool = False


@router.get("/overview")
async def overview(request: Request, user_id: str | None = _UserId, tier: str | None = _Tier) -> dict[str, Any]:
    _guard(request, user_id, tier)
    return {
        "journal": get_journal_service().stats(),
        "index": get_index_catalog_service().overview(),
    }


@router.get("/users")
async def users(request: Request, user_id: str | None = _UserId, tier: str | None = _Tier) -> dict[str, Any]:
    """Distinct owners across journal + index, with per-owner counts."""
    _guard(request, user_id, tier)
    index_owners = {o.owner_user_id: o for o in get_index_catalog_service().owners()}
    journal_owners = set(get_journal_service().owner_ids())
    merged: list[dict[str, Any]] = []
    for owner in sorted(set(index_owners) | journal_owners):
        summary = index_owners.get(owner)
        merged.append({
            "owner_user_id": owner,
            "index_files": summary.file_count if summary else 0,
            "index_chunks": summary.chunk_count if summary else 0,
            "index_hosts": summary.host_count if summary else 0,
            "index_projects": summary.project_count if summary else 0,
            "has_journal": owner in journal_owners,
        })
    return {"users": merged, "total": len(merged)}


@router.get("/journal/entries", response_model=JournalEntryListResponse)
async def list_entries(
    request: Request,
    owner_user_id: str | None = Query(default=None),
    project_slug: str | None = Query(default=None),
    source_kind: str | None = Query(default=None),
    q: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
):
    _guard(request, user_id, tier)
    return get_journal_service().admin_list_entries(
        owner_user_id=owner_user_id, project_slug=project_slug,
        source_kind=source_kind, q=q, limit=limit, offset=offset,
    )


@router.post("/journal/search", response_model=JournalSearchResponse)
async def search_entries(
    body: AdminJournalSearchRequest,
    request: Request,
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
):
    _guard(request, user_id, tier)
    return get_journal_service().admin_search(
        query=body.query, mode=body.mode, top_k=body.top_k,
        owner_user_id=body.owner_user_id, project_slug=body.project_slug,
        source_kind=body.source_kind, include_chunks=body.include_chunks,
        chunk_limit=body.chunk_limit, rerank=body.rerank,
    )


@router.get("/journal/entries/{entry_id}", response_model=JournalEntryResponse)
async def get_entry(
    entry_id: str,
    request: Request,
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
):
    _guard(request, user_id, tier)
    entry = get_journal_service().get_entry(entry_id)  # owner_user_id=None → any user
    if entry is None:
        raise HTTPException(status_code=404, detail="journal entry not found")
    return entry


@router.delete("/journal/entries/{entry_id}", response_model=JournalDeleteResponse)
async def delete_entry(
    entry_id: str,
    request: Request,
    reason: str | None = Query(default=None, max_length=500),
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
):
    _guard(request, user_id, tier)
    result = get_journal_service().delete_entry(entry_id, reason=reason)
    if not result.deleted:
        raise HTTPException(status_code=404, detail="journal entry not found")
    return result


@router.get("/journal/pending", response_model=JournalEntryListResponse)
async def journal_pending(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
):
    """The reindex queue: live entries whose vectors are missing (indexed_at NULL),
    e.g. recorded while the embed endpoint was down. Poll for a live view."""
    _guard(request, user_id, tier)
    return get_journal_service().pending_entries(limit=limit)


@router.post("/journal/reindex")
async def journal_reindex(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
) -> dict[str, int]:
    """Drain the reindex queue: re-embed + upsert pending entries so SQL and the
    vector index reconcile. Returns {reindexed, failed, remaining}. Idempotent."""
    _guard(request, user_id, tier)
    return get_journal_service().reindex_pending(limit=limit)


@router.get("/playbook/bullets", response_model=PlaybookListResponse)
async def playbook_bullets(
    request: Request,
    owner_user_id: str | None = Query(default=None),
    project_slug: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=1000),
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
):
    """Cross-user ACE playbook listing (omit owner_user_id for all users)."""
    _guard(request, user_id, tier)
    return get_playbook_service().admin_list(
        owner_user_id=owner_user_id, project_slug=project_slug, limit=limit,
    )


@router.delete("/playbook/bullets/{bullet_id}", response_model=PlaybookDeleteResponse)
async def delete_playbook_bullet(
    bullet_id: int,
    request: Request,
    owner_user_id: str = Query(..., description="owner of the bullet to delete"),
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
):
    _guard(request, user_id, tier)
    result = get_playbook_service().delete(bullet_id, owner_user_id=owner_user_id)
    if not result.deleted:
        raise HTTPException(status_code=404, detail="bullet not found")
    return result


@router.get("/index/inventory")
async def index_inventory(
    request: Request,
    owner_user_id: str | None = Query(default=None),
    host: str | None = Query(default=None),
    project: str | None = Query(default=None),
    kind: str | None = Query(default=None),
    detail: bool = Query(
        default=False,
        description=(
            "Include per-directory and per-file nodes. The default project summary "
            "keeps large cross-user inventories bounded; request detail only after "
            "selecting a project."
        ),
    ),
    user_id: str | None = _UserId,
    tier: str | None = _Tier,
):
    _guard(request, user_id, tier)

    def load_inventory():
        queue = get_existing_code_index_queue() if kind in (None, "code") else None
        code_scope_states = queue.inventory_scope_states(
            owner_user_id=owner_user_id, host=host, project=project,
        ) if queue is not None else None
        return get_index_catalog_service().inventory(
            owner_user_id=owner_user_id,
            host=host,
            project=project,
            kind=kind,
            detail=detail,
            code_scope_states=code_scope_states,
        )

    return await asyncio.to_thread(load_inventory)
