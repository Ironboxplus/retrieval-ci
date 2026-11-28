"""Per-user ACE playbook routes (owner from gateway or internal-admin on LAN)."""
from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Query, Request

from app.api.identity import resolve_owner
from app.services.playbook import get_playbook_service
from app.services.playbook.schemas import (
    PlaybookBulletResponse,
    PlaybookDeleteResponse,
    PlaybookListResponse,
    PlaybookMarkRequest,
    PlaybookRecordRequest,
    PlaybookSearchRequest,
)

router = APIRouter()

_UserId = Header(default=None, alias="X-Retrival-User-Id")


@router.post("/bullets", response_model=PlaybookBulletResponse)
async def record_bullet(body: PlaybookRecordRequest, request: Request, user_id: str | None = _UserId):
    """Curate a lesson into the caller's playbook (grow-and-refine dedup)."""
    owner = resolve_owner(request, user_id)
    try:
        return get_playbook_service().record(body, owner_user_id=owner)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/bullets", response_model=PlaybookListResponse)
async def list_bullets(
    request: Request,
    project_slug: str | None = Query(default=None),
    domain: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=500),
    user_id: str | None = _UserId,
):
    owner = resolve_owner(request, user_id)
    return get_playbook_service().list_bullets(
        owner_user_id=owner, project_slug=project_slug, domain=domain, limit=limit,
    )


@router.post("/search", response_model=PlaybookListResponse)
async def search_bullets(body: PlaybookSearchRequest, request: Request, user_id: str | None = _UserId):
    owner = resolve_owner(request, user_id)
    return get_playbook_service().search(body, owner_user_id=owner)


@router.post("/bullets/{bullet_id}/mark", response_model=PlaybookBulletResponse)
async def mark_bullet(bullet_id: int, body: PlaybookMarkRequest, request: Request, user_id: str | None = _UserId):
    owner = resolve_owner(request, user_id)
    result = get_playbook_service().mark(bullet_id, helpful=body.helpful, owner_user_id=owner)
    if result is None:
        raise HTTPException(status_code=404, detail="bullet not found")
    return result


@router.delete("/bullets/{bullet_id}", response_model=PlaybookDeleteResponse)
async def delete_bullet(bullet_id: int, request: Request, user_id: str | None = _UserId):
    owner = resolve_owner(request, user_id)
    result = get_playbook_service().delete(bullet_id, owner_user_id=owner)
    if not result.deleted:
        raise HTTPException(status_code=404, detail="bullet not found")
    return result
