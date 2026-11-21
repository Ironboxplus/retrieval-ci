"""Per-user index-inventory reporting (client reports what it indexed, where)."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Header, HTTPException, Query, Request

from app.api.identity import resolve_owner
from app.services.code_kb.jobs import get_existing_code_index_queue
from app.services.index_audit import (
    build_index_audit_record,
    emit_index_audit,
    push_index_audit_bg,
)
from app.services.index_catalog import get_index_catalog_service
from app.services.index_catalog.schemas import (
    IndexReportRequest,
    IndexReportResponse,
    IndexScopePolicyUpsert,
    InventoryResponse,
    ScopeOverlapResponse,
)

router = APIRouter()


@router.post("/policy", response_model=IndexScopePolicyUpsert)
async def upsert_index_policy(
    body: IndexScopePolicyUpsert,
    request: Request,
    user_id: str | None = Header(default=None, alias="X-Retrival-User-Id"),
):
    owner = resolve_owner(request, user_id)
    try:
        return get_index_catalog_service().upsert_policy(body, owner_user_id=owner)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/policy", response_model=IndexScopePolicyUpsert)
async def get_index_policy(
    request: Request,
    host: str = Query(..., min_length=1, max_length=200),
    project: str = Query(..., min_length=1, max_length=2000),
    kind: str = Query("code", min_length=1, max_length=50),
    user_id: str | None = Header(default=None, alias="X-Retrival-User-Id"),
):
    owner = resolve_owner(request, user_id)
    policy = get_index_catalog_service().get_policy(
        owner_user_id=owner, host=host, project=project, kind=kind)
    if policy is None:
        raise HTTPException(status_code=404, detail="index policy not found")
    return policy


@router.post("/report", response_model=IndexReportResponse)
async def report_index(
    body: IndexReportRequest,
    request: Request,
    user_id: str | None = Header(default=None, alias="X-Retrival-User-Id"),
):
    """The machine where the code lives reports its indexed-file inventory.

    Owner is taken from the gateway identity (or internal-admin on LAN) — the
    request body's ``owner_user_id`` is never trusted.
    """
    owner_user_id = resolve_owner(request, user_id)
    # Faithful audit at RECEIPT (W172): a doc/code index report is an index request —
    # record it the moment it arrives so LAN/MCP indexing lands in the audit log too.
    kind = body.files[0].kind if body.files else "doc"
    rec = emit_index_audit(build_index_audit_record(
        event="doc_index_reported", kind=kind, owner_user_id=owner_user_id,
        project=body.project, host=body.host, files=len(body.files),
        replace=body.replace))
    push_index_audit_bg(rec)
    return get_index_catalog_service().report(body, owner_user_id=owner_user_id)


@router.get("/inventory", response_model=InventoryResponse)
async def my_inventory(
    request: Request,
    host: str | None = Query(default=None),
    project: str | None = Query(default=None),
    kind: str | None = Query(default=None),
    detail: bool = Query(
        default=False,
        description="Include the per-dir/per-file tree (with content hashes). Default False "
                    "= compact summary (project counts + freshness only); the full tree is "
                    "~10MB for a heavy indexer and the inventory tool never renders files.",
    ),
    user_id: str | None = Header(default=None, alias="X-Retrival-User-Id"),
):
    """The caller's own indexed-file inventory tree (compact summary by default)."""
    owner_user_id = resolve_owner(request, user_id)

    def load_inventory() -> InventoryResponse:
        # Do not initialize a Qdrant-backed queue merely to render a read-only inventory.
        # If an index route did initialize it, overlay its active Code KB state so a
        # registered/preparing scope never looks equivalent to a completed catalog report.
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


@router.get("/scope-overlap", response_model=ScopeOverlapResponse)
async def scope_overlap(
    request: Request,
    project: str = Query(..., min_length=1, max_length=2000),
    host: str | None = Query(default=None),
    kind: str | None = Query(default=None),
    respect_gitignore: bool | None = Query(default=None),
    user_id: str | None = Header(default=None, alias="X-Retrival-User-Id"),
):
    """Parent/child overlap for a candidate index ``project`` scope, answered in KB.

    The MCP nested-scope guard calls this before indexing a directory: it returns the
    caller's existing ancestor scopes that would make the candidate redundant
    (``ancestor_blocks``) and existing descendant scopes (``child_warnings``) — replacing
    a per-call download of the entire per-file inventory (which grew to multi-MB and began
    truncating over the LAN overlay).

    ``respect_gitignore`` is the filter the candidate index will run under. Omitted by
    clients that predate it, which then see exactly the behaviour they shipped against."""
    owner_user_id = resolve_owner(request, user_id)
    return get_index_catalog_service().scope_overlap(
        owner_user_id=owner_user_id, project=project, host=host, kind=kind,
        respect_gitignore=respect_gitignore,
    )
