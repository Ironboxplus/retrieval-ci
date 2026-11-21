"""Gateway routes: OIDC login + sk-key management + metered compute proxy +
payment (mock) + admin. All sensitive state stays here on the web backend; the
compute routes forward to the 4090 with an internal token and nothing else.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import secrets
from collections.abc import Awaitable, Callable
from decimal import Decimal, InvalidOperation
from typing import Literal, TypeVar
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import (JSONResponse, PlainTextResponse, RedirectResponse,
                               Response, StreamingResponse)
from pydantic import BaseModel, Field, ValidationError, field_validator

from gateway.api.deps import (get_current_user, get_db, get_provider, get_proxy,
                              get_settings, require_admin, require_internal)
from gateway.auth import accounts, oauth2, oidc
from gateway.auth.deps import is_billable, is_internal
from gateway.billing import ledger, metering
from gateway.db.models import ApiKey, Tier, User
from gateway.payment.service import handle_notify, start_topup
from gateway.proxy import ComputeError, attach_user_headers

logger = logging.getLogger("gateway.api.routes")
router = APIRouter()
MeteredResult = TypeVar("MeteredResult")


class _BillableUpstreamFailure(Exception):
    """Trigger a metering refund while preserving the upstream response."""


def _require_successful_status(status_code: int) -> None:
    if not 200 <= status_code < 300:
        raise _BillableUpstreamFailure


def _api_idempotency_key(request: Request, user) -> str | None:
    return metering.api_idempotency_key(
        user.id, request.headers.get("X-Idempotency-Key"))


def _idempotency_conflict() -> HTTPException:
    return HTTPException(
        status_code=409,
        detail="X-Idempotency-Key was already used for a different request",
    )


async def _run_metered_proxy(
    request: Request,
    user,
    db,
    *,
    cost,
    ref: str,
    forward: Callable[[], Awaitable[MeteredResult]],
    status_code: Callable[[MeteredResult], int],
) -> MeteredResult:
    """Run one upstream call under the gateway's complete billing protocol."""
    request.state.billed_cost = cost if is_billable(user) else 0
    idempotency_key = _api_idempotency_key(request, user)
    try:
        with metering.charge(
            db,
            user,
            cost=cost,
            ref=ref,
            idempotency_key=idempotency_key,
        ) as receipt:
            request.state.billed_cost = cost if receipt.charged else 0
            result = await forward()
            _require_successful_status(status_code(result))
    except ledger.InsufficientCredits:
        request.state.billed_cost = 0
        raise HTTPException(status_code=402, detail="insufficient credits")
    except ledger.IdempotencyConflict:
        request.state.billed_cost = 0
        raise _idempotency_conflict()
    except ledger.IdempotencyInProgress:
        request.state.billed_cost = 0
        raise HTTPException(
            status_code=425,
            detail="A request with this X-Idempotency-Key is still in progress",
        )
    except _BillableUpstreamFailure:
        request.state.billed_cost = 0
    except ComputeError as exc:
        request.state.billed_cost = 0
        raise HTTPException(status_code=502, detail=f"compute error {exc.status_code}")
    except Exception:
        request.state.billed_cost = 0
        raise
    return result


# ============================ compute proxy (metered) ========================

async def _metered_forward(request: Request, path: str, user, db, settings, proxy, *,
                           cost=None, forward_headers=None) -> Response:
    """Charge `cost` (default settings.search_cost) then forward. `forward_headers`,
    if given, is applied to attach explicit identity headers (private per-user APIs
    like journal / code KB); otherwise the client's headers are forwarded as-is
    (public compute APIs, which don't need per-owner scoping)."""
    body = await request.body()
    cost = settings.search_cost if cost is None else cost
    headers = forward_headers(dict(request.headers), user) if forward_headers else request.headers
    billing_ref = metering.request_ref(request.method, path, body)
    resp = await _run_metered_proxy(
        request,
        user,
        db,
        cost=cost,
        ref=billing_ref,
        forward=lambda: proxy.request(
            request.method,
            path,
            headers=headers,
            params=request.query_params,
            content=body,
        ),
        status_code=lambda response: response.status_code,
    )
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=resp.headers.get("content-type"))


@router.post("/api/search")
async def search(request: Request, user=Depends(get_current_user), db=Depends(get_db),
                 settings=Depends(get_settings), proxy=Depends(get_proxy)):
    return await _metered_forward(request, "/api/search", user, db, settings, proxy)


@router.post("/api/match-doc")
async def match_doc(request: Request, user=Depends(get_current_user), db=Depends(get_db),
                    settings=Depends(get_settings), proxy=Depends(get_proxy)):
    return await _metered_forward(request, "/api/match-doc", user, db, settings, proxy)


@router.post("/api/search/stream")
async def search_stream(request: Request, user=Depends(get_current_user), db=Depends(get_db),
                        settings=Depends(get_settings), proxy=Depends(get_proxy)):
    body = await request.body()
    billing_ref = metering.request_ref("POST", "/api/search/stream", body)
    # Charge on successful initiation; a 5xx during connect refunds. We can't refund
    # a mid-stream failure (response already started) — accepted for v1.
    status, headers, aiter = await _run_metered_proxy(
        request,
        user,
        db,
        cost=settings.search_cost,
        ref=billing_ref,
        forward=lambda: proxy.stream(
                "POST", "/api/search/stream", headers=request.headers,
                params=request.query_params, content=body),
        status_code=lambda streamed: streamed[0],
    )
    return StreamingResponse(aiter, status_code=status,
                             media_type=headers.get("content-type", "text/event-stream"))


# Read-only aggregate endpoints (conferences / corpus stats / venue·year distribution):
# LOGIN-GATED, NOT metered. Anonymous → 401. (W127b: the operator chose to gate ALL
# data including the public aggregates — nothing is anonymous except the login flow.)
async def _gated_get(request: Request, path: str, proxy) -> Response:
    resp = await proxy.request("GET", path, headers=request.headers,
                               params=request.query_params)
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=resp.headers.get("content-type"))


async def _journal_forward(request: Request, path: str, user, proxy) -> Response:
    """Private-memory proxy (journal / code KB / index inventory / ACE playbook).

    This data is PRIVATE per-user memory: the request is forwarded with explicit,
    gateway-authenticated identity headers (client-supplied ones stripped, W147) so the
    compute tier scopes strictly to that owner — user A can never see user B's code /
    journal / inventory. Journal + code + index are open to any logged-in user (paying+);
    ACE playbook stays invite-only (internal+admin). This variant does not meter — used
    for free reads (search / GET / index bookkeeping) and billing-exempt tiers.
    """
    body = await request.body()
    headers = attach_user_headers(dict(request.headers), user)
    resp = await proxy.request(request.method, path, headers=headers,
                               params=request.query_params, content=body)
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=resp.headers.get("content-type"))


@router.api_route("/api/conferences", methods=["GET"])
async def conferences(request: Request, user=Depends(get_current_user),
                      proxy=Depends(get_proxy)):
    return await _gated_get(request, "/api/conferences", proxy)


@router.api_route("/api/stats", methods=["GET"])
async def stats(request: Request, user=Depends(get_current_user),
                proxy=Depends(get_proxy)):
    return await _gated_get(request, "/api/stats", proxy)


@router.api_route("/api/dashboard/distribution", methods=["GET"])
async def dashboard_distribution(request: Request, user=Depends(get_current_user),
                                 proxy=Depends(get_proxy)):
    return await _gated_get(request, "/api/dashboard/distribution", proxy)


_SCRUB_KEYS = {"txt_path", "pdf_path"}
_PAPER_FULLTEXT_SCRUB_KEYS = _SCRUB_KEYS | {"path"}
# Documents carry more internal-path fields than papers do (raw source file locations +
# the internal download URLs the read-only public surface must not expose).
_DOC_SCRUB_KEYS = _SCRUB_KEYS | {"file_path", "source_file_path",
                                 "download_url", "source_download_url"}


def _scrub_internal_paths(obj, keys=_SCRUB_KEYS):
    """Recursively drop internal filesystem path fields (txt_path/pdf_path — SMB/UNC
    infra paths) from a public API payload: useless to clients, leaks infrastructure."""
    if isinstance(obj, dict):
        return {k: _scrub_internal_paths(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [_scrub_internal_paths(x, keys) for x in obj]
    return obj


# Paper library: the catalog list (browse/batch) is a FREE login-gated read; opening
# a specific paper (/api/papers/{paper_id}) is METERED for paying users (paper_view_cost,
# 0.1) — member+ (internal/admin) is exempt. Sub-resources of an already-opened paper
# ({id}/fulltext, /pdf, /bibtex, /index-status) are free — you pay to open, not per byte.
# Anonymous → 401. JSON responses are scrubbed of internal filesystem paths.
# Catalog / metadata endpoints (list, filter facets, corpus stats) are FREE reads —
# only opening a specific paper is a metered "view". Keep this in sync with the
# non-`{paper_id}` GET routes the compute tier exposes under /api/papers.
_FREE_PAPER_ENDPOINTS = {"", "browse", "batch", "tags", "stats"}
_MEMBER_PAPER_ARTIFACTS = {"pdf", "both"}
_FILE_STREAM_HEADERS = {
    "accept-ranges",
    "cache-control",
    "content-disposition",
    "content-encoding",
    "content-length",
    "content-range",
    "etag",
    "last-modified",
    "retry-after",
    "x-retrieval-artifact-count",
    "x-retrieval-error-count",
}


def _scrub_response(resp, keys=_SCRUB_KEYS) -> Response:
    ctype = resp.headers.get("content-type", "")
    if ctype.startswith("application/json"):
        try:
            return JSONResponse(_scrub_internal_paths(json.loads(resp.content), keys),
                                status_code=resp.status_code)
        except (ValueError, TypeError):
            pass  # non-JSON body despite the header → pass through untouched
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=ctype or None)


def _streamed_file_response(status: int, upstream_headers: dict, body) -> StreamingResponse:
    response_headers = {
        key: value for key, value in upstream_headers.items()
        if key.lower() in _FILE_STREAM_HEADERS
    }
    return StreamingResponse(
        body,
        status_code=status,
        media_type=upstream_headers.get("content-type", "application/octet-stream"),
        headers=response_headers,
    )


class PaperBatchAccessRequest(BaseModel):
    """Gateway-side trust-boundary validation for the compute batch contract."""

    paper_ids: list[str] = Field(min_length=1, max_length=10)
    artifacts: Literal["pdf", "markdown", "both"] = "pdf"

    @field_validator("paper_ids")
    @classmethod
    def normalize_paper_ids(cls, paper_ids: list[str]) -> list[str]:
        normalized = [paper_id.strip() for paper_id in paper_ids]
        if any(not paper_id for paper_id in normalized):
            raise ValueError("paper_ids cannot contain blank values")
        if any(len(paper_id) > 512 for paper_id in normalized):
            raise ValueError("paper_ids cannot exceed 512 characters")
        return normalized


_PAPER_BATCH_MAX_BODY_BYTES = 16 * 1024


async def _read_limited_request_body(request: Request, limit: int) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > limit:
                raise HTTPException(status_code=413, detail="Request body is too large")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header")

    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise HTTPException(status_code=413, detail="Request body is too large")
    return bytes(body)


@router.post("/api/papers/download-batch")
async def papers_download_batch(request: Request, user=Depends(get_current_user),
                                db=Depends(get_db), settings=Depends(get_settings),
                                proxy=Depends(get_proxy)):
    """Stream an authenticated batch archive without buffering it on the gateway."""
    raw_content = await _read_limited_request_body(
        request, _PAPER_BATCH_MAX_BODY_BYTES)
    try:
        payload = PaperBatchAccessRequest.model_validate_json(raw_content)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="Invalid batch request") from exc
    content = payload.model_dump_json().encode("utf-8")
    artifacts = payload.artifacts
    paper_count = len(set(payload.paper_ids))
    if artifacts in _MEMBER_PAPER_ARTIFACTS and not is_internal(user):
        raise HTTPException(status_code=403, detail="PDF downloads require member access")
    cost = (settings.paper_view_cost * paper_count
            if artifacts == "markdown" and is_billable(user) else Decimal("0"))
    billing_ref = metering.request_ref(
        "POST", "/api/papers/download-batch", content)
    status, upstream_headers, body = await _run_metered_proxy(
        request,
        user,
        db,
        cost=cost,
        ref=billing_ref,
        forward=lambda: proxy.stream(
                "POST",
                "/api/papers/download-batch",
                headers=request.headers,
                params=request.query_params,
                content=content,
            ),
        status_code=lambda streamed: streamed[0],
    )
    return _streamed_file_response(status, upstream_headers, body)


@router.api_route("/api/papers/{rest:path}", methods=["GET"])
async def papers(rest: str, request: Request, user=Depends(get_current_user),
                 db=Depends(get_db), settings=Depends(get_settings),
                 proxy=Depends(get_proxy)):
    path = f"/api/papers/{rest}"
    paper_id, separator, subresource = rest.rpartition("/")
    is_paper_subresource = bool(separator and paper_id)
    if is_paper_subresource and subresource == "pdf":
        if not is_internal(user):
            raise HTTPException(status_code=403, detail="PDF downloads require member access")
        try:
            status, upstream_headers, body = await proxy.stream(
                "GET", path, headers=request.headers, params=request.query_params)
        except ComputeError as exc:
            raise HTTPException(status_code=502, detail=f"compute error {exc.status_code}")
        return _streamed_file_response(status, upstream_headers, body)
    if is_paper_subresource and subresource == "markdown":
        cost = settings.paper_view_cost if is_billable(user) else Decimal("0")
        billing_ref = metering.request_ref("GET", path)
        status, upstream_headers, body = await _run_metered_proxy(
            request,
            user,
            db,
            cost=cost,
            ref=billing_ref,
            forward=lambda: proxy.stream(
                "GET", path, headers=request.headers, params=request.query_params),
            status_code=lambda streamed: streamed[0],
        )
        return _streamed_file_response(status, upstream_headers, body)
    # A bare single segment that isn't a catalog endpoint == opening one paper → meter.
    is_detail = "/" not in rest and rest not in _FREE_PAPER_ENDPOINTS
    is_paid_markdown_view = is_paper_subresource and subresource == "fulltext"
    if (is_detail or is_paid_markdown_view) and is_billable(user):
        billing_ref = metering.request_ref("GET", path)
        resp = await _run_metered_proxy(
            request,
            user,
            db,
            cost=settings.paper_view_cost,
            ref=billing_ref,
            forward=lambda: proxy.request(
                "GET", path, headers=request.headers, params=request.query_params),
            status_code=lambda response: response.status_code,
        )
        return _scrub_response(
            resp,
            _PAPER_FULLTEXT_SCRUB_KEYS if is_paid_markdown_view else _SCRUB_KEYS,
        )
    resp = await proxy.request("GET", path, headers=request.headers,
                               params=request.query_params)
    return _scrub_response(resp)


# Document library (the SHARED corpus: papers / webpages / uploads). READ-ONLY on the
# public edge — only GET is routed here, so upload / edit / delete / crawl (POST/PUT/
# DELETE) never reach compute (405). Login-gated, FREE to browse (metadata + chunk
# previews); opening a *paper* still meters via /api/papers. Only metadata sub-routes
# are whitelisted — raw source-file bytes (/{id}/download, /{id}/source-download) are
# NOT exposed publicly (403). JSON is scrubbed of internal FS paths + internal download
# URLs. Not identity-forwarded: the corpus is shared, not per-owner.
_DOC_ALLOWED = {"", "index-status"}  # sub-route after the doc id ("" = detail itself)


def _document_read_allowed(rest: str) -> bool:
    """Whitelist the read-only metadata surface: list (""), detail ("{id}"),
    index-status ("{id}/index-status"). Everything else — download / source-download /
    any future file route — is blocked, so a blacklist can't be out-run."""
    if rest == "":
        return True  # list
    head, _, tail = rest.partition("/")
    return head.isdigit() and tail in _DOC_ALLOWED


@router.api_route("/api/documents", methods=["GET"])
@router.api_route("/api/documents/{rest:path}", methods=["GET"])
async def documents(request: Request, rest: str = "", user=Depends(get_current_user),
                    proxy=Depends(get_proxy)):
    if not _document_read_allowed(rest):
        raise HTTPException(status_code=403, detail="not available on the public document surface")
    # Forward the path verbatim — the compute list route is /api/documents (no trailing
    # slash; the slash variant hits the SPA catch-all).
    resp = await proxy.request("GET", request.url.path, headers=request.headers,
                               params=request.query_params)
    return _scrub_response(resp, _DOC_SCRUB_KEYS)


# Journal (per-user text memory). Open to any logged-in user (paying+). Writing an
# entry (record / batch index) costs journal_write_cost (0.1) for paying users; search
# + GET reads are free. Identity-forwarded so each user sees only their own.
@router.api_route("/api/journal/entries", methods=["GET", "POST"])
async def journal_entries(request: Request, user=Depends(get_current_user),
                          db=Depends(get_db), settings=Depends(get_settings),
                          proxy=Depends(get_proxy)):
    if request.method == "POST":
        return await _metered_forward(request, "/api/journal/entries", user, db, settings, proxy,
                                      cost=settings.journal_write_cost,
                                      forward_headers=attach_user_headers)
    return await _journal_forward(request, "/api/journal/entries", user, proxy)


@router.api_route("/api/journal/search", methods=["POST"])
async def journal_search(request: Request, user=Depends(get_current_user),
                         proxy=Depends(get_proxy)):
    return await _journal_forward(request, "/api/journal/search", user, proxy)


@router.api_route("/api/journal/entries/{entry_id}", methods=["GET", "DELETE"])
async def journal_entry(entry_id: str, request: Request, user=Depends(get_current_user),
                        proxy=Depends(get_proxy)):
    return await _journal_forward(request, f"/api/journal/entries/{entry_id}", user, proxy)


# Index-inventory: the client reports what it indexed (per-user), and can read back
# its own tree. Open to paying+; free (bookkeeping, not compute), identity-forwarded.
@router.api_route("/api/index/report", methods=["POST"])
async def index_report(request: Request, user=Depends(get_current_user),
                       proxy=Depends(get_proxy)):
    return await _journal_forward(request, "/api/index/report", user, proxy)


@router.api_route("/api/index/inventory", methods=["GET"])
async def index_inventory(request: Request, user=Depends(get_current_user),
                          proxy=Depends(get_proxy)):
    return await _journal_forward(request, "/api/index/inventory", user, proxy)


# ACE playbook (per-user curated lessons). Internal-gated + identity-forwarded, not
# metered (same policy as journal): the caller manages only their own bullets.
@router.api_route("/api/playbook/bullets", methods=["GET", "POST"])
async def playbook_bullets(request: Request, user=Depends(require_internal),
                           proxy=Depends(get_proxy)):
    return await _journal_forward(request, "/api/playbook/bullets", user, proxy)


@router.api_route("/api/playbook/search", methods=["POST"])
async def playbook_search(request: Request, user=Depends(require_internal),
                          proxy=Depends(get_proxy)):
    return await _journal_forward(request, "/api/playbook/search", user, proxy)


@router.api_route("/api/playbook/bullets/{rest:path}", methods=["POST", "DELETE"])
async def playbook_bullet_ops(rest: str, request: Request, user=Depends(require_internal),
                              proxy=Depends(get_proxy)):
    return await _journal_forward(request, f"/api/playbook/bullets/{rest}", user, proxy)


# Code KB (per-user semantic code search). Open to paying+; the client AST-chunks
# locally so only embeddings + search hit compute. Paying users pay code_index_cost (1)
# to index a repo and code_search_cost (0.1) to search; job polling + delete are free.
# Identity-forwarded so each user's code is private to them.
@router.api_route("/api/code/index", methods=["POST"])
async def code_index(request: Request, user=Depends(get_current_user),
                     db=Depends(get_db), settings=Depends(get_settings),
                     proxy=Depends(get_proxy)):
    return await _metered_forward(request, "/api/code/index", user, db, settings, proxy,
                                  cost=settings.code_index_cost,
                                  forward_headers=attach_user_headers)


@router.api_route("/api/code/search", methods=["POST"])
async def code_search(request: Request, user=Depends(get_current_user),
                      db=Depends(get_db), settings=Depends(get_settings),
                      proxy=Depends(get_proxy)):
    return await _metered_forward(request, "/api/code/search", user, db, settings, proxy,
                                  cost=settings.code_search_cost,
                                  forward_headers=attach_user_headers)


# Federated private+public code search (docs/federated_code_search_plan.md). Same
# billing/identity policy as /api/code/search: paying+ users pay code_search_cost,
# internal/admin are exempt; identity+tier are forwarded so compute can fail-closed
# 403 a non-admin caller's include_public=true instead of silently downgrading it.
@router.api_route("/api/code/federated-search", methods=["POST"])
async def code_federated_search(request: Request, user=Depends(get_current_user),
                                db=Depends(get_db), settings=Depends(get_settings),
                                proxy=Depends(get_proxy)):
    return await _metered_forward(request, "/api/code/federated-search", user, db, settings, proxy,
                                  cost=settings.code_search_cost,
                                  forward_headers=attach_user_headers)


# Async index jobs (W156): /index returns 202 {job_id}; clients poll these (free).
@router.api_route("/api/code/jobs", methods=["GET"])
async def code_jobs(request: Request, user=Depends(get_current_user),
                    proxy=Depends(get_proxy)):
    return await _journal_forward(request, "/api/code/jobs", user, proxy)


@router.api_route("/api/code/jobs/{job_id}", methods=["GET"])
async def code_job(job_id: str, request: Request, user=Depends(get_current_user),
                   proxy=Depends(get_proxy)):
    return await _journal_forward(request, f"/api/code/jobs/{job_id}", user, proxy)


@router.api_route("/api/code/{rest:path}", methods=["DELETE"])
async def code_delete(rest: str, request: Request, user=Depends(get_current_user),
                      proxy=Depends(get_proxy)):
    return await _journal_forward(request, f"/api/code/{rest}", user, proxy)


# Admin management console (cross-user journal + index inventory). ADMIN-ONLY:
# require_admin → 403 for non-admins; forwarded with identity headers (tier=admin)
# so the compute side's resolve_admin accepts it.
@router.api_route("/api/admin/console/{path:path}", methods=["GET", "POST", "DELETE"])
async def admin_console(path: str, request: Request, user=Depends(require_admin),
                        proxy=Depends(get_proxy)):
    return await _journal_forward(request, f"/api/admin/console/{path}", user, proxy)


# Shared public-code corpus. It is operator-managed rather than user-owned, so the
# public gateway exposes it only to admins and forwards the authenticated admin tier.
@router.api_route("/api/public-code/{path:path}", methods=["GET", "POST", "PUT"])
async def public_code_admin(path: str, request: Request, user=Depends(require_admin),
                            proxy=Depends(get_proxy)):
    return await _journal_forward(request, f"/api/public-code/{path}", user, proxy)


# ============================ account / sk-keys ==============================

class RegisterReq(BaseModel):
    username: str


@router.get("/auth/providers")
def auth_providers(request: Request):
    """Which login methods are enabled (so the UI shows only those). Public."""
    social = getattr(request.app.state, "social_providers", {})
    return {"cdk": True, "linuxdo": "linuxdo" in social, "google": "google" in social}


@router.get("/auth/me")
def me(user: User = Depends(get_current_user), db=Depends(get_db)):
    idents = accounts.list_identities(db, user)
    return {"id": user.id, "username": user.username, "tier": user.tier.value,
            "balance": user.balance,
            "identities": [{"id": i.id, "provider": i.provider, "sub": i.sub}
                           for i in idents]}


@router.get("/auth/identities")
def auth_list_identities(user: User = Depends(get_current_user), db=Depends(get_db)):
    return [{"id": i.id, "provider": i.provider, "sub": i.sub,
             "created_at": i.created_at.isoformat() if i.created_at else None}
            for i in accounts.list_identities(db, user)]


@router.delete("/auth/identities/{ident_id}")
def auth_unlink_identity(ident_id: int, user: User = Depends(get_current_user),
                         db=Depends(get_db)):
    # Refuses to strip the user's last login method (would orphan the account).
    if not accounts.remove_identity(db, user, ident_id):
        raise HTTPException(status_code=400, detail="cannot unlink last identity")
    db.commit()
    return {"ok": True}


@router.post("/auth/keys")
def create_key(user: User = Depends(get_current_user), db=Depends(get_db),
               settings=Depends(get_settings)):
    key, ak = accounts.issue_key(db, user, settings.key_pepper)
    db.commit()
    return {"id": ak.id, "key": key, "prefix": ak.prefix, "last4": ak.last4}  # plaintext ONCE


@router.get("/auth/keys")
def list_keys(user: User = Depends(get_current_user), db=Depends(get_db)):
    rows = db.query(ApiKey).filter(ApiKey.user_id == user.id).all()
    return [{"id": k.id, "prefix": k.prefix, "last4": k.last4,
             "revoked": k.revoked_at is not None,
             "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None}
            for k in rows]


@router.delete("/auth/keys/{key_id}")
def revoke_key(key_id: int, user: User = Depends(get_current_user), db=Depends(get_db)):
    ak = db.get(ApiKey, key_id)
    if ak is None or ak.user_id != user.id:
        raise HTTPException(status_code=404, detail="key not found")
    accounts.revoke_key(db, key_id)
    db.commit()
    return {"ok": True}


@router.post("/auth/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}


# ============================ OIDC login flow ================================

def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


async def default_oidc_exchange(app, settings, code: str, verifier: str, nonce: str):
    """Exchange the auth code at the Worker /token, then verify the id_token.
    Overridable via `app.state.oidc_exchange` (tests inject a fake)."""
    data = {"grant_type": "authorization_code", "code": code,
            "client_id": settings.oidc_client_id, "redirect_uri": settings.oidc_redirect_uri,
            "code_verifier": verifier}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(f"{settings.oidc_issuer.rstrip('/')}/token", data=data)
    if r.status_code != 200:
        raise HTTPException(status_code=401, detail="token exchange failed")
    id_token = r.json().get("id_token")
    cfg = oidc.OIDCConfig(issuer=settings.oidc_issuer, client_id=settings.oidc_client_id,
                          jwks_uri=settings.jwks_uri)
    try:
        return oidc.verify_id_token(id_token, cfg, jwks_client=app.state.jwks_client, nonce=nonce)
    except oidc.OIDCError:
        raise HTTPException(status_code=401, detail="invalid id_token")


@router.get("/auth/login")
def login(request: Request, settings=Depends(get_settings)):
    state, nonce = secrets.token_urlsafe(16), secrets.token_urlsafe(16)
    verifier, challenge = _pkce_pair()
    request.session["oidc"] = {"state": state, "nonce": nonce, "verifier": verifier}
    q = urlencode({"client_id": settings.oidc_client_id, "redirect_uri": settings.oidc_redirect_uri,
                   "response_type": "code", "scope": "openid", "state": state, "nonce": nonce,
                   "code_challenge": challenge, "code_challenge_method": "S256"})
    return RedirectResponse(f"{settings.oidc_issuer.rstrip('/')}/authorize?{q}")


@router.get("/auth/callback")
async def callback(request: Request, code: str, state: str, db=Depends(get_db),
                   settings=Depends(get_settings)):
    sess = request.session.get("oidc") or {}
    if not sess or not secrets.compare_digest(sess.get("state", ""), state):
        raise HTTPException(status_code=400, detail="bad state")
    exchange = getattr(request.app.state, "oidc_exchange", None) or default_oidc_exchange
    claims = await exchange(request.app, settings, code, sess["verifier"], sess["nonce"])
    request.session.pop("oidc", None)

    # CDK login is the "cdk" provider; claims.sub is "cdk:<key>".
    provider, sub = accounts.split_sub(claims.sub)
    user = accounts.resolve_identity(db, provider, sub)
    if user is not None:
        request.session["uid"] = user.id
        return RedirectResponse(settings.public_base_url + "/", status_code=303)
    # first login → choose a username
    request.session["pending"] = {"provider": provider, "sub": sub,
                                  "suggested": "", "is_admin": bool(claims.is_admin)}
    return RedirectResponse(settings.public_base_url + "/register", status_code=303)


@router.post("/auth/register")
def register(body: RegisterReq, request: Request, db=Depends(get_db)):
    pending = request.session.get("pending")
    if not pending:
        raise HTTPException(status_code=400, detail="no pending registration")
    try:
        user = accounts.register_with_identity(
            db, provider=pending.get("provider", "cdk"), sub=pending["sub"],
            username=body.username, is_admin=pending.get("is_admin", False))
    except accounts.UsernameTaken:
        raise HTTPException(status_code=409, detail="username taken")
    except accounts.AlreadyRegistered:
        raise HTTPException(status_code=409, detail="already registered")
    db.commit()
    request.session.pop("pending", None)
    request.session["uid"] = user.id
    return {"id": user.id, "username": user.username, "tier": user.tier.value}


# ====================== social login (linux.do / Google) =====================
# linux.do uses the URL segment "linuxdooid" (the app's configured callback path);
# its provider key is "linuxdo". Google uses "google" for both. Each provider is
# enabled iff its client_id is configured (app.state.social_providers).

def _social_provider(request: Request, name: str) -> oauth2.OAuth2Provider:
    p = getattr(request.app.state, "social_providers", {}).get(name)
    if p is None:
        raise HTTPException(status_code=404, detail="login provider not enabled")
    return p


async def _social_fetch(request: Request, provider, code: str):
    """Real fetch_identity, unless a test injected app.state.oauth_fetch."""
    fake = getattr(request.app.state, "oauth_fetch", None)
    if fake is not None:
        return await fake(provider, code, provider.redirect_uri)
    return await oauth2.fetch_identity(provider, code=code, redirect_uri=provider.redirect_uri)


def _begin_social(request: Request, name: str, *, mode: str) -> RedirectResponse:
    provider = _social_provider(request, name)
    state = secrets.token_urlsafe(16)
    # Stash state + mode (+ the current uid for link mode) keyed per provider.
    request.session[f"oauth_{name}"] = {
        "state": state, "mode": mode, "uid": request.session.get("uid")}
    url = oauth2.build_authorize_url(provider, redirect_uri=provider.redirect_uri, state=state)
    return RedirectResponse(url)


async def _finish_social(request: Request, name: str, code: str, state: str,
                         db, settings) -> RedirectResponse:
    provider = _social_provider(request, name)
    sess = request.session.get(f"oauth_{name}") or {}
    if not sess or not secrets.compare_digest(sess.get("state", ""), state):
        raise HTTPException(status_code=400, detail="bad state")
    mode = sess.get("mode", "login")
    link_uid = sess.get("uid")
    request.session.pop(f"oauth_{name}", None)

    try:
        ident = await _social_fetch(request, provider, code)
    except oauth2.OAuthError:
        raise HTTPException(status_code=401, detail="social login failed")

    base = settings.public_base_url.rstrip("/")
    # linux.do 最低等级 gate (Google has no trust_field → never gated)
    if provider.min_trust_level and (ident.trust_level is None
                                     or ident.trust_level < provider.min_trust_level):
        logger.info("social login blocked: low trust provider=%s sub=%s trust=%s",
                    name, ident.sub, ident.trust_level)
        return RedirectResponse(f"{base}/?login=trust_low", status_code=303)

    existing = accounts.resolve_identity(db, name, ident.sub)

    if mode == "link":
        cur = db.get(User, link_uid) if link_uid is not None else None
        if cur is None:
            raise HTTPException(status_code=401, detail="login required to bind")
        if existing is not None and existing.id != cur.id:
            return RedirectResponse(f"{base}/?bind=conflict", status_code=303)
        accounts.add_identity(db, cur, name, ident.sub)
        db.commit()
        request.session["uid"] = cur.id
        return RedirectResponse(f"{base}/?bind=ok", status_code=303)

    # login mode
    if existing is not None:
        request.session["uid"] = existing.id
        return RedirectResponse(f"{base}/", status_code=303)
    # first time → choose a username on /register
    request.session["pending"] = {"provider": name, "sub": ident.sub,
                                  "suggested": ident.username or "", "is_admin": False}
    return RedirectResponse(f"{base}/register", status_code=303)


@router.get("/api/linuxdooid/login")
def linuxdo_login(request: Request):
    return _begin_social(request, "linuxdo", mode="login")


@router.get("/api/linuxdooid/link")
def linuxdo_link(request: Request, user: User = Depends(get_current_user)):
    return _begin_social(request, "linuxdo", mode="link")


@router.get("/api/linuxdooid/callback")
async def linuxdo_callback(request: Request, code: str, state: str,
                           db=Depends(get_db), settings=Depends(get_settings)):
    return await _finish_social(request, "linuxdo", code, state, db, settings)


@router.get("/api/google/login")
def google_login(request: Request):
    return _begin_social(request, "google", mode="login")


@router.get("/api/google/link")
def google_link(request: Request, user: User = Depends(get_current_user)):
    return _begin_social(request, "google", mode="link")


@router.get("/api/google/callback")
async def google_callback(request: Request, code: str, state: str,
                          db=Depends(get_db), settings=Depends(get_settings)):
    return await _finish_social(request, "google", code, state, db, settings)


# ============================ payment (mock) =================================

class TopupReq(BaseModel):
    credits: int
    price: str  # Decimal as string


@router.post("/api/topup")
def topup(body: TopupReq, user: User = Depends(get_current_user), db=Depends(get_db),
          settings=Depends(get_settings), provider=Depends(get_provider)):
    if body.credits <= 0:
        raise HTTPException(status_code=400, detail="credits must be positive")
    try:
        price = Decimal(body.price)
    except (InvalidOperation, ValueError):
        raise HTTPException(status_code=400, detail="bad price")
    created = start_topup(db, user, credits=body.credits, price=price,
                          provider=provider, notify_url=settings.payment_notify_url,
                          return_url=settings.payment_return_url or None)
    db.commit()
    return {"out_trade_no": created.out_trade_no, "pay_url": created.pay_url}


async def _process_notify(request: Request, db, provider) -> PlainTextResponse:
    # epay sends the async notify as GET (query) and the return as GET; some
    # gateways POST a form. Accept both, merging query + form into one param map.
    params = dict(request.query_params)
    if request.method == "POST":
        form = await request.form()
        params.update({k: str(v) for k, v in form.items()})
    result = handle_notify(db, provider, params)
    db.commit()
    return PlainTextResponse(result)


@router.get("/api/payment/notify")
async def payment_notify_get(request: Request, db=Depends(get_db), provider=Depends(get_provider)):
    return await _process_notify(request, db, provider)


@router.post("/api/payment/notify")
async def payment_notify_post(request: Request, db=Depends(get_db), provider=Depends(get_provider)):
    return await _process_notify(request, db, provider)


# ============================ admin ==========================================

class AdjustReq(BaseModel):
    amount: int            # signed: positive grants, negative deducts
    reason: str | None = None


class TierReq(BaseModel):
    tier: str


@router.get("/admin/users")
def admin_list_users(admin: User = Depends(require_admin), db=Depends(get_db)):
    return [{"id": u.id, "username": u.username, "tier": u.tier.value, "balance": u.balance}
            for u in db.query(User).all()]


@router.post("/admin/users/{user_id}/adjust")
def admin_adjust(user_id: int, body: AdjustReq, admin: User = Depends(require_admin),
                 db=Depends(get_db)):
    if db.get(User, user_id) is None:
        raise HTTPException(status_code=404, detail="user not found")
    balance = ledger.credit(db, user_id, body.amount, kind="adjust", ref=body.reason)
    db.commit()
    return {"user_id": user_id, "balance": balance}


@router.post("/admin/users/{user_id}/tier")
def admin_set_tier(user_id: int, body: TierReq, admin: User = Depends(require_admin),
                   db=Depends(get_db)):
    try:
        tier = Tier(body.tier)
    except ValueError:
        raise HTTPException(status_code=400, detail="bad tier")
    target = db.get(User, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="user not found")
    target.tier = tier
    db.commit()
    return {"user_id": user_id, "tier": tier.value}
