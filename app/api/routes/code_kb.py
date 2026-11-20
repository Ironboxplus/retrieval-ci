"""Code KB routes (owner from gateway identity or internal-admin on LAN).

The client AST-chunks its repo locally and POSTs the chunks; the backend embeds +
stores them in the 4090 `code_chunks` collection. Owner is resolved server-side —
never from the request body — so users only ever touch their own code.

Indexing is ASYNC (W156): POST /index returns 202 {job_id} immediately — a large
repo's multi-minute embed no longer holds the HTTP request open (which read as a
hang to MCP clients). Poll GET /jobs/{job_id} or list GET /jobs for progress.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import time

from fastapi import APIRouter, Header, HTTPException, Query, Request

from app.api.identity import resolve_admin, resolve_owner
from app.api.routes.public_code import _run_service as _run_public_code_service
from app.services.code_federation import CodeFederationCandidate, fuse_code_hits
from app.services.code_kb import get_code_kb_service
from app.services.code_kb.analysis_store import claim_scope_notice
from app.services.code_kb.jobs import (
    CodeIndexJob,
    CodeIndexJobAlreadyEnded,
    CodeIndexJobCommitting,
    get_code_index_queue,
    get_existing_code_index_queue,
)
from app.services.index_audit import (
    build_index_audit_record,
    emit_index_audit,
    push_index_audit_bg,
)
from app.services.index_catalog.service import ResolvedScope, get_index_catalog_service
from app.services.public_code import get_public_code_search_service
from app.services.public_code.search_service import PublicCodeSearchRequest
from app.services.scope_paths import canonicalize_scope_path, is_windows_scope
from app.services.code_kb.schemas import (
    CodeDeleteResponse,
    CodeDuplicateScanRequest,
    CodeDuplicateScanResponse,
    CodeIndexJobList,
    CodeIndexJobOut,
    CodeIndexRequest,
    CodeSearchRequest,
    CodeSearchResponse,
    CodeSearchTimingsMs,
    FederatedCodeHit,
    FederatedCodeSearchRequest,
    FederatedCodeSearchResponse,
    FederatedLegDiagnostics,
)
from core.retrieval.reranker import get_shared_reranker

router = APIRouter()

_UserId = Header(default=None, alias="X-Retrival-User-Id")
_Tier = Header(default=None, alias="X-Retrival-User-Tier")
logger = logging.getLogger("code_kb.api")
# Retrieval pool per leg for federated-search: wider than top_k so RRF/rerank fusion
# has real candidates to work with, capped like PublicCodeSearchService's own rerank
# pool so a large top_k can't force an unbounded per-leg Qdrant query.
_FEDERATED_POOL_MIN = 20
_FEDERATED_POOL_MAX = 50
# CodeScopeVisibility is keyed by owner/project; host is only inventory display metadata.
# Deletes are not index jobs and have no pollable receipt, hence a named operation host and
# a null job_id rather than a synthetic job id clients could mistakenly poll.
_DELETE_VISIBILITY_HOST = "delete-operation"


def _job_out(job: CodeIndexJob) -> CodeIndexJobOut:
    return CodeIndexJobOut(
        job_id=job.job_id, project=job.project, host=job.host,
        status=job.status.value, progress=job.progress,
        files_total=job.files_total, chunks_total=job.chunks_total,
        removed_total=job.removed_total,
        chunks_done=job.chunks_done, indexed=job.indexed,
        duplicates_updated=job.duplicates_updated,
        queue_position=job.queue_position, error=job.error,
        created_at=job.created_at, started_at=job.started_at, finished_at=job.finished_at,
    )


def _join_search_notices(existing: str | None, addition: str) -> str:
    return f"{existing}\n{addition}" if existing else addition


def _withheld_scope_response(
    body: CodeSearchRequest,
    *,
    job_id: str | None,
    previous_completed_snapshot_available: bool,
    mutation_started: bool,
    recovery_required: bool = False,
    catalog_unavailable: bool = False,
) -> CodeSearchResponse:
    if catalog_unavailable:
        return CodeSearchResponse(
            hits=[], total=0,
            requested_mode=body.mode,
            effective_mode=body.mode,
            scope_snapshot_state="unavailable",
            recovery_required=False,
            active_index_job_id=None,
            previous_completed_snapshot_available=False,
            notice=(
                "Code KB search is temporarily withheld because Index Catalog visibility "
                "cannot be verified. Check Index Catalog health and retry shortly; do not "
                "treat this as a no-match result."
            ),
        )

    prior = (
        " A previous completed snapshot is recorded, but Code KB has no atomic revision "
        "promotion yet, so this request is withheld rather than reading mixed old/new batches."
        if previous_completed_snapshot_available else
        " No completed snapshot is available yet."
    )
    if recovery_required:
        return CodeSearchResponse(
            hits=[], total=0,
            requested_mode=body.mode,
            effective_mode=body.mode,
            scope_snapshot_state="recovery_required",
            recovery_required=True,
            active_index_job_id=None,
            previous_completed_snapshot_available=previous_completed_snapshot_available,
            notice=(
                "Code KB search is temporarily withheld because an interrupted in-place "
                "write requires a full recovery index for this scope. Submit a full recovery "
                "index, then retry; this recovery action has no live job receipt."
                + prior
            ),
        )

    phase = (
        "is writing" if mutation_started else "is preparing"
    )
    job_phrase = f"Code index job {job_id} {phase}" if job_id else f"Code index mutation {phase}"
    next_step = (
        f"Poll /api/code/jobs/{job_id} and retry after it completes."
        if job_id else
        "Check the active code-index status and retry after it completes."
    )
    return CodeSearchResponse(
        hits=[], total=0,
        requested_mode=body.mode,
        effective_mode=body.mode,
        scope_snapshot_state="preparing",
        active_index_job_id=job_id,
        previous_completed_snapshot_available=previous_completed_snapshot_available,
        notice=(
            f"{job_phrase}; search is temporarily withheld so a "
            "partial scope is not presented as a completed result. "
            + next_step
            + prior
        ),
    )


def _durable_completed_snapshot_available(
    *,
    owner_user_id: str,
    projects: list[str],
) -> bool:
    """Use catalog evidence after a queue/history restart, failing conservatively."""
    try:
        catalog = get_index_catalog_service()
        checker = getattr(catalog, "has_completed_code_scope", None)
        return bool(checker and any(
            checker(owner_user_id=owner_user_id, project=project)
            for project in projects
        ))
    except Exception:  # noqa: BLE001 - a catalog outage must not expose partial vectors
        logger.warning("code search could not verify previous completed scope snapshot")
        return False


def _durable_scope_visibility(
    *,
    owner_user_id: str,
    projects: list[str],
) -> dict | None:
    """Read the SQLite visibility marker without exposing a vector partial on failure."""
    try:
        catalog = get_index_catalog_service()
        getter = getattr(catalog, "code_scope_visibility", None)
        return getter(owner_user_id=owner_user_id, projects=projects) if getter else None
    except Exception:  # noqa: BLE001 - a marker read failure is handled fail-closed by caller
        logger.warning("code search could not read durable scope visibility")
        return {"state": "unavailable",
                "previous_completed_snapshot_available": False}


def _active_scopes_have_completed_snapshot(*, owner: str, queue) -> bool:
    """True only if EVERY scope an active job of this owner touches already has a
    completed snapshot, per the queue's own history or the durable catalog. Any doubt
    (no scope list, a read failure, one scope never completed) fails closed."""
    try:
        states = queue.inventory_scope_states(owner_user_id=owner)
    except Exception:  # noqa: BLE001 - cannot enumerate the active scopes: withhold
        logger.exception("code search could not read in-flight scope states")
        return False
    if not states:
        return False
    for (_owner, _host, project), scope_state in states.items():
        if scope_state.get("recovery_required"):
            return False
        if scope_state.get("previous_completed_snapshot_available"):
            continue
        if not _durable_completed_snapshot_available(
                owner_user_id=owner, projects=[project]):
            return False
    return True


def _durable_owner_visibility(*, owner_user_id: str) -> dict | None:
    try:
        catalog = get_index_catalog_service()
        getter = getattr(catalog, "owner_code_scope_visibility", None)
        return getter(owner_user_id=owner_user_id) if getter else None
    except Exception:  # noqa: BLE001 - do not mix a partial scope into an owner-wide search
        logger.warning("code search could not read owner visibility")
        return {"state": "unavailable",
                "previous_completed_snapshot_available": False}


@router.post("/index", response_model=CodeIndexJobOut, status_code=202)
async def index_code(body: CodeIndexRequest, request: Request, user_id: str | None = _UserId):
    """Queue a repo index job; returns 202 + job_id immediately (poll /jobs/{job_id})."""
    owner = resolve_owner(request, user_id)

    def submit_and_snapshot() -> tuple[CodeIndexJob, CodeIndexJobOut]:
        job = get_code_index_queue().submit(body, owner_user_id=owner)
        # Snapshot before audit side-effects. A tiny worker can finish within the few ms
        # audit serialization adds, which would otherwise change the receipt response.
        return job, _job_out(job)

    # Queue construction and request fingerprinting both scale with repository size and
    # must not stall unrelated async endpoints on FastAPI's event loop.
    try:
        job, out = await asyncio.to_thread(submit_and_snapshot)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    # Faithful audit at RECEIPT (W172): record the request the moment it is accepted,
    # before the async embed runs, so a job that later fails/crashes still left a trace.
    rec = emit_index_audit(build_index_audit_record(
        event="code_index_submitted", kind="code", owner_user_id=owner,
        project=body.project, host=body.host, files=len(body.files),
        chunks=job.chunks_total, job_id=job.job_id))
    push_index_audit_bg(rec)
    return out


@router.get("/jobs", response_model=CodeIndexJobList)
async def list_index_jobs(
    request: Request, user_id: str | None = _UserId, limit: int = Query(20, ge=1, le=100),
    project: str | None = Query(default=None, max_length=2000),
    active_only: bool = False,
):
    """The caller's recent index jobs, newest first."""
    owner = resolve_owner(request, user_id)
    project = canonicalize_scope_path(project) if project else None

    def load_jobs() -> list[CodeIndexJob]:
        queue = get_code_index_queue()
        scopes = [project] if project else []
        if project and is_windows_scope(project):
            try:
                scopes = get_index_catalog_service().equivalent_project_scopes(
                    owner_user_id=owner, project=project, kind="code")
            except Exception:  # noqa: BLE001 - job listing remains available on catalog outage
                logger.exception("code job scope-alias lookup failed for project=%s", project)
        if not scopes:
            return queue.list_jobs(
                owner_user_id=owner, limit=limit, project=None, active_only=active_only)
        merged: dict[str, CodeIndexJob] = {}
        for scope in dict.fromkeys(scopes):
            for job in queue.list_jobs(
                owner_user_id=owner, limit=limit, project=scope,
                active_only=active_only,
            ):
                merged[job.job_id] = job
        return sorted(
            merged.values(), key=lambda job: job.created_at, reverse=True)[:limit]

    jobs = await asyncio.to_thread(load_jobs)
    return CodeIndexJobList(jobs=[_job_out(j) for j in jobs])


@router.get("/jobs/{job_id}", response_model=CodeIndexJobOut)
async def get_index_job(
    job_id: str, request: Request, user_id: str | None = _UserId,
    scope_notice: bool = Query(
        default=False,
        description="Also claim and return the scope's undelivered analysis summary. "
                    "Off by default: a client's own progress polling must not consume a "
                    "message meant for a person."),
):
    """Progress of one index job (404 for unknown or another owner's job)."""
    owner = resolve_owner(request, user_id)
    # A poll is read-only.  Do not construct a Qdrant-backed queue/worker merely to
    # return an unknown receipt after a backend restart or before any index request.
    queue = get_existing_code_index_queue()
    job = queue.get(job_id, owner_user_id=owner) if queue is not None else None
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    out = _job_out(job)
    if scope_notice:
        notice = await asyncio.to_thread(claim_scope_notice, owner, job.project)
        if notice:
            out = out.model_copy(update={"scope_notice": notice})
    return out


@router.post("/jobs/{job_id}/cancel", response_model=CodeIndexJobOut)
async def cancel_index_job(job_id: str, request: Request, user_id: str | None = _UserId):
    """Cancel one of the caller's index jobs (F2-a), owner-isolated like every other
    job route: another owner's job_id looks like unknown (404), never a 403 oracle.

    queued -> cancelled immediately (200).
    running, no Qdrant mutation started yet -> a flag is set; the worker's mutation
    fence observes it before its first write and settles the job as cancelled with
    nothing committed (200 here — the transition itself happens asynchronously, so the
    returned job may still show status=running for an instant).
    running, mutation already started -> 409 {"code": "code_index_committing"}: a
    partial write cannot be safely interrupted without leaving a half-committed scope.
    already COMPLETED/FAILED/CANCELLED -> 409 {"code": "code_index_already_ended"}.
    """
    owner = resolve_owner(request, user_id)

    def do_cancel() -> CodeIndexJob:
        queue = get_existing_code_index_queue()
        if queue is None:
            raise KeyError(job_id)
        return queue.cancel(job_id, owner_user_id=owner)

    try:
        job = await asyncio.to_thread(do_cancel)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    except CodeIndexJobCommitting as exc:
        raise HTTPException(
            status_code=409, detail={"code": "code_index_committing"}) from exc
    except CodeIndexJobAlreadyEnded as exc:
        raise HTTPException(
            status_code=409, detail={"code": "code_index_already_ended"}) from exc
    return _job_out(job)


@dataclasses.dataclass(frozen=True)
class PrivateScopeDecision:
    """Can ONE scope (its canonical spelling plus Windows-case aliases) be searched
    right now? Extracted (W-federated-search A1, revision R1) from the single-project
    branch of ``_run_private_code_search`` so federated-search's per-scope exclusion
    (decision 10) applies the IDENTICAL rule instead of a parallel re-implementation
    that could drift from it."""
    allowed: bool
    active_job_id: str | None = None
    previous_completed_snapshot_available: bool | None = None
    mutation_started: bool = False
    recovery_required: bool = False
    catalog_unavailable: bool = False
    # Only meaningful when allowed=True: a newer job is preparing but the previous
    # completed snapshot is still safe to serve (mirrors the single-scope route's
    # existing "serve stale, note it" behaviour).
    allowed_with_newer_job: bool = False
    newer_job_active_id: str | None = None

    @property
    def scope_snapshot_state(self) -> str:
        """The state label a withheld response for this scope would carry."""
        if self.catalog_unavailable:
            return "unavailable"
        if self.recovery_required:
            return "recovery_required"
        return "preparing"


def _private_scope_decision(
    *, owner: str, projects: list[str], queue,
) -> PrivateScopeDecision:
    """Judge one scope. Must be called inside the caller's ``service.scope_snapshot``
    fence, exactly like the branch this was extracted from."""
    durable = _durable_scope_visibility(owner_user_id=owner, projects=projects)
    state = queue.scope_search_state(
        owner_user_id=owner, projects=projects) if queue is not None else None

    # A live process can retain an unsafe scope even if SQLite accepted the original
    # preparing marker but rejected its recovery upgrade. Prefer that stronger local
    # state so clients never poll the failed receipt.
    if state is not None and getattr(state, "recovery_required", False):
        return PrivateScopeDecision(
            allowed=False,
            previous_completed_snapshot_available=bool(
                state.previous_completed_snapshot_available),
            mutation_started=True,
            recovery_required=True,
        )
    if durable is not None and durable.get("state") != "completed":
        durable_state = str(durable.get("state") or "preparing")
        return PrivateScopeDecision(
            allowed=False,
            active_job_id=(str(durable["job_id"])
                            if durable_state == "preparing" and durable.get("job_id") else None),
            previous_completed_snapshot_available=bool(
                durable.get("previous_completed_snapshot_available")),
            mutation_started=True,
            recovery_required=(durable_state == "recovery_required"),
            catalog_unavailable=(durable_state == "unavailable"),
        )
    if state is None:
        return PrivateScopeDecision(allowed=True)

    prior_completed = bool(state.previous_completed_snapshot_available)
    if not prior_completed:
        prior_completed = _durable_completed_snapshot_available(
            owner_user_id=owner, projects=projects)
    if (state.mutation_started or getattr(state, "recovery_required", False)
            or not prior_completed):
        return PrivateScopeDecision(
            allowed=False,
            active_job_id=state.active_job_id,
            previous_completed_snapshot_available=prior_completed,
            mutation_started=state.mutation_started,
            recovery_required=bool(getattr(state, "recovery_required", False)),
        )
    return PrivateScopeDecision(
        allowed=True, allowed_with_newer_job=True, newer_job_active_id=state.active_job_id,
    )


def _resolve_indexed_ancestor(
    *, owner: str, project: str | None, aliases: list[str] | None,
) -> ResolvedScope | None:
    """The indexed ancestor that covers a requested directory nobody indexed, if any.

    Scopes are matched exactly, but the nested-scope guard refuses to index a directory
    inside an indexed one, so such a directory could never be searched. ``None`` means
    "search ``project`` as given" - including when the catalog cannot be read."""
    if not project:
        return None
    try:
        return get_index_catalog_service().resolve_indexed_scope(
            owner_user_id=owner, kind="code", scope=project, known_scopes=aliases)
    except Exception:  # noqa: BLE001 - preserve search availability on catalog outage
        logger.exception("code search ancestor-scope lookup failed for project=%s", project)
        return None


def _with_resolved_scope(
    response: CodeSearchResponse, resolved: ResolvedScope | None,
) -> CodeSearchResponse:
    if resolved is None:
        return response
    return response.model_copy(update={
        "resolved_scope": resolved.scope, "resolved_path_prefix": resolved.path_prefix})


def _run_private_code_search(body: CodeSearchRequest, owner: str) -> CodeSearchResponse:
    """The private-plane search body, off the event loop. Extracted (W-federated-search
    A1) so the federated-search route can reuse the scope-snapshot fencing/withheld-scope
    logic verbatim instead of re-implementing it or calling ``service.search`` directly,
    which would silently bypass the "index writing => scope withheld" invariant."""
    aliases: list[str] | None = None
    if body.project and is_windows_scope(body.project):
        # Catalog I/O belongs in this worker with embedding/vector search. Legacy
        # E:/JETB/... vectors remain discoverable after canonical writes begin.
        try:
            aliases = get_index_catalog_service().equivalent_project_scopes(
                owner_user_id=owner, project=body.project, kind="code")
        except Exception:  # noqa: BLE001 - preserve search availability on catalog outage
            logger.exception(
                "code search scope-alias lookup failed for project=%s", body.project)
            aliases = [body.project]
    resolved = _resolve_indexed_ancestor(owner=owner, project=body.project, aliases=aliases)
    if resolved is not None:
        # Everything below (visibility fence, withheld state, vector filter) must judge the
        # scope that actually holds the vectors, not the directory the caller named.
        body = body.model_copy(update={"project": resolved.scope})
        aliases = resolved.aliases if is_windows_scope(resolved.scope) else None
    response = _with_resolved_scope(
        _search_private_scope(
            body, owner, aliases=aliases,
            path_prefix=resolved.path_prefix if resolved is not None else None),
        resolved)
    # body.project is the scope that holds the vectors (the resolved ancestor when a
    # subdirectory was redirected). A withheld scope shows no results, so it does not
    # spend the one delivery either.
    if response.scope_snapshot_state == "completed":
        notice = claim_scope_notice(owner, body.project)
        if notice:
            return response.model_copy(update={"scope_notice": notice})
    return response


def _search_private_scope(
    body: CodeSearchRequest, owner: str, *, aliases: list[str] | None,
    path_prefix: str | None,
) -> CodeSearchResponse:
    service = get_code_kb_service()
    # Keep the base service contract unchanged for ordinary POSIX / unscoped
    # searches. Only Windows scopes need the legacy-case compatibility extension.
    def invoke_search() -> CodeSearchResponse:
        extra = {"path_prefix": path_prefix} if path_prefix else {}
        if aliases is None:
            return service.search(body, owner_user_id=owner, **extra)
        return service.search(body, owner_user_id=owner, project_aliases=aliases, **extra)

    # Code KB mutates Qdrant scopes in place.  Take the owner fence BEFORE looking at
    # queue state: a pre-fence ``None`` can race a writer that immediately starts a
    # batch. Owner-wide searches use the same fence because they can include a
    # fan-out target even without an explicit project filter.
    if not hasattr(service, "scope_snapshot"):
        return invoke_search()
    with service.scope_snapshot(owner_user_id=owner):
        queue = get_existing_code_index_queue()
        if body.project:
            projects = list(dict.fromkeys(aliases or [body.project]))
            decision = _private_scope_decision(owner=owner, projects=projects, queue=queue)
            if not decision.allowed:
                return _withheld_scope_response(
                    body,
                    job_id=decision.active_job_id,
                    previous_completed_snapshot_available=bool(
                        decision.previous_completed_snapshot_available),
                    mutation_started=decision.mutation_started,
                    recovery_required=decision.recovery_required,
                    catalog_unavailable=decision.catalog_unavailable,
                )
            if not decision.allowed_with_newer_job:
                return invoke_search()
            response = invoke_search()
            return response.model_copy(update={
                "scope_snapshot_state": "completed",
                "active_index_job_id": decision.newer_job_active_id,
                "previous_completed_snapshot_available": True,
                "notice": _join_search_notices(
                    response.notice,
                    "A newer code index job is preparing; results are from the previous "
                    "completed snapshot until its write phase begins.",
                ),
            })

        durable_owner = _durable_owner_visibility(owner_user_id=owner)
        state = None
        if queue is not None:
            getter = getattr(queue, "owner_search_state", None)
            state = getter(owner_user_id=owner) if getter else None
        if state is not None and getattr(state, "recovery_required", False):
            return _withheld_scope_response(
                body,
                job_id=None,
                previous_completed_snapshot_available=bool(
                    state.previous_completed_snapshot_available),
                mutation_started=True,
                recovery_required=True,
            )
        if durable_owner is not None:
            durable_state = str(durable_owner.get("state") or "preparing")
            return _withheld_scope_response(
                body,
                job_id=(str(durable_owner["job_id"])
                        if durable_state == "preparing" and durable_owner.get("job_id") else None),
                previous_completed_snapshot_available=bool(
                    durable_owner.get("previous_completed_snapshot_available")),
                mutation_started=True,
                recovery_required=(durable_state == "recovery_required"),
                catalog_unavailable=(durable_state == "unavailable"),
            )
        if state is None:
            return invoke_search()
        prior_completed = bool(state.previous_completed_snapshot_available)
        if (not prior_completed and not state.mutation_started
                and not getattr(state, "recovery_required", False)):
            # The queue only remembers completions from this process; after a restart
            # it cannot vouch for a scope that is merely refreshing. Ask the durable
            # catalog, exactly as the single-scope branch does.
            prior_completed = _active_scopes_have_completed_snapshot(
                owner=owner, queue=queue)
        if (state.mutation_started or getattr(state, "recovery_required", False)
                or not prior_completed):
            return _withheld_scope_response(
                body,
                job_id=state.active_job_id,
                previous_completed_snapshot_available=prior_completed,
                mutation_started=state.mutation_started,
                recovery_required=bool(getattr(state, "recovery_required", False)),
            )
        response = invoke_search()
        return response.model_copy(update={
            "scope_snapshot_state": "completed",
            "active_index_job_id": state.active_job_id,
            "previous_completed_snapshot_available": True,
            "notice": _join_search_notices(
                response.notice,
                "A newer code index job is preparing; results are from the previous "
                "completed snapshot until its write phase begins.",
            ),
        })


@router.post("/search", response_model=CodeSearchResponse)
async def search_code(body: CodeSearchRequest, request: Request, user_id: str | None = _UserId):
    """Semantic code search over the caller's indexed chunks (path:line citations)."""
    owner = resolve_owner(request, user_id)
    return await asyncio.to_thread(_run_private_code_search, body, owner)


def _federated_pool(top_k: int) -> int:
    return min(max(top_k * 2, _FEDERATED_POOL_MIN), _FEDERATED_POOL_MAX)


def _scope_aliases(*, owner: str, project: str) -> list[str]:
    """The canonical spelling plus any Windows-case aliases the catalog knows about."""
    canonical = canonicalize_scope_path(project)
    if not is_windows_scope(canonical):
        return [canonical]
    try:
        return get_index_catalog_service().equivalent_project_scopes(
            owner_user_id=owner, project=canonical, kind="code")
    except Exception:  # noqa: BLE001 - preserve search availability on catalog outage
        logger.exception(
            "federated code search scope-alias lookup failed for project=%s", canonical)
        return [canonical]


def _incomplete_scope_projects(*, owner_user_id: str, queue) -> list[str] | None:
    """Candidate incomplete scopes for owner-wide decision-10 discovery.

    ``None`` means the durable catalog read itself failed: the caller must fail the
    WHOLE private leg closed then (an unreadable catalog means we cannot know which
    scopes would need excluding). A successful read returns a (possibly empty) list,
    which is NOT the same thing.
    """
    try:
        catalog = get_index_catalog_service()
        lister = getattr(catalog, "list_incomplete_code_scopes", None)
        rows = lister(owner_user_id=owner_user_id) if lister else []
    except Exception:  # noqa: BLE001 - fail the whole leg closed; see docstring
        logger.warning("federated code search could not list incomplete code scopes")
        return None
    candidates = {row["project"] for row in rows if row.get("project")}
    if queue is not None:
        try:
            inventory = queue.inventory_scope_states(owner_user_id=owner_user_id)
        except Exception:  # noqa: BLE001 - the in-memory queue is best-effort evidence
            logger.exception("federated code search could not read in-flight scope states")
        else:
            candidates.update(key[2] for key in inventory)
    return sorted(candidates)


@dataclasses.dataclass(frozen=True)
class FederatedPrivatePlan:
    withheld: CodeSearchResponse | None
    project: str | None
    project_aliases: list[str] | None
    excluded_projects: list[str] | None
    excluded_scopes: list[dict]
    searched_projects: list[str]
    # Requested directories that had no index of their own: (ancestor scope spellings,
    # path prefix below it). Each needs its own query because a prefix is per scope.
    prefixed_groups: tuple[tuple[tuple[str, ...], str], ...] = ()
    # True when every requested scope is a prefixed group (nothing for the plain query).
    only_prefixed: bool = False


def _plan_federated_private_scopes(
    *, owner: str, body: FederatedCodeSearchRequest, queue,
) -> FederatedPrivatePlan:
    """Decision 10: exclude individually-withheld scopes instead of failing the whole
    private leg closed. Must run inside the caller's ``service.scope_snapshot`` fence.
    """
    excluded_scopes: list[dict] = []

    if body.projects:
        included_groups: list[list[str]] = []
        prefixed_groups: list[tuple[tuple[str, ...], str]] = []
        for raw in body.projects:
            aliases = _scope_aliases(owner=owner, project=raw)
            resolved = _resolve_indexed_ancestor(owner=owner, project=raw, aliases=aliases)
            if resolved is not None:
                # Judge the scope that holds the vectors, exactly like a plain request.
                aliases = list(resolved.aliases)
            decision = _private_scope_decision(owner=owner, projects=aliases, queue=queue)
            if decision.allowed and resolved is not None:
                prefixed_groups.append((tuple(aliases), resolved.path_prefix))
            elif decision.allowed:
                included_groups.append(aliases)
            else:
                excluded_scopes.append({
                    "project": aliases[0],
                    "state": decision.scope_snapshot_state,
                    "active_index_job_id": decision.active_job_id,
                })
        if not included_groups and not prefixed_groups:
            # Every requested scope is withheld: report the first one's state rather
            # than the "completed" default, which would read as a genuine no-match.
            return FederatedPrivatePlan(
                withheld=CodeSearchResponse(
                    hits=[], total=0, requested_mode=body.mode, effective_mode=body.mode,
                    scope_snapshot_state=excluded_scopes[0]["state"],
                    active_index_job_id=excluded_scopes[0]["active_index_job_id"],
                    notice=(
                        "Every requested code scope is withheld while it is being indexed "
                        "or needs recovery; see private.excluded_scopes. Do not treat this "
                        "as a no-match result."
                    ),
                ),
                project=None, project_aliases=None, excluded_projects=None,
                excluded_scopes=excluded_scopes, searched_projects=list(body.projects),
            )
        flat = list(dict.fromkeys(p for group in included_groups for p in group))
        return FederatedPrivatePlan(
            withheld=None, project=(flat[0] if flat else None),
            project_aliases=(flat[1:] or None),
            excluded_projects=None, excluded_scopes=excluded_scopes,
            searched_projects=list(body.projects),
            prefixed_groups=tuple(prefixed_groups), only_prefixed=not flat,
        )

    # body.projects == []: search everything the owner has, excluding individually
    # withheld scopes (decision 10) instead of the single-project route's all-or-
    # nothing fail closed.
    candidates = _incomplete_scope_projects(owner_user_id=owner, queue=queue)
    if candidates is None:
        return FederatedPrivatePlan(
            withheld=CodeSearchResponse(
                hits=[], total=0, requested_mode=body.mode, effective_mode=body.mode,
                scope_snapshot_state="unavailable",
                notice=(
                    "Code KB search is temporarily withheld because Index Catalog "
                    "visibility cannot be verified. Check Index Catalog health and "
                    "retry shortly; do not treat this as a no-match result."
                ),
            ),
            project=None, project_aliases=None, excluded_projects=None,
            excluded_scopes=[], searched_projects=[],
        )

    must_not: set[str] = set()
    for raw in candidates:
        aliases = _scope_aliases(owner=owner, project=raw)
        decision = _private_scope_decision(owner=owner, projects=aliases, queue=queue)
        if not decision.allowed:
            excluded_scopes.append({
                "project": aliases[0],
                "state": decision.scope_snapshot_state,
                "active_index_job_id": decision.active_job_id,
            })
            must_not.update(aliases)
    return FederatedPrivatePlan(
        withheld=None, project=None, project_aliases=None,
        excluded_projects=(sorted(must_not) or None),
        excluded_scopes=excluded_scopes, searched_projects=[],
    )


def _merge_private_responses(
    responses: list[CodeSearchResponse], pool: int,
) -> CodeSearchResponse:
    """One response from the per-scope-group searches of the private leg.

    Every group embeds the same query in the same mode, so their scores are comparable
    and a plain best-first merge keeps the pool's strongest candidates."""
    if len(responses) == 1:
        return responses[0]
    hits = sorted(
        (hit for response in responses for hit in response.hits),
        key=lambda hit: hit.score, reverse=True)[:pool]
    notices = [response.notice for response in responses if response.notice]
    return responses[0].model_copy(update={
        "hits": hits, "total": len(hits),
        "notice": " ".join(dict.fromkeys(notices)) or None,
    })


def _run_federated_private_leg(
    body: FederatedCodeSearchRequest, owner: str, *, pool: int,
) -> tuple[CodeSearchResponse, list[dict], list[str]]:
    """The private leg, off the event loop. Always runs (R1: no include_private flag).

    Returns (response, excluded_scopes, searched_projects) so the route can populate
    ``FederatedLegDiagnostics`` without re-deriving decision-10 bookkeeping.
    """
    service = get_code_kb_service()
    queue = get_existing_code_index_queue()

    def do_search(
        *, project: str | None, project_aliases: list[str] | None,
        excluded_projects: list[str] | None, path_prefix: str | None = None,
    ) -> CodeSearchResponse:
        request = CodeSearchRequest(
            query=body.query, project=project, top_k=pool, mode=body.mode,
            rerank=False, language=body.language, path_glob=body.path_glob,
            max_per_file=body.max_per_file,
        )
        extra = {"path_prefix": path_prefix} if path_prefix else {}
        return service.search(
            request, owner_user_id=owner, project_aliases=project_aliases,
            excluded_projects=excluded_projects, **extra,
        )

    if not hasattr(service, "scope_snapshot"):
        response = do_search(
            project=(body.projects[0] if body.projects else None),
            project_aliases=(body.projects[1:] or None) if body.projects else None,
            excluded_projects=None,
        )
        return response, [], list(body.projects)

    with service.scope_snapshot(owner_user_id=owner):
        plan = _plan_federated_private_scopes(owner=owner, body=body, queue=queue)
        if plan.withheld is not None:
            return plan.withheld, plan.excluded_scopes, plan.searched_projects
        responses = []
        if not plan.only_prefixed:
            responses.append(do_search(
                project=plan.project, project_aliases=plan.project_aliases,
                excluded_projects=plan.excluded_projects))
        for aliases, prefix in plan.prefixed_groups:
            responses.append(do_search(
                project=aliases[0], project_aliases=(list(aliases[1:]) or None),
                excluded_projects=None, path_prefix=prefix))
        return (
            _merge_private_responses(responses, pool),
            plan.excluded_scopes, plan.searched_projects)


def _private_candidates(response: CodeSearchResponse) -> list[CodeFederationCandidate]:
    return [
        CodeFederationCandidate(
            origin="private", path=hit.path, start_line=hit.start_line,
            end_line=hit.end_line, score=hit.score, label=hit.label,
            language=hit.language, code=hit.code, project=hit.project, host=hit.host,
        )
        for hit in response.hits
    ]


def _public_candidates(hits) -> list[CodeFederationCandidate]:
    return [
        CodeFederationCandidate(
            origin="public", path=hit.path, start_line=hit.start_line,
            end_line=hit.end_line, score=hit.score, label=hit.label,
            language=hit.language, code=(hit.code or hit.content),
            source_id=hit.source_id, revision=hit.revision, domain=hit.domain,
        )
        for hit in hits
    ]


@router.post("/federated-search", response_model=FederatedCodeSearchResponse)
async def federated_code_search(
    body: FederatedCodeSearchRequest, request: Request,
    user_id: str | None = _UserId, tier: str | None = _Tier,
):
    """Search the caller's private Code KB and the curated Public Code corpus in one
    fused ranking (RRF by default, or cross-encoder reranked with rerank=true).

    The private leg always runs (R1, operator-confirmed 2026-09-23): there is no
    include_private flag. ``projects`` scopes it (empty = every scope the caller
    owns); an individually-withheld scope is excluded and reported in
    ``private.excluded_scopes`` rather than withholding the whole leg (decision 10).

    Fail-closed, never a silent downgrade: include_public=true requires admin tier —
    a non-admin caller gets 403 rather than a private-only result
    (docs/federated_code_search_plan.md section 1). Both legs retrieve concurrently
    with rerank=False; this route's own fusion (app.services.code_federation) decides
    RRF vs rerank across the combined pool.
    """
    route_start = time.perf_counter()
    # Identity is always resolved via resolve_owner (matches /search's single-plane
    # auth), even when include_public=false, so an anonymous/non-LAN caller is still
    # fail-closed on the private leg that always runs.
    owner = resolve_owner(request, user_id)
    if body.include_public:
        resolve_admin(request, user_id, tier)

    pool = _federated_pool(body.top_k)

    def run_public_leg():
        return get_public_code_search_service().search(PublicCodeSearchRequest(
            query=body.query, top_k=pool, mode=body.mode,
            source_ids=tuple(body.source_ids), domains=tuple(body.domains),
            language=body.language, rerank=False,
        ))

    tasks = [asyncio.to_thread(_run_federated_private_leg, body, owner, pool=pool)]
    if body.include_public:
        # Reuse /api/public-code/search's own error mapping (unknown source_id/domain
        # etc.) instead of letting a public-service ValueError/PublicSourceSelectionError
        # escape as an unhandled 500 -- the public leg must fail the same way that
        # route already does for the identical input.
        tasks.append(_run_public_code_service(run_public_leg))
    results = await asyncio.gather(*tasks)
    private_response, excluded_scopes, searched_projects = results[0]
    public_response = results[1] if body.include_public else None

    private_candidates = _private_candidates(private_response)
    public_candidates = (
        _public_candidates(public_response.hits) if public_response else []
    )

    reranker = get_shared_reranker() if body.rerank else None
    fused = fuse_code_hits(
        private_candidates, public_candidates,
        query=body.query, top_k=body.top_k, reranker=reranker,
    )
    hits = [FederatedCodeHit(**dataclasses.asdict(hit)) for hit in fused.hits]

    private_diag = FederatedLegDiagnostics(
        included=True,
        candidate_count=len(private_candidates),
        effective_mode=private_response.effective_mode,
        notice=private_response.notice,
        scope_snapshot_state=private_response.scope_snapshot_state,
        excluded_scopes=excluded_scopes,
        searched_projects=searched_projects,
    )
    public_diag = (
        FederatedLegDiagnostics(
            included=True,
            candidate_count=len(public_candidates),
            effective_mode=public_response.effective_mode,
            notice=public_response.notice,
            selected_source_ids=list(public_response.selected_source_ids),
        )
        if public_response is not None else
        FederatedLegDiagnostics(included=False)
    )

    # embed/retrieve come from the private leg (R1: it always runs, so it is the
    # meaningful source for those buckets); fuse/rerank are measured directly around
    # fuse_code_hits above; total is this route's own wall time, which is always >=
    # the sum of the others since fuse+rerank run strictly after both legs (awaited
    # concurrently via gather) return.
    return FederatedCodeSearchResponse(
        query=body.query, hits=hits, total=len(hits),
        fusion=fused.fusion, rerank_applied=fused.rerank_applied,
        private=private_diag, public=public_diag,
        timings_ms=CodeSearchTimingsMs(
            embed=private_response.timings_ms.embed,
            retrieve=private_response.timings_ms.retrieve,
            fuse=fused.fuse_ms,
            rerank=fused.rerank_ms,
            total=(time.perf_counter() - route_start) * 1000.0,
        ),
    )


@router.post("/duplicates", response_model=CodeDuplicateScanResponse)
async def find_duplicate_code(
    body: CodeDuplicateScanRequest, request: Request, user_id: str | None = _UserId,
):
    """Report files in one indexed scope that duplicate each other.

    Runs on a worker thread like search: a full scope scan issues one nearest-neighbour
    query per chunk (~29s for 8.6k chunks in the reference measurement) and must not
    occupy the event loop.
    """
    owner = resolve_owner(request, user_id)

    def run_scan() -> CodeDuplicateScanResponse:
        aliases: list[str] | None = None
        if is_windows_scope(body.project):
            try:
                aliases = get_index_catalog_service().equivalent_project_scopes(
                    owner_user_id=owner, project=body.project, kind="code")
            except Exception:  # noqa: BLE001 - a catalog outage must not fail the scan
                logger.exception(
                    "duplicate scan scope-alias lookup failed for project=%s", body.project)
                aliases = [body.project]
        service = get_code_kb_service()
        if aliases is None:
            return service.find_duplicates(body, owner_user_id=owner)
        return service.find_duplicates(body, owner_user_id=owner, project_aliases=aliases)

    return await asyncio.to_thread(run_scan)


@router.delete("", response_model=CodeDeleteResponse)
async def delete_project(
    request: Request,
    project: str = Query(..., min_length=1, max_length=2000),
    user_id: str | None = _UserId,
):
    """Clear one of the caller's code projects: drop ALL its indexed chunks AND its code
    inventory rows. `project` is the indexed directory's absolute path — passed as a
    query param because a path segment can't hold a scope with slashes / a drive colon.

    This is the explicit deletion path: a full re-index of an emptied tree is deliberately
    a NO-OP (so a broken walk can't silently wipe a real index), so clearing is explicit.
    """
    owner = resolve_owner(request, user_id)
    project = canonicalize_scope_path(project)

    def delete_scope() -> CodeDeleteResponse:
        catalog = get_index_catalog_service()
        scopes = [project]
        if is_windows_scope(project):
            # Resolve aliases before the first mutation; run catalog, queue and Qdrant
            # I/O together in this worker so DELETE never blocks the event loop.
            scopes = catalog.equivalent_project_scopes(
                owner_user_id=owner, project=project, kind="code")
        scopes = list(dict.fromkeys(scope for scope in scopes if scope))
        queue = get_code_index_queue()
        for scope in scopes:
            queue.cancel_project(owner_user_id=owner, project=scope)

        def mark_visibility(method_name: str, scope: str) -> bool:
            method = getattr(catalog, method_name, None)
            if method is None:
                # Production IndexCatalogService has this durable API. Retain backwards
                # compatibility for small test/embedding fakes that only model rows.
                return False
            method(
                owner_user_id=owner,
                host=_DELETE_VISIBILITY_HOST,
                project=scope,
                job_id=None,
            )
            return True

        prepared_scopes: list[str] = []
        catalog_deleted_scopes: set[str] = set()
        try:
            # Persist every alias before the first vector delete. A search that reaches
            # the owner fence during this operation observes preparing instead of a
            # partial mixed-case Windows scope.
            for scope in scopes:
                if mark_visibility("mark_code_scope_preparing", scope):
                    prepared_scopes.append(scope)
            resp = get_code_kb_service().delete_project(
                project, owner_user_id=owner, project_aliases=scopes)
            resp.catalog_removed = 0
            for scope in scopes:
                resp.catalog_removed += catalog.delete_project(
                    owner_user_id=owner, project=scope, kind="code")
                catalog_deleted_scopes.add(scope)
            return resp
        except Exception:
            # A vector or catalog failure after preparation can leave only some aliases
            # gone. Preserve a durable recovery boundary for every not-yet-cleared scope
            # instead of allowing its remaining/partial vectors to look completed.
            for scope in prepared_scopes:
                if scope in catalog_deleted_scopes:
                    continue
                try:
                    mark_visibility("mark_code_scope_recovery_required", scope)
                except Exception:  # noqa: BLE001 - retain original delete failure
                    logger.exception(
                        "code delete could not persist recovery visibility owner=%s project=%s",
                        owner,
                        scope,
                    )
            raise

    return await asyncio.to_thread(delete_scope)
