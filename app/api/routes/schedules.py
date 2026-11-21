"""Recurring-job schedules (internal / gateway-admin only).

A schedule names a registered job kind, a trigger (daily / weekly / interval) and
the kind's parameters; the engine in app/services/scheduler fires it and records
every trigger as a tick. Not part of the public API surface.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import ValidationError

from app.api.identity import resolve_admin
from app.models.schemas import (
    ScheduleCreateRequest,
    ScheduleDeleteResponse,
    ScheduleFireResponse,
    SchedulePatchRequest,
    SchedulePreviewRequest,
    SchedulePreviewResponse,
    ScheduleKindResponse,
    ScheduleResponse,
    ScheduleTickResponse,
)
from app.services.scheduler.service import SchedulerService, get_scheduler_service
from app.services.scheduler.triggers import next_fire_times, parse_trigger

_AUTH_RESPONSES = {
    401: {"description": "Authentication required"},
    403: {"description": "Admin access required"},
}


def _admin(
    request: Request,
    user_id: Optional[str] = Header(default=None, alias="X-Retrival-User-Id"),
    tier: Optional[str] = Header(default=None, alias="X-Retrival-User-Tier"),
) -> str:
    return resolve_admin(request, user_id, tier)


def get_service() -> SchedulerService:
    return get_scheduler_service()


router = APIRouter(dependencies=[Depends(_admin)], responses=_AUTH_RESPONSES)


async def _run(operation: Callable[[], Any]) -> Any:
    """Run a blocking service call off the event loop, mapping its errors to HTTP."""
    try:
        return await asyncio.to_thread(operation)
    except ValidationError as exc:
        raise HTTPException(
            422, detail=exc.errors(include_url=False, include_context=False, include_input=False),
        ) from exc
    except LookupError as exc:
        raise HTTPException(404, detail=str(exc.args[0]) if exc.args else "not found") from exc
    except PermissionError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc


@router.get("/kinds", response_model=list[ScheduleKindResponse], summary="List schedulable job kinds")
async def list_kinds(service: SchedulerService = Depends(get_service)):
    return await _run(service.kinds_info)


@router.get("", response_model=list[ScheduleResponse], summary="List schedules with live state")
async def list_schedules(service: SchedulerService = Depends(get_service)):
    return await _run(service.list_schedules)


@router.post("", response_model=ScheduleResponse, status_code=201, summary="Create a schedule")
async def create_schedule(body: ScheduleCreateRequest, service: SchedulerService = Depends(get_service)):
    def create():
        try:
            service.create(**body.model_dump())
        except KeyError as exc:  # duplicate key
            raise HTTPException(409, detail=str(exc.args[0])) from exc
        return next(s for s in service.list_schedules() if s["key"] == body.key)

    return await _run(create)


@router.patch("/{schedule_id}", response_model=ScheduleResponse, summary="Edit a schedule")
async def patch_schedule(
    schedule_id: int, body: SchedulePatchRequest, service: SchedulerService = Depends(get_service),
):
    def patch():
        service.update(schedule_id, body.model_dump(exclude_unset=True))
        return next(s for s in service.list_schedules() if s["id"] == schedule_id)

    return await _run(patch)


@router.delete("/{schedule_id}", response_model=ScheduleDeleteResponse, summary="Delete a schedule")
async def delete_schedule(schedule_id: int, service: SchedulerService = Depends(get_service)):
    await _run(lambda: service.delete(schedule_id))
    return ScheduleDeleteResponse(id=schedule_id)


@router.post("/{schedule_id}/run", response_model=ScheduleFireResponse, summary="Run a schedule now")
async def run_schedule(schedule_id: int, service: SchedulerService = Depends(get_service)):
    return (await _run(lambda: service.run_now(schedule_id))).as_dict()


@router.get("/{schedule_id}/ticks", response_model=list[ScheduleTickResponse], summary="Trigger history")
async def list_ticks(
    schedule_id: int,
    limit: int = Query(50, ge=1, le=200),
    before_id: Optional[int] = Query(None, ge=1, description="Return ticks older than this tick id"),
    service: SchedulerService = Depends(get_service),
):
    def ticks():
        if service.get(schedule_id) is None:
            raise LookupError(f"schedule {schedule_id} not found")
        return service.list_ticks(schedule_id, limit, before_id)

    return await _run(ticks)


@router.post("/preview", response_model=SchedulePreviewResponse, summary="Preview a trigger's next fire times")
async def preview_trigger(body: SchedulePreviewRequest):
    spec = await _run(lambda: parse_trigger(body.trigger))
    return SchedulePreviewResponse(
        next_fire_times=[t.isoformat() for t in next_fire_times(spec, body.count)],
    )
