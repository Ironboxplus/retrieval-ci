"""Persisted public-code source update policies: one check-and-queue cycle, plus
the `public_code_updates` schedule kind that runs it every 30 s.

Timing belongs to app.services.scheduler; this module only knows what a cycle does.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any, Protocol

from pydantic import BaseModel

from app.services.scheduler.kinds import FireResult, JobKind, ScheduleSeed, register_kind


logger = logging.getLogger(__name__)


class ScheduledSourceRepository(Protocol):
    def list_due_scheduled_source_ids(self, *, now: datetime | None = None) -> tuple[str, ...]: ...


class ScheduledUpdateControl(Protocol):
    def check_updates(self, *, source_ids: tuple[str, ...], domains: tuple[str, ...]) -> Any: ...


class ScheduledIndexJobs(Protocol):
    def has_active_index(self, source_id: str) -> bool: ...

    def submit_index(self, *, source_ids: tuple[str, ...], domains: tuple[str, ...]) -> Any: ...


class ScheduledMaintenance(Protocol):
    def run_once(self) -> Any: ...


@dataclass(frozen=True, slots=True)
class PublicCodeUpdateCycle:
    checked_source_ids: tuple[str, ...]
    queued_source_ids: tuple[str, ...]
    skipped_active_source_ids: tuple[str, ...]
    errors: dict[str, str]


class PublicCodeUpdateScheduler:
    """Run remote checks and queue existing index jobs outside request/search paths."""

    def __init__(
        self,
        *,
        repository: ScheduledSourceRepository,
        control_service: ScheduledUpdateControl,
        job_service: ScheduledIndexJobs,
        maintenance_service: ScheduledMaintenance | None = None,
    ):
        self._repository = repository
        self._control_service = control_service
        self._job_service = job_service
        self._maintenance_service = maintenance_service

    def run_once(self, *, now: datetime | None = None) -> PublicCodeUpdateCycle:
        if self._maintenance_service is not None:
            self._maintenance_service.run_once()
        due_source_ids = self._repository.list_due_scheduled_source_ids(now=now)
        checked: list[str] = []
        queued: list[str] = []
        skipped_active: list[str] = []
        errors: dict[str, str] = {}
        if not due_source_ids:
            return PublicCodeUpdateCycle((), (), (), {})
        try:
            checks = self._control_service.check_updates(
                source_ids=due_source_ids,
                domains=(),
            )
        except Exception as exc:
            logger.exception("scheduled public-code update batch failed")
            return PublicCodeUpdateCycle(
                checked_source_ids=due_source_ids,
                queued_source_ids=(),
                skipped_active_source_ids=(),
                errors={source_id: str(exc) for source_id in due_source_ids},
            )
        update_source_ids: list[str] = []
        for check in checks:
            source_id = check.source_id
            checked.append(source_id)
            try:
                if check.status == "check_failed":
                    errors[source_id] = check.message or check.status
                    continue
                if check.status not in {"update_available", "not_mirrored"}:
                    continue
                if self._job_service.has_active_index(source_id):
                    skipped_active.append(source_id)
                    continue
                update_source_ids.append(source_id)
            except Exception as exc:
                errors[source_id] = str(exc)
                logger.exception("scheduled public-code update failed for %s", source_id)
        if update_source_ids:
            self._job_service.submit_index(
                source_ids=tuple(update_source_ids),
                domains=(),
            )
            queued.extend(update_source_ids)
        return PublicCodeUpdateCycle(
            checked_source_ids=tuple(checked),
            queued_source_ids=tuple(queued),
            skipped_active_source_ids=tuple(skipped_active),
            errors=errors,
        )


class PublicCodeUpdateParams(BaseModel):
    """No parameters: which sources are due is stored per source (update policy)."""


def fire_public_code_updates(params: PublicCodeUpdateParams, origin: dict) -> FireResult:
    from app.services.public_code import get_public_code_update_scheduler

    cycle = get_public_code_update_scheduler().run_once()
    if not cycle.checked_source_ids:
        return FireResult("noop")
    counts = (
        f"checked={len(cycle.checked_source_ids)} queued={len(cycle.queued_source_ids)} "
        f"skipped_active={len(cycle.skipped_active_source_ids)} errors={len(cycle.errors)}"
    )
    logger.info("public-code scheduler %s", counts)
    if cycle.errors:
        detail = "; ".join(f"{source}: {error}" for source, error in cycle.errors.items())
        raise RuntimeError(f"{counts} [{detail}]"[:1000])
    if cycle.queued_source_ids:
        return FireResult("enqueued", reason=counts)
    return FireResult("noop", reason=counts)


register_kind(JobKind(
    name="public_code_updates",
    title="Public code source update checks",
    params_model=PublicCodeUpdateParams,
    fire=fire_public_code_updates,
    # Polls every 30 s; a poll with nothing due is not worth a row.
    record_idle_ticks=False,
    default_misfire_grace_s=60,
    defaults=(ScheduleSeed(
        key="public_code_updates",
        title="Public code source update checks",
        trigger={"type": "interval", "seconds": 30, "timezone": "UTC"},
        # Polling: a missed poll is simply the next one, never replayed.
        catchup="skip",
    ),),
))
