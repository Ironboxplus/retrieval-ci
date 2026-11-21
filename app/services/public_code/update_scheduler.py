"""Background scheduler for persisted public-code source update policies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import threading
from typing import Any, Protocol


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
        poll_interval_seconds: float = 30.0,
    ):
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self._repository = repository
        self._control_service = control_service
        self._job_service = job_service
        self._maintenance_service = maintenance_service
        self._poll_interval_seconds = poll_interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="public-code-update-scheduler",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, timeout_seconds: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_seconds)

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

    def _run(self) -> None:
        while not self._stop_event.is_set():
            cycle = self.run_once()
            if cycle.checked_source_ids:
                logger.info(
                    "public-code scheduler checked=%d queued=%d skipped_active=%d errors=%d",
                    len(cycle.checked_source_ids),
                    len(cycle.queued_source_ids),
                    len(cycle.skipped_active_source_ids),
                    len(cycle.errors),
                )
            self._stop_event.wait(self._poll_interval_seconds)
