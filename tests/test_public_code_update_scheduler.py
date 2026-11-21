"""TDD for scheduled public-source checks that stay off the search path."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.public_code.update_checks import PublicCodeUpdateCheck
from app.services.public_code.update_scheduler import PublicCodeUpdateScheduler


@dataclass
class FakeRepository:
    due_source_ids: tuple[str, ...]

    def list_due_scheduled_source_ids(self, *, now):
        return self.due_source_ids


@dataclass
class FakeControlService:
    checks: dict[str, PublicCodeUpdateCheck]
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def check_updates(self, *, source_ids, domains):
        self.calls.append(tuple(source_ids))
        return tuple(self.checks[source_id] for source_id in source_ids)


@dataclass
class FakeJobService:
    active_source_ids: set[str] = field(default_factory=set)
    submitted: list[str] = field(default_factory=list)

    def has_active_index(self, source_id):
        return source_id in self.active_source_ids

    def submit_index(self, *, source_ids, domains):
        self.submitted.extend(source_ids)


def check(source_id: str, status: str) -> PublicCodeUpdateCheck:
    return PublicCodeUpdateCheck(
        source_id=source_id,
        domain="fixture",
        local_revision="old",
        remote_revision="new",
        status=status,
    )


def test_scheduler_checks_due_sources_and_only_queues_sources_that_need_indexing():
    repository = FakeRepository(("current", "updated", "missing", "failed"))
    control = FakeControlService({
        "current": check("current", "up_to_date"),
        "updated": check("updated", "update_available"),
        "missing": check("missing", "not_mirrored"),
        "failed": check("failed", "check_failed"),
    })
    jobs = FakeJobService(active_source_ids={"missing"})
    scheduler = PublicCodeUpdateScheduler(
        repository=repository,
        control_service=control,
        job_service=jobs,
    )

    result = scheduler.run_once()

    assert control.calls == [("current", "updated", "missing", "failed")]
    assert jobs.submitted == ["updated"]
    assert result.checked_source_ids == ("current", "updated", "missing", "failed")
    assert result.queued_source_ids == ("updated",)
    assert result.skipped_active_source_ids == ("missing",)
    assert result.errors == {"failed": "check_failed"}
