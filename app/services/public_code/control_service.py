"""Control-plane operations for public-code source maintenance."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Protocol

from app.services.public_code.source_catalog import PublicCodeSource
from app.services.public_code.update_checks import PublicCodeUpdateCheck


class PublicCodeSourceStateRepository(Protocol):
    """Persistent public-code source state required by the control plane."""

    def list_sources(self) -> Any:
        """List durable source state."""

    def record_update_checks(self, checks: Sequence[PublicCodeUpdateCheck]) -> None:
        """Persist observed remote/mirror revisions."""

    def set_update_policy(self, **kwargs: Any) -> Any:
        """Persist one source update policy."""

    def create_source(self, source: PublicCodeSource, **kwargs: Any) -> Any:
        """Persist one operator-managed source."""


class PublicCodeUpdateChecker(Protocol):
    """Read-only remote revision probe boundary."""

    def check(
        self,
        *,
        source_ids: Sequence[str] = (),
        domains: Sequence[str] = (),
        progress_cb: Callable[[PublicCodeUpdateCheck, int, int], None] | None = None,
    ) -> tuple[PublicCodeUpdateCheck, ...]:
        """Check selected sources without downloading or indexing."""


class PublicCodeControlService:
    """One safe entry point for source lists, update checks, and update policies."""

    def __init__(
        self,
        *,
        repository: PublicCodeSourceStateRepository,
        update_checks: PublicCodeUpdateChecker,
        source_validator: Callable[[PublicCodeSource], None] | None = None,
    ):
        self._repository = repository
        self._update_checks = update_checks
        self._source_validator = source_validator

    def list_sources(self) -> Any:
        return self._repository.list_sources()

    def check_updates(
        self,
        *,
        source_ids: Sequence[str] = (),
        domains: Sequence[str] = (),
        progress_cb: Callable[[PublicCodeUpdateCheck, int, int], None] | None = None,
    ) -> tuple[PublicCodeUpdateCheck, ...]:
        """Check one, many, or all sources and durably record the observation."""

        checks = self._update_checks.check(
            source_ids=source_ids,
            domains=domains,
            progress_cb=progress_cb,
        )
        self._repository.record_update_checks(checks)
        return checks

    def set_update_policy(
        self,
        *,
        source_id: str,
        update_policy: str,
        update_interval_minutes: int | None,
    ) -> Any:
        """Persist the operator policy used by one-click update jobs."""

        return self._repository.set_update_policy(
            source_id=source_id,
            update_policy=update_policy,
            update_interval_minutes=update_interval_minutes,
        )

    def register_source(
        self,
        source: PublicCodeSource,
        *,
        update_policy: str,
        update_interval_minutes: int | None,
        created_by_user_id: str | None = None,
    ) -> Any:
        """Register a validated source in the durable runtime registry."""

        if self._source_validator is not None:
            self._source_validator(source)
        return self._repository.create_source(
            source,
            update_policy=update_policy,
            update_interval_minutes=update_interval_minutes,
            created_by_user_id=created_by_user_id,
        )
