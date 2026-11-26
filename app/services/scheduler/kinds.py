"""Registry of the job kinds a schedule can run.

A *kind* is code: what to start (`fire`), how its parameters look
(`params_model`, which also drives the UI form via `model_json_schema()`), how a
failed run is retried, which gauges describe it. A *schedule* is data (a row in
`schedules`) that names a kind, a trigger and parameters. Adding a recurring job is
registering a kind and, if it should exist out of the box, listing a seed.

Kinds register themselves when their module is imported; `BUILTIN_KIND_MODULES` is
that list, loaded by `load_builtin_kinds()` (one failing module is logged, not fatal).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from loguru import logger
from pydantic import BaseModel

FireStatus = Literal["enqueued", "skipped", "noop"]


@dataclass(frozen=True)
class FireResult:
    """What one trigger came to.

    enqueued  work was started (`task_id` is the Task when there is one)
    skipped   deliberately not started, e.g. the same job is still running
    noop      there was nothing to do
    """

    status: FireStatus
    task_id: Optional[int] = None
    reason: Optional[str] = None

    def as_dict(self) -> dict:
        return {"status": self.status, "task_id": self.task_id, "reason": self.reason}


@dataclass(frozen=True)
class RetryPolicy:
    """Automatic re-drive of a failed task: Airflow's retries / retry_delay /
    retry_exponential_backoff / max_retry_delay."""

    max_retries: int = 0
    base_delay_s: int = 1800
    backoff: float = 2.0
    max_delay_s: int = 6 * 3600

    def delay_s(self, attempt: int) -> float:
        """Delay before retry number `attempt + 1` (attempt counts retries already made)."""
        return float(min(self.base_delay_s * self.backoff ** attempt, self.max_delay_s))


@dataclass(frozen=True)
class Metric:
    """One labelled gauge on a schedule card: name, value, unit."""

    name: str
    value: float | int | str | None
    unit: str = ""


@dataclass(frozen=True)
class ScheduleSeed:
    """A built-in schedule, inserted once if its key is absent (never overwritten)."""

    key: str
    title: str
    trigger: dict
    params: dict = field(default_factory=dict)
    catchup: Literal["latest", "skip"] = "latest"
    misfire_grace_s: Optional[int] = None
    enabled: bool = True


@dataclass(frozen=True)
class JobKind:
    name: str
    title: str
    params_model: type[BaseModel]
    # (validated params, origin) -> FireResult. Must return quickly: the long work
    # runs in the runner's own thread, not in the scheduler's.
    fire: Callable[[BaseModel, dict], FireResult]
    # Polling kinds (every few seconds) set this False so "nothing to do" does not
    # write a tick per poll.
    record_idle_ticks: bool = True
    # Seconds a due run may be late and still run (APScheduler misfire_grace_time).
    # None = never too late. A schedule row may override it.
    default_misfire_grace_s: Optional[int] = 3600
    retry: RetryPolicy = RetryPolicy()
    status: Optional[Callable[[], list[Metric]]] = None
    defaults: tuple[ScheduleSeed, ...] = ()


BUILTIN_KIND_MODULES: tuple[str, ...] = (
    "app.services.public_code.update_scheduler",
    "app.services.pipeline.arxiv_schedule",
)

_kinds: dict[str, JobKind] = {}
_builtins_loaded = False


def register_kind(kind: JobKind) -> None:
    _kinds[kind.name] = kind


def get_kind(name: str) -> JobKind:
    load_builtin_kinds()
    try:
        return _kinds[name]
    except KeyError:
        raise KeyError(f"unknown schedule kind {name!r}") from None


def all_kinds() -> dict[str, JobKind]:
    load_builtin_kinds()
    return dict(_kinds)


def load_builtin_kinds() -> None:
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True
    for module in BUILTIN_KIND_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:
            logger.warning(f"scheduler.kinds.load_failed module={module}: {exc}")
