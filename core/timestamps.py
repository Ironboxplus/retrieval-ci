"""Timestamps that leave the API are RFC 3339 with an explicit offset.

The Task tables store *naive* datetimes, in two different conventions depending
on who wrote the column:

* `created_at` / `updated_at` come from `server_default=func.now()` / `onupdate`,
  i.e. SQLite CURRENT_TIMESTAMP: naive **UTC**. (Task, TaskSubtask, TaskEvent,
  TaskResourceSample, TaskWorkflowEdge, Schedule.)
* `started_at` / `completed_at` / `last_heartbeat` are written by Python as
  `datetime.now()` (TaskStore, task_runtime, app/tasks/base.py): naive **server
  local** wall-clock time.

Serialising either one as-is makes every client guess. The stored data is left
alone; these helpers apply the writer's convention at the serialisation boundary
and return a timezone-aware value (UTC), or the RFC 3339 string for it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def from_utc_naive(value: Optional[datetime]) -> Optional[datetime]:
    """A naive column filled by the database default (CURRENT_TIMESTAMP) -> aware UTC."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def from_local_naive(value: Optional[datetime]) -> Optional[datetime]:
    """A naive column written with datetime.now() -> aware UTC.

    `astimezone()` on a naive datetime interprets it in the system's local zone,
    which is the zone `datetime.now()` used when it was written.
    """
    return None if value is None else value.astimezone(timezone.utc)


def from_epoch(value: Optional[float]) -> Optional[datetime]:
    return None if value is None else datetime.fromtimestamp(value, tz=timezone.utc)


def rfc3339(value: Optional[datetime]) -> Optional[str]:
    """Aware datetime -> `2026-11-03T03:35:12.123456Z` (UTC, same shape pydantic emits)."""
    if value is None:
        return None
    text = value.astimezone(timezone.utc).isoformat()
    return text[:-6] + "Z" if text.endswith("+00:00") else text


def utc_naive_iso(raw: Optional[str]) -> Optional[str]:
    """Re-stamp an `isoformat()` string of a naive-UTC column."""
    return None if raw is None else rfc3339(from_utc_naive(datetime.fromisoformat(raw)))


def local_naive_iso(raw: Optional[str]) -> Optional[str]:
    """Re-stamp an `isoformat()` string of a naive-local column."""
    return None if raw is None else rfc3339(from_local_naive(datetime.fromisoformat(raw)))
