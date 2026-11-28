"""Retry an idempotent operation after its failed SQLite transaction was rolled back."""
from __future__ import annotations

import sqlite3
import time
from sqlalchemy.exc import OperationalError


def is_lock_contention(exc: Exception) -> bool:
    original = getattr(exc, "orig", exc)
    code = getattr(original, "sqlite_errorcode", None)
    if code is not None:
        return (code & 0xff) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
    return any(s in str(original).lower() for s in ("database is locked", "database table is locked"))


def retry_transaction(operation, *, attempts: int = 4):
    """The caller must roll back/close each attempt and persist replay-safe checkpoints."""
    for attempt in range(attempts):
        try:
            return operation()
        except (sqlite3.OperationalError, OperationalError) as exc:
            if not is_lock_contention(exc) or attempt == attempts - 1:
                raise
            time.sleep(min(2.0, 0.25 * 2 ** attempt))
