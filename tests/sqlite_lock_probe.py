"""Measure how long other connections keep SQLite's single write lock.

``WriteLockProbe`` is a real second connection to the same database file. A thread loops
``BEGIN IMMEDIATE`` / ``ROLLBACK`` and records how long each ``BEGIN IMMEDIATE`` waited for the
write lock; the longest wait is the longest time any other connection held it. Nothing is
mocked: the holder is whatever production code runs against the same file.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from core.db import models  # noqa: F401 -- registers every table on Base
from core.db.database import Base


class WriteLockProbe:
    def __init__(self, path: Path, *, interval: float = 0.02, busy_timeout_s: float = 120.0):
        self.path = Path(path)
        self.interval = interval
        self.busy_timeout_s = busy_timeout_s
        self.waits: list[tuple[float, float]] = []      # (monotonic start, seconds waited)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        conn = sqlite3.connect(str(self.path), timeout=self.busy_timeout_s, isolation_level=None)
        try:
            conn.execute(f"PRAGMA busy_timeout={int(self.busy_timeout_s * 1000)}")
            while not self._stop.is_set():
                started = time.monotonic()
                conn.execute("BEGIN IMMEDIATE")
                self.waits.append((started, time.monotonic() - started))
                conn.execute("ROLLBACK")
                self._stop.wait(self.interval)
        finally:
            conn.close()

    def __enter__(self) -> "WriteLockProbe":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=self.busy_timeout_s + 5)

    @property
    def longest_wait(self) -> float:
        return max((waited for _, waited in self.waits), default=0.0)


class HeldWriteLock:
    """A second real connection that takes SQLite's write lock and keeps it until released."""

    def __init__(self, path: Path):
        self._conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self._conn.execute("BEGIN IMMEDIATE")

    def release(self) -> None:
        if self._conn is not None:
            self._conn.execute("ROLLBACK")
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "HeldWriteLock":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def make_wal_session_factory(path: Path, *, busy_timeout_ms: int = 30000):
    """The application's engine settings (core/db/database.py) on a WAL file, committing sessions.

    ``busy_timeout_ms`` defaults to the application's 30 s; a test that must see "locked" within a
    second passes a short one.
    """
    engine = create_engine(
        f"sqlite:///{Path(path).as_posix()}",
        connect_args={"check_same_thread": False, "timeout": busy_timeout_ms / 1000})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        cursor.close()

    Base.metadata.create_all(engine)
    local = sessionmaker(autoflush=False, bind=engine, expire_on_commit=False)

    @contextmanager
    def factory():
        session = local()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    factory.engine = engine
    return factory
