"""A real SQLite file with the application's tables, for tests that need the
Tasks / Schedules tables without touching data/db/retrieval.db.

Mirrors core/db/database.py: foreign keys enforced, 30 s busy timeout, a session
factory that commits on exit. Not a fake: it is the same engine type, schema and
SQL the application runs.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from core.db import models  # noqa: F401 -- registers every table on Base
from core.db.database import Base


def make_session_factory(path: Path):
    engine = create_engine(
        f"sqlite:///{path}", connect_args={"check_same_thread": False, "timeout": 30},
    )

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

    Base.metadata.create_all(engine)
    local = sessionmaker(bind=engine, expire_on_commit=False)

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
