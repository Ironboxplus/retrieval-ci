"""Engine / session factory for the gateway app-tier DB (its own SQLite, separate
from the 4090 compute DB)."""
from __future__ import annotations

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from gateway.db.models import Base


def _migrate_sqlite_schema(engine) -> None:
    """Apply the small additive migrations needed by the persistent gateway DB."""
    if engine.dialect.name != "sqlite":
        return
    inspector = inspect(engine)
    if "idempotency_requests" not in inspector.get_table_names():
        return
    columns = {
        column["name"]
        for column in inspector.get_columns("idempotency_requests")
    }
    if "lease_token" in columns:
        return

    with engine.begin() as connection:
        connection.execute(text(
            "ALTER TABLE idempotency_requests "
            "ADD COLUMN lease_token VARCHAR(32)"
        ))
        connection.execute(text(
            "UPDATE idempotency_requests "
            "SET lease_token = lower(hex(randomblob(16))) "
            "WHERE lease_token IS NULL OR lease_token = ''"
        ))


def make_engine(url: str = "sqlite://"):
    kw: dict = {}
    if url.startswith("sqlite"):
        kw["connect_args"] = {"check_same_thread": False}
        if url in ("sqlite://", "sqlite:///:memory:"):
            kw["poolclass"] = StaticPool  # one shared in-memory DB across connections
    engine = create_engine(url, **kw)
    Base.metadata.create_all(engine)
    _migrate_sqlite_schema(engine)
    return engine


def make_session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)
