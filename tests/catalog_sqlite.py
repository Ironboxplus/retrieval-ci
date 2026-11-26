"""A real SQLite index catalog (and the SQL it executes) for scope-resolution tests.

``capture_catalog_plans`` records every statement a block sends to SQLite and then runs
``EXPLAIN QUERY PLAN`` on each one with its real parameters, so a test can assert that a
code path never full-scans ``index_catalog`` without knowing which statements it issues.
"""
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.services.index_catalog.repository import IndexCatalogRepository
from app.services.index_catalog.service import IndexCatalogService
from core.db.database import Base


class CatalogDb:
    def __init__(self, path=None) -> None:
        """In memory unless ``path`` names a file (a catalog too big to hold in RAM)."""
        if path is None:
            self.engine = create_engine(
                "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        else:
            self.engine = create_engine(
                f"sqlite:///{path}", connect_args={"check_same_thread": False})

            @event.listens_for(self.engine, "connect")
            def _fast_scratch_file(connection, _record):
                # A throwaway fixture: durability is worthless, the bulk load is not.
                connection.execute("PRAGMA journal_mode=MEMORY")
                connection.execute("PRAGMA synchronous=OFF")
                connection.execute("PRAGMA cache_size=-262144")
        Base.metadata.create_all(self.engine)
        self._session_local = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.repository = IndexCatalogRepository(session_factory=self.session)
        self.service = IndexCatalogService(repository=self.repository)

    @contextmanager
    def session(self):
        session = self._session_local()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def add_files(self, project, paths, *, owner="owner", kind="code", host="host",
                  chunk_count=1, deleted=False) -> None:
        from core.db.models import IndexCatalog
        from datetime import datetime
        with self.session() as session:
            session.add_all([
                IndexCatalog(
                    owner_user_id=owner, host=host, project=project, root_dir=project,
                    rel_path=path, kind=kind, chunk_count=chunk_count,
                    deleted_at=datetime(2026, 1, 1) if deleted else None)
                for path in paths
            ])

    def bulk_noise(self, *, owner="owner", scopes=4, files=2000) -> None:
        """Rows of other scopes: a planner that ignores the project sees them all."""
        from sqlalchemy import insert
        from core.db.models import IndexCatalog
        with self.engine.begin() as conn:
            conn.execute(insert(IndexCatalog), [
                dict(owner_user_id=owner, host="host", project=f"/noise/scope{s}",
                     root_dir=f"/noise/scope{s}", rel_path=f"app/m{i // 20}/f{i}.py",
                     kind="code", chunk_count=1)
                for s in range(scopes) for i in range(files)
            ])

    def production_shaped(self, *, owner="internal-admin", rows=600_000,
                          scopes=300, stamped=False) -> None:
        """One owner holding nearly every row, spread over many scopes, with no
        ``sqlite_stat1`` (production never ran ANALYZE, so the planner has no statistics
        to rescue a query that reads more than it should).

        ``stamped`` gives every file an ``indexed_at`` (one stamp per scope, as one index run
        stamps its whole batch) and a ``git_commit`` in ``metadata_json``, like indexed rows."""
        per_scope = rows // scopes
        columns = "owner_user_id, host, project, root_dir, rel_path, kind, chunk_count"
        values = "?, 'host', ?, ?, ?, 'code', 1"
        if stamped:
            columns += ", indexed_at, metadata_json"
            values += ", ?, ?"
        with self.engine.begin() as conn:
            for start in range(0, scopes, 20):
                conn.exec_driver_sql(
                    f"INSERT INTO index_catalog ({columns}) VALUES ({values})",
                    [(owner, f"E:/work/scope{s:03d}", f"E:/work/scope{s:03d}",
                      f"app/m{i // 20}/f{i}.py",
                      *((f"2026-11-0{1 + s % 9} 12:00:00.000000", f'{{"git_commit": "c{s:03d}"}}')
                        if stamped else ()))
                     for s in range(start, min(start + 20, scopes))
                     for i in range(per_scope)])
        with self.engine.connect() as conn:
            assert not conn.exec_driver_sql(
                "SELECT 1 FROM sqlite_master WHERE name = 'sqlite_stat1'").fetchall()


@contextmanager
def capture_catalog_plans(db: CatalogDb):
    """Yield a list that is filled, on exit, with (sql, plan_details) for every
    statement that touched ``index_catalog`` inside the block."""
    statements: list[tuple[str, tuple]] = []

    def record(conn, cursor, statement, parameters, context, executemany):
        if "index_catalog" in statement and not statement.lstrip().upper().startswith(
                ("PRAGMA", "EXPLAIN")):
            statements.append((statement, tuple(parameters)))

    event.listen(db.engine, "before_cursor_execute", record)
    plans: list[tuple[str, list[str]]] = []
    try:
        yield plans
    finally:
        event.remove(db.engine, "before_cursor_execute", record)
        with db.engine.connect() as conn:
            raw = conn.connection.driver_connection
            for statement, parameters in statements:
                rows = raw.execute("EXPLAIN QUERY PLAN " + statement, parameters).fetchall()
                plans.append((statement, [row[3] for row in rows]))


def full_scans(plans) -> list[tuple[str, str]]:
    """Plan lines that read ``index_catalog`` without a key lookup (SQLite says SCAN)."""
    return [(statement, line) for statement, details in plans for line in details
            if line.startswith("SCAN") and "index_catalog" in line]


def _is_bounded(line: str) -> bool:
    return (
        "project=?" in line
        or ("rel_path>" in line and "rel_path<" in line)
        or ("<expr>>" in line and "<expr><" in line)
        or ("ix_index_catalog_live_project_fold" in line and "<expr>=?" in line)
        or "ix_index_catalog_code_odd_path" in line
        or "ix_index_catalog_live_odd_project" in line
    )


def unbounded_reads(plans) -> list[tuple[str, str]]:
    """Plan lines that read a whole owner's (or the whole table's) catalog rows.

    ``owner=? AND kind=? AND deleted_at=?`` is a SEARCH to SQLite yet still walks every
    code row of the owner, which is exactly the cost this change removes; so does a range
    with only one end. A bounded read pins the scope (``project=?`` or its ``lower(project)``
    spellings) or both ends of a path range (``rel_path`` / ``lower(rel_path)``); the
    partial indexes of legacy path and scope spellings hold only those rows, so they are
    exempt."""
    return [(statement, line) for statement, details in plans for line in details
            if "index_catalog" in line and (line.startswith("SCAN") or not _is_bounded(line))]
