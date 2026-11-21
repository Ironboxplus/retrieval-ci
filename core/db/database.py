"""
数据库连接管理
"""
from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from pathlib import Path

from config import settings

DB_PATH = Path(settings.data_dir) / "db" / "retrieval.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
    echo=False
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    """Ensure SQLite enforces foreign keys for every connection."""
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=30000")
    finally:
        cursor.close()

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
)
Base = declarative_base()


def get_db():
    """获取数据库会话（用于FastAPI依赖注入）"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session():
    """获取数据库会话（用于非FastAPI场景）"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """初始化数据库表"""
    from core.db import models
    Base.metadata.create_all(bind=engine)
    _apply_idempotent_migrations()
    cleanup_orphan_runtime_rows()


# Indexes of read paths (search-time path selection, scope spellings, the inventory summary)
# that create_all() cannot add to an existing table; installed by _apply_idempotent_migrations.
_CATALOG_READ_INDEXES = frozenset({
    "ix_index_catalog_code_scope_path",
    "ix_index_catalog_code_path_lookup",
    "ix_index_catalog_code_scope_lower_path",
    "ix_index_catalog_code_odd_path",
    "ix_index_catalog_code_odd_path_any_scope",
    "ix_index_catalog_live_project_fold",
    "ix_index_catalog_live_odd_project",
    "ix_index_catalog_live_summary",
})


def _apply_idempotent_migrations() -> None:
    """Schema patches that create_all() can't apply to pre-existing tables.

    SQLite has no native IF NOT EXISTS for ADD COLUMN, so we probe via
    PRAGMA table_info and add what's missing. Each migration is its own
    (table, column, ddl) triple — safe to re-run.
    """
    migrations = [
        ("crawl_records", "status_code", "INTEGER"),
        ("crawl_records", "retry_count", "INTEGER"),
        ("crawl_records", "crawl_time", "REAL"),
        ("crawl_records", "proxy_used", "VARCHAR(500)"),
        ("public_code_sources", "default_ref", "VARCHAR(255) NOT NULL DEFAULT 'HEAD'"),
        ("public_code_sources", "proxy_profile", "VARCHAR(100)"),
        ("public_code_sources", "credential_profile", "VARCHAR(100)"),
        ("public_code_sources", "language_overrides", "JSON NOT NULL DEFAULT '{}'"),
        ("public_code_sources", "source_origin", "VARCHAR(32) NOT NULL DEFAULT 'bootstrap'"),
        ("public_code_sources", "created_by_user_id", "VARCHAR(255)"),
        ("public_code_revisions", "cleanup_pending", "BOOLEAN NOT NULL DEFAULT 0"),
        ("tasks", "owner_user_id", "VARCHAR(255)"),
        ("papers", "source_version", "VARCHAR(32)"),
        ("papers", "withdrawn", "BOOLEAN NOT NULL DEFAULT 0"),
        ("papers", "withdrawn_at", "VARCHAR(40)"),
        ("papers", "full_text_source", "VARCHAR(16)"),
        # NULL default, no backfill: an existing file is "not known to have changed".
        ("index_catalog", "content_changed_at", "DATETIME"),
    ]
    with engine.begin() as conn:
        for table, column, ddl in migrations:
            cols = {row[1] for row in conn.execute(
                text(f"PRAGMA table_info({table})")
            ).fetchall()}
            if column not in cols:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS "
            "ix_public_code_revisions_cleanup_pending "
            "ON public_code_revisions(cleanup_pending)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_tasks_owner_user_id "
            "ON tasks(owner_user_id)"
        ))
        # index_catalog read paths that must not scan every file row of an owner/scope
        # (core/db/models.py::IndexCatalog); create_all() never adds indexes to an
        # existing table, so a deployed DB gets them here, before the code that uses them.
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_index_catalog_owner_projects "
            "ON index_catalog(owner_user_id, host, project, root_dir, kind, deleted_at)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_index_catalog_scope_lower_path "
            "ON index_catalog(owner_user_id, host, project, lower(rel_path))"
        ))
        # Search-time path ranges (IndexCatalogRepository.matching_code_paths) and scope
        # spellings (IndexCatalogRepository.project_spellings). Built from the model's own
        # Index objects so a partial index's WHERE is spelled once and queries can repeat
        # it verbatim; installed before the code that depends on them.
        from sqlalchemy.schema import CreateIndex
        from core.db.models import IndexCatalog
        for index in IndexCatalog.__table__.indexes:
            if index.name in _CATALOG_READ_INDEXES:
                conn.execute(CreateIndex(index, if_not_exists=True))
        # Duplicate-code analysis state (app/services/code_kb/analysis.py). create_all()
        # already creates missing tables at startup; doing it here too means this entry
        # point alone brings any database up to date, before the code that reads them.
        from core.db.models import CodeDuplicatePair, CodeScopeAnalysis
        for model in (CodeScopeAnalysis, CodeDuplicatePair):
            model.__table__.create(conn, checkfirst=True)
        tables = {row[0] for row in conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        ).fetchall()}
        if "index_scope_policies" in tables:
            # Automatic indexing runs at most once per 5 minutes per scope (2026-11-15);
            # policies stored with the old default of 10 are raised to that floor.
            conn.execute(text(
                "UPDATE index_scope_policies SET min_check_interval_s = 300 "
                "WHERE min_check_interval_s < 300"
            ))
        if "journal_entries" in tables:
            from core.db.models import JournalEntry
            for index in JournalEntry.__table__.indexes:
                if index.name == "ix_journal_entries_project_kind":
                    conn.execute(CreateIndex(index, if_not_exists=True))
        if "journal_projects" in tables:
            conn.execute(text("DROP INDEX IF EXISTS ix_journal_projects_slug"))
            conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "ux_journal_projects_owner_slug "
                "ON journal_projects(owner_user_id, slug)"
            ))


def cleanup_orphan_runtime_rows() -> None:
    """Remove dangling task runtime rows left behind by earlier non-cascading deletes."""
    statements = [
        "DELETE FROM task_events WHERE task_id NOT IN (SELECT id FROM tasks)",
        "DELETE FROM task_resource_samples WHERE task_id NOT IN (SELECT id FROM tasks)",
        "DELETE FROM task_workflow_edges WHERE task_id NOT IN (SELECT id FROM tasks)",
        "DELETE FROM task_subtasks WHERE task_id NOT IN (SELECT id FROM tasks)",
        "DELETE FROM crawl_records WHERE task_id IS NOT NULL AND task_id NOT IN (SELECT id FROM tasks)",
        "DELETE FROM task_events WHERE subtask_id IS NOT NULL AND subtask_id NOT IN (SELECT id FROM task_subtasks)",
        "DELETE FROM task_resource_samples WHERE subtask_id IS NOT NULL AND subtask_id NOT IN (SELECT id FROM task_subtasks)",
    ]

    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))
