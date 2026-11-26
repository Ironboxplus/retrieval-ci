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
        tables = {row[0] for row in conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        ).fetchall()}
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
