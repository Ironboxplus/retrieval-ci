"""Journal FTS5 MATCH builder: quoted terms, bounded ASCII prefix expansion (real SQLite)."""
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.database import Base
from app.services.journal import fts
from app.services.journal.repository import JournalRepository


def _repo():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_factory():
        session = session_local()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return JournalRepository(session_factory=session_factory)


def _add(repo, body):
    entry, _project, _deduped = repo.create_entry(
        project_slug="retrival", client="codex", source_kind="memory",
        title=None, summary=None, body=body, tags=[], metadata={},
    )
    return entry


def test_keyword_search_treats_fts_keywords_as_words():
    repo = _repo()
    first = _add(repo, "qdrant upsert near the end")
    _add(repo, "unrelated text")

    rows, total = repo.keyword_search(query="OR AND NOT NEAR qdrant")

    assert total == 1
    assert [entry.entry_id for entry, _score in rows] == [first.entry_id]


def test_short_ascii_token_matches_whole_word_only():
    repo = _repo()
    short = _add(repo, "abc marker")
    _add(repo, "abcdef marker")

    rows, total = repo.keyword_search(query="abc")

    assert total == 1
    assert [entry.entry_id for entry, _score in rows] == [short.entry_id]


def test_four_char_ascii_token_keeps_prefix():
    repo = _repo()
    entry = _add(repo, "observability")

    rows, total = repo.keyword_search(query="obse")

    assert total == 1
    assert rows[0][0].entry_id == entry.entry_id


def test_cjk_token_keeps_prefix():
    repo = _repo()
    entry = _add(repo, "后端内存压力很高")

    rows, total = repo.keyword_search(query="后端")

    assert total == 1
    assert rows[0][0].entry_id == entry.entry_id


def test_match_query_shape():
    assert fts._match_query("W58 mem watchdog a") == '"W58" OR "mem" OR "watchdog"* OR "a"'
    assert fts._match_query("后端 OR") == '"后端"* OR "OR"'
    assert fts._match_query("  ,;  ") is None
